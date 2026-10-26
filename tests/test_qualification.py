import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from runtime.integration.qualification import (
    BASE_GATES, PERFORMANCE_GATES, digest, eligible_rules, make_record, validate_record,
)
from runtime.kernel_package import registry, verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package():
    return dict(package_id="xrgpu.quest3.fp32.v1", package_sha256="a" * 64,
                runtime_revision="e" * 40)


@pytest.fixture
def device():
    return dict(model="Quest 3", gpu="Adreno (TM) 740", vendor_id=0x5143,
                device_id=740, driver="test-driver-1", api_version=4198400)


@pytest.fixture
def rule():
    return dict(request=dict(operation="linear", input_shape=[1, 32, 384],
                             output_shape=[1, 32, 1536], dtype="float32", layout="texture3d",
                             attributes=dict(bias=True, weight_shape=[1536, 384])),
                variant="m4_wg8x4_texture2d", kind="performance")


def sealed(record):
    record["qualification_sha256"] = digest(
        {k: v for k, v in record.items() if k != "qualification_sha256"}
    )
    return record


def qualified(package, device, rule):
    return make_record(package, device, [rule], dict.fromkeys(PERFORMANCE_GATES, True), ["b" * 64])


def test_automatic_eligibility_is_bound_to_exact_package_runtime_and_device(package, device, rule):
    record = qualified(package, device, rule)
    assert eligible_rules(record, package, device, package["runtime_revision"]) == [rule]
    for field, value in dict(model="Quest Pro", gpu="other GPU", vendor_id=1, device_id=2,
                             driver="test-driver-2", api_version=0).items():
        changed = dict(device, **{field: value})
        assert eligible_rules(record, package, changed, package["runtime_revision"]) == []
    assert eligible_rules(record, package, device, "f" * 40) == []
    with pytest.raises(ValueError, match="package/runtime mismatch"):
        eligible_rules(record, dict(package, package_sha256="c" * 64), device,
                       package["runtime_revision"])


@pytest.mark.parametrize("gate", PERFORMANCE_GATES)
def test_missing_gate_never_promotes_performance(package, device, rule, gate):
    gates = dict.fromkeys(PERFORMANCE_GATES, True)
    gates[gate] = False
    record = make_record(package, device, [rule], gates, ["b" * 64])
    assert record["status"] == "pending" and record["rules"] == []
    assert eligible_rules(record, package, device, package["runtime_revision"]) == []


def test_correctness_repair_does_not_require_performance_win(package, device, rule):
    repair = copy.deepcopy(rule)
    repair.update(kind="correctness_repair", variant="semantics_v1")
    repair["request"].update(operation="gelu", output_shape=[1, 32, 384],
                             attributes={"approximate": "none"})
    record = make_record(package, device, [repair], dict.fromkeys(BASE_GATES, True), ["b" * 64])
    assert record["status"] == "qualified"
    assert eligible_rules(record, package, device, package["runtime_revision"]) == [repair]


def test_encoded_vulkan_driver_and_package_target_constraints(package, device, rule):
    encoded = dict(device, driver=8388608)
    constrained = dict(package, device={k: device[k] for k in ("model", "gpu", "vendor_id")})
    record = qualified(constrained, encoded, rule)
    assert eligible_rules(record, constrained, encoded, package["runtime_revision"]) == [rule]
    with pytest.raises(ValueError, match="outside the kernel package"):
        qualified(constrained, dict(encoded, gpu="another GPU"), rule)


def test_simulated_missing_and_mutated_evidence_never_promotes(package, device, rule):
    gates = dict.fromkeys(PERFORMANCE_GATES, True)
    for hashes, synthetic in (([], False), (["b" * 64], True)):
        assert make_record(package, device, [rule], gates, hashes,
                           synthetic=synthetic)["status"] == "pending"
    record = qualified(package, device, rule)
    record["rules"][0]["request"]["input_shape"][1] = 128
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_record(record, package)


@pytest.mark.parametrize("damage", ["duplicate", "unknown", "synthetic", "gate", "shape"])
def test_resealed_invalid_rules_are_rejected(package, device, rule, damage):
    record = qualified(package, device, rule)
    if damage == "duplicate":
        record["rules"].append(copy.deepcopy(rule))
    elif damage == "unknown":
        record["rules"][0]["variant"] = "invented"
    elif damage == "synthetic":
        record["synthetic"] = True
    elif damage == "gate":
        record["gates"]["models"] = False
    else:
        record["rules"][0]["request"]["input_shape"] = [1, -1, 384]
    with pytest.raises(ValueError):
        validate_record(sealed(record), package)


def test_native_qualification_and_repeated_nodes_match_python(tmp_path, package, device, rule):
    compiler = shutil.which("c++")
    include = ROOT / "artifacts/deps/executorch/third-party/json/single_include"
    if not compiler or not include.exists():
        pytest.skip("Host compiler and pinned nlohmann/json required")
    record = qualified(package, device, rule)
    data = tmp_path / "qualification.json"
    data.write_text(json.dumps(dict(package=package, device=device, record=record)))
    source = tmp_path / "check.cpp"
    source.write_text(r'''
#include "runtime/integration/xrgpu_kernel_package.h"
#include <cassert>
#include <fstream>
int main(int argc,char**argv) {
  using namespace xrgpu_package;
  std::ifstream f(argv[1]); auto data=Json::parse(f);
  auto rules=eligible_rules(data["record"],data["package"],data["device"],
                            data["package"]["runtime_revision"]);
  assert(rules.size()==1); Context ctx;ctx.rules=rules; Scope scoped(ctx);
  auto request=rules[0]["request"];int graph;
  auto a=node_identity(&graph,1),b=node_identity(&graph,2);
  selections=0;
  assert(!choose(request,true,false,a).empty());
  assert(!choose(request,true,false,b).empty());
  assert(!choose(request,true,false,a).empty());
  assert(ctx.decisions.size()==2 && selections==2);
  auto changed=data["device"];changed["driver"]="another";
  assert(eligible_rules(data["record"],data["package"],changed,
                        data["package"]["runtime_revision"]).empty());
  auto bad=data["record"];bad["gates"]["sustained"]=false;bool failed=false;
  try {eligible_rules(bad,data["package"],data["device"],data["package"]["runtime_revision"]);}
  catch(const std::exception&){failed=true;}assert(failed);
}
''')
    output = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT), "-I", str(include),
                    str(source), "-o", str(output)], check=True)
    subprocess.run([str(output), str(data)], check=True)


def test_existing_package_embeds_identical_shader_binaries(tmp_path):
    path = ROOT / "artifacts/packages/quest3-fp32-host-repair"
    if not path.exists():
        pytest.skip("Locally built shader package required")
    result = registry(path, tmp_path / "registry.cpp")
    assert result["shaders"] == 18
    assert result["package_sha256"] == verify(path)["package_sha256"]


def test_package_rejects_symlink_even_for_matching_bytes(tmp_path):
    actual = tmp_path / "actual.txt"
    actual.write_text("kernel")
    path = tmp_path / "package"
    path.mkdir()
    (path / "kernel").symlink_to(actual)
    import hashlib
    manifest = dict(schema_version=1, package_id="xrgpu.quest3.fp32.v1", selections=[],
                    files={"kernel": hashlib.sha256(actual.read_bytes()).hexdigest()})
    manifest["package_sha256"] = digest(manifest)
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="file mismatch"):
        verify(path)
