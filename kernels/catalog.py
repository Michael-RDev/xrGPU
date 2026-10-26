"""Authoritative native XR catalog, generated registry checks, and local inspection.

python -m kernels.catalog validates all entries and their immutable provenance.
python -m kernels.catalog --generate updates derived runtime registries.
python -m kernels.catalog --refresh explicitly reindexes edited operator descriptors.
"""

# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
from pathlib import Path
import re

KERNELS = Path(__file__).resolve().parent
DTYPES = ["float32", "float16", "int32", "uint32", "int16", "int8", "uint8", "int64", "bool"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def regular(root, relative, *, directory=False):
    if (
        not isinstance(relative, str)
        or "\\" in relative
        or relative.startswith("/")
        or any(p in ("", ".", "..") for p in relative.split("/"))
    ):
        raise ValueError("Unsafe native catalog path: " + str(relative))
    path = Path(root)
    if path.is_symlink():
        raise ValueError("Native catalog root links are forbidden")
    for part in relative.split("/"):
        path /= part
        if path.is_symlink():
            raise ValueError("Native catalog symlinks are forbidden")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError("Native catalog source missing: " + relative)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shader_variants(value, root):
    variants = {}
    for entry in value["entries"]:
        for variant in entry["variants"]:
            output = variant["output"]
            if not isinstance(output, str) or not re.fullmatch(
                r"(?:extended/)?[a-z0-9_]+\.spv", output
            ):
                raise ValueError("Unsafe native shader output")
            source = regular(root, variant["source"])
            if source.suffix != ".comp" or not variant["source"].startswith(
                entry["id"] + "/shaders/"
            ):
                raise ValueError("Native entrypoint must belong to its operator")
            if not isinstance(variant["defines"], list) or any(
                not isinstance(d, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*=[A-Za-z0-9_]+", d)
                for d in variant["defines"]
            ):
                raise ValueError("Invalid native shader definition")
            for include in variant["include_dirs"]:
                if not include.startswith("_shared/"):
                    raise ValueError("Native include must use shared library sources")
                regular(root, include, directory=True)
            if output in variants:
                other = variants[output]
                if sha(source) != sha(regular(root, other["source"])) or any(
                    variant[k] != other[k] for k in ("defines", "include_dirs")
                ):
                    raise ValueError("Conflicting native shader dependency: " + output)
            else:
                variants[output] = variant
    return [variants[k] for k in sorted(variants)]


def load(root=KERNELS):
    """Validate native descriptors, complete shader inventory and frozen references."""
    root = Path(root)
    value = json.loads(regular(root, "catalog.json").read_text())
    if value.get("schema_version") != 1 or value.get("kind") != "xrgpu.native-kernel-catalog":
        raise ValueError("Unsupported native catalog")
    if digest({k: v for k, v in value.items() if k != "catalog_sha256"}) != value.get(
        "catalog_sha256"
    ):
        raise ValueError("Native catalog identity mismatch")
    if (
        value.get("backend"),
        value.get("target"),
        value.get("vulkan_min_version"),
        value.get("provenance_root"),
    ) != ("vulkan", "android-arm64", "1.1", "_provenance"):
        raise ValueError("Unsupported native catalog target")
    entries = value.get("entries", [])
    identities = [e["id"] for e in entries]
    if (
        len(entries) != 221
        or value.get("operator_count") != 221
        or len({i.casefold() for i in identities}) != 221
    ):
        raise ValueError("Native catalog requires exactly 221 unique qualified identities")
    from kernels.provenance import catalog as provenance_catalog

    upstream = provenance_catalog(root / "_provenance")
    frozen = {e["id"]: e for e in upstream["entries"]}
    if (
        set(identities) != set(frozen)
        or value.get("upstream_catalog_sha256") != upstream["catalog_sha256"]
    ):
        raise ValueError("Native catalog differs from frozen provenance")
    if (value.get("test_count"), value.get("benchmark_count")) != (13919, 5839):
        raise ValueError("Native fixture case inventory differs")
    legacy = []
    families = {}
    for entry in entries:
        identity = entry["id"]
        if (
            not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+", identity)
            or entry["path"] != identity
        ):
            raise ValueError("Invalid native operator path/identity")
        documents = {}
        for key, filename, kind in [
            ("contract", "contract.json", "xrgpu.native-operator"),
            ("tests", "test.json", "xrgpu.native-fixture-suite"),
            ("benchmarks", "bench.json", "xrgpu.native-benchmark-suite"),
        ]:
            if entry[key] != identity + "/" + filename:
                raise ValueError("Native operator descriptor path mismatch")
            file = regular(root, entry[key])
            if sha(file) != entry["descriptor_digests"][key]:
                raise ValueError("Native descriptor checksum mismatch: " + entry[key])
            document = json.loads(file.read_text())
            if (
                document.get("kind") != kind
                or document.get("id") != identity
                or document.get("backend") != "vulkan"
            ):
                raise ValueError("Native descriptor target/identity mismatch")
            documents[key] = document
        contract = documents["contract"]
        for key in ("native", "variants", "provenance", "status"):
            if contract[key] != entry[key]:
                raise ValueError("Native catalog and contract differ: " + identity + " " + key)
        if any(k in contract for k in ("bindings", "tunables", "derive", "when", "templates")):
            raise ValueError("Native contract contains upstream execution metadata")
        if (
            contract.get("capabilities", {}).get("cpu_fallback") is not False
            or contract["capabilities"].get("silent_type_conversion") is not False
        ):
            raise ValueError("Native contract must reject unsupported execution")
        original = frozen[identity]
        for key in ("repo_id", "revision", "contract", "tests", "benchmarks", "files"):
            if entry["provenance"].get(key) != original[key]:
                raise ValueError("Native provenance mismatch: " + identity)
        source_contract = json.loads(
            regular(root / "_provenance", original["contract"]).read_text()
        )
        for key, source_key in [
            ("inputs", "inputs"),
            ("outputs", "outputs"),
            ("type_constraints", "typeConstraints"),
            ("attributes", "attributes"),
            ("attribute_constraints", "attributeConstraints"),
        ]:
            if contract[key] != source_contract.get(source_key, {}):
                raise ValueError("Native portable contract differs: " + identity + " " + key)
        for key, source_key, count_key in [
            ("tests", "tests", "test_count"),
            ("benchmarks", "benchmarks", "benchmark_count"),
        ]:
            document = documents[key]
            if (
                document["fixture_source"] != "_provenance/" + original[source_key]
                or document["fixture_sha256"] != original["files"][original[source_key]]
                or document["case_count"] != original[count_key]
            ):
                raise ValueError("Native fixture descriptor mismatch")
        native = entry["native"]
        if native["api"] == "legacy":
            row = native["registry"]
            legacy.append(row)
            if row["id"] != identity or row["dtypes"] != [
                t for bit, t in enumerate(DTYPES) if row["dtype_mask"] & (1 << bit)
            ]:
                raise ValueError("Legacy native registry identity/type mask mismatch")
            if (
                row["kind"] not in ("Unary", "Binary", "MatMul")
                or type(row["opcode"]) is not int
                or row["opcode"] < 0
                or not re.fullmatch(r"[a-z0-9_]+", row["slug"])
                or set(row["input_names"]) != set(contract["inputs"])
                or {row["output_name"]} != set(contract["outputs"])
                or contract.get("supported_dtypes") != row["dtypes"]
            ):
                raise ValueError("Invalid legacy native API mapping")
            compiler_types = {
                "float32": ("f32", 0),
                "float16": ("f16", 1),
                "int32": ("i32", 2),
                "uint32": ("u32", 3),
                "int8": ("i8", 4),
                "uint8": ("u8", 5),
                "int16": ("i16", 6),
            }
            expected = []
            for dtype in row["dtypes"]:
                if dtype not in compiler_types:
                    raise ValueError("Legacy dtype has no native shader compiler support")
                tag, code = compiler_types[dtype]
                defines = [f"DTYPE={code}", f"OPCODE={row['opcode']}"]
                if row["kind"] == "Unary":
                    defines.append("UNARY=1")
                expected.append(
                    dict(
                        output=row["slug"] + "_" + tag + ".spv",
                        source=identity + "/shaders/kernel.comp",
                        defines=defines,
                        include_dirs=["_shared/shaders"],
                    )
                )
            if entry["variants"] != expected:
                raise ValueError("Legacy native dtype/variant mapping differs")
        elif native["api"] == "extended":
            if native["family"] not in ("tensor", "spatial", "complex"):
                raise ValueError("Unsupported native family")
            families[native["family"]] = families.get(native["family"], 0) + 1
            expected = {"extended/" + s for s in [native["shader"], *native["shader_dependencies"]]}
            if {v["output"] for v in entry["variants"]} != expected:
                raise ValueError("Incomplete native shader dependencies: " + identity)
        else:
            raise ValueError("Unsupported native API")
    if len(legacy) != 38 or families != {"tensor": 90, "spatial": 57, "complex": 36}:
        raise ValueError("Native implementation family inventory differs")
    if any(len({r[k] for r in legacy}) != 38 for k in ("id", "slug", "opcode")):
        raise ValueError("Duplicate legacy native numeric identity")
    variants = _shader_variants(value, root)
    if len(variants) != 315 or value.get("shader_count") != 315:
        raise ValueError("Native shader inventory requires 315 variants")
    return value


def legacy_records(root=KERNELS):
    return sorted(
        [e["native"]["registry"] for e in load(root)["entries"] if e["native"]["api"] == "legacy"],
        key=lambda r: r["opcode"],
    )


def extended_records(root=KERNELS):
    return [
        dict(id=e["id"], **{k: e["native"][k] for k in ("family", "shader", "shader_dependencies")})
        for e in load(root)["entries"]
        if e["native"]["api"] == "extended"
    ]


def shader_variants(root=KERNELS):
    return _shader_variants(load(root), Path(root))


def generate_registries(root=KERNELS, *, check=False):
    """Generate derived C++/Python membership files from the authoritative catalog."""
    root = Path(root)
    value = load(root)
    outputs = {
        name: "// SPDX-License-Identifier: Apache-2.0\n// Generated from kernels/catalog.json; edit the catalog, then regenerate.\n"
        for name in ("registry.def", "registry_binary.def", "registry_unary.def")
    }
    rows = sorted(
        [e["native"]["registry"] for e in value["entries"] if e["native"]["api"] == "legacy"],
        key=lambda r: r["opcode"],
    )
    for row in rows:
        file = (
            "registry.def"
            if row["opcode"] < 3
            else ("registry_binary.def" if row["kind"] == "Binary" else "registry_unary.def")
        )
        inputs = row["input_names"]
        outputs[file] += (
            "XRGPU_OPERATOR("
            + ", ".join(
                [
                    json.dumps(row["id"]),
                    json.dumps(row["slug"]),
                    row["kind"],
                    str(row["opcode"]),
                    str(row["dtype_mask"]),
                    json.dumps(inputs[0]),
                    json.dumps(inputs[1] if len(inputs) > 1 else ""),
                    json.dumps(row["output_name"]),
                ]
            )
            + ")\n"
        )
    extended = [e for e in value["entries"] if e["native"]["api"] == "extended"]
    for family in ("tensor", "spatial", "complex"):
        members = [e for e in extended if e["native"]["family"] == family]
        outputs["extended/" + family + "_ops.json"] = (
            json.dumps([e["id"] for e in members], indent=2) + "\n"
        )
        outputs["extended/" + family + "_shader_map.json"] = (
            json.dumps({e["id"]: e["native"]["shader"] for e in members}, indent=2, sort_keys=True)
            + "\n"
        )
    outputs["extended/shader_dependencies.json"] = (
        json.dumps(
            {
                e["id"]: e["native"]["shader_dependencies"]
                for e in extended
                if e["native"]["shader_dependencies"]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for name, text in outputs.items():
        path = regular(root, "_shared/" + name)
        if check:
            if path.read_text() != text:
                raise ValueError("Generated native registry drift: " + name)
        else:
            path.write_text(text)
    return len(outputs)


def refresh(root=KERNELS):
    """Explicitly reindex native descriptors; immutable provenance is never edited."""
    root = Path(root)
    value = json.loads(regular(root, "catalog.json").read_text())
    for entry in value["entries"]:
        contract = json.loads(regular(root, entry["contract"]).read_text())
        for key in ("native", "variants", "provenance", "status"):
            entry[key] = contract[key]
        entry["descriptor_digests"] = {
            key: sha(regular(root, entry[key])) for key in ("contract", "tests", "benchmarks")
        }
    value["catalog_sha256"] = digest({k: v for k, v in value.items() if k != "catalog_sha256"})
    # Validate candidate before replacing authoritative metadata.
    path = regular(root, "catalog.json")
    original = path.read_bytes()
    try:
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        load(root)
    except BaseException:
        path.write_bytes(original)
        raise
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        refresh()
    if args.generate:
        generate_registries()
    else:
        generate_registries(check=True)
    value = load()
    print(
        json.dumps(
            {
                k: value[k]
                for k in (
                    "catalog_sha256",
                    "operator_count",
                    "shader_count",
                    "test_count",
                    "benchmark_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
