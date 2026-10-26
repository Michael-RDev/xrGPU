"""Export and verify static, typed inference bundles for the xrGPU Session API.

Bundles contain a portable CPU program even when Vulkan lowering is unsupported.
They describe model programs and tensors; qualification is supplied by the library,
never by a model bundle.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import operator
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
DTYPES = {"float32": ("<f4", 4), "int64": ("<i8", 8)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def runtime_revision():
    return json.loads((ROOT / "dependencies.json").read_text())["executorch"]["commit"]


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid bundle path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Invalid bundle path: " + value)
    return path


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _tensor_spec(spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not spec["name"]:
        raise ValueError("Each tensor needs a name")
    if spec.get("dtype") not in DTYPES:
        raise ValueError("Session supports float32 and int64 tensors")
    shape = spec.get("shape")
    if not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Session requires positive static tensor dimensions")
    size = math.prod(shape) * DTYPES[spec["dtype"]][1]
    if size > 2**31 - 1:
        raise ValueError("Tensor exceeds the v1 Session size limit")
    return size


def verify_bundle(directory, expected_runtime=None):
    """Verify the model interface and all files without importing the ML stack."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "bundle.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("bundle_type") != "xrgpu.model":
        raise ValueError("Unsupported model bundle schema")
    for key in ("model_id", "model_revision", "runtime_revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError("Missing bundle identity: " + key)
    if manifest.get("precision") != "fp32":
        raise ValueError("Only FP32 model bundles are supported")
    if expected_runtime is not None and manifest["runtime_revision"] != expected_runtime:
        raise ValueError("Model runtime revision mismatch")
    for key in ("inputs", "outputs"):
        specs = manifest.get(key)
        if not isinstance(specs, list) or not specs:
            raise ValueError("Model bundle requires " + key)
        for spec in specs:
            _tensor_spec(spec)
        if len({spec["name"] for spec in specs}) != len(specs):
            raise ValueError("Duplicate tensor names")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Model bundle requires a file manifest")
    for name, digest in files.items():
        _relative(name)
        if name == "bundle.json" or not _digest(digest):
            raise ValueError("Invalid bundle file digest")
        target = directory / name
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != directory.parent):
            raise ValueError("Symlinks are not allowed in model bundles")
        if not target.resolve().is_relative_to(directory) or not target.is_file() or sha256(target) != digest:
            raise ValueError("Model bundle file mismatch: " + name)
    programs = manifest.get("programs", {})
    if not isinstance(programs, dict) or "cpu" not in programs or set(programs) - {"cpu", "vulkan"}:
        raise ValueError("A portable CPU fallback program is required")
    for backend, entry in programs.items():
        if not isinstance(entry, dict) or entry.get("file") not in files:
            raise ValueError("Missing program file for " + backend)
    package = manifest.get("kernel_package")
    if package is not None and (
        not isinstance(package, dict) or not package.get("package_id")
        or not _digest(package.get("package_sha256"))
    ):
        raise ValueError("Invalid required kernel package")
    if "qualification" in manifest or "selections" in manifest:
        raise ValueError("Model bundles cannot supply kernel qualification")
    if "fixtures" in manifest:
        if not isinstance(manifest["fixtures"], list) or not manifest["fixtures"]:
            raise ValueError("Invalid bundle fixtures")
        seeds = set()
        for fixture in manifest["fixtures"]:
            if not isinstance(fixture, dict) or type(fixture.get("seed")) is not int or fixture["seed"] in seeds:
                raise ValueError("Fixture seeds must be unique integers")
            seeds.add(fixture["seed"])
            for key in ("inputs", "outputs"):
                entries = fixture.get(key)
                if not isinstance(entries, list) or len(entries) != len(manifest[key]):
                    raise ValueError("Fixture tensor count mismatch")
                for entry, spec in zip(entries, manifest[key]):
                    if not isinstance(entry, dict) or {key: entry.get(key) for key in ("name", "shape", "dtype")} != spec:
                        raise ValueError("Fixture tensor specification mismatch")
                    if entry.get("file") not in files or (directory / entry["file"]).stat().st_size != _tensor_spec(spec):
                        raise ValueError("Fixture tensor byte count mismatch")
    if manifest.get("bundle_sha256") is not None:
        content = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
        if hashlib.sha256(canonical(content)).hexdigest() != manifest["bundle_sha256"]:
            raise ValueError("Model bundle manifest digest mismatch")
    return manifest


def _spec(tensor, name):
    import torch
    dtype = {torch.float32: "float32", torch.int64: "int64"}.get(tensor.dtype)
    spec = dict(name=name, shape=list(tensor.shape), dtype=dtype)
    _tensor_spec(spec)
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Export inputs and outputs must be contiguous CPU tensors")
    if dtype == "float32" and not torch.isfinite(tensor).all().item():
        raise ValueError("Non-finite reference tensor")
    return spec


def _outputs(value):
    import torch
    result = (value,) if isinstance(value, torch.Tensor) else tuple(value)
    if not result or not all(isinstance(item, torch.Tensor) for item in result):
        raise ValueError("Model must return a tensor or a flat tuple/list of tensors")
    return tuple(item.contiguous() for item in result)


def _lower(module, inputs, destination, vulkan):
    import torch
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.delegate import executorch_call_delegate

    partitions = []
    if vulkan:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        partitions = [VulkanPartitioner({"force_fp16": False})]
    edge = to_edge_transform_and_lower(torch.export.export(module, inputs, strict=True), partitioner=partitions)
    graph = edge.exported_program().graph
    calls = [node for node in graph.nodes if node.op == "call_function"]
    delegates = [node for node in calls if node.target == executorch_call_delegate]
    leftovers = [str(node.target) for node in calls if node.target not in (executorch_call_delegate, operator.getitem)]
    coverage = dict(delegate_count=len(delegates), non_delegate_ops=leftovers,
                    fully_delegated=bool(delegates) and not leftovers,
                    processor="vulkan" if vulkan else "cpu")
    if vulkan and not delegates:
        raise ValueError("Model has no supported Vulkan partitions")
    # Portable CPU nodes surrounding Vulkan partitions are supported, but exposed.
    destination.with_suffix(".graph.txt").write_text(str(graph))
    destination.write_bytes(edge.to_executorch().buffer)
    return coverage


def export_bundle(module, inputs, destination, *, model_id, model_revision, input_names=None,
                  output_names=None, kernel_package=None, preprocessing=None, metadata=None,
                  seed=20261025, strict_vulkan=False, overwrite=False, additional_fixtures=(),
                  extra_files=()):
    """Export an eval-mode module with exact static tensor inputs and CPU fallback.

    A failed Vulkan lowering is recorded and omitted unless strict_vulkan=True.
    Existing output is preserved until the complete replacement verifies.
    """
    import torch

    module = module.eval().cpu()
    inputs = tuple(inputs)
    if not inputs:
        raise ValueError("At least one input tensor is required")
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in (*module.parameters(), *module.buffers())):
        raise ValueError("Model floating parameters and buffers must be FP32")
    input_names = list(input_names or [f"input_{i}" for i in range(len(inputs))])
    if len(input_names) != len(inputs):
        raise ValueError("Input name count mismatch")
    input_specs = [_spec(value, name) for value, name in zip(inputs, input_names)]
    with torch.inference_mode():
        outputs = _outputs(module(*inputs))
    output_names = list(output_names or [f"output_{i}" for i in range(len(outputs))])
    if len(output_names) != len(outputs):
        raise ValueError("Output name count mismatch")
    output_specs = [_spec(value, name) for value, name in zip(outputs, output_names)]
    destination = Path(destination).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError("Model bundle exists; choose a new destination or use overwrite=True")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".xrgpu-model-", dir=destination.parent))
    try:
        manifest = dict(schema_version=SCHEMA_VERSION, bundle_type="xrgpu.model", model_id=model_id,
                        model_revision=model_revision, runtime_revision=runtime_revision(), precision="fp32",
                        inputs=input_specs, outputs=output_specs, seed=seed, programs={}, export_errors={},
                        preprocessing=preprocessing or {}, metadata=metadata or {},
                        environment={name: importlib.metadata.version(name) for name in ("torch", "executorch", "numpy")})
        if kernel_package is not None:
            manifest["kernel_package"] = {key: kernel_package[key] for key in ("package_id", "package_sha256")}
        for backend in ("cpu", "vulkan"):
            program = staging / (backend + ".pte")
            print(f"  Lower {model_id} for {backend}", flush=True)
            try:
                coverage = _lower(module, inputs, program, backend == "vulkan")
                manifest["programs"][backend] = dict(file=program.name, coverage=coverage)
            except Exception as error:
                program.unlink(missing_ok=True)
                program.with_suffix(".graph.txt").unlink(missing_ok=True)
                if backend == "cpu" or strict_vulkan:
                    raise
                manifest["export_errors"][backend] = f"{type(error).__name__}: {error}"
        manifest["fixtures"] = []
        seen_seeds = set()
        for fixture_index, (fixture_seed, fixture_inputs) in enumerate(((seed, inputs), *additional_fixtures)):
            if type(fixture_seed) is not int or fixture_seed in seen_seeds:
                raise ValueError("Fixture seeds must be unique integers")
            seen_seeds.add(fixture_seed)
            fixture_inputs = tuple(fixture_inputs)
            if len(fixture_inputs) != len(inputs) or [_spec(value, spec["name"]) for value, spec in zip(fixture_inputs, input_specs)] != input_specs:
                raise ValueError("Fixture input specification mismatch")
            with torch.inference_mode():
                fixture_outputs = _outputs(module(*fixture_inputs))
            if len(fixture_outputs) != len(outputs) or [_spec(value, spec["name"]) for value, spec in zip(fixture_outputs, output_specs)] != output_specs:
                raise ValueError("Fixture output specification mismatch")
            fixture = dict(seed=fixture_seed, inputs=[], outputs=[])
            for key, values, specs in (("inputs", fixture_inputs, input_specs), ("outputs", fixture_outputs, output_specs)):
                for i, (value, spec) in enumerate(zip(values, specs)):
                    filename = f"fixture.{fixture_index}.{key}.{i}.bin"
                    value.detach().numpy().astype(DTYPES[spec["dtype"]][0]).tofile(staging / filename)
                    fixture[key].append(dict(file=filename, **spec))
            manifest["fixtures"].append(fixture)
        (staging / "reference.json").write_bytes(canonical(manifest["fixtures"][0]) + b"\n")
        for source, relative in extra_files:
            _relative(relative)
            target = staging / relative
            if target.exists() or relative == "bundle.json":
                raise ValueError("Duplicate extra bundle file: " + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        manifest["files"] = {path.relative_to(staging).as_posix(): sha256(path) for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest["bundle_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        (staging / "bundle.json").write_bytes(canonical(manifest) + b"\n")
        verify_bundle(staging, expected_runtime=runtime_revision())
        # Execute the portable program before publishing a bundle.
        verify_cpu_reference(staging)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_cpu_reference(directory, atol=1e-4, rtol=1e-4):
    import numpy as np
    import torch
    from executorch.runtime import Runtime

    directory = Path(directory)
    manifest = verify_bundle(directory, expected_runtime=runtime_revision())
    program = Runtime.get().load_program(directory / manifest["programs"]["cpu"]["file"])
    method = program.load_method("forward")
    results = []
    references = manifest.get("fixtures") or [json.loads((directory / "reference.json").read_text())]
    for reference in references:
        arguments, expected = [], []
        for key, values in (("inputs", arguments), ("outputs", expected)):
            if len(reference[key]) != len(manifest[key]):
                raise ValueError("Reference tensor count mismatch")
            for entry, spec in zip(reference[key], manifest[key]):
                if {key: entry[key] for key in ("name", "shape", "dtype")} != spec or entry["file"] not in manifest["files"]:
                    raise ValueError("Reference tensor specification mismatch")
                data = np.fromfile(directory / entry["file"], dtype=DTYPES[spec["dtype"]][0])
                if data.nbytes != _tensor_spec(spec):
                    raise ValueError("Reference tensor byte count mismatch")
                values.append(torch.from_numpy(data.reshape(spec["shape"]).copy()))
        actual = method.execute(tuple(arguments))
        if len(actual) != len(expected):
            raise ValueError("CPU output count mismatch")
        for value, target, spec in zip(actual, expected, manifest["outputs"]):
            if list(value.shape) != spec["shape"] or value.dtype != target.dtype:
                raise ValueError("CPU output shape/dtype mismatch")
            same = torch.equal(value, target) if spec["dtype"] == "int64" else torch.allclose(value, target, atol=atol, rtol=rtol)
            if not same:
                raise ValueError("CPU reference mismatch: " + spec["name"])
            results.append(dict(name=spec["name"], fixture_seed=reference["seed"], passed=True,
                                max_abs_error=float((value.to(torch.float64) - target.to(torch.float64)).abs().max())))
    return dict(backend="executorch_cpu", passed=True, outputs=results,
                fixtures=len(references), bundle_sha256=manifest.get("bundle_sha256"), atol=atol, rtol=rtol)


def bind_package(source, package, destination, *, overwrite=False, execute_cpu=False):
    """Bind verified existing programs to a verified package without re-exporting.

    All model/program/fixture bytes and file digests remain identical. The new
    bundle must target the same pinned ExecuTorch runtime as the package.
    """
    from runtime.kernel_package import verify as verify_package

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Package binding requires a separate destination")
    if destination.exists() and not overwrite:
        raise FileExistsError("Bound model bundle destination already exists")
    manifest = verify_bundle(source, expected_runtime=runtime_revision())
    kernel = verify_package(package)
    if kernel["runtime_revision"] != manifest["runtime_revision"]:
        raise ValueError("Kernel package and model runtime revisions differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".xrgpu-bound-", dir=destination.parent))
    try:
        for filename in manifest["files"]:
            target = staging / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / filename, target)
        manifest.pop("bundle_sha256", None)
        manifest["kernel_package"] = {key: kernel[key] for key in ("package_id", "package_sha256")}
        manifest["bundle_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        (staging / "bundle.json").write_bytes(canonical(manifest) + b"\n")
        verify_bundle(staging, expected_runtime=kernel["runtime_revision"])
        if execute_cpu:
            verify_cpu_reference(staging)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--execute-cpu", action="store_true")
    parser.add_argument("--bind-package", type=Path, help="Bind programs to a verified kernel package without re-exporting")
    parser.add_argument("--output", type=Path, help="Destination for a separate bound bundle")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bind_package:
        if not args.output:
            parser.error("--bind-package requires --output")
        result = bind_package(args.bundle, args.bind_package, args.output, overwrite=args.overwrite,
                              execute_cpu=args.execute_cpu)
    else:
        if args.output or args.overwrite:
            parser.error("--output and --overwrite require --bind-package")
        result = verify_cpu_reference(args.bundle) if args.execute_cpu else verify_bundle(args.bundle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
