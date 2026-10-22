"""Build and verify local versioned xrGPU packages; never promotes a kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = 'xrgpu.quest3.fp32.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def verify(path):
    path = Path(path).resolve()
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('package_id') != PACKAGE_ID:
        raise ValueError('Unsupported kernel package')
    content = {k: v for k, v in manifest.items() if k != 'package_sha256'}
    if hashlib.sha256(canonical(content)).hexdigest() != manifest.get('package_sha256'):
        raise ValueError('Package manifest digest mismatch')
    for name, expected in manifest['files'].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or not target.is_file() or sha(target) != expected:
            raise ValueError('Package file mismatch: ' + name)
    if manifest.get('selections') != []:
        raise ValueError('v1 packages require external qualification; no automatic linear promotion')
    return manifest


def build(output, executorch, glslc, jobs=4):
    output, executorch, glslc = Path(output).resolve(), Path(executorch).resolve(), Path(glslc).resolve()
    pins = json.loads((ROOT / 'dependencies.json').read_text())
    revision = subprocess.check_output(['git', '-C', str(executorch), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != pins['executorch']['commit']:
        raise ValueError('ExecuTorch pin mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.xrgpu-package-', dir=output.parent))
    try:
        source = staging / 'sources'
        source.mkdir()
        upstream = executorch / 'backends/vulkan/runtime/graph/ops/glsl'
        for p in upstream.iterdir():
            if p.suffix in ('.glslh', '.h'):
                shutil.copyfile(p, source / p.name)
        for op in ('gelu', 'grouped_conv2d', 'linear'):
            shutil.copytree(ROOT / 'kernels' / op, staging / 'kernels' / op)
            for p in (ROOT / 'kernels' / op / 'vulkan/quest3').iterdir():
                if p.suffix in ('.glsl', '.yaml'):
                    shutil.copyfile(p, source / p.name)
        shutil.copytree(ROOT / 'runtime/integration', staging / 'integration',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(executorch / 'LICENSE', staging / 'EXECUTORCH-LICENSE')
        compiler = subprocess.check_output([str(glslc), '--version'], text=True).strip()
        generator = executorch / 'backends/vulkan/runtime/gen_vulkan_spv.py'
        subprocess.run([sys.executable, str(generator), '--glsl-paths', str(source),
            '--output-path', str(staging / 'spirv'), '--tmp-dir-path', str(staging / 'generated'),
            '--glslc-path', str(glslc), '--optimize', '--nthreads', str(jobs)], check=True)
        data = runpy.run_path(str(generator))
        env = {**data['DEFAULT_ENV'], **data['TYPE_MAPPINGS'], **data['UTILITY_FNS']}
        mapping = data['SPVGenerator'](str(source), env, None).output_file_map
        shaders = {}
        for name, (template, parameters) in mapping.items():
            if Path(template).suffix == '.glsl':
                binary = staging / 'spirv' / (name + '.spv')
                if not binary.is_file():
                    raise ValueError('Missing compiled shader: ' + name)
                operation = next(op for op in ('grouped_conv2d', 'linear', 'gelu') if name.startswith('xrgpu_' + op))
                shaders[name] = dict(operation=operation, source='sources/' + Path(template).name,
                    spirv_sha256=sha(binary), variant='candidate', validation_control='none')
        files = {p.relative_to(staging).as_posix(): sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        manifest = dict(schema_version=1, package_id=PACKAGE_ID, status='unqualified',
            backend='executorch_vulkan', precision='fp32', operations=['gelu', 'grouped_conv2d', 'linear'],
            device=dict(model='Quest 3', gpu='Adreno (TM) 740', vendor_id=0x5143),
            runtime_revision=revision, compiler=compiler, flags=['-O', '--target-env=vulkan1.1'],
            generator_sha256=sha(generator), files=files, shaders=shaders, selections=[],
            linear_variants=json.loads((ROOT / 'kernels/linear/contract.json').read_text())['variants'])
        manifest['package_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
        (staging / 'manifest.json').write_bytes(canonical(manifest) + b'\n')
        if output.exists():
            old = verify(output)
            if old['package_sha256'] != manifest['package_sha256']:
                raise FileExistsError('Immutable package destination exists; use a new output directory')
        else:
            staging.rename(output)
        return dict(path=str(output), package_sha256=manifest['package_sha256'], shaders=len(shaders), status='unqualified')
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    b = commands.add_parser('build')
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--executorch', type=Path, default=ROOT.parent / 'xrBench/artifacts/deps/executorch')
    b.add_argument('--glslc', type=Path, default=Path(os.environ.get('GLSLC_PATH', str(ROOT.parent / 'xrBench/artifacts/toolchain/shaderc/2026.4/bin/glslc'))))
    b.add_argument('--jobs', type=int, default=4)
    v = commands.add_parser('verify')
    v.add_argument('path', type=Path)
    args = parser.parse_args()
    result = build(args.output, args.executorch, args.glslc, args.jobs) if args.command == 'build' else verify(args.path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
