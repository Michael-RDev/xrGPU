# Native spatial, reduction, normalization, and signal kernels

`spatial_ops.json` lists 57 implemented catalog entry points. Every entry is paired
with host shape/control planning in `spatial_plan.cpp` and a compiled Vulkan shader
through `spatial_shader_map.json`. Numerical tensor work executes in GLSL/SPIR-V;
the host validates metadata, reads shape controls, allocates buffers, and dispatches.
These sources do not use WGSL, WebGPU, a browser, or a CPU numerical fallback.

The baseline favors readable scalar algorithms over performance: each output
invocation gathers its inputs. Convolutions support groups, dilation, automatic
padding, transposition, output padding, integer and per-channel quantized inputs,
and two-dimensional deformable sampling. Pooling supports padding, ceil sizing,
counting padded elements, optional exact Int64 indices, and inverse scatter.
Sampling includes two/three-dimensional grids, nearest/linear/cubic filtering,
border/reflection padding, Resize coordinate and antialias modes, affine grids,
and ROI sampling. Reductions, cumulative scans, axis softmax/arg reductions,
normalization and optional statistics, and direct DFT/STFT/window transforms have
separate shader algorithms. DFT/STFT use direct sums rather than an FFT.

The public frozen type constraints remain authoritative. Index tensors retain
logical Int64, including when the frozen wire format stores them as UInt32.
Numerical Int64 inputs are not part of this family's frozen contracts. Integer
reductions use widened arithmetic before the final declared integer cast; float16
normalization honors the explicit typed cast stages. Accurate float32 helpers
keep range reduction and sensitive normalization intermediates in double-float
arithmetic without requiring shaderFloat64.

## Local verification

Build an isolated, immutable source/test snapshot:

Use `kernels._shared` for the active native catalog. Older immutable snapshots
may instead contain `operators.*`; when verifying one, use its packaged tools
and original paths rather than changing the snapshot or importing a live checkout.

```sh
python -m kernels._shared.extended.develop --output /tmp/xrgpu-spatial --family spatial --jobs 4
```

The command reports exact source, runner, and shader digests. Set
`XRGPU_TEST_VULKAN_LIBRARY` and `VK_ICD_FILENAMES` to a native software Vulkan
loader/ICD for test-only execution. Then run the returned package and runner:

```sh
python -m kernels._shared.extended.verify_spatial \
  --package PACKAGE --runner RUNNER --features --output REPORT.json
python -m kernels._shared.extended.verify_spatial_published \
  --package PACKAGE --runner RUNNER --output PUBLISHED_REPORT.json
```

The independent suite contains one numerical case for every owned entry plus
parameterized NumPy/PyTorch references for groups/dilation, transpose padding,
quantization, interpolation, normalization, indices, scans, and signal transforms.
The published suite checks frozen literal output arrays under their original
absolute/relative/NaN policies. Missing fixture generators or output oracles are
recorded as unsupported fixtures. They are never numerical passes. Filtered
`--failed-from PREVIOUS_REPORT` runs are explicitly marked regression subsets.

Reference conflicts are preserved separately from failures and passes, with raw
input, actual, published, and independent output bytes. This includes an exact
stored-float32 cumulative-sum proof and ONNX21 explicit-half-stage normalization
reference checks. Neither reference conflicts nor skipped fixtures establish
complete catalog conformance.

Host planner and evidence classification tests are in
`tests/test_extended_spatial.py`. Reports in `artifacts/conformance/` record actual
software Vulkan dispatch, shader hashes, and `physical_qualification: false`.
Software results do not qualify an Android/XR device, prove performance, or cover
every shape, attribute, dtype, and exceptional value in the full frozen catalog.
Physical hardware qualification and broader generated-fixture coverage remain
separate gates.
