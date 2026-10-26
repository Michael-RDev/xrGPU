---
library_name: kernels
license: apache-2.0
tags:
- kernel
- webgpu
- wgsl
---
# ai.onnx.Unique

`ai.onnx`  ·  standard ONNX operator  ·  ONNX opset ≥ 11

## Description

Finds unique values or subtensors along an optional `axis`. Without an axis, `X` is flattened; results are sorted or retain first-occurrence order. Sub-32-bit integers and booleans use lossless widened 32-bit storage. Metadata outputs remain logical int64 but use lossless uint32 storage because all values are bounded by an addressable tensor extent. Exact data-dependent output shapes must be supplied. ONNX-permitted uint16, 64-bit, string, and complex inputs are unsupported by this package.

See the [ONNX `Unique` spec](https://onnx.ai/onnx/operators/onnx__Unique.html) for the reference semantics.

## Inputs

| Name | Upstream name | Logical dtype | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- |
| `x` | `X` | `T` | — | — | The N-D input tensor from which unique values or subtensors are extracted. When `axis` is omitted, tensors of any rank are flattened in row-major order. | required |

## Outputs

| Name | Upstream name | Logical dtype | WebGPU storage | Rank | Shape | Description | Presence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `y` | `Y` | `T` | runtime-selected; narrow integers and bool use 32-bit slots | derived | — | Tensor containing all unique values or subtensors of X, sorted or in first-occurrence order. | required |
| `indices` | — | `I` | `uint32` | `1` | — | Optional logical int64 indices of each `Y` value or slice's first occurrence in `X`; stored as bounded uint32 values by WebGPU. | optional |
| `inverse_indices` | — | `I` | `uint32` | `1` | — | Optional logical int64 mapping from each flattened input value, or each input-axis slice, to its corresponding index in `Y`; stored as bounded uint32 values by WebGPU. | optional |
| `counts` | — | `I` | `uint32` | `1` | — | Optional logical int64 occurrence count for each unique value or slice in `Y`; stored as bounded uint32 values by WebGPU. | optional |

## Attributes

Attributes and default values (overridable per request):

| Attribute | Default | Description |
| --- | --- | --- |
| `axis` | — | Optional axis along which unique subtensors are identified. Negative values count from the back; when omitted, the input is flattened. |
| `sorted` | `1` | Whether to sort unique elements in ascending order before output; 1 (default) sorts, 0 retains first-occurrence order. |

## Type constraints

| Variable | Allowed dtypes |
| --- | --- |
| `T` | `float32`, `float16`, `uint32`, `int32`, `int16`, `uint8`, `int8`, `bool` |
| `I` | `int64` |

## Implementation variants

One implementation is selected per call from the device capabilities, the request shapes and the dtypes; these notes say what each one covers.

- `single_class_y` — Copy the first representative and materialize requested metadata when the exact output contract proves there is one distinct class.
- `scalar_hash_parallel_y_unsorted` — Hash first occurrences, scan flag blocks in parallel, and scatter scalar representatives in input order with only the requested metadata.
- `scalar_hash_parallel_y_sorted` — Hash and compact scalar representatives in parallel, sort their order with shared and global bitonic stages, and materialize only the requested metadata.
- `single_class_indices` — Copy the first representative and materialize requested metadata when the exact output contract proves there is one distinct class.
- `scalar_hash_parallel_indices_unsorted` — Hash first occurrences, scan flag blocks in parallel, and scatter scalar representatives in input order with only the requested metadata.
- `scalar_hash_parallel_indices_sorted` — Hash and compact scalar representatives in parallel, sort their order with shared and global bitonic stages, and materialize only the requested metadata.
- `single_class_inverse` — Copy the first representative and materialize requested metadata when the exact output contract proves there is one distinct class.
- `scalar_hash_parallel_inverse_unsorted` — Hash first occurrences, scan flag blocks in parallel, and scatter scalar representatives in input order with only the requested metadata.
- `scalar_hash_parallel_inverse_sorted` — Hash and compact scalar representatives in parallel, sort their order with shared and global bitonic stages, and materialize only the requested metadata.
- `single_class_indices_inverse` — Copy the first representative and materialize requested metadata when the exact output contract proves there is one distinct class.
- `scalar_hash_parallel_indices_inverse_unsorted` — Hash first occurrences, scan flag blocks in parallel, and scatter scalar representatives in input order with only the requested metadata.
- `scalar_hash_parallel_indices_inverse_sorted` — Hash and compact scalar representatives in parallel, sort their order with shared and global bitonic stages, and materialize only the requested metadata.
- `single_class_counts` — Copy the first representative and materialize requested metadata when the exact output contract proves there is one distinct class.
- `scalar_hash_parallel_counts_unsorted` — Hash first occurrences, scan flag blocks in parallel, and scatter scalar representatives in input order with only the requested metadata.
- `scalar_hash_parallel_counts_sorted` — Hash and compact scalar representatives in parallel, sort their order with shared and global bitonic stages, and materialize only the requested metadata.
- `single_class_indices_counts` — Copy the first representative and materialize requested metadata when the exact output contract proves there is one distinct class.
- `scalar_hash_parallel_indices_counts_unsorted` — Hash first occurrences, scan flag blocks in parallel, and scatter scalar representatives in input order with only the requested metadata.
- `scalar_hash_parallel_indices_counts_sorted` — Hash and compact scalar representatives in parallel, sort their order with shared and global bitonic stages, and materialize only the requested metadata.
- `single_class_inverse_counts` — Copy the first representative and materialize requested metadata when the exact output contract proves there is one distinct class.
- `scalar_hash_parallel_inverse_counts_unsorted` — Hash first occurrences, scan flag blocks in parallel, and scatter scalar representatives in input order with only the requested metadata.
- `scalar_hash_parallel_inverse_counts_sorted` — Hash and compact scalar representatives in parallel, sort their order with shared and global bitonic stages, and materialize only the requested metadata.
- `single_class_indices_inverse_counts` — Copy the first representative and materialize requested metadata when the exact output contract proves there is one distinct class.
- `scalar_hash_parallel_indices_inverse_counts_unsorted` — Hash first occurrences, scan flag blocks in parallel, and scatter scalar representatives in input order with only the requested metadata.
- `scalar_hash_parallel_indices_inverse_counts_sorted` — Hash and compact scalar representatives in parallel, sort their order with shared and global bitonic stages, and materialize only the requested metadata.
- `hash_dedup` — Hashes flattened values and records each key's first input index. Long, low-cardinality f32 inputs coalesce duplicates in workgroup memory when the device's storage budget permits it.
- `axis_hash_bounded` — Hash axis slices, sort compacted representative indices in one workgroup, and scatter the exact output.
- `axis_hash_bounded_unsorted` — Hash axis slices and compact representative indices in first-occurrence order.
- `axis_hash_bounded_nan_safe` — Hash and sort finite axis slices, with an exact ordered fallback when any floating-point slice contains a NaN.
- `axis_hash_bounded_unsorted_nan_safe` — Hash finite axis slices in first-occurrence order, with an exact ordered fallback when any floating-point slice contains a NaN.
- `axis_hash_bounded_merge` — Hash axis slices, sort device-sized shared tiles, and merge by exact lexicographic ranks.
- `axis_hash_bounded_nan_safe_merge` — Hash axis slices, sort device-sized shared tiles, and merge by exact lexicographic ranks. Preserve the exact ordered fallback when a floating-point slice contains NaN.
- `flat_parallel_metadata_indices` — Builds the unique values with the parallel dedup and compaction passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_hash_metadata_indices` — Builds the unique values with the hash-set dedup passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_parallel_metadata_inverse` — Builds the unique values with the parallel dedup and compaction passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_hash_metadata_inverse` — Builds the unique values with the hash-set dedup passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_parallel_metadata_indices_inverse` — Builds the unique values with the parallel dedup and compaction passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_hash_metadata_indices_inverse` — Builds the unique values with the hash-set dedup passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_parallel_metadata_counts` — Builds the unique values with the parallel dedup and compaction passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_hash_metadata_counts` — Builds the unique values with the hash-set dedup passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_parallel_metadata_indices_counts` — Builds the unique values with the parallel dedup and compaction passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_hash_metadata_indices_counts` — Builds the unique values with the hash-set dedup passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_parallel_metadata_inverse_counts` — Builds the unique values with the parallel dedup and compaction passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_hash_metadata_inverse_counts` — Builds the unique values with the hash-set dedup passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_parallel_metadata_all` — Builds the unique values with the parallel dedup and compaction passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.
- `flat_hash_metadata_all` — Builds the unique values with the hash-set dedup passes, then resolves every metadata output from the finished result: one thread per input element for the bucket and the first-occurrence store, one thread per unique value for the tally.

## Files

- [`metadata.json`](build/webgpu/metadata.json) — kernel metadata (id, digests, per-variant templates, provenance)
- [`manifest.json`](build/webgpu/manifest.json) — the op contract (source of truth)
- [`test.json`](build/webgpu/test.json) — correctness cases
- [`bench.json`](build/webgpu/bench.json) — benchmark cases
- [`scan-block-prefix-u32.wgsl.jinja`](build/webgpu/scan-block-prefix-u32.wgsl.jinja)
- [`unique-axis-compact-sort.wgsl.jinja`](build/webgpu/unique-axis-compact-sort.wgsl.jinja)
- [`unique-axis-dedup.wgsl.jinja`](build/webgpu/unique-axis-dedup.wgsl.jinja)
- [`unique-axis-hash.wgsl.jinja`](build/webgpu/unique-axis-hash.wgsl.jinja)
- [`unique-axis-merge.wgsl.jinja`](build/webgpu/unique-axis-merge.wgsl.jinja)
- [`unique-axis-scalar-inverse.wgsl.jinja`](build/webgpu/unique-axis-scalar-inverse.wgsl.jinja)
- [`unique-axis-scalar-ranks.wgsl.jinja`](build/webgpu/unique-axis-scalar-ranks.wgsl.jinja)
- [`unique-axis-scatter.wgsl.jinja`](build/webgpu/unique-axis-scatter.wgsl.jinja)
- [`unique-axis.wgsl.jinja`](build/webgpu/unique-axis.wgsl.jinja)
- [`unique-compact-sort.wgsl.jinja`](build/webgpu/unique-compact-sort.wgsl.jinja)
- [`unique-dedup.wgsl.jinja`](build/webgpu/unique-dedup.wgsl.jinja)
- [`unique-flag-block-scan.wgsl.jinja`](build/webgpu/unique-flag-block-scan.wgsl.jinja)
- [`unique-flat-metadata.wgsl.jinja`](build/webgpu/unique-flat-metadata.wgsl.jinja)
- [`unique-global-sort-exchange.wgsl.jinja`](build/webgpu/unique-global-sort-exchange.wgsl.jinja)
- [`unique-global-sort-output.wgsl.jinja`](build/webgpu/unique-global-sort-output.wgsl.jinja)
- [`unique-global-sort-shared.wgsl.jinja`](build/webgpu/unique-global-sort-shared.wgsl.jinja)
- [`unique-hash-build.wgsl.jinja`](build/webgpu/unique-hash-build.wgsl.jinja)
- [`unique-hash-collect.wgsl.jinja`](build/webgpu/unique-hash-collect.wgsl.jinja)
- [`unique-hash-init.wgsl.jinja`](build/webgpu/unique-hash-init.wgsl.jinja)
- [`unique-hash-mark.wgsl.jinja`](build/webgpu/unique-hash-mark.wgsl.jinja)
- [`unique-hash-sort-collected-key-only.wgsl.jinja`](build/webgpu/unique-hash-sort-collected-key-only.wgsl.jinja)
- [`unique-scalar-compact.wgsl.jinja`](build/webgpu/unique-scalar-compact.wgsl.jinja)
- [`unique-scalar-metadata.wgsl.jinja`](build/webgpu/unique-scalar-metadata.wgsl.jinja)
- [`unique-scalar-output.wgsl.jinja`](build/webgpu/unique-scalar-output.wgsl.jinja)
- [`unique-single-class.wgsl.jinja`](build/webgpu/unique-single-class.wgsl.jinja)
- [`unique.wgsl.jinja`](build/webgpu/unique.wgsl.jinja)

## Use with `@huggingface/kernels`

```sh
npm install --save-exact @huggingface/kernels@0.0.1-preview.3
```

Outputs with inferable metadata are allocated automatically. Explicit `outputs` entries request optional results or provide metadata that cannot be inferred from the supplied inputs and attributes.

This example supplies explicit metadata for:

- `y`

The `version: 1` option selects the published kernel contract; it is independent of any operator opset, contrib `since_version`, or model version.
It follows the `v1` branch as fixes land. To pin exact artifact bytes, pass a 40-character commit `revision` instead of `version`.

Replace each `*Data` placeholder with a typed array containing the corresponding input data.

```js
import { getKernel } from "@huggingface/kernels";

const kernel = await getKernel("webgpu-kernels/ai.onnx.Unique", { version: 1 });
// Explicit destinations request optional results or supply metadata that cannot be inferred.
const { y } = await kernel({ x: { data: xData, shape: [1] } }, {
  outputs: { y: { shape: [1], dtype: "float32" } },
});
```
