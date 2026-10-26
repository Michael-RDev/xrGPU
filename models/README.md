# Pinned model export examples

Use the repository environment and pinned ExecuTorch checkout. Export cached
weights into local, inventoried model bundles with:

```sh
python -m models.export --offline --model all
python -m runtime.model_bundle artifacts/model-bundles/minilm-32 --execute-cpu
```

`--offline` requires the pinned weights in the local cache. Outputs, cached
weights, compiled programs and fixtures stay under ignored `artifacts/`.
To bind a verified optimized model kernel package, pass
`--package /absolute/path/to/package` to the exporter. See the
[automatic integration workflow](../docs/automatic-integration.md) for native
library builds and the separate hardware qualification gates.

## MiniLM

[minilm.py](minilm.py) pins `sentence-transformers/all-MiniLM-L6-v2` to revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` and checks the weight SHA-256:
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.
It exports sequence lengths 32, 128 and 256 with `input_ids`, `attention_mask`
and `token_type_ids`, using masked mean pooling and L2 normalization. The pinned
tokenizer handles padding and truncation; its assets are included in each bundle.

## MobileNetV2

[mobilenetv2.py](mobilenetv2.py) implements the MobileNetV2 1.0 architecture with
pinned `IMAGENET1K_V1` weights and SHA-256:
`b03531047ffacf1e2488318dcd2aba1126cde36e3bfe1aa5cb07700aeeee9889`.
Its normalization constants and seeded reference inputs are recorded in source
and exported bundle metadata. Preserve [the upstream license](TORCHVISION_LICENSE.txt).

CPU reference execution, supported Vulkan lowering, physical device correctness,
performance and sustained XR qualification are separate results. An exported
bundle does not qualify a replacement rule or the standalone native operator catalog.
