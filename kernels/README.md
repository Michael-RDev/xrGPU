# Kernel organization

Each operation owns a shared mathematical contract and reference implementation.
Backend/device implementations live beneath it:

```text
<operation>/
├── contract.json
├── reference.py
└── <backend>/<implementation>/
    ├── manifest.json
    ├── dispatch.h
    └── shader sources and variant definitions
```

The current implementation is `linear/vulkan/quest3`. Its manifest describes
precision, layouts, shapes, and variants; its dispatch policy supplies those
settings to the runtime. Variant IDs belong to a kernel set and zero is reserved
for the upstream baseline. Shader filenames/prefixes must be globally unique
within the ExecuTorch Vulkan backend.

Device detection belongs in `devices/`, not in shared operation contracts.
Reusable shader code can be factored out when a second implementation actually
needs it. Sharing code does not share acceptance evidence.
