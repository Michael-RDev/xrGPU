# Quest 3 MiniLM feasibility result — 2026-10-15

**Smaller repeatable gain.** Candidate 2 (2×8 tiles, texture weights) reduced
median end-to-end latency by **2.83%** in the first session and **2.60%** in a
fresh-process confirmation session. Every pair improved; both model and
end-to-end p95 stayed within the 5% regression limit. Neither session reached
the 10% target. Candidates 1 and 3 increased latency by 45.94% and 15.72%.

This demonstrates a modest whole-model improvement for the existing FP32 kernels
at 128 tokens. Sustained acceptance and VR frame timing remain untested. No
automatic selection was written; the normal inference path still defaults to stock.

## Comparison

All values below are milliseconds. Every row passed tokenizer, embedding,
normalization, and retrieval checks. CPU and initial stock baseline each have
100 samples; every paired arm has 400 samples. Custom dispatch is the number of
eligible model operations using a custom kernel, corroborated by GPU shader traces.

| Execution | Correct | Custom dispatch | Model median | Model p95 | End-to-end median | End-to-end p95 |
|---|---|---:|---:|---:|---:|---:|
| CPU reference | Pass | 0 | 42.348 | 46.899 | 42.866 | 47.410 |
| Initial stock Vulkan baseline | Pass | 0 | 32.160 | 35.188 | 32.831 | 35.788 |
| Stock paired with candidate 1 | Pass | 0 | 32.351 | 34.890 | 32.915 | 35.463 |
| Candidate 1: 4×8 texture weights | Pass | 12 | 47.441 | 50.858 | 48.036 | 51.429 |
| Stock paired with candidate 2 | Pass | 0 | 32.477 | 34.904 | 33.023 | 35.509 |
| Candidate 2: 2×8 texture weights | Pass | 12 | 31.541 | 33.696 | 32.088 | 34.312 |
| Stock paired with candidate 3 | Pass | 0 | 32.327 | 34.942 | 32.885 | 35.521 |
| Candidate 3: 4×8 buffer weights | Pass | 12 | 37.475 | 39.237 | 38.054 | 39.814 |
| Confirmation: stock | Pass | 0 | 32.383 | 34.855 | 33.066 | 35.601 |
| Confirmation: candidate 2 | Pass | 12 | 31.591 | 33.814 | 32.206 | 34.470 |

Speedup is matched stock median divided by candidate median; above 1 is faster.
Percentage reduction is `100 × (1 − candidate / stock)`. Each candidate is
compared with its own interleaved stock trials, not the initial baseline.

| Candidate/session | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Median latency reduction |
|---|---:|---:|---:|---:|---:|
| 1 | 0.6834× | 0.6880× | 0.6872× | 0.6845× | −45.94% |
| 2 | 1.0264× | 1.0313× | 1.0302× | 1.0260× | +2.83% |
| 3 | 0.8644× | 0.8666× | 0.8659× | 0.8608× | −15.72% |
| 2 confirmation | 1.0322× | 1.0228× | 1.0210× | 1.0265× | +2.60% |

Each candidate also passed all 12 layer cases on both stock and candidate arms,
with 20 warmups and 100 checked samples per case. Six supported cases dispatched
the custom shader; six unsupported cases correctly fell back to stock.

## What limits the gain

Across candidate 2's 400-sample comparison, median model execution fell from
32.477 to 31.541 ms. The measured submit-and-wait interval fell from 30.998 to
30.106 ms. Median input staging was about 0.0085 ms and output staging about
0.0063 ms for both arms. Submission-only cost is not separately measured;
submit-and-wait includes GPU execution. Staging is not a meaningful bottleneck
in this workload.

GPU traces identify the targeted feed-forward linear operations as meaningful
work. Across four stock profiles, their median summed operation time was
8.667 ms, about 36.8% of summed GPU operation times; candidate 2 reduced that
sampled total to 6.773 ms. All linear operations together remain the largest
operator group. Softmax, layout/copy operations, and attention matrix products
also contribute. Candidates 1 and 3 show that a different tile or weight-storage
choice can substantially worsen overall latency.

These detailed profiles capture the first measured inference after each warmup,
which is faster than the median inference at the 100 ms cadence. Their operation
totals are **not wall-clock latency or an additive critical-path breakdown**.
Use them to choose further investigations, not to predict a guaranteed speedup.
Further linear-kernel and layout work is plausible, but a 10% whole-model gain
has not been demonstrated. New kernels, FP16, and quantization were outside this pass.

## Conditions and provenance

- Quest 3, Adreno 740, device serial `2G97C5ZJ0300S4`, Vulkan driver `2150912009`.
- Batch 1, 128 tokens, FP32; unchanged MiniLM weights and tokenizer.
- Four pairs alternating stock/candidate order, 20 warmups and 100 samples per arm,
  at the existing 100 ms inference cadence. Confirmation used a fresh app process.
- Every accepted paired sample reported thermal status 0 and charging false;
  USB remained connected. The first baseline's charging transition was rejected.
- A temporary developer proximity override prevented wear-sensor sleep during
  the test. Normal wear detection was restored afterward. Ambient temperature
  and system clock frequencies were not measured.
- APK SHA-256: `39c905d1acc2a88de20be318b873f1ec4f1b4ccc93fa54ac58c319467ef5d4dc`.
- Runtime revision: `e4d02f41f7909e8ed5bf4a14ffc520d733453d9f`.
- Kernel/build revision: `a088e1775edf60c5b52534bf7c18ec6d30dbc1e51bcb450215d54377ebae488b`.
- Model revision: `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`.
- Vulkan model SHA-256: `f9691b5b222298c9ceacb811e21e3fb4b651713161e31ca000d0f0d60745d1c9`.

The complete OS fingerprint and per-trial identities are recorded in the raw JSON.
Local evidence is under
`artifacts/device-results/quest3/session3-128-20261015/`: `feasibility.html`,
`summary.json`, `evidence-manifest.json`, individual reports, command statuses,
requests, and logs. The summary identifies every accepted source run. Earlier
canceled attempts and the rejected baseline remain saved and excluded.

Host validation passed 54 Python tests, including exported CPU reference
execution, Windows asset paths, native dispatch, command failures, cancellation,
and feasibility/full-acceptance separation. All 69 Java tokenizer fixtures passed.
The Windows shader/native/APK build passed; all 103 packaged assets, native
fingerprints, APK signature, and offline/debug manifest checks passed.

Reproduce after installing the debug APK:

```powershell
. artifacts/toolchain/env.ps1
.venv/Scripts/python.exe -m runtime.device_benchmark --serial 2G97C5ZJ0300S4 --length 128 --keep-awake
```
