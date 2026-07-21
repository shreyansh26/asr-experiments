# Batched kernel optimization round 3

## Executive summary

This round started from main commit `b5e3955`, whose best path combines static
per-tensor FP8 decoder quantization with the exact Q/K RMSNorm, MRoPE, paged
KV-cache write, and prefill head-grouping kernel. The work was restricted to
out-of-tree patches and separate experiment branches; PyTorch, vLLM, Triton,
and the model were not upgraded or replaced.

The accepted result is `opt3/audio-cpu-metadata-pack`: implementation commit
`299bb3a`, the `int32` length robustness fix and regression test in `fddce01`,
and atomic prerequisite installation in `92c837e`.
It keeps audio lengths and derived attention metadata on the CPU, removes the
remaining device-to-host metadata synchronizations, and replaces dynamic
boolean row packing with one exact Triton copy kernel. Against a fresh full
main control it improved the latency-priority metric by 5.65% and throughput
by 5.91%, while CER and WER were slightly better. Full-set TTFT regressed by
12.87%, which is the important tradeoff.

| Full 550-file B16 run | Throughput | Avg latency | Avg TTFT | CER | WER |
|---|---:|---:|---:|---:|---:|
| Fresh main `b5e3955` | 4.485 files/s | 3.502 s | **0.474 s** | 0.163900 | 0.385612 |
| CPU metadata + Triton pack `299bb3a` | **4.750 files/s** | **3.304 s** | 0.535 s | **0.163614** | **0.384638** |
| Candidate delta | **+5.91%** | **-5.65%** | +12.87% | -0.000286 | -0.000974 |

The preceding CPU-max-seqlen branch removed 24 repeated FlashAttention scalar
readbacks per audio pass. The metadata branch then removed all seven remaining
steady-state stream synchronizations per pass. Together they move the audio
encoder's control metadata off the critical CUDA stream without changing its
convolution, attention, MLP, or projection arithmetic.

## Measurement discipline

- Only GPU 1 was free during the measured work, so service candidates ran one
  at a time. Other users' GPU processes were never stopped or modified.
- Every measured service run reset both `/reset_mm_cache` and
  `/reset_encoder_cache`; the benchmark runner reset the prefix cache.
- Uniform runs used 100 measured 50-second files, 20 warmups, and 16 workers.
- Full runs omitted `--num-files` and `--uniform-audio-length`; the runner
  completed and scored 550 measured files after 20 warmups.
- Input payload preparation happened before worker submission and outside the
  timed inference interval.
- Each experiment used a unique `--output-root`; generated predictions and CSV
  rows were excluded from source commits.
- Helper wins were not promoted without real-service evidence. Numerically
  changed candidates additionally required full CER/WER validation.

The common uniform command shape was:

```bash
curl -fsS -X POST http://127.0.0.1:8091/reset_mm_cache
curl -fsS -X POST http://127.0.0.1:8091/reset_encoder_cache

UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
uv run inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --overwrite \
  --uniform-audio-length 50 \
  --num-files 100 \
  --input-root /mnt/ssd1/shreyansh/home_dir/asr_experiments/data/prepared_data \
  --output-root /tmp/UNIQUE_EXPERIMENT_OUTPUT \
  --base-url http://127.0.0.1:8091/v1
```

The full-set command used the same arguments without the two uniform-subset
flags.

## Fresh control and trace

Three fresh 100 x 50-second main controls were:

```text
throughput: 20.629, 21.350, 21.720 files/s  (mean 21.233)
latency:     0.728,  0.705,  0.689 s        (mean 0.7073)
TTFT:        0.198,  0.171,  0.162 s        (mean 0.1770)
```

The matched B16 Nsight node trace was generated locally in the main worktree:

```text
inference/results/nsys/fp8static_qk_kvcache_fuse_b16_50s_node.nsys-rep
inference/results/nsys/fp8static_qk_kvcache_fuse_b16_50s_node.sqlite
```

These generated binary artifacts are intentionally not source-controlled. The
captured files used for this analysis were 12,235,280 and 28,712,960 bytes,
with SHA-256 digests `4dbbe04ebe718fbc6bf5b2a9f7716b2541a80597c788b310de566e07ab24faf8`
and `b312734369b6b3e13799bdfcc85689de6addb62172a74c83ed5fe5c87c0bfaad`
respectively. The aggregate evidence needed for the decision is retained
below even when those local profiling artifacts are absent from a clean clone.

Its dominant process-local graph had 246 replays, 309 nodes per replay,
1.739 ms of summed kernel time, and a 1.863 ms replay envelope. Important
per-replay regions were:

| Region | Time |
|---|---:|
| FA3 main kernels | 489.287 us |
| Gate/up FP8 GEMMs | 353.327 us |
| Down FP8 GEMMs | 216.295 us |
| QKV FP8 GEMMs | 158.260 us |
| O-projection FP8 GEMMs | 133.419 us |
| Fused Q/K RMSNorm + MRoPE + cache | 80.068 us |
| FA3 combine kernels | 72.419 us |
| Attention RMSNorm/static quant | 67.132 us |
| SwiGLU/static quant | 50.634 us |
| Attention-output static quant | 42.356 us |

The current attention-output quantization ceiling is therefore 42.356 us per
graph, not the older 91.8 us estimate in `optimization_ideas.md`.

## Accepted audio metadata path

### CPU max seqlen

Branch: `opt3/audio-cpu-maxseqlen`

Commits: `96443db`, documentation `9e78387`

The production convolution geometry proves a maximum attention sequence length
of 104 for this model and input window. Returning a cached CPU `int32` scalar
lets the installed FlashAttention wrapper keep its `.item()` call without a
CUDA synchronization. Nsight changed from 1,426 stream synchronizations across
46 stock passes (31/pass) to 455 across 65 candidate passes (7/pass).

Six fresh uniform runs averaged 21.641 files/s, 0.695 s latency, and 0.1872 s
TTFT. The full run produced 4.619 files/s, 3.397 s latency, 0.528 s TTFT,
0.160686 CER, and 0.382274 WER.

### CPU metadata plus exact valid-row pack

Branch: `opt3/audio-cpu-metadata-pack`

Commits: `299bb3a`, robustness fix `fddce01`, atomic install `92c837e`

Detailed note: [audio_cpu_metadata_pack.md](audio_cpu_metadata_pack.md)

The CUDA helper produced the exact `(5718, 1024)` BF16 output and changed:

| Helper path | Host call | CUDA time | Stream synchronizations |
|---|---:|---:|---:|
| Installed-style metadata | 156.688 us | 172.496 us | 3 |
| CPU metadata + Triton pack | **109.422 us** | **123.424 us** | **0** |

Six uniform candidate runs averaged 22.1708 files/s, 0.6743 s latency, and
0.1928 s TTFT. A three-run time-adjacent CPU-max return averaged 21.186
files/s, 0.7080 s latency, and 0.1987 s TTFT, so the candidate improved those
means by 4.65%, 4.76%, and 2.94% respectively. The full main comparison is in
the executive summary.

Decision: accept for the latency-prioritized batched track. The full TTFT
regression must remain visible in any README or deployment handoff.

## Evaluated but not promoted

| Branch / commit | Idea | Evidence and decision |
|---|---|---|
| `opt3/audio-full-row-fastpath` / `5eb45f5` | Alias an already-contiguous all-full CNN tensor instead of copying it | Exact helper removed allocation, H2D, and the pack kernel, but a time-adjacent service A/B regressed throughput 4.27%, latency 4.52%, and TTFT 14.80%. Rejected; detailed note: `opt3/audio-full-row-fastpath:ideas/audio_full_row_fastpath.md`. |
| `opt3/audio-fa3-scheduler-hoist` / `cabc609` | Prepare FA3 scheduler metadata once instead of in 24 layers | Exact and reduced preparation count 24 to 1, but isolated helper paths regressed 2.4-4.5%. Rejected before service. |
| `opt3/decode-blocktable-dirty-copy` / `0ba07f9` | Copy only dirty block-table rows | Three uniform runs averaged 21.166 files/s, 0.7080 s latency, and 0.1777 s TTFT versus fresh main 21.233/0.7073/0.1770. No repeatable win. |
| `opt3/torch-fp8-shape-hybrid` / `d48119e` | Use `torch._scaled_mm` for selected M=1/2/4/8/16 FP8 GEMMs | Helpers improved QKV, O, and gate/up by roughly 4-19%, but the full run regressed to 4.448 files/s, 3.532 s latency, and 0.518 s TTFT. CER/WER were 0.160394/0.382199. Rejected. |
| `opt3/audio-residual-layernorm` / `1bdcb57` | Fuse residual add and LayerNorm | Helper/service gate did not justify promotion. |
| `opt3/audio-pad-sequence-fastpath` / `2cc880b` | Replace one-copy-per-chunk padding with one Triton kernel | Helper was favorable, but service was not. Rejected. |
| `opt3/audio-conv-chunk-size` / `40b154d` | Increase convolution chunk size | Trace proved every captured pass already used one convolution stack and never reached the 500-chunk cap. Workload-inapplicable. |
| `opt3/audio-conv-channels-last` | Keep convolution data channels-last | The remaining removable layout-transform ceiling was too small and no reliable improvement emerged. No source commit. |
| `opt3/qk-paired-rotary-loads` / `e5fd1c9` | Pair rotary loads | No service improvement. Rejected. |
| `opt3/qk-batched-decode-groups` / `92c9534` | Decode-specific Q/K grouping | No service improvement. Rejected. |
| `opt3/qk-no-k-output` / `38e1e8b` | Avoid unused K output | No service improvement. Rejected. |
| `opt3/cutlass-batch-invariant-tile` / `217fb99` | Alternate compiled CUTLASS small-M tile | Slower for this shape mix. Rejected. |
| `opt3/fa3-splitk-cap` / `8a9b504` | Limit FA3 split count | No service improvement. Rejected. |

Earlier branches had already exhausted generic Torch/Triton linear backends,
Triton/FA4/FlashInfer attention replacements, FP8 KV cache, FP8/INT8 heads,
sampler/argmax variants, generic vLLM fusion flags, CUDA-graph tuning, and
server/runtime knobs. Those were not repeated.

## GPU-gated exact convolution fusions

Branch: `opt3/audio-conv1-bias-gelu`

Implementation commit: `530853e`

This candidate runs cuDNN conv1 without bias, then performs the BF16 bias add
and exact-erf GELU in one Triton kernel. A first helper exposed an extremely
rare last-ULP mismatch caused by Triton 3.6's bundled `__nv_erff`, not by the
formula or rounding order. A per-kernel override to the already-installed
`nvidia-cuda-nvcc-cu12==12.9.86` libdevice, guarded by exact SHA-256
`9e84b504f25d26ec775ef91f7122e68dac77fa1f776f1044128f5028356265d8`,
restored bitwise equality over all 65,280 finite BF16 values and at
C=21/90/448.

The exact helper changed stock-to-candidate CUDA time by:

```text
C=21:  278.016 -> 191.952 us  (-31.0%)
C=90: 1137.888 -> 764.336 us  (-32.8%)
C=448:5606.992 -> 3766.976 us (-32.8%)
```

Promotion is pending a real B16 service A/B. No package was installed or
upgraded; the alternate bitcode file was already present in the shared venv.

A separate follow-up branch, `opt3/audio-all-conv-bias-gelu` at commit
`1a28189`, generalizes the same guarded exact post-op to all three convolution
geometries:

```text
conv1: [C, 1,   128, 100] -> [C, 480, 64, 50]
conv2: [C, 480,  64,  50] -> [C, 480, 32, 25]
conv3: [C, 480,  32,  25] -> [C, 480, 16, 13]
```

It incorporates the accepted `int32` metadata and atomic-install fixes, pins
the final copied-source hash, fails loud on invalid environment values, and
rejects ambiguous in-process conv1/all-conv mode switches. Its 45 combined CPU
tests, fake-tensor/export custom-op gates, launcher checks, and fresh install
gate pass. The dedicated CUDA helper and service launcher are committed, but
were not run because no GPU was safely available after the branch became
ready. Detailed note: `opt3/audio-all-conv-bias-gelu:ideas/audio_all_conv_bias_gelu.md`.

## GPU-gated clustered split-K down projection

Branch: `opt3/decode-down-cluster-splitk`

CPU/compiler commit: `9d22bd8`

The current H100 small-M down projection is FP8 E4M3
`[M,6144] x [6144,2048]`, M in `{1,2,4,8,16}`. Its 32-CTA launch covers only
24.2% of the 132 SMs and averages 7.725 us/layer. The independent CuTeDSL 4.5.2
prototype uses four K-quarter CTAs per output tile, a `(4,1,1)` cluster, DSMEM
partial exchange, and a deterministic rank-0 reduction. The intended launch is
128 CTAs, one near-full H100 wave, one kernel, and zero global workspace.

All M=1/2/4/8/16 specializations lower successfully with `cute.compile` for
`sm_90a` while CUDA is hidden, and 12 integration/export/fallback/prewarm tests
pass. H100 runtime loading, correctness, graph replay, and timing remain
pending. The kill gate is bitwise or tightly characterized numerical parity
plus a net target at or below roughly 6.5 us/layer across every M bucket. A
second global reduction kernel is not acceptable because it is likely to erase
the gain.

## Remaining high-value opportunities

1. Native FA3 static-FP8 output epilogue: remove 28 quant nodes and up to
   42.356 us of kernel time per decode graph while preserving the observable
   FP32 -> BF16 -> scaled/clamped FP8 order. This targets the active SM90 paged
   split path, unlike the rejected FA4 experiment. Building the pinned source
   currently needs an isolated CMake >=3.26 and CUDA-compatible toolchain; the
   host has CMake 3.22.1, so no install was attempted without permission.
2. Gate/up GEMM epilogue with SwiGLU and static FP8 output. CUTLASS example 113
   demonstrates the required paired-half visitor. The exposed node is 50.634
   us/graph, but the custom epilogue must add less than about 0.9 us/layer to
   clear a meaningful service gate.
3. Audio FC1+bias+exact-GELU epilogue. The separate transformer GELU kernels
   cost 2.162 ms over the measured audio wave. A custom CUTLASS EVT must retain
   the intermediate BF16 rounding boundary and beat 13.858 us/layer for the
   combined FC1+GELU region.
4. Bucketed exact audio-encoder CUDA graphs. The audio tower remains eager, and
   the common M=264-273 bucket covers 13 of 23 passes. This is a launch-overhead
   optimization rather than new arithmetic and requires careful dynamic
   `cu_seqlens` and padded-tail validation.

## Current integration state

Nothing in this round has been merged into `main`. The accepted metadata branch
is the current promotion candidate. At the final runtime gate on 2026-07-20,
all eight local H100s were occupied by unrelated jobs; no process was stopped,
no shared-GPU timing was taken, and vLLM memory utilization was not reduced to
force coexistence. The conv1, all-conv, and clustered split-K branches therefore
remain independent and explicitly unpromoted until their CUDA helper/service
gates can run on an idle GPU. If one passes, compose it from the accepted
metadata branch in a separate integration branch and rerun the full quality
set.
