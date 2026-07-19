# Prefill-specific head grouping for the exact-order fused kernel

## Status

Selected single-replica winner. Branch: `opt2/qk-kvcache-prefill-heads`.

## Hypothesis

The decode-optimal fused kernel groups all 16 Q heads in one Triton program.
At large prefill token counts this raises per-program register pressure without
meaningful launch-overhead savings. Dispatching large inputs with eight heads
and two warps per program should improve prefill/TTFT while leaving the decode
path unchanged.

The runtime split is deliberately small:

- fewer than 512 tokens: 16 heads per program, 4 warps;
- 512 or more tokens: 8 heads per program, 2 warps.

Both paths retain the exact `R0_BLOCK=64` RMS reduction order from the parent
branch.

## Isolated Q/K + MRoPE timings

Median Triton timings on a free H100, in microseconds:

| Tokens | 16 heads / 4 warps | 8 heads / 2 warps | Delta |
| ---: | ---: | ---: | ---: |
| 660 | 10.00 | 8.85 | -11.5% |
| 1,320 | 13.51 | 12.30 | -9.0% |
| 1,980 | 17.10 | 15.56 | -9.0% |
| 4,096 | 28.17 | 25.23 | -10.4% |

## Natural-audio 100-file screen

After excluding the first symmetric JIT-bearing run, two repeated runs on GPU
4 gave:

| Variant | Files/s | Avg latency (s) | Avg TTFT (s) |
| --- | ---: | ---: | ---: |
| Exact-order parent, mean | 4.259 | 3.361 | 0.291 |
| Prefill grouping, mean | 4.270 | 3.345 | 0.263 |
| Relative delta | +0.3% | -0.5% | -9.6% |

## Full 550-file batched gate

| Variant | Files/s | Avg latency (s) | Avg TTFT (s) | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static FP8 paired control | 4.413 | 3.553 | 0.418 | 0.160771 | 0.381654 |
| Exact-order parent | 4.498 | 3.489 | 0.501 | 0.160947 | 0.382164 |
| Prefill grouping | 4.587 | 3.423 | 0.456 | 0.160834 | 0.382863 |

Versus static FP8, the final candidate improves throughput by 3.9% and average
latency by 3.7%. Average TTFT remains 38 ms higher on this heterogeneous run,
but the prefill dispatch removes 45 ms of the parent's 83 ms regression. CER
moves by only +0.006 percentage points and WER by +0.121 percentage points.

## Sequential 100 x 50-second screen

Three runs produced `3.801/0.262/0.058`, `4.020/0.248/0.045`, and
`4.037/0.247/0.044`, where each tuple is files/s, average latency, and average
TTFT. The warmed median is 4.020 files/s, 0.248 s latency, and 0.045 s TTFT.

## Full 550-file sequential gate

| Variant | Files/s | Avg latency (s) | Avg TTFT (s) | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original static FP8 | 0.682 | 1.465 | 0.209 | 0.162277 | 0.384183 |
| Final fused candidate | 0.735 | 1.360 | 0.195 | 0.162639 | 0.384508 |
| Relative/absolute delta | +7.8% | -7.2% | -6.7% | +0.000362 | +0.000325 |

All 550 files completed without failures. CER changes by +0.036 percentage
points and WER by +0.033 percentage points.

## Nsight Systems result

The final node-level capture uses the same 5-second request and 20 dominant
decode-graph replays as the original static-FP8 report.

| Metric per replay | Original static FP8 | Final candidate | Delta |
| --- | ---: | ---: | ---: |
| Kernel launches | 366 | 309 | -15.6% |
| Summed GPU time | 1.627 ms | 1.463 ms | -10.1% |
| Replay envelope | 1.649 ms | 1.475 ms | -10.6% |

The fused Q/K RMSNorm + MRoPE + cache node runs 28 times per replay and takes
75.91 us total. Across the full captured request, kernel launches fall from
8,087 to 6,890 and summed GPU time falls from 39.907 ms to 36.452 ms. The
whole-capture wall envelope is not used as the latency measure because it
contains host gaps between graph replays.

Reports:

- control: `inference/results/nsys/fp8static_c1_5s_node.sqlite` in the main
  checkout;
- candidate: `inference/results/nsys/qk_exact_prefill_heads_c1_5s_node.sqlite`
  in this worktree.

## Decision

Keep the prefill dispatch with the exact-order fused kernel. It improves the
latency-prioritized batched result, narrows the heterogeneous TTFT gap, and
preserves a strong, quality-safe sequential gain. This is the selected
single-replica configuration.
