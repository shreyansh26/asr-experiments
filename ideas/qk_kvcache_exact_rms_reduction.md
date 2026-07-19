# Exact-order RMS reduction for the fused Q/K + MRoPE + KV-cache kernel

## Hypothesis

The first fused kernel reduced all 128 head dimensions with one `tl.sum`. The
stock Inductor decode graph instead launches `triton_red_fused_4` with
`XBLOCK=64, R0_BLOCK=64`: it accumulates `x[d]^2 + x[d + 64]^2` and then
reduces the resulting 64 lanes. Matching that floating-point addition order
should retain the launch reduction while reducing transcript drift.

## Isolated numerical check

The candidate reshapes the 128 squared values to `[2, 64]`, reduces the first
axis, and then reduces the 64 lanes. Against a `torch.compile` stock-style
reference:

| Tokens | Q BF16 differences | K BF16 differences |
| ---: | ---: | ---: |
| 1 | 0 / 2,048 | 0 / 1,024 |
| 16 | 0 / 32,768 | 0 / 16,384 |
| 128 | 2 / 262,144 | 0 / 131,072 |

## Batched 100 x 50-second screen

Six runs used one replica on GPU 4 with 16 workers. Medians are shown against
the same-GPU static-FP8 control collected for the parent experiment.

| Variant | Files/s | Avg latency (s) | Avg TTFT (s) |
| --- | ---: | ---: | ---: |
| Static FP8 control | 22.107 | 0.675 | 0.128 |
| Exact-order fused kernel | 23.329 | 0.639 | 0.128 |
| Relative delta | +5.5% | -5.4% | approximately flat |

The six exact-order runs were:

`20.958/0.718/0.172`, `23.200/0.644/0.127`,
`23.458/0.633/0.114`, `23.843/0.626/0.118`,
`22.732/0.653/0.151`, and `23.647/0.630/0.129`, where each tuple is
files/s, average latency, and average TTFT.

## Full 550-file batched gate

| Variant | Files/s | Avg latency (s) | Avg TTFT (s) | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static FP8 paired control | 4.413 | 3.553 | 0.418 | 0.160771 | 0.381654 |
| Exact-order fused kernel | 4.498 | 3.489 | 0.501 | 0.160947 | 0.382164 |
| Delta | +1.9% | -1.8% | +83 ms | +0.000176 | +0.000510 |

The quality shift is only +0.018 percentage points CER and +0.051 percentage
points WER. This is much smaller than the parent direct-128-reduction variant
(about +0.30 percentage points on both metrics).

## Decision

Keep this arithmetic order. It is the leading quality-safe kernel and restores
TTFT parity on the controlled uniform screen. The remaining full-corpus TTFT
variance is being handled separately as a prefill tuning problem; the decode
configuration remains unchanged.
