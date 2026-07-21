# Natural-only audio CUDA graphs

Branch: `opt7/audio-natural-only-cudagraph`

Promoted into: `promote/audio-prefix-shared-suffix-bucketed`

Date: 2026-07-21

## Decision

The promoted configuration CUDA-graph captures only canonical 29--30 second
server chunks. It does **not** capture the workload-derived 20--21 second tail
family. Those tails, and every other unsupported final-chunk duration, use the
existing eager implementation.

This is a generality tradeoff, not a universal speedup. On the varying
550-file workload, removing tail graphs was essentially neutral within the
noise of runs on two physical H100s of the same SKU. On the intentionally
tail-heavy fixed-50 workload, it was measurably slower, especially in batched
mode. The natural-only configuration is promoted to avoid encoding a narrow
50-second benchmark artifact in the graph inventory, with that fixed-50 cost
recorded explicitly.

## Implementation

The general CPU metadata and Triton valid-row pack remain unchanged and still
support arbitrary valid audio lengths. Only CUDA-graph admission changes:

- `audio_prefix_cudagraph_patch.py` sets `_TAIL_PACKED_ROWS` to an empty set,
  admits only the 104 exact natural feature keys `F=2897..3000`, and limits
  graph signatures to the 14 natural post-CNN rows `M=377..390`.
- `audio_suffix_cudagraph_patch.py` sets `_TAIL_ROWS` to an empty set, removes
  the 273-row tail bucket, and retains one 390-row graph bucket for the 14
  natural rows.
- The helper benchmarks use only natural representatives.
- Prefix and suffix tests explicitly assert that former tail rows 263, 266,
  269, and 271 are rejected and stay eager.

The accepted natural prefix key still verifies the complete tensor shape,
stride, raw chunk lengths, pack lengths and offsets, feature and post-CNN
length tuples, and cumulative attention boundaries. This change does not
weaken graph guards.

## Coverage after the change

```text
F=2897..3000, M=377..390
  prefix graph: yes, after per-key probation
  suffix graph: yes, through the 390-row bucket

all other F and M, including M=263..273
  prefix graph: no
  suffix graph: no
  execution: accepted eager fallback
```

For a typical 50-second file split into approximately `29.x + 20.x` seconds,
the first chunk can use both graphs and the final chunk is eager. For a general
long file, each 29--30 second non-final chunk can use the graphs independently;
the arbitrary final chunk remains eager unless it independently falls in the
natural range.

## Validation gates

Focused CPU tests:

```bash
UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
uv run python -m unittest \
  tests.test_audio_prefix_cudagraph_patch \
  tests.test_audio_suffix_cudagraph_patch \
  tests.test_audio_prefix_suffix_cudagraph_patch
```

Result: 42 tests passed.

GPU4, NVIDIA H100 80GB HBM3:

- suffix helper passed all 14 rows `M=377..390`, bitwise equality, exact
  268-kernel ordering, alternating-key replay, and concurrent gates;
- chained prefix-plus-suffix helper passed for `M=384, F=2956`, including
  bitwise equality, observation-eight admission, and concurrency;
- combined eager host time was `6713.468 us` versus `183.188 us` for the
  graph call; eager CUDA time was `6750.800 us` versus `2522.080 us` including
  input copy, graph replay, and output clone.

The server log admitted only natural shapes and showed no tail probation or
capture. All service runs completed with zero failed or timed-out requests.

## Canonical AGENTS.md benchmark matrix

All four requested commands used the branch snapshot's default 20 warm-up
files. The candidate server ran on GPU4 at port 8094. Runs were executed in the
order shown on one server; later runs therefore inherited already compiled
kernels and any admitted natural graphs. Each benchmark resets vLLM's prefix
cache, but not the out-of-tree audio graph caches.

| Mode and workload | Files | Throughput | Mean latency | Latency p50/p95/p99 | Mean TTFT | TTFT p50/p95/p99 | CER | WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batched B16, fixed 50 s | 100 | 21.891 files/s | 0.678 s | 0.666 / 1.070 / 1.208 s | 0.204 s | 0.164 / 0.461 / 0.500 s | n/a | n/a |
| Batched B16, natural | 550 | 5.717 files/s | 2.738 s | 2.550 / 4.996 / 6.759 s | 0.613 s | 0.543 / 1.231 / 1.873 s | 16.47% | 38.62% |
| Sequential, fixed 50 s | 100 | 3.848 files/s | 0.259 s | 0.251 / 0.428 / 0.457 s | 0.052 s | 0.052 / 0.075 / 0.083 s | n/a | n/a |
| Sequential, natural | 550 | 0.763 files/s | 1.310 s | 1.224 / 2.489 / 3.060 s | 0.195 s | 0.198 / 0.310 / 0.353 s | 16.23% | 38.42% |

## Adjacent natural-workload comparison

The closest prior tail-graph service rows were produced earlier the same day
on GPU1, also an NVIDIA H100 80GB HBM3. The software stack and 20-file warm-up
policy match, but the physical GPU and server lifetime do not. Treat small
differences as run-to-run evidence rather than isolated kernel causality.

### Batched B16, 550 natural files

| Metric | Tail graphs | Natural-only | Change |
|---|---:|---:|---:|
| Throughput | 5.656 files/s | 5.717 files/s | +1.08% |
| Mean latency | 2.772 s | 2.738 s | -1.23% |
| Latency p50 | 2.604 s | 2.550 s | -2.07% |
| Latency p95 | 4.966 s | 4.996 s | +0.60% |
| Latency p99 | 6.603 s | 6.759 s | +2.36% |
| Mean TTFT | 0.681 s | 0.613 s | -9.99% |
| TTFT p50 | 0.627 s | 0.543 s | -13.40% |
| TTFT p95 | 1.296 s | 1.231 s | -5.02% |
| TTFT p99 | 1.843 s | 1.873 s | +1.63% |

CER moved from `16.25%` to `16.47%` (`+0.22` percentage points), and WER
moved from `38.39%` to `38.62%` (`+0.23` points). This is modest, but the
natural-only batched run is not bitwise transcript-identical to the prior run.

### Sequential, 550 natural files

| Metric | Tail graphs | Natural-only | Change |
|---|---:|---:|---:|
| Throughput | 0.766 files/s | 0.763 files/s | -0.39% |
| Mean latency | 1.305 s | 1.310 s | +0.38% |
| Latency p50 | 1.212 s | 1.224 s | +0.99% |
| Latency p95 | 2.499 s | 2.489 s | -0.40% |
| Latency p99 | 3.051 s | 3.060 s | +0.29% |
| Mean TTFT | 0.199 s | 0.195 s | -2.01% |
| TTFT p50 | 0.200 s | 0.198 s | -1.00% |
| TTFT p95 | 0.310 s | 0.310 s | 0.00% |
| TTFT p99 | 0.341 s | 0.353 s | +3.52% |

CER moved from `16.26%` to `16.23%` (`-0.04` percentage points), and WER
moved from `38.45%` to `38.42%` (`-0.03` points).

## Matched fixed-50 stress comparison

The historical fixed-50 tail-graph rows used 100 warm-up files. The branch
snapshot's wrapper no longer exposes that option, so supplementary matched
runs used the same newer runner from the main checkout read-only against the
natural-only server. These numbers isolate the steady, tail-heavy workload
more fairly than comparing it with the canonical default-20 rows.

### Batched B16, 100 measured files after 100 warm-ups

| Metric | Tail graphs | Natural-only | Change |
|---|---:|---:|---:|
| Throughput | 32.134 files/s | 30.211 files/s | -5.98% |
| Mean latency | 0.424 s | 0.489 s | +15.33% |
| Latency p50 | 0.360 s | 0.455 s | +26.39% |
| Latency p95 | 0.751 s | 0.834 s | +11.05% |
| Latency p99 | 0.863 s | 0.915 s | +6.03% |
| Mean TTFT | 0.096 s | 0.114 s | +18.75% |
| TTFT p50 | 0.078 s | 0.089 s | +14.10% |
| TTFT p95 | 0.202 s | 0.246 s | +21.78% |
| TTFT p99 | 0.208 s | 0.320 s | +53.85% |

### Sequential, 100 measured files after 100 warm-ups

| Metric | Tail graphs | Natural-only | Change |
|---|---:|---:|---:|
| Throughput | 5.161 files/s | 5.028 files/s | -2.58% |
| Mean latency | 0.193 s | 0.198 s | +2.59% |
| Latency p50 | 0.174 s | 0.181 s | +4.02% |
| Latency p95 | 0.312 s | 0.312 s | 0.00% |
| Latency p99 | 0.369 s | 0.379 s | +2.71% |
| Mean TTFT | 0.030 s | 0.031 s | +3.33% |
| TTFT p50 | 0.030 s | 0.031 s | +3.33% |
| TTFT p95 | 0.036 s | 0.038 s | +5.56% |
| TTFT p99 | 0.042 s | 0.042 s | 0.00% |

## Interpretation

- The 20--21 second graphs were not dead code: they materially accelerated the
  fixed-50 batched stress workload.
- They were not required to preserve the aggregate benefit on the varying
  natural workload. Natural-only results remained close to the adjacent
  tail-graph run because eligible 29--30 second chunks still occur frequently.
- Removing the tail family reduces graph inventory from 25 prefix signatures
  and two suffix buckets to 14 prefix signatures and one suffix bucket. It also
  eliminates probation, capture, and persistent graph-pool state for the
  20--21 second family.
- The promoted choice favors a simpler, less benchmark-specific graph policy.
  If 50-second or approximately 20--21 second final chunks become a documented
  production hot distribution, the measured fixed-50 regression is strong
  evidence for re-enabling the guarded tail family.
