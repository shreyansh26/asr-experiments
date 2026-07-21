# Exact-shape audio suffix CUDA graph experiment

## Status

This is a CUDA-unvalidated research candidate on branch
`opt3/audio-suffix-cudagraph-probation`, based on concurrent-safe suffix graph
commit `f6c6967436ce690c84f7bc9095db3dc33ff0a4b5`. No GPU helper, service
benchmark, or CER/WER run has been performed yet. The launcher must not be
treated as a winner until all gates below pass.

No PyTorch, vLLM, Triton, or other dependency version changed.

## Narrow graph boundary

The accepted path still performs all of the following eagerly:

1. CPU length and chunk metadata construction;
2. audio splitting and padding;
3. all three convolutions and GELUs;
4. convolution projection and positional addition;
5. valid-row packing into contiguous `[M, 1024]` BF16 storage;
6. `cu_seqlens` transfer and exact-key cache lookup;
7. stable-input copies and all output ownership outside capture.

Only the existing post-pack suffix is passed to CUDA stream capture, through
the same `run_audio_suffix_eager` callable used by fallback and validation:

```text
24 encoder layers
-> ln_post
-> proj1
-> act
-> proj2
-> [M, 2048]
```

The captured callable therefore has the same suffix kernels and ordering as
eager execution. It does not include padding, convolutions, CPU/GPU metadata
construction, pack allocation, or cache allocation.

## Exact key and stable state

This bounded experiment accepts only the observed common batched family:

```text
cu_seqlens.numel() == 4
M in {264, 265, 267, 268, 270, 272, 273}
```

Each encoder instance owns at most seven graph entries. Sequential warmup calls
have `cu_seqlens.numel() == 2`, so they remain eager and cannot fill or starve
the intended B16 keys. Each supported exact key also owns a probation counter.
There is no eviction, LRU, or shape padding. Once full, any unseen exact-content
key runs the identical eager suffix.

The key contains:

```text
(
  M,
  cu_seqlens.numel(),
  hidden dtype,
  hidden device type/index,
  accepted CPU max_seqlen value,
  every CPU-built cumulative cu_seqlens value,
)
```

Including the full cumulative-length tuple is required: two requests can have
the same `M` and number of sequences but different segment boundaries, which
changes variable-length attention. The tuple comes directly from the accepted
CPU metadata builder, so keying does not add a CUDA readback.

Each admitted key retains stable hidden-state, `cu_seqlens`, `max_seqlen`, and
graph-output tensors plus the `CUDAGraph` for the encoder lifetime. Because a
later replay overwrites the stable graph output, every replay clones it after
graph launch and returns independently owned `[M, 2048]` storage. Reuse is
safe across caller CUDA streams: each entry owns one execution stream, and a
per-entry host lock serializes the enqueue transaction `input copy -> graph
replay -> output clone`. Asynchronous caller-to-entry and entry-to-caller stream
waits preserve producer/consumer ordering without a device synchronization on
the hot path. A per-cache lock also serializes cold entry creation, and a
one-time global lock prevents concurrent first calls from attaching different
caches to the same encoder. An unsupported layout, training/grad mode, nested
capture, unexpected 24-layer/1024-wide contract, cache overflow, capture
exception, or output other than `[M, 2048]` falls back to eager execution.

## Probation, admission, and exactness

The fixed probation threshold is eight observations per full exact key. The
counter update is protected by the existing cache lock:

```text
observations 1-7 -> identical eager suffix
observation 8    -> capture and bitwise admission exactly once
observation 9+   -> admitted graph replay
```

The eighth call performs three eager side-stream warmups, captures the exact
eager suffix callable, then runs both eager and graph replay on the same input.
The entry is admitted only when the outputs are bitwise equal. Capture failure
or any mismatch permanently rejects that exact key for the process and returns
the eager output. The capture call itself also returns the independently
allocated eager result; graph-owned stable output is exposed only internally
and is cloned before every later exact-key return. The lock stays held through
observation-eight capture and admission, so simultaneous eighth calls cannot
capture the same key twice.

`torch.equal` is used only for observation-eight admission and therefore
synchronizes only that cold capture call. Admitted replay calls do not compare
or synchronize. The CUDA helper advances every key through probation before
timing. A service measurement is valid only after warmup has emitted all
expected capture markers and no new capture marker appears inside a measured
run.

Probation logs include the full `cu_seqlens` tuple, exact observation count, and
current graph-cache occupancy. Admission logs add cold-path wall-clock capture
duration and start the cumulative replay counter at zero. Replay logs emit
power-of-two milestones for the per-key cumulative replay count:

```text
Audio suffix CUDA graph probation uses eager suffix cu_seqlens=<tuple> observation=<n>/8 occupancy=<n>/7
Captured bitwise-exact audio suffix CUDA graph cu_seqlens=<tuple> observation=8/8 occupancy=<n>/7 capture_duration_ms=<ms> cumulative_replays=0
ASR audio post-pack suffix CUDA graph replay active cu_seqlens=<tuple> observation=8/8 cumulative_replays=<n>
```

No per-replay CUDA duration is collected in the service hot path. PyTorch CUDA
event elapsed-time reporting requires completed events, so collecting it inline
would add synchronization or a deferred-event queue. The exact replay counter
requires neither and remains protected by the existing per-entry enqueue lock.

The earlier `opt/mm-encoder-cudagraph` branch is negative protocol evidence
only. It enabled vLLM's broad `cudagraph_mm_encoder` setting and measured
`3.318 files/s` sequential plus `21.188 files/s`, `0.711 s` latency, and
`0.184 s` TTFT batched, without a consistent win. Its available Nsight trace
was hot-cache and excluded the audio tower. No code or positive attribution is
carried from that experiment; this candidate uses a narrower observable seam
and requires exact per-key admission.

## Environment and launcher

The strict independent gate is:

```text
ASR_AUDIO_SUFFIX_CUDAGRAPH=1
```

It also requires both accepted prerequisites:

```text
ASR_AUDIO_CPU_MAXSEQLEN=1
ASR_AUDIO_CPU_METADATA_PACK=1
```

Launcher:

```bash
inference/run_vllm_fp8_static_qk_prefill_audio_suffix_cudagraph.sh
```

Invalid suffix boolean values fail loudly. Missing prerequisites or a metadata
patch installed before the suffix runner fail closed without replacing part of
the model forward.

## CUDA helper gate

Do not run this without an explicitly allocated SM90 GPU:

```bash
rtk env \
  CUDA_VISIBLE_DEVICES=<free-gpu> \
  UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
  uv run inference/vllm_static_fp8/bench_audio_suffix_cudagraph.py \
    --segments 88,88,88 \
    --segments 88,88,89 \
    --segments 89,89,89 \
    --segments 89,89,90 \
    --segments 90,90,90 \
    --segments 90,91,91 \
    --segments 91,91,91 \
    --warmup 3 \
    --repeats 10
```

The helper instantiates the exact 24-layer, 1024-wide, 16-head, 2048-output
Qwen3-ASR audio architecture with synthetic BF16 weights. Its defaults cover
all seven whitelisted row counts; pass additional three-segment cases with the
same sum to test different `cu_seqlens` contents. It explicitly requires no
entry after observations one through seven and exactly one new entry on
observation eight. For every admitted case it then requires:

- eager versus graph replay bitwise equality on a fresh input;
- exact CUDA suffix kernel-name ordering, excluding the required graph input
  copy from that ordering comparison;
- one distinct admitted graph for every content key;
- timing that includes both the required stable hidden-state input copy and the
  independent output clone for replay;
- repeated same-key calls from two host threads on two distinct CUDA streams,
  with every retained output bitwise checked only after all replays finish.

The timing table reports CUDA-event duration and synchronized host-call enqueue
duration separately. CUDA-event replay numbers include the stable input copy,
graph launch, and independent output clone; host-call numbers include key
construction, lookup, both copy enqueues, and graph launch.

Required terminal line:

```text
concurrency_gate=PASS threads=2 streams=2 iterations=5
gate=PASS_EXACT_AUDIO_SUFFIX_CUDAGRAPH
```

Run a second equal-shape/different-content CUDA gate as well:

```bash
rtk env \
  CUDA_VISIBLE_DEVICES=<free-gpu> \
  UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
  uv run inference/vllm_static_fp8/bench_audio_suffix_cudagraph.py \
    --segments 88,88,88 \
    --segments 80,88,96 \
    --warmup 3 \
    --repeats 10
```

Both cases have `(M, cu_seqlens.numel()) == (264, 4)` but different cumulative
boundaries, so this invocation proves the content tuple selects distinct graphs.

The helper is a capture-protocol gate, while production FP8 correctness is
additionally checked by the per-key admission logic in the real encoder. Before
service attribution, run the helper over every observed production segmentation
shape, require the definitive replay marker, and confirm no key-rejection or
cache-full warning appeared during measured calls.

## Promotion gate

1. Pass the CUDA helper bitwise and kernel-order gate for representative and
   observed exact shapes, including equal `(M, N)` with different segment
   boundaries.
2. Run adjacent accepted/candidate/accepted B16 service controls on the same GPU
   and require replay markers for measured candidate calls.
3. Reject if priority average latency is neutral or worse, even if host launch
   time or TTFT improves.
4. Only after a latency win, run the full 550-file batched CER/WER gate and
   compare throughput, latency, TTFT, CER, and WER with the accepted path.
