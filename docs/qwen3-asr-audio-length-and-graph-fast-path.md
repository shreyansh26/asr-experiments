# Qwen3-ASR audio lengths and CUDA-graph fast-path coverage

This note explains how decoded audio duration becomes a Qwen3-ASR feature
length, how that feature length becomes the post-CNN row count used by the
audio transformer, and which resulting shapes are admitted by the current
audio-prefix and audio-suffix CUDA graphs.

The implementation described here starts from the selected round-four
candidate on branch `opt5/audio-prefix-shared-suffix-bucketed`. Follow-up
branch `opt6/audio-tail-rows-263-271` expands the compatible 21-chunk tail row
family from seven rows to the contiguous range `M=263..273`. It combines:

- the accepted static-FP8 decoder with fused Q/K RMSNorm, MRoPE, and KV-cache
  write;
- general CPU construction of audio length and packing metadata;
- exact-admitted audio-prefix CUDA graphs with a shared graph memory pool;
- exact-admitted audio-suffix CUDA graphs grouped into two padded row buckets.

The key distinction is:

> The optimization does not recognize an audio file, speaker, language, or
> waveform. It recognizes exact tensor shapes and metadata after decoding and
> resampling. An unseen audio file can use the graph fast path when it maps to
> an admitted shape.

## End-to-end length pipeline

For one server-side audio chunk, the relevant pipeline is:

```text
encoded audio file
  -> decode, channel normalization, and resampling to 16 kHz
  -> valid 128-bin log-mel frames, feature length F
  -> split F into 100-frame convolution chunks
  -> three stride-two convolutions per chunk
  -> pack valid post-CNN rows, total length M
  -> divide M into local-attention sequences of at most 104 rows
  -> audio transformer and projection
```

Files longer than 30 seconds are split into several server-side chunks before
this model pipeline. Graph admission happens independently for each resulting
chunk, not once for the original file.

## Decoded samples to feature length

Let `N` be the number of mono samples after resampling to 16 kHz. The Whisper
feature extractor uses a hop length of 160 samples, which is 10 ms at 16 kHz.

For the single-audio server processing path, the valid feature length is:

```text
F = floor(N / 160)
```

Equivalently, for decoded duration `T = N / 16000` seconds:

```text
F = floor(100 * T)
```

The exact runtime authority is the sum of the rescaled feature attention mask.
The feature extractor subsamples the sample-domain mask with `hop_length=160`
and trims the extra STFT frame. Thus an exact feature length `F` corresponds to
the following 16 kHz sample interval:

```text
160 * F <= N < 160 * (F + 1)
```

or the following half-open duration interval:

```text
F / 100 <= T < (F + 1) / 100 seconds
```

Original sample rate, channel count, and container format do not directly
select a graph. They matter only insofar as decoding and resampling change the
final 16 kHz sample count. For compressed formats or resampled inputs, inspect
the decoded sample count or runtime feature attention mask when a boundary
must be classified exactly.

The installed feature extractor logic is in:

```text
.venv/lib/python3.12/site-packages/transformers/models/whisper/
  feature_extraction_whisper.py
```

## Feature length to post-CNN rows

The audio encoder first divides `F` into raw chunks of 100 feature frames:

```text
q, r = divmod(F, 100)
```

Here `q` is the number of full chunks and `r` is the final partial chunk
length. One 100-frame chunk represents approximately one second.

Each raw chunk passes through three convolutions with stride two and padding
one. For a chunk with `L` valid frames, each convolution applies:

```text
L -> floor((L - 1) / 2) + 1 = ceil(L / 2)
```

After three convolutions:

```text
C(L) = ceil(L / 8)
```

A full 100-frame chunk therefore produces:

```text
100 -> 50 -> 25 -> 13 rows
```

The whole-audio post-CNN row count is consequently:

```text
M = 13 * floor(F / 100) + ceil((F % 100) / 8)
```

For integer arithmetic:

```text
M = 13 * (F // 100) + ((F % 100 + 7) // 8)
```

It is important not to replace this with `ceil(F / 8)`. The model performs
rounding independently inside every 100-frame chunk, so every full chunk
contributes 13 rows rather than 12.5 rows.

The installed vLLM formula is `_get_feat_extract_output_lengths()` in:

```text
.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_asr.py
```

Our exact CPU mirror is
[`_expected_audio_output_lengths()`](../inference/vllm_static_fp8/audio_cpu_metadata_pack_patch.py).

### Worked examples

| Decoded duration | Feature frames `F` | Calculation | Post-CNN rows `M` |
|---:|---:|---|---:|
| 1.0 s | 100 | `13 * 1 + 0` | 13 |
| 10.0 s | 1000 | `13 * 10 + 0` | 130 |
| 20.0 s | 2000 | `13 * 20 + 0` | 260 |
| 20.3 s | 2030 | `13 * 20 + ceil(30/8)` | 264 |
| 29.0 s | 2900 | `13 * 29 + 0` | 377 |
| 29.5 s | 2950 | `13 * 29 + ceil(50/8)` | 384 |
| 30.0 s | 3000 | `13 * 30 + 0` | 390 |

## Why the local-attention window is 104 rows

The number eight is a model configuration ratio, not an HTTP batch size and
not a new choice made by this optimization:

```text
n_window       = 50
raw chunk      = n_window * 2 = 100 feature frames
n_window_infer = 800 feature frames
chunks/window  = 800 // 100 = 8
```

Because each full raw chunk produces 13 post-CNN rows, one configured
inference attention window contains:

```text
window_aftercnn = 8 * 13 = 104 rows
```

The installed encoder computes the same value as:

```python
window_aftercnn = padded_mask_after_cnn.shape[-1] * (
    self.n_window_infer // (self.n_window * 2)
)
```

The transformer receives cumulative sequence boundaries for consecutive
windows. Examples are:

```text
M=264 -> lengths 104 + 104 + 56
         cu_seqlens = (0, 104, 208, 264)

M=390 -> lengths 104 + 104 + 104 + 78
         cu_seqlens = (0, 104, 208, 312, 390)
```

FlashAttention treats each interval as a separate local-attention sequence.
Our metadata patch reconstructs these boundaries on the CPU in
[`_build_cpu_metadata()`](../inference/vllm_static_fp8/audio_cpu_metadata_pack_patch.py),
avoiding the previous GPU-to-CPU length readback while preserving the model's
existing algorithm.

## Which parts are general and which are shape-specialized

There are three distinct optimization scopes.

### CPU metadata and valid-row packing

The CPU metadata path derives raw chunk lengths, per-chunk post-CNN lengths,
packing offsets, and attention boundaries for arbitrary positive feature
lengths accepted by the guarded installed model implementation. This part is
not restricted to 20-second, 29-second, or 30-second audio.

The Triton valid-row pack copies only the valid CNN rows into their calculated
output offsets. Unsupported model versions or inconsistent metadata fail
closed to the accepted implementation.

Implementation:

- [`audio_cpu_metadata_pack_patch.py`](../inference/vllm_static_fp8/audio_cpu_metadata_pack_patch.py)
- [`audio_cpu_maxseqlen_patch.py`](../inference/vllm_static_fp8/audio_cpu_maxseqlen_patch.py)

### Prefix CUDA graphs

The prefix graph covers convolution, projection, positional embedding, and
valid-row packing. It validates the complete runtime key before replay:

- padded shape and stride;
- dtype and device;
- every raw chunk length;
- every pack length and output offset;
- the full `feature_lens` and `aftercnn_lens` tuples;
- all cumulative attention boundaries.

The natural family admits exactly one audio with `F=2897..3000`. Those 104
exact feature-length keys reduce to 14 graph-static signatures, one for each
post-CNN row count `M=377..390`. The signatures share a CUDA graph memory pool
and are serialized because captures from that pool can alias allocations.

The tail family admits exactly 21 raw convolution chunks with
`M in {263, ..., 273}` when all other exact metadata guards also pass. The
21-chunk restriction preserves the captured prefix input topology
`(21,1,128,100)`; it is an exact CUDA-graph shape guard, not a model limit.

Implementation:

- [`audio_prefix_cudagraph_patch.py`](../inference/vllm_static_fp8/audio_prefix_cudagraph_patch.py)
- [Detailed prefix design](audio-prefix-cudagraph.md)

### Suffix CUDA graphs

The suffix graph covers the 24 audio-transformer layers and output projection.
Exact runtime keys are admitted into two padded graph families:

```text
tail rows M in {263, ..., 273}
  -> padded graph bucket of 273 rows

natural rows M in {377, ..., 390}
  -> padded graph bucket of 390 rows
```

Padding reduces 25 exact suffix shapes to two captured graphs, while exact
pre-admission prevents an unsupported shape or attention layout from replaying
through a merely equal-sized bucket.

Implementation:

- [`audio_suffix_cudagraph_patch.py`](../inference/vllm_static_fp8/audio_suffix_cudagraph_patch.py)
- [Detailed suffix design](../ideas/audio_suffix_cudagraph.md)

## Exact natural graph coverage

Natural graph admission covers the following continuous feature-length range:

```text
2897 <= F <= 3000
```

This is approximately 28.97 through 30.00 seconds for chunks supplied by the
server API, whose maximum chunk duration is 30 seconds.

| Post-CNN rows `M` | Exact feature lengths `F` | Approximate duration |
|---:|---:|---:|
| 377 | 2897..2900 | 28.97..<29.01 s |
| 378 | 2901..2908 | 29.01..<29.09 s |
| 379 | 2909..2916 | 29.09..<29.17 s |
| 380 | 2917..2924 | 29.17..<29.25 s |
| 381 | 2925..2932 | 29.25..<29.33 s |
| 382 | 2933..2940 | 29.33..<29.41 s |
| 383 | 2941..2948 | 29.41..<29.49 s |
| 384 | 2949..2956 | 29.49..<29.57 s |
| 385 | 2957..2964 | 29.57..<29.65 s |
| 386 | 2965..2972 | 29.65..<29.73 s |
| 387 | 2973..2980 | 29.73..<29.81 s |
| 388 | 2981..2988 | 29.81..<29.89 s |
| 389 | 2989..2996 | 29.89..<29.97 s |
| 390 | 2997..3000 | 29.97..30.00 s under the API cap |

The mathematical interval for `F=3000` extends to just below 30.01 seconds,
but an original API input longer than 30 seconds is chunked before feature
extraction. The effective server-created natural range therefore ends at
30.00 seconds.

## Exact observed tail coverage

The 50-second workload produced a second family of approximately 20--21 second
final chunks. Follow-up branch `opt6/audio-tail-rows-263-271` makes the admitted
single-audio feature range continuous across these rows:

| Post-CNN rows `M` | Exact feature lengths `F` | Approximate duration |
|---:|---:|---:|
| 263 | 2017..2024 | 20.17..<20.25 s |
| 264 | 2025..2032 | 20.25..<20.33 s |
| 265 | 2033..2040 | 20.33..<20.41 s |
| 266 | 2041..2048 | 20.41..<20.49 s |
| 267 | 2049..2056 | 20.49..<20.57 s |
| 268 | 2057..2064 | 20.57..<20.65 s |
| 269 | 2065..2072 | 20.65..<20.73 s |
| 270 | 2073..2080 | 20.73..<20.81 s |
| 271 | 2081..2088 | 20.81..<20.89 s |
| 272 | 2089..2096 | 20.89..<20.97 s |
| 273 | 2097..2100 | 20.97..<21.01 s |

The expanded family covers `F=2017..2100`, approximately
`[20.17, 21.01)` seconds, without changing the 273-row suffix bucket. It is
still workload-informed rather than complete coverage of possible final
chunks in `(0, 30]` seconds.

### Follow-up GPU validation

On 2026-07-21, GPU1 passed the SM90 suffix helper for all 25 exact keys: the
11 tail rows `M=263..273` and 14 natural rows `M=377..390`. Every key was
bitwise exact, bucket-eager and replay kernel order matched at 268 kernels,
and alternating plus two-thread/two-stream shared-bucket gates passed.

The chained prefix-plus-suffix helper then passed each newly admitted row with
eight-observation probation, changed-content equality, and two-thread/two-
stream concurrency:

| New row | Eager CUDA | Copy + graph replay + clone | Eager host call | Graph host call |
|---:|---:|---:|---:|---:|
| 263 | 6493.632 us | 2134.784 us | 6507.907 us | 165.693 us |
| 266 | 6804.240 us | 2135.920 us | 6834.688 us | 170.253 us |
| 269 | 6683.440 us | 2137.824 us | 6722.245 us | 172.673 us |
| 271 | 6722.048 us | 2133.536 us | 6700.560 us | 167.908 us |

These focused helper measurements were followed by a clean end-to-end service
run from commit `98f6992` on GPU1. The 500-file, 50-second batched workload
completed with zero failures/timeouts at 29.572 files/s, 0.526 s mean latency,
and 0.199 s mean TTFT. The server log also showed the expanded 21-chunk family
entering probation and replay through the existing 273-row suffix bucket.

## Long-file splitting and arbitrary final tails

The vLLM transcription server first decodes and resamples the complete file.
When its duration is greater than 30 seconds, it repeatedly searches for a
low-energy split point during the final second of the next 30-second region.

With the current configuration:

```text
sample rate               = 16000 Hz
maximum chunk             = 30 seconds
split search region       = [29, 30) seconds relative to the chunk start
energy window             = 1600 samples = 0.1 seconds
candidate non-final sizes = 29.0, 29.1, ..., 29.8 seconds
```

Despite the configuration name `overlap_chunk_second`, the current splitter
uses the one-second interval as a split-search region; it does not duplicate
that audio in both emitted chunks. The next chunk begins at the selected split
point.

Every non-final chunk therefore falls inside the natural graph range. Once the
remaining audio is at most 30 seconds, the server emits it unchanged as the
final chunk. For an arbitrary long file:

```text
0 < final tail duration <= 30 seconds
```

Consequently, full CUDA-graph coverage cannot be inferred from total file
duration alone. Audio energy affects the chosen split position, and the final
tail can have any positive duration up to 30 seconds. Unsupported tails remain
correct and run through the eager fallback; earlier eligible chunks from the
same file can still use their graphs.

### Exact 50-second example

For an exact 50-second decoded input with one split:

| First chunk | Final tail | Tail rows | Current tail graph? |
|---:|---:|---:|:---:|
| 29.0 s | 21.0 s | 273 | Yes |
| 29.1 s | 20.9 s | 272 | Yes |
| 29.2 s | 20.8 s | 270 | Yes |
| 29.3 s | 20.7 s | 269 | Yes |
| 29.4 s | 20.6 s | 268 | Yes |
| 29.5 s | 20.5 s | 267 | Yes |
| 29.6 s | 20.4 s | 265 | Yes |
| 29.7 s | 20.3 s | 264 | Yes |
| 29.8 s | 20.2 s | 263 | Yes |

The non-final chunk is graph-eligible in every row. Seven of the nine possible
exact tails were covered by the original observed tail row family; the
follow-up expansion covers all nine.

### Other illustrative file lengths

| Original duration | Typical server chunks | Likely graph behavior |
|---:|---|---|
| 35 s | approximately 29.x + 5.x s | natural graph, then eager tail |
| 40 s | approximately 29.x + 10.x s | natural graph, then eager tail |
| 50 s | approximately 29.x + 20.x s | natural graph, often graphed tail |
| 59 s | approximately 29.x + 29.x s | both chunks usually natural graphs |
| 65 s | approximately 29.x + 29.x + 6.x s | two natural graphs, then eager tail |

These are illustrations rather than admission guarantees because the actual
low-energy split depends on the waveform.

## Why the varying 550-file workload still improves

The full 550-file natural workload is not restricted to one audio duration.
Nevertheless, the selected same-server steady run improved aggregate
throughput and latency substantially:

| Path | Throughput | Average latency | Average TTFT |
|---|---:|---:|---:|
| Accepted adjacent mean | 4.803 files/s | 3.269 s | 0.533 s |
| Selected winner, steady | 5.809 files/s | 2.696 s | 0.629 s |
| Change | +20.95% | -17.53% | +18.01% |

This result demonstrates useful partial coverage, not uniform acceleration of
every file:

- non-final 29--30 second chunks from long recordings frequently enter the
  natural prefix and suffix graphs;
- eligible final chunks also enter a graph;
- unsupported final tails retain the general CPU metadata path and use eager
  prefix/suffix execution;
- graph-eligible invocations occur often enough to improve aggregate batched
  throughput and mean latency;
- TTFT regressed in this comparison and remains the main measured tradeoff.

The workload result therefore supports the description **shape-specialized
but workload-general**. The implementation is not tied to the 550 source files
or their contents, but graph replay is restricted to the admitted runtime
shapes.

Quality measurements and the exact service methodology are recorded in
[the round-four report](batched-kernel-optimization-round4.md).

## Cold admission versus steady replay

Graph eligibility does not mean that the first occurrence immediately replays
a graph. Each exact runtime key passes probation:

```text
observations 1..7 -> eager execution
observation 8     -> capture or alias validation against eager output
observation 9+    -> hot graph replay when admission succeeded
```

Capture errors, numerical mismatches, changed metadata, unsupported layouts,
training or gradient mode, and cache-capacity limits all fail closed to eager
execution. This is why same-server steady measurements best represent the hot
graph path, while fresh-service measurements include admission overhead.

## Multi-audio and batch-layout qualification

HTTP concurrency does not weaken graph validation. A graph key includes the
full length and cumulative-attention metadata, not only the summed row count.
An unseen single audio with an admitted length and canonical metadata can use
the graph regardless of its content.

A different packed multi-audio partition does not qualify merely because its
total post-CNN rows equal an admitted `M`. If `feature_lens`, `aftercnn_lens`,
raw chunking, pack offsets, or `cu_seqlens` differ, the exact guard rejects the
graph replay and retains eager correctness.

## Small calculator

The following reproduces the single-chunk length mapping after decoding to
16 kHz:

```python
def qwen3_asr_lengths(samples_16khz: int) -> tuple[int, int]:
    feature_frames = samples_16khz // 160
    full_chunks, tail_frames = divmod(feature_frames, 100)
    post_cnn_rows = 13 * full_chunks + (tail_frames + 7) // 8
    return feature_frames, post_cnn_rows
```

For runtime admission, these two integers are necessary but not sufficient.
The implementation additionally validates tensor layout, every chunk and pack
offset, the per-audio length tuples, and the cumulative attention boundaries.

## Practical conclusions

- Raw audio content does not select the fast path; decoded shape and metadata
  do.
- Feature length advances once per 10 ms after resampling to 16 kHz.
- The three CNNs produce 13 rows per full one-second feature chunk.
- Eight such chunks form the configured 104-row local-attention window.
- Non-final chunks created by the current long-file splitter naturally land in
  the 29--30 second graph family.
- The final chunk of a long file can be anywhere in `(0, 30]` seconds, so the
  current 20--21 second family is useful but not comprehensive.
- Unsupported shapes use safe eager fallback rather than an approximate graph.
- The varying 550-file workload benefits because enough of its individual
  server chunks are graph-eligible, even though not every audio is fully
  covered.
