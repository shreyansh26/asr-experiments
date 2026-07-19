# ASR Experiments

Utilities for preparing ASR audio/text pairs, running Qwen ASR through a vLLM
OpenAI-compatible server, and scoring prediction files against prepared ground
truth. The serving and benchmark workflow supports BF16, dynamic FP8, and
calibrated static-activation FP8.

Run commands below from the repository root:

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments
```

## Technical Documentation

- [Static FP8 activation calibration](docs/fp8-calibration.md) explains sample
  selection, batched collection, global per-layer absmax aggregation, scale
  calculation, and the portable JSON artifact.
- [Out-of-tree vLLM static-FP8 extension](docs/vllm-static-fp8-extension.md)
  explains Python/vLLM registration, layer-name mapping, scale injection,
  runtime behavior, coverage diagnostics, and upgrade boundaries.
- [Nsight Systems dynamic-versus-static FP8 guide](docs/nsys-fp8-dynamic-vs-static.md)
  covers capture commands, report interpretation, CUDA-graph timing, exact
  node-level differences, screenshots, and row-by-row decoder pseudocode.
- [FlashAttention forward combine in Nsight Systems](docs/flashattention-forward-combine.md)
  explains the split-KV decode path, stable softmax recombination, and how to
  interpret `FlashAttnFwdCombine` in the node trace.

## Setup

Install the pinned environment with `uv`:

```bash
uv sync
```

The inference scripts expect a vLLM OpenAI-compatible server by default at:

```text
http://localhost:8090/v1
```

Choose one server precision:

```bash
# BF16.
bash inference/run_vllm.sh

# vLLM dynamic FP8 activations.
bash inference/run_vllm_fp8_dynamic.sh

# Calibrated static FP8 activations. The default artifact must exist.
bash inference/run_vllm_fp8_static.sh
```

The inference and benchmark scripts reset vLLM prefix cache before each run with
`POST /reset_prefix_cache`. All three launchers enable the vLLM development API
endpoints needed for that reset.

The static-FP8 launcher defaults to
`inference/results/fp8_static_scales_128x50.json`. Generate it with:

```bash
bash inference/run_record_fp8_static_scales.sh \
  --input-root data/prepared_data \
  --num-files 128 \
  --batch-size 8 \
  --max-audio-seconds 50 \
  --output inference/results/fp8_static_scales_128x50.json
```

Override the artifact when launching:

```bash
SCALES_JSON=inference/results/fp8_static_scales_custom.json \
  bash inference/run_vllm_fp8_static.sh
```

If the server runs elsewhere, pass `--base-url`.

## Data Layout

Default input and output locations:

```text
data/manifest.json                    # source manifest mapping for data prep
data/prepared_data/                   # default ground-truth audio/text tree
predictions/results/sequential_predicted/            # default sequential prediction tree
predictions/results/batched_predicted/               # default batched prediction tree
predictions/results/sequential_predicted_uniform_audio_length_* # default sequential clipped trees
predictions/results/batched_predicted_uniform_audio_length_*    # default batched clipped trees
predictions/results_bf16/                 # BF16 comparison benchmark outputs
predictions/results_fp8_dynamic/          # dynamic-FP8 comparison benchmark outputs
predictions/results_fp8_static/           # static-FP8 comparison/default wrapper outputs
```

Prepared data and predictions use the same relative path layout, for example:

```text
data/prepared_data/<dataset>/<sample>/channel_0.wav
data/prepared_data/<dataset>/<sample>/channel_0.txt
predictions/results/batched_predicted/<dataset>/<sample>/channel_0.txt
```

## Prepare Data

Script:

```bash
uv run python data/read_data.py
```

Current fixed behavior:

- Reads `data/manifest.json`.
- Writes to `data/prepared_data`.
- Copies audio files.
- Overwrites existing prepared audio/text files.

This script does not currently expose CLI flags. To change the manifest,
output directory, placement mode, or overwrite behavior, edit the `prepare_dataset`
call at the bottom of `data/read_data.py`.

Supported placement modes in code:

- `copy`: copy audio files into prepared data.
- `symlink`: symlink prepared audio files to source audio files.
- `hardlink`: hardlink source audio files.
- `auto`: try hardlink first, then symlink.

## Single-File / Sequential Inference

Script:

```bash
uv run python inference/run_infer.py <audio_path>
```

Single-file example:

```bash
uv run python inference/run_infer.py data/prepared_data/carta_september_2024/example/channel_0.wav
```

Arguments:

```bash
uv run python inference/run_infer.py <audio_path> \
  --model Qwen/Qwen3-ASR-1.7B \
  --base-url http://localhost:8090/v1
```

Streaming mode:

```bash
uv run python inference/run_infer.py <audio_path> --stream
```

Sequential directory mode:

```bash
uv run python inference/run_infer.py \
  --input-root data/prepared_data \
  --num-files 10 \
  --stream \
  --no-print-text
```

Uniform clipping with RMS no-speech filtering:

```bash
uv run python inference/run_infer.py \
  --num-files 10 \
  --uniform-audio-length 10 \
  --no-speech-rms-threshold 1
```

Sequential streaming with clipping, RMS filtering, and no transcript printing:

```bash
uv run python inference/run_infer.py \
  --input-root data/prepared_data/carta_september_2024 \
  --num-files 10 \
  --uniform-audio-length 10 \
  --no-speech-rms-threshold 1 \
  --stream \
  --no-print-text
```

Default behavior:

- With `audio_path`, runs one file and prints transcript text by default.
- Without `audio_path`, scans `.wav` files under `--input-root`.
- Runs requests sequentially and writes predictions under `predictions/results/sequential_predicted`.
- Resets vLLM prefix cache before warmup and measured inference.
- Uses 20 warmup audio files before measured files in directory mode.
- `--num-files` means measured benchmark files after warmup and after filtering.
- If an output file exists, inference still runs; without `--overwrite`, the
  existing file is not rewritten. With `--overwrite`, the output is written.
- `--print-text` prints transcripts; `--no-print-text` suppresses them.
- Final output reports completed/failed counts, throughput, audio throughput,
  avg/p50/p95/p99 latency, and avg/p50/p95/p99 TTFT (`n/a` for non-streaming).

## Batch Inference / Load Test

Script:

```bash
uv run python inference/run_infer_batched.py
```

Default behavior:

- Reads `.wav` files under `data/prepared_data`.
- Writes predictions under `predictions/results/batched_predicted`.
- Uses model `Qwen/Qwen3-ASR-1.7B`.
- Uses base URL `http://localhost:8090/v1`.
- Uses `--workers 1`.
- Uses `--timeout-seconds 10`.
- Uses `--max-tokens 512`.
- Resets vLLM prefix cache before warmup and measured inference.
- Uses 20 warmup audio files before measured files.
- `--num-files` means measured benchmark files after warmup and after filtering.
- If an output file exists, inference still runs; without `--overwrite`, the
  existing file is not rewritten. With `--overwrite`, the output is written.
- Continues past per-request failures and reports `failed` and `timed_out`.

Common commands:

```bash
# Full run.
uv run python inference/run_infer_batched.py

# Full run, overwrite existing predictions.
uv run python inference/run_infer_batched.py --overwrite

# Parallel run.
uv run python inference/run_infer_batched.py --workers 4 --overwrite

# Longer per-request timeout.
uv run python inference/run_infer_batched.py --workers 4 --overwrite --timeout-seconds 30

# Limit generated transcription tokens.
uv run python inference/run_infer_batched.py --workers 4 --overwrite --max-tokens 512

# Smoke test on the first N measured files after warmup/filtering.
uv run python inference/run_infer_batched.py --num-files 20 --overwrite

# Use a custom server or model.
uv run python inference/run_infer_batched.py \
  --base-url http://localhost:8090/v1 \
  --model Qwen/Qwen3-ASR-1.7B

# Read from and write to custom directories.
uv run python inference/run_infer_batched.py \
  --input-root data/prepared_data \
  --output-root predictions/results/batched_predicted_custom \
  --workers 4 \
  --overwrite
```

### Clipped-Audio Mode

Use `--uniform-audio-length <seconds>` to send only the first fixed-length clip
from each audio file. Files shorter than or equal to the clip length are skipped.

```bash
# Send only first 10 seconds of each eligible audio file.
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --workers 4 \
  --overwrite
```

Default clipped output path is a separate sibling directory:

```text
predictions/results/batched_predicted_uniform_audio_length_10s
```

The clipped mode also applies a no-speech guard before creating inference jobs.
The default threshold is `--no-speech-rms-threshold 1`; clipped payloads with
RMS less than or equal to that value are skipped before submission to vLLM.

```bash
# Disable most no-speech filtering.
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --no-speech-rms-threshold 0 \
  --workers 4 \
  --overwrite

# More aggressive no-speech filtering.
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --no-speech-rms-threshold 5 \
  --workers 4 \
  --overwrite

# Override clipped output directory.
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --output-root predictions/results/batched_predicted_10s_custom \
  --workers 4 \
  --overwrite
```

All load-test options:

```bash
uv run python inference/run_infer_batched.py \
  --input-root data/prepared_data \
  --output-root predictions/results/batched_predicted \
  --model Qwen/Qwen3-ASR-1.7B \
  --base-url http://localhost:8090/v1 \
  --workers 4 \
  --num-files 100 \
  --uniform-audio-length 10 \
  --timeout-seconds 30 \
  --max-tokens 512 \
  --no-speech-rms-threshold 1 \
  --overwrite
```

## Benchmark Wrapper

Script:

```bash
uv run python inference/run_benchmark.py --mode sequential
uv run python inference/run_benchmark.py --mode batched
```

The wrapper runs `run_infer.py` or `run_infer_batched.py`, streams the child
script output live, parses the final metrics, and appends one row to:

```text
inference/results/sequential.csv
inference/results/batched.csv
```

Default behavior:

- `--mode sequential` uses
  `predictions/results_fp8_static/sequential_predicted`.
- `--mode batched` uses `predictions/results_fp8_static/batched_predicted`.
- For BF16 or dynamic-FP8 comparison runs, pass an explicit `--output-root`
  containing `results_bf16` or `results_fp8_dynamic`. The analysis script uses
  those path markers to identify precision.
- Streaming is enabled by default; use `--no-stream` for non-streaming requests.
- `--workers` is only valid with `--mode batched`; default is `1`.
- `--num-files` is measured files after the 20-file warmup and after filtering.
- `--uniform-audio-length` filters eligible audio first, then clips submitted
  audio to that length.
- `--overwrite` writes outputs even when prediction files already exist. Without
  it, inference still runs but existing outputs are not rewritten.
- `--no-speech-rms-threshold` defaults to `1`.
- `--input-root`, `--model`, and `--base-url` default to the same values as the
  underlying scripts. The wrapper's default output roots are the static-FP8
  paths listed above.

CSV metrics include counts, skipped/no-speech counts, wall time, file
throughput, audio throughput, avg/p50/p95/p99 latency, avg/p50/p95/p99 TTFT,
and the exact command that was run. For full overwrite benchmarks, the wrapper
also runs aggregate error-rate scoring and appends CER/WER fields to the CSV.
That scoring step runs only when all of these are true:

- `--overwrite` is passed.
- `--num-files` is not passed.
- `--uniform-audio-length` is not passed.

The scoring step uses `eval/compute_error_rates.py <output_root> --ref-root
<input_root>` and does not pass `--per-file`.

Common benchmark commands:

```bash
# Sequential streaming benchmark on 50 measured files.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --num-files 50

# Sequential streaming benchmark with 10-second clipped audio.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --num-files 50 \
  --uniform-audio-length 10

# Batched streaming benchmark with 4 concurrent workers.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --num-files 50

# Batched 10-second clipped benchmark.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --num-files 50 \
  --uniform-audio-length 10

# Non-streaming benchmark.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --num-files 50 \
  --no-stream

# Write or overwrite prediction files during benchmarking.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --num-files 50 \
  --overwrite

# Full batched benchmark with aggregate CER/WER scoring.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 4 \
  --overwrite

# Custom roots, model, and server.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --input-root data/prepared_data \
  --output-root predictions/results/sequential_predicted_custom \
  --model Qwen/Qwen3-ASR-1.7B \
  --base-url http://localhost:8090/v1 \
  --num-files 50
```

### Precision comparison workflow

Start exactly one matching server, run its benchmarks, stop it, and repeat for
the next precision:

| Precision label | Server command |
| --- | --- |
| `bf16` | `bash inference/run_vllm.sh` |
| `fp8_dynamic` | `bash inference/run_vllm_fp8_dynamic.sh` |
| `fp8_static` | `bash inference/run_vllm_fp8_static.sh` |

Use precision-specific output roots. This keeps prediction files isolated and
allows `analyse_results.py` to infer the precision from the recorded path:

```bash
# Set this to bf16, fp8_dynamic, or fp8_static for the matching running server.
PRECISION=fp8_static

# Full sequential benchmark; --overwrite also enables aggregate CER/WER scoring.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --output-root "predictions/results_${PRECISION}/sequential_predicted" \
  --overwrite

# Sequential benchmark limited to the first 50 seconds of eligible files.
uv run python inference/run_benchmark.py \
  --mode sequential \
  --uniform-audio-length 50 \
  --output-root "predictions/results_${PRECISION}/sequential_predicted_uniform_audio_length_50s"

# Full 16-worker load test.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --output-root "predictions/results_${PRECISION}/batched_predicted" \
  --overwrite

# 16-worker, 50-second load test.
uv run python inference/run_benchmark.py \
  --mode batched \
  --workers 16 \
  --uniform-audio-length 50 \
  --output-root "predictions/results_${PRECISION}/batched_predicted_uniform_audio_length_50s"
```

Print the comparison tables from the accumulated CSV files:

```bash
uv run python inference/analyse_results.py --mode sequential
uv run python inference/analyse_results.py --mode batched
```

### Current benchmark results

These tables are the current workspace snapshot from
`inference/results/sequential.csv` and `inference/results/batched.csv`. Latency
and TTFT values are seconds; throughput is completed files per second.

Sequential, full benchmark with 550 measured files:

| Precision | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 1.576 | 3.261 | 4.229 | 0.203 | 0.318 | 0.377 | 0.583 | 0.168 | 0.388 |
| FP8 dynamic | 1.473 | 2.929 | 3.732 | 0.201 | 0.309 | 0.339 | 0.640 | 0.165 | 0.385 |
| FP8 static | 1.381 | 2.745 | 3.439 | 0.207 | 0.339 | 0.372 | 0.682 | 0.162 | 0.384 |

Sequential, 50-second audio limit:

| Precision | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.365 | 0.564 | 1.411 | 0.063 | 0.074 | 0.092 | 2.677 |
| FP8 dynamic | 0.286 | 0.479 | 0.634 | 0.046 | 0.060 | 0.064 | 3.409 |
| FP8 static | 0.273 | 0.468 | 0.489 | 0.059 | 0.075 | 0.090 | 3.538 |

Batched, full benchmark with 550 measured files:

| Precision | Workers | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 1.977 | 4.006 | 5.276 | 0.227 | 0.402 | 0.509 | 1.875 | n/a | n/a |
| BF16 | 8 | 2.428 | 5.082 | 6.409 | 0.260 | 0.558 | 0.935 | 3.001 | n/a | n/a |
| BF16 | 16 | 3.520 | 7.136 | 9.130 | 0.351 | 0.946 | 1.946 | 4.138 | 0.166 | 0.387 |
| FP8 dynamic | 4 | 1.900 | 3.831 | 4.719 | 0.228 | 0.420 | 0.566 | 1.980 | n/a | n/a |
| FP8 dynamic | 8 | 2.435 | 4.959 | 5.848 | 0.270 | 0.525 | 0.775 | 3.055 | n/a | n/a |
| FP8 dynamic | 16 | 3.563 | 7.166 | 9.903 | 0.343 | 1.121 | 1.838 | 4.104 | 0.158 | 0.380 |
| FP8 static | 16 | 3.346 | 6.675 | 9.124 | 0.426 | 1.177 | 1.984 | 4.352 | 0.162 | 0.384 |

Batched, 50-second audio limit:

| Precision | Workers | Lat p50 | Lat p95 | Lat p99 | TTFT p50 | TTFT p95 | TTFT p99 | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 0.429 | 0.641 | 1.659 | 0.049 | 0.085 | 0.122 | 9.130 |
| BF16 | 8 | 0.542 | 0.864 | 2.047 | 0.073 | 0.179 | 0.288 | 13.922 |
| BF16 | 16 | 0.720 | 1.338 | 2.768 | 0.138 | 0.448 | 0.456 | 18.554 |
| FP8 dynamic | 4 | 0.351 | 0.582 | 0.727 | 0.052 | 0.106 | 0.111 | 10.489 |
| FP8 dynamic | 8 | 0.441 | 0.764 | 0.846 | 0.053 | 0.149 | 0.157 | 16.686 |
| FP8 dynamic | 16 | 0.735 | 1.279 | 1.481 | 0.154 | 0.432 | 0.499 | 20.731 |
| FP8 static | 16 | 0.652 | 1.098 | 1.212 | 0.119 | 0.306 | 0.332 | 22.171 |

Static FP8 currently has batched comparison rows only at 16 workers; the table
does not imply unmeasured 4- or 8-worker static results. CER/WER is `n/a` for
rows that were not recorded through the wrapper's full overwrite-and-score
path. The 50-second runs are intentionally not scored by the wrapper.

## Nsight Systems Profiling

Nsight Systems must launch the vLLM server so that CUDA work from the engine
process is visible. The client request then starts and stops collection through
the named Nsight session. Run the workflow from the repository root in two
terminals.

The two supported modes answer different questions:

| Mode | Nsight configuration | Use it for |
| --- | --- | --- |
| `node` | `--cuda-graph-trace=node` | Executed CUDA graph nodes, exact kernel order and names, node counts, and per-replay GPU timing |
| `pytrace_graph` | `--cuda-graph-trace=graph --pytorch=functions-trace` | CUDA graph launches plus Python/PyTorch/NVTX ownership and CPU call-path context |

Use `node` for kernel-level dynamic-versus-static FP8 comparisons. Use
`pytrace_graph` to determine which Python or vLLM region initiated a graph
launch. The server wrapper supplies the required Python `nvtx` package through
`uv run --with nvtx`, so this does not modify the project dependencies.

### Configure the server capture

Edit these variables near the top of
`inference/start_vllm_server_with_nsys.sh`:

```bash
SESSION_NAME=fp8static_c1
REPORT_NAME=fp8static_c1_5s
VLLM_SCRIPT=inference/run_vllm_fp8_static.sh
```

Common precision configurations are:

| Precision | `SESSION_NAME` | `REPORT_NAME` | `VLLM_SCRIPT` |
| --- | --- | --- | --- |
| BF16 | `bf16_c1` | `bf16_c1_5s` | `inference/run_vllm.sh` |
| Dynamic FP8 | `fp8dyn_c1` | `fp8dyn_c1_5s` | `inference/run_vllm_fp8_dynamic.sh` |
| Static FP8 | `fp8static_c1` | `fp8static_c1_5s` | `inference/run_vllm_fp8_static.sh` |

The wrapper appends the selected mode to both names. For example, the static
configuration produces:

```text
node session:          fp8static_c1_node
node report:           inference/results/nsys/fp8static_c1_5s_node
pytrace_graph session: fp8static_c1_pytrace_graph
pytrace_graph report:  inference/results/nsys/fp8static_c1_5s_pytrace_graph
```

### Configure the profiled request

Edit the matching values near the top of `inference/run_nsys_profile.sh`:

```bash
SESSION=fp8static_c1_node
REPORT=inference/results/nsys/fp8static_c1_5s_node
AUDIO=data/prepared_data/carta_september_2024/call-156003_0.8311693678982316_4.230553711834489/channel_0.wav
```

`SESSION` must exactly equal `<SESSION_NAME>_<mode>` from the server wrapper.
`REPORT` should equal
`inference/results/nsys/<REPORT_NAME>_<mode>`. The server wrapper owns the
actual Nsight output path; the request script keeps `REPORT` only to print the
expected artifact at the end.

The request script waits for `/v1/models`, runs three unprofiled warmups, starts
collection, sends one streaming 5-second request, and stops collection.

### Capture node-level CUDA graph activity

For the static configuration above, use:

```bash
# Terminal 1: launch the server under an inactive Nsight session.
bash inference/start_vllm_server_with_nsys.sh node
```

After the server starts, run in another terminal:

```bash
# Terminal 2: warm up, profile one request, and stop collection.
bash inference/run_nsys_profile.sh
```

For dynamic FP8, first set the server variables to:

```bash
SESSION_NAME=fp8dyn_c1
REPORT_NAME=fp8dyn_c1_5s
VLLM_SCRIPT=inference/run_vllm_fp8_dynamic.sh
```

and set the request variables to:

```bash
SESSION=fp8dyn_c1_node
REPORT=inference/results/nsys/fp8dyn_c1_5s_node
```

Then run the same two commands.

### Capture Python/PyTorch ownership and graph launches

Change the request script to the matching `pytrace_graph` session and report:

```bash
SESSION=fp8static_c1_pytrace_graph
REPORT=inference/results/nsys/fp8static_c1_5s_pytrace_graph
```

Then run:

```bash
# Terminal 1.
bash inference/start_vllm_server_with_nsys.sh pytrace_graph

# Terminal 2, after the server is ready.
bash inference/run_nsys_profile.sh
```

For dynamic FP8, use `fp8dyn_c1_pytrace_graph` and
`fp8dyn_c1_5s_pytrace_graph` instead.

Each capture writes both files because the server wrapper passes
`--export=sqlite`:

```text
inference/results/nsys/<report-name>.nsys-rep
inference/results/nsys/<report-name>.sqlite
```

If the vLLM process remains active after report generation, stop it with
Ctrl-C in Terminal 1 before starting the next precision or trace mode. Do not
run two servers on port `8090` at the same time.

See the
[Nsight Systems dynamic-versus-static FP8 guide](docs/nsys-fp8-dynamic-vs-static.md)
for direct `nsys` commands, timing definitions, Events View filtering, SQLite
queries, graph identification, and interpretation of the captured kernels.

## Error-Rate Evaluation

Script:

```bash
uv run python eval/compute_error_rates.py <prediction_root>
```

The evaluator computes case-insensitive, whitespace-normalized:

- CER: character error rate.
- WER: word error rate.

It iterates over prediction `.txt` files and matches each prediction to the
same relative path under the reference root. This supports partial prediction
runs where some requests failed or timed out.

Default reference root:

```text
data/prepared_data
```

Commands:

```bash
# Score full-audio predictions.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted

# Score clipped-audio predictions.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted_uniform_audio_length_10s

# Use a custom reference tree.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted_custom \
  --ref-root data/prepared_data

# Print per-file tab-separated metrics before the aggregate summary.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted \
  --per-file

# Save per-file and aggregate output.
uv run python eval/compute_error_rates.py predictions/results/batched_predicted \
  --per-file > error_rates.tsv
```

Summary output includes:

- number of prediction text files found;
- number of files matched to references;
- number of missing references;
- aggregate CER with `char_edits/ref_chars`;
- aggregate WER with `word_edits/ref_words`.

## Typical Workflows

Prepare data, run inference, score predictions:

```bash
uv run python data/read_data.py
uv run python inference/run_infer_batched.py --workers 4 --overwrite --timeout-seconds 30
uv run python eval/compute_error_rates.py predictions/results/batched_predicted
```

Run a short smoke test:

```bash
uv run python inference/run_infer_batched.py --num-files 20 --workers 4 --overwrite
uv run python eval/compute_error_rates.py predictions/results/batched_predicted
```

Run clipped 10-second inference and score it:

```bash
uv run python inference/run_infer_batched.py \
  --uniform-audio-length 10 \
  --workers 4 \
  --overwrite \
  --timeout-seconds 30

uv run python eval/compute_error_rates.py predictions/results/batched_predicted_uniform_audio_length_10s
```
