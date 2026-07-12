# ASR Experiments

Utilities for preparing ASR audio/text pairs, running Qwen ASR through a vLLM
OpenAI-compatible server, and scoring prediction files against prepared ground
truth.

Run commands below from the repository root:

```bash
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments
```

## Setup

Install the pinned environment with `uv`:

```bash
uv sync
```

The inference scripts expect a vLLM OpenAI-compatible server by default at:

```text
http://localhost:8090/v1
```

Start the local vLLM server with:

```bash
bash inference/run_vllm.sh
```

The inference and benchmark scripts reset vLLM prefix cache before each run with
`POST /reset_prefix_cache`. `inference/run_vllm.sh` enables the vLLM development
API endpoints needed for that reset.

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

- `--mode sequential` uses `predictions/results/sequential_predicted`.
- `--mode batched` uses `predictions/results/batched_predicted`.
- Streaming is enabled by default; use `--no-stream` for non-streaming requests.
- `--workers` is only valid with `--mode batched`; default is `1`.
- `--num-files` is measured files after the 20-file warmup and after filtering.
- `--uniform-audio-length` filters eligible audio first, then clips submitted
  audio to that length.
- `--overwrite` writes outputs even when prediction files already exist. Without
  it, inference still runs but existing outputs are not rewritten.
- `--no-speech-rms-threshold` defaults to `1`.
- `--input-root`, `--output-root`, `--model`, and `--base-url` default to the
  same values as the underlying scripts.

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
