# Audio prefix CUDA graph candidate

Branch: `opt3/audio-prefix-cudagraph`

Base: accepted `opt3/audio-cpu-metadata-pack` commit `10bbc1c`

## Scope

This is a separate opt-in candidate for the eager Qwen3-ASR audio prefix before
the 24 encoder layers. It starts after CPU metadata construction and padded
audio chunk creation, captures:

1. `conv2d1` + exact PyTorch GELU
2. `conv2d2` + exact PyTorch GELU
3. `conv2d3` + exact PyTorch GELU
4. `conv_out`
5. positional embedding add
6. valid-row pack to `[M, 1024]`

The transformer/audio suffix remains the accepted CPU-metadata path.

## Guardrails

- Env gate: `ASR_AUDIO_PREFIX_CUDAGRAPH=1`.
- Requires `ASR_AUDIO_CPU_MAXSEQLEN=1` and `ASR_AUDIO_CPU_METADATA_PACK=1`.
- Admits only the observed common 21-chunk / 13-row B16 trace bucket with
  packed rows in `{264, 265, 267, 268, 270, 272, 273}`.
- The graph key includes padded tensor shape/stride, chunk lengths, pack
  metadata, `cu_seqlens`, whole-audio feature lengths, and after-CNN lengths.
- Each exact key stays eager for seven observations, captures on observation
  eight, and replays only on later calls. Probation state is bounded and LRU.
- Cache misses capture and admit only after bitwise equality against eager.
- Capture failures or equality failures mark the exact key rejected and fall
  back to eager.
- Replays use long-lived static input/metadata/output buffers. Each entry has a
  host lock plus a dedicated replay stream and CUDA event so cross-stream calls
  serialize around the shared buffers without a hot-path device synchronize.
- Replay returns `output.clone()` so downstream cache/storage cannot observe
  the next replay's graph-owned output.

## CUDA gate

Run on an idle GPU from this worktree:

```bash
UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
uv run python inference/vllm_static_fp8/bench_audio_prefix_cudagraph.py \
  --rows 272 \
  --repeats 30
```

The helper checks:

- bitwise equality for the first admitted shape;
- bitwise equality for the same shape with different input contents;
- two-thread / two-stream same-key replay stress;
- median eager host/CUDA time;
- median candidate input-copy + graph-replay + output-clone CUDA time;
- eager and candidate kernel order from `torch.profiler`.

No GPU gate has been run in this commit.

## Service command

Terminal 1:

```bash
UV_PROJECT_ENVIRONMENT=/mnt/ssd1/shreyansh/home_dir/asr_experiments/.venv \
bash inference/run_vllm_fp8_static_qk_prefill_audio_prefix_cudagraph.sh
```

Terminal 2, uniform B16 run with an isolated output root:

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
  --output-root /tmp/audio_prefix_cudagraph_b16_50s \
  --base-url http://127.0.0.1:8091/v1
```

Full-set validation should use the same server and command shape without
`--uniform-audio-length` or `--num-files`, and with a fresh output root.
