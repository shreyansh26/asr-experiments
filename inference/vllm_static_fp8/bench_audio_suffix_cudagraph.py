#!/usr/bin/env python3
"""Gate exact-shape capture for the 24-layer post-pack audio suffix on CUDA."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch

from audio_cpu_metadata_pack_patch import run_audio_suffix_eager
from audio_suffix_cudagraph_patch import (
    _MAX_CACHE_ENTRIES,
    _SUPPORTED_ROWS,
    ExactShapeAudioSuffixGraphCache,
)


def _parse_segments(raw_value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in raw_value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "segments must be comma-separated positive integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "segments must be comma-separated positive integers"
        )
    return values


def _cu_metadata(
    segments: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    cumulative = [0]
    for length in segments:
        cumulative.append(cumulative[-1] + length)
    values = tuple(cumulative)
    cu_seqlens = torch.tensor(values, dtype=torch.int32, device=device)
    max_seqlen = torch.tensor(
        104,
        dtype=torch.int32,
        device="cpu",
    )
    return cu_seqlens, max_seqlen, values


def _initialize_single_gpu_distributed(port: int) -> None:
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


def _new_encoder(device: torch.device) -> torch.nn.Module:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from vllm.model_executor.models.qwen3_asr import Qwen3OmniMoeAudioEncoder
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    config = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=128,
        encoder_layers=24,
        encoder_attention_heads=16,
        encoder_ffn_dim=4096,
        d_model=1024,
        output_dim=2048,
        activation_function="gelu",
        n_window=50,
        n_window_infer=800,
        conv_chunksize=500,
        downsample_hidden_size=480,
    )
    with set_default_torch_dtype(torch.bfloat16), torch.device(device):
        encoder = Qwen3OmniMoeAudioEncoder(
            config,
            prefix="audio_tower",
        )
    encoder.eval()
    if len(encoder.layers) != 24:
        raise RuntimeError("Expected the Qwen3-ASR 24-layer audio encoder")
    if encoder.attn_backend != AttentionBackendEnum.FLASH_ATTN:
        raise RuntimeError(
            f"Expected FLASH_ATTN audio backend, got {encoder.attn_backend}"
        )

    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            if parameter.ndim == 1:
                if name.endswith("weight"):
                    parameter.fill_(1)
                else:
                    parameter.zero_()
            else:
                parameter.normal_(mean=0.0, std=0.01, generator=generator)
    return encoder


def _median_cuda_us(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000)
        del output
    return statistics.median(samples)


def _median_host_call_us(
    function: Callable[[], torch.Tensor],
    *,
    repeats: int,
) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = function()
        samples.append((time.perf_counter_ns() - start) / 1_000)
        del output
    torch.cuda.synchronize()
    return statistics.median(samples)


def _cuda_kernel_names(function: Callable[[], object]) -> list[str]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        function()
        torch.cuda.synchronize()
    return [
        event.name
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segments",
        action="append",
        type=_parse_segments,
        help=(
            "Comma-separated exact sequence lengths. Repeat for every observed "
            "shape/content key. Defaults cover all whitelisted row counts."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--master-port", type=int, default=29617)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup <= 0 or args.repeats <= 0:
        raise ValueError("--warmup and --repeats must be positive")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError("This production-shape gate requires an SM90 GPU")

    cases = args.segments or [
        (88, 88, 88),
        (88, 88, 89),
        (89, 89, 89),
        (89, 89, 90),
        (90, 90, 90),
        (90, 91, 91),
        (91, 91, 91),
    ]
    if any(max(segments) > 104 for segments in cases):
        raise ValueError(
            "Each segment must be <=104 to match the accepted CPU max-seqlen "
            "contract"
        )
    if (
        len(cases) > _MAX_CACHE_ENTRIES
        or any(len(segments) != 3 for segments in cases)
        or any(sum(segments) not in _SUPPORTED_ROWS for segments in cases)
    ):
        raise ValueError(
            "Expected at most seven three-segment cases with a supported total "
            f"row count in {sorted(_SUPPORTED_ROWS)}"
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _initialize_single_gpu_distributed(args.master_port)
    encoder = _new_encoder(device)
    cache = ExactShapeAudioSuffixGraphCache(
        max_entries=_MAX_CACHE_ENTRIES,
        warmup_iterations=args.warmup,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(
        "case,rows,sequences,bitwise_exact,kernel_order_exact,"
        "eager_kernels,replay_kernels,eager_cuda_us,"
        "replay_with_input_copy_cuda_us,cuda_speedup_pct,eager_host_call_us,"
        "replay_with_input_copy_host_call_us,host_speedup_pct"
    )
    with torch.inference_mode():
        for case_index, segments in enumerate(cases):
            cu_seqlens, max_seqlen, cu_values = _cu_metadata(segments, device)
            rows = sum(segments)
            initial_hidden = torch.randn(
                (rows, 1024),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )

            # First call performs side-stream warmup, capture, and its own
            # bitwise eager-vs-replay admission check for this exact key.
            cache.run(
                encoder,
                initial_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            replay_hidden = torch.randn(
                (rows, 1024),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            eager_output = run_audio_suffix_eager(
                encoder,
                replay_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            replay_output = cache.run(
                encoder,
                replay_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            torch.cuda.synchronize()
            bitwise_exact = torch.equal(eager_output, replay_output)
            if not bitwise_exact:
                raise AssertionError(
                    f"suffix replay is not bitwise exact for {segments}"
                )

            key = next(
                key
                for key in cache._entries
                if key.cu_seqlens_values == cu_values
            )
            entry = cache._entries[key]
            eager_kernels = _cuda_kernel_names(
                lambda: run_audio_suffix_eager(
                    encoder,
                    replay_hidden,
                    cu_seqlens,
                    max_seqlen,
                    cu_seqlens_values=cu_values,
                )
            )
            entry.static_hidden_states.copy_(replay_hidden)
            torch.cuda.synchronize()
            replay_kernels = _cuda_kernel_names(
                lambda: cache._backend.replay(entry.graph)
            )
            kernel_order_exact = eager_kernels == replay_kernels
            if not kernel_order_exact:
                raise AssertionError(
                    "captured suffix kernel names/order differ from eager for "
                    f"{segments}"
                )

            eager_call = lambda: run_audio_suffix_eager(  # noqa: E731
                encoder,
                replay_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            replay_call = lambda: cache.run(  # noqa: E731
                encoder,
                replay_hidden,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=cu_values,
            )
            eager_us = _median_cuda_us(
                eager_call,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            replay_us = _median_cuda_us(
                replay_call,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            eager_host_us = _median_host_call_us(
                eager_call,
                repeats=args.repeats,
            )
            replay_host_us = _median_host_call_us(
                replay_call,
                repeats=args.repeats,
            )
            cuda_speedup_pct = (eager_us / replay_us - 1) * 100
            host_speedup_pct = (eager_host_us / replay_host_us - 1) * 100
            print(
                f"{case_index},{rows},{len(segments)},{bitwise_exact},"
                f"{kernel_order_exact},{len(eager_kernels)},"
                f"{len(replay_kernels)},{eager_us:.3f},{replay_us:.3f},"
                f"{cuda_speedup_pct:.3f},{eager_host_us:.3f},"
                f"{replay_host_us:.3f},{host_speedup_pct:.3f}"
            )

    if cache.entry_count != len(cases):
        raise AssertionError(
            f"expected {len(cases)} exact graph keys, got {cache.entry_count}"
        )
    print("gate=PASS_EXACT_AUDIO_SUFFIX_CUDAGRAPH")


def main() -> None:
    # Direct module construction needs the same config context that vLLM's
    # production model loader installs around initialization and execution.
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        try:
            _main()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
