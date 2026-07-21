#!/usr/bin/env python3
"""CUDA gate for the chained Qwen3-ASR audio prefix and suffix graphs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

import torch

from audio_cpu_metadata_pack_patch import (
    run_audio_prefix_eager,
    run_audio_suffix_eager,
)
from audio_prefix_cudagraph_patch import ExactShapeAudioPrefixGraphCache
from audio_suffix_cudagraph_patch import ExactShapeAudioSuffixGraphCache
from bench_audio_prefix_cudagraph import (
    _cuda_timed_us,
    _host_timed_us,
    _metadata,
)
from bench_audio_suffix_cudagraph import (
    _initialize_single_gpu_distributed,
    _new_encoder,
)
from vllm.utils.torch_utils import async_tensor_h2d


def _concurrent_chain_gate(
    encoder,
    prefix_cache,
    suffix_cache,
    metadata,
    cu_seqlens,
    max_seqlen,
    *,
    device: torch.device,
    generator: torch.Generator,
    iterations: int,
) -> None:
    inputs_by_thread = [
        [
            torch.randn(
                (21, 1, 128, 100),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in range(iterations)
        ]
        for _ in range(2)
    ]

    def eager(padded_feature):
        hidden_states = run_audio_prefix_eager(
            encoder,
            padded_feature,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )
        return run_audio_suffix_eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=metadata[2],
        )

    with torch.inference_mode():
        expected_by_thread = [
            [eager(padded_feature) for padded_feature in thread_inputs]
            for thread_inputs in inputs_by_thread
        ]
    torch.cuda.synchronize(device)

    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    barrier = threading.Barrier(2)

    def replay_many(thread_index: int) -> list[torch.Tensor]:
        stream = streams[thread_index]
        torch.cuda.set_device(device)
        outputs = []
        try:
            with torch.inference_mode(), torch.cuda.stream(stream):
                for padded_feature in inputs_by_thread[thread_index]:
                    barrier.wait()
                    hidden_states = prefix_cache.run(
                        encoder,
                        padded_feature,
                        *metadata,
                        async_tensor_h2d=async_tensor_h2d,
                    )
                    outputs.append(
                        suffix_cache.run(
                            encoder,
                            hidden_states,
                            cu_seqlens,
                            max_seqlen,
                            cu_seqlens_values=metadata[2],
                        )
                    )
            stream.synchronize()
            return outputs
        except BaseException:
            barrier.abort()
            raise

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs_by_thread = list(executor.map(replay_many, (0, 1)))

    for thread_index, (outputs, expected_outputs) in enumerate(
        zip(outputs_by_thread, expected_by_thread, strict=True)
    ):
        for iteration, (output, expected) in enumerate(
            zip(outputs, expected_outputs, strict=True)
        ):
            if not torch.equal(output, expected):
                raise AssertionError(
                    "combined replay is not bitwise exact for "
                    f"thread={thread_index}, iteration={iteration}"
                )


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=272)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--concurrency-iterations", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29619)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows not in {264, 265, 267, 268, 270, 272, 273}:
        raise ValueError("rows must be one of the admitted trace bucket sizes")
    if args.warmup <= 0 or args.repeats <= 0 or args.concurrency_iterations <= 0:
        raise ValueError(
            "--warmup, --repeats, and --concurrency-iterations must be positive"
        )
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError("This production-shape gate requires an SM90 GPU")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _initialize_single_gpu_distributed(args.master_port)
    encoder = _new_encoder(device)
    metadata = _metadata(args.rows)
    cu_seqlens = torch.tensor(metadata[2], dtype=torch.int32, device=device)
    max_seqlen = torch.tensor(104, dtype=torch.int32, device="cpu")
    prefix_cache = ExactShapeAudioPrefixGraphCache(
        warmup_iterations=args.warmup,
    )
    suffix_cache = ExactShapeAudioSuffixGraphCache(
        warmup_iterations=args.warmup,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    padded = torch.randn(
        (21, 1, 128, 100),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    def eager(input_tensor=padded):
        hidden_states = run_audio_prefix_eager(
            encoder,
            input_tensor,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )
        return run_audio_suffix_eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=metadata[2],
        )

    def candidate(input_tensor=padded):
        hidden_states = prefix_cache.run(
            encoder,
            input_tensor,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )
        return suffix_cache.run(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=metadata[2],
        )

    with torch.inference_mode():
        eager_output = eager()
        for _ in range(7):
            probation_output = candidate()
            torch.cuda.synchronize(device)
            if not torch.equal(probation_output, eager_output):
                raise AssertionError("combined probation output is not exact")
            if prefix_cache.entry_count != 0:
                raise AssertionError("prefix graph captured before observation 8")

        eighth_output = candidate()
        torch.cuda.synchronize(device)
        if not torch.equal(eighth_output, eager_output):
            raise AssertionError("combined observation-8 output is not exact")
        if prefix_cache.entry_count != 1 or suffix_cache.entry_count != 1:
            raise AssertionError("both graph caches must be admitted by call 8")

        other_padded = torch.randn_like(padded)
        other_eager = eager(other_padded)
        other_candidate = candidate(other_padded)
        torch.cuda.synchronize(device)
        if not torch.equal(other_candidate, other_eager):
            raise AssertionError("different-content combined replay is not exact")

        _concurrent_chain_gate(
            encoder,
            prefix_cache,
            suffix_cache,
            metadata,
            cu_seqlens,
            max_seqlen,
            device=device,
            generator=generator,
            iterations=args.concurrency_iterations,
        )

        for _ in range(args.warmup):
            eager()
            candidate()
        torch.cuda.synchronize(device)

        print("correctness=bitwise_exact")
        print("probation=PASS observations=8")
        print("concurrency=PASS threads=2 streams=2")
        print(f"rows={args.rows}")
        print(f"eager_host_us={_host_timed_us(eager, args.repeats):.3f}")
        print(f"combined_host_us={_host_timed_us(candidate, args.repeats):.3f}")
        print(f"eager_cuda_us={_cuda_timed_us(eager, args.repeats):.3f}")
        print(
            "combined_copy_replay_clone_cuda_us="
            f"{_cuda_timed_us(candidate, args.repeats):.3f}"
        )
        print("gate=PASS_EXACT_AUDIO_PREFIX_SUFFIX_CUDAGRAPH")


def main() -> None:
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
