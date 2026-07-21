"""CUDA gate for the exact-shape Qwen3-ASR audio prefix graph candidate."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import statistics
import time

import torch
from torch import nn

from audio_cpu_metadata_pack_patch import run_audio_prefix_eager
from audio_prefix_cudagraph_patch import (
    ExactShapeAudioPrefixGraphCache,
)
from vllm.utils.torch_utils import async_tensor_h2d


class PrefixOnlyEncoder(nn.Module):
    def __init__(self, chunks: int) -> None:
        super().__init__()
        self.training = False
        self.conv_chunksize = chunks + 1
        self.conv2d1 = nn.Conv2d(
            1,
            480,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=torch.bfloat16,
        )
        self.conv2d2 = nn.Conv2d(
            480,
            480,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=torch.bfloat16,
        )
        self.conv2d3 = nn.Conv2d(
            480,
            480,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=torch.bfloat16,
        )
        self.conv_out = nn.Linear(480 * 16, 1024, dtype=torch.bfloat16)
        self.positional_embedding = _PositionalEmbedding()


class _PositionalEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(13, 1024, dtype=torch.bfloat16),
            requires_grad=False,
        )


def _metadata(total_rows: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    chunks = 21
    lengths = [13] * chunks
    lengths[-1] = total_rows - sum(lengths[:-1])
    if lengths[-1] <= 0 or lengths[-1] > 13:
        raise ValueError("The helper targets the observed 21-chunk shape bucket")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return (
        torch.tensor([100] * chunks, dtype=torch.int64),
        torch.tensor(lengths + offsets[:-1], dtype=torch.int32),
        (0, 104, 208, total_rows),
        (800, 800, 500),
        (104, 104, total_rows - 208),
    )


def _cuda_timed_us(function, repeats: int) -> float:
    samples = []
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


def _host_timed_us(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        output = function()
        samples.append((time.perf_counter_ns() - start) / 1_000)
        del output
    torch.cuda.synchronize()
    return statistics.median(samples)


def _kernel_order(function) -> list[str]:
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        output = function()
    del output
    return [
        event.key
        for event in profile.events()
        if event.device_type == torch.profiler.ProfilerActivity.CUDA
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=272)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the prefix graph gate")
    if args.rows not in {264, 265, 267, 268, 270, 272, 273}:
        raise ValueError("rows must be one of the admitted trace bucket sizes")

    torch.manual_seed(0)
    device = torch.device("cuda")
    encoder = PrefixOnlyEncoder(chunks=21).to(device=device).eval()
    metadata = _metadata(args.rows)
    padded = torch.randn(
        (21, 1, 128, 100),
        device=device,
        dtype=torch.bfloat16,
    )
    other_padded = torch.randn_like(padded)
    cache = ExactShapeAudioPrefixGraphCache()

    def eager(input_tensor: torch.Tensor = padded) -> torch.Tensor:
        return run_audio_prefix_eager(
            encoder,
            input_tensor,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )

    def candidate(input_tensor: torch.Tensor = padded) -> torch.Tensor:
        return cache.run(
            encoder,
            input_tensor,
            *metadata,
            async_tensor_h2d=async_tensor_h2d,
        )

    with torch.inference_mode():
        eager_output = eager()
        for _ in range(7):
            candidate_output = candidate()
            if cache.entry_count != 0:
                raise AssertionError("prefix graph captured before observation 8")
        candidate_output = candidate()
        if cache.entry_count != 1:
            raise AssertionError("prefix graph was not captured on observation 8")
        torch.cuda.synchronize()
        torch.testing.assert_close(candidate_output, eager_output, rtol=0, atol=0)
        other_eager_output = eager(other_padded)
        other_candidate_output = candidate(other_padded)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            other_candidate_output,
            other_eager_output,
            rtol=0,
            atol=0,
        )

        def threaded(seed: int) -> torch.Tensor:
            stream = torch.cuda.Stream()
            local = torch.randn(
                (21, 1, 128, 100),
                device=device,
                dtype=torch.bfloat16,
            )
            local.mul_(seed)
            with torch.cuda.stream(stream):
                output = candidate(local)
            stream.synchronize()
            torch.testing.assert_close(output, eager(local), rtol=0, atol=0)
            return output

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(threaded, (3, 5)))

        for _ in range(3):
            eager()
            candidate()
        torch.cuda.synchronize()

        print("correctness=exact")
        print(f"output_shape={tuple(candidate_output.shape)}")
        print(f"cache_entries={cache.entry_count}")
        print(f"eager_host_us={_host_timed_us(eager, args.repeats):.3f}")
        print(f"candidate_host_us={_host_timed_us(candidate, args.repeats):.3f}")
        print(f"eager_cuda_us={_cuda_timed_us(eager, args.repeats):.3f}")
        print(f"candidate_copy_replay_clone_cuda_us={_cuda_timed_us(candidate, args.repeats):.3f}")
        print("eager_kernel_order=" + " | ".join(_kernel_order(eager)))
        print("candidate_kernel_order=" + " | ".join(_kernel_order(candidate)))


if __name__ == "__main__":
    main()
