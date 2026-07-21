"""Exact-shape CUDA graph cache for the Qwen3-ASR pre-layer audio prefix."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import os
import threading
from typing import Any, Callable

import torch
import torch.nn.functional as F

from audio_cpu_metadata_pack_patch import (
    ENV_NAME as METADATA_ENV_NAME,
    MAX_SEQLEN_ENV_NAME,
    _PATCH_MARKER as METADATA_PATCH_MARKER,
    install_audio_cpu_metadata_pack_patch,
    run_audio_prefix_eager,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton


logger = init_logger("vllm.qwen3_asr_audio_prefix_cudagraph")

ENV_NAME = "ASR_AUDIO_PREFIX_CUDAGRAPH"
_PATCH_MARKER = "_asr_audio_prefix_cudagraph_patch"
_CACHE_ATTR = "_asr_audio_prefix_cudagraph_cache"
_CACHE_CREATION_LOCK = threading.Lock()
_INPUT_CHANNELS = 1
_INPUT_MELS = 128
_INPUT_FRAMES = 100
_PACKED_WIDTH = 1024
_CONV_OUTPUT_ROWS = 13
_SUPPORTED_CHUNKS = frozenset({21})
_SUPPORTED_PACKED_ROWS = frozenset({264, 265, 267, 268, 270, 272, 273})
_MAX_CACHE_ENTRIES = len(_SUPPORTED_PACKED_ROWS)
_WARMUP_ITERATIONS = 3
_CAPTURE_OBSERVATIONS = 8


@triton.jit
def _pack_valid_rows_device_kernel(
    padded_ptr,
    metadata_ptr,
    output_ptr,
    padded_batch_stride,
    padded_row_stride,
    padded_hidden_stride,
    num_chunks,
    padded_rows,
    hidden_size: tl.constexpr,
    block_hidden: tl.constexpr,
):
    program = tl.program_id(0)
    chunk = program // padded_rows
    row = program % padded_rows
    columns = tl.arange(0, block_hidden)

    valid_rows = tl.load(metadata_ptr + chunk).to(tl.int64)
    output_offset = tl.load(metadata_ptr + num_chunks + chunk).to(tl.int64)
    valid = (row < valid_rows) & (columns < hidden_size)

    values = tl.load(
        padded_ptr
        + chunk * padded_batch_stride
        + row * padded_row_stride
        + columns * padded_hidden_stride,
        mask=valid,
    )
    tl.store(
        output_ptr + (output_offset + row) * hidden_size + columns,
        values,
        mask=valid,
    )


@dataclass(frozen=True)
class PrefixGraphKey:
    """All static launch and CPU-metadata state for one prefix graph."""

    padded_shape: tuple[int, int, int, int]
    padded_stride: tuple[int, int, int, int]
    packed_rows: int
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    chunk_lengths_values: tuple[int, ...]
    pack_metadata_values: tuple[int, ...]
    cu_seqlens_values: tuple[int, ...]
    feature_lens_values: tuple[int, ...]
    aftercnn_lens_values: tuple[int, ...]


@dataclass
class _PrefixGraphEntry:
    key: PrefixGraphKey
    static_padded_feature: torch.Tensor
    static_pack_metadata: torch.Tensor
    graph: Any
    output: torch.Tensor
    replay_stream: Any
    done_event: Any
    lock: threading.Lock


def audio_prefix_cudagraph_enabled() -> bool:
    """Return the strict boolean environment gate for this experiment."""
    raw_value = os.environ.get(ENV_NAME, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(f"{ENV_NAME} must be 0 or 1, got {raw_value!r}")
    return raw_value == "1"


def _metadata_tuple(values: torch.Tensor | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(values, tuple):
        return tuple(int(value) for value in values)
    if values.device.type != "cpu" or values.ndim != 1:
        raise ValueError("Audio prefix metadata must be one-dimensional CPU data")
    return tuple(int(value) for value in values.tolist())


def _validate_pack_metadata_values(
    pack_metadata_values: tuple[int, ...],
    *,
    num_chunks: int,
) -> int | None:
    if len(pack_metadata_values) != num_chunks * 2:
        return None
    lengths = pack_metadata_values[:num_chunks]
    offsets = pack_metadata_values[num_chunks:]
    running_offset = 0
    for length, offset in zip(lengths, offsets, strict=True):
        if length <= 0 or length > _CONV_OUTPUT_ROWS or offset != running_offset:
            return None
        running_offset += length
    return running_offset


def _make_prefix_graph_key(
    padded_feature: torch.Tensor,
    chunk_lengths_cpu: torch.Tensor,
    pack_metadata_cpu: torch.Tensor,
    cu_seqlens_values: tuple[int, ...],
    feature_lens_values: tuple[int, ...],
    aftercnn_lens_values: tuple[int, ...],
) -> PrefixGraphKey | None:
    if (
        not isinstance(padded_feature, torch.Tensor)
        or padded_feature.ndim != 4
        or padded_feature.shape[1:] != (_INPUT_CHANNELS, _INPUT_MELS, _INPUT_FRAMES)
        or padded_feature.dtype != torch.bfloat16
        or not isinstance(chunk_lengths_cpu, torch.Tensor)
        or chunk_lengths_cpu.device.type != "cpu"
        or chunk_lengths_cpu.ndim != 1
        or chunk_lengths_cpu.dtype not in {torch.int32, torch.int64}
        or not isinstance(pack_metadata_cpu, torch.Tensor)
        or pack_metadata_cpu.device.type != "cpu"
        or pack_metadata_cpu.ndim != 1
        or pack_metadata_cpu.dtype != torch.int32
    ):
        return None

    num_chunks = int(padded_feature.shape[0])
    chunk_lengths_values = _metadata_tuple(chunk_lengths_cpu)
    pack_metadata_values = _metadata_tuple(pack_metadata_cpu)
    packed_rows = _validate_pack_metadata_values(
        pack_metadata_values,
        num_chunks=num_chunks,
    )
    if (
        packed_rows is None
        or num_chunks not in _SUPPORTED_CHUNKS
        or packed_rows not in _SUPPORTED_PACKED_ROWS
        or len(chunk_lengths_values) != num_chunks
        or any(length <= 0 or length > _INPUT_FRAMES for length in chunk_lengths_values)
        or not cu_seqlens_values
        or cu_seqlens_values[0] != 0
        or cu_seqlens_values[-1] != packed_rows
        or any(
            left >= right
            for left, right in zip(cu_seqlens_values, cu_seqlens_values[1:])
        )
        or sum(aftercnn_lens_values) != packed_rows
        or len(feature_lens_values) != len(aftercnn_lens_values)
    ):
        return None

    return PrefixGraphKey(
        padded_shape=tuple(int(value) for value in padded_feature.shape),
        padded_stride=tuple(int(value) for value in padded_feature.stride()),
        packed_rows=packed_rows,
        dtype=padded_feature.dtype,
        device_type=padded_feature.device.type,
        device_index=padded_feature.device.index,
        chunk_lengths_values=chunk_lengths_values,
        pack_metadata_values=pack_metadata_values,
        cu_seqlens_values=tuple(int(value) for value in cu_seqlens_values),
        feature_lens_values=tuple(int(value) for value in feature_lens_values),
        aftercnn_lens_values=tuple(int(value) for value in aftercnn_lens_values),
    )


def _output_is_supported(output: Any, key: PrefixGraphKey) -> bool:
    return (
        isinstance(output, torch.Tensor)
        and output.shape == (key.packed_rows, _PACKED_WIDTH)
        and output.dtype == key.dtype
        and output.device.type == key.device_type
        and output.device.index == key.device_index
    )


def pack_valid_rows_from_device_metadata(
    padded: torch.Tensor,
    metadata: torch.Tensor,
    *,
    total_rows: int,
) -> torch.Tensor:
    """Pack `[chunk, row, hidden]` using graph-static device metadata."""
    if (
        padded.ndim != 3
        or padded.dtype != torch.bfloat16
        or padded.shape[1:] != (_CONV_OUTPUT_ROWS, _PACKED_WIDTH)
        or metadata.ndim != 1
        or metadata.dtype != torch.int32
        or metadata.numel() != padded.shape[0] * 2
        or metadata.device != padded.device
    ):
        raise ValueError("Unsupported tensor layout for graph audio row pack")

    num_chunks, padded_rows, hidden_size = padded.shape
    if padded.is_cuda:
        output = torch.empty(
            (total_rows, hidden_size),
            device=padded.device,
            dtype=padded.dtype,
        )
        block_hidden = triton.next_power_of_2(hidden_size)
        _pack_valid_rows_device_kernel[(num_chunks * padded_rows,)](
            padded,
            metadata,
            output,
            padded.stride(0),
            padded.stride(1),
            padded.stride(2),
            num_chunks,
            padded_rows,
            hidden_size,
            block_hidden,
            num_warps=8,
        )
        return output

    metadata_values = _metadata_tuple(metadata.cpu())
    lengths = metadata_values[:num_chunks]
    return torch.cat(
        [padded[index, :length] for index, length in enumerate(lengths)],
        dim=0,
    )


def run_audio_prefix_with_device_metadata(
    encoder: Any,
    padded_feature: torch.Tensor,
    pack_metadata: torch.Tensor,
    *,
    total_rows: int,
) -> torch.Tensor:
    """Run the exact prefix with already-materialized metadata."""
    padded_embed = F.gelu(encoder.conv2d1(padded_feature))
    padded_embed = F.gelu(encoder.conv2d2(padded_embed))
    padded_embed = F.gelu(encoder.conv2d3(padded_embed))

    batch, channels, frequency, time = padded_embed.size()
    padded_embed = encoder.conv_out(
        padded_embed.permute(0, 3, 1, 2)
        .contiguous()
        .view(batch, time, channels * frequency)
    )
    positional_embedding = (
        encoder.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding
    return pack_valid_rows_from_device_metadata(
        padded_embed,
        pack_metadata,
        total_rows=total_rows,
    )


class _TorchCudaGraphBackend:
    """Small CUDA API seam so cache behavior has CUDA-hidden fake tests."""

    def supports(self, padded_feature: torch.Tensor) -> bool:
        return (
            torch.cuda.is_available()
            and padded_feature.is_cuda
            and padded_feature.dtype == torch.bfloat16
        )

    def is_current_stream_capturing(self) -> bool:
        return torch.cuda.is_current_stream_capturing()

    def make_device_metadata(
        self,
        metadata_cpu: torch.Tensor,
        *,
        device: torch.device,
        async_tensor_h2d: Any,
    ) -> torch.Tensor:
        return async_tensor_h2d(metadata_cpu, dtype=torch.int32, device=device)

    def capture(
        self,
        function: Callable[[], torch.Tensor],
        *,
        device: torch.device,
        warmup_iterations: int,
    ) -> tuple[Any, torch.Tensor, Any, Any]:
        caller_stream = torch.cuda.current_stream(device)
        replay_stream = torch.cuda.Stream(device=device)
        replay_stream.wait_stream(caller_stream)
        with torch.cuda.stream(replay_stream):
            for _ in range(warmup_iterations):
                function()
        caller_stream.wait_stream(replay_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=replay_stream,
            capture_error_mode="thread_local",
        ):
            output = function()
        done_event = torch.cuda.Event(blocking=False)
        done_event.record(replay_stream)
        caller_stream.wait_event(done_event)
        return graph, output, replay_stream, done_event

    def replay_context(self, replay_stream: Any) -> Any:
        return torch.cuda.stream(replay_stream)

    def wait_stream(self, replay_stream: Any, device: torch.device) -> None:
        replay_stream.wait_stream(torch.cuda.current_stream(device))

    def record_done(self, done_event: Any, replay_stream: Any) -> None:
        done_event.record(replay_stream)

    def wait_done(self, done_event: Any, device: torch.device) -> None:
        torch.cuda.current_stream(device).wait_event(done_event)

    def replay(self, graph: Any) -> None:
        graph.replay()

    def replay_and_clone(
        self,
        entry: _PrefixGraphEntry,
        padded_feature: torch.Tensor,
    ) -> torch.Tensor:
        caller_stream = torch.cuda.current_stream(padded_feature.device)
        replay_stream = entry.replay_stream

        replay_stream.wait_stream(caller_stream)
        with torch.cuda.stream(replay_stream):
            entry.static_padded_feature.copy_(
                padded_feature,
                non_blocking=True,
            )
            # The asynchronous copy may outlive the caller's Python reference.
            padded_feature.record_stream(replay_stream)
            entry.graph.replay()
            output = entry.output.clone()
            entry.done_event.record(replay_stream)

        caller_stream.wait_event(entry.done_event)
        # Keep clone storage live through its asynchronous caller-stream use.
        output.record_stream(caller_stream)
        return output

    def equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        return torch.equal(left, right)


class ExactShapeAudioPrefixGraphCache:
    """Capture one verified prefix graph per exact padded shape and metadata."""

    def __init__(
        self,
        *,
        backend: Any | None = None,
        max_entries: int = _MAX_CACHE_ENTRIES,
        warmup_iterations: int = _WARMUP_ITERATIONS,
    ) -> None:
        if max_entries <= 0 or warmup_iterations <= 0:
            raise ValueError("CUDA graph cache limits must be positive")
        self._backend = backend or _TorchCudaGraphBackend()
        self._max_entries = max_entries
        self._warmup_iterations = warmup_iterations
        self._entries: dict[PrefixGraphKey, _PrefixGraphEntry] = {}
        self._rejected_keys: set[PrefixGraphKey] = set()
        self._probation_counts: OrderedDict[PrefixGraphKey, int] = OrderedDict()
        self._global_lock = threading.Lock()
        self._logged_capacity = False
        self._logged_replay = False

    @property
    def entry_count(self) -> int:
        with self._global_lock:
            return len(self._entries)

    @property
    def rejected_count(self) -> int:
        with self._global_lock:
            return len(self._rejected_keys)

    @property
    def probation_key_count(self) -> int:
        with self._global_lock:
            return len(self._probation_counts)

    def _observe_key_locked(self, key: PrefixGraphKey) -> bool:
        observations = self._probation_counts.pop(key, 0) + 1
        if observations >= _CAPTURE_OBSERVATIONS:
            return True

        self._probation_counts[key] = observations
        while len(self._probation_counts) > self._max_entries:
            self._probation_counts.popitem(last=False)
        return False

    def _eager(
        self,
        encoder: Any,
        padded_feature: torch.Tensor,
        chunk_lengths_cpu: torch.Tensor,
        pack_metadata_cpu: torch.Tensor,
        cu_seqlens_values: tuple[int, ...],
        feature_lens_values: tuple[int, ...],
        aftercnn_lens_values: tuple[int, ...],
        async_tensor_h2d: Any,
    ) -> torch.Tensor:
        return run_audio_prefix_eager(
            encoder,
            padded_feature,
            chunk_lengths_cpu,
            pack_metadata_cpu,
            cu_seqlens_values,
            feature_lens_values,
            aftercnn_lens_values,
            async_tensor_h2d=async_tensor_h2d,
        )

    def run(
        self,
        encoder: Any,
        padded_feature: torch.Tensor,
        chunk_lengths_cpu: torch.Tensor,
        pack_metadata_cpu: torch.Tensor,
        cu_seqlens_values: tuple[int, ...],
        feature_lens_values: tuple[int, ...],
        aftercnn_lens_values: tuple[int, ...],
        *,
        async_tensor_h2d: Any,
    ) -> torch.Tensor:
        key = _make_prefix_graph_key(
            padded_feature,
            chunk_lengths_cpu,
            pack_metadata_cpu,
            cu_seqlens_values,
            feature_lens_values,
            aftercnn_lens_values,
        )
        if (
            key is None
            or bool(getattr(encoder, "training", True))
            or torch.is_grad_enabled()
            or padded_feature.size(0) > getattr(encoder, "conv_chunksize", 0)
            or not self._backend.supports(padded_feature)
            or self._backend.is_current_stream_capturing()
        ):
            return self._eager(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                tuple(cu_seqlens_values),
                tuple(feature_lens_values),
                tuple(aftercnn_lens_values),
                async_tensor_h2d,
            )

        if key in self._rejected_keys:
            return self._eager(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                key.cu_seqlens_values,
                key.feature_lens_values,
                key.aftercnn_lens_values,
                async_tensor_h2d,
            )

        capture_output: torch.Tensor | None = None
        with self._global_lock:
            entry = self._entries.get(key)
            if entry is None and key in self._rejected_keys:
                pass
            elif entry is None and len(self._entries) < self._max_entries:
                if self._observe_key_locked(key):
                    entry, capture_output = self._capture_entry(
                        encoder,
                        padded_feature,
                        chunk_lengths_cpu,
                        pack_metadata_cpu,
                        key,
                        async_tensor_h2d,
                    )
            elif entry is None and not self._logged_capacity:
                self._probation_counts.pop(key, None)
                logger.warning(
                    "Audio prefix CUDA graph cache is full at %d exact keys; "
                    "using eager prefix for shape=%s",
                    self._max_entries,
                    key.padded_shape,
                )
                self._logged_capacity = True

        if capture_output is not None:
            return capture_output
        if entry is None:
            return self._eager(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                key.cu_seqlens_values,
                key.feature_lens_values,
                key.aftercnn_lens_values,
                async_tensor_h2d,
            )

        with entry.lock:
            output = self._backend.replay_and_clone(entry, padded_feature)
        if not self._logged_replay:
            logger.info(
                "ASR audio prefix CUDA graph replay active for shape=%s rows=%d",
                key.padded_shape,
                key.packed_rows,
            )
            self._logged_replay = True
        return output

    def _capture_entry(
        self,
        encoder: Any,
        padded_feature: torch.Tensor,
        chunk_lengths_cpu: torch.Tensor,
        pack_metadata_cpu: torch.Tensor,
        key: PrefixGraphKey,
        async_tensor_h2d: Any,
    ) -> tuple[_PrefixGraphEntry | None, torch.Tensor | None]:
        reference_output: torch.Tensor | None = None
        try:
            static_padded_feature = torch.empty_strided(
                key.padded_shape,
                key.padded_stride,
                device=padded_feature.device,
                dtype=padded_feature.dtype,
            )
            static_pack_metadata = self._backend.make_device_metadata(
                pack_metadata_cpu,
                device=padded_feature.device,
                async_tensor_h2d=async_tensor_h2d,
            )
            static_padded_feature.copy_(padded_feature, non_blocking=True)

            def static_prefix() -> torch.Tensor:
                return run_audio_prefix_with_device_metadata(
                    encoder,
                    static_padded_feature,
                    static_pack_metadata,
                    total_rows=key.packed_rows,
                )

            graph, graph_output, replay_stream, done_event = self._backend.capture(
                static_prefix,
                device=padded_feature.device,
                warmup_iterations=self._warmup_iterations,
            )
            if not _output_is_supported(graph_output, key):
                raise RuntimeError("Captured audio prefix did not return [M, 1024]")

            reference_output = self._eager(
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                key.cu_seqlens_values,
                key.feature_lens_values,
                key.aftercnn_lens_values,
                async_tensor_h2d,
            )
            if not _output_is_supported(reference_output, key):
                raise RuntimeError("Eager audio prefix did not return [M, 1024]")
            self._backend.wait_stream(replay_stream, padded_feature.device)
            with self._backend.replay_context(replay_stream):
                static_padded_feature.copy_(padded_feature, non_blocking=True)
                self._backend.replay(graph)
                self._backend.record_done(done_event, replay_stream)
            self._backend.wait_done(done_event, padded_feature.device)
            if not self._backend.equal(graph_output, reference_output):
                raise RuntimeError("Captured audio prefix is not bitwise exact")

            entry = _PrefixGraphEntry(
                key=key,
                static_padded_feature=static_padded_feature,
                static_pack_metadata=static_pack_metadata,
                graph=graph,
                output=graph_output,
                replay_stream=replay_stream,
                done_event=done_event,
                lock=threading.Lock(),
            )
            self._entries[key] = entry
            logger.info(
                "Captured bitwise-exact audio prefix CUDA graph for shape=%s "
                "rows=%d",
                key.padded_shape,
                key.packed_rows,
            )
            return entry, reference_output
        except Exception as error:
            self._rejected_keys.add(key)
            logger.warning(
                "Audio prefix CUDA graph capture failed closed for shape=%s "
                "rows=%d: %s",
                key.padded_shape,
                key.packed_rows,
                error,
            )
            return None, None


class _NoopPrefixBackend:
    """CPU fake backend used by tests and the standalone helper's dry paths."""

    def supports(self, padded_feature: torch.Tensor) -> bool:
        return True

    def is_current_stream_capturing(self) -> bool:
        return False

    def make_device_metadata(
        self,
        metadata_cpu: torch.Tensor,
        *,
        device: torch.device,
        async_tensor_h2d: Any,
    ) -> torch.Tensor:
        del async_tensor_h2d
        return metadata_cpu.to(device=device)

    def capture(
        self,
        function: Callable[[], torch.Tensor],
        *,
        device: torch.device,
        warmup_iterations: int,
    ) -> tuple[Any, torch.Tensor, Any, Any]:
        del device
        for _ in range(warmup_iterations):
            function()
        output = function()
        return {"function": function, "output": output}, output, object(), object()

    def replay_context(self, replay_stream: Any) -> Any:
        del replay_stream
        return nullcontext()

    def wait_stream(self, replay_stream: Any, device: torch.device) -> None:
        del replay_stream, device

    def record_done(self, done_event: Any, replay_stream: Any) -> None:
        del done_event, replay_stream

    def wait_done(self, done_event: Any, device: torch.device) -> None:
        del done_event, device

    def replay(self, graph: Any) -> None:
        graph["output"].copy_(graph["function"]())

    def replay_and_clone(
        self,
        entry: _PrefixGraphEntry,
        padded_feature: torch.Tensor,
    ) -> torch.Tensor:
        entry.static_padded_feature.copy_(padded_feature)
        self.replay(entry.graph)
        return entry.output.clone()

    def equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        return torch.equal(left, right)


def run_audio_prefix_cudagraph(
    encoder: Any,
    padded_feature: torch.Tensor,
    chunk_lengths_cpu: torch.Tensor,
    pack_metadata_cpu: torch.Tensor,
    cu_seqlens_values: tuple[int, ...],
    feature_lens_values: tuple[int, ...],
    aftercnn_lens_values: tuple[int, ...],
    *,
    async_tensor_h2d: Any,
) -> torch.Tensor:
    """Run an exact cached graph or the identical eager prefix."""
    cache = getattr(encoder, _CACHE_ATTR, None)
    if cache is None:
        with _CACHE_CREATION_LOCK:
            cache = getattr(encoder, _CACHE_ATTR, None)
            if cache is None:
                cache = ExactShapeAudioPrefixGraphCache()
                setattr(encoder, _CACHE_ATTR, cache)
    if not isinstance(cache, ExactShapeAudioPrefixGraphCache):
        return run_audio_prefix_eager(
            encoder,
            padded_feature,
            chunk_lengths_cpu,
            pack_metadata_cpu,
            cu_seqlens_values,
            feature_lens_values,
            aftercnn_lens_values,
            async_tensor_h2d=async_tensor_h2d,
        )
    return cache.run(
        encoder,
        padded_feature,
        chunk_lengths_cpu,
        pack_metadata_cpu,
        cu_seqlens_values,
        feature_lens_values,
        aftercnn_lens_values,
        async_tensor_h2d=async_tensor_h2d,
    )


def install_audio_prefix_cudagraph_patch() -> bool:
    """Install the prefix runner through the accepted CPU-metadata forward."""
    if not audio_prefix_cudagraph_enabled():
        return False
    if (
        os.environ.get(METADATA_ENV_NAME, "0") != "1"
        or os.environ.get(MAX_SEQLEN_ENV_NAME, "0") != "1"
    ):
        logger.warning(
            "%s requires %s=1 and %s=1; leaving the model unchanged",
            ENV_NAME,
            METADATA_ENV_NAME,
            MAX_SEQLEN_ENV_NAME,
        )
        return False

    from vllm.model_executor.models import qwen3_asr as asr_module
    from vllm.model_executor.models.qwen3_asr import Qwen3OmniMoeAudioEncoder

    current_forward = Qwen3OmniMoeAudioEncoder.forward
    current_field_config = asr_module._qwen3asr_field_config
    prefix_patched = bool(getattr(current_forward, _PATCH_MARKER, False))
    metadata_forward_patched = bool(
        getattr(current_forward, METADATA_PATCH_MARKER, False)
    )
    metadata_field_patched = bool(
        getattr(current_field_config, METADATA_PATCH_MARKER, False)
    )
    if prefix_patched:
        if metadata_forward_patched and metadata_field_patched:
            return True
        raise RuntimeError("Audio prefix CUDA graph patch is partially installed")
    if metadata_forward_patched or metadata_field_patched:
        logger.warning(
            "Audio CPU metadata patch was already installed without the prefix "
            "runner; refusing a partial replacement"
        )
        return False

    if not install_audio_cpu_metadata_pack_patch(
        prefix_runner=run_audio_prefix_cudagraph,
    ):
        return False
    setattr(Qwen3OmniMoeAudioEncoder.forward, _PATCH_MARKER, True)
    return True
