"""Exact-shape CUDA graph cache for the Qwen3-ASR post-pack audio suffix."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import torch

from audio_cpu_metadata_pack_patch import (
    ENV_NAME as METADATA_ENV_NAME,
    MAX_SEQLEN_ENV_NAME,
    _PATCH_MARKER as METADATA_PATCH_MARKER,
    install_audio_cpu_metadata_pack_patch,
    run_audio_suffix_eager,
)
from vllm.logger import init_logger


logger = init_logger("vllm.qwen3_asr_audio_suffix_cudagraph")

ENV_NAME = "ASR_AUDIO_SUFFIX_CUDAGRAPH"
_PATCH_MARKER = "_asr_audio_suffix_cudagraph_patch"
_CACHE_ATTR = "_asr_audio_suffix_cudagraph_cache"
_EXPECTED_LAYER_COUNT = 24
_INPUT_WIDTH = 1024
_OUTPUT_WIDTH = 2048
_EXPECTED_MAX_SEQLEN = 104
_EXPECTED_CU_SEQLENS_NUMEL = 4
_SUPPORTED_ROWS = frozenset({264, 265, 267, 268, 270, 272, 273})
_MAX_CACHE_ENTRIES = len(_SUPPORTED_ROWS)
_WARMUP_ITERATIONS = 3


@dataclass(frozen=True)
class SuffixGraphKey:
    """All launch-static state that can change suffix attention behavior."""

    rows: int
    cu_seqlens_numel: int
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    max_seqlen_value: int
    cu_seqlens_values: tuple[int, ...]


@dataclass
class _SuffixGraphEntry:
    key: SuffixGraphKey
    static_hidden_states: torch.Tensor
    static_cu_seqlens: torch.Tensor
    static_max_seqlen: torch.Tensor
    graph: Any
    output: torch.Tensor
    replay_stream_id: int


def audio_suffix_cudagraph_enabled() -> bool:
    """Return the strict boolean environment gate for this experiment."""
    raw_value = os.environ.get(ENV_NAME, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(f"{ENV_NAME} must be 0 or 1, got {raw_value!r}")
    return raw_value == "1"


def _make_suffix_graph_key(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor | None,
    cu_seqlens_values: tuple[int, ...],
) -> SuffixGraphKey | None:
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or hidden_states.shape[0] <= 0
        or hidden_states.shape[1] != _INPUT_WIDTH
        or not hidden_states.is_contiguous()
        or not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.ndim != 1
        or cu_seqlens.dtype != torch.int32
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.device != hidden_states.device
        or not isinstance(max_seqlen, torch.Tensor)
        or max_seqlen.ndim != 0
        or max_seqlen.dtype != torch.int32
        or max_seqlen.device.type != "cpu"
    ):
        return None

    values = tuple(int(value) for value in cu_seqlens_values)
    rows = hidden_states.shape[0]
    max_seqlen_value = int(max_seqlen.item())
    if (
        len(values) != cu_seqlens.numel()
        or len(values) != _EXPECTED_CU_SEQLENS_NUMEL
        or rows not in _SUPPORTED_ROWS
        or values[0] != 0
        or values[-1] != rows
        or any(left >= right for left, right in zip(values, values[1:]))
        or max_seqlen_value != _EXPECTED_MAX_SEQLEN
        or any(
            right - left > max_seqlen_value
            for left, right in zip(values, values[1:])
        )
    ):
        return None

    return SuffixGraphKey(
        rows=rows,
        cu_seqlens_numel=cu_seqlens.numel(),
        dtype=hidden_states.dtype,
        device_type=hidden_states.device.type,
        device_index=hidden_states.device.index,
        max_seqlen_value=max_seqlen_value,
        cu_seqlens_values=values,
    )


def _output_is_supported(output: Any, key: SuffixGraphKey) -> bool:
    return (
        isinstance(output, torch.Tensor)
        and output.shape == (key.rows, _OUTPUT_WIDTH)
        and output.dtype == key.dtype
        and output.device.type == key.device_type
        and output.device.index == key.device_index
    )


class _TorchCudaGraphBackend:
    """Small CUDA API seam so cache behavior has CUDA-hidden fake tests."""

    def supports(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> bool:
        return (
            torch.cuda.is_available()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and cu_seqlens.is_cuda
            and max_seqlen.device.type == "cpu"
            and max_seqlen.dtype == torch.int32
        )

    def is_current_stream_capturing(self) -> bool:
        return torch.cuda.is_current_stream_capturing()

    def current_stream_id(self, device: torch.device) -> int:
        return int(torch.cuda.current_stream(device).cuda_stream)

    def capture(
        self,
        function: Callable[[], torch.Tensor],
        *,
        device: torch.device,
        warmup_iterations: int,
    ) -> tuple[Any, torch.Tensor, int]:
        caller_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(caller_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(warmup_iterations):
                function()
        caller_stream.wait_stream(capture_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            output = function()
        return graph, output, int(caller_stream.cuda_stream)

    def replay(self, graph: Any) -> None:
        graph.replay()

    def equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        return torch.equal(left, right)


class ExactShapeAudioSuffixGraphCache:
    """Capture one verified suffix graph per exact shape and segmentation."""

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
        self._entries: dict[SuffixGraphKey, _SuffixGraphEntry] = {}
        self._rejected_keys: set[SuffixGraphKey] = set()
        self._logged_capacity = False
        self._logged_replay = False

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def rejected_count(self) -> int:
        return len(self._rejected_keys)

    def _eager(
        self,
        encoder: Any,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        cu_seqlens_values: tuple[int, ...],
    ) -> torch.Tensor:
        return run_audio_suffix_eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=cu_seqlens_values,
        )

    def run(
        self,
        encoder: Any,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
        *,
        cu_seqlens_values: tuple[int, ...],
    ) -> torch.Tensor:
        key = _make_suffix_graph_key(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values,
        )
        if (
            key is None
            or max_seqlen is None
            or len(getattr(encoder, "layers", ())) != _EXPECTED_LAYER_COUNT
            or bool(getattr(encoder, "training", True))
            or torch.is_grad_enabled()
            or not self._backend.supports(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )
            or self._backend.is_current_stream_capturing()
        ):
            return self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                tuple(cu_seqlens_values),
            )

        if key in self._rejected_keys:
            return self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                key.cu_seqlens_values,
            )

        caller_stream_id = self._backend.current_stream_id(hidden_states.device)
        entry = self._entries.get(key)
        if entry is not None:
            if entry.replay_stream_id != caller_stream_id:
                return self._eager(
                    encoder,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    key.cu_seqlens_values,
                )
            entry.static_hidden_states.copy_(hidden_states, non_blocking=True)
            self._backend.replay(entry.graph)
            if not self._logged_replay:
                logger.info(
                    "ASR audio post-pack suffix CUDA graph replay active "
                    "for M=%d and %d sequences",
                    key.rows,
                    key.cu_seqlens_numel - 1,
                )
                self._logged_replay = True
            # The graph-owned output is overwritten by the next exact-key
            # replay. Return independent storage so downstream encoder-cache
            # views cannot observe a later request's result.
            return entry.output.clone()

        if len(self._entries) >= self._max_entries:
            if not self._logged_capacity:
                logger.warning(
                    "Audio suffix CUDA graph cache is full at %d exact keys; "
                    "using eager suffix for M=%d",
                    self._max_entries,
                    key.rows,
                )
                self._logged_capacity = True
            return self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                key.cu_seqlens_values,
            )

        reference_output: torch.Tensor | None = None
        try:
            static_hidden_states = torch.empty_like(hidden_states)
            static_cu_seqlens = torch.empty_like(cu_seqlens)
            static_max_seqlen = torch.empty_like(max_seqlen)
            static_hidden_states.copy_(hidden_states, non_blocking=True)
            static_cu_seqlens.copy_(cu_seqlens, non_blocking=True)
            static_max_seqlen.copy_(max_seqlen, non_blocking=True)

            def static_suffix() -> torch.Tensor:
                return self._eager(
                    encoder,
                    static_hidden_states,
                    static_cu_seqlens,
                    static_max_seqlen,
                    key.cu_seqlens_values,
                )

            graph, graph_output, replay_stream_id = self._backend.capture(
                static_suffix,
                device=hidden_states.device,
                warmup_iterations=self._warmup_iterations,
            )
            if replay_stream_id != caller_stream_id:
                raise RuntimeError("CUDA graph capture changed the caller stream")
            if not _output_is_supported(graph_output, key):
                raise RuntimeError(
                    "Captured audio suffix did not return [M, 2048]"
                )

            # Gate the real captured module stack before admitting this key.
            reference_output = self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                key.cu_seqlens_values,
            )
            if not _output_is_supported(reference_output, key):
                raise RuntimeError("Eager audio suffix did not return [M, 2048]")
            static_hidden_states.copy_(hidden_states, non_blocking=True)
            self._backend.replay(graph)
            # This comparison synchronizes once during key admission. There is
            # no equality check or host synchronization on admitted replays.
            if not self._backend.equal(graph_output, reference_output):
                raise RuntimeError("Captured audio suffix is not bitwise exact")

            self._entries[key] = _SuffixGraphEntry(
                key=key,
                static_hidden_states=static_hidden_states,
                static_cu_seqlens=static_cu_seqlens,
                static_max_seqlen=static_max_seqlen,
                graph=graph,
                output=graph_output,
                replay_stream_id=replay_stream_id,
            )
            logger.info(
                "Captured bitwise-exact audio suffix CUDA graph for M=%d and "
                "%d sequences",
                key.rows,
                key.cu_seqlens_numel - 1,
            )
            # Preserve eager output ownership on the capture call. Replays use
            # the stable graph-owned output from the next exact-key hit onward.
            return reference_output
        except Exception as error:
            self._rejected_keys.add(key)
            logger.warning(
                "Audio suffix CUDA graph capture failed closed for M=%d and "
                "%d sequences: %s",
                key.rows,
                key.cu_seqlens_numel - 1,
                error,
            )
            if reference_output is not None:
                return reference_output
            return self._eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                key.cu_seqlens_values,
            )


def run_audio_suffix_cudagraph(
    encoder: Any,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor | None,
    *,
    cu_seqlens_values: tuple[int, ...],
) -> torch.Tensor:
    """Run an exact cached graph or the identical eager suffix."""
    cache = getattr(encoder, _CACHE_ATTR, None)
    if cache is None:
        cache = ExactShapeAudioSuffixGraphCache()
        setattr(encoder, _CACHE_ATTR, cache)
    if not isinstance(cache, ExactShapeAudioSuffixGraphCache):
        return run_audio_suffix_eager(
            encoder,
            hidden_states,
            cu_seqlens,
            max_seqlen,
            cu_seqlens_values=cu_seqlens_values,
        )
    return cache.run(
        encoder,
        hidden_states,
        cu_seqlens,
        max_seqlen,
        cu_seqlens_values=cu_seqlens_values,
    )


def install_audio_suffix_cudagraph_patch() -> bool:
    """Install the suffix runner through the accepted CPU-metadata forward."""
    if not audio_suffix_cudagraph_enabled():
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
    suffix_patched = bool(getattr(current_forward, _PATCH_MARKER, False))
    metadata_forward_patched = bool(
        getattr(current_forward, METADATA_PATCH_MARKER, False)
    )
    metadata_field_patched = bool(
        getattr(current_field_config, METADATA_PATCH_MARKER, False)
    )
    if suffix_patched:
        if metadata_forward_patched and metadata_field_patched:
            return True
        raise RuntimeError("Audio suffix CUDA graph patch is partially installed")
    if metadata_forward_patched or metadata_field_patched:
        logger.warning(
            "Audio CPU metadata patch was already installed without the suffix "
            "runner; refusing a partial replacement"
        )
        return False

    if not install_audio_cpu_metadata_pack_patch(
        suffix_runner=run_audio_suffix_cudagraph
    ):
        return False
    setattr(Qwen3OmniMoeAudioEncoder.forward, _PATCH_MARKER, True)
    return True
