from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time
import unittest
from unittest.mock import patch

import torch


PATCH_DIR = Path(__file__).resolve().parents[1] / "inference" / "vllm_static_fp8"
sys.path.insert(0, str(PATCH_DIR))

import audio_prefix_cudagraph_patch as prefix_patch  # noqa: E402
from audio_cpu_metadata_pack_patch import run_audio_prefix_eager  # noqa: E402
from audio_prefix_cudagraph_patch import (  # noqa: E402
    ENV_NAME,
    ExactShapeAudioPrefixGraphCache,
    _NoopPrefixBackend,
    _make_prefix_graph_key,
    audio_prefix_cudagraph_enabled,
)


class _FakeConv:
    def __init__(self, shape: tuple[int, int, int], scale: float) -> None:
        self.shape = shape
        self.scale = scale

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        base = inputs.float().mean(dim=tuple(range(1, inputs.ndim)), keepdim=True)
        values = torch.arange(
            self.shape[0] * self.shape[1] * self.shape[2],
            device=inputs.device,
            dtype=torch.float32,
        ).reshape(1, *self.shape)
        output = base.reshape(batch, 1, 1, 1) + values * self.scale
        return output.to(inputs.dtype)


class _FakeLinear:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        base = inputs.float().mean(dim=-1, keepdim=True)
        columns = torch.arange(
            1024,
            device=inputs.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1024)
        return (base + columns / 1024.0).to(inputs.dtype)


class _FailCaptureBackend(_NoopPrefixBackend):
    def __init__(self) -> None:
        self.capture_calls = 0

    def capture(self, *args, **kwargs):
        self.capture_calls += 1
        raise RuntimeError("forced capture failure")


class _RejectAdmissionBackend(_NoopPrefixBackend):
    def equal(self, left: torch.Tensor, right: torch.Tensor) -> bool:
        del left, right
        return False


class _CountingPrefixBackend(_NoopPrefixBackend):
    def __init__(self) -> None:
        self.capture_calls = 0
        self.replay_calls = 0

    def capture(self, *args, **kwargs):
        self.capture_calls += 1
        return super().capture(*args, **kwargs)

    def replay(self, graph) -> None:
        self.replay_calls += 1
        super().replay(graph)


class _TrackingReplayBackend(_NoopPrefixBackend):
    def __init__(self) -> None:
        self.active_transactions = 0
        self.max_active_transactions = 0
        self._state_lock = threading.Lock()

    def replay_and_clone(self, entry, padded_feature):
        with self._state_lock:
            self.active_transactions += 1
            self.max_active_transactions = max(
                self.max_active_transactions,
                self.active_transactions,
            )
        try:
            time.sleep(0.01)
            return super().replay_and_clone(entry, padded_feature)
        finally:
            with self._state_lock:
                self.active_transactions -= 1


def _encoder() -> SimpleNamespace:
    return SimpleNamespace(
        training=False,
        conv_chunksize=1024,
        conv2d1=_FakeConv((1, 64, 50), 0.0001),
        conv2d2=_FakeConv((1, 32, 25), 0.0002),
        conv2d3=_FakeConv((1, 16, 13), 0.0003),
        conv_out=_FakeLinear(),
        positional_embedding=SimpleNamespace(
            positional_embedding=torch.arange(
                13 * 1024,
                dtype=torch.bfloat16,
            ).reshape(13, 1024)
            / 8192
        ),
    )


def _metadata(total_rows: int = 272):
    chunks = 21
    lengths = [13] * chunks
    lengths[-1] = total_rows - sum(lengths[:-1])
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


def _padded(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        (21, 1, 128, 100),
        generator=generator,
        dtype=torch.bfloat16,
    )


class AudioPrefixCudaGraphPatchTest(unittest.TestCase):
    def test_environment_gate_is_strict(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(audio_prefix_cudagraph_enabled())
        with patch.dict(os.environ, {ENV_NAME: "1"}, clear=True):
            self.assertTrue(audio_prefix_cudagraph_enabled())
        for value in ("", "true", "2", "-1"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {ENV_NAME: value}, clear=True):
                    with self.assertRaises(ValueError):
                        audio_prefix_cudagraph_enabled()

    def test_key_includes_full_cpu_metadata_and_padded_shape(self) -> None:
        chunk_lengths, pack_metadata, cu_seqlens, feature_lens, aftercnn_lens = (
            _metadata()
        )
        key = _make_prefix_graph_key(
            _padded(),
            chunk_lengths,
            pack_metadata,
            cu_seqlens,
            feature_lens,
            aftercnn_lens,
        )
        self.assertIsNotNone(key)
        assert key is not None
        self.assertEqual(key.padded_shape, (21, 1, 128, 100))
        self.assertEqual(key.packed_rows, 272)
        self.assertEqual(key.chunk_lengths_values, tuple([100] * 21))
        self.assertEqual(key.pack_metadata_values, tuple(pack_metadata.tolist()))
        self.assertEqual(key.cu_seqlens_values, cu_seqlens)
        self.assertEqual(key.feature_lens_values, feature_lens)
        self.assertEqual(key.aftercnn_lens_values, aftercnn_lens)

        altered_feature_lens = (801, 800, 499)
        altered_key = _make_prefix_graph_key(
            _padded(),
            chunk_lengths,
            pack_metadata,
            cu_seqlens,
            altered_feature_lens,
            aftercnn_lens,
        )
        self.assertNotEqual(key, altered_key)

    def test_unsupported_shape_is_not_admitted(self) -> None:
        chunk_lengths, pack_metadata, cu_seqlens, feature_lens, _ = _metadata(266)
        self.assertIsNone(
            _make_prefix_graph_key(
                _padded(),
                chunk_lengths,
                pack_metadata,
                cu_seqlens,
                feature_lens,
                (104, 104, 58),
            )
        )

    def test_fake_graph_replay_is_exact_for_same_shape_different_content(self) -> None:
        encoder = _encoder()
        cache = ExactShapeAudioPrefixGraphCache(
            backend=_NoopPrefixBackend(),
            warmup_iterations=1,
        )
        metadata = _metadata()
        first = _padded(1)
        second = _padded(2)

        with torch.inference_mode():
            for _ in range(8):
                output_first = cache.run(
                    encoder,
                    first,
                    *metadata,
                    async_tensor_h2d=None,
                )
            output_second = cache.run(
                encoder,
                second,
                *metadata,
                async_tensor_h2d=None,
            )
            reference_second = run_audio_prefix_eager(
                encoder,
                second,
                *metadata,
                async_tensor_h2d=None,
            )

        self.assertEqual(cache.entry_count, 1)
        self.assertTrue(torch.equal(output_second, reference_second))
        self.assertFalse(torch.equal(output_first, output_second))
        self.assertNotEqual(output_second.data_ptr(), reference_second.data_ptr())

    def test_probation_captures_on_eighth_observation_then_replays(self) -> None:
        encoder = _encoder()
        backend = _CountingPrefixBackend()
        cache = ExactShapeAudioPrefixGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        metadata = _metadata()
        padded = _padded(9)

        with torch.inference_mode():
            for _ in range(7):
                output = cache.run(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=None,
                )
                self.assertEqual(cache.entry_count, 0)
                self.assertEqual(backend.capture_calls, 0)
            eighth = cache.run(
                encoder,
                padded,
                *metadata,
                async_tensor_h2d=None,
            )
            replay_input = _padded(10)
            ninth = cache.run(
                encoder,
                replay_input,
                *metadata,
                async_tensor_h2d=None,
            )
            expected_eighth = run_audio_prefix_eager(
                encoder,
                padded,
                *metadata,
                async_tensor_h2d=None,
            )
            expected_ninth = run_audio_prefix_eager(
                encoder,
                replay_input,
                *metadata,
                async_tensor_h2d=None,
            )

        self.assertTrue(torch.equal(output, expected_eighth))
        self.assertTrue(torch.equal(eighth, expected_eighth))
        self.assertTrue(torch.equal(ninth, expected_ninth))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.probation_key_count, 0)
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(backend.replay_calls, 2)

    def test_probation_state_is_bounded_for_one_off_keys(self) -> None:
        encoder = _encoder()
        backend = _CountingPrefixBackend()
        cache = ExactShapeAudioPrefixGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )

        with torch.inference_mode():
            for rows in (264, 265, 267):
                cache.run(
                    encoder,
                    _padded(rows),
                    *_metadata(rows),
                    async_tensor_h2d=None,
                )

        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.probation_key_count, 2)
        self.assertEqual(backend.capture_calls, 0)

    def test_cuda_backend_records_cross_stream_tensor_lifetimes(self) -> None:
        calls = []

        class _FakeStream:
            def __init__(self, name: str) -> None:
                self.name = name

            def wait_stream(self, other) -> None:
                calls.append((self.name, "wait_stream", other.name))

            def wait_event(self, event) -> None:
                calls.append((self.name, "wait_event", event.name))

        class _FakeStreamContext:
            def __init__(self, stream) -> None:
                self.stream = stream

            def __enter__(self):
                calls.append((self.stream.name, "enter"))

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                calls.append((self.stream.name, "exit"))

        class _FakeTensor:
            def __init__(self, name: str) -> None:
                self.name = name
                self.device = torch.device("cuda:0")

            def copy_(self, other, *, non_blocking=False):
                calls.append(
                    (self.name, "copy", other.name, non_blocking)
                )

            def clone(self):
                calls.append((self.name, "clone"))
                return clone

            def record_stream(self, stream) -> None:
                calls.append((self.name, "record_stream", stream.name))

        class _FakeGraph:
            def replay(self) -> None:
                calls.append(("graph", "replay"))

        class _FakeEvent:
            name = "done"

            def record(self, stream) -> None:
                calls.append((self.name, "record", stream.name))

        caller_stream = _FakeStream("caller")
        replay_stream = _FakeStream("replay")
        padded_feature = _FakeTensor("input")
        static_padded_feature = _FakeTensor("static")
        graph_output = _FakeTensor("graph_output")
        clone = _FakeTensor("clone")
        entry = SimpleNamespace(
            replay_stream=replay_stream,
            static_padded_feature=static_padded_feature,
            graph=_FakeGraph(),
            output=graph_output,
            done_event=_FakeEvent(),
        )

        backend = prefix_patch._TorchCudaGraphBackend()
        with (
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=caller_stream,
            ),
            patch.object(
                torch.cuda,
                "stream",
                side_effect=_FakeStreamContext,
            ),
        ):
            actual = backend.replay_and_clone(entry, padded_feature)

        self.assertIs(actual, clone)
        self.assertEqual(
            calls,
            [
                ("replay", "wait_stream", "caller"),
                ("replay", "enter"),
                ("static", "copy", "input", True),
                ("input", "record_stream", "replay"),
                ("graph", "replay"),
                ("graph_output", "clone"),
                ("done", "record", "replay"),
                ("replay", "exit"),
                ("caller", "wait_event", "done"),
                ("clone", "record_stream", "caller"),
            ],
        )

    def test_concurrent_first_wrapper_call_attaches_one_cache(self) -> None:
        encoder = _encoder()
        creation_count = 0
        creation_lock = threading.Lock()

        class _SlowCache:
            def __init__(self) -> None:
                nonlocal creation_count
                time.sleep(0.02)
                with creation_lock:
                    creation_count += 1
                self.run_calls = 0
                self.run_lock = threading.Lock()

            def run(
                self,
                encoder,
                padded_feature,
                chunk_lengths_cpu,
                pack_metadata_cpu,
                cu_seqlens_values,
                feature_lens_values,
                aftercnn_lens_values,
                *,
                async_tensor_h2d,
            ):
                del (
                    encoder,
                    chunk_lengths_cpu,
                    pack_metadata_cpu,
                    cu_seqlens_values,
                    feature_lens_values,
                    aftercnn_lens_values,
                    async_tensor_h2d,
                )
                with self.run_lock:
                    self.run_calls += 1
                return padded_feature.clone()

        inputs = [_padded(7), _padded(8)]
        metadata = _metadata()
        barrier = threading.Barrier(2)

        def invoke(padded_feature) -> torch.Tensor:
            barrier.wait()
            return prefix_patch.run_audio_prefix_cudagraph(
                encoder,
                padded_feature,
                *metadata,
                async_tensor_h2d=None,
            )

        with patch.object(
            prefix_patch,
            "ExactShapeAudioPrefixGraphCache",
            _SlowCache,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = list(executor.map(invoke, inputs))

        attached_cache = getattr(encoder, prefix_patch._CACHE_ATTR)
        self.assertEqual(creation_count, 1)
        self.assertEqual(attached_cache.run_calls, 2)
        for padded_feature, output in zip(inputs, outputs, strict=True):
            self.assertTrue(torch.equal(padded_feature, output))

    def test_capture_failure_falls_closed_and_marks_key_rejected(self) -> None:
        encoder = _encoder()
        backend = _FailCaptureBackend()
        cache = ExactShapeAudioPrefixGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        metadata = _metadata()
        padded = _padded(3)

        with torch.inference_mode():
            for _ in range(8):
                output = cache.run(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=None,
                )
            reference = run_audio_prefix_eager(
                encoder,
                padded,
                *metadata,
                async_tensor_h2d=None,
            )
            cache.run(
                encoder,
                padded,
                *metadata,
                async_tensor_h2d=None,
            )

        self.assertTrue(torch.equal(output, reference))
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.rejected_count, 1)
        self.assertEqual(backend.capture_calls, 1)

    def test_bitwise_admission_failure_falls_closed(self) -> None:
        encoder = _encoder()
        cache = ExactShapeAudioPrefixGraphCache(
            backend=_RejectAdmissionBackend(),
            warmup_iterations=1,
        )

        with torch.inference_mode():
            for _ in range(8):
                output = cache.run(
                    encoder,
                    _padded(4),
                    *_metadata(),
                    async_tensor_h2d=None,
                )

        self.assertEqual(output.shape, (272, 1024))
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.rejected_count, 1)

    def test_two_thread_same_key_replay_is_serialized_and_exact(self) -> None:
        encoder = _encoder()
        backend = _TrackingReplayBackend()
        cache = ExactShapeAudioPrefixGraphCache(
            backend=backend,
            warmup_iterations=1,
        )
        metadata = _metadata()
        with torch.inference_mode():
            for _ in range(8):
                cache.run(
                    encoder,
                    _padded(5),
                    *metadata,
                    async_tensor_h2d=None,
                )
        barrier = threading.Barrier(2)

        def run(seed: int) -> torch.Tensor:
            padded = _padded(seed)
            with torch.inference_mode():
                barrier.wait()
                output = cache.run(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=None,
                )
                reference = run_audio_prefix_eager(
                    encoder,
                    padded,
                    *metadata,
                    async_tensor_h2d=None,
                )
            self.assertTrue(torch.equal(output, reference))
            return output

        with ThreadPoolExecutor(max_workers=2) as executor:
            left, right = list(executor.map(run, (6, 7)))

        self.assertEqual(cache.entry_count, 1)
        self.assertFalse(torch.equal(left, right))
        self.assertEqual(backend.max_active_transactions, 1)


if __name__ == "__main__":
    unittest.main()
