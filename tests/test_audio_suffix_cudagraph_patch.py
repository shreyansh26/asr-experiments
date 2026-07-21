from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn


PATCH_DIR = Path(__file__).resolve().parents[1] / "inference" / "vllm_static_fp8"
sys.path.insert(0, str(PATCH_DIR))

import audio_suffix_cudagraph_patch as suffix_patch  # noqa: E402
from audio_cpu_metadata_pack_patch import run_audio_suffix_eager  # noqa: E402


class _FakeLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None,
    ) -> torch.Tensor:
        if max_seqlen is None:
            raise AssertionError("fake suffix requires max_seqlen")
        segment_value = cu_seqlens[1].to(hidden_states.dtype) / 1024
        max_value = max_seqlen.to(hidden_states.dtype) / 2048
        return hidden_states + segment_value + max_value


class _DoubleWidth(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.cat((hidden_states, hidden_states), dim=-1)


class _FakeEncoder(nn.Module):
    def __init__(self, layer_count: int = 24) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(layer_count)])
        self.ln_post = nn.Identity()
        self.proj1 = nn.Identity()
        self.act = nn.Identity()
        self.proj2 = _DoubleWidth()
        self.eval()


class _FakeGraph:
    def __init__(self, function, output: torch.Tensor) -> None:
        self.function = function
        self.output = output

    def replay(self) -> None:
        self.output.copy_(self.function())


class _FakeBackend:
    def __init__(
        self,
        *,
        capture_error: Exception | None = None,
        capture_delay: float = 0.0,
        transaction_delay: float = 0.0,
    ) -> None:
        self.capture_error = capture_error
        self.capture_delay = capture_delay
        self.transaction_delay = transaction_delay
        self.capture_calls = 0
        self.replay_calls = 0
        self.equal_calls = 0
        self.active_transactions = 0
        self.max_active_transactions = 0
        self.execution_stream = object()
        self._state_lock = threading.Lock()

    def supports(self, hidden_states, cu_seqlens, max_seqlen) -> bool:
        del hidden_states, cu_seqlens, max_seqlen
        return True

    def is_current_stream_capturing(self) -> bool:
        return False

    def capture(self, function, *, device, warmup_iterations):
        del device
        with self._state_lock:
            self.capture_calls += 1
        if self.capture_delay:
            time.sleep(self.capture_delay)
        if self.capture_error is not None:
            raise self.capture_error
        for _ in range(warmup_iterations):
            function()
        output = function()
        return _FakeGraph(function, output), output, self.execution_stream

    def replay(self, graph) -> None:
        with self._state_lock:
            self.replay_calls += 1
        graph.replay()

    def replay_and_clone(self, entry, hidden_states) -> torch.Tensor:
        with self._state_lock:
            self.active_transactions += 1
            self.max_active_transactions = max(
                self.max_active_transactions,
                self.active_transactions,
            )
            self.replay_calls += 1
        try:
            entry.static_hidden_states.copy_(hidden_states)
            if self.transaction_delay:
                time.sleep(self.transaction_delay)
            entry.graph.replay()
            return entry.output.clone()
        finally:
            with self._state_lock:
                self.active_transactions -= 1

    def equal(self, left, right) -> bool:
        with self._state_lock:
            self.equal_calls += 1
        return torch.equal(left, right)


def _inputs(
    values: tuple[int, ...],
    *,
    fill: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = values[-1]
    hidden_states = torch.full(
        (rows, 1024),
        fill,
        dtype=torch.bfloat16,
    )
    cu_seqlens = torch.tensor(values, dtype=torch.int32)
    max_seqlen = torch.tensor(
        104,
        dtype=torch.int32,
    )
    return hidden_states, cu_seqlens, max_seqlen


class AudioSuffixCudagraphPatchTest(unittest.TestCase):
    def test_environment_gate_is_strict(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(suffix_patch.audio_suffix_cudagraph_enabled())
        with patch.dict(
            os.environ,
            {suffix_patch.ENV_NAME: "1"},
            clear=True,
        ):
            self.assertTrue(suffix_patch.audio_suffix_cudagraph_enabled())
        for value in ("", "true", "2", "-1"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {suffix_patch.ENV_NAME: value},
                    clear=True,
                ):
                    with self.assertRaises(ValueError):
                        suffix_patch.audio_suffix_cudagraph_enabled()

    def test_exact_key_includes_full_cu_seqlens_contents(self) -> None:
        hidden_states, cu_seqlens, max_seqlen = _inputs((0, 88, 176, 264))
        first = suffix_patch._make_suffix_graph_key(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            (0, 88, 176, 264),
        )
        second = suffix_patch._make_suffix_graph_key(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            (0, 80, 176, 264),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.rows, 264)
        self.assertEqual(first.cu_seqlens_numel, 4)
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertEqual(first.max_seqlen_value, 104)
        self.assertNotEqual(first, second)

    def test_malformed_cpu_metadata_fails_key_closed(self) -> None:
        hidden_states, cu_seqlens, max_seqlen = _inputs((0, 88, 176, 264))
        for values in (
            (1, 88, 176, 264),
            (0, 88, 176, 263),
            (0, 88, 88, 264),
            (0, 264),
        ):
            with self.subTest(values=values):
                key = suffix_patch._make_suffix_graph_key(
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    values,
                )
                self.assertIsNone(key)

        invalid_max = max_seqlen.clone().fill_(103)
        self.assertIsNone(
            suffix_patch._make_suffix_graph_key(
                hidden_states,
                cu_seqlens,
                invalid_max,
                (0, 88, 176, 264),
            )
        )

    def test_eager_suffix_runs_all_layers_and_preserves_rows(self) -> None:
        encoder = _FakeEncoder()
        hidden_states, cu_seqlens, max_seqlen = _inputs((0, 88, 176, 264))

        with torch.inference_mode():
            output = run_audio_suffix_eager(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=(0, 88, 176, 264),
            )

        self.assertEqual(output.shape, (264, 2048))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_fake_capture_replays_stable_buffers_bitwise(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = suffix_patch.ExactShapeAudioSuffixGraphCache(
            backend=backend,
            max_entries=4,
            warmup_iterations=2,
        )
        hidden_states, cu_seqlens, max_seqlen = _inputs((0, 88, 176, 264))

        with torch.inference_mode():
            first = cache.run(
                encoder,
                hidden_states,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            replay_input = torch.full_like(hidden_states, 0.75)
            expected = run_audio_suffix_eager(
                encoder,
                replay_input,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            replayed = cache.run(
                encoder,
                replay_input,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            replay_pointer = replayed.data_ptr()
            replay_input_again = replay_input + 0.25
            expected_again = run_audio_suffix_eager(
                encoder,
                replay_input_again,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            replayed_again = cache.run(
                encoder,
                replay_input_again,
                cu_seqlens,
                max_seqlen,
                cu_seqlens_values=(0, 88, 176, 264),
            )

        self.assertEqual(first.shape, (264, 2048))
        self.assertFalse(torch.equal(expected, expected_again))
        self.assertTrue(torch.equal(expected, replayed))
        self.assertTrue(torch.equal(expected_again, replayed_again))
        self.assertNotEqual(replay_pointer, replayed_again.data_ptr())
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(cache.rejected_count, 0)
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(backend.replay_calls, 3)
        self.assertEqual(backend.equal_calls, 1)

    def test_same_shape_different_segments_get_distinct_graphs(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = suffix_patch.ExactShapeAudioSuffixGraphCache(
            backend=backend,
            max_entries=4,
            warmup_iterations=1,
        )
        first_inputs = _inputs((0, 88, 176, 264))
        second_inputs = _inputs((0, 80, 176, 264))

        with torch.inference_mode():
            first = cache.run(
                encoder,
                *first_inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            second = cache.run(
                encoder,
                *second_inputs,
                cu_seqlens_values=(0, 80, 176, 264),
            )

        self.assertFalse(torch.equal(first, second))
        self.assertEqual(cache.entry_count, 2)
        self.assertEqual(backend.capture_calls, 2)

    def test_sequential_warmup_shape_cannot_starve_batched_key(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = suffix_patch.ExactShapeAudioSuffixGraphCache(
            backend=backend,
            max_entries=1,
            warmup_iterations=1,
        )
        sequential_inputs = _inputs((0, 264))
        batched_inputs = _inputs((0, 88, 176, 264))

        with torch.inference_mode():
            sequential_output = cache.run(
                encoder,
                *sequential_inputs,
                cu_seqlens_values=(0, 264),
            )
            self.assertEqual(cache.entry_count, 0)
            batched_output = cache.run(
                encoder,
                *batched_inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )

        self.assertEqual(sequential_output.shape, (264, 2048))
        self.assertEqual(batched_output.shape, (264, 2048))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(backend.capture_calls, 1)

    def test_concurrent_first_wrapper_call_attaches_one_cache(self) -> None:
        encoder = _FakeEncoder()
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
                hidden_states,
                cu_seqlens,
                max_seqlen,
                *,
                cu_seqlens_values,
            ) -> torch.Tensor:
                del encoder, cu_seqlens, max_seqlen, cu_seqlens_values
                with self.run_lock:
                    self.run_calls += 1
                return hidden_states.clone()

        inputs_by_thread = [
            _inputs((0, 88, 176, 264), fill=0.5),
            _inputs((0, 88, 176, 264), fill=0.75),
        ]
        barrier = threading.Barrier(2)

        def invoke(inputs) -> torch.Tensor:
            barrier.wait()
            return suffix_patch.run_audio_suffix_cudagraph(
                encoder,
                *inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )

        with patch.object(
            suffix_patch,
            "ExactShapeAudioSuffixGraphCache",
            _SlowCache,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(invoke, inputs)
                    for inputs in inputs_by_thread
                ]
                outputs = [future.result() for future in futures]

        attached_cache = getattr(encoder, suffix_patch._CACHE_ATTR)
        self.assertEqual(creation_count, 1)
        self.assertEqual(attached_cache.run_calls, 2)
        for inputs, output in zip(inputs_by_thread, outputs, strict=True):
            self.assertTrue(torch.equal(inputs[0], output))

    def test_concurrent_cold_same_key_captures_once(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend(capture_delay=0.02)
        cache = suffix_patch.ExactShapeAudioSuffixGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        first_inputs = _inputs((0, 88, 176, 264), fill=0.5)
        second_inputs = _inputs((0, 88, 176, 264), fill=0.75)
        barrier = threading.Barrier(2)

        def invoke(inputs) -> torch.Tensor:
            with torch.inference_mode():
                barrier.wait()
                return cache.run(
                    encoder,
                    *inputs,
                    cu_seqlens_values=(0, 88, 176, 264),
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(invoke, first_inputs)
            second_future = executor.submit(invoke, second_inputs)
            first_output = first_future.result()
            second_output = second_future.result()

        with torch.inference_mode():
            first_expected = run_audio_suffix_eager(
                encoder,
                *first_inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            second_expected = run_audio_suffix_eager(
                encoder,
                *second_inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )
        self.assertTrue(torch.equal(first_output, first_expected))
        self.assertTrue(torch.equal(second_output, second_expected))
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(cache.entry_count, 1)

    def test_concurrent_hot_same_key_transactions_are_serialized(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend(transaction_delay=0.01)
        cache = suffix_patch.ExactShapeAudioSuffixGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        initial_inputs = _inputs((0, 88, 176, 264), fill=0.25)
        with torch.inference_mode():
            cache.run(
                encoder,
                *initial_inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )

        inputs_by_thread = [
            [
                _inputs((0, 88, 176, 264), fill=fill)
                for fill in (0.5, 0.625, 0.75)
            ],
            [
                _inputs((0, 88, 176, 264), fill=fill)
                for fill in (1.0, 1.125, 1.25)
            ],
        ]
        barrier = threading.Barrier(2)

        def replay_many(inputs_list) -> list[torch.Tensor]:
            outputs = []
            with torch.inference_mode():
                for inputs in inputs_list:
                    barrier.wait()
                    outputs.append(
                        cache.run(
                            encoder,
                            *inputs,
                            cu_seqlens_values=(0, 88, 176, 264),
                        )
                    )
            return outputs

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(replay_many, inputs_list)
                for inputs_list in inputs_by_thread
            ]
            outputs_by_thread = [future.result() for future in futures]

        with torch.inference_mode():
            for inputs_list, outputs in zip(
                inputs_by_thread,
                outputs_by_thread,
                strict=True,
            ):
                for inputs, output in zip(inputs_list, outputs, strict=True):
                    expected = run_audio_suffix_eager(
                        encoder,
                        *inputs,
                        cu_seqlens_values=(0, 88, 176, 264),
                    )
                    self.assertTrue(torch.equal(output, expected))

        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(backend.max_active_transactions, 1)

    def test_capture_error_is_rejected_once_then_eager(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend(capture_error=RuntimeError("capture rejected"))
        cache = suffix_patch.ExactShapeAudioSuffixGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        inputs = _inputs((0, 88, 176, 264))

        with torch.inference_mode():
            expected = run_audio_suffix_eager(
                encoder,
                *inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            first = cache.run(
                encoder,
                *inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            second = cache.run(
                encoder,
                *inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )

        self.assertTrue(torch.equal(first, expected))
        self.assertTrue(torch.equal(second, expected))
        self.assertEqual(backend.capture_calls, 1)
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(cache.rejected_count, 1)

    def test_full_cache_falls_back_without_evicting_stable_graph(self) -> None:
        encoder = _FakeEncoder()
        backend = _FakeBackend()
        cache = suffix_patch.ExactShapeAudioSuffixGraphCache(
            backend=backend,
            max_entries=1,
            warmup_iterations=1,
        )
        first_inputs = _inputs((0, 88, 176, 264))
        second_inputs = _inputs((0, 80, 176, 264))

        with torch.inference_mode():
            cache.run(
                encoder,
                *first_inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )
            expected = run_audio_suffix_eager(
                encoder,
                *second_inputs,
                cu_seqlens_values=(0, 80, 176, 264),
            )
            actual = cache.run(
                encoder,
                *second_inputs,
                cu_seqlens_values=(0, 80, 176, 264),
            )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(cache.entry_count, 1)
        self.assertEqual(backend.capture_calls, 1)

    def test_wrong_layer_count_uses_eager_without_capture(self) -> None:
        encoder = _FakeEncoder(layer_count=23)
        backend = _FakeBackend()
        cache = suffix_patch.ExactShapeAudioSuffixGraphCache(
            backend=backend,
            max_entries=2,
            warmup_iterations=1,
        )
        inputs = _inputs((0, 88, 176, 264))

        with torch.inference_mode():
            output = cache.run(
                encoder,
                *inputs,
                cu_seqlens_values=(0, 88, 176, 264),
            )

        self.assertEqual(output.shape, (264, 2048))
        self.assertEqual(backend.capture_calls, 0)
        self.assertEqual(cache.entry_count, 0)

    def test_missing_metadata_prerequisites_leave_model_unimported(self) -> None:
        with patch.dict(
            os.environ,
            {suffix_patch.ENV_NAME: "1"},
            clear=True,
        ):
            self.assertFalse(suffix_patch.install_audio_suffix_cudagraph_patch())


if __name__ == "__main__":
    unittest.main()
