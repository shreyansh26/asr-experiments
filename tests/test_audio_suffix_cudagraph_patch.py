from __future__ import annotations

import os
from pathlib import Path
import sys
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
    def __init__(self, *, capture_error: Exception | None = None) -> None:
        self.capture_error = capture_error
        self.capture_calls = 0
        self.replay_calls = 0
        self.equal_calls = 0
        self.stream_id = 17

    def supports(self, hidden_states, cu_seqlens, max_seqlen) -> bool:
        del hidden_states, cu_seqlens, max_seqlen
        return True

    def is_current_stream_capturing(self) -> bool:
        return False

    def current_stream_id(self, device) -> int:
        del device
        return self.stream_id

    def capture(self, function, *, device, warmup_iterations):
        del device
        self.capture_calls += 1
        if self.capture_error is not None:
            raise self.capture_error
        for _ in range(warmup_iterations):
            function()
        output = function()
        return _FakeGraph(function, output), output, self.stream_id

    def replay(self, graph) -> None:
        self.replay_calls += 1
        graph.replay()

    def equal(self, left, right) -> bool:
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
