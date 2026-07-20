from __future__ import annotations

import unittest

from inference.analyse_results import PRECISIONS, precision_for_row


class AnalyseResultsPrecisionTest(unittest.TestCase):
    def test_cpu_metadata_output_root_has_its_own_precision(self) -> None:
        row = {
            "output_root": (
                "predictions/results_fp8_static_qk_audio_cpu_metadata/"
                "batched_predicted"
            )
        }
        self.assertEqual(
            precision_for_row(row),
            "fp8_static_qk_audio_cpu_metadata",
        )

    def test_longer_cpu_metadata_marker_precedes_static_fallback(self) -> None:
        row = {
            "output_root": (
                "inference/results_fp8_static_qk_audio_cpu_metadata_pack"
            )
        }
        self.assertEqual(
            precision_for_row(row),
            "fp8_static_qk_audio_cpu_metadata",
        )
        self.assertEqual(PRECISIONS[-1], "fp8_static_qk_audio_cpu_metadata")


if __name__ == "__main__":
    unittest.main()
