"""Load the local vLLM static-FP8 extension before vLLM parses its CLI."""

import os


if os.environ.get("ASR_FP8_STATIC_SCALES_JSON"):
    import vllm_static_fp8_plugin  # noqa: F401
