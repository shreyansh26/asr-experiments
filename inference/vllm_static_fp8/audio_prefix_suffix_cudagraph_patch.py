"""Install the exact-shape audio prefix and suffix CUDA graphs together."""

from __future__ import annotations

import os

from audio_cpu_metadata_pack_patch import (
    ENV_NAME as METADATA_ENV_NAME,
    MAX_SEQLEN_ENV_NAME,
    _PATCH_MARKER as METADATA_PATCH_MARKER,
    install_audio_cpu_metadata_pack_patch,
)
from audio_prefix_cudagraph_patch import (
    _PATCH_MARKER as PREFIX_PATCH_MARKER,
    audio_prefix_cudagraph_enabled,
    install_audio_prefix_cudagraph_patch,
    run_audio_prefix_cudagraph,
)
from audio_suffix_cudagraph_patch import (
    _PATCH_MARKER as SUFFIX_PATCH_MARKER,
    audio_suffix_cudagraph_enabled,
    install_audio_suffix_cudagraph_patch,
    run_audio_suffix_cudagraph,
)
from vllm.logger import init_logger


logger = init_logger("vllm.qwen3_asr_audio_prefix_suffix_cudagraph")

_PATCH_MARKER = "_asr_audio_prefix_suffix_cudagraph_patch"


def audio_prefix_suffix_cudagraph_enabled() -> bool:
    """Return whether both strict graph gates are enabled."""
    prefix_enabled = audio_prefix_cudagraph_enabled()
    suffix_enabled = audio_suffix_cudagraph_enabled()
    return prefix_enabled and suffix_enabled


def install_audio_prefix_suffix_cudagraph_patch() -> bool:
    """Install both graph runners through one CPU-metadata forward wrapper."""
    if not audio_prefix_suffix_cudagraph_enabled():
        return False
    if (
        os.environ.get(METADATA_ENV_NAME, "0") != "1"
        or os.environ.get(MAX_SEQLEN_ENV_NAME, "0") != "1"
    ):
        logger.warning(
            "Combined audio CUDA graphs require %s=1 and %s=1; leaving the "
            "model unchanged",
            METADATA_ENV_NAME,
            MAX_SEQLEN_ENV_NAME,
        )
        return False

    from vllm.model_executor.models import qwen3_asr as asr_module
    from vllm.model_executor.models.qwen3_asr import Qwen3OmniMoeAudioEncoder

    current_forward = Qwen3OmniMoeAudioEncoder.forward
    current_field_config = asr_module._qwen3asr_field_config
    combined_patched = bool(getattr(current_forward, _PATCH_MARKER, False))
    prefix_patched = bool(getattr(current_forward, PREFIX_PATCH_MARKER, False))
    suffix_patched = bool(getattr(current_forward, SUFFIX_PATCH_MARKER, False))
    metadata_forward_patched = bool(
        getattr(current_forward, METADATA_PATCH_MARKER, False)
    )
    metadata_field_patched = bool(
        getattr(current_field_config, METADATA_PATCH_MARKER, False)
    )

    if combined_patched:
        if (
            prefix_patched
            and suffix_patched
            and metadata_forward_patched
            and metadata_field_patched
        ):
            return True
        raise RuntimeError(
            "Combined audio prefix/suffix CUDA graph patch is partially installed"
        )

    if (prefix_patched or suffix_patched) and not (
        metadata_forward_patched and metadata_field_patched
    ):
        raise RuntimeError("An audio CUDA graph patch is partially installed")
    if metadata_forward_patched != metadata_field_patched:
        raise RuntimeError("Audio CPU metadata patch is only partially installed")
    if prefix_patched or suffix_patched or metadata_forward_patched:
        logger.warning(
            "An audio forward patch was already installed without both graph "
            "runners; refusing a partial replacement"
        )
        return False

    if not install_audio_cpu_metadata_pack_patch(
        prefix_runner=run_audio_prefix_cudagraph,
        suffix_runner=run_audio_suffix_cudagraph,
    ):
        return False

    installed_forward = Qwen3OmniMoeAudioEncoder.forward
    setattr(installed_forward, PREFIX_PATCH_MARKER, True)
    setattr(installed_forward, SUFFIX_PATCH_MARKER, True)
    setattr(installed_forward, _PATCH_MARKER, True)
    return True


def install_requested_audio_cudagraph_patches() -> bool:
    """Install the exact requested combination without partial fallback."""
    prefix_enabled = audio_prefix_cudagraph_enabled()
    suffix_enabled = audio_suffix_cudagraph_enabled()
    if prefix_enabled and suffix_enabled:
        return install_audio_prefix_suffix_cudagraph_patch()
    if prefix_enabled:
        return install_audio_prefix_cudagraph_patch()
    if suffix_enabled:
        return install_audio_suffix_cudagraph_patch()
    if os.environ.get(METADATA_ENV_NAME, "0") == "1":
        return install_audio_cpu_metadata_pack_patch()
    return False
