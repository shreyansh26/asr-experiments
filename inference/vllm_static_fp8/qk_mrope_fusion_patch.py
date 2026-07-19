"""Fuse Q/K RMSNorm and Qwen3-ASR multi-axis RoPE in one Triton kernel."""

from __future__ import annotations

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton


logger = init_logger("vllm.qwen3_asr_qk_mrope")


@triton.jit
def _qk_norm_mrope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    positions_ptr,
    cache_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    position_axis_stride,
    position_token_stride,
    cache_position_stride: tl.constexpr,
    eps: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_rotary_dim: tl.constexpr,
    mrope_h_end: tl.constexpr,
    mrope_w_end: tl.constexpr,
    heads_per_program: tl.constexpr,
    groups_per_token: tl.constexpr,
    block_heads: tl.constexpr,
):
    program = tl.program_id(0)
    token = program // groups_per_token
    head_group = program % groups_per_token
    heads = head_group * heads_per_program + tl.arange(0, block_heads)[:, None]
    dims = tl.arange(0, head_dim)[None, :]
    valid_head = heads < num_q_heads + num_kv_heads
    is_q = heads < num_q_heads
    local_head = tl.where(is_q, heads, heads - num_q_heads)

    q_input = q_ptr + token * q_token_stride + local_head * head_dim + dims
    k_input = k_ptr + token * k_token_stride + local_head * head_dim + dims
    input_ptrs = tl.where(is_q, q_input, k_input)
    values = tl.load(input_ptrs, mask=valid_head, other=0.0).to(tl.float32)

    squared_mean = tl.sum(values * values, axis=1) / head_dim
    inv_rms = tl.rsqrt(squared_mean + eps)
    q_weights = q_weight_ptr + dims
    k_weights = k_weight_ptr + dims
    weights = tl.load(tl.where(is_q, q_weights, k_weights), mask=valid_head)
    # Inductor fuses the RMSNorm epilogue into its MRoPE kernel, so this
    # intermediate remains FP32 in the control graph.
    normalized = values * inv_rms[:, None] * weights.to(tl.float32)

    frequency = dims % half_rotary_dim
    h_axis = (frequency % 3 == 1) & (frequency <= mrope_h_end)
    w_axis = (frequency % 3 == 2) & (frequency <= mrope_w_end)
    axis = tl.where(h_axis, 1, tl.where(w_axis, 2, 0))
    position = tl.load(
        positions_ptr
        + axis * position_axis_stride
        + token * position_token_stride
    )
    cache_base = position * cache_position_stride + frequency
    cosine = tl.load(cache_ptr + cache_base).to(tl.float32)
    sine = tl.load(cache_ptr + cache_base + half_rotary_dim).to(tl.float32)

    partner_dims = (dims + half_rotary_dim) % head_dim
    partner_q = q_ptr + token * q_token_stride + local_head * head_dim + partner_dims
    partner_k = k_ptr + token * k_token_stride + local_head * head_dim + partner_dims
    partner_values = tl.load(
        tl.where(is_q, partner_q, partner_k), mask=valid_head, other=0.0
    ).to(tl.float32)
    partner_q_weight = q_weight_ptr + partner_dims
    partner_k_weight = k_weight_ptr + partner_dims
    partner_weights = tl.load(
        tl.where(is_q, partner_q_weight, partner_k_weight), mask=valid_head
    ).to(tl.float32)
    partner_normalized = partner_values * inv_rms[:, None] * partner_weights

    sign = tl.where(dims < half_rotary_dim, -1.0, 1.0)
    rotated = normalized * cosine + sign * partner_normalized * sine

    q_output = q_out_ptr + token * (num_q_heads * head_dim) + local_head * head_dim + dims
    k_output = k_out_ptr + token * (num_kv_heads * head_dim) + local_head * head_dim + dims
    tl.store(tl.where(is_q, q_output, k_output), rotated, mask=valid_head)


def fused_qk_norm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    mrope_section: list[int],
    heads_per_program: int = 16,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K head-wise RMSNorm and interleaved-section MRoPE."""
    if positions.ndim != 2 or positions.shape[0] != 3:
        raise ValueError("The fused kernel requires three-axis MRoPE positions")
    if q.shape[-1] != 2048 or k.shape[-1] != 1024:
        raise ValueError("The fused kernel is specialized for Qwen3-ASR-1.7B")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("The fused kernel currently supports BF16 Q/K inputs")
    if mrope_section != [24, 20, 20]:
        raise ValueError("Unexpected Qwen3-ASR MRoPE section layout")

    q_out = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=q.dtype)
    k_out = torch.empty((k.shape[0], k.shape[1]), device=k.device, dtype=k.dtype)
    total_heads = 24
    groups_per_token = triton.cdiv(total_heads, heads_per_program)
    block_heads = triton.next_power_of_2(heads_per_program)
    _qk_norm_mrope_kernel[(q.shape[0] * groups_per_token,)](
        q,
        k,
        q_out,
        k_out,
        positions,
        cos_sin_cache,
        q_weight,
        k_weight,
        q.stride(0),
        k.stride(0),
        positions.stride(0),
        positions.stride(1),
        cos_sin_cache.stride(0),
        eps,
        16,
        8,
        128,
        64,
        3 * mrope_section[1],
        3 * mrope_section[2],
        heads_per_program,
        groups_per_token,
        block_heads,
        num_warps=num_warps,
    )
    return q_out, k_out


def install_qk_mrope_fusion_patch() -> None:
    """Patch the Qwen3 attention forward only for compatible ASR layers."""
    from vllm.model_executor.models import qwen3

    original_forward = qwen3.Qwen3Attention.forward
    if getattr(original_forward, "_asr_qk_mrope_fusion", False):
        return

    def patched_forward(
        self: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        rotary = self.rotary_emb
        compatible = (
            positions.ndim == 2
            and getattr(rotary, "mrope_interleaved", False)
            and getattr(rotary, "mrope_section", None) == [24, 20, 20]
            and self.num_heads == 16
            and self.num_kv_heads == 8
            and self.head_dim == 128
        )
        if not compatible:
            return original_forward(self, positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        cache = rotary._match_cos_sin_cache_dtype(q)
        q, k = fused_qk_norm_mrope(
            q,
            k,
            positions,
            cache,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
            rotary.mrope_section,
        )
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    patched_forward._asr_qk_mrope_fusion = True
    qwen3.Qwen3Attention.forward = patched_forward
    logger.info("Installed Qwen3-ASR Q/K RMSNorm + MRoPE fusion patch")
