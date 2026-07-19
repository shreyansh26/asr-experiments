# FlashAttention forward combine in Nsight Systems

This note explains the `FlashAttnFwdCombine` kernel that follows
`FlashAttnFwdSm90` in the Qwen3-ASR Nsight Systems node traces. It is a normal
part of FlashAttention's split-KV decoding path, not an FP8 scaling kernel.

For the complete dynamic-versus-static FP8 profiling workflow and decoder
kernel order, see the
[Nsight Systems dynamic-versus-static FP8 guide](nsys-fp8-dynamic-vs-static.md).

## What the two kernels represent

During autoregressive decoding, the query is short—often one token—while the
KV cache can contain many tokens. Processing the whole KV sequence with one
thread block per query/head can leave the GPU underutilized. FlashAttention can
therefore divide the KV sequence into multiple chunks and process those chunks
in parallel:

```text
KV cache
┌──────────┬──────────┬──────────┬──────────┐
│ split 0  │ split 1  │ split 2  │ split 3  │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     │          │          │          │
     └──────────┴──── FlashAttnFwdSm90 ┘
                       partial outputs
                       and local LSEs
                              │
                              v
                    FlashAttnFwdCombine
                              │
                              v
                    final attention output
```

The main `FlashAttnFwdSm90` work reads the KV chunks and produces a partial
attention output plus softmax normalization information for each split.
`FlashAttnFwdCombine` then merges those partial results into the exact output
that an unsplit attention calculation would produce.

FlashAttention exposes this behavior through `num_splits` in
`flash_attn_with_kvcache`:

- `num_splits > 1` explicitly splits K/V along the sequence dimension;
- `num_splits == 1` does not use the split-KV combine path;
- `num_splits == 0` lets FlashAttention choose with a performance heuristic.

The installed `flash_attn_3` package uses the heuristic path unless the caller
supplies another value.

## Why partial outputs cannot simply be added

For KV split `i`, define:

```text
scores_i = Q K_i^T / sqrt(head_dim)
lse_i    = logsumexp(scores_i)
output_i = softmax(scores_i) V_i
```

Each `output_i` is normalized only over its own KV chunk. The combine kernel
first recovers the global softmax normalization:

```text
global_lse = logsumexp(lse_0, lse_1, ..., lse_n)
```

It then weights every partial output by the probability mass represented by
that split:

```text
final_output = sum_i(exp(lse_i - global_lse) * output_i)
```

This is a numerically stable recombination. A plain sum, average, or
concatenation would be incorrect.

For example, suppose two splits produce:

```text
split 0: lse = 5, output = [1, 0]
split 1: lse = 4, output = [0, 2]
```

Then:

```text
global_lse = log(exp(5) + exp(4)) ~= 5.313
weight_0   = exp(5 - 5.313)        ~= 0.731
weight_1   = exp(4 - 5.313)        ~= 0.269

final_output = 0.731 * [1, 0] + 0.269 * [0, 2]
             = [0.731, 0.538]
```

## Pseudocode for the profiled sequence

The relevant decoder sequence is approximately:

```python
write_new_kv_to_paged_cache(k, v)

partial_outputs, partial_lses = flash_attention_over_kv_splits(
    q,
    paged_kv_cache,
)

attention_output = combine_attention_splits(
    partial_outputs,
    partial_lses,
)

# The following Triton/CUTLASS work prepares the combined result and applies
# the attention output projection.
o_proj_input_fp8, input_scale = quantize_for_o_proj(attention_output)
attention_delta = fp8_gemm(o_proj_input_fp8, o_proj_weight_fp8, input_scale)
```

In Events View this appears as:

```text
reshape_and_cache_flash_kernel
FlashAttnFwdSm90
FlashAttnFwdCombine
triton_poi_fused_0
...
```

The combine kernel runs before `o_proj`; it does not perform the projection
itself.

## What the current captures show

Each dominant decode-graph replay contains:

| Kernel | Dynamic FP8 | Static FP8 |
| --- | ---: | ---: |
| `FlashAttnFwdSm90` | 28 | 28 |
| `FlashAttnFwdCombine` | 28 | 28 |

The mean combine-kernel duration is approximately 1.69 microseconds in the
dynamic capture and 1.71 microseconds in the static capture. The small
difference is not evidence of a static-FP8 optimization; both precisions run
the same attention structure.

Static FP8 changes how decoder linear inputs are scaled before CUTLASS FP8
GEMMs. It does not remove or modify the mathematically required attention
combine step. In the current traces, Q/K/V and the KV cache are BF16.

## Interpreting the kernel in Nsight Systems

Use the node trace and filter Events View to one CUDA graph replay. The expected
pattern is one main FlashAttention kernel immediately followed by one combine
kernel for every decoder layer.

Keep these distinctions in mind:

- It combines partial results from KV-sequence splits for the same attention
  operation.
- It does not combine attention heads.
- It does not concatenate Q, K, and V.
- It does not write the KV cache; `reshape_and_cache_flash_kernel` does that.
- It is not one of the dynamic FP8 activation `absmax` reductions.
- The CUDA grid dimensions do not directly equal the number of KV splits;
  they also reflect batch, head, query-tile, and kernel scheduling choices.

If the selected FlashAttention configuration uses only one KV split, a
separate combine kernel is unnecessary and may not appear.

## References

- [FlashAttention repository](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention Python interface and `flash_attn_with_kvcache`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py)

