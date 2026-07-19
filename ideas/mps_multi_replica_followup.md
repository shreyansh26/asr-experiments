# Same-GPU MPS multi-replica follow-up

## Status

Deferred to the multi-replica phase. The selected result on this branch is a
verified one-replica-per-GPU kernel improvement, so an MPS experiment can now
use it as the per-replica baseline without confusing replica scaling with the
kernel result.

## vLLM 0.24 execution model

The installed `vllm serve --help=ParallelConfig` explicitly says that its data
parallel flags are for MoE deployments and that non-MoE models should use
independent vLLM instances. Qwen3-ASR-1.7B therefore needs:

1. one CUDA MPS control daemon for the selected physical GPU;
2. two independent static-FP8 vLLM servers on that GPU, each constrained with
   `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` and a memory-utilization limit;
3. an external least-loaded or round-robin request router in front of the two
   API ports.

`--api-server-count` is not a substitute: it creates more HTTP frontends for
one engine rather than independent GPU replicas.

## Required comparison

Use the same free GPU and compare the selected single replica against two MPS
replicas. Run the 16-worker uniform and full-corpus batched commands, then the
full quality gate. Report aggregate throughput and per-request latency/TTFT;
also confirm that each backend receives requests. The MPS result is successful
only if it improves aggregate batched service without undoing the selected
branch's latency gain or changing CER/WER beyond normal run-to-run variation.

Profile both server processes. Per-process traces must not be compared as if
either represented whole-GPU time; combine their overlapping GPU intervals or
collect a system-wide trace.

## Isolation decision

Do not add MPS launch or routing changes to `opt2/qk-kvcache-prefill-heads`.
Create a child branch/worktree after the single-replica branch is accepted.
This preserves a clean kernel-only benchmark and avoids repeating the earlier
mistake of treating two replicas as a faster single-replica configuration.

Reference: <https://sgl-project.github.io/sglang-omni/basic_usage/same_gpu_dp.html>
