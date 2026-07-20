#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ASR_QK_MROPE_FUSION=1

echo "Q/K RMSNorm + MRoPE + KV-cache fusion: enabled"

exec "$SCRIPT_DIR/run_vllm_fp8_static.sh" "$@"
