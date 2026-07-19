#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

exec uv run \
  --project "$REPO_ROOT" \
  --frozen \
  --with qwen-asr==0.0.6 \
  --with torch==2.11.0+cu128 \
  --with torchvision==0.26.0+cu128 \
  --with torchaudio==2.11.0+cu128 \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  python "$SCRIPT_DIR/record_fp8_static_scales.py" "$@"
