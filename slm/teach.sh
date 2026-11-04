#!/usr/bin/env bash
# Teacher labels for real diagram follow-ups (teach.py), on a GPU VM with FP8 support.
set -euo pipefail
cd "$(dirname "$0")"

echo "== start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m pip install -q "vllm==0.30.0" "pydantic>=2.12,<3"
python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)"
echo "== teach"
python teach.py --input /content/input/teach-input.jsonl --output /content/runs/teach
