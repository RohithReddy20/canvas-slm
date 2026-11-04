#!/usr/bin/env bash
# Rescore the script model: free or grammar-constrained decoding, exact or fuzzy names.
set -euo pipefail
cd "$(dirname "$0")"
DATA=/content/data/v7
RUNS=/content/runs

echo "== start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$DATA"
[ -f "$DATA/examples.jsonl" ] || tar -xJf /content/data.tar.xz -C "$DATA"
python -m pip install -q "pydantic>=2.12,<3"

echo "== prepare"
[ -f "$RUNS/scripts-data/dev-compounds.jsonl" ] || \
  python prepare_scripts.py --data "$DATA" --output "$RUNS/scripts-data"

echo "== evaluate"
[ -x /content/vllm-env/bin/python ] || uv venv -q --python 3.12 /content/vllm-env
uv pip install -q --python /content/vllm-env/bin/python "vllm==0.30.0" "pydantic>=2.12,<3"
# vLLM compiles kernels on first use with the environment's own ninja.
PATH=/content/vllm-env/bin:$PATH /content/vllm-env/bin/python evaluate_scripts.py \
  --model Qwen/Qwen3-1.7B --adapter /content/input/adapter --data "$DATA" \
  --compounds "$RUNS/scripts-data/dev-compounds.jsonl" --real /content/input/real-test-v1.jsonl \
  --variants free-exact,free-fuzzy,constrained-exact,constrained-fuzzy \
  --output "$RUNS/rescore/report.json"
