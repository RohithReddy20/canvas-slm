#!/usr/bin/env bash
# Capacity comparison on the frozen v7 data: one LoRA pass per base model, scored on validation.
set -euo pipefail
cd "$(dirname "$0")"
DATA=/content/data/v7
RUNS=/content/runs

echo "== start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m pip install -q "transformers==4.57.6" "peft>=0.17,<0.23" "pydantic>=2.12,<3"
# PEFT refuses the torchao Colab preinstalls (0.10); nothing here uses it.
python -m pip uninstall -q -y torchao
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
mkdir -p "$DATA"
[ -f "$DATA/examples.jsonl" ] || tar -xJf /content/data.tar.xz -C "$DATA"

echo "== smoke"
python train.py --model Qwen/Qwen3-0.6B --data "$DATA" --output "$RUNS/smoke" --rows 320
python evaluate.py --model Qwen/Qwen3-0.6B --adapter "$RUNS/smoke/adapter" --data "$DATA" \
  --limit 48 --session-limit 1 --output "$RUNS/smoke/valid.json"

for spec in "unsloth/functiongemma-270m-it fg270m" "Qwen/Qwen3-0.6B q06" "Qwen/Qwen3-1.7B q17"; do
  set -- $spec
  echo "== train $2"
  python train.py --model "$1" --data "$DATA" --output "$RUNS/$2"
  echo "== evaluate $2"
  python evaluate.py --model "$1" --adapter "$RUNS/$2/adapter" --data "$DATA" \
    --output "$RUNS/$2/valid.json"
done
