#!/usr/bin/env bash
# Script model: data from the v7 bundle plus compounds, autotuned LoRA training on
# Qwen3-1.7B, then vLLM scoring on dev commands, sessions, compounds and real follow-ups.
set -euo pipefail
cd "$(dirname "$0")"
DATA=/content/data/v7
RUNS=/content/runs
OUT=$RUNS/scripts-q17

echo "== start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m pip install -q "transformers[hub-kernels]==4.57.6" "peft>=0.17,<0.23" "pydantic>=2.12,<3" liger-kernel
# PEFT refuses the torchao Colab preinstalls (0.10); nothing here uses it.
python -m pip uninstall -q -y torchao
python -c "import torch, transformers, peft, kernels, liger_kernel; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
mkdir -p "$DATA"
[ -f "$DATA/examples.jsonl" ] || tar -xJf /content/data.tar.xz -C "$DATA"

echo "== prepare"
python prepare_scripts.py --data "$DATA" --output "$RUNS/scripts-data"

echo "== train"
python train.py --model Qwen/Qwen3-1.7B --rows "$RUNS/scripts-data/train.jsonl" --output "$OUT" \
  --autotune --merge

echo "== evaluate"
# vLLM pins its own torch, so it gets an environment of its own.
uv venv -q --python 3.12 /content/vllm-env
uv pip install -q --python /content/vllm-env/bin/python "vllm==0.30.0" "pydantic>=2.12,<3"
/content/vllm-env/bin/python evaluate_scripts.py --model "$OUT/merged" --data "$DATA" \
  --compounds "$RUNS/scripts-data/dev-compounds.jsonl" --real /content/input/real-test-v1.jsonl \
  --output "$OUT/report.json"
