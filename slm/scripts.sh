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
# Colab's default PyTorch is newer than any prebuilt FlashAttention build, so training runs
# in its own environment on PyTorch 2.8 (CUDA 12.8). Liger 0.7+ needs transformers 5.
TRAIN=/content/train-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FLASH=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
[ -x "$TRAIN" ] || uv venv -q --python 3.12 /content/train-env
uv pip install -q --python "$TRAIN" "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python "$TRAIN" "transformers[hub-kernels]==4.57.6" "peft>=0.17,<0.23" \
  "pydantic>=2.12,<3" "liger-kernel==0.6.4"
uv pip install -q --python "$TRAIN" "$FLASH" || echo "flash-attn wheel unavailable"
"$TRAIN" -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
mkdir -p "$DATA"
[ -f "$DATA/examples.jsonl" ] || tar -xJf /content/data.tar.xz -C "$DATA"

echo "== prepare"
[ -f "$RUNS/scripts-data/train.jsonl" ] || \
  python prepare_scripts.py --data "$DATA" --output "$RUNS/scripts-data"

echo "== train"
[ -f "$OUT/merged/config.json" ] || \
  "$TRAIN" train.py --model Qwen/Qwen3-1.7B --rows "$RUNS/scripts-data/train.jsonl" --output "$OUT" \
    --autotune --merge

echo "== evaluate"
# vLLM pins its own torch, so it gets an environment of its own.
[ -x /content/vllm-env/bin/python ] || uv venv -q --python 3.12 /content/vllm-env
uv pip install -q --python /content/vllm-env/bin/python "vllm==0.30.0" "pydantic>=2.12,<3"
# vLLM compiles kernels on first use with the environment's own ninja.
PATH=/content/vllm-env/bin:$PATH /content/vllm-env/bin/python evaluate_scripts.py --model "$OUT/merged" \
  --data "$DATA" --compounds "$RUNS/scripts-data/dev-compounds.jsonl" \
  --real /content/input/real-test-v1.jsonl \
  --variants free-exact,free-fuzzy,constrained-exact,constrained-fuzzy --output "$OUT/report.json"
