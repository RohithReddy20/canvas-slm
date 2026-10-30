#!/usr/bin/env bash
# Script model: data from the v7 bundle plus compounds, autotuned LoRA training on
# Qwen3-1.7B, then vLLM scoring on dev commands, sessions, compounds and real follow-ups.
# Paths default to Colab's; kaggle_run.py sets them for a Kaggle kernel.
set -euo pipefail
cd "$(dirname "$0")"
WORK=${WORK:-/content}
DATA=${DATA:-$WORK/data/v7}
RUNS=${RUNS:-$WORK/runs}
INPUT=${INPUT:-$WORK/input}
OUT=$RUNS/scripts-q17

echo "== start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
GPUS=$(nvidia-smi -L | wc -l)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
command -v uv >/dev/null || python -m pip install -q uv
# Colab's default PyTorch is newer than any prebuilt FlashAttention build, so training runs
# in its own environment on PyTorch 2.8 (CUDA 12.8). Liger 0.7+ needs transformers 5.
TRAIN=$WORK/train-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FLASH=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
[ -x "$TRAIN" ] || uv venv -q --python 3.12 "$WORK/train-env"
uv pip install -q --python "$TRAIN" "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python "$TRAIN" "transformers[hub-kernels]==4.57.6" "peft>=0.17,<0.23" \
  "pydantic>=2.12,<3" "liger-kernel==0.6.4"
# FlashAttention needs Ampere or newer; T4s train with padded SDPA.
if [ "${CAP%%.*}" -ge 8 ]; then
  uv pip install -q --python "$TRAIN" "$FLASH" || echo "flash-attn wheel unavailable"
fi
"$TRAIN" -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
mkdir -p "$DATA"
[ -f "$DATA/examples.jsonl" ] || tar -xJf "$WORK/data.tar.xz" -C "$DATA"

echo "== prepare"
[ -f "$RUNS/scripts-data/train.jsonl" ] || \
  "$TRAIN" prepare_scripts.py --data "$DATA" --output "$RUNS/scripts-data"

echo "== train"
TRAINING=(--model Qwen/Qwen3-1.7B --rows "$RUNS/scripts-data/train.jsonl" --output "$OUT")
# LIMIT trains on that many rows only, for a smoke run.
[ -z "${LIMIT:-}" ] || TRAINING+=(--limit-rows "$LIMIT")
if [ -f "$OUT/merged/config.json" ]; then
  echo "reusing $OUT/merged"
elif [ "$GPUS" -gt 1 ]; then
  # Tune on one GPU, then train the chosen setting on all of them. SETTINGS (train.py
  # options) skips tuning when an earlier run on the same GPU already chose.
  if [ -n "${SETTINGS:-}" ]; then
    read -r -a CHOSEN <<< "$SETTINGS"
  else
    "$TRAIN" train.py "${TRAINING[@]}" --autotune --tune-only
    read -r -a CHOSEN < <("$TRAIN" -c 'import json, sys
chosen = json.load(open(sys.argv[1]))["chosen"]
print(" ".join("--" + k.replace("_", "-") + " " + str(v) for k, v in chosen.items()))' "$OUT/autotune.json")
  fi
  "$TRAIN" -m torch.distributed.run --standalone --nproc-per-node "$GPUS" train.py \
    "${TRAINING[@]}" "${CHOSEN[@]}" --merge
else
  "$TRAIN" train.py "${TRAINING[@]}" --autotune --merge
fi

echo "== evaluate"
# vLLM pins its own torch, so it gets an environment of its own.
[ -x "$WORK/vllm-env/bin/python" ] || uv venv -q --python 3.12 "$WORK/vllm-env"
uv pip install -q --python "$WORK/vllm-env/bin/python" "vllm==0.30.0" "pydantic>=2.12,<3"
# vLLM compiles kernels on first use with the environment's own ninja.
export PATH=$WORK/vllm-env/bin:$PATH
EVALUATION=(evaluate_scripts.py --model "$OUT/merged" --data "$DATA"
  --compounds "$RUNS/scripts-data/dev-compounds.jsonl" --real "$INPUT/real-test-v1.jsonl")
if [ "$GPUS" -gt 1 ]; then
  # Free and constrained decoding share no generations, so each gets a GPU.
  CUDA_VISIBLE_DEVICES=0 python "${EVALUATION[@]}" --variants free-exact,free-fuzzy \
    --output "$OUT/report-free.json" &
  free=$!
  CUDA_VISIBLE_DEVICES=1 python "${EVALUATION[@]}" --variants constrained-exact,constrained-fuzzy \
    --output "$OUT/report-constrained.json" &
  constrained=$!
  wait "$free"
  wait "$constrained"
else
  python "${EVALUATION[@]}" --variants free-exact,free-fuzzy,constrained-exact,constrained-fuzzy \
    --output "$OUT/report.json"
fi
echo "== done $(date -u +%FT%TZ)"
