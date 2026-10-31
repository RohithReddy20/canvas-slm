#!/usr/bin/env bash
# Which 4-bit teacher runs on these GPUs, and how fast (teacher_probe.py); MODELS lists the
# candidates, each tried in its own process so a failed load frees the GPUs.
set -euo pipefail
cd "$(dirname "$0")"
WORK=${WORK:-/content}
DATA=${DATA:-$WORK/data/v7}
RUNS=${RUNS:-$WORK/runs}
MODELS=${MODELS:-"sokada4/Qwen3.8-27B-GPTQ-Int4 Max73333/Qwen3.8-27B-GPTQ-Int4-V100"}

echo "== start $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
df -h "$WORK" | tail -1
command -v uv >/dev/null || python -m pip install -q uv
[ -x "$WORK/vllm-env/bin/python" ] || uv venv -q --python 3.12 "$WORK/vllm-env"
uv pip install -q --python "$WORK/vllm-env/bin/python" "vllm==0.30.0" "pydantic>=2.12,<3"
export PATH=$WORK/vllm-env/bin:$PATH HF_HOME=$WORK/hf
for model in $MODELS; do
  echo "== probe $model"
  if python teacher_probe.py --model "$model" --data "$DATA" --output "$RUNS/teacher-probe"; then
    break
  fi
  rm -rf "$WORK/hf/hub"
done
echo "== done $(date -u +%FT%TZ)"
