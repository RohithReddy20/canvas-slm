#!/usr/bin/env bash
# Teacher-written training data (teach_data.py): real-style requests and v7 paraphrases,
# each checked by an independent teacher reply. TEACH holds extra teach_data.py options.
set -euo pipefail
cd "$(dirname "$0")"
WORK=${WORK:-/content}
DATA=${DATA:-$WORK/data/v7}
RUNS=${RUNS:-$WORK/runs}
INPUT=${INPUT:-$WORK/input}

echo "== start $(date -u +%FT%TZ)"
mkdir -p "$DATA"
[ -f "$DATA/examples.jsonl" ] || tar -xJf "$WORK/data.tar.xz" -C "$DATA"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
command -v uv >/dev/null || python -m pip install -q uv
[ -x "$WORK/vllm-env/bin/python" ] || uv venv -q --python 3.12 "$WORK/vllm-env"
uv pip install -q --python "$WORK/vllm-env/bin/python" "vllm==0.30.0" "pydantic>=2.12,<3"
export PATH=$WORK/vllm-env/bin:$PATH HF_HOME=$WORK/hf
echo "== teach"
read -r -a OPTIONS <<< "${TEACH:-}"
python teach_data.py run --data "$DATA" --input "$INPUT" --output "$RUNS/teach-data" "${OPTIONS[@]}"
# Packed for the trip to the training job; the plain files are too large to collect.
tar -cJf "$RUNS/teach-data/teach-data.tar.xz" -C "$RUNS" teach-data/paraphrases.jsonl teach-data/requests.jsonl
ls -l "$RUNS/teach-data"
echo "== done $(date -u +%FT%TZ)"
