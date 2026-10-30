#!/usr/bin/env bash
# Real diagram conversations from WildChat-4.8M and ShareGPT, on a CPU VM.
set -euo pipefail
cd "$(dirname "$0")"

echo "== start $(date -u +%FT%TZ)"
echo "cpus $(nproc), memory $(free -g | awk '/Mem:/ {print $2}') GB"
python -c "import huggingface_hub, pyarrow; print('huggingface_hub', huggingface_hub.__version__, 'pyarrow', pyarrow.__version__)"

echo "== smoke"
python mine_chats.py --output /content/runs/mine-smoke --shards 2 --sample 20
echo "== mine"
python mine_chats.py --output /content/runs/mine
