"""Merge a LoRA adapter trained elsewhere (distil labs' model-adapter/ folder) into the base
model, so scripts.sh can evaluate it like a student trained here.

    python merge_adapter.py --adapter /tmp/work/model-adapter --output /kaggle/working/runs/scripts-q17/merged
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
