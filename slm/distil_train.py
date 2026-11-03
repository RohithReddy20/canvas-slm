"""Upload folder for training the student on distil labs' GPUs instead of Kaggle.

Takes prepare_scripts.py output: every training row becomes a two-turn conversation, the
user message exactly as the student sees it and the reply as the answer. Our system
prompt goes in as the task description, which distil labs trains and serves as the
system prompt, so the adapter it returns answers our own prompt format. The test set is
held-out compounds rendered the same way (distil labs' judge scores them; the canvas
scoring stays in evaluate_scripts.py).

    python distil_train.py --rows ../runs/distil/train6/scripts-data --output ../runs/distil/train6/dataset
    python distil_train.py --rows ... --output ... --thin accuracy-train-single=0.25 \\
        --drop-conversations ../runs/colab/real-test-v3.jsonl     # a quarter of the v7 single-command rows
    distil dataset create --data ../runs/distil/train6/dataset
    distil slm create-from-dataset --smoke <dataset-id>     # 128 longest rows, under a dollar
    distil slm create-from-dataset <dataset-id>             # the whole set
"""

import argparse
import json
import random
from pathlib import Path

from contract import SYSTEM, messages, script_text
from convert import session_for

CONFIG = """base:
  task: question-answering
  student_model_name: {student}
  should_expand_dataset: false
synthgen:
  output_is_json: true
tuning:
  learning_rate: 2e-4
  learning_rate_scheduler: cosine
  warmup_ratio: 0.05
  lora_r: 64
  lora_alpha_multiplier: 2
  num_train_epochs: {epochs}
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  max_completion_length: 2048
"""
JUDGE_INSTRUCTIONS = (
    "The answer is correct when it is a JSON edit script whose actions, applied in order to "
    "the canvas in the input, do what the command asks and nothing more, or the refusal "
    '{"actions": [], "reason": "unsupported_request"} when no tool can carry the command out.'
)


def conversation_of(row_id):
    """wildchat:<hash>:d12:r3 and wildchat:<hash>:2 both come from the chat wildchat:<hash>."""
    return ":".join(row_id.split(":")[:2])


def keep(row_id, thin, dropped, rng):
    if row_id.startswith(("wildchat:", "sharegpt:")) and conversation_of(row_id) in dropped:
        return False
    for prefix, fraction in thin.items():
        if row_id.startswith(prefix):
            return rng.random() < fraction
    return True


def conversation(chat, answer):
    system, user = [m["content"] for m in chat]
    assert system == SYSTEM, "a training row carries a different system prompt"
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True, help="prepare_scripts.py output folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student", default="Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--test-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--thin", action="append", default=[], metavar="PREFIX=FRACTION",
                        help="keep only this fraction of the rows whose id starts with PREFIX")
    parser.add_argument("--drop-conversations", type=Path,
                        help="a real test key: training rows from its chats stay out (their canvases leak)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    thin = {prefix: float(fraction) for prefix, fraction in (pair.split("=") for pair in args.thin)}
    dropped = set()
    if args.drop_conversations:
        dropped = {conversation_of(json.loads(line)["id"]) for line in open(args.drop_conversations)}
    rng = random.Random(args.seed)

    seen, skipped = set(), 0
    with open(args.rows / "train.jsonl") as source, open(args.output / "train.jsonl", "w") as out:
        for row in map(json.loads, source):
            if not keep(row["id"], thin, dropped, rng):
                skipped += 1
                continue
            item = conversation(row["messages"], row["reply"])
            if item["messages"][0]["content"] in seen:  # the same canvas and command twice
                continue
            seen.add(item["messages"][0]["content"])
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    dev = [json.loads(line) for line in open(args.rows / "dev-compounds.jsonl")]
    random.Random(args.seed).shuffle(dev)
    tests = []
    for row in dev:
        item = conversation(messages(row["command"], session_for(row)), script_text(row["script"]))
        if item["messages"][0]["content"] not in seen:
            tests.append(item)
        if len(tests) == args.test_rows:
            break
    with open(args.output / "test.jsonl", "w") as out:
        for item in tests:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    json.dump({"task_description": SYSTEM, "llm_as_a_judge_instructions": JUDGE_INSTRUCTIONS},
              open(args.output / "job_description.json", "w"), indent=2)
    (args.output / "config.yaml").write_text(CONFIG.format(student=args.student, epochs=args.epochs))
    print(json.dumps({"train": len(seen), "test": len(tests), "skipped": skipped}))


if __name__ == "__main__":
    main()
