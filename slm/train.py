"""LoRA fine-tuning on the action contract, with loss on the reply tokens only."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import messages, parse, reply


def end_marker(tokenizer):
    vocabulary = tokenizer.get_vocab()
    return next(
        (m for m in ("<|im_end|>", "<end_of_turn>") if m in vocabulary),
        tokenizer.eos_token,
    )


def prompt_ids(tokenizer, chat):
    text = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def read_rows(path, split):
    with open(path) as source:
        return [row for line in source if (row := json.loads(line))["split"] == split]


def encode(tokenizer, rows, limit):
    marker, examples, dropped = end_marker(tokenizer), [], 0
    for row in rows:
        answer = reply(row["expected"], row["canvas"])
        if parse(answer, row["canvas"]) != row["expected"]:
            raise ValueError(f"Reply format does not round-trip: {row['id']}")
        prompt = prompt_ids(
            tokenizer, messages(row["command"], row["canvas"], row.get("history"))
        )
        completion = tokenizer(answer + marker, add_special_tokens=False)["input_ids"]
        if len(prompt) + len(completion) > limit:
            dropped += 1
            continue
        examples.append((prompt, completion))
    return examples, dropped


def micro_batches(examples, max_tokens, max_rows, rng):
    order = sorted(
        range(len(examples)), key=lambda i: (sum(map(len, examples[i])), rng.random())
    )
    batches, current, longest = [], [], 0
    for index in order:
        length = sum(map(len, examples[index]))
        if current and (
            max(longest, length) * (len(current) + 1) > max_tokens
            or len(current) == max_rows
        ):
            batches.append(current)
            current, longest = [], 0
        current.append(index)
        longest = max(longest, length)
    batches.append(current)
    rng.shuffle(batches)
    return batches


def optimizer_steps(batches, rows_per_step):
    steps, current, count = [], [], 0
    for batch in batches:
        current.append(batch)
        count += len(batch)
        if count >= rows_per_step:
            steps.append(current)
            current, count = [], 0
    if current:
        steps.append(current)
    return steps


def collate(examples, indices, pad_id):
    rows = [examples[i][0] + examples[i][1] for i in indices]
    width = max(map(len, rows))
    keep = max(len(examples[i][1]) for i in indices) + 1
    # Left padding puts every reply at the end, so logits are kept for the last positions only.
    ids = torch.full((len(rows), width), pad_id)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), keep - 1), -100)
    for row, (index, tokens) in enumerate(zip(indices, rows, strict=True)):
        ids[row, width - len(tokens) :] = torch.tensor(tokens)
        mask[row, width - len(tokens) :] = 1
        completion = examples[index][1]
        labels[row, keep - 1 - len(completion) :] = torch.tensor(completion)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, positions, labels, keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--rows", type=int, help="use only this many training rows")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--rows-per-step", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = read_rows(args.data / "examples.jsonl", "train")
    rng.shuffle(rows)
    if args.rows:
        rows = rows[: args.rows]
    examples, dropped = encode(tokenizer, rows, args.max_length)
    if not examples:
        raise SystemExit(f"No training row fits within --max-length {args.max_length}.")
    lengths = sorted(sum(map(len, example)) for example in examples)
    print(
        f"{len(examples)} rows, {dropped} over {args.max_length} tokens dropped; "
        f"tokens median {lengths[len(lengths) // 2]}, max {lengths[-1]}, "
        f"total {sum(lengths)}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    epochs = math.ceil(args.epochs)
    plan = []
    for epoch in range(epochs):
        steps = optimizer_steps(
            micro_batches(examples, args.max_tokens, 64, rng), args.rows_per_step
        )
        if epoch + 1 > args.epochs:
            steps = steps[: round(len(steps) * (args.epochs - epoch))]
        plan.extend(steps)
    warmup = max(1, round(0.03 * len(plan)))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup
        if step < warmup
        else 0.1
        + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, len(plan) - warmup))),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    model.train()
    start, seen_tokens, recent = time.time(), 0, []
    for number, step in enumerate(plan, 1):
        targets = sum(len(examples[i][1]) for batch in step for i in batch)
        total = 0.0
        for batch in step:
            ids, mask, positions, labels, keep = collate(examples, batch, pad_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=ids.cuda(),
                    attention_mask=mask.cuda(),
                    position_ids=positions.cuda(),
                    logits_to_keep=keep,
                ).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().flatten(0, 1),
                labels.cuda().flatten(),
                ignore_index=-100,
                reduction="sum",
            )
            (loss / targets).backward()
            total += loss.item()
            seen_tokens += int(mask.sum())
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        recent.append(total / targets)
        if number % 20 == 0 or number == len(plan):
            elapsed = time.time() - start
            print(
                f"step {number}/{len(plan)} loss {sum(recent) / len(recent):.4f} "
                f"lr {scheduler.get_last_lr()[0]:.2e} {seen_tokens / elapsed:.0f} tok/s "
                f"eta {elapsed / number * (len(plan) - number) / 60:.1f} min",
                flush=True,
            )
            recent = []
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    (args.output / "training.json").write_text(
        json.dumps(
            {
                **{key: str(value) for key, value in vars(args).items()},
                "rows": len(examples),
                "dropped": dropped,
                "optimizer_steps": len(plan),
                "train_tokens": seen_tokens,
                "seconds": time.time() - start,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            },
            indent=2,
        )
    )
    print(f"saved {args.output / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
