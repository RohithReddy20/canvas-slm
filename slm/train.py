"""LoRA fine-tuning with loss on reply tokens only, tuned to the GPU it runs on.

Rows are v7 bundle rows (--data; one-action JSON replies from prompt.py) or rendered
{"messages", "reply"} rows (--rows). Two batch layouts:
- packed: examples concatenated into one row, position IDs restarting per example, so a
  variable-length FlashAttention kernel keeps them apart and no compute goes to padding;
- padded: left-padded rows through SDPA, the fallback when no FlashAttention kernel loads.
Either way only reply positions are projected to the vocabulary.

--autotune benchmarks settings (attention kernel, Liger kernels, activation checkpointing,
tokens per micro-batch) in separate processes on real batches, rejects any whose loss on a
fixed probe set drifts from the plain padded baseline, and trains with the fastest one.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import messages, parse, reply

PACKED_ATTENTION = ("kernels-community/flash-attn", "flash_attention_2")
# Dense BF16 tensor peaks, for the utilisation estimate only.
PEAK_FLOPS = {"H100": 989e12, "A100": 312e12, "RTX PRO 6000": 500e12, "L4": 121e12, "T4": 65e12}
PROBE_ROWS = 48
TOLERANCE = 0.02
HEADROOM = 0.8


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


def rendered_rows(args):
    """(id, messages, reply) for every training row."""
    if args.rows:
        with open(args.rows) as source:
            return [(r["id"], r["messages"], r["reply"]) for r in map(json.loads, source)]
    rendered = []
    for row in read_rows(args.data / "examples.jsonl", "train"):
        answer = reply(row["expected"], row["canvas"])
        if parse(answer, row["canvas"]) != row["expected"]:
            raise ValueError(f"Reply format does not round-trip: {row['id']}")
        rendered.append((row["id"], messages(row["command"], row["canvas"], row.get("history")), answer))
    return rendered


def encode(tokenizer, rows, limit):
    marker, examples, ids, dropped = end_marker(tokenizer), [], [], 0
    for identifier, chat, answer in rows:
        prompt = prompt_ids(tokenizer, chat)
        completion = tokenizer(answer + marker, add_special_tokens=False)["input_ids"]
        if len(prompt) + len(completion) > limit:
            dropped += 1
            continue
        examples.append((prompt, completion))
        ids.append(identifier)
    return examples, ids, dropped


def micro_batches(examples, max_tokens, packed, rng):
    """Groups of example indices: packed groups sum to at most max_tokens; padded groups
    have similar lengths, with their padded size within max_tokens."""
    lengths = [len(p) + len(c) for p, c in examples]
    if packed:
        order = rng.sample(range(len(examples)), len(examples))
    else:
        order = sorted(range(len(examples)), key=lambda i: (lengths[i], rng.random()))
    batches, current, size = [], [], 0
    for index in order:
        grown = size + lengths[index] if packed else max(size, lengths[index]) * (len(current) + 1)
        if current and (grown > max_tokens or len(current) == 64):
            batches.append(current)
            current, size = [], 0
        current.append(index)
        size = size + lengths[index] if packed else max(size, lengths[index])
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


class ReplyLoss:
    """Summed cross-entropy over reply tokens for a micro-batch, in either layout."""

    def __init__(self, model, packed, pad_id):
        base = model.get_base_model()
        self.model, self.decoder, self.head = model, base.model, base.lm_head
        self.cap = getattr(base.config, "final_logit_softcapping", None)
        self.packed, self.pad_id = packed, pad_id

    def logits(self, hidden):
        logits = self.head(hidden).float()
        return torch.tanh(logits / self.cap) * self.cap if self.cap else logits

    def __call__(self, examples, indices):
        if self.packed:
            ids, positions, predictors, targets, offset = [], [], [], [], 0
            for i in indices:
                prompt, completion = examples[i]
                tokens = prompt + completion
                ids += tokens
                positions += range(len(tokens))
                start = offset + len(prompt) - 1
                predictors += range(start, start + len(completion))
                targets += completion
                offset += len(tokens)
            hidden = self.decoder(
                input_ids=torch.tensor([ids], device="cuda"),
                position_ids=torch.tensor([positions], device="cuda"),
            ).last_hidden_state[0]
            logits = self.logits(hidden[torch.tensor(predictors, device="cuda")])
            target = torch.tensor(targets, device="cuda")
            return torch.nn.functional.cross_entropy(logits, target, reduction="sum"), len(ids)
        rows = [examples[i][0] + examples[i][1] for i in indices]
        width = max(map(len, rows))
        keep = max(len(examples[i][1]) for i in indices) + 1
        # Left padding puts every reply at the end, so hidden states are kept for the
        # last positions only.
        ids = torch.full((len(rows), width), self.pad_id)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), keep - 1), -100)
        for row, (index, tokens) in enumerate(zip(indices, rows, strict=True)):
            ids[row, width - len(tokens) :] = torch.tensor(tokens)
            mask[row, width - len(tokens) :] = 1
            completion = examples[index][1]
            labels[row, keep - 1 - len(completion) :] = torch.tensor(completion)
        hidden = self.decoder(
            input_ids=ids.cuda(),
            attention_mask=mask.cuda(),
            position_ids=(mask.cumsum(-1) - 1).clamp(min=0).cuda(),
        ).last_hidden_state[:, -keep:-1]
        logits = self.logits(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), labels.cuda().flatten(), ignore_index=-100, reduction="sum"
        )
        return loss, int(mask.sum())


def load_model(args, attention):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=attention
    ).cuda()
    if args.liger:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        _apply_liger_kernel_to_instance(
            model=model, rope=True, rms_norm=True, swiglu=True,
            cross_entropy=False, fused_linear_cross_entropy=False,
        )
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Checkpointing only a share of the layers trades part of the recompute for memory.
        layers = model.model.layers
        for layer in layers[round(len(layers) * args.checkpointing) :]:
            layer.gradient_checkpointing = False
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                   target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    return model


def peak_flops():
    name = torch.cuda.get_device_name()
    return next((value for key, value in PEAK_FLOPS.items() if key in name), None)


def autotune(args):
    """The fastest setting whose probe loss matches the baseline, measured in subprocesses."""
    results = []

    def run(setting):
        command = [c for c in [sys.executable, __file__, *sys.argv[1:]] if c != "--autotune"]
        command += ["--benchmark", str(args.benchmark_steps), "--limit-rows", "2000"]
        for key, value in setting.items():
            command += [f"--{key.replace('_', '-')}", str(value)]
        started = time.time()
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=1800)
            output, error = done.stdout, done.stderr
        except subprocess.TimeoutExpired:
            output, error = "", "timed out"
        line = next((x for x in output.splitlines() if x.startswith("BENCH ")), None)
        result = {**setting, "seconds": round(time.time() - started), "valid": False}
        if line:
            result.update(json.loads(line[6:]))
            reference = next((r["probe_loss"] for r in results if "probe_loss" in r), result["probe_loss"])
            result["drift"] = round(abs(result["probe_loss"] - reference) / reference, 4)
            result["valid"] = result["drift"] <= TOLERANCE
        else:
            result["error"] = (error or output)[-600:]
        results.append(result)
        print("autotune", json.dumps(result), flush=True)
        return result

    # Short benchmarks understate the peak of a full run, so a setting needs headroom.
    total_gb = float(
        subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split()[0]
    ) / 1024

    def best():
        valid = [r for r in results if r["valid"] and r["peak_gb"] <= HEADROOM * total_gb]
        if not valid:
            raise SystemExit("No benchmarked setting worked.")
        top = max(valid, key=lambda r: r["tokens_per_second"])
        return {k: top[k] for k in ("attention", "liger", "checkpointing", "max_tokens")}

    run({"attention": "sdpa", "liger": 0, "checkpointing": 1, "max_tokens": args.max_tokens})
    for attention in PACKED_ATTENTION:
        run({**best(), "attention": attention})
    run({**best(), "liger": 1})
    chosen = best()
    for share in (0, 0.5):
        for tokens in (chosen["max_tokens"], chosen["max_tokens"] // 2):
            run({**chosen, "checkpointing": share, "max_tokens": tokens})
    chosen = best()
    run({**chosen, "max_tokens": chosen["max_tokens"] * 2})
    return best(), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, help="v7 bundle folder")
    parser.add_argument("--rows", type=Path, help="rendered messages/reply rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--limit-rows", type=int, help="use only this many training rows")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--rows-per-step", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--liger", type=int, default=0)
    parser.add_argument("--checkpointing", type=float, default=1, help="share of layers")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--benchmark", type=int, help="time this many steps, print BENCH, exit")
    parser.add_argument("--benchmark-steps", type=int, default=8)
    parser.add_argument("--merge", action="store_true", help="also save merged weights")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if bool(args.data) == bool(args.rows):
        raise SystemExit("Give exactly one of --data and --rows.")

    torch.backends.cuda.matmul.allow_tf32 = True
    if args.autotune:
        best, results = autotune(args)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "autotune.json").write_text(json.dumps(results, indent=2))
        for key in ("attention", "liger", "checkpointing", "max_tokens"):
            setattr(args, key, best[key])
        print(f"training with {json.dumps(best)}", flush=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = rendered_rows(args)
    rows.sort(key=lambda r: r[0])
    rng.shuffle(rows)
    if args.limit_rows:
        rows = rows[: args.limit_rows]
    examples, ids, dropped = encode(tokenizer, rows, args.max_length)
    if not examples:
        raise SystemExit(f"No training row fits within --max-length {args.max_length}.")
    lengths = sorted(sum(map(len, example)) for example in examples)
    print(
        f"{len(examples)} rows, {dropped} over {args.max_length} tokens dropped; "
        f"tokens median {lengths[len(lengths) // 2]}, max {lengths[-1]}, total {sum(lengths)}",
        flush=True,
    )

    packed = args.attention in PACKED_ATTENTION
    model = load_model(args, args.attention)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loss_of = ReplyLoss(model, packed, pad_id)
    parameters = sum(p.numel() for p in model.parameters())

    # The same rows for every setting, in groups of eight so that packed examples attending
    # to each other would show up as a different loss.
    probe = sorted(range(len(examples)), key=ids.__getitem__)[:PROBE_ROWS]
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        probe_sum = sum(
            loss_of(examples, probe[i : i + 8])[0].item() for i in range(0, len(probe), 8)
        )
    probe_loss = probe_sum / sum(len(examples[i][1]) for i in probe)
    model.train()

    epochs = math.ceil(args.epochs)
    plan = []
    for epoch in range(epochs):
        steps = optimizer_steps(micro_batches(examples, args.max_tokens, packed, rng), args.rows_per_step)
        if epoch + 1 > args.epochs:
            steps = steps[: round(len(steps) * (args.epochs - epoch))]
        plan.extend(steps)
    if args.benchmark:
        plan = plan[: args.benchmark]
    warmup = max(1, round(0.03 * len(plan)))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup
        if step < warmup
        else 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, len(plan) - warmup))),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    start, seen_tokens, recent, timed = time.time(), 0, [], None
    for number, step in enumerate(plan, 1):
        if args.benchmark and number == 3:
            torch.cuda.synchronize()
            timed = (time.time(), seen_tokens)
        targets = sum(len(examples[i][1]) for batch in step for i in batch)
        total = 0.0
        for batch in step:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, tokens = loss_of(examples, batch)
            (loss / targets).backward()
            total += loss.item()
            seen_tokens += tokens
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        recent.append(total / targets)
        if not args.benchmark and (number % 20 == 0 or number == len(plan)):
            elapsed = time.time() - start
            rate = seen_tokens / elapsed
            peak = peak_flops()
            utilisation = (4 + 2 * args.checkpointing) * parameters * rate / peak if peak else 0
            print(
                f"step {number}/{len(plan)} loss {sum(recent) / len(recent):.4f} "
                f"lr {scheduler.get_last_lr()[0]:.2e} {rate:.0f} tok/s ~{utilisation:.0%} of peak "
                f"eta {elapsed / number * (len(plan) - number) / 60:.1f} min",
                flush=True,
            )
            recent = []
    if args.benchmark:
        torch.cuda.synchronize()
        seconds, tokens = time.time() - timed[0], seen_tokens - timed[1]
        print("BENCH " + json.dumps({
            "tokens_per_second": round(tokens / seconds), "probe_loss": probe_loss,
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        }), flush=True)
        return
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    (args.output / "training.json").write_text(json.dumps({
        **{key: str(value) for key, value in vars(args).items()},
        "rows": len(examples), "dropped": dropped, "optimizer_steps": len(plan),
        "train_tokens": seen_tokens, "seconds": time.time() - start, "probe_loss": probe_loss,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }, indent=2))
    print(f"saved {args.output / 'adapter'}", flush=True)
    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output / "merged")
        tokenizer.save_pretrained(args.output / "merged")
        print(f"saved {args.output / 'merged'}", flush=True)


if __name__ == "__main__":
    main()
