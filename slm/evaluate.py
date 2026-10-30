"""Exact-action evaluation on independent commands and closed-loop sessions.

Scoring follows lab.evaluate and lab.evaluate_sessions: a prediction is correct when
the validated action equals the label, and session turns act on the model's own
simulated canvas. Session turns are batched across sessions, one turn at a time.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import messages, predict
from sessions import CanvasSession, compare_session_state, summarize_sessions
from train import end_marker, prompt_ids, read_rows


def load(model_name, adapter):
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompts, max_new_tokens=192, budget=65536):
    stop = sorted(
        {
            tokenizer.convert_tokens_to_ids(end_marker(tokenizer)),
            *([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []),
        }
    )
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop[0]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    texts = [None] * len(prompts)
    position = 0
    while position < len(order):
        batch = [order[position]]
        while (
            position + len(batch) < len(order)
            and len(batch) < 96
            and (len(prompts[order[position + len(batch)]]) + max_new_tokens)
            * (len(batch) + 1)
            <= budget
        ):
            batch.append(order[position + len(batch)])
        position += len(batch)
        width = max(len(prompts[i]) for i in batch)
        ids = torch.full((len(batch), width), pad)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, index in enumerate(batch):
            ids[row, width - len(prompts[index]) :] = torch.tensor(prompts[index])
            mask[row, width - len(prompts[index]) :] = 1
        output = model.generate(
            input_ids=ids.cuda(),
            attention_mask=mask.cuda(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            eos_token_id=stop,
            pad_token_id=pad,
        )
        for row, index in enumerate(batch):
            tokens = output[row, width:].tolist()
            ends = [tokens.index(token) for token in stop if token in tokens]
            texts[index] = tokenizer.decode(
                tokens[: min(ends)] if ends else tokens, skip_special_tokens=True
            )
    return texts


def score_commands(model, tokenizer, rows):
    texts = generate(
        model,
        tokenizer,
        [
            prompt_ids(
                tokenizer, messages(row["command"], row["canvas"], row.get("history"))
            )
            for row in rows
        ],
    )
    results = []
    for row, text in zip(rows, texts, strict=True):
        result = {"id": row["id"], "raw_output": text, "expected": row["expected"]}
        for mode, guarded in (("raw", False), ("guarded", True)):
            outcome = predict(
                text, row["command"], row["canvas"], row.get("history"), guarded
            )
            result[mode] = {**outcome, "correct": outcome["prediction"] == row["expected"]}
        result["stratum"] = row.get("stratum") or row["expected"]["name"]
        results.append(result)
    summary = {}
    for mode in ("raw", "guarded"):
        strata = Counter(r["stratum"] for r in results)
        right = Counter(r["stratum"] for r in results if r[mode]["correct"])
        summary[mode] = {
            "total": len(results),
            "correct": sum(r[mode]["correct"] for r in results),
            "exact_action_accuracy": sum(r[mode]["correct"] for r in results)
            / len(results),
            "valid_call_rate": sum(r[mode]["prediction"] is not None for r in results)
            / len(results),
            "per_stratum": {
                key: f"{right[key]}/{strata[key]}" for key in sorted(strata)
            },
        }
    return summary, results


def score_sessions(model, tokenizer, cases, guarded):
    live = [
        (case, CanvasSession(case["initial_canvas"]), CanvasSession(case["initial_canvas"]))
        for case in cases
    ]
    rows = []
    for turn in range(max(len(case["turns"]) for case in cases)):
        active = [entry for entry in live if turn < len(entry[0]["turns"])]
        for case, actual, oracle in active:
            actual.external(case["turns"][turn]["before"])
            oracle.external(case["turns"][turn]["before"])
        texts = generate(
            model,
            tokenizer,
            [
                prompt_ids(
                    tokenizer,
                    messages(case["turns"][turn]["command"], actual.canvas, actual.history),
                )
                for case, actual, _ in active
            ],
        )
        for (case, actual, oracle), text in zip(active, texts, strict=True):
            step = case["turns"][turn]
            before = actual.snapshot()
            matched_before = before == oracle.snapshot()
            result = predict(text, step["command"], actual.canvas, actual.history, guarded)
            created_id = f"{case['id']}:created-{turn}"
            try:
                actual.execute(step["command"], result["prediction"], created_id)
            except ValueError as error:
                result.update(prediction=None, error=str(error))
                actual.execute(step["command"], None, created_id)
            oracle.execute(step["command"], step["expected"], created_id)
            actual_state, oracle_state = actual.snapshot(), oracle.snapshot()
            rows.append(
                {
                    **result,
                    "raw_output": text,
                    "id": step["id"],
                    "session_id": case["id"],
                    "turn": turn,
                    "command": step["command"],
                    "expected": step["expected"],
                    "correct": result["prediction"] == step["expected"],
                    "mutated": before != actual_state,
                    "state_matched_before": matched_before,
                    **compare_session_state(actual_state, oracle_state),
                }
            )
    summary = summarize_sessions(rows)
    matched = [row for row in rows if row["state_matched_before"]]
    summary["accuracy_when_state_matched"] = (
        sum(row["correct"] for row in matched) / len(matched) if matched else None
    )
    summary["turns_with_state_matched"] = len(matched)
    firsts = [turn for turn in summary["first_error_turns"].values() if turn]
    summary["mean_first_error_turn"] = sum(firsts) / len(firsts) if firsts else None
    return summary, rows


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--limit", type=positive_integer, help="score only this many commands")
    parser.add_argument(
        "--session-limit", type=positive_integer, help="score only this many sessions"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load(args.model, args.adapter)
    rows = read_rows(args.data / "examples.jsonl", args.split)
    cases = read_rows(args.data / "sessions.jsonl", args.split)
    subsets = {f"{args.split}_all": (rows, cases)}
    if args.split == "valid":
        receipt = json.loads((args.data / "selection-receipt.json").read_text())
        subsets["valid_receipt"] = (
            [row for row in rows if row["id"] in set(receipt["example_ids"])],
            [case for case in cases if case["id"] in set(receipt["session_ids"])],
        )
    report = {"model": args.model, "adapter": args.adapter, "split": args.split}
    details = {}
    for name, (subset_rows, subset_cases) in subsets.items():
        if args.limit:
            subset_rows = subset_rows[: args.limit]
        if args.session_limit:
            subset_cases = subset_cases[: args.session_limit]
        start = time.time()
        commands, command_rows = score_commands(model, tokenizer, subset_rows)
        report[name] = {"commands": commands}
        details[name] = {"commands": command_rows}
        for mode, guarded in (("raw", False), ("guarded", True)):
            summary, session_rows = score_sessions(model, tokenizer, subset_cases, guarded)
            report[name][f"sessions_{mode}"] = summary
            details[name][f"sessions_{mode}"] = session_rows
        report[name]["seconds"] = time.time() - start
        print(
            name,
            json.dumps(
                {
                    "commands": {m: commands[m]["exact_action_accuracy"] for m in commands},
                    **{
                        f"sessions_{m}": {
                            key: report[name][f"sessions_{m}"][key]
                            for key in (
                                "exact_action_accuracy",
                                "state_agreement_rate",
                                "accuracy_when_state_matched",
                                "mean_first_error_turn",
                            )
                        }
                        for m in ("raw", "guarded")
                    },
                }
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_suffix(".rows.json").write_text(json.dumps(details))


if __name__ == "__main__":
    main()
