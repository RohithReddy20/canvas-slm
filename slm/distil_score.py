"""Score distil labs teacher-evaluation predictions by the canvas each script produces.

Dev rows are correct when the script leaves the canvas exactly as v7's label does, as in
evaluate_scripts.commands; real rows when the diagram content matches the hand-checked
result, as in evaluate_scripts.real. A reply is read leniently (fences, prose and
reasoning around the JSON are dropped), since teacher output is cleaned before training.

    python distil_score.py --bundle ../runs/colab/v7-bundle --index ../runs/distil/eval/index.jsonl \\
        --real ../runs/colab/real-test-v1.jsonl ../runs/distil/eval/predictions/*.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from contract import semantic_state, session_from
from convert import session_for
from evaluate_scripts import apply
from real_match import relaxed
from teach_data import json_reply


def lenient(text):
    try:
        return json.dumps(json_reply(text))
    except ValueError:
        return text


def score(path, index, dev, real):
    predictions = [json.loads(line) for line in open(path)]
    by_user = {entry["user"]: entry for entry in index}
    totals, outcomes, misses = Counter(), Counter(), []
    for prediction in predictions:
        prompt = prediction.get("prompt")
        if isinstance(prompt, list):  # the platform's chat: system, then the user message
            prompt = next(m["content"] for m in reversed(prompt) if m["role"] == "user")
        entry = by_user.get(prompt) or next((e for u, e in by_user.items() if u in prompt), None)
        if entry is None:
            outcomes["unmatched"] += 1
            continue
        reply = prediction.get("prediction", "")
        text = lenient(reply.get("content") or "" if isinstance(reply, dict) else str(reply))
        if entry["kind"] == "dev":
            row = dev[entry["id"]]
            predicted, gold = session_for(row), session_for(row)
            outcome = apply(text, predicted, row["command"], "new")
            if row["expected"]["name"] != "no_action":
                gold.execute(row["command"], row["expected"], "new")
            correct = predicted.snapshot() == gold.snapshot()
        else:
            row = real[entry["id"]]
            predicted = session_from(row["canvas"])
            outcome = apply(text, predicted, row["command"], "new")
            correct = semantic_state(predicted) == row["after"]
            before = semantic_state(session_from(row["canvas"]))
            loose = relaxed(before, semantic_state(predicted), row["after"], row["command"])
            totals["real_relaxed"] += loose
            english = row["language"] in ("English", "unknown")  # the product is English-only
            totals["real_en"] += english
            totals["real_en_relaxed"] += english and loose
        totals[entry["kind"]] += 1
        totals[entry["kind"] + "_correct"] += correct
        outcomes[outcome] += 1
        if not correct:
            misses.append({"kind": entry["kind"], "id": entry["id"], "command": row["command"],
                           "outcome": outcome, "relaxed": entry["kind"] == "real" and loose,
                           "reply": text[:1500]})
    report = {kind: f"{totals[kind + '_correct']}/{totals[kind]}" for kind in ("dev", "real") if totals[kind]}
    if totals["real"]:
        report["real_relaxed"] = f"{totals['real_relaxed']}/{totals['real']}"
        report["real_english_relaxed"] = f"{totals['real_en_relaxed']}/{totals['real_en']}"
    return {"file": Path(path).name, **report, "outcomes": dict(outcomes), "misses": misses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--real", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("predictions", type=Path, nargs="+")
    args = parser.parse_args()

    index = [json.loads(line) for line in open(args.index)]
    ids = {e["id"] for e in index if e["kind"] == "dev"}
    dev = {}
    for line in open(args.bundle / "examples.jsonl"):
        row = json.loads(line)
        if row["id"] in ids:
            dev[row["id"]] = row
    real = {json.loads(line)["id"]: json.loads(line) for line in open(args.real)} if args.real else {}
    reports = [score(path, index, dev, real) for path in args.predictions]
    for report in reports:
        print(json.dumps({k: v for k, v in report.items() if k != "misses"}))
    if args.output:
        json.dump(reports, open(args.output, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
