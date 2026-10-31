"""Script-model data from the frozen v7 bundle: converted rows plus compound requests.

Writes train.jsonl (rendered messages and replies), dev-compounds.jsonl (held-out
compounds built on validation canvases) and a summary. Validation commands and sessions
are scored from the bundle itself by evaluate_scripts.py.

Rows that are session turns are converted and rendered on the replayed session, which
keeps the links and history a row's canvas alone does not carry.

With --teacher (teach_data.py output), verified paraphrases join training on their
source row's canvas, and verified real-style requests join it on their own canvases,
except those from a tenth of the diagrams, kept as dev-requests.jsonl.

    python prepare_scripts.py --data /content/data/v7 --output /content/runs/scripts-data
"""

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from contract import messages, script_text, session_from
from convert import session_for, to_script
from sessions import CanvasSession
from synth import compound


def replayed(path, splits):
    """The session each session turn starts from, by turn ID."""
    states = {}
    for line in open(path):
        case = json.loads(line)
        if case["split"] not in splits:
            continue
        session = CanvasSession(case["initial_canvas"])
        for turn, step in enumerate(case["turns"]):
            session.external(step["before"])
            states[step["id"]] = copy.deepcopy(session)
            session.execute(step["command"], step["expected"], f"{case['id']}:created-{turn}")
    return states


def teacher_rows(folder, train, live):
    """Verified paraphrases as training rows, and verified requests split into train and dev."""
    by_id = {row["id"]: row for row in train}
    paraphrases, counts = [], Counter()
    for line in open(folder / "paraphrases.jsonl"):
        item = json.loads(line)
        base = by_id.get(item["source"])
        if not item["verified"] or base is None:
            continue
        counts[item["source"]] += 1
        row = {**base, "id": f"{item['source']}:p{counts[item['source']]}", "command": item["command"],
               "source": item["source"]}
        # The new wording may point at targets differently ("this" for a name); both resolve alike.
        paraphrases.append({**row, "script": to_script(row, live.get(item["source"])) or base["script"]})
    requests = {"train": [], "dev": []}
    for line in open(folder / "requests.jsonl"):
        item = json.loads(line)
        if item["verified"]:
            held = int(hashlib.sha1(item["source"].encode()).hexdigest(), 16) % 10 == 0
            requests["dev" if held else "train"].append(item)
    return paraphrases, requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compounds", type=int, default=9000)
    parser.add_argument("--dev-compounds", type=int, default=400)
    parser.add_argument("--teacher", type=Path, help="teach_data.py output")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    converted, counts = {"train": [], "valid": []}, Counter()
    live = replayed(args.data / "sessions.jsonl", converted)
    for line in open(args.data / "examples.jsonl"):
        row = json.loads(line)
        if row["split"] not in converted:
            continue
        script = to_script(row, live.get(row["id"]))
        outcome = "converted" if script else "dropped"
        counts[f"{row['split']}:{outcome}"] += 1
        counts[f"{row['split']}:{row['expected']['name']}:{outcome}"] += 1
        if script:
            converted[row["split"]].append({**row, "script": script})

    def compounds(rows, count):
        made = []
        for row in rng.sample(rows, len(rows)):
            if len(made) == count:
                break
            if result := compound(row, rng):
                made.append(result)
        return made

    made = compounds(converted["train"], args.compounds)
    paraphrases, requests = (teacher_rows(args.teacher, converted["train"], live) if args.teacher
                             else ([], {"train": [], "dev": []}))
    train = converted["train"] + made + paraphrases
    dev = compounds(converted["valid"], args.dev_compounds)
    rendered = []
    for row in train:
        source = row.get("source", row["id"])
        session = live[source] if source in live else session_for(row)
        rendered.append({"id": row["id"], "messages": messages(row["command"], session),
                         "reply": script_text(row["script"])})
    for row in requests["train"]:
        rendered.append({"id": row["id"], "messages": messages(row["command"], session_from(row["canvas"])),
                         "reply": script_text(row["script"])})
    rng.shuffle(rendered)
    with open(args.output / "train.jsonl", "w") as out:
        for row in rendered:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name, rows in (("dev-compounds", dev), ("dev-requests", requests["dev"])):
        with open(args.output / f"{name}.jsonl", "w") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts.update(train_rows=len(rendered), train_compounds=len(made), train_paraphrases=len(paraphrases),
                  train_requests=len(requests["train"]), dev_compounds=len(dev),
                  dev_requests=len(requests["dev"]))
    summary = {"counts": dict(counts),
               "actions_per_train_row": dict(Counter(
                   len(r["script"]["actions"]) for r in train + requests["train"]))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
