"""Script-model data from the frozen v7 bundle: converted rows plus compound requests.

Writes train.jsonl (rendered messages and replies), dev-compounds.jsonl (held-out
compounds built on validation canvases) and a summary. Validation commands and sessions
are scored from the bundle itself by evaluate_scripts.py.

Rows that are session turns are converted and rendered on the replayed session, which
keeps the links and history a row's canvas alone does not carry.

    python prepare_scripts.py --data /content/data/v7 --output /content/runs/scripts-data
"""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from contract import messages, script_text
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compounds", type=int, default=9000)
    parser.add_argument("--dev-compounds", type=int, default=400)
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

    train = converted["train"] + compounds(converted["train"], args.compounds)
    dev = compounds(converted["valid"], args.dev_compounds)
    rng.shuffle(train)
    with open(args.output / "train.jsonl", "w") as out:
        for row in train:
            session = live[row["id"]] if row["id"] in live else session_for(row)
            rendered = {"id": row["id"], "messages": messages(row["command"], session),
                        "reply": script_text(row["script"])}
            out.write(json.dumps(rendered, ensure_ascii=False) + "\n")
    with open(args.output / "dev-compounds.jsonl", "w") as out:
        for row in dev:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts.update(train_rows=len(train), train_compounds=len(train) - len(converted["train"]),
                  dev_compounds=len(dev))
    summary = {"counts": dict(counts),
               "actions_per_train_row": dict(Counter(len(r["script"]["actions"]) for r in train))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
