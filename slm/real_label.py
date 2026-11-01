"""Grow the real test set: label mined follow-ups with a strong model, get a second opinion,
and leave the disagreements for a human.

Real follow-ups (teach_data.py inputs: the user's next message after diagram code in a
public chat) are the only genuinely real requests we have. One model reads the diagram code
and the message and writes the category, the canvas and the edit script (teach.PROMPT). A
second model sees only what the student sees (the canvas and the message) and writes its
own script. When both make the same edit (real_match.relaxed) the case is accepted with both
outcomes; otherwise it is written out for review. Closed models label the test set only;
nothing here feeds training.

    python real_label.py label --input ../runs/colab/teach-input.jsonl \\
        --exclude ../runs/colab/real-test-v2.jsonl --cli agy --model gemini-3.8-flash-high \\
        --output ../runs/colab/real-v3/flash.jsonl --batch 4
    python real_label.py second --labels ../runs/colab/real-v3/flash.jsonl \\
        --cli claude --model claude-opus-5-5 --output ../runs/colab/real-v3/opus.jsonl
    python real_label.py build --labels ../runs/colab/real-v3/flash.jsonl \\
        --second ../runs/colab/real-v3/opus.jsonl --output ../runs/colab/real-v3/
"""

import argparse
import json
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cli_eval import ask_agy, ask_claude
from contract import messages, run_script, semantic_state, session_from
from real_match import relaxed
from teach import PROMPT, check, message, parse
from teach_data import GUIDE, json_reply

LABEL_PROMPT = PROMPT + (
    "\nAlways give the canvas, for every category, so refusals can be tested on the same "
    "diagram; give a script only for atomic and compound edits (null otherwise)."
)
REFUSED = ("unsupported_edit", "other", "new_diagram", "fix_error")
QUOTA = re.compile(r"RESOURCE_EXHAUSTED|quota|credits|429|usage limit|rate limit|limit reached", re.I)


OTHER_SCRIPTS = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]")


def english(row):
    """English, or unlabelled text that at least is not written in another script."""
    return row.get("language") in ("English", None, "unknown") and not OTHER_SCRIPTS.search(row["text"])


def calls(ask, model, guide, prompts, ending, batch, workers, timeout, wait, tries, on_answer):
    """Ask for every prompt, several per call; `on_answer(prompt, text)` runs as each finishes,
    under a lock, so a long run keeps what it has when it is interrupted."""
    lock, failures = threading.Lock(), []
    done = [0]

    def call(text, tail):
        for _ in range(tries):
            try:
                with tempfile.TemporaryDirectory() as empty:
                    return ask(model, guide, text, timeout, empty, tail)[0]
            except RuntimeError as error:
                if not QUOTA.search(str(error)):
                    raise
                time.sleep(wait)
        raise RuntimeError("the quota stayed exhausted")

    def work(group):
        try:
            if len(group) == 1:
                answers = [call(group[0], ending)]
            else:
                joined = "\n\n".join(f"### Input {i}\n{p}" for i, p in enumerate(group, 1))
                text = call(joined, f"Answer each of the {len(group)} inputs on its own. Reply with JSON only: "
                                    f'{{"answers": [the JSON object for input 1, ..., for input {len(group)}]}}.')
                reply = json.loads(text[text.find("{"): text.rfind("}") + 1])
                answers = [json.dumps(a, ensure_ascii=False) for a in reply["answers"]]
                if len(answers) != len(group):
                    raise ValueError(f"{len(answers)} answers for {len(group)} inputs")
        except (RuntimeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            with lock:
                failures.append(str(error)[:200])
            return
        with lock:
            for prompt, answer in zip(group, answers):
                on_answer(prompt, answer)
            done[0] += len(group)
            if done[0] % 30 < len(group):
                print(f"{done[0]}/{len(prompts)} answered, {len(failures)} calls failed", flush=True)

    groups = [prompts[i: i + batch] for i in range(0, len(prompts), batch)]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, groups))
    return failures


def label(args):
    rows = [json.loads(line) for line in open(args.input)]
    exclude = set()
    for path in args.exclude:
        exclude |= {json.loads(line)["id"] for line in open(path)}
    todo = [r for r in rows if english(r) and r["id"] not in exclude]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in open(args.output)}
    todo = [r for r in todo if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    by_prompt = {message(r): r for r in todo}
    ask = {"agy": ask_agy, "claude": ask_claude}[args.cli]
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a") as out:
        def keep(prompt, text):
            row = by_prompt[prompt]
            record = {**row, "labeller": args.model}
            try:
                answer = parse(text)
                record.update(answer=answer, **check(row, answer))
                counts[answer["category"]] += 1
                counts["applied"] += bool(record.get("applied"))
            except (ValueError, json.JSONDecodeError) as error:
                record.update(answer=None, parse_error=str(error)[:300], raw=text[:2000])
                counts["unparsed"] += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

        failures = calls(ask, args.model, LABEL_PROMPT, list(by_prompt), "Reply with the JSON object only.",
                         args.batch, args.workers, args.timeout, args.quota_wait, args.quota_tries, keep)
    print(json.dumps({"asked": len(todo), "failed": len(failures), "examples": failures[:3], **counts}))


def student_view(canvas, command):
    return messages(command, session_from(canvas))[1]["content"]


def second(args):
    """A second model writes the script from the canvas and message alone."""
    labels = [json.loads(line) for line in open(args.labels)]
    todo = [r for r in labels if r.get("applied")]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in open(args.output)}
    todo = [r for r in todo if r["id"] not in done]
    by_prompt = {student_view(r["answer"]["canvas"], r["text"]): r for r in todo}
    ask = {"agy": ask_agy, "claude": ask_claude}[args.cli]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a") as out:
        def keep(prompt, text):
            out.write(json.dumps({"id": by_prompt[prompt]["id"], "model": args.model, "reply": text}, ensure_ascii=False) + "\n")
            out.flush()

        failures = calls(ask, args.model, GUIDE, list(by_prompt), "Reply with the JSON script only.",
                         args.batch, args.workers, args.timeout, args.quota_wait, args.quota_tries, keep)
    print(json.dumps({"asked": len(todo), "failed": len(failures), "examples": failures[:3]}))


def apply_reply(canvas, command, text):
    session = session_from(canvas)
    try:
        refused = run_script(json_reply(text), session, command, "new")
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        return None, f"invalid: {str(error)[:120]}"
    return semantic_state(session), refused or "applied"


def build(args):
    """Accepted cases, refusal cases, and a review file for the disagreements."""
    labels = [json.loads(line) for line in open(args.labels)]
    seconds = {}
    for path in args.second:
        for line in open(path):
            item = json.loads(line)
            seconds.setdefault(item["id"], []).append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    counts, accepted, review = Counter(), [], []
    for row in labels:
        answer = row.get("answer")
        if not answer or not answer.get("canvas"):
            counts["no_canvas"] += 1
            continue
        case = {"id": row["id"], "language": row.get("language") or "unknown", "kinds": row.get("kinds"),
                "command": row["text"], "canvas": answer["canvas"], "category": answer["category"],
                "edit_kind": answer.get("edit_kind"), "source": f"real follow-up; labelled by {row['labeller']}",
                "labeller_reason": answer.get("reason")}
        if answer["category"] in REFUSED:
            try:
                before = semantic_state(session_from(answer["canvas"]))
            except (KeyError, TypeError, ValueError, AttributeError):
                counts["bad_canvas"] += 1
                continue
            case.update(after=before, accepted=[], script={"actions": [], "reason": answer["category"]},
                        expect="refusal")
            accepted.append(case)
            counts["refusal"] += 1
            continue
        if not row.get("applied"):
            counts["edit_not_applied:" + str(answer.get("edit_kind"))] += 1
            continue
        case.update(script=answer["script"], after=row["after"], accepted=[], expect="edit")
        opinions = seconds.get(row["id"], [])
        if not opinions:
            counts["no_second"] += 1
            review.append({**case, "second": None, "why": "no second opinion"})
            continue
        states, verdicts = [], []
        for opinion in opinions:
            state, outcome = apply_reply(answer["canvas"], row["text"], opinion["reply"])
            agree = state is not None and relaxed(row["before"], state, row["after"], row["text"])
            states.append(state)
            verdicts.append({"model": opinion["model"], "outcome": outcome, "agree": agree, "reply": opinion["reply"][:1500]})
        if all(v["agree"] for v in verdicts):
            case["accepted"] = [s for s in states if s != row["after"]]
            accepted.append(case)
            counts["agreed"] += 1
        else:
            review.append({**case, "second": verdicts, "why": "second opinion made a different edit"})
            counts["disagreed"] += 1
    with open(args.output / "accepted.jsonl", "w") as out:
        for case in accepted:
            out.write(json.dumps(case, ensure_ascii=False) + "\n")
    with open(args.output / "review.jsonl", "w") as out:
        for case in review:
            out.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(counts))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    def cli_options(command):
        command.add_argument("--cli", choices=("agy", "claude"), required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--batch", type=int, default=1)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--timeout", type=int, default=600)
        command.add_argument("--quota-wait", type=int, default=600)
        command.add_argument("--quota-tries", type=int, default=36)
        command.add_argument("--limit", type=int, default=0)

    labelling = commands.add_parser("label")
    labelling.add_argument("--input", type=Path, required=True)
    labelling.add_argument("--exclude", type=Path, nargs="*", default=[])
    cli_options(labelling)
    opinion = commands.add_parser("second")
    opinion.add_argument("--labels", type=Path, required=True)
    cli_options(opinion)
    building = commands.add_parser("build")
    building.add_argument("--labels", type=Path, required=True)
    building.add_argument("--second", type=Path, nargs="*", default=[])
    building.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"label": label, "second": second, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
