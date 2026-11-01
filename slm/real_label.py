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

The second opinion can also come from a distil labs teacher (an open model, independent of
the labeller): `export-second` writes the dataset folder, `distil teacher-evaluation
create-from-dataset --config <folder>/<teacher>.yaml <dataset-id>` runs it, and
`from-distil` turns the downloaded predictions into the file `build` reads.
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
from distil_synth import zero_based
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


def viewable(rows):
    """Rows whose canvas the app can hold (it caps shapes at 60); the rest are no test cases."""
    kept = []
    for row in rows:
        try:
            student_view(row["answer"]["canvas"], row["text"])
            kept.append(row)
        except Exception:  # noqa: BLE001  pydantic limits, bad ids
            pass
    return kept


def second(args):
    """A second model writes the script from the canvas and message alone."""
    labels = [json.loads(line) for line in open(args.labels)]
    todo = viewable([r for r in labels if r.get("applied")])
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


ADDITIVE_GUIDE = """You help build a test for a diagram-editing assistant. The input is a canvas and a \
user's message that asks for more content than it spells out ("add more detail", "expand X", \
"add the steps I mentioned"). Such a request has many right answers, so the test only checks \
the shape of the edit. Decide:
- kind: "add" when a right answer adds objects (and arrows) and removes nothing; "restyle" when \
it only changes colours, fills or text of existing objects; "skip" when no canvas edit could \
satisfy it (it needs information that is not in the message or the canvas, or restructures \
the whole diagram).
- anchors: for "add", the names (exactly as on the canvas) of existing objects the new content \
should be connected to; [] when it may attach anywhere.
- min_new: for "add", the smallest number of new objects a serious answer would add (1 to 6).
Reply with JSON only: {"kind": ..., "anchors": [...], "min_new": n, "why": "one sentence"}."""


def additive(args):
    """Opus proposes the pass criteria for high-level requests: anchors and a minimum of new objects."""
    labels = [json.loads(line) for line in open(args.labels)]
    todo = viewable([r for r in labels if (r.get("answer") or {}).get("edit_kind") == "high_level" and r["answer"].get("canvas")])
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in open(args.output)}
    todo = [r for r in todo if r["id"] not in done]
    by_prompt = {student_view(r["answer"]["canvas"], r["text"]): r for r in todo}
    ask = {"agy": ask_agy, "claude": ask_claude}[args.cli]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "a") as out:
        def keep(prompt, text):
            row = by_prompt[prompt]
            try:
                answer = json.loads(text[text.find("{"): text.rfind("}") + 1])
            except (ValueError, json.JSONDecodeError):
                answer = {"kind": "skip", "why": "unparsed: " + text[:200]}
            out.write(json.dumps({"id": row["id"], "model": args.model, **answer}, ensure_ascii=False) + "\n")
            out.flush()

        failures = calls(ask, args.model, ADDITIVE_GUIDE, list(by_prompt), "Reply with the JSON object only.",
                         args.batch, args.workers, args.timeout, args.quota_wait, args.quota_tries, keep)
    print(json.dumps({"asked": len(todo), "failed": len(failures), "examples": failures[:3]}))


def export_second(args):
    """A distil labs dataset of the applied labels, in the student's view, for an open teacher."""
    from distil_export import JUDGE_INSTRUCTIONS, TEACHERS, config

    labels = viewable([json.loads(line) for line in open(args.labels) if json.loads(line).get("applied")])
    dataset = args.output / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    with open(dataset / "test.jsonl", "w") as test, open(args.output / "index.jsonl", "w") as index:
        for row in labels:
            user = student_view(row["answer"]["canvas"], row["text"])
            test.write(json.dumps({"messages": [{"role": "user", "content": user},
                                                {"role": "assistant", "content": json.dumps(row["answer"]["script"])}]},
                                  ensure_ascii=False) + "\n")
            index.write(json.dumps({"id": row["id"], "user": user}, ensure_ascii=False) + "\n")
    json.dump({"task_description": GUIDE, "llm_as_a_judge_instructions": JUDGE_INSTRUCTIONS},
              open(dataset / "job_description.json", "w"), indent=2)
    for name, teacher in TEACHERS.items():
        (args.output / f"{name}.yaml").write_text(config(teacher))
    (dataset / "config.yaml").write_text(config(TEACHERS["deepseek-v4-pro"]))
    print(json.dumps({"rows": len(labels)}))


def from_distil(args):
    """Downloaded teacher predictions as second opinions keyed by case id."""
    index = {json.loads(line)["user"]: json.loads(line)["id"] for line in open(args.index)}
    count = 0
    with open(args.output, "w") as out:
        for line in open(args.predictions):
            item = json.loads(line)
            prompt = item["prompt"]
            if isinstance(prompt, list):
                prompt = next(m["content"] for m in reversed(prompt) if m["role"] == "user")
            identifier = index.get(prompt) or next((i for u, i in index.items() if u in prompt), None)
            if identifier is None:
                continue
            reply = item["prediction"]
            text = reply.get("content") or "" if isinstance(reply, dict) else str(reply)
            out.write(json.dumps({"id": identifier, "model": args.model, "reply": text}, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"converted": count}))


def apply_reply(canvas, command, text):
    session = session_from(canvas)
    try:
        refused = run_script(zero_based(json_reply(text)), session, command, "new")
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
    criteria = commands.add_parser("additive")
    criteria.add_argument("--labels", type=Path, required=True)
    cli_options(criteria)
    exporting = commands.add_parser("export-second")
    exporting.add_argument("--labels", type=Path, required=True)
    exporting.add_argument("--output", type=Path, required=True)
    converting = commands.add_parser("from-distil")
    converting.add_argument("--index", type=Path, required=True)
    converting.add_argument("--predictions", type=Path, required=True)
    converting.add_argument("--model", required=True)
    converting.add_argument("--output", type=Path, required=True)
    building = commands.add_parser("build")
    building.add_argument("--labels", type=Path, required=True)
    building.add_argument("--second", type=Path, nargs="*", default=[])
    building.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"label": label, "second": second, "additive": additive, "export-second": export_second,
     "from-distil": from_distil, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
