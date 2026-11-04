"""Summarise observation sessions (server.mjs logs) and export every step as a row.

    python3 analyze.py ../runs/observe [--rows ../runs/observe/rows.jsonl]

Per session and overall: how people phrase requests (length, single vs several edits,
pronouns, corrections), which actions and components the agent reaches for, what fails and
why, and the capabilities it wished for. Each act step becomes one row: the canvas before,
the sentence, the actions (short IDs), their results and the canvas after.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PRONOUNS = re.compile(r"\b(it|that|this|those|these|them|they|its)\b", re.I)
CORRECTIONS = re.compile(r"\b(actually|instead|no wait|wait|undo|scratch that|never mind|oops|rather)\b", re.I)
VAGUE = re.compile(r"\b(nicer|cleaner|better|tidy|clean up|organi[sz]e|somewhere|a bit|kind of|stuff)\b", re.I)


def failure_kind(error):
    if error.startswith("Unknown action"):
        return "unknown action"
    if " takes " in error:
        return "wrong arguments"
    if "no longer on this canvas" in error:
        return "missing target"
    if "canvas changed" in error:
        return "stale canvas"
    return error.split(":")[0][:50]


def sessions(root):
    for folder in sorted(p for p in root.iterdir() if (p / "events.jsonl").exists()):
        meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
        events = [json.loads(line) for line in open(folder / "events.jsonl")]
        yield folder.name, meta, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--rows", type=Path)
    args = parser.parse_args()

    groups = {}
    wishes, rows = [], []
    for name, meta, events in sessions(args.root):
        kind = "edit" if meta.get("from") else "build"
        group = groups.setdefault(kind, {key: Counter() for key in ("total", "actions", "components", "failures", "sizes", "refusals")})
        total, actions, components, failures, sizes = (group[k] for k in ("total", "actions", "components", "failures", "sizes"))
        acts = [e for e in events if e["type"] == "act"]
        if not acts:
            continue
        end = next((e["canvas"] for e in reversed(events) if e["type"] == "end"), acts[-1]["after"])
        failed = [e for e in acts if any(not r["ok"] for r in e["results"])]
        words = [len(e["command"].split()) for e in acts]
        print(f"\n## {name}  ({meta.get('model', '?')})  goal: {meta.get('goal', '')}")
        print(f"steps {len(acts)}, failed {len(failed)}, words/sentence {sum(words) / len(words):.1f}, "
              f"final {len(end['boxes'])} boxes / {len(end['shapes'])} shapes / {len(end['links'])} links")
        for e in acts[:: max(1, len(acts) // 6)][:6]:
            print(f"   - {e['command'][:120]}")
        for e in acts:
            n = len(e["actions"])
            sizes["1" if n == 1 else "2" if n == 2 else "3+"] += 1
            total["steps"] += 1
            total["pronoun"] += bool(PRONOUNS.search(e["command"]))
            total["correction"] += bool(CORRECTIONS.search(e["command"]))
            total["vague"] += bool(VAGUE.search(e["command"]))
            total["words"] += len(e["command"].split())
            for action in e["actions"]:
                actions[action.get("name")] += 1
                if action.get("name") == "no_action":
                    group["refusals"][action.get("arguments", {}).get("reason")] += 1
                if action.get("name") == "create_component":
                    components[action.get("arguments", {}).get("component")] += 1
            for result in e["results"]:
                if not result["ok"]:
                    failures[failure_kind(result["error"])] += 1
            rows.append({"session": name, "goal": meta.get("goal"), "model": meta.get("model"), "step": e["step"],
                         "command": e["command"], "before": e["before"], "actions": e["actions"],
                         "results": e["results"], "after": e["after"]})
        total["sessions"] += 1
        total["clicks"] += sum(e["type"] == "user_select" for e in events)
        total["failed_steps"] += len(failed)
        wishes += [(name, e["command"], e["missing"]) for e in events if e["type"] == "wish"]

    for kind, group in groups.items():
        total = group["total"]
        steps = max(1, total["steps"])
        print(f"\n# Overall: {kind} sessions")
        print(f"{total['sessions']} sessions, {total['steps']} steps, {total['failed_steps']} with a failure "
              f"({100 * total['failed_steps'] / steps:.0f}%), {total['words'] / steps:.1f} words per sentence, "
              f"{total['clicks']} clicks (user_select)")
        print(f"actions per sentence: {dict(group['sizes'])}")
        print(f"sentences with a pronoun {100 * total['pronoun'] / steps:.0f}%, a correction {100 * total['correction'] / steps:.0f}%, "
              f"vague wording {100 * total['vague'] / steps:.0f}%")
        print("actions:", dict(group["actions"].most_common()))
        print("refusals:", dict(group["refusals"]))
        print("components:", dict(group["components"].most_common()))
        print("failures:", dict(group["failures"].most_common()))
    print(f"wishes ({len(wishes)}):")
    for session, command, missing in wishes:
        print(f"   - [{session}] {command[:90]}  ->  {missing[:110]}")
    if args.rows:
        with open(args.rows, "w") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(rows)} rows to {args.rows}")


if __name__ == "__main__":
    main()
