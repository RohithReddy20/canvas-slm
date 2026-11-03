"""The teacher's side of the precise rows: realistic system designs, and instructions
written the way a diagram's author types them.

precise_synth.py keeps every script right by construction, but its diagrams and phrasing
come from templates. Here a teacher (Opus through `claude -p`; see
closed-model-outputs policy: these rows carry "author") writes the realistic part only:

    python precise_rewrite.py designs --output ../runs/colab/designs.jsonl --per-concept 3
        system designs for 30 well-known concepts (URL shortener, ride hailing, ...),
        each checked to rebuild on the canvas;
    python precise_rewrite.py rewrite --input ../runs/colab/precise-designs.jsonl \\
        --output ../runs/colab/precise-designs.opus.jsonl --batch 10
        each generated instruction reworded by the teacher, same meaning and names; the
        script stays the generated one, and a rewrite is kept only when it still names
        every object the script touches.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from cli_eval import ask_claude
from distil_synth import session_of
from extensions import COMPONENTS
from precise_synth import Canvas, applied
from real_label import calls

MODEL = "claude-opus-5-5"
CONCEPTS = [
    "URL shortener", "chat and messaging app", "ride hailing service", "food delivery platform",
    "e-commerce checkout", "video streaming service", "social media feed", "payment processing system",
    "web search engine", "notification system", "API rate limiting service", "cloud file storage like Dropbox",
    "online code judge", "event ticket booking", "hotel reservation system", "news aggregator",
    "transactional email service", "clickstream analytics pipeline", "IoT device telemetry platform",
    "multiplayer game backend", "CI/CD pipeline", "ad click aggregation", "recommendation system",
    "fraud detection pipeline", "warehouse inventory management", "image upload and CDN delivery",
    "web crawler", "stock trading platform", "healthcare appointment booking", "online learning platform",
]
DESIGN_GUIDE = f"""You draw system-design diagrams for an editable canvas. A diagram is JSON:
{{"boxes": [], "shapes": [{{"id": "c1", "kind": "component:<icon>", "text": "<label>", "x": <int>, "y": <int>}}, ...],
 "connections": [{{"from": "c1", "to": "c2", "label": "<short verb or protocol, or empty>"}}, ...]}}
Icons: {", ".join(COMPONENTS)}. Every label is unique, short (1-3 words) and concrete for the
system (e.g. "Ride Matching Service", "Trips DB", not "Service 1"). Lay components out left to
right along the request flow: x in steps of 240 from 80, y in steps of 180 from 120, no two
components at the same spot. Arrows go in the direction of calls or data flow; no duplicate
arrows, no self-arrows. Reply with JSON only."""
DESIGN_ASK = """Draw a {size} system design for: {concept}.
Reply with {{"diagram": <the diagram>}} only."""
SIZES = ["minimal (4-5 components)", "typical (6-8 components)", "detailed (9-12 components)"]
REWRITE_GUIDE = """You are the author of a system-design diagram, typing short instructions to an assistant
that edits the diagram exactly as told. You will be given the diagram's component names and
an instruction written from a template. Rewrite the instruction the way you would really type
it: your own words and tone (terse, chatty, imperative, with or without punctuation), but the
SAME meaning, no more and no less. Keep every component name spelled exactly as given (quotes
are fine), keep any new component's name and icon type as given, keep the same direction of
arrows and the same labels. Do not add explanations, questions or extra requests. Reply with
JSON only: {"instruction": "..."}."""


def designs(args):
    prompts = [DESIGN_ASK.format(size=size, concept=concept)
               for concept in CONCEPTS[: args.limit or None] for size in SIZES[: args.per_concept]]
    kept, counts = [], Counter()

    def keep(prompt, text):
        try:
            diagram = json.loads(text[text.find("{"): text.rfind("}") + 1])["diagram"]
            diagram = {k: diagram.get(k, []) for k in ("boxes", "shapes", "connections")}
            canvas = Canvas(diagram)
            kinds = {s["kind"].split(":", 1)[1] for s in diagram["shapes"]}
            assert len(canvas.nodes) >= 3 and kinds <= set(COMPONENTS)
            assert len(canvas.nodes) == len(diagram["shapes"]), "labels are not unique"
            assert canvas.edges, "no arrows"
        except (ValueError, KeyError, TypeError, AssertionError, AttributeError) as error:
            counts[f"dropped: {str(error)[:40]}"] += 1
            return
        concept = re.search(r"for: (.*)\.\nReply", prompt).group(1)
        kept.append({"concept": concept, "diagram": diagram, "author": MODEL})
        counts["kept"] += 1

    failures = calls(ask_claude, args.model, DESIGN_GUIDE, prompts, "", 1, args.workers, args.timeout,
                     args.quota_wait, args.quota_tries, keep)
    with open(args.output, "w") as out:
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"designs": len(kept), "failed_calls": len(failures), **counts}, indent=1))


def names_in(script):
    """Every name a script resolves or creates; a rewrite must still say them all."""
    found = set()

    def walk(value):
        if isinstance(value, dict):
            if "name" in value and isinstance(value["name"], str) and len(value) == 1:
                found.add(value["name"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for action in script["actions"]:
        args = action["arguments"]
        if action["name"] == "create_component":
            found.add(args["label"])
        elif action["name"] == "create_schema_box":
            found.add(args["name"])
        elif action["name"] in ("create_shape", "set_text"):
            found.add(args["text"])
        elif action["name"] == "rename_schema":
            found.add(args["new_name"])
        elif action["name"] == "connect_schemas" and args.get("label"):
            found.add(args["label"])
        for key in ("schema_id", "shape_id", "source_id", "target_id", "near"):
            walk(args.get(key))
        walk(args.get("shape_ids"))
    return {name for name in found if name}


def rewrite(args):
    rows = [json.loads(line) for line in open(args.input)]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["command"] for line in open(args.output)}
    rng = random.Random(args.seed)
    if args.limit:
        rows = rng.sample(rows, min(args.limit, len(rows)))
    todo = [r for r in rows if r["command"] not in done]
    by_prompt, prompts = {}, []
    for row in todo:
        names = [o["name"] for o in Canvas(row["diagram"]).nodes]
        prompt = f"Components: {', '.join(names)}\nTemplate instruction: {row['command']}"
        if prompt in by_prompt:
            continue
        by_prompt[prompt] = row
        prompts.append(prompt)
    counts = Counter()
    out = open(args.output, "a")

    def keep(prompt, text):
        row = by_prompt[prompt]
        try:
            instruction = json.loads(text[text.find("{"): text.rfind("}") + 1])["instruction"].strip()
        except (ValueError, KeyError, AttributeError):
            counts["unparsed"] += 1
            return
        missing = [n for n in names_in(row["script"]) if n.casefold() not in instruction.casefold()]
        if missing or not instruction or instruction.casefold() == row["command"].casefold():
            counts["dropped_missing_name" if missing else "dropped_unchanged"] += 1
            return
        if applied(row["diagram"], row["script"]) is None:
            counts["dropped_script"] += 1
            return
        out.write(json.dumps({**row, "command": instruction, "template": row["command"],
                              "source": "precise-opus", "author": args.model}, ensure_ascii=False) + "\n")
        out.flush()
        counts["kept"] += 1

    failures = calls(ask_claude, args.model, REWRITE_GUIDE, prompts, "", args.batch, args.workers,
                     args.timeout, args.quota_wait, args.quota_tries, keep)
    print(json.dumps({"asked": len(prompts), "failed_calls": len(failures), **counts}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("designs", designs), ("rewrite", rewrite)):
        command = commands.add_parser(name)
        command.set_defaults(fn=fn)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--model", default=MODEL)
        command.add_argument("--workers", type=int, default=3)
        command.add_argument("--timeout", type=int, default=600)
        command.add_argument("--quota-wait", type=int, default=600)
        command.add_argument("--quota-tries", type=int, default=36)
    commands.choices["designs"].add_argument("--per-concept", type=int, default=3)
    commands.choices["designs"].add_argument("--limit", type=int, default=0, help="concepts, for a smoke test")
    commands.choices["rewrite"].add_argument("--input", type=Path, required=True)
    commands.choices["rewrite"].add_argument("--batch", type=int, default=10)
    commands.choices["rewrite"].add_argument("--limit", type=int, default=0)
    commands.choices["rewrite"].add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
