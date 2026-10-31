"""Teacher-written training data through distil labs' synthetic generation, checked by our code.

The teacher sees seed rows and writes new ones: a realistic diagram plus the next message
its author sends, and the edit script for it. Diagrams use the plain format of the real
follow-ups (boxes, shapes, connections by ID, plus the selection and the last created and
edited objects), which a teacher can write naturally; `session_of` rebuilds the canvas.

    python distil_synth.py export --bundle ../runs/colab/v7-bundle --observed ../runs/observe/rows.jsonl \\
        --output ../runs/distil/synth
    distil dataset create --data ../runs/distil/synth/dataset
    distil dataset generate-synthetic-data-train --smoke <dataset-id>
    distil dataset download -d ../runs/distil/synth/generated <generated-dataset-id>
    python distil_synth.py check --input ../runs/distil/synth/generated/train.jsonl \\
        --output ../runs/distil/synth/checked.jsonl

`check` keeps a row only when its diagram rebuilds and its script runs on it (a refusal
included), dropping scripts with no actions and no reason. Then a second teacher answers the
kept rows on its own, and `agree` keeps those where both replies reach the same result:

    python distil_synth.py second-opinion --input checked.jsonl --output second/
    python cli_eval.py --dataset second/ --cli agy --model gemini-4-argon-medium --output second/gemini.jsonl
    python distil_synth.py agree --input checked.jsonl --predictions second/gemini.jsonl --output agreed.jsonl
"""

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

from contract import parse_script, run_script, script_text, semantic_state, session_from
from convert import session_for, to_script
from synth import compound
from teach_data import GUIDE

CANVAS_PARAGRAPH = GUIDE.split("\n\n")[1]
DIAGRAM_PARAGRAPH = (
    'The input is a diagram as JSON: "boxes" are classes or tables with fields and methods; '
    '"shapes" have a kind and text, where "component:database" and the like are system-design '
    'components (an icon with a label); "connections" are arrows between objects by ID, with a '
    'label; x, y, w, h place each object (right and down are positive) and "parent" is the frame '
    'holding it; "selected" lists the IDs the person has selected; "last_created" and '
    '"last_edited" are the objects created and edited most recently (or null). Colors are given '
    'only when not black. After the diagram comes the message the diagram\'s author sends next.'
)
assert CANVAS_PARAGRAPH.startswith("The canvas is compact JSON")
TASK = GUIDE.replace(CANVAS_PARAGRAPH, DIAGRAM_PARAGRAPH)

GENERATION = """Write inputs as they occur when a person edits their own diagram with a voice \
assistant. The diagram is one people really draw, usually a system design (services, gateways, \
databases, caches, queues, workers as components, often grouped in frames) and also class \
diagrams, database tables, flowcharts, state machines and mind maps. It has 4 to 25 objects with \
realistic, domain-specific names (not placeholders), consistent IDs, sensible positions and \
connections. The message is the next thing its author says, and real people are messy:
- about three in four messages ask for one thing ("add a cache", "make it red", "undo that"); \
the rest combine two or three edits;
- about half point with "it", "this", "these", "that one" or "the one I just added": "these" \
and "this" usually mean what the person has selected, so the diagram's "selected" then lists \
those objects;
- names are often loose or slightly wrong ("the redis thing", "the payments one", "notif \
service"), abbreviated, or garbled by speech-to-text;
- some messages correct or undo ("no wait, the other one", "undo that"), and some are vague \
("tidy this up", "lay it out left to right");
- about one in twelve names something missing or matching several objects, and the script \
still points at it the way the message does; a few ask for what no tool does (explain, export, \
change the theme) and get the refusal.
Every message is in English; names on the diagram are English too."""

MUTATORS = [
    {"name": "request", "values": [
        "the message asks for one edit",
        "the message asks for one edit, short and casual",
        "the message asks for one edit to an object the person selected first, calling it this or these",
        "the message asks for one edit to the object just created or edited, calling it it or that",
        "the message asks for two or three edits in one sentence",
        "the message adds system-design components and wires them in with labelled arrows",
        "the message puts objects in a frame, or moves objects into an existing frame",
        "the message asks to lay out or tidy the diagram or part of it",
        "the message renames, relabels, restyles, reconnects or removes existing objects, naming them loosely",
        "the message corrects or undoes the previous change",
        "the message names an object that is not on the diagram or that matches several objects",
        "the message asks for something no tool does (explain the design, export it, change the theme)",
    ]},
    {"name": "diagram", "values": [  # mostly system designs, as in the observed sessions
        "a system architecture with components", "a system architecture with components",
        "a system architecture with components grouped in frames",
        "a system architecture with components grouped in frames",
        "a data pipeline with queues, workers and storage", "an event-driven backend with components",
        "a class diagram", "database tables", "a flowchart", "a state machine or mind map",
    ]},
    {"name": "voice", "values": [  # English only: the product serves English speakers
        "plain English", "casual English", "English with typos", "English as speech-to-text, with slips",
        "terse English, a few words", "polite, wordy English",
    ]},
]

LABELS = {
    "cache": ["Redis Cache", "Session Cache", "CDN Cache"], "queue": ["Job Queue", "Order Queue", "Event Queue"],
    "database": ["Postgres", "Orders DB", "User DB"], "load_balancer": ["Load Balancer"], "cdn": ["CDN"],
    "monitoring": ["Metrics", "Monitoring"], "search": ["Search Index"], "auth": ["Auth Service"],
    "worker": ["Worker", "Email Worker"], "notification": ["Notifier", "Push Service"],
    "object_storage": ["S3 Bucket", "Blob Storage"], "service": ["Billing Service", "Profile Service"],
}
FRAMES = ["Backend", "Data Layer", "Edge", "Async Pipeline", "Core Services", "Storage", "Clients"]


def diagram_from_observed(canvas):
    """An observation-rig canvas (observe/server.mjs) as a plain diagram."""
    keep = lambda item, keys: {k: item[k] for k in keys if k in item}  # noqa: E731
    return {
        "boxes": [keep(b, ("id", "name", "fields", "methods", "x", "y", "w", "h", "color", "fill", "parent"))
                  for b in canvas["boxes"]],
        "shapes": [keep(s, ("id", "kind", "text", "x", "y", "w", "h", "color", "fill", "parent"))
                   for s in canvas["shapes"]],
        "connections": [{"id": link["id"], "from": link["from"], "to": link["to"], "label": link["label"]}
                        for link in canvas["links"] if link["from"] and link["to"]],
        "selected": list(canvas["selected"]),
        "last_created": canvas["last_created"],
        "last_edited": canvas["last_edited"],
    }


def observed_seed_rows(path, count, rng):
    """Rows for the new tools on observed canvases, with commands and scripts made by code."""
    canvases, seen = [], set()
    for line in open(path):
        canvas = json.loads(line)["before"]
        key = json.dumps(canvas, sort_keys=True)
        if len(canvas["boxes"]) + len(canvas["shapes"]) >= 4 and key not in seen:
            seen.add(key)
            canvases.append(canvas)
    rng.shuffle(canvases)
    kept, kinds = [], Counter()
    for canvas in canvases:
        if len(kept) >= count:
            break
        diagram = diagram_from_observed(canvas)
        top = [o for o in diagram["boxes"] + diagram["shapes"] if not o.get("parent") and o.get("kind") != "frame"]
        name = lambda o: o.get("name") or o.get("text")  # noqa: E731
        named = [o for o in top if name(o)]
        frames = [o for o in diagram["shapes"] if o.get("kind") == "frame" and o.get("text")]
        if len(named) < 3:
            continue
        kind = rng.choice(["component", "frame", "frame_selected", "add_to_frame", "layout"])
        picked = rng.sample(named, 3)
        if kind == "component":
            component = rng.choice(sorted(LABELS))
            label = rng.choice(LABELS[component])
            side = rng.choice(["right", "left", "above", "below"])
            where = {"right": "next to", "left": "left of", "above": "above", "below": "below"}[side]
            command = f"add a {label.lower()} {where} {name(picked[0]).lower()} and connect it"
            script = {"actions": [
                {"name": "create_component", "arguments": {"component": component, "label": label,
                                                           "near": {"name": name(picked[0])}, "side": side}},
                {"name": "connect_schemas", "arguments": {"source_id": {"name": name(picked[0])}, "target_id": {"new": 1}, "label": ""}},
            ]}
        elif kind == "frame":
            frame = rng.choice(FRAMES)
            command = f"put {name(picked[0])}, {name(picked[1])} and {name(picked[2])} in a frame called {frame}"
            script = {"actions": [{"name": "frame_shapes", "arguments": {
                "shape_ids": [{"name": name(o)} for o in picked], "name": frame}}]}
        elif kind == "frame_selected":
            frame = rng.choice(FRAMES)
            diagram["selected"] = [o["id"] for o in picked[:2]]
            command = f"frame these as {frame}"
            script = {"actions": [{"name": "frame_shapes", "arguments": {"shape_ids": [{"selected": True}], "name": frame}}]}
        elif kind == "add_to_frame":
            if not frames:
                continue
            frame = rng.choice(frames)
            command = f"move {name(picked[0])} into the {frame['text']} frame"
            script = {"actions": [{"name": "add_to_frame", "arguments": {
                "shape_ids": [{"name": name(picked[0])}], "frame_id": {"name": frame["text"]}}}]}
        else:
            direction = rng.choice(["left_to_right", "top_to_bottom"])
            command = "lay it all out left to right" if direction == "left_to_right" else "tidy this up top to bottom"
            script = {"actions": [{"name": "auto_layout", "arguments": {"direction": direction}}]}
        try:
            objects = session_of(diagram).objects()
            diagram["selected"] = [i for i in diagram["selected"] if i in objects]
            for key in ("last_created", "last_edited"):  # recent objects since deleted
                diagram[key] = diagram[key] if diagram[key] in objects else None
            result, _ = outcome(script, session_of(diagram), command)
        except (ValueError, KeyError, TypeError, IndexError):
            continue
        if result != "applied":
            continue
        kinds[kind] += 1
        kept.append({"id": f"observed:{len(kept)}", "kind": kind, "user": user_text(diagram, command),
                     "answer": script_text(script)})
    return kept, kinds


def diagram_of(session):
    """The plain diagram for a session, with selection and recent objects."""
    linked = {link["shape_id"] for link in session.connections}

    def styled(item, entry):
        if item.get("color", "black") != "black":
            entry["color"] = item["color"]
        if item.get("fill", "none") != "none":
            entry["fill"] = item["fill"]
        return entry

    present = lambda i: i if i in session.objects() else None  # noqa: E731
    def placed(item, entry):
        entry.update({key: round(item[key]) for key in ("x", "y", "w", "h")})
        if item.get("parent_id"):
            entry["parent"] = item["parent_id"]
        return styled(item, entry)

    boxes = [placed(b, {"id": b["id"], "name": b["name"], "fields": b["properties"], "methods": b["methods"]})
             for b in session.canvas["schemas"]]
    shapes = [placed(s, {"id": s["id"], "kind": s["kind"], "text": s["text"]})
              for s in session.canvas["shapes"] if s["id"] not in linked and s["kind"] != "group"]
    return {
        "boxes": boxes,
        "shapes": shapes,
        "connections": [{"id": link["shape_id"], "from": link["source_id"], "to": link["target_id"],
                         "label": link["label"]} for link in session.connections],
        "selected": list(session.canvas["selected_ids"]),
        # An object since deleted is no object at all.
        "last_created": present(session.history.get("last_created_id")),
        "last_edited": present(session.history.get("last_edited_id")),
    }


def shortened(diagram, script):
    """The diagram and script with short IDs (box1, shape1), as a person's canvas would show."""
    names = {}
    for prefix, key in (("box", "boxes"), ("shape", "shapes"), ("arrow", "connections")):
        for number, item in enumerate(diagram[key], 1):
            names[item["id"]] = f"{prefix}{number}"
    rename = lambda value: names.get(value, value)  # noqa: E731

    def walk(value):
        if isinstance(value, dict):
            return {k: rename(v) if k == "id" and isinstance(v, str) else walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    short = copy.deepcopy(diagram)
    for key in ("boxes", "shapes"):
        for item in short[key]:
            item["id"] = rename(item["id"])
            if item.get("parent"):
                item["parent"] = rename(item["parent"])
    for link in short["connections"]:
        link["id"], link["from"], link["to"] = rename(link["id"]), rename(link["from"]), rename(link["to"])
    short["selected"] = [rename(i) for i in short["selected"]]
    for key in ("last_created", "last_edited"):
        short[key] = rename(short[key]) if short[key] else None
    return short, walk(script)


def session_of(diagram):
    """The session for a plain diagram; raises on a diagram the canvas cannot hold.

    Teachers write arrows as from/to or source/target, sometimes with their own ID, which
    the arrow then keeps so a script can point at it."""
    links = [{"from": link.get("from", link.get("source")), "to": link.get("to", link.get("target")),
              "label": link.get("label") or ""} for link in diagram.get("connections", [])]
    session = session_from({**diagram, "connections": links})
    for given, link in zip(diagram.get("connections", []), session.connections):
        if isinstance(given.get("id"), str) and given["id"] not in session.objects():
            old_id = link["shape_id"]
            for shape in session.canvas["shapes"]:
                if shape["id"] == old_id:
                    shape["id"] = given["id"]
            link["shape_id"] = given["id"]
    objects = session.objects()
    for item in [*diagram.get("boxes", []), *diagram.get("shapes", [])]:
        if item.get("id") not in objects:
            continue
        target = objects[item["id"]]
        for key in ("color", "fill"):
            if key in item:
                target[key] = item[key]
        for key in ("x", "y", "w", "h"):  # positions and sizes when the diagram gives them
            if isinstance(item.get(key), (int, float)) and not isinstance(item.get(key), bool):
                target[key] = float(item[key]) if key in "xy" else max(1.0, float(item[key]))
        parent = item.get("parent")
        if parent:
            if objects.get(parent, {}).get("kind") not in ("frame", "group") or parent == item["id"]:
                raise ValueError(f"{item['id']} sits in {parent!r}, which is no frame")
            target["parent_id"] = parent
    session.normalize_context()
    session.canvas["selected_ids"] = sorted(i for i in diagram.get("selected") or [] if i in objects)
    for key in ("last_created", "last_edited"):
        value = diagram.get(key)
        session.history[f"{key}_id"] = value if value in objects else None
    return session


def user_text(diagram, command):
    return f"Diagram: {json.dumps(diagram, ensure_ascii=False)}\nMessage: {command}"


def parse_user(text):
    match = re.match(r"Diagram: (\{.*\})\nMessage: (.*)\Z", text.strip(), re.S)
    if not match:
        raise ValueError("not a diagram and message")
    return json.loads(match.group(1)), match.group(2).strip()


COORDINATES = re.compile(r"\b[xy]\s*[=:]?\s*-?\d+|\(\s*-?\d+\s*,\s*-?\d+\s*\)|\bcoordinates?\b", re.I)


def _side(anchor, x, y):
    dx = x - (anchor["x"] + anchor["w"] / 2)
    dy = y - (anchor["y"] + anchor["h"] / 2)
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "below" if dy >= 0 else "above"


def placed(script, session, command):
    """The script with guessed coordinates replaced by placement the app computes.

    A small model should not guess pixels: a component goes `near` the object it is wired
    to (else the nearest object) on the side its guessed position implied, and a shape's
    guessed x and y are dropped. Coordinates stay when the person says them."""
    from contract import ReferenceProblem, resolve

    if COORDINATES.search(command) or not script.get("actions"):
        return script
    script = copy.deepcopy(script)
    objects = session.objects()
    for index, action in enumerate(script["actions"], 1):
        args = action.get("arguments", {})
        if action.get("name") == "create_shape":
            args.pop("x", None)
            args.pop("y", None)
        if action.get("name") != "create_component" or not isinstance(args.get("x"), (int, float)) \
                or not isinstance(args.get("y"), (int, float)):
            continue
        x, y = args.pop("x"), args.pop("y")
        if args.get("near"):
            continue
        partner = None
        for other in script["actions"]:
            other_args = other.get("arguments", {})
            if other.get("name") != "connect_schemas":
                continue
            ends = (other_args.get("source_id"), other_args.get("target_id"))
            if {"new": index} in ends:
                partner = next((e for e in ends if e != {"new": index}), None)
                break
        anchor = None
        if isinstance(partner, dict) and "new" not in partner:
            try:
                found = resolve(partner, session, {})
                anchor = objects[found[0]] if len(found) == 1 else None
            except (ReferenceProblem, ValueError, KeyError):
                anchor = None
        if anchor is None:
            tops = [o for o in objects.values() if o.get("kind") != "arrow"]
            if not tops:
                continue
            anchor = min(tops, key=lambda o: (o["x"] + o["w"] / 2 - x) ** 2 + (o["y"] + o["h"] / 2 - y) ** 2)
            names = [o for o in objects.values() if o["name"] == anchor["name"]]
            partner = {"name": anchor["name"]} if len(names) == 1 else None
            if partner is None:
                continue
        args["near"] = partner
        args["side"] = _side(anchor, x, y)
    return script


def state(session):
    return session.snapshot(), session.canvas["camera"]


def outcome(script, session, command):
    """(outcome, state) of a script on a copy of the session; the state includes the camera."""
    session = copy.deepcopy(session)
    refused = run_script(script, session, command, "new")
    return refused or "applied", state(session)


def seed_rows(bundle, count, rng):
    """v7 training rows (never validation or test) whose plain diagram replays like the canvas."""
    rows = [json.loads(line) for line in open(bundle / "examples.jsonl")]
    rows = [r for r in rows if r["split"] == "train"]
    rng.shuffle(rows)
    by_kind, kept = Counter(), []
    quota = max(2, count // 40)
    for row in rows:
        script = to_script(row)
        if script is None:
            continue
        candidates = [(row["command"], script)]
        made = compound({**row, "script": script}, rng)
        if made:
            candidates.append((made["command"], made["script"]))
        for command, script in candidates:
            kind = "compound" if len(script.get("actions", [])) > 1 else (row.get("stratum") or row["expected"]["name"])
            if by_kind[kind] >= (quota * 4 if kind == "compound" else quota):
                continue
            original = session_for(row)
            try:
                diagram, short_script = shortened(diagram_of(original), script)
                rebuilt = session_of(diagram)
                first, second = outcome(script, original, command), outcome(short_script, rebuilt, command)
                same = first[0] == second[0] and semantic_state(_after(script, original, command)) == \
                    semantic_state(_after(short_script, rebuilt, command))
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            if not same:
                continue
            by_kind[kind] += 1
            kept.append({"id": row["id"], "kind": kind, "user": user_text(diagram, command),
                         "answer": script_text(short_script)})
        if len(kept) >= count:
            break
    return kept, by_kind


def _after(script, session, command):
    session = copy.deepcopy(session)
    run_script(script, session, command, "new")
    return session


def config(teacher, target):
    lines = [
        "base:",
        "  task: question-answering",
        "  student_model_name: Qwen3-1.7B",
        f"  teacher_model_name: {teacher}",
        "  should_expand_dataset: false",
        "synthgen:",
        "  output_is_json: true",
        "  teacher_temperature: 0.6",
        f"  train_generation_target: {target}",
        "  mutators:",
    ]
    for mutator in MUTATORS:
        lines.append(f"    - name: {mutator['name']}")
        lines.append("      values:")
        lines += [f"        - {json.dumps(value)}" for value in mutator["values"]]
    lines += ["evaluation:", "  llm_as_a_judge_model_name: openai.gpt-oss-120b", ""]
    return "\n".join(lines)


def export(args):
    rng = random.Random(args.seed)
    rows, kinds = seed_rows(args.bundle, args.seeds + args.test, rng)
    rng.shuffle(rows)
    test, rows = rows[: args.test], rows[args.test:]
    if args.observed:  # new-tool seeds on observed canvases, for training only
        extra, extra_kinds = observed_seed_rows(args.observed, args.observed_seeds, rng)
        rows += extra
        kinds.update({f"observed:{k}": v for k, v in extra_kinds.items()})
        rng.shuffle(rows)
    dataset = args.output / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    splits = {"test": test, "train": rows}
    for split, items in splits.items():
        with open(dataset / f"{split}.jsonl", "w") as out:
            for item in items:
                out.write(json.dumps({"messages": [{"role": "user", "content": item["user"]},
                                                   {"role": "assistant", "content": item["answer"]}]},
                                     ensure_ascii=False) + "\n")
    json.dump({"task_description": TASK, "synthetic_data_generation_instructions": GENERATION},
              open(dataset / "job_description.json", "w"), indent=2, ensure_ascii=False)
    (dataset / "config.yaml").write_text(config(args.teacher, args.target))
    print(json.dumps({"seeds": len(splits["train"]), "test": len(splits["test"]), "kinds": dict(kinds)}))


def check(args):
    counts, kept = Counter(), []
    for line in open(args.input):
        messages = json.loads(line)["messages"]
        user = next(m["content"] for m in messages if m["role"] == "user")
        answer = next(m["content"] for m in reversed(messages) if m["role"] == "assistant")
        try:
            diagram, command = parse_user(user)
            session = session_of(diagram)
        except (ValueError, KeyError, TypeError, IndexError):
            counts["bad_diagram"] += 1
            continue
        named = set(session.objects())
        pointed = set(diagram.get("selected") or []) | {diagram.get(k) for k in ("last_created", "last_edited")}
        if pointed - named - {None}:  # "it" or "the selected ones" would mean something else
            counts["bad_reference"] += 1
            continue
        try:
            script = placed(parse_script(answer), session, command)
            result, snapshot = outcome(script, session, command)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            counts["bad_script"] += 1
            continue
        if result == "applied" and not script["actions"]:
            counts["empty"] += 1
            continue
        if result == "applied" and snapshot == state(session):
            counts["no_change"] += 1  # kept: "zoom to 100%" at 100% is still the right script
        counts[result] += 1
        kept.append({"diagram": diagram, "command": command, "script": script, "outcome": result})
    with open(args.output, "w") as out:
        for item in kept:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"kept": len(kept), **counts}))


def same_edit(first, second, base):
    """Whether two sessions hold the same result of one edit: content, positions and sizes of
    the objects that were already there, the camera and the selected names."""
    if semantic_state(first) != semantic_state(second) or first.canvas["camera"] != second.canvas["camera"]:
        return False
    for identifier in base.objects():
        a, b = first.objects().get(identifier), second.objects().get(identifier)
        if (a is None) != (b is None):
            return False
        if a is not None and any(a.get(k) != b.get(k) for k in ("x", "y", "w", "h", "parent_id")):
            return False
    names = lambda s: sorted(s.objects()[i]["name"] for i in s.canvas["selected_ids"] if i in s.objects())  # noqa: E731
    return names(first) == names(second)


def second_opinion(args):
    """The checked rows as a test set for cli_eval.py or distil teacher evaluation."""
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    json.dump({"task_description": TASK}, open(folder / "job_description.json", "w"), indent=2)
    with open(folder / "test.jsonl", "w") as out:
        for line in open(args.input):
            item = json.loads(line)
            out.write(json.dumps({"messages": [
                {"role": "user", "content": user_text(item["diagram"], item["command"])},
                {"role": "assistant", "content": script_text(item["script"])}]}, ensure_ascii=False) + "\n")


def agree(args):
    """Keep checked rows whose second teacher reply reaches the same result."""
    from teach_data import json_reply

    replies = {}
    for line in open(args.predictions):
        item = json.loads(line)
        prompt = item["prompt"]
        if isinstance(prompt, list):
            prompt = next(m["content"] for m in reversed(prompt) if m["role"] == "user")
        reply = item["prediction"]
        replies[prompt] = reply.get("content") or "" if isinstance(reply, dict) else reply
    counts, kept, disagreements = Counter(), [], []
    for line in open(args.input):
        item = json.loads(line)
        text = replies.get(user_text(item["diagram"], item["command"]))
        if text is None:
            counts["no_reply"] += 1
            continue
        base = session_of(item["diagram"])
        mine, theirs = copy.deepcopy(base), copy.deepcopy(base)
        first = run_script(item["script"], mine, item["command"], "new") or "applied"
        try:
            second = run_script(parse_script(json.dumps(json_reply(text))), theirs, item["command"], "new") or "applied"
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            second = "invalid"
        if first == second and same_edit(mine, theirs, base):
            counts["agree"] += 1
            kept.append(item)
        else:
            counts["disagree"] += 1
            disagreements.append({"command": item["command"], "first": first, "second": second,
                                  "script": script_text(item["script"]), "reply": text[:600]})
    with open(args.output, "w") as out:
        for item in kept:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    json.dump(disagreements, open(args.output.with_suffix(".disagreements.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(counts))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("export")
    making.add_argument("--bundle", type=Path, required=True)
    making.add_argument("--output", type=Path, required=True)
    making.add_argument("--seeds", type=int, default=240)
    making.add_argument("--test", type=int, default=40)
    making.add_argument("--teacher", default="moonshotai.kimi-k3-low-thinking")
    making.add_argument("--target", type=int, default=5000)
    making.add_argument("--seed", type=int, default=17)
    making.add_argument("--observed", type=Path, help="observation rows (observe/analyze.py --rows)")
    making.add_argument("--observed-seeds", type=int, default=80)
    checking = commands.add_parser("check")
    checking.add_argument("--input", type=Path, required=True)
    checking.add_argument("--output", type=Path, required=True)
    opinion = commands.add_parser("second-opinion")
    opinion.add_argument("--input", type=Path, required=True)
    opinion.add_argument("--output", type=Path, required=True)
    agreeing = commands.add_parser("agree")
    agreeing.add_argument("--input", type=Path, required=True)
    agreeing.add_argument("--predictions", type=Path, required=True)
    agreeing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"export": export, "check": check, "second-opinion": second_opinion, "agree": agree}[args.command](args)


if __name__ == "__main__":
    main()
