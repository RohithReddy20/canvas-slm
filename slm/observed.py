"""Observation-rig canvases (observe/server.mjs logs) as sessions, and how faithfully the
Python simulator replays what tldraw did.

    python observed.py parity --rows ../runs/observe/rows.jsonl

`session_of_observed` keeps the logged IDs (box1, shape3, arrow2), positions, sizes,
colors, frame membership, selection and recent objects. `parity` replays every logged act
step from its canvas before and compares the result with tldraw's canvas after, by
content (names, kinds, text, colors, fields), arrows and frame membership.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import extensions  # noqa: F401  (teaches sessions the new tools)
from contract import semantic_state, session_from


def session_of_observed(canvas):
    """A CanvasSession holding an observed canvas with its own IDs and geometry."""
    links = [link for link in canvas["links"] if link["from"] and link["to"]]
    loose = [link for link in canvas["links"] if not (link["from"] and link["to"])]
    diagram = {
        "boxes": [{"id": b["id"], "name": b["name"], "fields": b["fields"], "methods": b["methods"]} for b in canvas["boxes"]],
        "shapes": [{"id": s["id"], "kind": s["kind"], "text": s["text"]} for s in canvas["shapes"]]
        + [{"id": link["id"], "kind": "arrow", "text": link["label"]} for link in loose],
        "connections": [{"from": link["from"], "to": link["to"], "label": link["label"]} for link in links],
    }
    session = session_from(diagram)
    for given, link in zip(links, session.connections):  # arrows keep their logged IDs
        old = link["shape_id"]
        for shape in session.canvas["shapes"]:
            if shape["id"] == old:
                shape["id"] = given["id"]
        link["shape_id"] = given["id"]
    objects = session.objects()
    for item in [*canvas["boxes"], *canvas["shapes"]]:
        target = objects[item["id"]]
        for key in ("x", "y", "w", "h"):
            target[key] = item[key]
        target["color"] = item.get("color", "black")
        target["fill"] = item.get("fill", "none")
        target["parent_id"] = item.get("parent")
    session.canvas["selected_ids"] = sorted(i for i in canvas["selected"] if i in objects)
    session.history["last_created_id"] = canvas["last_created"] if canvas["last_created"] in objects else None
    session.history["last_edited_id"] = canvas["last_edited"] if canvas["last_edited"] in objects else None
    session.canvas["camera"] = dict(canvas["camera"])
    session.normalize_context()
    return session


def memberships(session):
    objects = session.objects()
    return sorted((o["name"], objects[o["parent_id"]]["name"]) for o in objects.values()
                  if o["parent_id"] and o.get("kind") != "arrow")


def replay(row):
    """The session after the logged actions; created objects take the IDs tldraw gave them."""
    session = session_of_observed(row["before"])
    for index, (action, result) in enumerate(zip(row["actions"], row["results"]), 1):
        created = result.get("created") or f"replay-{row['step']}-{index}"
        session.execute(row["command"], action, created)
    return session


def parity(args):
    rows = [json.loads(line) for line in open(args.rows)]
    rows = [r for r in rows if r["actions"] and all(x["ok"] for x in r["results"])]
    counts, by_action, misses = Counter(), Counter(), []
    for row in rows:
        names = sorted({a["name"] for a in row["actions"]})
        if any(a["name"] == "canvas_command" and a["arguments"].get("operation") in ("undo", "redo") for a in row["actions"]):
            counts["needs_history"] += 1  # a single replayed step has no undo stack
            continue
        try:
            simulated = replay(row)
        except ValueError as error:
            counts["python_rejected"] += 1
            misses.append({"command": row["command"], "actions": names, "why": f"rejected: {error}"[:160]})
            continue
        expected = session_of_observed(row["after"])
        same_content = semantic_state(simulated) == semantic_state(expected)
        same_frames = memberships(simulated) == memberships(expected)
        ok = same_content and same_frames
        counts["agree" if ok else "differ"] += 1
        for name in names:
            by_action[f"{name}:{'ok' if ok else 'differ'}"] += 1
        if not ok:
            a, b = semantic_state(simulated), semantic_state(expected)
            diff = {k: [x for x in a[k] if x not in b[k]] for k in a if a[k] != b[k]}
            diff["expected"] = {k: [x for x in b[k] if x not in a[k]] for k in b if a[k] != b[k]}
            if not same_frames:
                diff["frames"] = {"python": memberships(simulated), "tldraw": memberships(expected)}
            misses.append({"command": row["command"], "actions": names, "why": json.dumps(diff, ensure_ascii=False)[:400]})
    print(json.dumps({"steps": len(rows), **counts}))
    actions = sorted({k.rsplit(":", 1)[0] for k in by_action})
    for name in actions:
        print(f"  {name:18} {by_action[name + ':ok']}/{by_action[name + ':ok'] + by_action[name + ':differ']}")
    for miss in misses[: args.show]:
        print(f"- {miss['command'][:90]} {miss['actions']}\n    {miss['why']}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("parity")
    checking.add_argument("--rows", type=Path, required=True)
    checking.add_argument("--show", type=int, default=12)
    args = parser.parse_args()
    {"parity": parity}[args.command](args)


if __name__ == "__main__":
    main()
