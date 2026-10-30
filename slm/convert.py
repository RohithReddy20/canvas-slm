"""Single-action v7 rows as edit scripts (contract.py), kept only when they resolve back.

Each target ID in a label becomes the reference the command used: the object's name,
the selection (the one selected object for "it" or "the selected shape"), or the last
created or edited object. A converted script must resolve to exactly the original label
on the row's canvas.

A missing_target or ambiguous_target label does not say which edit was meant, so for
v7's templated refusals (move it, delete a name, remove a field from the selected shape)
the edit is read from the wording, and kept only when the resolver refuses it for the
labelled reason. Other refusals are dropped.
"""

import copy
import json
import re
import sys
from collections import Counter

from contract import PLURAL, SINGLE, resolve_action, run_script
from sessions import CanvasSession

SINGULAR = re.compile(
    r"\b(?:it|this|that)\b|\b(?:selected|chosen|highlighted)\s+"
    r"(?:box|schema|class|shape|object|rectangle|circle|text|note|frame|group)\b",
    re.I,
)
# v7 corrections ("Explain this. No, actually ...") keep only the request after them.
CORRECTION = re.compile(r"No, (?:actually|instead)|cancel that\.", re.I)
DIRECTIONS = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}


def request(command):
    return CORRECTION.split(command)[-1]


def selection_reference(row, selected):
    singular = len(selected) == 1 and SINGULAR.search(request(row["command"]))
    return {"it": True} if singular else {"selected": True}


def session_for(row):
    session = CanvasSession(row["canvas"])
    history = row.get("history") or {}
    if isinstance(history, dict):
        session.history = {
            "turns": history.get("turns", []),
            "last_created_id": history.get("last_created_id"),
            "last_edited_id": history.get("last_edited_id"),
        }
    # Rows say undo or redo is available without the history behind it; a copy of the
    # document stands in so those commands can run.
    if session.canvas["can_undo"] and not session.undo_stack:
        session.undo_stack.append(session.document())
    if session.canvas["can_redo"] and not session.redo_stack:
        session.redo_stack.append(session.document())
    return session


def reference(identifier, row, session):
    item = session.objects()[identifier]
    kind = row.get("reference_kind")
    for last in ("created", "edited"):
        if kind == f"last_{last}" and session.history.get(f"last_{last}_id") == identifier:
            return {"last": last}
    selected = session.canvas["selected_ids"]
    if kind == "selection" and identifier in selected:
        return selection_reference(row, selected)
    if item["name"].casefold() in row["command"].casefold():
        return {"name": item["name"]}
    if identifier in selected:
        return selection_reference(row, selected)
    return None


def plural_references(ids, row, session):
    selected = session.canvas["selected_ids"]
    top = [
        i
        for i, item in session.objects().items()
        if item["parent_id"] is None and item.get("kind") != "arrow"
    ]
    kind = row.get("reference_kind")
    if kind == "selection" and sorted(ids) == sorted(selected):
        return [selection_reference(row, selected)]
    if kind == "all" and sorted(ids) == sorted(top):
        return [{"all": True}]
    references = [reference(identifier, row, session) for identifier in ids]
    return None if None in references else list({json.dumps(r): r for r in references}.values())


def intended(row):
    """The edit a templated v7 refusal asked for, or None."""
    family, kind = (row.get("wording_family") or "").split(":"), row.get("reference_kind")
    command = request(row["command"])
    quoted = re.search(r'"([^"]+)"', command)
    if kind in ("absent_selection", "multiple_selection"):
        target = {"it": True}
    elif kind in ("removed_last_created", "removed_last_edited"):
        target = {"last": kind.rsplit("_", 1)[1]}
    elif kind in ("missing_name", "duplicate_name") and quoted:
        target = {"name": quoted.group(1)}
    elif kind == "missing_field" and SINGULAR.search(command):
        target = {"it": True}
    else:
        return None
    action = family[1] if len(family) > 2 else None
    if action == "move_shapes":
        found = re.search(r"(\d+(?:\.\d+)?)\D*?\b(right|left|up|down)\b", command) or re.search(
            r"\b(right|left|up|down)\b\D*?(\d+(?:\.\d+)?)", command)
        if not found:
            return None
        amount, direction = sorted(found.groups(), key=lambda g: g in DIRECTIONS)
        x, y = DIRECTIONS[direction]
        return {"name": "move_shapes", "arguments": {
            "shape_ids": [target], "dx": float(x * float(amount)), "dy": float(y * float(amount))}}
    if action == "delete_shapes":
        return {"name": "delete_shapes", "arguments": {"shape_ids": [target]}}
    if action == "remove_property":
        field = re.search(r"\b(?:property|field)\s+(?:called\s+)?(\w+)|\bthe\s+(\w+)\s+attribute\b", command)
        if not field:
            return None
        name = field.group(1) or field.group(2)
        return {"name": "remove_property", "arguments": {"schema_id": target, "property_name": name}}
    return None


def to_script(row, live=None):
    """The row's label as a script, or None when it cannot be expressed faithfully.

    `live` is the session the row's turn starts from, when the row comes from a session."""
    expected = row["expected"]

    def fresh():
        return copy.deepcopy(live) if live is not None else session_for(row)

    if expected["name"] == "no_action":
        reason = expected["arguments"]["reason"]
        if reason == "unsupported_request":
            return {"actions": [], "reason": reason}
        action = intended(row)
        if action is None:
            return None
        script = {"actions": [action]}
        try:
            refused = run_script(script, fresh(), row["command"])
        except ValueError:
            return None
        return script if refused == reason else None
    session = fresh()
    arguments = dict(expected["arguments"])
    for key in SINGLE:
        if key in arguments:
            arguments[key] = reference(arguments[key], row, session)
            if arguments[key] is None:
                return None
    if PLURAL in arguments:
        arguments[PLURAL] = plural_references(arguments[PLURAL], row, session)
        if arguments[PLURAL] is None:
            return None
    script = {"actions": [{"name": expected["name"], "arguments": arguments}]}
    try:
        resolved = resolve_action(script["actions"][0], fresh(), {})
    except ValueError:
        return None
    return script if resolved == expected else None


def main():
    counts = Counter()
    for line in sys.stdin:
        row = json.loads(line)
        script = to_script(row)
        name = row["expected"]["name"]
        counts[f"{name}:{'kept' if script else 'dropped'}"] += 1
        if script:
            print(json.dumps({**row, "script": script}, ensure_ascii=False))
    print(json.dumps(dict(sorted(counts.items())), indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
