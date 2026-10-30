"""Single-action v7 rows as edit scripts (contract.py), kept only when they resolve back.

Each target ID in a label becomes the reference the command used: the object's name,
the selection, or the last created or edited object. A converted script must resolve to
exactly the original label on the row's canvas. Rows labelled missing_target or
ambiguous_target are dropped: under scripts that decision belongs to the resolver, and
the label does not say which edit was meant.
"""

import json
import sys
from collections import Counter

from contract import PLURAL, SINGLE, resolve_action
from sessions import CanvasSession


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
        return {"selected": True}
    if item["name"].casefold() in row["command"].casefold():
        return {"name": item["name"]}
    if identifier in selected:
        return {"selected": True}
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
        return [{"selected": True}]
    if kind == "all" and sorted(ids) == sorted(top):
        return [{"all": True}]
    references = [reference(identifier, row, session) for identifier in ids]
    return None if None in references else list({json.dumps(r): r for r in references}.values())


def to_script(row):
    """The row's label as a script, or None when it cannot be expressed faithfully."""
    expected = row["expected"]
    if expected["name"] == "no_action":
        if expected["arguments"]["reason"] != "unsupported_request":
            return None
        return {"actions": [], "reason": "unsupported_request"}
    session = session_for(row)
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
        resolved = resolve_action(script["actions"][0], session_for(row), {})
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
