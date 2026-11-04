"""Compound requests: a converted v7 row plus further edits on the same canvas.

Real follow-ups mostly ask for several changes at once ("add Payment with amount and link
it to Order"). Each compound joins a row's own command with one or two clauses built from
the names on its canvas, and is kept only when the whole script applies in the simulator.

    python synth.py --count 8000 < scripts.jsonl > compounds.jsonl
"""

import argparse
import json
import random
import sys

from contract import run_script
from convert import session_for

FIELDS = ["email", "status", "created_at", "price", "quantity", "address", "phone", "total",
          "due_date", "owner_id", "title", "description", "rating", "currency", "zip_code"]
METHODS = ["validate()", "save()", "load()", "refresh()", "archive()", "notify()", "export()"]
NEW_NAMES = ["Payment", "Invoice", "Supplier", "Shipment", "Review", "Coupon", "Ticket",
             "Session", "Address", "Category", "Warehouse", "Refund", "Notification", "Report"]
COLORS = ["red", "blue", "green", "orange", "violet", "grey", "yellow"]
JOINERS = [" and ", ", then ", ". Also ", "; ", " and also ", ". After that, ", ", and then "]


def clause(session, rng):
    """One further request on this canvas, as (text, actions), or None."""
    objects = session.objects()
    counts = {}
    for item in objects.values():
        counts[item["name"].casefold()] = counts.get(item["name"].casefold(), 0) + 1
    unique = [o for o in objects.values() if o.get("kind") != "arrow" and counts[o["name"].casefold()] == 1]
    boxes = [o for o in unique if "kind" not in o]
    if not unique:
        return None
    ref = lambda item: {"name": item["name"]}  # noqa: E731
    choice = rng.choice(["field", "method", "connect", "rename", "delete", "style", "create"])
    if choice == "field" and boxes:
        box, field = rng.choice(boxes), rng.choice(FIELDS)
        text = rng.choice(["add {f} to {b}", "give {b} a {f} field", "{b} also needs {f}",
                           "put {f} on {b}"]).format(f=field, b=box["name"])
        return text, [{"name": "add_property", "arguments": {"schema_id": ref(box), "property_name": field}}]
    if choice == "method" and boxes:
        box, method = rng.choice(boxes), rng.choice(METHODS)
        text = rng.choice(["add a {m} method to {b}", "{b} should have {m}"]).format(m=method, b=box["name"])
        return text, [{"name": "add_method", "arguments": {"schema_id": ref(box), "method_name": method}}]
    if choice == "connect" and len(unique) > 1:
        source, target = rng.sample(unique, 2)
        label = rng.choice(["", "", "uses", "owns", "has many"])
        text = rng.choice(["connect {s} to {t}", "link {s} with {t}", "draw an arrow from {s} to {t}"])
        text = text.format(s=source["name"], t=target["name"]) + (f' labelled "{label}"' if label else "")
        return text, [{"name": "connect_schemas", "arguments": {"source_id": ref(source), "target_id": ref(target), "label": label}}]
    if choice == "rename" and boxes:
        box, name = rng.choice(boxes), rng.choice(NEW_NAMES)
        text = rng.choice(["rename {b} to {n}", "call {b} {n} instead"]).format(b=box["name"], n=name)
        return text, [{"name": "rename_schema", "arguments": {"schema_id": ref(box), "new_name": name}}]
    if choice == "delete":
        item = rng.choice(unique)
        text = rng.choice(["delete {x}", "remove {x}", "get rid of {x}"]).format(x=item["name"])
        return text, [{"name": "delete_shapes", "arguments": {"shape_ids": [ref(item)]}}]
    if choice == "style":
        item, color = rng.choice(unique), rng.choice(COLORS)
        text = rng.choice(["make {x} {c}", "color {x} {c}", "turn {x} {c}"]).format(x=item["name"], c=color)
        return text, [{"name": "style_shapes", "arguments": {"shape_ids": [ref(item)], "color": color}}]
    if choice == "create":
        name, fields = rng.choice(NEW_NAMES), rng.sample(FIELDS, rng.randint(1, 3))
        target = rng.choice(unique)
        text = rng.choice([
            "create a {n} box with {f} and connect it to {t}",
            "add {n} with {f}, linked to {t}",
            "make a new {n} class with {f} and draw an arrow from it to {t}",
        ]).format(n=name, f=", ".join(fields), t=target["name"])
        return text, [
            {"name": "create_schema_box", "arguments": {"name": name, "fields": fields, "methods": []}},
            {"name": "connect_schemas", "arguments": {"source_id": {"new": 1}, "target_id": ref(target), "label": ""}},
        ]
    return None


def offset(actions, by):
    """The actions with {"new": n} references shifted past `by` earlier actions."""
    text = json.dumps(actions)
    for number in range(9, 0, -1):
        text = text.replace(f'{{"new": {number}}}', f'{{"new": {number + by}}}')
    return json.loads(text)


def compound(row, rng):
    if row["script"].get("reason") is not None:
        return None
    command, actions = row["command"].rstrip(" ."), list(row["script"]["actions"])
    session = session_for(row)
    for _ in range(rng.choice([1, 1, 2])):
        extra = clause(session, rng)
        if extra is None:
            return None
        text, more = extra
        command += rng.choice(JOINERS) + text
        actions += offset(more, len(actions))
    script = {"actions": actions}
    try:
        if run_script(script, session_for(row), command) is not None:
            return None
    except ValueError:
        return None
    return {**row, "id": f"{row['id']}:compound", "command": command + ".", "script": script,
            "stratum": "compound"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in sys.stdin]
    made = 0
    for row in rng.sample(rows, len(rows)):
        result = compound(row, rng)
        if result:
            print(json.dumps(result, ensure_ascii=False))
            made += 1
            if made == args.count:
                break
    print(f"{made} compounds from {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
