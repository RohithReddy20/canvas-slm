"""A relaxed check for real follow-ups, where one gold script is only one right answer.

Strict scoring compares the whole diagram with the gold one (semantic_state). Real requests
are open: "add the load balancer" may be named "Balanceador de Carga", wired to the UI, or
given fields. `relaxed` accepts a prediction that makes the same edit as the gold:

- it deletes exactly the objects and arrows the gold deletes, and nothing else;
- every object the gold adds is added, with a similar name when the command names it (any
  name of the same kind when the model has to invent one, as for "a big title"; text and
  notes are matched by kind, since their wording is always the model's), with the members
  the gold gives it; no other object is added;
- kept boxes gain and lose the same members (similar names), and every object keeps the
  color and fill the gold gives it;
- every arrow the gold adds is there, labelled alike when the command quotes the label, in the
  same direction unless it wires up a new object; extra arrows are allowed only when they
  wire up a new object.

A real request often has several right answers. A test row carries the reviewed primary
outcome in "after" and the other acceptable outcomes in "accepted"; `matches` passes a
prediction that makes any of them. A row with "arrow_direction": "any" (UML "implements"
and "uses" arrows, whose direction is convention) ignores arrow direction.
"""

import re
from collections import Counter
from difflib import SequenceMatcher

BOX_LIKE = {"box", "rectangle"}
WORDED = {"text", "note"}


def _key(text):
    return re.sub(r"[^0-9a-z]", "", str(text).casefold())


def similar(a, b):
    a, b = _key(a), _key(b)
    if not a or not b:
        return a == b
    if a == b or (min(len(a), len(b)) >= 4 and (a in b or b in a)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.8


def named_in(name, command):
    """Whether the command spells out this name, so the model must use (about) it."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name))  # ProductCategory: two words
    words = [w for w in re.findall(r"\w+", spaced.casefold()) if len(w) >= 3]
    if not words:
        return False
    said = re.findall(r"\w+", command.casefold())
    hits = sum(any(similar(w, s) or s.startswith(w[:5]) for s in said) for w in words)
    return hits * 2 >= len(words)


def _objects(state):
    objects = [
        {"kind": "box", "name": name, "members": list(props) + list(methods), "style": (color, fill)}
        for name, props, methods, color, fill in state["boxes"]
    ]
    objects += [
        {"kind": kind, "name": text, "members": [], "style": (color, fill)}
        for kind, text, color, fill in state["shapes"]
    ]
    return objects


def _split(before, after):
    """Objects of `after` kept from `before` (by exact name and family), added, and removed."""
    pool = Counter((_key(o["name"]), o["kind"] in BOX_LIKE) for o in before)
    kept, added = [], []
    for item in after:
        slot = (_key(item["name"]), item["kind"] in BOX_LIKE)
        if pool[slot]:
            pool[slot] -= 1
            kept.append(item)
        else:
            added.append(item)
    return kept, added, +pool


def _pair_members(gold, predicted):
    """Every gold member has a similar predicted one; returns False when one is missing."""
    left = list(predicted)
    for member in gold:
        match = next((m for m in left if similar(m, member)), None)
        if match is None:
            return False
        left.remove(match)
    return True


def relaxed(before, predicted, gold, command, directed=True):
    """True when `predicted` (a semantic_state) makes the edit `gold` makes to `before`."""
    base = _objects(before)
    kept_g, added_g, removed_g = _split(base, _objects(gold))
    kept_p, added_p, removed_p = _split(base, _objects(predicted))
    if removed_g != removed_p:
        return False

    # New objects: pair each gold one with a predicted one.
    names, free = {}, list(added_p)
    for item in added_g:
        family = item["kind"] in BOX_LIKE
        candidates = [p for p in free if (p["kind"] in BOX_LIKE) == family or p["kind"] == item["kind"]]
        if item["kind"] in WORDED:
            match = next((p for p in candidates if p["kind"] == item["kind"]), None)
        elif named_in(item["name"], command) or not candidates:
            match = next((p for p in candidates if similar(p["name"], item["name"])), None)
        else:
            match = next((p for p in candidates if similar(p["name"], item["name"])), candidates[0])
        if match is None or not _pair_members(item["members"], match["members"]):
            return False
        if item["style"] != ("black", "none") and match["style"] != item["style"]:
            return False
        free.remove(match)
        names[_key(item["name"])] = _key(match["name"])
    if free:
        return False

    # Kept objects: the same member changes and the same style.
    by_name = {}
    for item in kept_p:
        by_name.setdefault(_key(item["name"]), []).append(item)
    before_by = {}
    for item in base:
        before_by.setdefault(_key(item["name"]), []).append(item)
    for item in kept_g:
        options = by_name.get(_key(item["name"]), [])
        if not options:
            return False
        mine = options.pop(0)
        if mine["style"] != item["style"]:
            return False
        original = Counter(before_by[_key(item["name"])][0]["members"])
        gold_members, my_members = Counter(item["members"]), Counter(mine["members"])
        if original - gold_members != original - my_members:  # removed members
            return False
        if not _pair_members(list((gold_members - original).elements()), list((my_members - original).elements())):
            return False
        if sum((my_members - original).values()) != sum((gold_members - original).values()):
            return False

    # Arrows, by endpoint names; new gold objects are renamed to their predicted partners.
    def links(state, rename=None):
        rename = rename or {}
        return Counter((rename.get(_key(a), _key(a)), rename.get(_key(b), _key(b))) for a, b, _ in state["connections"])

    new_objects = {_key(p["name"]) for p in added_p}

    def undirected(counter):  # an arrow to a new object may point either way
        return Counter({(tuple(sorted(pair)) if not directed or set(pair) & new_objects else pair): n
                        for pair, n in counter.items()})

    old = links(before)
    gold_links, my_links = undirected(links(gold, names)), undirected(links(predicted))
    if old - gold_links != old - my_links:  # removed arrows
        return False
    missing = (gold_links - old) - (my_links - old)
    if missing:
        return False
    extra = (my_links - old) - (gold_links - old)
    if any(a not in new_objects and b not in new_objects for a, b in extra):
        return False
    # Labels the command spells out must match.
    for a, b, label in gold["connections"]:
        if len(_key(label)) >= 3 and _key(label) in _key(command):  # the command quotes the label
            pair = (names.get(_key(a), _key(a)), names.get(_key(b), _key(b)))
            if not any(((_key(x), _key(y)) == pair or (not directed and (_key(y), _key(x)) == pair)) and similar(lab, label)
                       for x, y, lab in predicted["connections"]):
                return False
    return True


def _names(state):
    return Counter(_key(o["name"]) for o in _objects(state))


def additive(before, predicted, anchors=(), min_new=1):
    """A request for more content ("add more detail", "expand X"): nothing is removed, at least
    `min_new` objects are added, and when `anchors` are named, some new arrow joins one of them."""
    old_names, new_names = _names(before), _names(predicted)
    if old_names - new_names:  # an object disappeared
        return False
    old_links = Counter((_key(a), _key(b)) for a, b, _ in before["connections"])
    new_links = Counter((_key(a), _key(b)) for a, b, _ in predicted["connections"])
    if old_links - new_links:  # an arrow disappeared
        return False
    added = sum((new_names - old_names).values())
    if added < min_new:
        return False
    if anchors:
        fresh = set((new_names - old_names).elements())
        wanted = {_key(a) for a in anchors}
        return any((a in wanted and b in fresh) or (b in wanted and a in fresh) for a, b in (new_links - old_links).elements())
    return True


def restyle(before, predicted):
    """Only looks or wording change: no object or arrow added or removed, and something (a
    colour, a fill, a text, a member or a label) is different."""
    if len(_objects(before)) != len(_objects(predicted)) or len(before["connections"]) != len(predicted["connections"]):
        return False
    return before != predicted


def matches(row, before, predicted, script=None):
    """Whether `predicted` passes the row: the edit of its primary or any accepted outcome, or,
    by the row's "expect", a refusal (nothing changes), an additive edit, a restyle, or a
    layout (judged by the script, since positions are not part of the state)."""
    expect = row.get("expect", "edit")
    if expect == "refusal":
        return predicted == before
    if expect == "additive":
        return additive(before, predicted, row.get("anchors") or (), row.get("min_new", 1))
    if expect == "restyle":
        return restyle(before, predicted)
    if expect == "layout":
        actions = (script or {}).get("actions") or []
        return any(a.get("name") == "auto_layout" and a.get("arguments", {}).get("direction") == row.get("layout")
                   for a in actions) and predicted == before
    directed = row.get("arrow_direction") != "any"
    return any(relaxed(before, predicted, gold, row["command"], directed)
               for gold in [row["after"], *row.get("accepted", [])])
