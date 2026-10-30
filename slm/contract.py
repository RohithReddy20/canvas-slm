"""Edit scripts: several canvas actions per command, with targets named as the user did.

A script is {"actions": [...]} or a refusal {"actions": [], "reason": "unsupported_request"}.
Actions use the tools of `actions.ACTION_MODELS`, except that target arguments
(schema_id, shape_id, source_id, target_id, shape_ids) hold references instead of IDs:

    {"name": "User"}                 the object with that name
    {"selected": true}               the current selection
    {"last": "created" | "edited"}   the object last created or edited
    {"new": 2}                       the object created by action 2 of this script
    {"kind": "diamond"}              every object of a kind ("schema" for boxes)
    {"all": true}                    every top-level object
    {"connection": {"from": ref, "to": ref, "label"?: text}}   the arrow between two objects
    {"id": "box1"}                   a short ID shown in the canvas context

`run_script` resolves references against the canvas as each action runs and applies the
actions through the frozen simulator, all or nothing. Whether a target is missing or
ambiguous is decided here, not by the model.
"""

import copy
import re

from actions import ACTION_MODELS, model_canvas
from sessions import CanvasSession, schema_height, shape_name

SINGLE = ("schema_id", "shape_id", "source_id", "target_id")
PLURAL = "shape_ids"
REASONS = ("unsupported_request",)
GRID = (5, 360, 320)


class ReferenceProblem(ValueError):
    """A reference that names nothing (missing_target) or several things (ambiguous_target)."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def _key(text):
    return re.sub(r"[^0-9a-z]", "", str(text).casefold())


def _loose(objects, connections, name):
    """Objects a loosely written name may mean: case, punctuation, spacing and a plural s
    ignored; then an arrow by its label; then an object whose whole name the reference
    contains ("the Order table"). Never a reference that is only part of a name."""
    wanted = re.sub(r"(arrow|line|link|edge|box|class|table|entity|node|shape)s?$", "", _key(name)) or _key(name)
    if len(wanted) < 3:
        return []
    forms = {wanted, wanted.removesuffix("s"), wanted + "s"}
    shapes = {i: item for i, item in objects.items() if item.get("kind") != "arrow"}
    return (
        [i for i, item in shapes.items() if _key(item["name"]) in forms]
        or [link["shape_id"] for link in connections if link["label"] and _key(link["label"]) in forms]
        or [i for i, item in shapes.items() if len(_key(item["name"])) >= 4 and _key(item["name"]) in wanted]
    )


def _named(session, name, fuzzy):
    objects = session.objects()
    found = [
        identifier
        for identifier, item in objects.items()
        if item.get("kind") != "arrow" and item["name"].casefold() == str(name).casefold()
    ]
    if not found and fuzzy:
        found = _loose(objects, session.connections, name)
    if not found:
        raise ReferenceProblem("missing_target", f"Nothing is named {name!r}.")
    if len(found) > 1:
        raise ReferenceProblem("ambiguous_target", f"Several objects are named {name!r}.")
    return found


def resolve(reference, session, created, plural=False, fuzzy=False):
    """The IDs a reference names on the current canvas; `fuzzy` also accepts loose names."""
    if not isinstance(reference, dict) or len(reference) != 1:
        raise ValueError(f"A reference is one keyed object: {reference!r}")
    (key, value), = reference.items()
    objects = session.objects()
    if key == "name":
        return _named(session, value, fuzzy)
    if key == "id":
        _, identifiers = model_canvas(session.canvas)
        aliases = {alias: identifier for identifier, alias in identifiers.items()}
        if value not in aliases:
            raise ReferenceProblem("missing_target", f"No object has the ID {value!r}.")
        return [aliases[value]]
    if key == "selected":
        selected = list(session.canvas["selected_ids"])
        if not selected or (not plural and len(selected) != 1):
            raise ReferenceProblem(
                "ambiguous_target" if objects else "missing_target",
                "This needs one selected object." if not plural else "Nothing is selected.",
            )
        return selected
    if key == "last":
        identifier = session.history.get(f"last_{value}_id")
        if value not in ("created", "edited") or identifier not in objects:
            raise ReferenceProblem("missing_target", f"There is no last {value} object.")
        return [identifier]
    if key == "new":
        if value not in created:
            raise ValueError(f"Action {value} of this script created nothing.")
        return [created[value]]
    if key in ("kind", "all"):
        if not plural:
            raise ValueError(f"A {key!r} reference needs a plural target.")
        found = [
            identifier
            for identifier, item in objects.items()
            if item["parent_id"] is None
            and (
                item.get("kind", "schema") == value
                if key == "kind"
                else item.get("kind") != "arrow"
            )
        ]
        if not found:
            raise ReferenceProblem("missing_target", f"Nothing matches {reference!r}.")
        return found
    if key == "connection":
        ends = [resolve(value[end], session, created, fuzzy=fuzzy)[0] for end in ("from", "to")]
        # An optional label picks one of several arrows between the same two objects.
        found = [
            link["shape_id"]
            for link in session.connections
            if [link["source_id"], link["target_id"]] == ends
            and ("label" not in value or link["label"] == value["label"])
        ]
        if not found:
            raise ReferenceProblem("missing_target", "Those objects are not connected.")
        return found
    raise ValueError(f"Unknown reference: {key!r}")


def resolve_action(action, session, created, fuzzy=False):
    if (
        not isinstance(action, dict)
        or set(action) != {"name", "arguments"}
        or action["name"] not in ACTION_MODELS
        or action["name"] == "no_action"
        or not isinstance(action["arguments"], dict)
    ):
        raise ValueError(f"Expected a known action with arguments: {action!r}")
    arguments = dict(action["arguments"])
    for key in SINGLE:
        if key in arguments:
            ids = resolve(arguments[key], session, created, fuzzy=fuzzy)
            if len(ids) != 1:
                raise ReferenceProblem("ambiguous_target", f"{key} names several objects.")
            arguments[key] = ids[0]
    if PLURAL in arguments:
        references = arguments[PLURAL]
        if not isinstance(references, list):
            raise ValueError("shape_ids holds a list of references.")
        ids = [
            i for r in references for i in resolve(r, session, created, plural=True, fuzzy=fuzzy)
        ]
        arguments[PLURAL] = list(dict.fromkeys(ids))
    return {"name": action["name"], "arguments": arguments}


def run_script(script, session, command, prefix="new", fuzzy=False):
    """Apply a script to the session; returns None, or why nothing was applied."""
    if not isinstance(script, dict) or not isinstance(script.get("actions"), list):
        raise ValueError("A script is an object with an actions list.")
    if script.get("reason") is not None:
        if script["reason"] not in REASONS or script["actions"]:
            raise ValueError("A refusal is an empty actions list with a known reason.")
        return script["reason"]
    if not script["actions"]:
        raise ValueError("A script without actions needs a reason.")
    state = ("canvas", "connections", "history", "undo_stack", "redo_stack")
    snapshot = copy.deepcopy([getattr(session, name) for name in state])
    created = {}
    try:
        for index, action in enumerate(script["actions"], 1):
            # The first created object keeps the plain prefix, as one-action sessions name it.
            identifier = prefix if index == 1 else f"{prefix}-{index}"
            call = resolve_action(action, session, created, fuzzy)
            session.execute(command, call, identifier)
            if identifier in session.objects():
                created[index] = identifier
    except ValueError as error:
        for name, value in zip(state, snapshot, strict=True):
            setattr(session, name, value)
        if isinstance(error, ReferenceProblem):
            return error.reason
        raise
    return None


def session_from(diagram):
    """A session holding a diagram given as boxes, shapes and connections between them.

    {"boxes": [{"id", "name", "fields", "methods"}], "shapes": [{"id", "kind", "text"}],
     "connections": [{"from", "to", "label"}]}; positions are laid out on a grid.
    """
    columns, width, height = GRID
    schemas, shapes = [], []
    for index, box in enumerate(diagram.get("boxes", [])):
        schemas.append(
            {
                "id": box["id"],
                "name": box["name"],
                "properties": list(box.get("fields", [])),
                "methods": list(box.get("methods", [])),
                "x": (index % columns) * width,
                "y": (index // columns) * height,
                "h": schema_height(box.get("fields", []), box.get("methods", [])),
                "order": index,
            }
        )
    offset = len(schemas)
    for index, shape in enumerate(diagram.get("shapes", []), offset):
        text = shape.get("text", "")
        shapes.append(
            {
                "id": shape["id"],
                "name": text[:100] or shape_name(shape["kind"]),
                "kind": shape["kind"],
                "text": text,
                "x": (index % columns) * width,
                "y": (index // columns) * height,
                "order": index,
            }
        )
    session = CanvasSession({"schemas": schemas, "shapes": shapes})
    for number, link in enumerate(diagram.get("connections", []), 1):
        session.execute(
            "",
            {
                "name": "connect_schemas",
                "arguments": {
                    "source_id": link["from"],
                    "target_id": link["to"],
                    "label": link.get("label", ""),
                },
            },
            f"link-{number}",
        )
    session.canvas["selected_ids"] = []
    session.undo_stack, session.redo_stack = [], []
    session.history = {"turns": [], "last_created_id": None, "last_edited_id": None}
    session.sync_history()
    return session


def semantic_state(session):
    """The diagram's content, ignoring IDs, positions, sizes and ordering."""
    objects = session.objects()

    def name(identifier):
        return objects[identifier]["name"] if identifier in objects else None

    return {
        "boxes": sorted(
            [b["name"], sorted(b["properties"]), sorted(b["methods"]), b["color"], b["fill"]]
            for b in session.canvas["schemas"]
        ),
        "shapes": sorted(
            [s["kind"], s["text"], s["color"], s["fill"]]
            for s in session.canvas["shapes"]
            if s["kind"] not in ("arrow", "group")
        ),
        "connections": sorted(
            [name(link["source_id"]), name(link["target_id"]), link["label"]]
            for link in session.connections
        ),
    }


SYSTEM = (
    'Turn the command into an edit script {"actions": [...]} for this canvas. Target '
    'arguments are references: {"name": ...}, {"selected": true}, {"last": "created"} or '
    '{"last": "edited"}, {"new": n} for the object action n creates, {"kind": ...}, '
    '{"all": true}, or {"connection": {"from": ref, "to": ref}}. Refuse a request no action '
    'can carry out with {"actions": [], "reason": "unsupported_request"}.'
)


def messages(command, session):
    """Chat messages for a script model: the compact canvas, its links and recent outcomes."""
    from prompt import context_and_recent, dumps

    context, recent, identifiers = context_and_recent(session.canvas, session.history)
    links = [
        [identifiers.get(link["source_id"]), identifiers.get(link["target_id"]), link["label"]]
        for link in session.connections
    ]
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Canvas: {dumps(context)}\nLinks: {dumps(links)}\n"
            f"Recent: {dumps(recent)}\nCommand: {command}",
        },
    ]


def script_text(script):
    from prompt import compact_call, dumps

    if script.get("reason") is not None:
        return dumps({"actions": [], "reason": script["reason"]})
    return dumps({"actions": [compact_call(action) for action in script["actions"]]})


def parse_script(text):
    import json

    from prompt import _unique_object

    script = json.loads(text.strip(), object_pairs_hook=_unique_object)
    if not isinstance(script, dict) or not isinstance(script.get("actions"), list):
        raise ValueError("Expected a script object with an actions list.")
    return script


def _keyed(key, value):
    return {"type": "object", "properties": {key: value}, "required": [key], "additionalProperties": False}


def script_schema():
    """A JSON schema that only valid scripts satisfy, for constrained decoding."""
    single = [
        _keyed("name", {"type": "string", "minLength": 1}),
        _keyed("selected", {"const": True}),
        _keyed("last", {"enum": ["created", "edited"]}),
        _keyed("new", {"type": "integer", "minimum": 1}),
        _keyed("id", {"type": "string"}),
    ]
    ends = {"anyOf": single}
    connection = _keyed(
        "connection",
        {"type": "object", "properties": {"from": ends, "to": ends, "label": {"type": "string"}},
         "required": ["from", "to"], "additionalProperties": False},
    )
    tools = []
    for name, (model, _) in ACTION_MODELS.items():
        if name == "no_action":
            continue
        schema = model.model_json_schema()
        properties = {}
        for key, value in schema["properties"].items():
            value = {k: v for k, v in value.items() if k not in ("title", "description", "default")}
            if key in SINGLE:
                value = {"anyOf": [*single, connection]}
            elif key == PLURAL:
                items = {"anyOf": [*single, _keyed("kind", {"type": "string"}),
                                   _keyed("all", {"const": True}), connection]}
                value = {**value, "items": items}
            properties[key] = value
        tools.append(
            {"type": "object", "properties": {"name": {"const": name}, "arguments": {
                "type": "object", "properties": properties, "required": schema.get("required", []),
                "additionalProperties": False}}, "required": ["name", "arguments"],
             "additionalProperties": False}
        )
    return {"anyOf": [
        {"type": "object", "properties": {"actions": {"type": "array", "items": {"anyOf": tools}, "minItems": 1}},
         "required": ["actions"], "additionalProperties": False},
        {"type": "object", "properties": {"actions": {"type": "array", "maxItems": 0},
                                          "reason": {"enum": list(REASONS)}},
         "required": ["actions", "reason"], "additionalProperties": False},
    ]}
