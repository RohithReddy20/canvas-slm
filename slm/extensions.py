"""Canvas tools beyond the frozen v7 contract, as the observation sessions asked for them.

- create_component: a system-design icon with a label (observe/src/components.tsx);
- frame_shapes: a new frame around shapes, which become its children;
- add_to_frame: shapes move into an existing frame, which then fits its content;
- auto_layout: a layered layout along the arrows, left to right or top to bottom.

actions.py and sessions.py stay frozen: importing this module teaches CanvasSession the new
actions by wrapping `_execute` (old actions run unchanged), lets components take color,
opacity and text, and SCRIPT_ACTIONS lists every tool a script may use. The geometry
follows the tldraw implementation in observe/src (frame padding, layout gaps); exact pixel
parity is not needed, since scripts name the action, not coordinates.

Two tldraw behaviours are copied for the new tools only, so v7 actions run as frozen:
a new component or frame created inside a frame's area becomes that frame's child; and
styles pass to children through groups but not through frames (slm/observed.py parity
checks the simulator against the logged tldraw sessions).
"""

import copy
from typing import Literal

from pydantic import Field

import sessions
from actions import ACTION_MODELS, CanvasShape, Name, Names, Number, StrictModel

# Kept in step with ICONS in observe/src/components.tsx (test_extensions.py checks it).
COMPONENTS = (
    "user", "browser", "mobile", "server", "service", "database", "cache", "queue", "stream",
    "load_balancer", "api_gateway", "cdn", "dns", "object_storage", "function", "container",
    "auth", "monitoring", "search", "firewall", "cloud", "external_api", "worker", "scheduler",
    "data_warehouse", "notification", "payment", "email", "ml_model", "document",
)
COMPONENT_SIZE = (120, 110)
FRAME_PADDING = 48
LAYER_GAP, ITEM_GAP = 220, 70


Side = Literal["right", "left", "above", "below"]
PLACEMENT_GAP = 80


class CreateComponent(StrictModel):
    component: Literal[COMPONENTS]
    label: str = Field(default="", max_length=100)
    near: Name | None = Field(default=None, description="The object it goes next to.")
    side: Side = Field(default="right", description="Which side of `near` it goes on.")
    x: Number | None = Field(default=None, description="Only when the person gives coordinates.")
    y: Number | None = Field(default=None, description="Only when the person gives coordinates.")


class FrameShapes(StrictModel):
    shape_ids: Names = Field(min_length=1)
    name: str = Field(default="", max_length=100)


class AddToFrame(StrictModel):
    shape_ids: Names = Field(min_length=1)
    frame_id: Name


class AutoLayout(StrictModel):
    direction: Literal["left_to_right", "top_to_bottom"]
    shape_ids: Names | None = None


EXTRA_ACTION_MODELS = {
    "create_component": (
        CreateComponent,
        "Create a system-design component (an icon with a label), e.g. a database or a cache.",
    ),
    "frame_shapes": (FrameShapes, "Put a new frame around shapes; they move inside it."),
    "add_to_frame": (AddToFrame, "Move shapes into an existing frame, which grows to fit."),
    "auto_layout": (
        AutoLayout,
        "Lay the diagram (or the given shapes) out along its arrows.",
    ),
}
SCRIPT_ACTIONS = {**ACTION_MODELS, **EXTRA_ACTION_MODELS}


def is_component(shape):
    return str(shape.get("kind", "")).startswith("component:")


def _validated(action, session):
    """Checked arguments of a new action, raising ValueError like actions.validate_call."""
    if set(action) != {"name", "arguments"}:
        raise ValueError("Expected a known function name and arguments.")
    model = EXTRA_ACTION_MODELS[action["name"]][0]
    arguments = model.model_validate(action["arguments"]).model_dump()
    objects = session.objects()
    ids = arguments.get("shape_ids") or []
    if len(set(ids)) != len(ids):
        raise ValueError("Shape IDs must be unique.")
    if any(identifier not in objects for identifier in ids):
        raise ValueError("Unknown canvas ID in shape targets.")
    if arguments.get("near") is not None and arguments["near"] not in objects:
        raise ValueError("Unknown canvas ID for near.")
    if action["name"] == "add_to_frame":
        frame = objects.get(arguments["frame_id"])
        if frame is None or frame.get("kind") != "frame":
            raise ValueError("The target is not a frame.")
        ancestors, current = set(), frame
        while current is not None:
            ancestors.add(current["id"])
            current = objects.get(current["parent_id"])
        if ancestors & set(ids):
            raise ValueError("A frame cannot go inside itself.")
    return {"name": action["name"], "arguments": arguments}


def _roots(session, ids):
    """The given objects without those whose ancestor is also given, in the given order."""
    objects, chosen = session.objects(), set(ids)

    def nested(identifier):
        parent = objects[identifier]["parent_id"]
        while parent:
            if parent in chosen:
                return True
            parent = objects[parent]["parent_id"]
        return False

    return [identifier for identifier in ids if not nested(identifier)]


def _top(session, identifier):
    objects = session.objects()
    while objects[identifier]["parent_id"]:
        identifier = objects[identifier]["parent_id"]
    return identifier


def _bounds(items):
    return (min(i["x"] for i in items), min(i["y"] for i in items),
            max(i["x"] + i["w"] for i in items), max(i["y"] + i["h"] for i in items))


def _shift(session, identifier, dx, dy):
    objects = session.objects()
    for target in session.descendants([identifier]):
        objects[target]["x"] += dx
        objects[target]["y"] += dy


def _frame_at(session, x, y, exclude=()):
    """The frontmost frame whose area holds the point, as tldraw parents a new shape."""
    frames = [
        o for o in session.objects().values()
        if o.get("kind") == "frame" and o["id"] not in exclude
        and o["x"] <= x <= o["x"] + o["w"] and o["y"] <= y <= o["y"] + o["h"]
    ]
    return max(frames, key=lambda o: o["order"])["id"] if frames else None


def place(session, width, height, near=None, side="right"):
    """A free spot for a new object: beside `near` (or the last created or edited object,
    else right of everything) on `side`, moved on along that side until it overlaps no
    other object (frames do not count).
    observe/src/layout.ts places the same way on the real canvas."""
    objects = session.objects()
    anchor = objects.get(near) or objects.get(session.history.get("last_created_id")) \
        or objects.get(session.history.get("last_edited_id"))
    # Frames are room to place in, not obstacles: a shape placed inside one joins it.
    tops = [o for o in objects.values() if o.get("kind") not in ("arrow", "group", "frame")]
    if anchor is None:
        if not tops:
            return 0.0, 0.0
        right = max(o["x"] + o["w"] for o in tops)
        anchor = {"x": right - width, "y": min(o["y"] for o in tops), "w": width, "h": height}
        side = "right"
    step = {"right": (anchor["w"] + PLACEMENT_GAP, 0), "left": (-(width + PLACEMENT_GAP), 0),
            "above": (0, -(height + PLACEMENT_GAP)), "below": (0, anchor["h"] + PLACEMENT_GAP)}[side]
    x = anchor["x"] + step[0] + ((anchor["w"] - width) / 2 if side in ("above", "below") else 0)
    y = anchor["y"] + step[1] + ((anchor["h"] - height) / 2 if side in ("left", "right") else 0)
    advance = {"right": (width + PLACEMENT_GAP, 0), "left": (-(width + PLACEMENT_GAP), 0),
               "above": (0, -(height + PLACEMENT_GAP)), "below": (0, height + PLACEMENT_GAP)}[side]
    for _ in range(40):
        if not any(x < o["x"] + o["w"] and o["x"] < x + width and y < o["y"] + o["h"] and o["y"] < y + height
                   for o in tops):
            break
        x, y = x + advance[0], y + advance[1]
    return float(round(x)), float(round(y))


def _fit_frame(session, frame_id):
    objects = session.objects()
    children = [o for o in objects.values() if o["parent_id"] == frame_id]
    if not children:
        return
    left, top, right, bottom = _bounds(children)
    frame = objects[frame_id]
    frame["x"], frame["y"] = left - FRAME_PADDING, top - FRAME_PADDING
    frame["w"], frame["h"] = right - left + 2 * FRAME_PADDING, bottom - top + 2 * FRAME_PADDING


def auto_layout(session, only, direction):
    """Moves top-level objects into layers along the arrows; returns the first node's ID."""
    objects = session.objects()
    chosen = only if only is not None else list(objects)
    nodes = []
    for identifier in sorted(chosen, key=lambda i: (objects[i]["order"], i)):
        top = _top(session, identifier)
        if objects[top].get("kind") != "arrow" and top not in nodes:
            nodes.append(top)
    if len(nodes) < 2:
        raise ValueError("There is nothing to lay out.")
    members = set(nodes)
    edges = []
    for link in session.connections:
        source, target = _top(session, link["source_id"]), _top(session, link["target_id"])
        if source != target and source in members and target in members:
            edges.append((source, target))
    linked = {end for edge in edges for end in edge}

    layer, visiting = {}, set()

    def depth(identifier):
        if identifier in layer:
            return layer[identifier]
        visiting.add(identifier)
        best = 0
        for source, target in edges:
            if target == identifier and source not in visiting:
                best = max(best, depth(source) + 1)
        visiting.discard(identifier)
        layer[identifier] = best
        return best

    for node in nodes:
        if node in linked:
            depth(node)
    layers = []
    for node in nodes:
        if node in linked:
            while len(layers) <= layer[node]:
                layers.append([])
            layers[layer[node]].append(node)
    for _ in range(3):  # order each layer by where its incoming arrows come from
        for i in range(1, len(layers)):
            index = {identifier: k for k, identifier in enumerate(layers[i - 1])}

            def score(identifier, index=index):
                ins = [index[s] for s, t in edges if t == identifier and s in index]
                return sum(ins) / len(ins) if ins else float("inf")

            layers[i].sort(key=score)

    across = direction == "left_to_right"
    start_x = min(objects[n]["x"] for n in nodes)
    start_y = min(objects[n]["y"] for n in nodes)
    length = lambda i: objects[i]["h"] if across else objects[i]["w"]  # noqa: E731
    thick = lambda i: objects[i]["w"] if across else objects[i]["h"]  # noqa: E731
    spans = [sum(length(i) for i in group) + ITEM_GAP * (len(group) - 1) for group in layers]
    widest = max(spans, default=0)
    places, offset = {}, 0
    for group, span in zip(layers, spans):
        thickness = max(thick(i) for i in group)
        along = (widest - span) / 2
        for identifier in group:
            box = objects[identifier]
            inner = offset + (thickness - thick(identifier)) / 2
            places[identifier] = (inner, along) if across else (along, inner)
            along += length(identifier) + ITEM_GAP
        offset += thickness + LAYER_GAP
    along = 0
    for node in nodes:
        if node not in linked:
            places[node] = (along, widest + LAYER_GAP if layers else 0) if across else (offset, along)
            along += thick(node) if across else length(node)
            along += ITEM_GAP
    for identifier, (x, y) in places.items():
        box = objects[identifier]
        _shift(session, identifier, start_x + x - box["x"], start_y + y - box["y"])
    return nodes[0]


def _finish(session, command, action, before, target, made):
    """The bookkeeping CanvasSession._execute does after a document edit."""
    session.normalize_context()
    if session.document() != before:
        session.undo_stack.append(before)
        session.redo_stack.clear()
    if made:
        session.history["last_created_id"] = target
    if target:
        session.history["last_edited_id"] = target
    session.sync_history()
    recorded = copy.deepcopy(action)
    if recorded["arguments"].get("shape_ids", []) is None:  # frozen recent_context wants a list
        del recorded["arguments"]["shape_ids"]
    entry = {"command": command, "action": recorded}
    if made:
        entry["created_id"] = target
    session.history["turns"] = [*session.history["turns"], entry][-3:]
    session.canvas = sessions.Canvas.model_validate(session.canvas).model_dump()


def _execute_extra(session, command, action, created_id):
    action = _validated(action, session)
    name, args = action["name"], action["arguments"]
    before = session.document()
    objects = session.objects()
    target, made = None, False
    if name in ("create_component", "frame_shapes") and created_id in objects:
        raise ValueError("Creation reuses an ID.")
    if name == "create_component":
        width, height = COMPONENT_SIZE
        if args["x"] is None or args["y"] is None:
            args["x"], args["y"] = place(session, width, height, args["near"], args["side"])
        session.canvas["shapes"].append(
            CanvasShape(
                id=created_id,
                name=args["label"][:100] or args["component"],
                kind=f"component:{args['component']}",
                text=args["label"],
                x=args["x"],
                y=args["y"],
                w=width,
                h=height,
                order=len(objects),
                parent_id=_frame_at(session, args["x"], args["y"]),
            ).model_dump()
        )
        target, made = created_id, True
        session.canvas["selected_ids"] = [created_id]
    elif name == "frame_shapes":
        roots = _roots(session, args["shape_ids"])
        left, top, right, bottom = _bounds([objects[i] for i in roots])
        lowest = min((o["order"] for o in objects.values()), default=0)
        session.canvas["shapes"].append(
            CanvasShape(
                id=created_id,
                name=args["name"][:100] or "frame",
                kind="frame",
                text=args["name"],
                x=left - FRAME_PADDING,
                y=top - FRAME_PADDING,
                w=right - left + 2 * FRAME_PADDING,
                h=bottom - top + 2 * FRAME_PADDING,
                order=lowest - 1,  # sent to the back, behind its children
                parent_id=_frame_at(session, left - FRAME_PADDING, top - FRAME_PADDING, exclude=set(roots)),
            ).model_dump()
        )
        for identifier in roots:
            session.objects()[identifier]["parent_id"] = created_id
        target, made = created_id, True
        session.canvas["selected_ids"] = [created_id]
    elif name == "add_to_frame":
        for identifier in _roots(session, args["shape_ids"]):
            objects[identifier]["parent_id"] = args["frame_id"]
        _fit_frame(session, args["frame_id"])
        target = args["frame_id"]
        session.canvas["selected_ids"] = [target]
    else:
        only = _roots(session, args["shape_ids"]) if args["shape_ids"] else None
        target = auto_layout(session, only, args["direction"])
    _finish(session, command, action, before, target, made)


def _set_component_text(session, command, action):
    arguments = action["arguments"]
    if set(arguments) != {"shape_id", "text"} or len(arguments["text"]) > 100:
        raise ValueError("A component label has at most 100 characters.")
    before = session.document()
    shape = session.objects()[arguments["shape_id"]]
    shape["text"] = arguments["text"]
    shape["name"] = arguments["text"][:100] or shape["kind"].split(":", 1)[1]
    session.canvas["selected_ids"] = [shape["id"]]
    _finish(session, command, action, before, shape["id"], False)


def _style_targets(self, targets):
    """Styled shapes: the targets, and the children of groups among them (not of frames)."""
    objects, found, stack = self.objects(), [], list(targets)
    while stack:
        identifier = stack.pop()
        shape = objects[identifier]
        if shape.get("kind") == "group":
            stack += [o["id"] for o in objects.values() if o["parent_id"] == identifier]
        elif identifier not in found:
            found.append(identifier)
    has_frame_children = any(
        o["parent_id"] and objects[o["parent_id"]].get("kind") == "frame" for o in objects.values()
    )
    if not has_frame_children:  # nothing new on the canvas: exactly the frozen behaviour
        return _original_style_targets(self, targets)
    return [objects[i] for i in sorted(found)]


_original_execute = sessions.CanvasSession._execute
_original_style_targets = sessions.CanvasSession.style_targets
_original_supports_style = sessions.supports_style


def _execute(self, command, action, created_id):
    if action and action.get("name") in EXTRA_ACTION_MODELS:
        return _execute_extra(self, command, action, created_id)
    if (action and action.get("name") == "set_text"
            and is_component(self.objects().get((action.get("arguments") or {}).get("shape_id"), {}))):
        return _set_component_text(self, command, action)
    return _original_execute(self, command, action, created_id)


def _supports_style(shape, style):
    if is_component(shape):  # icons take a color and opacity, as in tldraw
        return style in ("color", "opacity")
    return _original_supports_style(shape, style)


sessions.CanvasSession._execute = _execute
sessions.CanvasSession.style_targets = _style_targets
sessions.supports_style = _supports_style
