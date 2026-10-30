"""Model-agnostic prompt and reply format for the canvas action contract.

The canvas, recent outcomes and command carry the same information as
`actions.messages_for`, as compact JSON without default-valued keys. The reply is
one JSON call using the short IDs from the canvas, mapped back and validated by
`actions.validate_call`.
"""

import json

from actions import (
    ACTION_MODELS,
    execution_guard,
    model_call,
    model_canvas,
    no_action,
    recent_context,
    validate_call,
)
from sessions import shape_name

SYSTEM = (
    "Turn the command into exactly one canvas action. Reply with one JSON object "
    '{"name": ..., "arguments": {...}} that uses the short IDs shown in Canvas.'
)
DEFAULTS = {
    "color": "black",
    "fill": "none",
    "opacity": 1,
    "parent_id": None,
    "rotation": 0,
    "text": "",
}
TARGET_KEYS = ("schema_id", "source_id", "target_id", "shape_id")
# lab._validate_prediction turns these validation errors into missing targets.
GUARDED_ERRORS = (
    "Cannot remove a property that does not exist.",
    "Cannot remove a method that does not exist.",
    "No canvas history is available for that operation.",
)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def number(value):
    return int(value) if isinstance(value, float) and value.is_integer() else value


def compact_object(item):
    kept = {
        key: number(value)
        for key, value in item.items()
        if key not in DEFAULTS or DEFAULTS[key] != value
    }
    if "kind" in item and item["name"] == (
        item["text"][:100] or shape_name(item["kind"])
    ):
        del kept["name"]
    return kept


def compact_call(call):
    model = ACTION_MODELS[call["name"]][0]
    arguments = {}
    for key, value in call["arguments"].items():
        field = model.model_fields.get(key)
        if field is not None and not field.is_required() and field.default == value:
            continue
        arguments[key] = number(value)
    return {"name": call["name"], "arguments": arguments}


def messages(command, canvas, history=None):
    context, identifiers = model_canvas(canvas)
    context["schemas"] = [compact_object(box) for box in context["schemas"]]
    context["shapes"] = [compact_object(shape) for shape in context["shapes"]]
    context["camera"] = {key: number(value) for key, value in context["camera"].items()}
    recent = recent_context(history or [], identifiers)
    recent["turns"] = [
        {
            "command": turn["command"],
            "outcome": turn["outcome"]
            if turn["outcome"] == "rejected"
            else compact_call(turn["outcome"]),
        }
        for turn in recent["turns"]
    ]
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Canvas: {dumps(context)}\nRecent: {dumps(recent)}"
            f"\nCommand: {command}",
        },
    ]


def reply(call, canvas):
    return dumps(compact_call(model_call(call, canvas)))


def parse(text, canvas):
    call = json.loads(text.strip(), object_pairs_hook=_unique_object)
    if (
        not isinstance(call, dict)
        or set(call) != {"name", "arguments"}
        or not isinstance(call["name"], str)
        or not isinstance(call["arguments"], dict)
    ):
        raise ValueError("Expected one JSON call with a name and arguments.")
    _, identifiers = model_canvas(canvas)
    originals = {alias: identifier for identifier, alias in identifiers.items()}
    arguments = dict(call["arguments"])
    for key in TARGET_KEYS:
        if isinstance(arguments.get(key), str):
            arguments[key] = originals.get(arguments[key], arguments[key])
    if isinstance(arguments.get("shape_ids"), list):
        arguments["shape_ids"] = [
            originals.get(value, value) if isinstance(value, str) else value
            for value in arguments["shape_ids"]
        ]
    return validate_call({"name": call["name"], "arguments": arguments}, canvas)


def predict(text, command, canvas, history, guarded):
    """The validated action for a model reply, as lab._validate_prediction scores it."""
    try:
        result = {"prediction": parse(text, canvas), "error": None}
    except ValueError as error:
        result = {"prediction": None, "error": str(error)}
    if guarded:
        if result["error"] in GUARDED_ERRORS:
            result = {"prediction": no_action("missing_target"), "error": None}
        if result["prediction"] is not None:
            result["prediction"] = execution_guard(
                command, result["prediction"], canvas, session_history(history, canvas)
            )
    return result


def session_history(history, canvas):
    """The mapping execution_guard reads, also for the list form `messages` accepts."""
    if isinstance(history, dict) or not history:
        return history or {}
    _, identifiers = model_canvas(canvas)
    references = recent_context(history, {key: key for key in identifiers})
    return {
        "turns": history,
        "last_created_id": references["last_created_id"],
        "last_edited_id": references["last_edited_id"],
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate argument: {key}")
        result[key] = value
    return result
