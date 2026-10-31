"""Checks for the new canvas tools (extensions.py): python test_extensions.py"""

import json
import re
from pathlib import Path

from contract import messages, parse_script, run_script, script_schema, script_text, semantic_state, session_from
from extensions import COMPONENTS


def canvas():
    return session_from({
        "boxes": [{"id": "b1", "name": "Orders", "fields": ["id"], "methods": []}],
        "shapes": [{"id": "s1", "kind": "rectangle", "text": "API"}, {"id": "s2", "kind": "ellipse", "text": "Client"}],
        "connections": [{"from": "s2", "to": "s1", "label": "HTTPS"}, {"from": "s1", "to": "b1", "label": "SQL"}],
    })


def run(session, actions, command="test"):
    outcome = run_script({"actions": actions}, session, command)
    assert outcome is None, outcome
    return session


def by_name(session, name):
    return next(o for o in session.objects().values() if o["name"] == name)


def test_components_match_the_page():
    source = Path(__file__).parent.parent / "observe" / "src" / "components.tsx"
    if source.exists():
        icons = re.findall(r"^\t(\w+):", source.read_text().split("const ICONS = {")[1].split("} as const")[0], re.M)
        assert tuple(icons) == COMPONENTS, set(icons) ^ set(COMPONENTS)


def test_component_create_style_and_label():
    session = run(canvas(), [
        {"name": "create_component", "arguments": {"component": "cache", "label": "Redis", "x": 400, "y": 0}},
        {"name": "connect_schemas", "arguments": {"source_id": {"name": "API"}, "target_id": {"new": 1}, "label": "reads"}},
        {"name": "style_shapes", "arguments": {"shape_ids": [{"name": "Redis"}], "color": "red", "fill": "solid"}},
        {"name": "set_text", "arguments": {"shape_id": {"name": "Redis"}, "text": "Redis cache"}},
    ])
    cache = by_name(session, "Redis cache")
    assert cache["kind"] == "component:cache" and cache["color"] == "red" and cache["fill"] == "none"
    assert ["API", "Redis cache", "reads"] in semantic_state(session)["connections"]
    assert session.history["last_edited_id"] == cache["id"]


def test_frame_shapes_and_moves():
    session = run(canvas(), [{"name": "frame_shapes", "arguments": {"shape_ids": [{"name": "API"}, {"name": "Orders"}], "name": "Backend"}}])
    frame = by_name(session, "Backend")
    api, orders = by_name(session, "API"), by_name(session, "Orders")
    assert frame["kind"] == "frame" and api["parent_id"] == orders["parent_id"] == frame["id"]
    for child in (api, orders):
        assert frame["x"] < child["x"] and child["x"] + child["w"] < frame["x"] + frame["w"]
    assert session.history["last_created_id"] == frame["id"]
    x = api["x"]
    run(session, [{"name": "move_shapes", "arguments": {"shape_ids": [{"name": "Backend"}], "dx": 50, "dy": 0}}])
    assert by_name(session, "API")["x"] == x + 50
    run(session, [{"name": "delete_shapes", "arguments": {"shape_ids": [{"name": "Backend"}]}}])
    assert [s[1] for s in semantic_state(session)["shapes"]] == ["Client"] and not semantic_state(session)["boxes"]


def test_add_to_frame_fits():
    session = run(canvas(), [
        {"name": "frame_shapes", "arguments": {"shape_ids": [{"name": "API"}], "name": "Edge"}},
        {"name": "add_to_frame", "arguments": {"shape_ids": [{"name": "Client"}], "frame_id": {"name": "Edge"}}},
    ])
    frame, client = by_name(session, "Edge"), by_name(session, "Client")
    assert client["parent_id"] == frame["id"]
    assert frame["x"] <= client["x"] and client["x"] + client["w"] <= frame["x"] + frame["w"]
    for shapes, frame in (("Edge", "Edge"), ("Orders", "API")):  # itself; not a frame
        try:
            run_script({"actions": [{"name": "add_to_frame", "arguments": {
                "shape_ids": [{"name": shapes}], "frame_id": {"name": frame}}}]}, session, "x")
            raise AssertionError(f"{shapes} went into {frame}")
        except ValueError:
            pass


def test_auto_layout_follows_arrows():
    for direction, axis in (("left_to_right", "x"), ("top_to_bottom", "y")):
        session = run(canvas(), [{"name": "auto_layout", "arguments": {"direction": direction}}])
        client, api, orders = (by_name(session, n) for n in ("Client", "API", "Orders"))
        assert client[axis] < api[axis] < orders[axis], (direction, client[axis], api[axis], orders[axis])


def test_scripts_render_and_parse():
    session = canvas()
    script = {"actions": [
        {"name": "create_component", "arguments": {"component": "database", "label": "Postgres"}},
        {"name": "frame_shapes", "arguments": {"shape_ids": [{"new": 1}, {"name": "Orders"}], "name": "Data"}},
        {"name": "auto_layout", "arguments": {"direction": "left_to_right"}},
    ]}
    text = script_text(script)
    assert parse_script(text) == json.loads(text)
    run_script(parse_script(text), session, "add postgres, frame it with orders and tidy up")
    content = messages("make it blue", session)[1]["content"]
    assert "component:database" in content and "Recent" in content
    json.dumps(script_schema())  # the grammar still builds with the new tools


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("ok", test.__name__)
