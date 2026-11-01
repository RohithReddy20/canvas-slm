"""The relaxed real-case check with several accepted outcomes and undirected arrows."""

from real_match import matches, relaxed

BEFORE = {"boxes": [["A", [], [], "black", "none"], ["B", [], [], "black", "none"]], "shapes": [], "connections": []}
A_TO_B = {**BEFORE, "connections": [["A", "B", "uses"]]}
B_TO_A = {**BEFORE, "connections": [["B", "A", "uses"]]}
FIELD = {**BEFORE, "boxes": [["A", ["x"], [], "black", "none"], ["B", [], [], "black", "none"]]}


def test_direction_matters_unless_the_row_says_otherwise():
    assert relaxed(BEFORE, A_TO_B, A_TO_B, "connect a to b")
    assert not relaxed(BEFORE, B_TO_A, A_TO_B, "connect a to b")
    assert relaxed(BEFORE, B_TO_A, A_TO_B, "connect a to b", directed=False)
    assert relaxed(BEFORE, B_TO_A, A_TO_B, "label it uses", directed=False)  # a quoted label, either way round


def test_any_accepted_outcome_passes():
    row = {"command": "add x", "after": A_TO_B, "accepted": [FIELD]}
    assert matches(row, BEFORE, A_TO_B)
    assert matches(row, BEFORE, FIELD)
    assert not matches(row, BEFORE, B_TO_A)
    assert matches({**row, "arrow_direction": "any"}, BEFORE, B_TO_A)


def test_additive_restyle_layout_and_refusal():
    from real_match import additive, restyle

    added = {**BEFORE, "boxes": BEFORE["boxes"] + [["C", [], [], "black", "none"]], "connections": [["A", "C", ""]]}
    assert additive(BEFORE, added, anchors=["A"], min_new=1)
    assert not additive(BEFORE, added, anchors=["B"], min_new=1)  # wired to the wrong object
    assert not additive(BEFORE, added, min_new=2)
    assert not additive(A_TO_B, BEFORE)  # removed an arrow
    red = {**BEFORE, "boxes": [["A", [], [], "red", "none"], ["B", [], [], "black", "none"]]}
    renamed = {**BEFORE, "boxes": [["A", [], [], "black", "none"], ["Bee", [], [], "black", "none"]]}
    assert restyle(BEFORE, red) and restyle(BEFORE, renamed)
    assert not restyle(BEFORE, added) and not restyle(BEFORE, BEFORE)
    assert matches({"command": "x", "after": BEFORE, "expect": "refusal"}, BEFORE, BEFORE)
    assert not matches({"command": "x", "after": BEFORE, "expect": "refusal"}, BEFORE, red)
    layout = {"command": "x", "after": BEFORE, "expect": "layout", "layout": "left_to_right"}
    script = {"actions": [{"name": "auto_layout", "arguments": {"direction": "left_to_right"}}]}
    assert matches(layout, BEFORE, BEFORE, script)
    assert not matches(layout, BEFORE, BEFORE, {"actions": []})
