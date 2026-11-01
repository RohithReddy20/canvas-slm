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
