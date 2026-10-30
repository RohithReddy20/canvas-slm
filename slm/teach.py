"""Label real diagram follow-ups with a teacher model, checked against the simulator.

For each user message that followed diagram code in a real conversation, the teacher
decides whether it asks for a concrete diagram edit, turns the diagram into a canvas, and
writes the edit as a script (see contract.py). Scripts are applied with `run_script` so
only edits the simulator can carry out become test cases.

    python teach.py --input teach-input.jsonl --output /content/runs/teach
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from actions import ACTION_MODELS, Color
from contract import run_script, semantic_state, session_from

CATEGORIES = ("edit", "unsupported_edit", "fix_error", "new_diagram", "other")
EDIT_KINDS = ("atomic", "compound", "high_level")


def tool_lines():
    lines = []
    for name, (model, description) in ACTION_MODELS.items():
        if name == "no_action":
            continue
        fields = ", ".join(
            field + ("" if info.is_required() else "?")
            for field, info in model.model_fields.items()
        )
        lines.append(f"- {name}({fields}): {description}")
    return "\n".join(lines)


PROMPT = f"""You label real user messages sent to a diagramming assistant. Each message followed \
diagram code the assistant had produced (Mermaid, PlantUML, Graphviz). Decide what the user \
wants and, for a concrete edit, express it as actions on a canvas.

Categories:
- edit: changes the existing diagram's content (add, remove, rename, connect, restyle...).
- unsupported_edit: a diagram edit the tools below cannot express (layout direction, \
themes, sequence-diagram ordering, notes on lines...). Say what is missing in "reason".
- fix_error: reports that the code fails to render or has a syntax error.
- new_diagram: asks for a different or new diagram instead of changing this one.
- other: anything else (questions, code, prose, translations).

For "edit", also give edit_kind: atomic (one action), compound (several concrete actions), \
or high_level (needs invented content, like "add more detail"). Give a canvas and a script \
only for atomic and compound edits; the script must do exactly what the user asked.

Canvas: the diagram before the edit, as JSON:
{{"boxes": [{{"id": "b1", "name": "User", "fields": ["id", "email"], "methods": ["login()"]}}],
 "shapes": [{{"id": "s1", "kind": "rectangle", "text": "Start"}}],
 "connections": [{{"from": "b1", "to": "s1", "label": ""}}]}}
Classes, entities, tables and participants are boxes; flowchart steps are shapes \
(rectangle for steps, diamond for decisions, ellipse for start/end, text or note otherwise). \
Every connection joins two ids. Keep names exactly as written in the code.

Script: {{"actions": [{{"name": ..., "arguments": {{...}}}}]}} with these tools ("?" marks \
optional arguments):
{tool_lines()}
Target arguments (schema_id, shape_id, source_id, target_id, and each item of shape_ids) are \
references, never raw ids:
{{"name": "User"}} the object with that name; {{"new": 2}} the object created by action 2 of \
this script; {{"kind": "diamond"}} or {{"all": true}} for plural targets; \
{{"connection": {{"from": {{"name": "A"}}, "to": {{"name": "B"}}}}}} for the arrow between two \
objects (delete it with delete_shapes). rename_schema renames boxes; set_text relabels shapes. \
Colors: {", ".join(Color.__args__)}.

Example. Diagram: classDiagram with User and Order, User --> Order. Message: "add a Payment \
class with amount and link it to Order". Answer:
{{"category": "edit", "edit_kind": "compound", "canvas": {{"boxes": [{{"id": "b1", "name": "User", \
"fields": [], "methods": []}}, {{"id": "b2", "name": "Order", "fields": [], "methods": []}}], \
"shapes": [], "connections": [{{"from": "b1", "to": "b2", "label": ""}}]}}, "script": {{"actions": \
[{{"name": "create_schema_box", "arguments": {{"name": "Payment", "fields": ["amount"], \
"methods": []}}}}, {{"name": "connect_schemas", "arguments": {{"source_id": {{"name": "Order"}}, \
"target_id": {{"new": 1}}, "label": ""}}}}]}}, "reason": "adds a class and a link"}}

Reply with one JSON object only: {{"category", "edit_kind", "canvas", "script", "reason"}}, \
using null where a field does not apply."""


def message(row):
    context = "\n".join(
        f"{label}: {row[key]}"
        for label, key in (
            ("First request", "first_request"),
            ("Previous request", "previous_request"),
        )
        if row[key]
    )
    return (
        f"{context}\nDiagram code the assistant produced:\n```\n{row['diagram']}\n```\n"
        f"User's next message: {row['text']}"
    )


def parse(text):
    # The chat template opens the thinking block in the prompt, so only its end appears here.
    text = text.rsplit("</think>", 1)[-1].strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced[-1] if fenced else text[text.find("{") : text.rfind("}") + 1]
    answer = json.loads(candidate)
    if answer.get("category") not in CATEGORIES:
        raise ValueError(f"Unknown category: {answer.get('category')!r}")
    return answer


def check(row, answer):
    """Whether the canvas builds and the script applies; the states before and after."""
    if answer["category"] != "edit" or answer.get("edit_kind") not in ("atomic", "compound"):
        return {}
    try:
        session = session_from(answer["canvas"])
        before = semantic_state(session)
        refused = run_script(answer["script"], session, row["text"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return {"applied": False, "error": str(error)[:300]}
    if refused:
        return {"applied": False, "error": refused}
    after = semantic_state(session)
    return {"applied": before != after, "before": before, "after": after,
            "error": None if before != after else "no visible change"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=12288)
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(args.input)]
    args.output.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model, max_model_len=20480, gpu_memory_utilization=0.92, max_num_seqs=64)
    # Qwen's recommended sampling in thinking mode; greedy decoding loops.
    sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_tokens)
    counts, start = Counter(), time.time()
    with open(args.output / "labels.jsonl", "w") as labels:
        for offset in range(0, len(rows), args.chunk):
            chunk = rows[offset : offset + args.chunk]
            outputs = llm.chat(
                [
                    [{"role": "system", "content": PROMPT}, {"role": "user", "content": message(r)}]
                    for r in chunk
                ],
                sampling,
                chat_template_kwargs={"enable_thinking": True},
                use_tqdm=False,
            )
            for row, output in zip(chunk, outputs, strict=True):
                text = output.outputs[0].text
                record = {**row, "finish": output.outputs[0].finish_reason}
                try:
                    answer = parse(text)
                    record.update(answer=answer, **check(row, answer))
                    counts[answer["category"]] += 1
                    counts[f"kind:{answer.get('edit_kind')}"] += answer["category"] == "edit"
                    counts["applied"] += bool(record.get("applied"))
                except (ValueError, json.JSONDecodeError) as error:
                    record.update(answer=None, parse_error=str(error)[:300], raw=text)
                    counts["unparsed"] += 1
                labels.write(json.dumps(record, ensure_ascii=False) + "\n")
            labels.flush()
            done = offset + len(chunk)
            print(f"{done}/{len(rows)} in {time.time() - start:.0f} s: {dict(counts)}", flush=True)
            if offset == 0 and counts["unparsed"] == len(chunk):
                raise SystemExit("No answer in the first chunk could be parsed.")
    (args.output / "summary.json").write_text(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
