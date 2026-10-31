"""Teacher-written training data, each row kept only when an independent teacher reply agrees.

- requests: a real diagram (assistant code mined from chats) or a diagram drawn for a real
  diagram request, as a canvas plus messages a user might send next, with scripts. A
  message is kept when its script changes the diagram (or refuses an unsupported edit)
  and a separate greedy teacher reply to it, seeing only what the student sees, reaches
  the same diagram content.
- paraphrases: rewordings of v7 training commands. A paraphrase is kept when the
  separate reply leaves the canvas exactly as the row's label does; the original
  command goes through the same check, which measures the check itself.

Inputs come from mined conversations (mine_chats.py) without the real test's
conversations, prepared off the GPU:

    python teach_data.py inputs --conversations conversations.jsonl.gz \\
        --exclude real-test-v1.jsonl --output teach-inputs/
    python teach_data.py run --data /content/data/v7 --input /content/input \\
        --output /content/runs/teach-data --hours 6
"""

import argparse
import copy
import gzip
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from actions import Color
from contract import messages, run_script, semantic_state, session_from
from convert import session_for, to_script
from evaluate_scripts import FAILURES, apply
from teach import tool_lines

REFERENCES = """Target arguments (schema_id, shape_id, source_id, target_id, and each item of \
shape_ids) are references, never raw values:
{"name": "User"} the object with that name or text, spelled as on the canvas;
{"it": true} what the command calls "it", "this" or "the selected shape": the one selected object;
{"selected": true} the current selection as a group;
{"last": "created"} or {"last": "edited"} the object created or edited most recently;
{"new": 2} the object that action 2 of this script creates;
{"kind": "diamond"} every object of a kind ("schema" for boxes) and {"all": true} every \
object, for shape_ids only;
{"connection": {"from": ref, "to": ref}} the arrow between two objects;
{"id": "box1"} an object by its canvas ID, when nothing else names it.
Colors: """ + ", ".join(Color.__args__) + "."

RULES = """Rules of this app: moves and pans default to 100 units, right and down are \
positive. Fill works on boxes, geometric shapes and arrows only; frames take opacity only; \
text and notes take color and opacity; notes cannot be resized. Requests for information, \
explanations or code, and anything no tool does (layout direction, themes, fonts, notes on \
arrows) get {"actions": [], "reason": "unsupported_request"}."""

GUIDE = f"""You turn a command for a diagramming canvas into an edit script.

The canvas is compact JSON: "schemas" are boxes (classes, tables) with fields and methods; \
"shapes" have a kind, text and color; "selected_ids" is the selection. Links are arrows \
between objects as [from, to, label]. Recent holds earlier commands and the last created \
and edited objects.

Reply with {{"actions": [...]}}: the actions, in order, that do exactly what the command \
asks and nothing more. Each action is {{"name": tool, "arguments": {{...}}}} with these \
tools ("?" marks optional arguments):
{tool_lines()}

{REFERENCES}
Point at objects the way the command does: "it" is {{"it": true}} and a name is \
{{"name": ...}}, even when the canvas makes that reference missing or ambiguous; the app \
checks references itself.
{RULES}
Reply with the JSON only, never prose: a command that cannot work on this canvas still gets \
the script it asks for, or the refusal above."""

PARAPHRASE = """You help build training data for an assistant that edits diagrams by voice \
or text. Given a command, what was on the canvas, and the edit it means, write {count} \
other ways a person might say the same command.

Vary them: casual typed (lowercase, short), spoken as a speech recognizer writes it (no \
quotes or symbols, numbers may be words), a polite question, a terse fragment, other verbs \
and word order. Prefer everyday words to the app's tool names.

Each must ask for exactly the same edit on this canvas:
- the same objects, named the same way (quotes optional); when the command points at the \
selection or the last created or edited object, point at it the same way, keeping singular \
and plural as they are;
- the same numbers, colors, names and text (spoken numbers may be words);
- nothing added or dropped.
If the command cannot work here (a missing or ambiguous object, or an unsupported request), \
keep it that way.
Reply with JSON only: {{"paraphrases": ["...", ...]}}"""

def requests_prompt(count):
    return f"""You write realistic training data for an assistant that edits diagrams on a canvas.

The input is diagram code an assistant produced in a real chat, or a real user's request \
for a diagram.
1. Write the diagram as a canvas, drawing the one you would make when given a request:
{{"boxes": [{{"id": "b1", "name": "User", "fields": ["id", "email"], "methods": ["login()"]}}],
 "shapes": [{{"id": "s1", "kind": "rectangle", "text": "Start"}}],
 "connections": [{{"from": "b1", "to": "s1", "label": ""}}]}}
Classes, entities, tables and participants are boxes; other nodes are shapes (rectangle \
for steps, diamond for decisions, ellipse for start and end, text or note otherwise). At \
most 20 objects; names exactly as written. If the input is not about a diagram of boxes, \
shapes and arrows, reply {{"canvas": null}}.
2. Write {count} messages a user might send next to change it, each with the script that \
does exactly what it asks on that canvas: 2 with one change, {count - 3} with two to four \
changes in one message, and 1 that no tool can do.
Write like real users, in English: casual, terse or chatty, loose names ("the order \
table", "user class"), pronouns across clauses ("add a Payment class and link it to \
Order"). Refer only to objects on the canvas or ones the message creates.

Scripts are {{"actions": [...]}}; each action is {{"name": tool, "arguments": {{...}}}} with \
these tools ("?" marks optional arguments):
{tool_lines()}

{REFERENCES}
{RULES}
Reply with JSON only: {{"canvas": {{...}}, "requests": [{{"message": "...", \
"kind": "atomic" | "compound" | "unsupported", "script": {{...}}}}]}}"""


def code_key(code):
    return hashlib.sha1(re.sub(r"\s+", " ", code).strip().encode()).hexdigest()


def english(language, text):
    """English chats, and unlabelled ones (ShareGPT has no language) written in ASCII letters."""
    letters = [c for c in text if c.isalpha()]
    return language == "English" or (
        language == "unknown" and sum(c.isascii() for c in letters) >= 0.98 * len(letters))


def inputs(args):
    """Diagrams, diagram requests and follow-up wording from mined chats, minus the test's chats."""
    from mine_chats import CODE, kinds_of

    excluded = {json.loads(line)["id"].rsplit(":", 1)[0] for line in open(args.exclude)}
    conversations = [json.loads(line) for line in gzip.open(args.conversations, "rt")]
    banned = {
        code_key(code)
        for c in conversations if c["id"] in excluded
        for t in c["turns"] for code in CODE.findall(t["text"])
    }
    diagrams, topics, styles, seen = [], [], [], set(banned)
    for c in conversations:
        if c["id"] in excluded:
            continue
        language = c.get("language") or "unknown"
        for turn in c["turns"]:
            text = turn["text"].strip()
            if not english(language, text):
                continue
            if turn["role"] == "assistant":
                for code in CODE.findall(text):
                    if len(code) >= 80 and code_key(code) not in seen:
                        seen.add(code_key(code))
                        diagrams.append({"id": f"{c['id']}:d{len(diagrams)}", "language": language,
                                         "code": code[:4000]})
            # A bare "diagram" is often geometry or prose, so requests must name a kind.
            elif (turn["index"] == 0 and 20 <= len(text) <= 1200 and code_key(text) not in seen
                  and set(kinds_of(text)) - {"diagram", "tool"}):
                seen.add(code_key(text))
                topics.append({"id": f"{c['id']}:t", "language": language, "text": text})
            elif turn.get("edit_like") and turn.get("after_diagram") and len(text) <= 300:
                styles.append({"language": language, "text": text})
    rng = random.Random(7)
    rng.shuffle(topics)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("diagrams", diagrams), ("topics", topics[: args.topics]), ("styles", styles)):
        with open(args.output / f"{name}.jsonl", "w") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, len(rows), f"{(args.output / f'{name}.jsonl').stat().st_size / 1e6:.1f} MB")


class Teacher:
    def __init__(self, model, max_num_seqs):
        import torch
        from vllm import LLM

        # Turing GPUs have no bfloat16.
        dtype = "auto" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
        self.llm = LLM(model=model, tensor_parallel_size=torch.cuda.device_count(), dtype=dtype,
                       max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=max_num_seqs,
                       limit_mm_per_prompt={"image": 0, "video": 0})

    def replies(self, chats, max_tokens, temperature, thinking=False):
        from vllm import SamplingParams

        # Greedy for checks; Qwen's recommended sampling for writing.
        options = {"top_p": 0.95, "top_k": 20} if temperature else {}
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, **options)
        outputs = self.llm.chat(chats, sampling, chat_template_kwargs={"enable_thinking": thinking},
                                use_tqdm=False)
        # Without its code fence, a reply reads as the student's bare JSON would.
        return [re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", o.outputs[0].text.rsplit("</think>", 1)[-1])
                for o in outputs]


def json_reply(text):
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    return json.loads(fenced[-1] if fenced else text[text.find("{") : text.rfind("}") + 1])


def chat(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def check_chat(command, session):
    return chat(GUIDE, messages(command, session)[1]["content"])


def outcome_of(text, session, command):
    """A teacher reply run on a copy of the session: (outcome, snapshot)."""
    session = copy.deepcopy(session)
    outcome = apply(text, session, command, "new")
    return outcome, session.snapshot()


def situation(session):
    objects = session.objects()
    named = lambda identifier: (  # noqa: E731
        f'"{objects[identifier]["name"]}" ({objects[identifier].get("kind", "box")})'
        if identifier in objects else None
    )
    selected = [named(i) for i in session.canvas["selected_ids"]]
    return "\n".join([
        "On the canvas: " + ", ".join(named(i) for i in list(objects)[:40]),
        "Selected: " + (", ".join(selected) if selected else "nothing"),
        f"Last created: {named(session.history.get('last_created_id')) or 'nothing'}; "
        f"last edited: {named(session.history.get('last_edited_id')) or 'nothing'}",
    ])


def paraphrase_rows(data, count, rng):
    """Converted v7 training rows spread evenly over actions and wording families."""
    from prepare_scripts import replayed

    live = replayed(data / "sessions.jsonl", {"train"})
    groups = defaultdict(list)
    for line in open(data / "examples.jsonl"):
        row = json.loads(line)
        if row["split"] != "train":
            continue
        groups[row["stratum"], row.get("wording_family")].append(row)
    for rows in groups.values():
        rng.shuffle(rows)
    order = sorted(groups, key=lambda _: rng.random())
    picked = []
    while len(picked) < count and any(groups.values()):
        for key in order:
            if groups[key] and len(picked) < count:
                row = groups[key].pop()
                script = to_script(row, live.get(row["id"]))
                if script:
                    picked.append((row, script, live.get(row["id"])))
    return picked


def gold(row, session, command):
    """The label's outcome and canvas for this wording: (refusal reason or "applied", snapshot)."""
    session = copy.deepcopy(session)
    if row["expected"]["name"] == "no_action":
        return row["expected"]["arguments"]["reason"], session.snapshot()
    session.execute(command, row["expected"], "new")
    return "applied", session.snapshot()


def paraphrase_chunk(teacher, chunk, args, out, counts):
    sessions = [live if live is not None else session_for(row) for row, _, live in chunk]
    prompts = [
        chat(PARAPHRASE.format(count=args.paraphrases),
             f"{situation(s)}\nCommand: {row['command']}\nEdit: {json.dumps(script, ensure_ascii=False)}")
        for (row, script, _), s in zip(chunk, sessions, strict=True)
    ]
    candidates = []
    for (row, _, _), session, text in zip(chunk, sessions, teacher.replies(prompts, 1024, 0.8), strict=True):
        try:
            found = [p.strip() for p in json_reply(text)["paraphrases"] if isinstance(p, str)]
        except (*FAILURES, json.JSONDecodeError):
            counts["paraphrase_unparsed"] += 1
            continue
        found = [p for p in dict.fromkeys(found) if p and p.casefold() != row["command"].casefold()]
        candidates += [(row, session, row["command"], True)]
        candidates += [(row, session, p, False) for p in found[: args.paraphrases]]
    replies = teacher.replies([check_chat(c, s) for _, s, c, _ in candidates], 768, 0)
    for (row, session, command, control), text in zip(candidates, replies, strict=True):
        outcome, snapshot = outcome_of(text, session, command)
        agreed = (outcome, snapshot) == gold(row, session, command)
        kind = "control" if control else "paraphrase"
        counts[f"{kind}s"] += 1
        counts[f"{kind}s_agreed"] += agreed
        counts[f"{kind}s_agreed:{row['stratum']}"] += agreed
        counts[f"{kind}s:{row['stratum']}"] += 1
        if not control:
            out.write(json.dumps({"source": row["id"], "command": command, "verified": agreed,
                                  "outcome": outcome}, ensure_ascii=False) + "\n")


def request_chunk(teacher, chunk, styles, args, out, counts, rng):
    prompts = []
    for item in chunk:
        examples = "\n".join(f"- {s['text']}" for s in rng.sample(styles, min(6, len(styles))))
        source = (f"Diagram code:\n```\n{item['code']}\n```" if "code" in item
                  else f"A user's request for a diagram:\n{item['text']}")
        prompts.append(chat(
            requests_prompt(args.requests),
            f"{source}\n\nReal messages people sent after a diagram, for style only:\n{examples}"))
    candidates = []
    for item, text in zip(chunk, teacher.replies(prompts, 4096, 0.7), strict=True):
        try:
            answer = json_reply(text)
            canvas = answer["canvas"]
            if canvas is None:
                counts["not_a_diagram"] += 1
                continue
            session = session_from(canvas)
            requests = list(answer["requests"])
            if len(session.objects()) < 2:
                raise ValueError("A diagram needs at least two objects.")
        except (*FAILURES, json.JSONDecodeError):
            counts["diagram_unparsed"] += 1
            continue
        counts["diagrams"] += 1
        before = semantic_state(session)
        for number, request in enumerate(requests):
            try:
                command, kind = str(request["message"]), request.get("kind")
                done = copy.deepcopy(session)
                refused = run_script(request["script"], done, command)
            except (*FAILURES, json.JSONDecodeError):
                counts["request_invalid"] += 1
                continue
            after = semantic_state(done)
            expected = "unsupported_request" if kind == "unsupported" else "applied"
            if (refused or "applied") != expected or (not refused and after == before):
                counts["request_not_applied"] += 1
                continue
            row = {"id": f"{item['id']}:r{number}", "source": item["id"], "language": item["language"],
                   "edit_kind": kind, "command": command, "canvas": canvas,
                   "script": request["script"], "outcome": refused or "applied", "after": after}
            candidates.append((row, session))
    replies = teacher.replies([check_chat(r["command"], s) for r, s in candidates], 1024, 0)
    for (row, session), text in zip(candidates, replies, strict=True):
        done = copy.deepcopy(session)
        outcome = apply(text, done, row["command"], "new")
        row["verified"] = outcome == row["outcome"] and semantic_state(done) == row["after"]
        counts[f"requests:{row['edit_kind']}"] += 1
        counts[f"requests_agreed:{row['edit_kind']}"] += row["verified"]
        out.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args):
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(args.model, args.max_num_seqs)
    deadline = time.time() + args.hours * 3600
    counts, started = Counter(), time.time()

    def progress(stage, done, total):
        print(f"{stage} {done}/{total} at {(time.time() - started) / 60:.0f} min: "
              f"{json.dumps({k: v for k, v in counts.items() if ':' not in k})}", flush=True)

    def read(name):
        path = args.input / f"{name}.jsonl"
        return [json.loads(line) for line in open(path)] if path.exists() else []

    styles = read("styles")
    sources = read("diagrams")[: args.diagrams] + read("topics")[: args.topics]
    rng.shuffle(sources)
    # Real-style requests come first: they are scarcer than rewordings of v7 commands.
    request_deadline = started + args.hours * 3600 * args.request_share
    # A restart on the same folder continues it, skipping the sources already written.
    path = args.output / "requests.jsonl"
    written = {json.loads(line)["source"] for line in open(path)} if path.exists() else set()
    sources = [s for s in sources if s["id"] not in written]
    with open(path, "a") as out:
        for offset in range(0, len(sources), args.chunk):
            if time.time() > request_deadline:
                break
            request_chunk(teacher, sources[offset : offset + args.chunk], styles, args, out, counts, rng)
            out.flush()
            progress("requests", min(offset + args.chunk, len(sources)), len(sources))
    path = args.output / "paraphrases.jsonl"
    written = {json.loads(line)["source"] for line in open(path)} if path.exists() else set()
    rows = [r for r in paraphrase_rows(args.data, args.paraphrase_rows, rng) if r[0]["id"] not in written]
    with open(path, "a") as out:
        for offset in range(0, len(rows), args.chunk):
            if time.time() > deadline:
                break
            paraphrase_chunk(teacher, rows[offset : offset + args.chunk], args, out, counts)
            out.flush()
            progress("paraphrases", min(offset + args.chunk, len(rows)), len(rows))
    summary = {"model": args.model, "minutes": round((time.time() - started) / 60),
               "counts": dict(sorted(counts.items())), "samples": {}}
    # A few kept and rejected rows of each kind, to read the teacher's work by eye.
    for name in ("requests", "paraphrases"):
        written = [json.loads(line) for line in open(args.output / f"{name}.jsonl")]
        for verified in (True, False):
            pool = [{k: v for k, v in r.items() if k not in ("after", "canvas")}
                    for r in written if r["verified"] == verified]
            summary["samples"][f"{name}:{'kept' if verified else 'rejected'}"] = rng.sample(pool, min(25, len(pool)))
    (args.output / "report-teach.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["counts"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("inputs")
    prepare.add_argument("--conversations", type=Path, required=True)
    prepare.add_argument("--exclude", type=Path, required=True, help="the real test set")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--topics", type=int, default=4000)
    generate = commands.add_parser("run")
    generate.add_argument("--data", type=Path, required=True)
    generate.add_argument("--input", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--model", default="sokada4/Qwen3.8-27B-GPTQ-Int4")
    generate.add_argument("--max-num-seqs", type=int, default=48)
    generate.add_argument("--hours", type=float, default=6)
    generate.add_argument("--request-share", type=float, default=0.5,
                          help="the part of the time real-style requests may use")
    generate.add_argument("--diagrams", type=int, default=2000)
    generate.add_argument("--topics", type=int, default=2000)
    generate.add_argument("--requests", type=int, default=6)
    generate.add_argument("--paraphrase-rows", type=int, default=8000)
    generate.add_argument("--paraphrases", type=int, default=4)
    generate.add_argument("--chunk", type=int, default=256)
    generate.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    {"inputs": inputs, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
