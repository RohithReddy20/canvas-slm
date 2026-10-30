"""Real diagram conversations from public chat logs (WildChat-4.8M, ShareGPT).

Keeps conversations whose user turns ask for a diagram, with each user turn and the
diagram code the assistant had produced before it, so follow-up turns show how
people actually phrase edits ("add email to User", "connect it to Order"). Location,
IP and browser headers are dropped; WildChat conversations flagged toxic are skipped.

    python mine_chats.py --output /content/runs/mine
"""

import argparse
import gzip
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

WILDCHAT = "allenai/WildChat-4.8M"
SHAREGPT = ("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json")
# A cheap substring screen before the word-boundary patterns below.
SCREEN = (
    "diagram", "mermaid", "plantuml", "uml", "flowchart", "flow chart", "erd", "draw.io",
    "drawio", "excalidraw", "tldraw", "graphviz", "digraph", "mind map", "mindmap",
    "whiteboard", "lucidchart", "visio",
)
KINDS = {
    "mermaid": r"\bmermaid\b",
    "plantuml": r"\bplant\s?uml\b|@startuml",
    "class": r"\bclass\s+diagrams?\b|\buml\b",
    # A bare "er" is German for "he", so it counts only before diagram or model.
    "er": r"\b(?:er|entity[- ]relationship)\s+(?:diagrams?|models?)\b|\berds?\b",
    "flowchart": r"\bflow\s?charts?\b|\bflow\s+diagrams?\b",
    "sequence": r"\bsequence\s+diagrams?\b",
    "state": r"\bstate\s+(?:machine\s+)?diagrams?\b",
    "activity": r"\bactivity\s+diagrams?\b|\buse[- ]case\s+diagrams?\b",
    "architecture": r"\b(?:architecture|system|network|deployment|component)\s+diagrams?\b",
    "mindmap": r"\bmind\s?maps?\b",
    "tool": r"\bdraw\.io\b|\bdrawio\b|\bdiagrams\.net\b|\bexcalidraw\b|\btldraw\b|\blucidchart\b|\bvisio\b|\bwhiteboard\b",
    "graphviz": r"\bgraphviz\b|\bdigraph\b|\bdot\s+language\b",
    "diagram": r"\bdiagrams?\b",
}
KIND_PATTERNS = {kind: re.compile(pattern, re.I) for kind, pattern in KINDS.items()}
CODE = re.compile(
    r"```\s*(?:mermaid|plantuml|puml|dot|graphviz)\b.*?```|@startuml.*?@enduml|"
    r"^\s*(?:classDiagram|erDiagram|sequenceDiagram|stateDiagram(?:-v2)?|"
    r"flowchart\s+(?:TD|TB|LR|RL|BT)|graph\s+(?:TD|TB|LR|RL|BT)|mindmap)\b.*?(?:```|\Z)|"
    r"\bdigraph\s+\w*\s*\{.*?\}",
    re.S | re.M | re.I,
)
EDIT = re.compile(
    r"\b(?:add|remove|delete|drop|rename|connect|link|join|move|change|make|put|replace|"
    r"insert|include|swap|split|merge|update|modify|turn|set|show|label|group|align|"
    r"arrange|reorder|color|colour|instead|also|without)\b",
    re.I,
)
TARGET = re.compile(
    r"\b(?:box|boxes|node|nodes|class|classes|entity|entities|table|tables|field|fields|"
    r"attribute|attributes|column|columns|method|methods|property|properties|arrow|arrows|"
    r"edge|edges|line|lines|relationship|relationships|relation|link|connection|step|steps|"
    r"shape|shapes|label|labels|title|color|colour|diagram|chart|flowchart|it|them|this)\b",
    re.I,
)
TURN_LIMIT = 6000


def kinds_of(text):
    return [kind for kind, pattern in KIND_PATTERNS.items() if pattern.search(text)]


def diagram_code(text):
    blocks = CODE.findall(text or "")
    return "\n\n".join(blocks)[:TURN_LIMIT] if blocks else None


def conversation_record(source, identifier, turns, extra):
    """The diagram conversation as kept, or None when no user turn asks for a diagram."""
    users = [t["content"] or "" for t in turns if t["role"] == "user"]
    joined = "\n".join(users).lower()
    if not any(word in joined for word in SCREEN):
        return None
    kinds = kinds_of("\n".join(users))
    if not kinds:
        return None
    kept, code, user_index = [], None, 0
    for turn in turns:
        content = turn["content"] or ""
        if turn["role"] == "user":
            # Assistants often describe a diagram in prose, so any later turn may edit it.
            followup = user_index > 0
            kept.append(
                {
                    "role": "user",
                    "index": user_index,
                    "text": content[:TURN_LIMIT],
                    "after_diagram": code is not None,
                    "edit_like": bool(followup and EDIT.search(content) and TARGET.search(content)),
                    "previous_diagram": code,
                }
            )
            user_index += 1
        else:
            code = diagram_code(content) or code
            kept.append({"role": "assistant", "text": content[:TURN_LIMIT]})
    return {"id": f"{source}:{identifier}", "source": source, "kinds": kinds, **extra, "turns": kept}


def wildchat_shard(path):
    import pyarrow.parquet as pq

    from huggingface_hub import hf_hub_download

    local = hf_hub_download(WILDCHAT, path, repo_type="dataset")
    table = pq.read_table(
        local, columns=["conversation_hash", "model", "timestamp", "conversation", "language", "toxic"]
    )
    records, seen = [], 0
    for row in table.to_pylist():
        seen += 1
        if row["toxic"]:
            continue
        turns = [{"role": t["role"], "content": t["content"]} for t in row["conversation"]]
        record = conversation_record(
            "wildchat",
            row["conversation_hash"],
            turns,
            {"model": row["model"], "language": row["language"], "timestamp": str(row["timestamp"])},
        )
        if record:
            records.append(record)
    os.remove(os.path.realpath(local))
    return path, seen, records


def sharegpt():
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(SHAREGPT[0], SHAREGPT[1], repo_type="dataset")
    rows = json.load(open(local))
    roles = {"human": "user", "user": "user", "gpt": "assistant", "chatgpt": "assistant"}
    records = []
    for row in rows:
        turns = [
            {"role": roles.get(t.get("from"), "assistant"), "content": t.get("value") or ""}
            for t in row.get("conversations", [])
        ]
        record = conversation_record("sharegpt", row.get("id"), turns, {"model": None, "language": None})
        if record:
            records.append(record)
    return len(rows), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, help="only the first N WildChat shards (smoke runs)")
    parser.add_argument("--sample", type=int, default=600)
    args = parser.parse_args()
    from huggingface_hub import list_repo_files

    args.output.mkdir(parents=True, exist_ok=True)
    shards = sorted(f for f in list_repo_files(WILDCHAT, repo_type="dataset") if f.endswith(".parquet"))
    shards = shards[: args.shards] if args.shards else shards
    counts, kinds, languages = Counter(), Counter(), Counter()
    sample_pool, create_pool = [], []
    with gzip.open(args.output / "conversations.jsonl.gz", "wt") as conversations:

        def keep(records):
            for record in records:
                conversations.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[f"{record['source']}_diagram_conversations"] += 1
                kinds.update(record["kinds"])
                languages[record.get("language") or "unknown"] += 1
                users = [t for t in record["turns"] if t["role"] == "user"]
                counts["user_turns"] += len(users)
                counts["followups"] += sum(t["index"] > 0 for t in users)
                counts["followups_after_diagram_code"] += sum(t["after_diagram"] for t in users)
                for turn in users:
                    if turn["index"] == 0:
                        create_pool.append({"id": record["id"], "kinds": record["kinds"],
                                            "language": record.get("language"), "text": turn["text"]})
                    if turn["edit_like"]:
                        counts["edit_like_followups"] += 1
                        sample_pool.append({"id": record["id"], "kinds": record["kinds"],
                                            "language": record.get("language"), **turn})

        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
            for done, (path, seen, records) in enumerate(pool.map(wildchat_shard, shards), 1):
                counts["wildchat_conversations_read"] += seen
                keep(records)
                print(f"wildchat {done}/{len(shards)} {path}: {len(records)} kept; "
                      f"{counts['wildchat_diagram_conversations']} so far", flush=True)
        if not args.shards:
            read, records = sharegpt()
            counts["sharegpt_conversations_read"] += read
            keep(records)
            print(f"sharegpt: {len(records)} kept of {read}", flush=True)
    for name, pool in (("sample-edit-turns", sample_pool), ("sample-create-requests", create_pool)):
        english = [t for t in pool if (t["language"] or "English") == "English"]
        random.Random(7).shuffle(english)
        with open(args.output / f"{name}.jsonl", "w") as sample:
            for turn in english[: args.sample]:
                sample.write(json.dumps(turn, ensure_ascii=False) + "\n")
    summary = {"counts": dict(counts), "kinds": dict(kinds.most_common()),
               "languages": dict(languages.most_common(15))}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
