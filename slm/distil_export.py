"""Upload folders for distil labs (distillabs.ai) teacher evaluation, one per teacher.

The test set is the 256-command v7 dev subset (selection-receipt.json) and, with --real,
the hand-checked real follow-ups. Each row is the user message the student sees with the
gold script as the answer; the teacher's system prompt is teach_data.GUIDE, the prompt our
own teacher gets. distil labs only runs the teachers: distil_score.py scores their
predictions by the canvas they produce. The v7 test split is never read.

    python distil_export.py --bundle ../runs/colab/v7-bundle --output ../runs/distil/eval
    distil dataset create --data ../runs/distil/eval/dataset
    distil teacher-evaluation create-from-dataset --config ../runs/distil/eval/<teacher>.yaml <id>
"""

import argparse
import json
from pathlib import Path

from contract import messages, script_text, session_from
from convert import session_for, to_script
from teach_data import GUIDE

TEACHERS = {
    "qwen38-27b": "qwen.qwen3.8-27b",  # the teacher teach_data.py runs today: the baseline
    "gpt-oss-120b": "openai.gpt-oss-120b-thinking",
    "glm53": "zai.glm-5.3-low-thinking",
    "deepseek-v4-pro": "deepseek.v4-pro-thinking",
    "kimi-k3": "moonshotai.kimi-k3-low-thinking",
    "opus55": "anthropic.claude-opus-5-5",  # evaluation only: a ceiling, cannot generate data
}
JUDGE = "openai.gpt-oss-120b"  # one judge for every run; its scores are not ours anyway

JUDGE_INSTRUCTIONS = (
    "The answer is correct when it is a JSON edit script whose actions, applied in order to "
    "the canvas in the input, do exactly what the command asks and nothing more, with targets "
    "given as references the way the command names them; or, when no tool can carry the "
    "command out, the refusal {\"actions\": [], \"reason\": \"unsupported_request\"}."
)


def config(teacher):
    return "\n".join([
        "base:",
        "  task: question-answering",
        "  student_model_name: Qwen3-1.7B",
        f"  teacher_model_name: {teacher}",
        "  should_expand_dataset: false",
        "synthgen:",
        "  output_is_json: true",
        "  teacher_temperature: 0.6",
        "evaluation:",
        f"  llm_as_a_judge_model_name: {JUDGE}",
        "",
    ])


def dev_rows(bundle):
    wanted = json.load(open(bundle / "selection-receipt.json"))["example_ids"]
    rows = {}
    for line in open(bundle / "examples.jsonl"):
        row = json.loads(line)
        if row["id"] in wanted:
            assert row["split"] == "valid", f"{row['id']} is not a validation row"
            rows[row["id"]] = row
    assert len(rows) == len(wanted), "missing dev rows"
    for identifier in wanted:
        row = rows[identifier]
        script = to_script(row)
        # The answer is only a reference for distil's own metrics; scoring replays v7's label.
        answer = script_text(script) if script else json.dumps({"actions": []})
        yield identifier, messages(row["command"], session_for(row))[1]["content"], answer


def real_rows(path):
    for line in open(path):
        row = json.loads(line)
        user = messages(row["command"], session_from(row["canvas"]))[1]["content"]
        yield row["id"], user, script_text(row["script"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--real", type=Path, help="also upload the real follow-ups")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.output / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    rows = [("dev", *row) for row in dev_rows(args.bundle)]
    if args.real:
        rows += [("real", *row) for row in real_rows(args.real)]
    users = [user for _, _, user, _ in rows]
    assert len(set(users)) == len(users), "two test rows share a prompt"

    with open(dataset / "test.jsonl", "w") as test, open(args.output / "index.jsonl", "w") as index:
        for kind, identifier, user, answer in rows:
            test.write(json.dumps({"messages": [{"role": "user", "content": user},
                                                {"role": "assistant", "content": answer}]},
                                  ensure_ascii=False) + "\n")
            index.write(json.dumps({"kind": kind, "id": identifier, "user": user}, ensure_ascii=False) + "\n")
    json.dump({"task_description": GUIDE, "llm_as_a_judge_instructions": JUDGE_INSTRUCTIONS},
              open(dataset / "job_description.json", "w"), indent=2)
    (dataset / "config.yaml").write_text(config(TEACHERS["glm53"]))
    for name, teacher in TEACHERS.items():
        (args.output / f"{name}.yaml").write_text(config(teacher))
    kinds = {kind: sum(r[0] == kind for r in rows) for kind in ("dev", "real")}
    print(json.dumps({"rows": kinds, "system_prompt_chars": len(GUIDE),
                      "mean_user_chars": sum(map(len, users)) // len(users)}))


if __name__ == "__main__":
    main()
