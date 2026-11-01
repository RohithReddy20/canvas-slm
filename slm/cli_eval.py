"""Answer a distil_export.py test set with a model behind a local CLI: `agy` (Gemini) or
`claude` (Claude Code, for Claude models).

Writes predictions in distil labs' format ({"prompt", "prediction"}), so distil_score.py
scores them like the platform's teacher evaluations. Finished rows are kept on rerun.
Claude Code gets the guide as its whole system prompt and no tools, so each call carries
only the task; agy adds its own agent prompt (about 9k tokens) to every call, which
--batch N spreads over N rows (one reply lists N scripts). When the quota runs out, a
batch waits and tries again, so a long run resumes on its own.

    python cli_eval.py --dataset ../runs/distil/eval/dataset --cli agy --model gemini-4-argon-medium \\
        --output ../runs/distil/eval/predictions/agy-gemini-4-argon-medium.jsonl
    python cli_eval.py --dataset ../runs/distil/eval/dataset --cli claude --model claude-opus-5-5 \\
        --output ../runs/distil/eval/predictions/claude-opus-5-5.jsonl
"""

import argparse
import json
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def ask_agy(model, guide, user, timeout, cwd, ending="Reply with the JSON script only."):
    result = subprocess.run(
        ["agy", "--model", model, "--sandbox", "--disable-slash-commands", "--output-format", "json",
         "--print-timeout", f"{timeout}s", f"-p={guide}\n\n{user}\n\n{ending}"],
        capture_output=True, text=True, timeout=timeout + 60, cwd=cwd,
    )
    reply = json.loads(result.stdout[result.stdout.find("{"):])
    if reply.get("status") != "SUCCESS":
        raise RuntimeError(reply.get("error") or result.stdout[-300:])
    return reply["response"], reply.get("usage", {})


def ask_claude(model, guide, user, timeout, cwd, ending=""):
    result = subprocess.run(
        ["claude", "-p", "--model", model, "--system-prompt", guide, "--tools", "", "--strict-mcp-config",
         "--setting-sources", "", "--no-session-persistence", "--output-format", "json",
         f"{user}\n\n{ending}" if ending else user],
        capture_output=True, text=True, timeout=timeout, cwd=cwd,
    )
    reply = json.loads(result.stdout)
    if reply.get("is_error") or reply.get("subtype") != "success":
        raise RuntimeError(str(reply.get("result") or result.stderr)[-300:])
    return reply["result"], reply.get("usage", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cli", choices=("agy", "claude"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1, help="rows per call")
    parser.add_argument("--quota-wait", type=int, default=600, help="seconds to wait when the quota is out")
    parser.add_argument("--quota-tries", type=int, default=36)
    args = parser.parse_args()

    ask = {"agy": ask_agy, "claude": ask_claude}[args.cli]
    guide = json.load(open(args.dataset / "job_description.json"))["task_description"]
    users = [json.loads(line)["messages"][0]["content"] for line in open(args.dataset / "test.jsonl")]
    done = set()
    if args.output.exists():
        done = {json.loads(line)["prompt"] for line in open(args.output)}
    todo = [user for user in users if user not in done]
    lock, failures = threading.Lock(), []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def call(prompt, ending):
        for _ in range(args.quota_tries):
            try:
                with tempfile.TemporaryDirectory() as empty:  # nothing for the agent to read or edit
                    return ask(args.model, guide, prompt, args.timeout, empty, ending)
            except RuntimeError as error:
                if not re.search(r"RESOURCE_EXHAUSTED|quota|credits|429|usage limit|rate limit|limit reached", str(error), re.I):
                    raise
                time.sleep(args.quota_wait)
        raise RuntimeError("the quota stayed exhausted")

    def work(batch):
        try:
            if len(batch) == 1:
                text, usage = call(batch[0], "Reply with the JSON script only.")
                answers = [text]
            else:
                prompt = "\n\n".join(f"### Input {i}\n{user}" for i, user in enumerate(batch, 1))
                text, usage = call(prompt, f"Answer each of the {len(batch)} inputs on its own. Reply with JSON only: "
                                           f'{{"answers": [the script for input 1, ..., the script for input {len(batch)}]}}.')
                reply = json.loads(text[text.find("{"): text.rfind("}") + 1])
                answers = [json.dumps(answer) for answer in reply["answers"]]
                if len(answers) != len(batch):
                    raise ValueError(f"{len(answers)} answers for {len(batch)} inputs")
        except (RuntimeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
            with lock:
                failures.append(str(error)[:200])
            return
        with lock, open(args.output, "a") as out:
            for user, answer in zip(batch, answers):
                out.write(json.dumps({"prompt": user, "prediction": answer, "usage": usage}, ensure_ascii=False) + "\n")

    batches = [todo[i: i + args.batch] for i in range(0, len(todo), args.batch)]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, batches))
    print(json.dumps({"model": args.model, "asked": len(todo), "failed": len(failures),
                      "examples": failures[:3]}))


if __name__ == "__main__":
    main()
