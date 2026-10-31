"""Answer a distil_export.py test set with a model behind a local CLI: `agy` (Gemini) or
`claude` (Claude Code, for Claude models).

Writes predictions in distil labs' format ({"prompt", "prediction"}), so distil_score.py
scores them like the platform's teacher evaluations. Finished rows are kept on rerun.
Claude Code gets the guide as its whole system prompt and no tools, so each call carries
only the task; agy adds its own agent prompt (about 9k tokens) to every call.

    python cli_eval.py --dataset ../runs/distil/eval/dataset --cli agy --model gemini-4-argon-medium \\
        --output ../runs/distil/eval/predictions/agy-gemini-4-argon-medium.jsonl
    python cli_eval.py --dataset ../runs/distil/eval/dataset --cli claude --model claude-opus-5-5 \\
        --output ../runs/distil/eval/predictions/claude-opus-5-5.jsonl
"""

import argparse
import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def ask_agy(model, guide, user, timeout, cwd):
    result = subprocess.run(
        ["agy", "--model", model, "--sandbox", "--disable-slash-commands", "--output-format", "json",
         "--print-timeout", f"{timeout}s", f"-p={guide}\n\n{user}\n\nReply with the JSON script only."],
        capture_output=True, text=True, timeout=timeout + 60, cwd=cwd,
    )
    reply = json.loads(result.stdout[result.stdout.find("{"):])
    if reply.get("status") != "SUCCESS":
        raise RuntimeError(reply.get("error") or result.stdout[-300:])
    return reply["response"], reply.get("usage", {})


def ask_claude(model, guide, user, timeout, cwd):
    result = subprocess.run(
        ["claude", "-p", "--model", model, "--system-prompt", guide, "--tools", "", "--strict-mcp-config",
         "--setting-sources", "", "--no-session-persistence", "--output-format", "json", user],
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

    def work(user):
        try:
            with tempfile.TemporaryDirectory() as empty:  # nothing for the agent to read or edit
                text, usage = ask(args.model, guide, user, args.timeout, empty)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            with lock:
                failures.append(str(error)[:200])
            return
        with lock, open(args.output, "a") as out:
            out.write(json.dumps({"prompt": user, "prediction": text, "usage": usage}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))
    print(json.dumps({"model": args.model, "asked": len(todo), "failed": len(failures),
                      "examples": failures[:3]}))


if __name__ == "__main__":
    main()
