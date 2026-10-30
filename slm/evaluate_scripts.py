"""Score a script model with vLLM by the canvas it produces.

- commands: every v7 validation command; correct when the script leaves the same canvas as
  the label (refusals leave it unchanged), comparable to the one-action exact accuracy;
- sessions: v7 validation sessions closed-loop, the model acting on its own canvas;
- compounds: held-out compound requests on validation canvases;
- real: hand-checked real follow-ups, compared by diagram content (semantic_state).

Each variant combines free or grammar-constrained decoding (script_schema) with the exact
or the fuzzy name resolver; generations are cached, so variants share them where the
prompts agree.

    python evaluate_scripts.py --model Qwen/Qwen3-1.7B --adapter adapter/ --data /content/data/v7 \\
        --compounds dev-compounds.jsonl --real real-test-v1.jsonl --output scripts-report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from contract import (
    messages,
    parse_script,
    run_script,
    script_schema,
    semantic_state,
    session_from,
)
from convert import session_for
from sessions import CanvasSession

FAILURES = (ValueError, IndexError, KeyError, TypeError, AttributeError)


class Model:
    def __init__(self, path, adapter=None):
        from vllm import LLM, SamplingParams

        options = {"enable_lora": True, "max_lora_rank": 64} if adapter else {}
        self.llm = LLM(model=str(path), max_model_len=8192, gpu_memory_utilization=0.85, **options)
        self.tokenizer = self.llm.get_tokenizer()
        self.lora = None
        if adapter:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("scripts", 1, str(adapter))
        self.sampling = {False: SamplingParams(temperature=0, max_tokens=768)}
        try:
            from vllm.sampling_params import StructuredOutputsParams

            grammar = {"structured_outputs": StructuredOutputsParams(json=script_schema())}
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams

            grammar = {"guided_decoding": GuidedDecodingParams(json=script_schema())}
        self.sampling[True] = SamplingParams(temperature=0, max_tokens=768, **grammar)
        self.cache = {}

    def scripts(self, chats, constrained=False):
        prompts = [
            self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
            for chat in chats
        ]
        missing = list(dict.fromkeys(p for p in prompts if (constrained, p) not in self.cache))
        if missing:
            outputs = self.llm.generate(
                missing, self.sampling[constrained], use_tqdm=False, lora_request=self.lora
            )
            for prompt, output in zip(missing, outputs, strict=True):
                self.cache[constrained, prompt] = output.outputs[0].text
        return [self.cache[constrained, p] for p in prompts]


def apply(text, session, command, prefix, fuzzy=False):
    """Run a model reply on the session; returns its outcome label."""
    try:
        script = parse_script(text)
    except FAILURES:
        return "unparsed"
    try:
        refused = run_script(script, session, command, prefix, fuzzy)
    except FAILURES:
        return "invalid"
    return refused or "applied"


def commands(model, rows, constrained=False, fuzzy=False):
    texts = model.scripts([messages(r["command"], session_for(r)) for r in rows], constrained)
    outcomes, right, strata = Counter(), Counter(), Counter()
    for row, text in zip(rows, texts, strict=True):
        predicted, gold = session_for(row), session_for(row)
        outcome = apply(text, predicted, row["command"], "new", fuzzy)
        if row["expected"]["name"] != "no_action":
            gold.execute(row["command"], row["expected"], "new")
        stratum = row.get("stratum") or row["expected"]["name"]
        strata[stratum] += 1
        correct = predicted.snapshot() == gold.snapshot()
        right[stratum] += correct
        outcomes[outcome] += 1
        outcomes["correct"] += correct
    return {
        "total": len(rows),
        "state_accuracy": outcomes["correct"] / len(rows),
        "outcomes": dict(outcomes),
        "per_stratum": {k: f"{right[k]}/{strata[k]}" for k in sorted(strata)},
    }


def sessions(model, cases, constrained=False, fuzzy=False):
    live = [(c, CanvasSession(c["initial_canvas"]), CanvasSession(c["initial_canvas"])) for c in cases]
    turns = matched = matched_right = agree = 0
    first_errors = {}
    for turn in range(max(len(c["turns"]) for c in cases)):
        active = [entry for entry in live if turn < len(entry[0]["turns"])]
        for case, actual, oracle in active:
            actual.external(case["turns"][turn]["before"])
            oracle.external(case["turns"][turn]["before"])
        texts = model.scripts(
            [messages(c["turns"][turn]["command"], actual) for c, actual, _ in active], constrained
        )
        for (case, actual, oracle), text in zip(active, texts, strict=True):
            step = case["turns"][turn]
            was_matched = actual.snapshot() == oracle.snapshot()
            prefix = f"{case['id']}:created-{turn}"
            apply(text, actual, step["command"], prefix, fuzzy)
            oracle.execute(step["command"], step["expected"], prefix)
            same = actual.snapshot() == oracle.snapshot()
            turns += 1
            agree += same
            matched += was_matched
            matched_right += was_matched and same
            if not same:
                first_errors.setdefault(case["id"], turn + 1)
    return {
        "sessions": len(cases),
        "turns": turns,
        "state_agreement_rate": agree / turns,
        "accuracy_when_state_matched": matched_right / matched if matched else None,
        "mean_first_error_turn": sum(first_errors.values()) / len(first_errors) if first_errors else None,
        "perfect_sessions": len(cases) - len(first_errors),
    }


def compounds(model, rows, constrained=False, fuzzy=False):
    texts = model.scripts([messages(r["command"], session_for(r)) for r in rows], constrained)
    outcomes, exact, semantic = Counter(), 0, 0
    for row, text in zip(rows, texts, strict=True):
        predicted, gold = session_for(row), session_for(row)
        outcomes[apply(text, predicted, row["command"], "new", fuzzy)] += 1
        run_script(row["script"], gold, row["command"], "new")
        exact += predicted.snapshot() == gold.snapshot()
        semantic += semantic_state(predicted) == semantic_state(gold)
    return {"total": len(rows), "state_accuracy": exact / len(rows),
            "content_accuracy": semantic / len(rows), "outcomes": dict(outcomes)}


def real(model, rows, constrained=False, fuzzy=False):
    texts = model.scripts(
        [messages(r["command"], session_from(r["canvas"])) for r in rows], constrained
    )
    results = []
    for row, text in zip(rows, texts, strict=True):
        session = session_from(row["canvas"])
        outcome = apply(text, session, row["command"], "new", fuzzy)
        results.append({"id": row["id"], "language": row["language"], "edit_kind": row["edit_kind"],
                        "outcome": outcome, "correct": semantic_state(session) == row["after"],
                        "reply": text[:600]})
    by = lambda key: {  # noqa: E731
        value: f"{sum(r['correct'] for r in results if r[key] == value)}/{sum(r[key] == value for r in results)}"
        for value in sorted({r[key] for r in results})
    }
    return {"total": len(rows), "content_accuracy": sum(r["correct"] for r in results) / len(rows),
            "by_kind": by("edit_kind"), "by_language": by("language"),
            "outcomes": dict(Counter(r["outcome"] for r in results)), "cases": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="merged weights, or the base with --adapter")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--compounds", type=Path, required=True)
    parser.add_argument("--real", type=Path)
    parser.add_argument("--variants", default="free-exact", help="comma-separated decoding-resolver pairs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = Model(args.model, args.adapter)
    rows = [json.loads(line) for line in open(args.data / "examples.jsonl")]
    cases = [json.loads(line) for line in open(args.data / "sessions.jsonl")]
    receipt = json.loads((args.data / "selection-receipt.json").read_text())
    valid = [r for r in rows if r["split"] == "valid"]
    receipt_rows = [r for r in valid if r["id"] in set(receipt["example_ids"])]
    compound_rows = [json.loads(line) for line in open(args.compounds)]
    real_rows = [json.loads(line) for line in open(args.real)] if args.real and args.real.exists() else []
    report = {"model": args.model, "adapter": str(args.adapter) if args.adapter else None}
    for variant in args.variants.split(","):
        decoding, resolver = variant.split("-")
        options = {"constrained": decoding == "constrained", "fuzzy": resolver == "fuzzy"}
        report[variant] = {
            "commands_all": commands(model, valid, **options),
            "commands_receipt": commands(model, receipt_rows, **options),
            "sessions_all": sessions(model, [c for c in cases if c["split"] == "valid"], **options),
            "compounds": compounds(model, compound_rows, **options),
        }
        if real_rows:
            report[variant]["real"] = real(model, real_rows, **options)
        brief = {
            name: {k: v for k, v in part.items() if k not in ("cases", "per_stratum")}
            for name, part in report[variant].items()
        }
        print(variant, json.dumps(brief), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
