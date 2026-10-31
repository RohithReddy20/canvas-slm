"""Whether a teacher model runs on this machine's GPUs, how fast, and whether it writes sense.

Times three workloads shaped like teacher data jobs on v7 training rows: writing a script
for a command on its canvas (with and without thinking), and paraphrasing a command.

    python teacher_probe.py --model sokada4/Qwen3.8-27B-GPTQ-Int4 --data /content/data/v7 \\
        --output /content/runs/teacher-probe
"""

import argparse
import json
import random
import time
from pathlib import Path

from contract import SYSTEM, messages
from convert import session_for
from teach import tool_lines

SCRIPT = f"{SYSTEM}\nTools (\"?\" marks optional arguments):\n{tool_lines()}\nReply with the script only."
PARAPHRASE = (
    "Write five different ways a person might say this command to a diagramming app, by voice "
    "or typed, keeping its exact meaning: the same names, numbers and colors, and the same way "
    "of pointing at objects. One per line, no numbering."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    args = parser.parse_args()
    import torch
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(args.data / "examples.jsonl")]
    rows = random.Random(11).sample([r for r in rows if r["split"] == "train"], args.prompts)
    started = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), dtype="float16",
              max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"image": 0, "video": 0})
    report = {"model": args.model, "gpus": torch.cuda.get_device_name(0),
              "count": torch.cuda.device_count(), "load_seconds": round(time.time() - started)}
    print(json.dumps(report), flush=True)

    def chats(system, rows):
        if system == PARAPHRASE:
            return [[{"role": "system", "content": system}, {"role": "user", "content": r["command"]}]
                    for r in rows]
        return [[{"role": "system", "content": system},
                 messages(r["command"], session_for(r))[1]] for r in rows]

    workloads = {
        "script": (chats(SCRIPT, rows), False, 300),
        "paraphrase": (chats(PARAPHRASE, rows), False, 300),
        "script_thinking": (chats(SCRIPT, rows[:32]), True, 2048),
    }
    for name, (batch, thinking, limit) in workloads.items():
        sampling = SamplingParams(temperature=0.6 if thinking else 0.7, top_p=0.95, top_k=20,
                                  max_tokens=limit)
        started = time.time()
        outputs = llm.chat(batch, sampling, chat_template_kwargs={"enable_thinking": thinking},
                           use_tqdm=False)
        seconds = time.time() - started
        prompt = sum(len(o.prompt_token_ids) for o in outputs)
        generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        report[name] = {
            "prompts": len(batch), "seconds": round(seconds, 1),
            "prompt_tokens": prompt, "generated_tokens": generated,
            "generated_per_second": round(generated / seconds), "seconds_per_prompt": round(seconds / len(batch), 2),
            "truncated": sum(o.outputs[0].finish_reason == "length" for o in outputs),
            "samples": [{"command": r["command"], "reply": o.outputs[0].text[-700:]}
                        for r, o in zip(rows[:4], outputs[:4])],
        }
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "samples"}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "--")
    (args.output / f"report-{slug}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
