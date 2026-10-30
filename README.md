# canvas-slm

A small local model that turns spoken or typed commands into validated tldraw canvas
actions. Speech recognition (Parakeet) and the browser executor live in the
[tldraw fork](https://github.com/RohithReddy20/tldraw/tree/workspace-greeting-11155aa2/templates/agent);
this repository holds the model work.

| Path | Contents |
|---|---|
| `lab/` | The original FunctionGemma-270M lab (MLX), moved with its history: data generators, simulator, training, Kaggle/Modal/Colab jobs. See `lab/README.md`. |
| `slm/` | Model-agnostic LoRA training and closed-loop evaluation (PyTorch) on Colab GPUs, with the frozen v7 `actions.py` and `sessions.py`. |
| `data-sources.md` | Survey of public data that could replace or supplement the synthetic data. |

## Capacity comparison (2026-10-30)

Each model had one LoRA pass over the 24,000 frozen v7 training rows, on a Colab A100.
Scores are exact validated actions on the v7 development split; the reserved v7 test set
is still unused.

| Model | 256-command subset | 6 subset sessions, action accuracy | All 1,700 commands | 8 sessions, action accuracy | 8 sessions, state agreement |
|---|---|---|---|---|---|
| v6 adapter (reported in `lab/`) | 78.5% (guarded) | 33.3% | | | |
| v7, 500 updates (reported) | 88.7% (guarded) | 67.2% | | | |
| FunctionGemma-270M | 80.5% | 63.5% | 61.6% | 61.7% | 5.1% |
| Qwen3-0.6B | 91.4% | 65.6% | 73.5% | 64.8% | 27.3% |
| Qwen3-1.7B | 93.0% | 96.4% | 75.0% | 95.3% | 93.4% |

Raw (unguarded) scores unless marked. Most remaining command errors are refusals
(`no_action`) in the historical v5 rows: Qwen3-0.6B gets 99.5% of those rows' edits
right but 10% of their refusals, because rules such as duplicate-name ambiguity are
nearly absent from the v7 training data.

## Real diagram conversations (2026-10-30)

`slm/mine_chats.py` (run on a Colab CPU VM by `slm/mine.sh`, 0.07 compute units) read
3,199,860 WildChat-4.8M and 94,145 ShareGPT conversations and kept 14,043 that ask for a
diagram, with 43,279 user turns. Of those, 1,098 follow-ups come right after the
assistant produced diagram code (402 PlantUML, 299 Mermaid, the rest graph/DOT/sequence);
802 are English or unlabelled. The broader keyword filter for edit-like follow-ups
(6,987) is mostly noise ("mermaid" hair colours, questions that mention a diagram).

The follow-ups after diagram code show how people actually edit: mostly compound or
high-level requests ("add more objects and arrows to manage multiple companies",
"make it top to bottom", "include steps for a lien or a damaged title", "the compliance
check is done by the sending bank, not the recipient bank"), plus fix requests after
syntax errors. Few are single atomic edits like the current one-action contract expects.
The mined files stay in `runs/colab/mine/` and are not committed, since they hold users'
chat text.

### Teacher labels and the first real test set

`slm/teach.py` (Qwen3.8-27B-FP8 with thinking, on a Colab G4, about 12 compute units)
labelled the 1,097 follow-ups: 411 other, 277 new-diagram requests, 195 edits (135
high-level, 41 compound, 19 atomic), 117 error reports, 79 edits the tools cannot express
(mostly layout direction and themes), 18 unparsed. 47 scripts applied in the simulator;
reading them by hand kept 32 as test cases (about 68% teacher precision). The rest were
translations, pasted code or open-ended rewrites.

The 32 real cases are 21 compound and 11 atomic edits, in English (22 including
unlabelled ShareGPT) and five other languages. Two thirds need several actions, which
the one-action contract cannot express. The set is `runs/colab/real-test-v1.jsonl`,
kept out of git like the mined data.

## Edit scripts

`slm/contract.py` defines the multi-action format: targets are references (a name, the
selection, an object created earlier in the script, the connection between two objects)
that code resolves, so missing and ambiguous targets are decided by rules instead of the
model. `slm/convert.py` rewrites v7 labels as scripts and keeps a row only when its
script resolves back to the original label (90% of a 2,000-row sample; missing and
ambiguous refusals are dropped by design).

## Running on Colab

`slm/colab_run.py` starts `slm/job.sh` on a Colab GPU VM, follows its log, and collects
results into `runs/colab/` (not committed). The VM releases itself after the job.

```sh
python3 slm/colab_run.py start --gpu A100 --bundle /path/to/canvas-270m-v7-accuracy-workflow
python3 slm/colab_run.py watch --adapters q17
```

The training data are the private Kaggle datasets `rohithresearch/canvas-270m-v7-accuracy-workflow`
and `rohithresearch/canvas-270m-v6-spoken-workflow`.
