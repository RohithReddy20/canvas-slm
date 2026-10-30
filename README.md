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

## Running on Colab

`slm/colab_run.py` starts `slm/job.sh` on a Colab GPU VM, follows its log, and collects
results into `runs/colab/` (not committed). The VM releases itself after the job.

```sh
python3 slm/colab_run.py start --gpu A100 --bundle /path/to/canvas-270m-v7-accuracy-workflow
python3 slm/colab_run.py watch --adapters q17
```

The training data are the private Kaggle datasets `rohithresearch/canvas-270m-v7-accuracy-workflow`
and `rohithresearch/canvas-270m-v6-spoken-workflow`.
