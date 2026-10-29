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

## Running on Colab

`slm/colab_run.py` starts `slm/job.sh` on a Colab GPU VM, follows its log, and collects
results into `runs/colab/` (not committed). The VM releases itself after the job.

```sh
python3 slm/colab_run.py start --gpu A100 --bundle <Kaggle canvas-270m-v7-accuracy-workflow folder>
python3 slm/colab_run.py watch --adapters q17
```

The training data are the private Kaggle datasets `rohithresearch/canvas-270m-v7-accuracy-workflow`
and `rohithresearch/canvas-270m-v6-spoken-workflow`.
