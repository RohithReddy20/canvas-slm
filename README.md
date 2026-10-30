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

### First script model (2026-10-30)

Qwen3-1.7B, one LoRA pass over 31,080 script rows (22,080 converted v7 rows and 9,000
compounds), on a Colab A100. Scored by the canvas each script leaves (`evaluate_scripts.py`):

| Set | One-action Qwen3-1.7B | Script Qwen3-1.7B |
|---|---|---|
| All 1,700 dev commands | 75.0% (exact action) | 92.6% (same canvas) |
| 256-command subset | 93.0% | 90.6% |
| 8 dev sessions, state agreement | 93.4% | 23.8% |
| 400 held-out compounds | cannot express | 95.5% |
| 32 real follow-ups (diagram content) | cannot express most | 9.4% (3 atomic, 0 compound) |

Refusals now come from the resolver, which removes the historical-refusal ceiling on the dev
commands. The session regression is not yet explained: replaying converted gold labels
already caps that harness at 68% agreement, since session turns lack reference metadata.
On real follow-ups the model invents tools and arguments (`add_note`, `update_diagram`,
`"required": true`; 12 of 32 invalid or unparsed) and names objects loosely ("Ecommerce" for
"ECommerce", an arrow by its label; 13 of 32 missing targets). Constrained decoding against
the script grammar and fuzzier reference matching are the next, training-free fixes.

### Constrained decoding and fuzzy names (retrained model, same data)

| Variant | Dev commands | Subset | Sessions, agreement | Compounds | Real (32) |
|---|---|---|---|---|---|
| free decoding, exact names | 92.5% | 92.2% | 19.5% | 96.5% | 2 |
| free, fuzzy | 92.5% | 92.2% | 19.5% | 96.5% | 3 |
| grammar-constrained, exact | 91.3% | 91.4% | 19.5% | 96.2% | 3 |
| grammar-constrained, fuzzy | 91.3% | 91.4% | 19.5% | 96.2% | 4 |

The grammar cut invalid real scripts from 12 to 2, but valid is not correct: correct real
edits only rose from 2 to 4, all atomic, none compound. Fuzzy names changed nothing on the
dev sets. The gap is the training distribution (real canvases, phrasing and multi-step
intent), not decoding.

### Why sessions regressed

The script model's sessions failed early: 84.5% accuracy while its canvas still matched,
first error at turn 7 on average, no perfect session. Every dev session contains "Move it a
distance of 100 towards the right." with no single object selected, which v7 labels as an
ambiguous target; those turns sit at turn 6.75 on average. Scripts had no singular
selection reference, so the model wrote `shape_ids: [{"selected": true}]` and moved every
selected object. The conversion had three more gaps:

- all 890 missing/ambiguous refusals in the training data were dropped, because a refusal
  label does not say which edit was meant;
- training canvases never carried links (rows hold no connection data), while 46% of
  session turns show some;
- exact names skipped arrows, but v7 canvases have free-standing labelled arrows that
  commands name by their label.

Fixes: a `{"it": true}` reference (the one selected object, otherwise a refusal decided by
the resolver); refusals rebuilt from v7's wording families (move it, delete a name, remove
a field from the selected shape) and kept only when the resolver refuses them for the
labelled reason (889 of 890); session-turn rows rendered on the replayed session, which
adds links and real history to 877 training prompts; arrows matched by exact label when
nothing else has that name. Conversion keeps 23,993 of 24,000 training rows (was about
22,000), and replaying gold scripts through the dev sessions now expresses 254 of 256
turns with no divergence.

Retrained with these fixes on Kaggle (2×T4, FP16, 32,993 rows, 24.8M tokens, 704 steps in
3.5 hours at 1,950 tokens/s, no overflowed steps; free):

| Variant | Dev commands | Subset | Sessions, agreement | Compounds | Real (32) |
|---|---|---|---|---|---|
| free decoding, before the fixes | 92.5% | 92.2% | 19.5% | 96.5% | 2 |
| free decoding | 93.2% | 92.2% | 36.7% | 96.2% | 3 |
| grammar-constrained | 93.6% | 93.8% | 36.7% | 96.2% | 4 |

Per-turn accuracy while the session canvas still matched rose from 84.5% to 93.1%, the
first error moved from turn 7.1 to 9.9, and one session is now perfect. Six of the seven
remaining first errors are the same held-out dev wording, "Remove all objects from the
selection." (clear the selection): training's nearest phrase is "Remove the selection from
the canvas." (delete), so the model deletes the selection and the session diverges from
there. The seventh is a fill style v7 calls unsupported on that shape. Both are wording and
rule coverage, not harness errors; real follow-ups stay at 3–4 of 32 with no compound right.

### Training speed

`train.py --autotune` measured on the A100 (Qwen3-1.7B LoRA, rank 64):

| Setting | Tokens/s | Peak memory |
|---|---|---|
| SDPA, padded, checkpointing | 7,812 | 13.5 GB |
| FlashAttention-2 varlen, packed, 24K tokens per micro-batch | 8,765 | 11.4 GB |
| Same, 48K tokens per micro-batch | 9,010 | 18.0 GB |
| + Liger 0.6.4 (RMSNorm, RoPE, SwiGLU), 24K tokens | 10,080 | 10.7 GB |
| + Liger, half the layers checkpointed, 12K tokens (chosen) | 12,069 | 22.8 GB |
| + Liger, no checkpointing, 12K tokens | 13,618 | 38.3 GB (too close to 39.5) |

The full run trained at 12,100 tokens/s, about 35% of the A100's peak and 55% faster than
the baseline. Settings need a benchmark peak within 80% of GPU memory: the no-checkpointing
setting ran out of memory on the full data. Colab's default PyTorch (2.11, CUDA 13) has no
prebuilt FlashAttention or Hub-kernel build, so `scripts.sh` trains in a PyTorch 2.8
environment; Liger 0.7+ needs transformers 5 and is pinned to 0.6.4.

## Running on Kaggle

The Mac's Kaggle account runs jobs in private kernels on two free T4s (about 30 GPU-hours
a week, two sessions at a time; A100 and L4 requests fall back to T4). `slm/kaggle_run.py`
embeds `slm/` in the kernel script, mounts the v7 dataset and a small private inputs
dataset, and leaves models in the kernel output on Kaggle, where later kernels can mount
them. Only logs and reports come back.

```sh
python3 slm/kaggle_run.py start --name scripts3-t4 --with-inputs \
  --env "SETTINGS=--attention sdpa --liger 1 --checkpointing 1 --max-tokens 12288"
kaggle kernels logs -f rohithresearch/canvas-slm-scripts3-t4
python3 slm/kaggle_run.py collect --name scripts3-t4
```

T4s have no BF16 or FlashAttention-2, so `train.py` trains in FP16 with loss scaling and,
under torchrun, sums LoRA gradients across both GPUs. Autotune on one T4 chose Liger with
full checkpointing at 12K tokens per micro-batch (1,250 tokens/s per GPU); dropping
checkpointing ran out of memory. `SETTINGS` skips the 20 minutes of tuning.

## Running on Colab

`slm/colab_run.py` starts `slm/job.sh` on a Colab GPU VM, follows its log, and collects
results into `runs/colab/` (not committed). The VM releases itself after the job.

```sh
python3 slm/colab_run.py start --gpu A100 --bundle /path/to/canvas-270m-v7-accuracy-workflow
python3 slm/colab_run.py watch --adapters q17
```

The training data are the private Kaggle datasets `rohithresearch/canvas-270m-v7-accuracy-workflow`
and `rohithresearch/canvas-270m-v6-spoken-workflow`.
