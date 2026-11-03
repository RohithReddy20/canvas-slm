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

### Teacher-written data (2026-10-31)

`slm/teach_data.py` has Qwen3.8-27B (FP8, Colab G4) write English training data, keeping
a row only when an independent greedy teacher reply leaves the same canvas:

- real-style requests: the teacher draws a canvas from a mined diagram or diagram request,
  then writes follow-ups for it (single edits, compound edits, one no tool can do) with
  their scripts; 1,508 of 3,497 kept;
- paraphrases of v7 training commands, checked against the v7 label on the source row's
  canvas; 10,405 of 15,239 kept.

A 4-bit 27B loads on Kaggle's two T4s (`teacher_probe.py`) but takes about 2.9 s per script
there, so the job ran on a G4 instead: 1.1 hours of teaching, 12 compute units. The
teacher's tool list first lacked the allowed `operation` values; the checks then agreed
with v7 on 35% of unchanged commands, and on 79% once the values were listed.

Retrained on Kaggle with the teacher rows (44,769 rows, 34.0M tokens, 954 steps in 4.9
hours; free). A tenth of the request diagrams are held out as a new dev set:

| Free decoding | Before | With teacher data |
|---|---|---|
| Dev commands (1,700) | 93.2% | 95.3% |
| Subset (256) | 92.2% | 92.2% |
| Sessions, agreement | 36.7% | 54.3% |
| Sessions, first error turn | 9.9 | 13.5 |
| Compounds | 96.2% | 95.0% |
| Held-out teacher requests (137) | – | 81.0% (atomic 26/33, compound 27/45, refusal 58/59) |
| Real (32) | 3 | 4 |
| Real, English and unlabelled (22) | 1 | 3 |

Sessions improved most; "Remove all objects from the selection." is still the first error
in four of eight. Real follow-ups barely moved, and they now fail differently: 14 of 32
are refused as unsupported (before: 9 invalid, 6 unparsed, 9 missing targets). Refusals
verify far more often than compound edits (74% against 27%), so they make up 37% of the
teacher's requests, and the model now refuses vague real requests ("can you integrate
SQL MP database") that the hand-checked labels treat as edits.

### Kimi K3 synthetic data (2026-11-01)

`slm/distil_synth.py` has Kimi K3 (distil labs, $38.88) write 5,064 rows of messy spoken
edits on system-design canvases, seeded from v7 rows and the observation rig; 4,672 pass
the replay check and 3,735 also agree with a second model (Gemini, then Claude). The
retrain scripts5-t4 (48,147 rows, 6.1 hours, free Kaggle), constrained decoding with the
fuzzy resolver:

| | scripts4 | scripts5 |
|---|---|---|
| Dev commands (1,700) | 93.4% | 94.6% |
| Subset (256) | 91.4% | 96.1% |
| Sessions, agreement | 54.3% | 95.7% |
| Sessions, first error turn | 13.5 | 25.5 |
| Compounds | 95.0% | 96.5% |
| Held-out teacher requests (137) | 80.3% | 83.9% |
| Held-out synthetic rows (357) | – | 91.6% |
| Real, English, v1 key (22) | 3 | 3 |
| Real, English, reviewed v2 key (19) | 5 | 4 |

Sessions are fixed: the clear-selection wording no longer deletes, and four of eight dev
sessions replay perfectly. Real follow-ups did not move, and 11 of 19 are still refused.

### The real answer key, reviewed (2026-11-01)

The v1 real cases were labelled by Qwen3.8-27B, the weakest teacher in the comparison, so
better models were marked wrong for disagreeing with it. Reviewing the 22 English cases
against their source chats: 2 cannot be answered from the canvas (dropped), 1 had lost the
box its command corrects (repaired), 16 had several reasonable readings and now carry them
as accepted alternatives (`real_match.matches`; `"arrow_direction": "any"` for UML
implements/uses arrows). One v1 key was itself wrong: it deleted every arrow between two
participants instead of the one named. Under the reviewed key (19 comparable cases):

| Model | v1 key (22) | v2 key (19) |
|---|---|---|
| Opus 5.5 | 10 | 14 |
| Gemini 4 Argon | 7 | 12 |
| DeepSeek V4 Pro | 7 | 10 |
| Kimi K3 | 5 | 8 |
| Qwen3.8-27B | 4 | 7 |
| scripts4 student | 3 | 5 |

Where the data falls short: 16 of the 19 real cases add a domain-named object to a UML,
flowchart or mind-map diagram of 10 to 25 nodes and expect it wired in; the Kimi rows are
mostly styling, text, layout and deletion on 6-node system designs (120 create-and-connect
rows of 3,735). The next generation round (`distil_synth.py export --profile real`) seeds
from canvases transcribed from real chats and describes how real users ask; `real_label.py`
grows the English test set from the mined follow-ups with a strong model proposing, a
second model answering from the canvas alone, and a human reviewing the disagreements.

### Real-style synthetic data and the v3 test set (2026-11-03)

`real_label.py` turned 729 English follow-ups from the mined chats into the v3 test set:
127 cases scored by what they expect (`real_match.matches`): 28 concrete edits, 37
additions ("add a cache" on a 15-node diagram: something new, wired to the object named),
9 restyles, 3 layouts and 50 refusals (explanations, code, "more detail everywhere").
Only 17 of the 729 follow-ups were edits that replay exactly, which is why the kinds exist.

The generation round `distil_synth.py export --profile real` seeded Kimi K3 from
canvases transcribed from real chats and the ways real users ask (5,434 rows, $48.77);
Opus 5.5 checked every row and, where it disagreed, wrote the script itself (the user
accepted the terms risk; such rows stay in separate `*.fixed.jsonl` files with an
`author` field). Merged with the earlier rounds: 9,800 rows, 6,972 by Kimi and 2,828 by
Opus, 2,624 of them create-and-connect (120 before). The retrain scripts6-t4 (53,592 rows,
7 hours on Kaggle) against scripts5 scored on the same sets:

| | scripts5 | scripts6 |
|---|---|---|
| Dev commands (1,700) | 94.6% | 94.6% |
| Sessions, agreement | 95.7% | 54.3% |
| Compounds | 96.5% | 96.3% |
| Held-out teacher requests (137) | 83.9% | 88.3% |
| Held-out synthetic rows (977, new set) | 58.0% | 67.1% |
| Real v3 English, edits (28) | 7 | 13 |
| Real v3 English, additions (37) | 1 | 6 |
| Real v3 English, restyle (9) / layout (3) | 0 / 0 | 0 / 0 |
| Real v3 English, refusals (50) | 48 | 49 |

Real edits nearly doubled and additions moved off the floor without more false edits on
the refusal cases. Additions still mostly fail by refusing (10) or by creating the object
without an arrow to the right anchor (12 applied but unwired). Restyle and layout have no
training data at all: every case is refused.

Sessions are a coin flip, not a regression: four of the eight dev sessions contain v7's
"Remove all objects from the selection" (clear the selection), while v7 also labels
"Remove the selection from the canvas" as delete, and each retrain lands on one side
(scripts4 54.3%, scripts5 95.7%, scripts6 54.3%). Per-turn accuracy when the state
matches is 95.9%. A caveat on v3: its canvases overlap the conversations the teacher
requests were seeded from, so the key's labels are unseen but some canvases are not.

Two training lessons. The merged rows have replies up to 3,600 characters (580 before),
and the loss materialised float32 logits for every row at the longest reply, about 3 GB
for one micro-batch; `train.py` now runs the reply positions through the head 1,024 at a
time under checkpointing. And distil labs trains the same student for $7.77 in under two
hours (`distil_train.py` exports the rows; the adapter comes back as PEFT LoRA on our
system prompt and `scripts.sh` scores it with `ADAPTER=`), so the loop no longer needs
7 Kaggle hours per try; `kaggle_run.py start --reuse <name>` re-scores an earlier
student in 30 minutes.

### A quarter of the v7 rows, trained on distil labs (2026-11-03)

Of the 53,592 training rows, about 43,000 are v7's own single commands, compounds and
paraphrases. `distil_train.py --thin accuracy-train-single=0.25 --drop-conversations
real-test-v3.jsonl` keeps a quarter of the single-command rows (and drops the 61 teacher
requests whose chats the v3 key came from): 23,413 rows, trained on distil labs in about
90 minutes for about $3 (job 1e89dfd3), scored on a Colab T4. Against scripts6, all
languages, constrained decoding:

| | scripts6 (53,592 rows) | distil7 (23,413 rows) |
|---|---|---|
| Dev commands (1,700) | 94.6% | 93.2% |
| Sessions, per turn when the state matches | 95.9% | 93.1% |
| Compounds | 96.2% | 95.2% |
| Held-out teacher requests (137) | 122 | 119 |
| Held-out synthetic rows (977) | 67.1% | 61.3% |
| Real v3 edits (38) / additions (37) / refusals (50) | 17 / 6 / 49 | 15 / 4 / 48 |

A point or so on the v7 sets and six on the synthetic set: the v7 rows do carry weight,
but at a quarter of the data the student is within reach of the full one for a fifth of
the time. Two operational lessons cost a day: Colab's uploader and Kaggle's datasets both
unpack `.tar` files, so `ADAPTER=.../model.tar` was never found and both platforms
silently trained from scratch instead (the "hung" Kaggle kernel was a 7-hour retrain that
used up the weekly 30 GPU hours); `scripts.sh` now also takes the unpacked
`model-adapter/` folder.

### Precise instructions: the author dictates, the model writes (2026-11-03)

The product use is literal: "between Client and Load Balancer add a Rate Limiter" means
one new node, the right icon, wired into that path, nothing else. The v3 "additions" are
mostly "add more detail", which is not that, and the training data had almost no
rewiring: "between" appears in 242 of 53,592 rows and no row inserts a node on a path.
`precise_synth.py` generates such instructions with their exact scripts (insert between,
add linked, add beside, replace keeping arrows, redirect an arrow, delete or relabel an
arrow, connect, rename, delete, style, and pairs of these) on three kinds of diagrams:
generated system designs, 90 designs Opus drew for 30 known systems (URL shortener, ride
hailing, ...; `precise_rewrite.py designs`), and the earlier synthetic diagrams. Every row
is kept only when its script applies in the simulator and changes the diagram, so the
scripts are right by construction; `precise_rewrite.py rewrite` then has Opus reword the
instructions as an author would type them ("label the Redis Cache -> Cron Scheduler
arrow \"reads\""), kept when the template's names are still all named. Results follow
(train8: 10,000 template rows; train9: plus 4,191 rewrites).

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
