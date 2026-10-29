# Data sources for the canvas action model

Collected 2026-10-30. Nothing here has been downloaded into training yet.

**Bottom line:** no public dataset of real tldraw command logs exists. The closest
genuine human language is real chat logs where people asked an AI to draw and
then edit diagrams (WildChat, ShareGPT). Real diagram *states* are plentiful
(tldraw/Excalidraw/draw.io files, UML models, database schemas). Everything
labelled "model-written" below is synthetic, like the current data.

## 1. Genuine human requests (highest value)

| Source | What it is | Size | License | Use |
|---|---|---|---|---|
| [allenai/WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) | Real user conversations with ChatGPT, many multi-turn | 4.8M conversations | ODC-BY, not gated | Filter for diagram requests (mermaid, UML, ER, flowchart, class diagram, draw.io) and their follow-up edits ("add email to User", "connect X to Y"). Real phrasing for a test set and teacher-model seeds. Counts pending: HF search index was still loading. |
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | Earlier subset of the above | 1M | ODC-BY | Same |
| [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | Real shared ChatGPT conversations | ~90k | Apache-2.0 (as listed) | Same filtering |
| [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | Real user conversations, 25 models | 1M | Gated, custom terms | Same, after accepting terms |
| [DocEdit (Adobe, AAAI 2023)](https://research.adobe.com/publication/docedit-language-guided-document-editing) | Human edit requests on PDF/design pages, with direct and indirect object references and executable commands | ~28k (public split ~12.7k, no boxes) | Check before use | Closest human-written "edit this visual object" language |
| [CoDraw](https://github.com/facebookresearch/CoDraw) | Human dialogues instructing someone to place clip art on a canvas | ~10k dialogues, ~138k messages | Repo says NOASSERTION; a paper reports CC BY-NC 4.0 | Spatial language: "put it on the left, a bit smaller" |
| [PPTC](https://github.com/gydpku/PPTC) ([paper](https://arxiv.org/abs/2311.01767)) | Multi-turn PowerPoint editing sessions with API-call solutions | 279 sessions | MIT | Multi-turn edit sessions; reports the same error-accumulation problem (GPT-4: 75% per turn, 6% per session) |
| [Edit Me (LREC 2018)](https://aclanthology.org/L18-1683/) | Elicited human image-edit requests mapped to commands | small | Check | Phrasing of edit requests |
| [kixlab/CANVAS](https://huggingface.co/datasets/kixlab/CANVAS) | UI-design replication and modification via tool calls, with human tool paths | ~4k files | CC-BY-4.0 | Design-tool modification tasks |

## 2. Real diagram states (realistic canvases, names, layouts)

| Source | What it is | Size | License | Use |
|---|---|---|---|---|
| [steveruizoktldraw/tldraw-datasets](https://huggingface.co/datasets/steveruizoktldraw/tldraw-datasets) | tldraw room history: initial snapshot plus every edit diff, with a render per step | 1 room (271 MB) on HF | MIT | The only real tldraw edit trace found; consecutive diffs are real edit operations. Too small to train on; useful to test the simulator against real records |
| [GitHub `.tldr` files](https://github.com/search?q=extension%3Atldr&type=code) | Committed tldraw documents | ~4.2k matches (mixed with tldr-pages man pages; filter on JSON content) | Per-repo | Real tldraw layouts, labels and arrows as starting canvases |
| [Infatoshi/excalidraw-workspace](https://huggingface.co/datasets/Infatoshi/excalidraw-workspace) | One person's technical Excalidraw drawings | ~120 | Unspecified | Real whiteboard diagrams; Obsidian vaults on GitHub have many more `.excalidraw` files |
| [dimsavva/drawio-sft](https://huggingface.co/datasets/dimsavva/drawio-sft), [masamasa4/drawio-xml](https://huggingface.co/datasets/masamasa4/drawio-xml), [miracle10/DwT_drawio](https://huggingface.co/datasets/miracle10/DwT_drawio) | draw.io diagrams, some paired with instructions | 10k–100k (drawio-sft) | MIT / unspecified / other | Diagram structures with boxes, labels, edges |
| [ModelSet](https://github.com/modelset) ([paper](https://link.springer.com/doi/10.1007/s10270-021-00929-3)) | Labelled real UML and Ecore models from GitHub | 5,120 UML + 5,466 Ecore | LGPL-3.0 | Real class names, attributes, methods and associations for schema boxes, replacing the synthetic name lists |
| [iisys-hof/uml-class-diagrams](https://huggingface.co/datasets/iisys-hof/uml-class-diagrams) | UML class diagrams | 1k–10k | Unspecified | Same |
| [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context), [xlangai/spider](https://huggingface.co/datasets/xlangai/spider), [BIRD](https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106) | Real database schemas (tables, columns, foreign keys) | 78k / 1k+ | CC-BY-4.0 / CC-BY-SA-4.0 | Realistic entities and fields; foreign keys become connections |

## 3. When to act vs refuse (the main error source in the 0.6B run)

| Source | What it is | License | Use |
|---|---|---|---|
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | When to call a tool, ask a follow-up, or decline | CC-BY-4.0 | Refusal and ambiguity behaviour |
| [Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Includes multi-turn cases with missing functions or parameters | Apache-2.0 | Evaluation of refusals on missing targets |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Verified function-calling data (APIGen) | CC-BY-4.0, auto-gated | General tool-call robustness |

## 4. Model-written text-to-diagram data (synthetic, like ours)

- [13point5/tldraw-vf-env](https://huggingface.co/datasets/13point5/tldraw-vf-env): 7k "draw a flowchart / state machine / architecture / sequence diagram" prompts for a tldraw RL environment (MIT). Whole-diagram creation, not edits.
- [DiagramGenBenchmark (CVPR 2025)](https://arxiv.org/abs/2411.11916): diagram generation, coding and *editing* tasks; model-written instructions.
- [mikronai/VectorEdits](https://huggingface.co/datasets/mikronai/VectorEdits): 271k SVG edit instructions written by vision-language models; [SVGEdit-30k](https://huggingface.co/datasets/steve329/SVGEdit-30k) is similar.
- Mermaid/PlantUML: [ibm-research/MermaidSeqBench](https://huggingface.co/datasets/ibm-research/MermaidSeqBench) (Apache-2.0), [Celiadraw/text-to-mermaid](https://huggingface.co/datasets/Celiadraw/text-to-mermaid), [rakitha/mermaid-flowchart-transformer](https://huggingface.co/datasets/rakitha/mermaid-flowchart-transformer), [vinzur/Prompt-to-PlantUML](https://huggingface.co/datasets/vinzur/Prompt-to-PlantUML), [sequelbox/UML-Generator-Dataset-DeepSeek-V3.2](https://huggingface.co/datasets/sequelbox/UML-Generator-Dataset-DeepSeek-V3.2).

## Not found

- No public tldraw usage or command logs. The [tldraw agent starter kit](https://tldraw.dev/starter-kits/agent) and [tldraw AI docs](https://tldraw.dev/docs/ai) describe the agent action system but publish no data. `github.com/tldraw/tldraw-datasets`, linked from the HF card, is not public.

## Suggested order for tomorrow

1. On a Colab CPU VM (not this Mac), filter WildChat-4.8M and ShareGPT for diagram conversations and pull out the user's edit turns. Hand-check a few hundred to build a **real-phrasing test set**.
2. Swap the generator's synthetic names for real ones: ModelSet/UML classes, SQL schemas, and labels from `.tldr`/Excalidraw/draw.io files.
3. Use DocEdit, CoDraw and PPTC phrasing as seeds for the teacher-model paraphrases.
