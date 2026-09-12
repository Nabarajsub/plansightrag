# PlanSightRAG

**A Visual-First Multimodal RAG for Automating Question Answering and Compliance
Checking for Civil Standard Plans**

Nabaraj Subedi, Shuvo Dip Datta, Ahmed Abdelaty, Shivanand Venkanna Sheshappanavar
— University of Wyoming

Reproducibility package: code, evaluation reports, and benchmark metadata.

Every number in every table of the manuscript is reproduced by a script in `code/`
that writes a JSON report in `reports/`. The mapping from each manuscript table to
its backing report and generating script is in **`TABLE_PROVENANCE.md`**.

## Demo

A two-minute silent walkthrough of the pipeline, built from the actual plan sheets
and the actual sharpened-MaxSim heatmaps used in the study — not mock-ups.

<div align="center">
  <video src="https://github.com/user-attachments/assets/e15d74a8-6033-41c6-b6d8-8e0023acd069" width="880" controls muted></video>
</div>

<p align="center">
2:02 &nbsp;·&nbsp; 1920×1080 &nbsp;·&nbsp; H.264, no audio &nbsp;·&nbsp; 6.9 MB<br>
<sub>Player not loading? <a href="https://github.com/Nabarajsub/plansightrag/raw/main/docs/PlanSightRAG_demo.mp4">download the video</a> or open the <a href="docs/demo_preview.gif">animated preview</a> &nbsp;·&nbsp; <a href="docs/DEMO_PROVENANCE.md">where every on-screen number comes from</a></sub>
</p>

| # | Scene | Length | What it shows |
|---|---|---|---|
| 1 | Title | 9 s | Paper, authors, corpus scale |
| 2 | The problem | 13 s | One sheet as an engineer reads it vs. as OCR hands it over, and the retrieval numbers that follow |
| 3 | Phase 1 · Ingestion | 12 s | 66 PDFs → 1,898 pages at 200 DPI across five DOTs |
| 4 | Phase 2 · Visual indexing | 12 s | ColNomic-3B patch grid → multi-vector embedding; 7.3 min offline |
| 5 | Phase 3 · Retrieval & grounding | 16 s | MaxSim scoring, ranked candidates, then the heatmap reveal over WYDOT 203-2A |
| 6 | Phase 4a · Visual QA | 12 s | A grounded answer beside the evidence gallery it came from |
| 7 | Phase 4b · Agentic compliance | 20 s | Planner → Retriever → Auditor → Synthesizer, step by step, to a verdict |
| 8 | Results | 16 s | Recall@5 across seven retrievers; compliance, transfer and rule-grounding |
| 9 | Reproducibility | 12 s | table → report JSON → generator script |

Every figure on screen is traced to a committed report in
**`docs/DEMO_PROVENANCE.md`**, which also flags the three elements that are
illustrative rather than measured.

## Quick start

```bash
git clone git@github.com:Nabarajsub/plansightrag.git && cd plansightrag
pip install -r requirements.txt

# Plan images are not redistributed. Rebuild them from the public DOT PDFs,
# then point PLANS_ROOT at the rasterized output (per-document DPI in data/render_map.json).
export PLANS_ROOT=/path/to/rasterized/pages
export PSR_ROOT=$(pwd)          # optional; defaults to the repo root
```

## Paths and page identifiers

No absolute machine paths appear anywhere in this package. Scripts resolve two
environment variables, both with sensible defaults:

| Variable | Meaning | Default |
|---|---|---|
| `PSR_ROOT` | this repository | inferred from the script location |
| `PLANS_ROOT` | rasterized plan pages | `$PSR_ROOT/data/pages` |

Benchmark records identify pages by a stable **`page_id`** of the form
`<agency>/<plan_id>/p<page>` (e.g. `WYDOT/608-1B/p7`), not by file path.
`data/page_index.json` maps all 1,854 page_ids to their provenance — agency, plan
ID, sheet title and the original source filename — so the corpus can be rebuilt
from the public DOT documents.

## What is in this package

```
plansightrag/
├── README.md                 # this file
├── LICENSE                   # MIT (code) + CC BY 4.0 (annotations)
├── CITATION.cff
├── requirements.txt          # pinned environment
├── TABLE_PROVENANCE.md       # manuscript table  ->  report JSON  ->  generating script
├── reports/                  # source-of-truth result JSONs (back every table)
│   ├── retrieval/            # H1 retrieval comparison, rank metrics, tiling, rerank, CIs, latency, circularity
│   ├── compliance/           # H2 CAD compliance, OCR baseline, rule-grounding, real-plan, agentic
│   └── transfer_lora/        # H3 cross-agency transfer + LoRA ablation
├── code/                     # the generators (run-as-is on the cached index)
│   ├── retrieval/            # bench_*.py, retrieval_metrics.py, circularity, cross-family judge, latency
│   ├── compliance/           # CAD generators (archetypes), auditor, OCR baseline, rule-grounding
│   ├── lora/                 # split, train, eval for the LoRA ablation
│   └── benchmark_gen/        # refine-loop QnA drafters + Qwen-7B verifier + Michigan index
└── data/                     # releasable text artifacts (NOT the image corpus — see below)
    ├── prompts.json          # all 42 prompts, with source file:line and placeholders
    ├── seeds.json            # every random seed and what it controls
    ├── models.json           # model ids pinned to Hugging Face commit SHAs + decoding
    ├── page_index.json       # page_id -> provenance for all 1,854 pages
    ├── splits/               # page-disjoint train/dev/test query splits (jsonl)
    ├── transfer/             # 93 verified Michigan transfer queries
    ├── vqa/                  # 424 Qwen-7B zero-shot answers used by the cross-family judge
    └── real_plan/            # real WYDOT-project audit items (ground-truth encoded)
```

## What is NOT included (and where to get it)

- **The 1,898-page five-DOT plan image corpus** is not redistributed here. The
  source documents are public: WYDOT Standard Plans are available from the Wyoming
  DOT website; the other four agencies (Caltrans, AZDOT, CDOT, FDOT) publish their
  standard plans on their respective DOT sites. Page-rasterization settings are in
  the methodology (200 DPI page-level; 400 DPI for the tiling study).
- **The 2 GB CAD-generated compliance image sets** (`compliance/scale500`, etc.) are
  regenerable from `code/compliance/make_*.py` + `archetypes*.py`. **Verified:** a
  clean re-run of `make_500.py` under the pinned environment reproduces all 500
  drawings **byte-identically** (500/500 filenames, bytes, pixels and manifest
  records). Note that `matplotlib` must match the pin in `requirements.txt` —
  text metrics differ across minor versions, so a different matplotlib gives
  visually equivalent but not byte-identical drawings.
- **Model weights** are pulled from Hugging Face by `model_id` (see each script header).

## Environment

- Python 3.10+, PyTorch 2.x, CUDA 12.x.
- Retrieval: `colpali_engine` (ColQwen2_5 / ColQwen2_5_Processor), `transformers`,
  `Pillow`, `numpy`, `tqdm`.
- Compliance/judge: `transformers` with `bitsandbytes` (4-bit NF4) for
  Qwen2.5-VL-72B and InternVL2.5-8B.
- Retriever backbone: `nomic-ai/colnomic-embed-multimodal-3b` (headline results).
- All judging/answering uses **local** models only (Qwen2.5-VL, InternVL) — no
  hosted-API calls are required to reproduce any number.

## Reproducing the headline numbers

The retrieval scripts reuse a cached page-embedding tensor
(`baselines_v2/cache_colnomic_pages.pt`, 1,898 pages) so they run in minutes rather
than re-encoding the corpus. With the corpus rasterized and the cache built:

```bash
# H1 — rank-sensitive retrieval metrics (Table: rankmetrics) + Michigan joint (Table: michjoint)
python code/retrieval/retrieval_metrics.py        # -> reports/retrieval/retrieval_metrics.json

# H1 — full baseline comparison (Table: comparison)
python code/retrieval/bench_colx.py               # ColNomic / ColQwen / Nemotron ...
python code/retrieval/bench_rerank.py             # MonoQwen2-VL rerank (Table: rerank)

# Judge circularity control (cross-family) — kappa=0.75
python code/retrieval/crossfamily_judge.py        # -> reports/retrieval/crossfamily_judge.json

# H2 — autonomous rule-grounding (Table: rule-grounding)
python code/compliance/rule_grounding_hard.py     # -> reports/compliance/rule_grounding_hard_report.json
```

Each script prints its JSON to stdout and writes it under `reports/`; compare against
the committed report of the same name.


## Reproducibility manifest

Everything needed for exact re-execution is committed:

| Artifact | File | Contents |
|---|---|---|
| Page identifiers | `data/page_index.json` | 1,854 pages: agency, plan ID, sheet title, source filename |
| Source documents | `data/sources.json` | 66 PDFs with SHA-256, page count and per-document attribution |
| Questions & answers | `data/splits/*.jsonl` | 3,211 train / 421 dev / 424 test, page-disjoint |
| Refine-loop trace | same files | `attempts[]`, `n_attempts`, `final_rank`, `final_hit` per question |
| Transfer set | `data/transfer/verified_michigan.jsonl` | 93 held-out Michigan DOT pairs |
| Prompts | `data/prompts.json` | 42 prompts, parsed from source, with `{placeholders}` |
| Seeds | `data/seeds.json` | 11 declarations across 11 scripts |
| Model revisions | `data/models.json` | 20 models pinned to commit SHAs |
| Decoding settings | `data/models.json` | greedy throughout (`do_sample=False`), per-task `max_new_tokens` |
| Environment | `requirements.txt` | pinned versions |
| Evaluation code | `code/` | 88 scripts; table→report→script map in `TABLE_PROVENANCE.md` |

Regenerate the derived artifacts with:

```bash
python code/make_release_artifacts.py     # prompts, seeds, models, requirements
```

## Question fields

| Field | Meaning |
|---|---|
| `query` / `answer` | **as evaluated** — every published metric is computed on these |
| `malformed` | present on 25 of 4,056 records (0.62%) whose drafter output was a markdown header rather than a question |
| `query_regenerated` / `answer_regenerated` | redrafted with Qwen2.5-VL-72B and re-verified, supplied so the released set is complete |

`query` is never overwritten, so the shipped splits reproduce the reported numbers
exactly. To work with the complete set instead:

```python
q = r.get("query_regenerated") or r["query"]
```

## Baseline checkpoint versions

Two ColPali checkpoints were evaluated under the identical 424-query / 1,898-page
protocol. They are different models, not conflicting measurements of one model:

| Checkpoint | Recall@5 | Report |
|---|---|---|
| `vidore/colpali-v1.2` | **76.89%** | `reports/retrieval/tiling_full_page_424.json` |
| `vidore/colpali-v1.3` | **69.58%** | `reports/retrieval/colpali_v13.json` |

**v1.2 is the companion backbone** used for the tiling, re-ranking, binary-quantization
and bootstrap-CI analyses, and it is the ColPali figure quoted in the manuscript.
v1.3 is included for completeness. The adopted retriever for all headline results is
`nomic-ai/colnomic-embed-multimodal-3b` at 92.69%.

Unrelated coincidence worth flagging, because the numbers collide: **69.58** also
appears in the manuscript as the percentage-point gap between ColNomic-3B (92.69)
and the strongest hybrid baseline (23.11). That is a different quantity from
ColPali-v1.3's 69.58% Recall@5.

## Data sources and licensing

State DOT Standard Plans are public records published by the issuing agencies and
are **not redistributed here**. `data/sources.json` records every source document
with its **SHA-256**, page count, and how many benchmark pages came from it, so
you can confirm you hold the same edition — DOT plans are revised, and a later
edition will not reproduce the index. Verify with `sha256sum <file>`.

Documents flagged `used_in_reported_index: false` are present in the working
corpus but back no reported result (four superseded Caltrans editions and four
Road Design Manual chapters used only in Section 5.9).

Download from:

| Agency | Corpus | Pages |
|---|---|---|
| Wyoming DOT | Standard Plans | 237 |
| Caltrans | 2025 Standard Plans | 638 |
| Arizona DOT | 2025 Standard Drawings | 181 |
| Colorado DOT | 2025 Bridge Detail Worksheets | 62 |
| Florida DOT | 2026 Design Standards | 780 |
| Michigan DOT | Standard Plans (held-out transfer set) | 298 |

**WYDOT Standard Plans are copyrighted by the Wyoming Department of Transportation**
and are distributed through a click-through acceptance page. Obtain them directly
from WYDOT. This is one reason no plan imagery is redistributed in this repository.

Rebuild the page corpus with the shipped renderer, which reproduces the original
scale and file naming per document (`data/render_map.json`):

```bash
python code/render_pages.py --pdf-dir /path/to/downloaded/pdfs --out data/pages
export PLANS_ROOT=$PWD/data/pages
```

Source PDFs are named as in `data/sources.json`, which also carries their URLs and
sha256. Rendered pages line up with `data/page_index.json` one-to-one. See `LICENSE` for the split between code (MIT) and benchmark
annotations (CC BY 4.0).

Funded by the Wyoming Department of Transportation, grant RS03225.

## Additional analyses (`code/analysis/`)

Each script is self-contained and reads only artefacts shipped in this repository. Set
`PSR_ROOT` to the repository root; scripts that need the full page corpus also
honour `CLUSTER_ROOT`.

| Script | What it produces | Cost |
|---|---|---|
| `capability_ladder.py` | Per-stage decomposition of compliance accuracy: retrieval → rule ID → rule interpretation → value extraction → arithmetic → verdict | CPU, seconds |
| `numeric_match.py` | Judge-free arithmetic scoring of Dimensional Accuracy, and its agreement with the LLM judge | CPU, seconds |
| `threshold_sensitivity.py` | Flip points for the H1–H4 acceptance thresholds, with effect sizes | CPU, seconds |
| `colnomic_tiling_v2.py` | Six tile-score aggregations on the adopted backbone, two grids | 1 GPU, ~35 min/grid |
| `compliance_retrieval_colnomic.py` | Compliance-set retrieval re-measured on ColNomic-3B | 1 GPU, ~2 min |
| `stats_all_methods.py` | Wilson, bootstrap and page-clustered intervals for all 57 retrieval runs | CPU, ~2 min |
| `split_similarity.py` | Duplicate / near-duplicate rates and cross-split similarity, page and question level | CPU, ~10 min |
| `baseline_settings.py` | Per-baseline implementation settings across eight reproducibility fields | CPU, seconds |
| `baseline_settings_table.py` | Per-baseline implementation settings, with each generating script's documentation | CPU, seconds |
| `compliance_calibration.py` | ECE, MCE, Brier and abstention for the compliance judge | 1 GPU, ~1.5 h |
| `qa_grounding_cis.py` | Wilson, bootstrap and page-clustered intervals for the QA and grounding families | CPU, seconds |
| `bench_tiling_judge_colnomic_424.py` | H4 judge-accuracy endpoint on the adopted ColNomic-3B backbone: full-page vs. tile-level retrieval, 397 dense queries | 2 GPU, ~7 h |
| `table9_rank1.py` | Compliance verdict accuracy conditioned on rank-1 retrieval: the auditor is shown the rank-1 retrieved page instead of the supplied sheet, frozen question set | 1 GPU, ~25 min |
| `table9_frozen.py` | Judge configurations on a frozen question set, with and without supplied design facts | 1 GPU, ~2 h |
| `faithfulness_probe.py` | Causal occlusion probe of the MaxSim attribution maps: the top-5% patches masked vs. the same pattern translated to the least-salient position | 1 GPU, ~5 min |

