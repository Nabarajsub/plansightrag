# PlanSightRAG

**A Visual-First Multimodal RAG for Automating Question Answering and Compliance
Checking for Civil Standard Plans**

Nabaraj Subedi, Shuvo Dip Datta, Ahmed Abdelaty, Shivanand Venkanna Sheshappanavar
— University of Wyoming

Reproducibility package: code, evaluation reports, and benchmark metadata.

Every number in every table of the manuscript is reproduced by a script in `code/`
that writes a JSON report in `reports/`. The mapping from each manuscript table to
its backing report and generating script is in **`TABLE_PROVENANCE.md`**.

## Quick start

```bash
git clone https://github.com/<ORG>/plansightrag && cd plansightrag
pip install -r requirements.txt

# Plan images are not redistributed. Rebuild them from the public DOT PDFs,
# then point PLANS_ROOT at the rasterized output (200 DPI page-level).
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

Everything reviewer-requested for exact re-execution is committed:

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
| Evaluation code | `code/` | 72 scripts; table→report→script map in `TABLE_PROVENANCE.md` |

Regenerate the derived artifacts with:

```bash
python code/make_release_artifacts.py     # prompts, seeds, models, requirements
```

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

Rasterize at 200 DPI page-level (400 DPI for the tiling study) and set
`PLANS_ROOT`. See `LICENSE` for the split between code (MIT) and benchmark
annotations (CC BY 4.0).

Funded by the Wyoming Department of Transportation, grant RS03225.
