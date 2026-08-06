# Provenance for `PlanSightRAG_demo.mp4`

The demo video states numbers on screen. This file maps each of them back to a
committed report, in the same spirit as `TABLE_PROVENANCE.md`. Paths are relative
to the repository root.

| Scene | Claim on screen | Source |
|---|---|---|
| 1 Title | 1,898 indexed plan sheets · five state DOTs · 4,056 benchmark pairs | manuscript abstract and §Methodology |
| 2 Problem | LayoutLMv3 Recall@5 **0.00** | `reports/retrieval/layoutlmv3.json` |
| 2 Problem | BGE-M3 + OCR Recall@5 **36.79** | `reports/retrieval/bge_m3_ocr.json` |
| 2 Problem | VisRAG Recall@5 **53.77** | `reports/retrieval/visrag_ret.json` |
| 2 Problem | PlanSightRAG Recall@5 **92.69** | `reports/retrieval/colnomic_3b.json` |
| 3 Ingestion | 66 source PDFs | `data/sources.json` |
| 3 Ingestion | 1,898 pages · 200 DPI · five DOTs | §`sec:meth-ingest` |
| 3 Ingestion | WYDOT 237 / Caltrans 638 / AZDOT 181 / CDOT 62 / FDOT 780; Michigan 298 held out | §`sec:meth-ingest`; README corpus table |
| 4 Indexing | ColNomic-3B multi-vector late interaction, dynamic-resolution patch grid | §`sec:meth-colpali` |
| 4 Indexing | 7.3 min one-time index · 4.34 pages/s | `reports/retrieval/latency_bench.json` |
| 5 Retrieval | `Score(Q,P) = Σ_i max_j (q_i · p_j)` | Eqs. 1–3, §`sec:meth-retrieval-grounding` |
| 5 Retrieval | retrieval p50 **0.10 s** over 1,898 pages | `reports/retrieval/latency_424.json` (p50 101.26 ms) |
| 5 Retrieval | Recall@5 **92.69%** on the 424-pair test split | `reports/retrieval/colnomic_3b.json` |
| 5 Grounding | top 5% of patches · γ = 3.0 · α = 0.4 overlay | §`sec:meth-retrieval-grounding`, "Sharpened MaxSim Heatmap" |
| 6 VQA | best prompting technique **82.31%** judge accuracy | `reports/retrieval/vqa_results.json` |
| 6 VQA | union over techniques **97.16%** | `reports/retrieval/vqa_results.json` |
| 6 VQA | cross-family judge agreement **92.92%**, κ = 0.7532 | `reports/retrieval/crossfamily_judge.json` |
| 6 VQA | answer text (MDC depth, 2 ft [600], scarify 6 in [150], 90%) | WYDOT 203-2A Grading Notes H and I, read off the sheet on screen |
| 7 Agentic | Planner–Retriever–Auditor–Synthesizer | §`sec:meth-compliance` |
| 7 Agentic | ~**8.6** sequential VLM steps per verdict | §`sec:meth-deploy` |
| 7 Agentic | **100%** verdict accuracy, 500 single-doc + 100 multi-plan CAD | `reports/compliance/scale500_path_a_report.json`, `reports/compliance/multi100_path_b_report_v2.json` |
| 7 Agentic | **60.9 s/query** | `reports/retrieval/latency_bench.json` |
| 7 Agentic | audit categories unit / dimension / label / logic | §`sec:meth-compliance` |
| 8 Results | ColPali-v1.2 **76.89** | `reports/retrieval/tiling_full_page_424.json` (see Note A in `TABLE_PROVENANCE.md`) |
| 8 Results | ColPali + MonoQwen rerank **85.14** | `reports/retrieval/rerank_monoqwen_424.json` |
| 8 Results | Nemotron-ColEmbed-8B **95.28** | `reports/retrieval/nemotron_colembed_8b.json` |
| 8 Results | **99.85%** = 673/674 across six CAD compliance sets | `TABLE_PROVENANCE.md` row 8 |
| 8 Results | **93.55%** zero-shot Michigan transfer | `reports/retrieval/retrieval_metrics.json` → `michigan_only_index.recall@5` |
| 8 Results | rule grounding R@5 **100%**, 1,913 candidates | `reports/compliance/rule_grounding_hard_report.json` |
| 8 Results | **91.47%** Recall@5 on the full 4,056-pair benchmark | manuscript `tbl:overall` |
| 8 Results | full-LM LoRA tuning collapses to **35.38%** | `reports/transfer_lora/eval_report.json` |
| 9 Reproducibility | 72 scripts · 1,854 page IDs · 3,211/421/424 splits · 42 prompts · 20 pinned models · 11 seeds · 66 PDFs with SHA-256 | README reproducibility manifest |

## Two numbers that deliberately differ

- **92.69% vs 91.47%.** 92.69 is ColNomic-3B on the **424-pair page-disjoint test
  split**, the split every baseline comparison uses. 91.47 is the same retriever
  over the **full 4,056-pair benchmark**. The video labels each one on screen.
- **Nemotron-8B (95.28) scores above the adopted ColNomic-3B (92.69).** The bar is
  shown at full length and tagged `non-commercial`, matching the paper's claim of
  adopting the strongest *openly-licensed* retriever.

## Illustrative, not measured

Three elements are visual scaffolding built around real sheets, not evaluation
output. They are called out here so nothing in the video reads as a result that
isn't one:

- The four similarity scores in scene 5 (0.912 / 0.781 / 0.744 / 0.664) are
  plausible MaxSim magnitudes chosen to show ranking behaviour. They are not from
  a report.
- The four planner step descriptions in scene 7 are written to match the WYDOT
  215-1 and 203-2A sheets on screen. The measured quantity is the step *count*
  (8.6 average).
- The OCR token spill in scene 2 is a hand-built illustration of fragmentation.
  Its *consequence* (36.79% / 0.00%) is measured.

Every plan sheet shown, and the sharpened MaxSim heatmap in scene 5, are the real
artifacts from the study.
