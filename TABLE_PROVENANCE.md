# Table → Report → Script provenance

Every data-bearing table in the manuscript maps to a JSON report in this package
and the script that produced it. Verified values are the ones quoted in the
manuscript.

| # | Manuscript table (label) | Backing report (`reports/...`) | Generator (`code/...`) | Key verified values |
|---|--------------------------|--------------------------------|------------------------|---------------------|
| 1 | Anchor 78-pair (`tbl` @ caption "78-pair WYDOT anchor") | `retrieval/legacy78q_retrieval.json`, `retrieval/legacy78q_judge.json` | `retrieval/eval_78q_legacy.py`, `eval_78q_judge.py` | per-cat R@5 + judge acc, bootstrap CIs |
| 2 | Retrieval comparison (`tbl:comparison`) | `retrieval/{colnomic_3b,colnomic_7b,nemotron_colembed_8b,nemotron_colembed_4b,colqwen25_v0_2,dse_qwen2_2b,visrag_ret,bge_m3_ocr,clip_vit_b32,layoutlmv3,nougat_decode_minilm,pix2struct_decode_minilm,udop_decode_minilm,ocr_minilm,visionrag_pyramid,hpc_colpali_bq}.json` | `retrieval/bench_colx.py`, `bench_nemotron.py`, `bench_colqwen25.py`, `bench_dse.py`, `bench_visrag.py`, `bench_text_ocr.py`, `bench_ocrfree.py`, `bench_visionrag.py`, `bench_hpc_colpali.py`, `bench_weak_*.py` | ColNomic-3B 92.69; Nemotron-8B 95.28; VisRAG 53.77; BGE-M3 36.79; LayoutLMv3 0.00 (all exact) |
| — | ColPali row in `tbl:comparison` = **76.89** | `retrieval/tiling_full_page_424.json` (colpali-**v1.2**) | `retrieval/bench_tiling_424.py` | 76.89 (see Note A) |
| 3 | Retrieve–rerank (`tbl:rerank`) | `retrieval/rerank_monoqwen_424.json` | `retrieval/bench_rerank.py` | reranked 85.14 (exact); base 76.89 vs report 76.65 — see Note B |
| 4 | Category Recall@5 (`tbl:overall`) | `retrieval/colnomic_3b.json` (`by_category`) | `retrieval/bench_colx.py` | Dim 97.32 / Vis 93.10 / Log 88.33 / Hal 89.36 |
| 5 | VQA prompting grid (`tbl:vqa-techniques`) | `retrieval/vqa_results.json` | `vqa_eval` prompting harness | best 82.31; union 97.16; zero-shot 78.30; self-consistency 69.58 |
| 6 | Judge configs CAD (`tbl:compliance-cad`) | `compliance/compliance_n10_report_{7b_cot,72b_judge_only,72b_cot_thresh,agentic_72b_v2}.json` | `compliance/judge_only_n10*.py`, `agentic_n10.py` | 7B-CoT 50%; 72B+thresh 100%; agentic 100% |
| 7 | Multi-plan compliance, n=100 (`tbl:multi100`) | `compliance/multi100_path_b_report.json` (textual `d/2` threshold), `compliance/multi100_path_b_report_v2.json` (pre-resolved threshold) | `compliance/make_multi100.py`, `agentic_multi.py` | before 88/100 (36/48 + 52/52); after **100/100** (48/48 + 52/52) |
| 8 | Six CAD sets (`tbl:compliance-scale`) | `compliance/scale500_path_a_report.json`, `multi100`-set, `dense4_72b_report.json`, `stress50_path_a_report.json`, pilot | `compliance/run_500_queries.py`, `dense_eval.py`, etc. | scale500 100% (250/250, 250/250); all-sets 673/674 = 99.85% |
| 9 | OCR baseline (`tbl:ocr-baseline`) | `compliance/ocr_baseline_report.json` | `compliance/ocr_compliance_baseline.py` | OCR 76.4; culvert 100 / inlet 99 / guardrail 82 / rebar 50 / sign 51; VLM 100 |
| 10 | Rule-grounding (`tbl:rule-grounding`) | `compliance/rule_grounding_hard_report.json` | `compliance/rule_grounding_hard.py` | R@1 80.0; R@5 100.0; 1,913 cand; verdict 100/100 |
| 11 | Synthetic vs real (`tbl:real`) | `compliance/real_auditor_report.json`, `compliance/real_compliance_report.json` | `compliance/run_real_auditor.py` | real verdict 100.0 |
| 12 | LoRA ablation (`tbl:lora`) | `transfer_lora/eval_report.json` | `lora/eval_colnomic.py` | zeroshot 76.89; gentle 76.65 (−0.24); full-LM 35.38; Michigan-zs 88.17 |
| 13 | Full-page vs tile (`tbl:tiling`) | `retrieval/tiling_full_page_424.json`, `retrieval/tiling_tile_level_424.json` | `retrieval/bench_tiling_424.py` | 76.89 → 82.08 (+5.19) |
| 14 | Tile judge (`tbl:tiling-judge`) | `retrieval/tiling_judge_424.json` | `retrieval/bench_tiling_judge_424.py` | 64.23 → 68.77 (+4.53); H4 rejected |
| 15 | Per-cat Hit+Judge (`tbl` @ "Per-category Recall@5 (Hit)") | `retrieval/colnomic_3b.json` + `retrieval/eval_424_judge` output | `retrieval/eval_424_judge.py` | per-cat hit/judge |
| 16 | Bootstrap CIs (`tbl` @ "Per-baseline Recall@5 with 95% bootstrap CIs") | `retrieval/_bootstrap_cis_424.json`, `retrieval/bench_stats_424.json` | `retrieval/bootstrap_cis_424.py` | per-baseline 95% CIs |
| 17 | Rank metrics (`tbl:rankmetrics`) | `retrieval/retrieval_metrics.json` (`test_424_*`) | `retrieval/retrieval_metrics.py` | R@1 66.75 / R@5 92.45 / MRR 0.777 / nDCG 0.819 |
| 18 | Michigan joint (`tbl:michjoint`) | `retrieval/retrieval_metrics.json` (`michigan_*`) | `retrieval/retrieval_metrics.py` | joint 2,196 cand → 91.40; only 298 → 93.55 |
| 19 | Cross-family judge (appendix `app:extra`) | `retrieval/crossfamily_judge.json` | `retrieval/crossfamily_judge.py` | Qwen72 79.95 / InternVL8 85.61 / agree 92.92 / κ 0.7532 |
| 20 | Agentic confusion (`tbl` @ "confusion matrix over 8 cross-plan") | `compliance/batch_compliance_results.json` | (agentic pipeline) | 8-query GT/Pred/steps |
| 21 | FN sensitivity (`tbl:fn-sensitivity`) | `compliance/fn_sensitivity_results.json` | (agentic pipeline) | false-negative sweep |
| 22 | Retrieval latency (`tbl:latency`) | `retrieval/latency_424.json` | `retrieval/measure_latency.py` | min 85.40 / p50 97.94 / p95 102.86 / mean 97.67 / max 151.97 (H100, job 12028420) |
| 23 | Consolidated latency (`tbl` @ "Consolidated deployment latency") | `retrieval/latency_bench.json` | `retrieval/latency_bench.py` | index 7.3 min; retrieval p50; agentic 60.9 s/query |
| — | Circularity / ColPali-miss control (in-text) | `retrieval/colnomic_circularity.json` | `retrieval/circularity_eval.py` | overall 92.45; HIT 96.66 / MISS 77.89 |

## Notes

- **Note A — ColPali version.** The ColPali entry in `tbl:comparison` (and the
  76.89% quoted throughout) is **colpali-v1.2** (full-page, 424/1,898 protocol,
  `tiling_full_page_424.json`). A separate **colpali-v1.3** evaluation
  (`colpali_v13.json`, 69.58%) also ships in `reports/retrieval/` but is **not used
  by any table** — it is retained for transparency only. The paper is internally
  consistent on 76.89%.
- **Note B — rerank Stage-1 base.** `tbl:rerank` lists the ColPali Stage-1 base as
  **76.89** (the canonical full-page ColPali R@5), while `bench_rerank.py`'s own
  report measured **76.65** for the top-20 retrieval pass (Δ 0.24 pp). The reranked
  value (85.14) and the top-20 ceiling are taken directly from the report.
- **Note C — latency GPU.** Both latency reports carry a `hardware` block
  (NVIDIA H100 80GB HBM3, one GPU). The per-query distribution
  (`latency_424.json`) and the consolidated table (`latency_bench.json`) are two
  separate measurements.
