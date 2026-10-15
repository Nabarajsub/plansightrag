---
layout: default
title: Supplementary
permalink: /supplementary/
---

## Supplementary information

Supporting tables for *A Visual-First Multimodal RAG for Retrieval-Assisted
Question Answering and Compliance Checking of Civil Standard Plans*
(submitted to *Automation in Construction*, AUTCON-D-26-04051).

Additional material supporting the article is published here. Nothing in this
page is needed to follow the argument of the paper: each table either records
implementation provenance or gives a breakdown whose headline figures are
already stated in the article text.

Every number below is reproduced by a script in `code/` and written to
`reports/`. See [`TABLE_PROVENANCE.md`](https://github.com/Nabarajsub/plansightrag/blob/main/TABLE_PROVENANCE.md) for the mapping from
each table to its backing report and script.

## Table S1. Implementation provenance for every retrieval run

Implementation provenance for every retrieval run. The generating script is the authoritative record of resolution, chunking, pooling, and query formulation for that run; its module docstring documents them in prose.

| Run | R@5 | OCR engine / model | Generating script |
|---|---|---|---|
| nemotron_colembed_8b | 95.28% | nvidia/nemotron-colembed-vl-8b-v2 | full_benchmark_retrieval.py |
| colnomic_rerank_monoqwen_424 | 94.58% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_page_plus_tiles | 93.87% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_patch_union | 93.87% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_page_plus_tiles_coarse | 93.63% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_patch_union_coarse | 93.40% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| nemotron_colembed_4b | 92.92% | nvidia/nemotron-colembed-vl-4b-v2 | full_benchmark_retrieval.py |
| colnomic_3b | 92.69% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_max_page_tile | 92.69% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_full_page_424 | 92.45% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_max_page_tile_coarse | 92.45% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_circularity | 92.45% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_blend_coarse | 91.98% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_7b | 91.27% | nomic-ai/colnomic-embed-multimodal-7b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_blend | 91.27% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_binary_quantized | 90.33% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_published_max_coarse | 90.09% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colqwen25_v0_2 | 87.26% | vidore/colqwen2.5-v0.2 | full_benchmark_retrieval.py |
| colnomic_tiling_v2_top3_sum_coarse | 84.20% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_v2_published_max | 83.96% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| colnomic_tiling_tile_level_424 | 83.73% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| tiling_tile_level_424 | 82.08% | vidore/colpali-v1.2 | -- |
| colnomic_tiling_v2_top3_sum | 81.84% | nomic-ai/colnomic-embed-multimodal-3b | full_benchmark_retrieval.py |
| tiling_full_page_424 | 76.89% | vidore/colpali-v1.2 | -- |
| colpali_v13 | 69.58% | vidore/colpali-v1.3 | full_benchmark_retrieval.py |
| hpc_colpali_bq | 65.09% | vidore/colpali-v1.2 (binary quantized) | full_benchmark_retrieval.py |
| ocr_bm25_meta | 58.02% | Tesseract OCR | bench_ocr_bm25.py |
| ocr_bm25 | 54.72% | Tesseract OCR | bench_ocr_bm25.py |
| visrag_ret | 53.77% | openbmb/VisRAG-Ret | bench_visrag.py |
| bge_m3_ocr_dense_sparse | 50.00% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_bge_m3_tuned.py |
| bge_m3_ocr_tuned | 50.00% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_bge_m3_tuned.py |
| bge_m3_ocr_chunk_maxpool | 46.93% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_bge_m3_tuned.py |
| visionrag_pyramid_fixed | 45.99% | sentence-transformers/all-MiniLM-L6-v2 | bench_visual_fixed.py |
| bge_m3_ocr_chunk_sparse | 44.81% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_bge_m3_tuned.py |
| ocr_minilm_chunked | 40.57% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_text_fixed.py |
| bge_m3_ocr | 36.79% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_bge_m3_tuned.py |
| bge_m3_ocr_full_single | 36.79% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_bge_m3_tuned.py |
| clip_l14_336_tiled | 36.79% | openai/clip-vit-large-patch14-336 | bench_weak_fixed.py |
| bge_m3_ocr_chunk_meanpool | 36.32% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_bge_m3_tuned.py |
| ocr_minilm | 25.24% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_text_fixed.py |
| visionrag_pyramid | 23.11% | VisionRAG (Pyramid, RRF-fused) | bench_visual_fixed.py |
| dse_qwen2_2b | 22.17% | MrLight/dse-qwen2-2b-mrl-v1 | full_benchmark_retrieval.py |
| pix2struct_textcaps_fixed | 16.27% | google/pix2struct-textcaps-base | bench_weak_fixed.py |
| clip_b32_tiled | 13.92% | openai/clip-vit-base-patch32 | bench_weak_fixed.py |
| pix2struct_docvqa_fixed | 9.67% | google/pix2struct-docvqa-base | bench_weak_fixed.py |
| pix2struct_decode_minilm | 7.31% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_weak_fixed.py |
| nougat_base_fixed | 2.36% | facebook/nougat-base | bench_weak_fixed.py |
| clip_b32_whole | 1.89% | openai/clip-vit-base-patch32 | bench_weak_fixed.py |
| clip_vit_b32 | 1.89% | openai/clip-vit-base-patch32 | bench_weak_fixed.py |
| nougat_decode_minilm | 0.47% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_weak_fixed.py |
| layoutlmv3 | 0.00% | microsoft/layoutlmv3-base | bench_weak_fixed.py |
| layoutlmv3_cls | 0.00% | microsoft/layoutlmv3-base | bench_weak_fixed.py |
| layoutlmv3_cls_centered | 0.00% | microsoft/layoutlmv3-base | bench_weak_fixed.py |
| layoutlmv3_meanpool | 0.00% | microsoft/layoutlmv3-base | bench_weak_fixed.py |
| layoutlmv3_meanpool_centered | 0.00% | microsoft/layoutlmv3-base | bench_weak_fixed.py |
| udop_decode_minilm | 0.00% | Tesseract OCR (baselines_v2/ocr_text.json) | bench_weak_fixed.py |
| udop_encoder_meanpool | 0.00% | microsoft/udop-large | bench_weak_fixed.py |
| udop_encoder_meanpool_centered | 0.00% | microsoft/udop-large | bench_weak_fixed.py |

## Table S2. Benchmark distribution by agency and reasoning category

Distribution of the 4,056-pair five-DOT benchmark by agency and reasoning category. The held-out Michigan DOT zero-shot set (93 pairs) is not included in this table.

| Category | WYDOT | Caltrans | AZDOT | ARDOT | FDOT | Total |
|---|---|---|---|---|---|---|
| Visual Interpretation | 201 | 416 | 149 | 85 | 549 | 1,400 |
| Dimensional Accuracy | 170 | 287 | 144 | 87 | 327 | 1,015 |
| Logical Reasoning | 172 | 331 | 104 | 104 | 446 | 1,157 |
| Hallucination Rate | 15 | 192 | 17 | 2 | 258 | 484 |
| Total | 558 | 1,226 | 414 | 278 | 1,580 | 4,056 |


*Note: the agency column shown as ARDOT is Arkansas material. It was recorded as
Colorado DOT (CDOT) in the initial release; the sheets carry Arkansas State Highway
Commission title blocks, and `data/sources.json` documents the correction. The local
collection key remains `colorado_2025` so that the frozen reports stay consistent.*

## Table S3. Human-curated 78-pair WYDOT anchor set

Human-curated 78-pair WYDOT anchor set: per-category Recall@5 and end-to-end judge accuracy with 95% bootstrap CIs (10,000 resamples, seed42). Retrieval is over the 237-page WYDOT-only index; judge accuracy uses the local Qwen2.5-VL-7B answerer with a Qwen2.5-VL-72B judge.

| Category | N | Recall@5 (%) | Judge (%) |
|---|---|---|---|
| Dimensional | 29 | 100.00 [100.0, 100.0] | 79.31 [62.1, 93.1] |
| Visual | 16 | 87.50 [68.8, 100.0] | 68.75 [43.8, 87.5] |
| Logical | 17 | 94.12 [82.4, 100.0] | 94.12 [82.4, 100.0] |
| Hallucination | 16 | 75.00 [50.0, 93.8] | 81.25 [62.5, 100.0] |
| Overall | 78 | 91.03 [84.6, 96.2] | 80.77 [71.8, 89.7] |

## Table S4. Retrieval-only latency distribution

Retrieval-only latency distribution (ms) for full-page ColNomic MaxSim retrieval on the 424-pair page-disjoint test split over the combined 1,898-page five-DOT visual index. Wall-clock time collected with time.perf_counter() on a single NVIDIA H100 80 GB HBM3 (PyTorch 2.5.1, CUDA 12.1; the device is recorded in the released result file), one query at a time (batch size 1).

| Percentile | Full-page (424 queries, 1,898-page index) |
|---|---|
| min | 85.40 |
| p25 | 94.43 |
| p50 | 97.94 |
| p75 | 100.61 |
| p95 | 102.86 |
| p99 | 103.66 |
| max | 151.97 |
| mean | 97.67 |

## Table S5. Multi-plan compliance before and after threshold pre-resolution

Multi-plan compliance verdict accuracy on the n=100 scale set, before and after replacing the textual d/2 threshold in the rules registry with a per-drawing pre-resolved numeric value.

| N plans | Before d/2 fix | After d/2 fix |
|---|---|---|
| N=2 | 25/25 | 25/25 |
| N=3 | 25/25 | 25/25 |
| N=4 | 25/25 | 25/25 |
| N=5 | 13/25 | 25/25 |
| Total | 88/100 (88%) | 100/100 (100%) |
