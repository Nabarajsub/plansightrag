---
layout: default
title: Overview
---

## PlanSightRAG

Reproducibility package for *A Visual-First Multimodal RAG for Retrieval-Assisted
Question Answering and Compliance Checking of Civil Standard Plans*, submitted to
*Automation in Construction*.

Compliance review of civil infrastructure still means an engineer reading dense 2D
standard plans. OCR-based automation discards the geometry, layout and cross-view
symbols that carry the meaning of those sheets. PlanSightRAG indexes the plans as
page images and retrieves over them directly, pairing multi-vector late-interaction
retrieval with a vision–language reasoning stage and an agentic compliance pipeline.

### What is here

- **[Supplementary]({{ '/supplementary/' | relative_url }})** — additional tables
  supporting the article (S1–S5).
- **[Reproducibility]({{ '/reproducibility/' | relative_url }})** — how every number
  in every table is regenerated.
- **[Demo]({{ '/demo/' | relative_url }})** — the system running on a real query.

### Scope of the evidence

Retrieval is the strongest result and is measured on real standard plans across five
state agencies. Compliance checking is evaluated primarily on synthetic drawings with
controlled ground truth, with the target sheet supplied rather than retrieved.
Autonomous rule grounding against a real specification corpus retrieves the governing
sheet reliably but extracts the numeric limit in fewer than one case in five, and that
gap is the open problem the work leaves. The evidence supports retrieval-assisted
engineering review, not end-to-end automated compliance checking.

### Citation

See [`CITATION.cff`](https://github.com/Nabarajsub/plansightrag/blob/main/CITATION.cff).
