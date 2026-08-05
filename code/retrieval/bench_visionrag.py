"""Weak baseline (re-run): VisionRAG (Pyramid) on the 1,898-page / 424-query eval.

Approximates the original 'Pyramid' VisionRAG: three parallel text-side
streams (global caption / structural / factual) fused with RRF. Without
re-running the Qwen-VL captioner on 1,898 pages we approximate the three
streams from the OCR text with three different text-encoder prompts.
This is the most faithful low-compute reproduction of the original
hybrid baseline.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, os, sys, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

OCR = f"{PSR_ROOT}/baselines_v2/ocr_text.json"
TAG = "visionrag_pyramid"
INDEX = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"
RRF_K = 60


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    pages = load_pages()
    ocr = json.load(open(OCR))
    assert [r["image_path"] for r in ocr] == [p["image_path"] for p in pages]

    if os.path.exists(INDEX):
        d = torch.load(INDEX, weights_only=False)
        e_global, e_struct, e_fact = d["global"], d["struct"], d["fact"]
    else:
        global_texts = [f"Document: {r['ocr_text'][:1000]}" for r in ocr]
        struct_texts = [f"Section views, tables and dimensioned details: {r['ocr_text'][:1500]}" for r in ocr]
        fact_texts = [f"Specific numeric values and labels: {r['ocr_text'][:2000]}" for r in ocr]
        e_global = torch.tensor(model.encode(global_texts, batch_size=64, normalize_embeddings=True))
        e_struct = torch.tensor(model.encode(struct_texts, batch_size=64, normalize_embeddings=True))
        e_fact   = torch.tensor(model.encode(fact_texts,   batch_size=64, normalize_embeddings=True))
        os.makedirs(os.path.dirname(INDEX), exist_ok=True)
        torch.save({"global": e_global, "struct": e_struct, "fact": e_fact}, INDEX)
    t_index = time.time() - t0

    test = load_test()
    qs = [r["query"] for r in test]
    q_g = torch.tensor(model.encode(qs, batch_size=128, normalize_embeddings=True))
    q_s = torch.tensor(model.encode([f"Sections, tables, dimensions: {q}" for q in qs],
                                     batch_size=128, normalize_embeddings=True))
    q_f = torch.tensor(model.encode([f"Specific values and labels: {q}" for q in qs],
                                     batch_size=128, normalize_embeddings=True))
    s_g = (q_g @ e_global.T).numpy()
    s_s = (q_s @ e_struct.T).numpy()
    s_f = (q_f @ e_fact.T).numpy()

    # Reciprocal Rank Fusion across the three streams
    def rrf(scores):
        Nq, Np = scores.shape
        ranks = np.argsort(-scores, axis=1)  # rank=0 best
        rr = np.zeros_like(scores)
        for qi in range(Nq):
            for r, pi in enumerate(ranks[qi]):
                rr[qi, pi] += 1.0 / (RRF_K + r + 1)
        return rr

    fused = rrf(s_g) + rrf(s_s) + rrf(s_f)
    overall, per_agency, per_cat, hits = recall_at_k(fused, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": "VisionRAG (Pyramid, RRF-fused)",
                                          "rrf_k": RRF_K, "t_index_sec": t_index})


if __name__ == "__main__":
    main()
