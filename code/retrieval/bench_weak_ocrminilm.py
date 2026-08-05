"""Weak baseline (re-run): OCR + MiniLM dense text retrieval.

Uses the shared ocr_text.json dump. Embedder = sentence-transformers/all-MiniLM-L6-v2.
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
import torch, torch.nn.functional as F
from sentence_transformers import SentenceTransformer

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

OCR = f"{PSR_ROOT}/baselines_v2/ocr_text.json"
TAG = "ocr_minilm"
INDEX = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"


def main():
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                device="cuda" if torch.cuda.is_available() else "cpu")
    pages = load_pages()
    ocr = json.load(open(OCR))
    assert [r["image_path"] for r in ocr] == [p["image_path"] for p in pages]
    if os.path.exists(INDEX):
        page_embs = torch.load(INDEX, weights_only=False)
    else:
        page_embs = torch.tensor(
            model.encode([r["ocr_text"] or " " for r in ocr], batch_size=128, normalize_embeddings=True)
        )
        os.makedirs(os.path.dirname(INDEX), exist_ok=True)
        torch.save(page_embs, INDEX)
    t_index = time.time() - t0
    test = load_test()
    q_embs = torch.tensor(model.encode([r["query"] for r in test], batch_size=128, normalize_embeddings=True))
    score_matrix = (q_embs @ page_embs.T).numpy()
    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": "all-MiniLM-L6-v2", "t_index_sec": t_index})


if __name__ == "__main__":
    main()
