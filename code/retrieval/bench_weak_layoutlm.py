"""Weak baseline (re-run): LayoutLMv3 on the 1,898-page / 424-query eval.

LayoutLMv3 ingests page image + OCR token boxes; uses the [CLS] embedding
as a global page representation. Falls back to tesseract OCR boxes per page.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pytesseract
from transformers import LayoutLMv3Processor, LayoutLMv3Model

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

MODEL_ID = "microsoft/layoutlmv3-base"
TAG = "layoutlmv3"
INDEX = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"


def _ocr_boxes(pil):
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    W, H = pil.size
    for i, w in enumerate(data["text"]):
        if not w.strip():
            continue
        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append(w)
        # LayoutLMv3 wants 0-1000 normalized
        boxes.append([int(1000 * x / W), int(1000 * y / H),
                      int(1000 * (x + ww) / W), int(1000 * (y + hh) / H)])
    if not words:
        words, boxes = ["page"], [[0, 0, 1000, 1000]]
    return words, boxes


@torch.no_grad()
def main():
    device = "cuda"
    t0 = time.time()
    processor = LayoutLMv3Processor.from_pretrained(MODEL_ID, apply_ocr=False)
    model = LayoutLMv3Model.from_pretrained(MODEL_ID).to(device).eval()
    pages = load_pages()
    if os.path.exists(INDEX):
        page_embs = torch.load(INDEX, weights_only=False)
    else:
        embs = []
        for p in tqdm(pages, desc="encode pages"):
            pil = Image.open(p["image_path"]).convert("RGB")
            try:
                words, boxes = _ocr_boxes(pil)
                enc = processor(pil, words, boxes=boxes, return_tensors="pt",
                                truncation=True, max_length=512).to(device)
                out = model(**enc)
                emb = F.normalize(out.last_hidden_state[:, 0], dim=-1).float().cpu()
            except Exception as e:
                emb = torch.zeros(1, 768)
            embs.append(emb)
        page_embs = torch.cat(embs, dim=0)
        os.makedirs(os.path.dirname(INDEX), exist_ok=True)
        torch.save(page_embs, INDEX)
    t_index = time.time() - t0

    test = load_test()
    qs = [r["query"] for r in test]
    # Encode queries via the same encoder with a blank-page image and word list.
    blank = Image.new("RGB", (224, 224), color="white")
    q_embs = []
    for q in tqdm(qs, desc="encode queries"):
        toks = q.split()[:200] or ["query"]
        boxes = [[0, 0, 1000, 1000]] * len(toks)
        enc = processor(blank, toks, boxes=boxes, return_tensors="pt",
                        truncation=True, max_length=512).to(device)
        out = model(**enc)
        q_embs.append(F.normalize(out.last_hidden_state[:, 0], dim=-1).float().cpu())
    q_embs = torch.cat(q_embs, dim=0)
    score_matrix = (q_embs @ page_embs.T).numpy()
    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": MODEL_ID, "t_index_sec": t_index})


if __name__ == "__main__":
    main()
