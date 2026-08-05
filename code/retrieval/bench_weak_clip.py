"""Weak baseline (re-run): CLIP ViT-B/32 on the 1,898-page / 424-query eval.

Single global-vector visual retrieval. Cosine similarity.
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
from transformers import CLIPModel, CLIPProcessor

# openai/clip-vit-base-patch32 ships only pytorch_model.bin (no safetensors), and
# torch 2.5.1 (<2.6) trips transformers' CVE-2025-32434 guard. The checkpoint is
# the official OpenAI release and is still loaded with weights_only=True, so we
# disable the guard for this trusted file.
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

MODEL_ID = "openai/clip-vit-base-patch32"
TAG = "clip_vit_b32"
INDEX = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"
BATCH = 32


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    pages = load_pages()
    if os.path.exists(INDEX):
        page_embs = torch.load(INDEX, weights_only=False)
    else:
        embs = []
        for i in tqdm(range(0, len(pages), BATCH), desc="encode pages"):
            chunk = pages[i:i + BATCH]
            imgs = [Image.open(p["image_path"]).convert("RGB") for p in chunk]
            inp = processor(images=imgs, return_tensors="pt").to(device)
            e = model.get_image_features(**inp)
            embs.append(F.normalize(e.float(), dim=-1).cpu())
        page_embs = torch.cat(embs, dim=0)
        os.makedirs(os.path.dirname(INDEX), exist_ok=True)
        torch.save(page_embs, INDEX)
    t_index = time.time() - t0

    test = load_test()
    qs = [r["query"] for r in test]
    q_embs = []
    for i in tqdm(range(0, len(qs), BATCH), desc="encode queries"):
        chunk = qs[i:i + BATCH]
        inp = processor(text=chunk, return_tensors="pt", padding=True, truncation=True).to(device)
        e = model.get_text_features(**inp)
        q_embs.append(F.normalize(e.float(), dim=-1).cpu())
    q_embs = torch.cat(q_embs, dim=0)
    score_matrix = (q_embs @ page_embs.T).numpy()
    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": MODEL_ID, "t_index_sec": t_index})


if __name__ == "__main__":
    main()
