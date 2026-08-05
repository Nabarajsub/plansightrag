"""Strong baseline: ColQwen2.5-v0.2 -- ColPali successor with Qwen2.5-VL backbone.

Index 1,898 pages, evaluate Recall@5 on the 424-pair page-disjoint test split.
Same evaluation protocol as eval_retrieval.py (MaxSim, k=5).
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
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

MODEL_ID = "vidore/colqwen2.5-v0.2"
TAG = "colqwen25_v0_2"
INDEX_PATH = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"
BATCH_PAGES = 2  # multi-vector, keep small
BATCH_QUERIES = 16


@torch.no_grad()
def main():
    device = "cuda"
    print(f"[load] {MODEL_ID}")
    t0 = time.time()
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_ID)
    model = ColQwen2_5.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa",
    ).eval()
    print(f"[load] {time.time() - t0:.1f}s")

    pages = load_pages()
    print(f"[index] {len(pages)} pages")

    # ---- Encode pages (with persistent on-disk cache so reruns are cheap) ----
    if os.path.exists(INDEX_PATH):
        print(f"[index] loading cached embeddings from {INDEX_PATH}")
        page_embs_cpu = torch.load(INDEX_PATH, weights_only=False)
    else:
        page_embs_cpu = []
        for i in tqdm(range(0, len(pages), BATCH_PAGES), desc="encode pages"):
            chunk = pages[i:i + BATCH_PAGES]
            imgs = [Image.open(p["image_path"]).convert("RGB") for p in chunk]
            batch = processor.process_images(imgs).to(device)
            out = model(**batch)  # (B, N_patches, D)
            for j in range(out.shape[0]):
                page_embs_cpu.append(out[j].to(torch.float16).cpu())
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        torch.save(page_embs_cpu, INDEX_PATH)
        print(f"[index] saved -> {INDEX_PATH}")
    t_index = time.time() - t0

    # ---- Encode queries ----
    test = load_test()
    queries = [r["query"] for r in test]
    print(f"[query] encoding {len(queries)} queries")
    q_embs = []
    for i in tqdm(range(0, len(queries), BATCH_QUERIES), desc="encode queries"):
        chunk = queries[i:i + BATCH_QUERIES]
        batch = processor.process_queries(chunk).to(device)
        out = model(**batch)  # (B, Lq, D)
        for j in range(out.shape[0]):
            q_embs.append(out[j].to(torch.float16).cpu())

    # ---- MaxSim scoring ----
    print("[score] MaxSim")
    score_matrix = np.zeros((len(q_embs), len(page_embs_cpu)), dtype=np.float32)
    # Push pages to GPU one at a time (multi-vector, memory-careful)
    q_embs_gpu = [q.to(device).to(torch.float32) for q in q_embs]
    for pi, pe in enumerate(tqdm(page_embs_cpu, desc="score")):
        pe_gpu = pe.to(device).to(torch.float32)
        for qi, qe in enumerate(q_embs_gpu):
            inter = torch.matmul(qe, pe_gpu.T)  # (Lq, Np)
            score_matrix[qi, pi] = inter.max(dim=-1).values.sum().item()

    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": MODEL_ID, "t_index_sec": t_index})


if __name__ == "__main__":
    main()
