"""Nemotron ColEmbed V2 (NVIDIA) -- current ViDoRe-V3 SOTA late-interaction
visual retriever -- on the 424-pair test split, same protocol as the other Col*
baselines.

Backbones: -4b-v2 = Qwen3-VL-4B, -8b-v2 = Qwen3-VL-8B (newer than the Qwen2.5-VL
that ColQwen2.5 uses). License: CC-BY-NC-4.0 (research-only) -- baseline comparison
is research use; flagged in the paper.

Usage: python bench_nemotron.py --model nvidia/nemotron-colembed-vl-4b-v2 --tag nemotron_colembed_4b
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import argparse, sys, time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None

PAGE_CHUNK = 160  # encode + score this many page images at a time (memory control)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = "cuda"

    print(f"[load] {args.model}")
    t0 = time.time()
    # flash_attn isn't installed in this env -> sdpa
    model = AutoModel.from_pretrained(
        args.model, device_map=dev, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    print(f"[load] {time.time()-t0:.1f}s")

    pages = load_pages()
    test = load_test()
    queries = [r["query"] for r in test]
    print(f"[data] {len(pages)} pages, {len(queries)} queries")

    with torch.no_grad():
        q_embs = model.forward_queries(queries, batch_size=8)

    scores = np.zeros((len(queries), len(pages)), dtype=np.float32)
    for s in tqdm(range(0, len(pages), PAGE_CHUNK), desc="encode+score pages"):
        chunk = pages[s:s + PAGE_CHUNK]
        imgs = []
        ok_idx = []
        for k, p in enumerate(chunk):
            try:
                imgs.append(Image.open(p["image_path"]).convert("RGB"))
                ok_idx.append(k)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            img_embs = model.forward_images(imgs, batch_size=4)
            sc = model.get_scores(q_embs, img_embs)  # [n_queries x n_imgs]
        sc = sc.detach().to(torch.float32).cpu().numpy()
        for col, k in enumerate(ok_idx):
            scores[:, s + k] = sc[:, col]

    overall, per_agency, per_cat, hits = recall_at_k(scores, test, pages)
    save_report(args.tag, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": args.model, "license": "CC-BY-NC-4.0"})
    print(f"\n[done] {args.tag}: Recall@5 = {overall:.2f}%")
    for a, v in sorted(per_agency.items()):
        print(f"    {a:10s} R@5={v['recall@5']:.2f}%")


if __name__ == "__main__":
    main()
