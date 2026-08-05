"""Construct-validity / circularity experiment for manuscript_swap.

The QnA benchmark was built with ColPali in the refine loop (a question is
accepted when ColPali ranks its target page in the top-5; otherwise it is
rephrased up to 5x). This raises the concern that the benchmark is merely
"ColPali-friendly" and that ColNomic's strong score is an artifact of the
curation retriever.

We test this directly: compute the ADOPTED ColNomic-3B Recall@5 on the 424-pair
test split, split by whether ColPali succeeded (final_hit=True) or FAILED
(final_hit=False) during curation. If ColNomic retrieves the ColPali-MISS
subset well, the benchmark is not merely ColPali-friendly and ColNomic's
advantage is genuine, not curation-induced.

Also reports the breakdown by category, drafter, and n_attempts.
Page embeddings are cached to disk for reuse.
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
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

ROOT = f"{PSR_ROOT}"
MODEL_ID = "nomic-ai/colnomic-embed-multimodal-3b"
PAGES = f"{ROOT}/baselines_v2/pages.json"
TEST = f"{ROOT}/lora_finetune/split_test.jsonl"
CACHE = f"{ROOT}/baselines_v2/cache_colnomic_pages.pt"
OUT = f"{ROOT}/baselines_v2/reports/colnomic_circularity.json"
DEV = "cuda"


def encode_images(model, proc, paths):
    embs = []
    for s in tqdm(range(0, len(paths), 4), desc="enc pages"):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + 4]]
        with torch.no_grad():
            out = model(**proc.process_images(imgs).to(DEV))
        embs += [out[j].to(torch.float16).cpu() for j in range(out.shape[0])]
    return embs


def encode_queries(model, proc, queries):
    embs = []
    for s in range(0, len(queries), 16):
        with torch.no_grad():
            out = model(**proc.process_queries(queries[s:s + 16]).to(DEV))
        embs += [out[j].to(torch.float16).cpu() for j in range(out.shape[0])]
    return embs


def per_query_hits(qembs, pembs, qrows, ppaths, k=5):
    bn = [os.path.basename(p) for p in ppaths]
    hits = []
    for qi, r in enumerate(qrows):
        qe = qembs[qi].to(DEV).float()
        sc = np.array([torch.matmul(qe, pembs[pi].to(DEV).float().T).max(-1).values.sum().item()
                       for pi in range(len(pembs))])
        top = {bn[j] for j in np.argsort(sc)[::-1][:k]}
        hits.append(int(os.path.basename(r["image_path"]) in top))
    return hits


def group_recall(rows, hits, keyfn):
    agg = {}
    for r, h in zip(rows, hits):
        key = keyfn(r)
        a = agg.setdefault(key, [0, 0])
        a[0] += h; a[1] += 1
    return {str(k): {"n": v[1], "recall@5": round(100.0 * v[0] / max(v[1], 1), 2)}
            for k, v in sorted(agg.items(), key=lambda kv: str(kv[0]))}


def main():
    t0 = time.time()
    pages = json.load(open(PAGES))
    test = [json.loads(l) for l in open(TEST)]
    ppaths = [p["image_path"] for p in pages]
    print(f"[data] {len(pages)} index pages | {len(test)} test queries")

    model = ColQwen2_5.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEV)
    proc = ColQwen2_5_Processor.from_pretrained(MODEL_ID)
    model.eval()

    if os.path.exists(CACHE):
        print(f"[cache] loading page embeddings from {CACHE}")
        blob = torch.load(CACHE, weights_only=False)
        pembs = blob["pembs"]
        assert blob["paths"] == ppaths, "cache paths mismatch; delete cache"
    else:
        pembs = encode_images(model, proc, ppaths)
        torch.save({"pembs": pembs, "paths": ppaths}, CACHE)
        print(f"[cache] saved page embeddings to {CACHE}")

    qembs = encode_queries(model, proc, [r["query"] for r in test])
    hits = per_query_hits(qembs, pembs, test, ppaths)
    overall = round(100.0 * sum(hits) / len(hits), 2)

    # Key split: ColPali curation hit vs miss
    by_colpali = group_recall(test, hits, lambda r: "colpali_HIT" if r.get("final_hit") else "colpali_MISS")
    by_cat = group_recall(test, hits, lambda r: r.get("category", "?"))
    by_drafter = group_recall(test, hits, lambda r: r.get("drafter", "?"))
    by_attempts = group_recall(test, hits, lambda r: f"att_{r.get('n_attempts','?')}")

    rep = {
        "experiment": "circularity / construct-validity (ColNomic on ColPali-hit vs ColPali-miss)",
        "retriever": MODEL_ID,
        "n_test": len(test),
        "recall@5_overall": overall,
        "by_colpali_curation": by_colpali,
        "by_category": by_cat,
        "by_drafter": by_drafter,
        "by_n_attempts": by_attempts,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps({"overall": overall, "by_colpali_curation": by_colpali}, indent=2))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
