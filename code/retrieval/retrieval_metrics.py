"""Rank-sensitive retrieval metrics and the Michigan transfer over the joint index.

  Rank-sensitive metrics on the 424-pair test split: Recall@1, Recall@5, MRR, nDCG@10
  Michigan transfer over the JOINT 1,898+298 index (not the 298-only pool), to
    remove the pool-size confound; report joint vs Michigan-only.
Reuses the cached ColNomic page embeddings for the 1,898 plan pages.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import glob, json, math, os
import numpy as np, torch
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

ROOT = f"{PSR_ROOT}"
CACHE = f"{ROOT}/baselines_v2/cache_colnomic_pages.pt"
TEST = f"{ROOT}/lora_finetune/split_test.jsonl"
MICH_DIR = f"{ROOT}/data/michigan_2025/images"
MICH_QA = f"{ROOT}/qna_expansion/verified_michigan.jsonl"
MODEL = "nomic-ai/colnomic-embed-multimodal-3b"
OUT = f"{ROOT}/baselines_v2/reports/retrieval_metrics.json"
DEV = "cuda"


def encode_q(model, proc, qs):
    out = []
    for s in range(0, len(qs), 16):
        with torch.no_grad():
            e = model(**proc.process_queries(qs[s:s+16]).to(DEV))
        out += [e[j].to(torch.float16).cpu() for j in range(e.shape[0])]
    return out


def encode_img(model, proc, paths):
    out = []
    for s in tqdm(range(0, len(paths), 4), desc="enc img"):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s+4]]
        with torch.no_grad():
            e = model(**proc.process_images(imgs).to(DEV))
        out += [e[j].to(torch.float16).cpu() for j in range(e.shape[0])]
    return out


def ranks(qembs, pembs, qrows, ppaths):
    bn = [os.path.basename(p) for p in ppaths]
    pg = [pe.to(DEV).float() for pe in pembs]
    rks = []
    for qi, r in enumerate(qrows):
        qe = qembs[qi].to(DEV).float()
        sc = np.array([torch.matmul(qe, pe.T).max(-1).values.sum().item() for pe in pg])
        order = np.argsort(sc)[::-1]
        gold = os.path.basename(r["image_path"])
        rank = next((i+1 for i, j in enumerate(order) if bn[j] == gold), None)
        rks.append(rank)
    del pg; torch.cuda.empty_cache()
    return rks


def metrics(rks, ks=(1, 5)):
    n = len(rks)
    out = {f"recall@{k}": round(100.0*sum(1 for r in rks if r and r <= k)/n, 2) for k in ks}
    out["MRR"] = round(sum(1.0/r for r in rks if r)/n, 4)
    out["nDCG@10"] = round(sum(1.0/math.log2(r+1) for r in rks if r and r <= 10)/n, 4)
    out["n"] = n
    return out


def main():
    blob = torch.load(CACHE, weights_only=False)
    pembs, ppaths = blob["pembs"], blob["paths"]
    test = [json.loads(l) for l in open(TEST)]
    mich = [json.loads(l) for l in open(MICH_QA)]
    mpaths = sorted(glob.glob(f"{MICH_DIR}/*.png"))
    print(f"[data] {len(ppaths)} plan pages | {len(test)} test Q | {len(mpaths)} Michigan pages | {len(mich)} Michigan Q")

    model = ColQwen2_5.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=DEV).eval()
    proc = ColQwen2_5_Processor.from_pretrained(MODEL)

    # #9: 424 test split rank metrics over the 1,898-page index
    tq = encode_q(model, proc, [r["query"] for r in test])
    test_rks = ranks(tq, pembs, test, ppaths)
    by_cat = {}
    for r, rk in zip(test, test_rks):
        by_cat.setdefault(r.get("category", "?"), []).append(rk)

    # #2: Michigan over joint index vs Michigan-only
    membs = encode_img(model, proc, mpaths)
    mq = encode_q(model, proc, [r["query"] for r in mich])
    joint_embs = pembs + membs
    joint_paths = ppaths + mpaths
    mich_joint = ranks(mq, joint_embs, mich, joint_paths)
    mich_only = ranks(mq, membs, mich, mpaths)

    rep = {
        "test_424_rank_metrics": metrics(test_rks),
        "test_424_by_category": {c: metrics(v) for c, v in sorted(by_cat.items())},
        "michigan_joint_index": {**metrics(mich_joint), "n_candidates": len(joint_paths)},
        "michigan_only_index":  {**metrics(mich_only),  "n_candidates": len(mpaths)},
    }
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
