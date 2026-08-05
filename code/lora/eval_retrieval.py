"""Evaluate retrieval Recall@5: zero-shot ColPali vs LoRA-tuned ColPali.

Runs on:
  - split_test.jsonl   (424 pairs, 5 DOTs, page-disjoint from train)  -> in-distribution
  - verified_michigan  ( 93 pairs, MDOT)                              -> zero-shot 6th DOT

Index = the v3 1898-page index (+ Michigan index for the MDOT eval).
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---


import argparse, json, os, sys, time
from collections import defaultdict

import numpy as np
import torch
from peft import PeftModel
from colpali_engine.models import ColPali, ColPaliProcessor

COLPALI_ID = "vidore/colpali-v1.2"
V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"
MICHIGAN_INDEX = f"{PSR_ROOT}/qna_expansion/michigan_index.pt"
TEST = f"{PSR_ROOT}/lora_finetune/split_test.jsonl"
MICHIGAN_QNA = f"{PSR_ROOT}/qna_expansion/verified_michigan.jsonl"
TOP_K = 5


def load_index(path, device):
    recs = torch.load(path, weights_only=False, map_location="cpu")
    return [{"embedding": r["embedding"].to(device).to(torch.bfloat16),
             "metadata": r["metadata"]} for r in recs]


@torch.no_grad()
def recall_at_k(model, processor, qna_rows, db, device, k=TOP_K):
    by_agency = defaultdict(lambda: [0, 0])  # agency -> [hits, total]
    total_hit = 0
    for r in qna_rows:
        inputs = processor.process_queries([r["query"]]).to(device)
        q_emb = model(**inputs)
        scores = []
        for doc in db:
            inter = torch.matmul(q_emb, doc["embedding"].T)
            scores.append(inter.max(dim=-1).values.sum(dim=-1).item())
        top = np.argsort(scores)[::-1][:k]
        hit = any(db[i]["metadata"]["image_path"] == r["image_path"] for i in top)
        total_hit += int(hit)
        by_agency[r["agency"]][0] += int(hit)
        by_agency[r["agency"]][1] += 1
    overall = 100.0 * total_hit / max(len(qna_rows), 1)
    per_agency = {a: {"n": v[1], "recall@5": 100.0 * v[0] / max(v[1], 1)}
                  for a, v in by_agency.items()}
    return overall, per_agency


def build_model(adapter_path, device):
    base = ColPali.from_pretrained(COLPALI_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
    if adapter_path and os.path.exists(adapter_path):
        print(f"[model] loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path).eval()
        return model
    print("[model] zero-shot ColPali (no adapter)")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="", help="path to LoRA adapter; empty = zero-shot")
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--out", default=f"{PSR_ROOT}/lora_finetune/eval_report.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    processor = ColPaliProcessor.from_pretrained(COLPALI_ID)
    model = build_model(args.adapter, device)

    print("[eval] loading v3 index")
    v3 = load_index(V3_INDEX, device)
    test_rows = [json.loads(l) for l in open(TEST)]
    print(f"[eval] in-distribution test: {len(test_rows)} pairs over {len(v3)}-page index")
    test_overall, test_per_agency = recall_at_k(model, processor, test_rows, v3, device)

    print("[eval] loading Michigan index")
    mich = load_index(MICHIGAN_INDEX, device)
    mich_rows = [json.loads(l) for l in open(MICHIGAN_QNA)]
    print(f"[eval] Michigan zero-shot: {len(mich_rows)} pairs over {len(mich)}-page index")
    mich_overall, _ = recall_at_k(model, processor, mich_rows, mich, device)

    report = {
        "tag": args.tag, "adapter": args.adapter or "zero-shot",
        "test_recall@5_overall": test_overall,
        "test_per_agency": test_per_agency,
        "michigan_zeroshot_recall@5": mich_overall,
        "michigan_n": len(mich_rows),
        "elapsed_min": (time.time() - t0) / 60,
    }
    # Merge into a multi-config report file
    allrep = {}
    if os.path.exists(args.out):
        allrep = json.load(open(args.out))
    allrep[args.tag] = report
    with open(args.out, "w") as f:
        json.dump(allrep, f, indent=2)

    print(f"\n=== {args.tag} ===")
    print(f"  in-distribution test Recall@5: {test_overall:.2f}%")
    for a, v in sorted(test_per_agency.items()):
        print(f"    {a:10s} n={v['n']:4d}  R@5={v['recall@5']:.2f}%")
    print(f"  Michigan zero-shot Recall@5: {mich_overall:.2f}%  (n={len(mich_rows)})")
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
