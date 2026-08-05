"""Shared retrieval-eval helpers for all strong + weak baselines.

A baseline only needs to provide:
  - encode_pages(image_paths) -> page_embs (any tensor/array indexed 0..N-1)
  - encode_queries(queries) -> query_embs
  - score(query_embs, page_embs) -> (Nq, Np) score matrix

Then run_eval() handles loading the 424-pair test split, computing Recall@5
overall and per-agency, and writing a JSON report.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, os, time
from collections import defaultdict

import numpy as np

PAGES = f"{PSR_ROOT}/baselines_v2/pages.json"
TEST = f"{PSR_ROOT}/lora_finetune/split_test.jsonl"
REPORTS = f"{PSR_ROOT}/baselines_v2/reports"

TOP_K = 5


def load_pages():
    return json.load(open(PAGES))


def load_test():
    return [json.loads(l) for l in open(TEST)]


def recall_at_k(score_matrix, queries, pages, k=TOP_K):
    """score_matrix: (Nq, Np); queries: list with image_path each; pages: list with image_path each.
    Returns overall_recall, per_agency dict, top1_hit_array (for bootstrap CIs)."""
    path_to_idx = {p["image_path"]: i for i, p in enumerate(pages)}
    by_agency = defaultdict(lambda: [0, 0])
    by_category = defaultdict(lambda: [0, 0])
    hits = []
    total = 0
    for qi, q in enumerate(queries):
        scores = score_matrix[qi]
        top = np.argsort(scores)[::-1][:k]
        gold_idx = path_to_idx.get(q["image_path"])
        hit = int(gold_idx in top.tolist()) if gold_idx is not None else 0
        hits.append(hit)
        total += hit
        by_agency[q["agency"]][0] += hit
        by_agency[q["agency"]][1] += 1
        cat = q.get("category", "unknown")
        by_category[cat][0] += hit
        by_category[cat][1] += 1
    overall = 100.0 * total / max(len(queries), 1)
    per_agency = {a: {"n": v[1], "recall@5": 100.0 * v[0] / max(v[1], 1)}
                  for a, v in by_agency.items()}
    per_cat = {c: {"n": v[1], "recall@5": 100.0 * v[0] / max(v[1], 1)}
               for c, v in by_category.items()}
    return overall, per_agency, per_cat, hits


def save_report(tag, overall, per_agency, per_cat, hits, n_pages, elapsed, extra=None):
    os.makedirs(REPORTS, exist_ok=True)
    rep = {
        "tag": tag,
        "n_test_queries": len(hits),
        "n_index_pages": n_pages,
        "recall@5_overall": overall,
        "per_agency_recall@5": per_agency,
        "per_category_recall@5": per_cat,
        "hits": hits,
        "elapsed_sec": elapsed,
    }
    if extra:
        rep.update(extra)
    out = os.path.join(REPORTS, f"{tag}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[done] {tag}: Recall@5 = {overall:.2f}%  ({sum(hits)}/{len(hits)})  -> {out}")
    for a, v in sorted(per_agency.items()):
        print(f"    {a:10s}  n={v['n']:4d}  R@5={v['recall@5']:.2f}%")
    return rep
