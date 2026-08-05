"""Legacy 78-question WYDOT comparison -- retrieval recompute (gemini-free).

Recomputes ColPali Recall@5 with 95% bootstrap CIs on the original 78-question
WYDOT subset, retrieving over the 237-page WYDOT-only slice of the v3 index
(faithful to the legacy "237-page WYDOT-only index" framing). Also caches the
top-3 retrieved pages per query so the downstream local-model VQA + judge stage
(eval_78q_judge via vqa_techniques.py / vqa_judge.py) can reuse them without
co-residing ColPali with the answerer VLM.

Outputs:
  reports/legacy78q_retrieval.json   per-category Recall@5 + bootstrap CIs
  retrieval_78q.json                 {query: [top3 image paths]} for the VQA stage

No Gemini anywhere: retrieval is generator-independent.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, os
import numpy as np
import torch
from colpali_engine.models import ColPali, ColPaliProcessor

ROOT = f"{PSR_ROOT}"
V3_INDEX = f"{ROOT}/build_v3/all_dot_index_v3.pt"
Q_FILES = [f"{ROOT}/iclr_2027_research/data_cache/wydot78_train.jsonl",
           f"{ROOT}/iclr_2027_research/data_cache/wydot78_dev.jsonl"]
OUT_JSON = f"{ROOT}/baselines_v2/reports/legacy78q_retrieval.json"
OUT_RETR = f"{ROOT}/baselines_v2/retrieval_78q.json"

N_BOOT = 10000
SEED = 42
# Match the published table caption ordering / labels.
CAT_ORDER = [
    ("Dimensional Accuracy", "Dimensional"),
    ("Visual Interpretation", "Visual"),
    ("Logical Reasoning", "Logical"),
    ("Hallucination Rate", "Hallucination"),
]


def load_questions():
    rows = []
    for f in Q_FILES:
        rows += [json.loads(l) for l in open(f)]
    return rows


def bootstrap_recall(hits, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    h = np.asarray(hits, dtype=np.float32)
    if h.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, h.size, size=(n_boot, h.size))
    samples = h[idx].mean(axis=1) * 100.0
    return (float(h.mean() * 100.0),
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))


def main():
    dev = "cuda"
    print("[load] ColPali vidore/colpali-v1.2")
    cp = ColPali.from_pretrained("vidore/colpali-v1.2", torch_dtype=torch.bfloat16,
                                 device_map=dev).eval()
    cpp = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    print(f"[load] {V3_INDEX}")
    recs = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    db = [{"emb": r["embedding"].to(dev).to(torch.bfloat16),
           "path": r["metadata"]["image_path"],
           "base": os.path.basename(r["metadata"]["image_path"])}
          for r in recs if r["metadata"]["agency"] == "WYDOT"]
    print(f"[index] {len(db)} WYDOT pages")

    rows = load_questions()
    print(f"[q] {len(rows)} questions")

    retr_out = {}
    hits_by_cat = {c: [] for c, _ in CAT_ORDER}
    all_hits = []
    for r in rows:
        inp = cpp.process_queries([r["query"]]).to(dev)
        with torch.no_grad():
            q = cp(**inp)
        scores = np.array([torch.matmul(q, d["emb"].T).max(dim=-1).values.sum(dim=-1).item()
                           for d in db])
        order = np.argsort(scores)[::-1]
        top5 = [db[j]["base"] for j in order[:5]]
        target = os.path.basename(r["image_path"])
        hit = int(target in top5)
        all_hits.append(hit)
        if r["category"] in hits_by_cat:
            hits_by_cat[r["category"]].append(hit)
        retr_out[r["query"]] = [db[j]["path"] for j in order[:3]]

    # ---- assemble report -----------------------------------------------------
    report = {"n": len(rows), "index_pages": len(db), "n_boot": N_BOOT, "seed": SEED}
    pt, lo, hi = bootstrap_recall(all_hits)
    report["overall"] = {"n": len(all_hits), "point": pt, "lo": lo, "hi": hi}
    report["per_category"] = {}
    for cat, label in CAT_ORDER:
        h = hits_by_cat[cat]
        pt, lo, hi = bootstrap_recall(h)
        report["per_category"][label] = {"n": len(h), "point": pt, "lo": lo, "hi": hi}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(report, open(OUT_JSON, "w"), indent=2)
    json.dump(retr_out, open(OUT_RETR, "w"), indent=2)

    print(f"[done] overall Recall@5 = {report['overall']['point']:.2f}% "
          f"({sum(all_hits)}/{len(all_hits)})")
    for cat, label in CAT_ORDER:
        c = report["per_category"][label]
        print(f"    {label:14s} n={c['n']:3d}  R@5={c['point']:.2f}% "
              f"[{c['lo']:.1f}, {c['hi']:.1f}]")
    print(f"[done] -> {OUT_JSON}")
    print(f"[done] -> {OUT_RETR}")


if __name__ == "__main__":
    main()
