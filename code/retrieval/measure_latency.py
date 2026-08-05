"""Re-measure retrieval-only latency on the 424-query test split.

Drop-in replacement for the legacy N=503 latency numbers. Measures the
ColPali (multi-vector MaxSim) retrieval step on the full 1,898-page
five-DOT index.
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
from colpali_engine.models import ColPali, ColPaliProcessor

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_test

COLPALI_ID = "vidore/colpali-v1.2"
V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"
OUT = f"{PSR_ROOT}/baselines_v2/reports/latency_424.json"


@torch.no_grad()
def main():
    device = "cuda"
    processor = ColPaliProcessor.from_pretrained(COLPALI_ID)
    model = ColPali.from_pretrained(COLPALI_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
    print("[load] loading index")
    recs = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    docs = [{"emb": r["embedding"].to(device).to(torch.bfloat16),
             "meta": r["metadata"]} for r in recs]
    print(f"[load] {len(docs)} pages")

    test = load_test()
    timings = []
    # Warmup
    for q in test[:5]:
        inputs = processor.process_queries([q["query"]]).to(device)
        q_emb = model(**inputs)
        torch.cuda.synchronize()
    # Real measurement
    for r in test:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        inputs = processor.process_queries([r["query"]]).to(device)
        q_emb = model(**inputs)
        scores = []
        for d in docs:
            inter = torch.matmul(q_emb, d["emb"].T)
            scores.append(inter.max(dim=-1).values.sum(dim=-1).item())
        _ = np.argsort(scores)[::-1][:5]
        torch.cuda.synchronize(); timings.append((time.perf_counter() - t0) * 1000)

    a = np.asarray(timings)
    out = {
        "n_queries": len(test), "n_pages": len(docs), "unit": "ms",
        "min": float(a.min()), "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
        "max": float(a.max()), "mean": float(a.mean()),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
