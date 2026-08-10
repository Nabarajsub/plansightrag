"""Compliance-set retrieval on the adopted backbone (A29b follow-up, Cluster E).

The retrieval rung of the capability ladder was measured with ColPali-v1.2 -- the
predecessor backbone -- because `compliance/compliance_smoke_n10.py` hardcodes
`vidore/colpali-v1.2`. That leaves the compliance table inconsistent with the rest
of the paper, which is exactly the objection R1.8/R2.7/R2-repro 5 raise.

This re-measures it with ColNomic-3B under the identical protocol: the 100
CAD mockups are embedded and added to the 1,898-page five-DOT index, each
compliance question is issued against the joint index, and we ask whether the
mockup the question was written about lands in the top 5.

It also reports the diagnosis that matters for interpreting the number: when the
target is missed, is the top-5 filled with *other synthetic mockups of the same
archetype*? The generator produced 100 drawings from only 5 archetypes, so ~20
sheets differ from each other only in printed dimension values. If the misses are
dominated by same-archetype confusions, the figure measures near-duplicate
disambiguation on synthetic data and is not comparable to H1.

    python compliance_retrieval_colnomic.py
"""

# --- release path resolution (added for the release copy; the run-time originals
# under baselines_v2/ and explanability/ are unchanged and still carry the
# absolute ARCC paths the experiments were executed with) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter

import torch
from PIL import Image

ROOT = CLUSTER_ROOT
CACHE = f"{ROOT}/baselines_v2/cache_colnomic_pages.pt"
MANIFEST = f"{ROOT}/compliance/table9_n100/manifest.json"
QSRC = f"{ROOT}/compliance/compliance_n10_report_n100_72b_plain.json"
OUT = f"{PSR_ROOT}/reports/analysis/_compliance_retrieval_colnomic.json"
MODEL_ID = "nomic-ai/colnomic-embed-multimodal-3b"
DEV = "cuda"
K = 5
Image.MAX_IMAGE_PIXELS = None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, (c - h) * 100), min(100.0, (c + h) * 100))


def maxsim(Q, P):
    """Late-interaction score: sum over query tokens of max over page patches."""
    return (Q @ P.T).max(dim=1).values.sum().item()


def main():
    t0 = time.time()
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    manifest = json.load(open(MANIFEST))
    cases = json.load(open(QSRC))
    cases = cases if isinstance(cases, list) else (cases.get("results") or cases.get("cases"))
    by_name = {d["name"]: d for d in manifest}
    arche = {d["name"]: d["archetype"] for d in manifest}

    print(f"[cmp] loading ColNomic-3B", flush=True)
    model = ColQwen2_5.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                       device_map=DEV).eval()
    proc = ColQwen2_5_Processor.from_pretrained(MODEL_ID)

    print(f"[cmp] loading cached {os.path.basename(CACHE)}", flush=True)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    paths, pembs = cache["paths"], cache["pembs"]
    print(f"[cmp] {len(paths)} real pages cached", flush=True)

    # ---- embed the 100 mockups with the SAME retriever -------------------
    print(f"[cmp] embedding {len(manifest)} CAD mockups", flush=True)
    mock_paths, mock_embs = [], []
    for i, d in enumerate(manifest, 1):
        img = Image.open(d["image_path"]).convert("RGB")
        img.thumbnail((1400, 1400))
        b = proc.process_images([img]).to(DEV)
        with torch.no_grad():
            e = model(**b)[0].to(torch.float16).cpu()
        mock_paths.append(d["image_path"])
        mock_embs.append(e)
        if i % 25 == 0:
            print(f"    {i}/{len(manifest)}", flush=True)
            torch.cuda.empty_cache()

    all_paths = list(paths) + mock_paths
    all_embs = list(pembs) + mock_embs
    mock_start = len(paths)
    name_of_index = {mock_start + i: manifest[i]["name"] for i in range(len(manifest))}
    print(f"[cmp] joint index = {len(all_paths)} pages "
          f"({len(paths)} real + {len(mock_embs)} mockups)", flush=True)

    # ---- score every compliance question --------------------------------
    hits, records = [], []
    for qi, c in enumerate(cases, 1):
        q = c.get("question") or c.get("query")
        gold_name = c.get("name")
        gold_idx = next((i for i, n in name_of_index.items() if n == gold_name), None)
        if gold_idx is None or not q:
            continue
        bq = proc.process_queries([q]).to(DEV)
        with torch.no_grad():
            Q = model(**bq)[0].float()

        scores = torch.empty(len(all_embs))
        for i, P in enumerate(all_embs):
            scores[i] = maxsim(Q, P.to(DEV).float())
        top = torch.topk(scores, K).indices.tolist()
        hit = gold_idx in top
        hits.append(hit)

        # what filled the top-5 when we missed
        kinds = []
        for t in top:
            if t == gold_idx:
                kinds.append("gold")
            elif t >= mock_start:
                other = name_of_index[t]
                kinds.append("same_archetype_mockup"
                             if arche.get(other) == arche.get(gold_name)
                             else "other_mockup")
            else:
                kinds.append("real_page")
        records.append({"name": gold_name, "question": q, "hit": hit,
                        "rank": (top.index(gold_idx) + 1) if hit else None,
                        "top5_kinds": kinds})
        if qi % 20 == 0:
            print(f"    [{qi}/{len(cases)}] hits={sum(hits)}", flush=True)
            torch.cuda.empty_cache()

    n = len(hits)
    k = sum(hits)
    lo, hi = wilson(k, n)
    miss = [r for r in records if not r["hit"]]
    filler = Counter(x for r in miss for x in r["top5_kinds"])

    out = {"note": "Compliance-set retrieval on ColNomic-3B (A29b / Cluster E).",
           "model_id": MODEL_ID, "k": K,
           "n_queries": n, "n_index_pages": len(all_paths),
           "n_real_pages": len(paths), "n_mockups": len(mock_embs),
           "recall@5": round(k / n * 100, 2) if n else None,
           "wilson95": [round(lo, 2), round(hi, 2)],
           "colpali_v12_baseline": 32.0,
           "miss_top5_composition": dict(filler),
           "elapsed_min": round((time.time() - t0) / 60, 1),
           "records": records}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    W = 74
    print(f"\n{'=' * W}\nCOMPLIANCE RETRIEVAL ON THE ADOPTED BACKBONE\n{'=' * W}")
    print(f"  index                 {len(all_paths)} pages "
          f"({len(paths)} real + {len(mock_embs)} mockups)")
    print(f"  ColNomic-3B  R@5      {k / n * 100:6.2f}%  ({k}/{n})  [{lo:.1f}, {hi:.1f}]")
    print(f"  ColPali-v1.2 R@5      {32.0:6.2f}%   (as previously reported)")
    print(f"  delta                 {k / n * 100 - 32.0:+6.2f} pp")
    if miss:
        tot = sum(filler.values())
        print(f"\n  on the {len(miss)} misses, the top-5 contained:")
        for kk, v in filler.most_common():
            print(f"    {kk:<26} {v:4d}  ({v / tot * 100:.0f}%)")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
