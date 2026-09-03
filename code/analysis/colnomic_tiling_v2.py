"""Tiling aggregations designed for a late-interaction retriever.

The published tiling study applied ColPali's recipe unchanged: crop the page into
1024 px tiles, score each tile independently, and take the MAX over tile scores. On
ColNomic-3B that loses 8.73 pp. Two reasons, both fixable:

1. MAX-OVER-TILE-SCORES IS THE WRONG AGGREGATION FOR LATE INTERACTION.
   MaxSim already lets each query token choose its own best patch. Scoring tiles
   separately and taking the best tile forces every query token to be satisfied by
   ONE tile, so a question needing a note in one corner and a dimension in another
   can never be fully matched. The correct move is to pool every tile's patches into
   a single bag and run MaxSim once - then each query token picks its best patch
   from anywhere on the sheet, which is what cross-view engineering queries need.

2. THE TILES CARRY NO EXTRA RESOLUTION.
   Measured: indexed pages are 3400x4400, which IS the 200 DPI render. The tiler
   crops that PNG rather than re-rendering at 400 DPI, so a tile is the same pixels.
   For ColPali (fixed 448x448 encoder) cropping still helps, because a 1024 px crop
   downsampled to 448 keeps more detail per unit area. For a dynamic-resolution
   backbone there is nothing to gain and context to lose.

So this evaluates aggregations rather than re-running the same recipe. One encoding
pass per page, every variant scored from the same embeddings:

    published_max      max over per-tile MaxSim scores        (reproduces -8.73 pp)
    patch_union        all tile patches pooled, ONE MaxSim     (the principled fix)
    page_plus_tiles    full-page patches UNION tile patches    (context + detail)
    top3_sum           sum of the three best tile scores       (multi-region evidence)
    max_page_tile      max(full-page score, best-tile score)   (tiling can only help)
    blend              0.7*full-page + 0.3*best-tile           (soft fusion)

Same 424-query split, same 1,898-page corpus, same tile geometry as published, so
every row is comparable to the manuscript.

    python colnomic_tiling_v2.py [--grid coarse]
"""

from __future__ import annotations

# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

import argparse, json, math, os, sys, time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _eval_lib import load_pages, load_test, recall_at_k, save_report, REPORTS

Image.MAX_IMAGE_PIXELS = None
ROOT = CLUSTER_ROOT
COLNOMIC = "nomic-ai/colnomic-embed-multimodal-3b"
CACHE = f"{ROOT}/baselines_v2/cache_colnomic_pages.pt"
DEV = "cuda"


def tiles_published(img, size=1024, overlap=256):
    """Exactly the geometry used in the manuscript's tiling study."""
    W, H = img.size
    stride = size - overlap
    out = []
    for i in range(max(1, math.ceil((W - overlap) / stride))):
        for j in range(max(1, math.ceil((H - overlap) / stride))):
            x, y = i * stride, j * stride
            out.append(img.crop((x, y, min(x + size, W), min(y + size, H))))
    return out


def tiles_coarse(img, grid=2, overlap=0.18):
    """Few large tiles: keeps far more context per crop than 30 small ones."""
    W, H = img.size
    tw, th = W / grid, H / grid
    ox, oy = tw * overlap, th * overlap
    out = []
    for i in range(grid):
        for j in range(grid):
            l, u = max(0, int(i * tw - ox)), max(0, int(j * th - oy))
            r, b = min(W, int((i + 1) * tw + ox)), min(H, int((j + 1) * th + oy))
            out.append(img.crop((l, u, r, b)))
    return out


def maxsim(q, P):
    return torch.matmul(q, P.T).max(-1).values.sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="published", choices=["published", "coarse"])
    a = ap.parse_args()
    t0 = time.time()

    pages, test = load_pages(), load_test()
    qs = [r["query"] for r in test]
    nq, npg = len(qs), len(pages)

    model = ColQwen2_5.from_pretrained(COLNOMIC, torch_dtype=torch.bfloat16,
                                       device_map=DEV).eval()
    proc = ColQwen2_5_Processor.from_pretrained(COLNOMIC)
    Q = []
    for s in range(0, nq, 16):
        with torch.no_grad():
            e = model(**proc.process_queries(qs[s:s + 16]).to(DEV))
        Q += [e[j].float() for j in range(e.shape[0])]

    cache = torch.load(CACHE, map_location="cpu")
    pembs = cache["pembs"]
    tiler = tiles_published if a.grid == "published" else tiles_coarse
    print(f"[v2] {npg} pages, {nq} queries, grid={a.grid}", flush=True)

    VARIANTS = ["published_max", "patch_union", "page_plus_tiles",
                "top3_sum", "max_page_tile", "blend"]
    KEYS = VARIANTS + ["full_page"]
    S = {v: np.zeros((nq, npg), dtype=np.float32) for v in KEYS}
    ntiles, start = 0, 0

    # --- resume support: this partition preempts and the run is multi-hour ---
    CKPT = f"{REPORTS}/_tiling_v2_{a.grid}.ckpt.npz"
    if os.path.exists(CKPT):
        try:
            z = np.load(CKPT)
            start = int(z["done"]); ntiles = int(z["ntiles"])
            for v in KEYS:
                S[v] = z[v]
            print(f"[resume] {start}/{npg} pages already scored", flush=True)
        except Exception as e:
            print(f"[resume] checkpoint unreadable ({e}); starting fresh", flush=True)
            start = 0

    for pi, p in enumerate(tqdm(pages, desc="score")):
        if pi < start:
            continue
        Pfull = pembs[pi].to(DEV).float()
        img = Image.open(p["image_path"]).convert("RGB")
        ts = tiler(img)
        ntiles += len(ts)

        tile_embs = []
        for s in range(0, len(ts), 2):
            with torch.no_grad():
                e = model(**proc.process_images(ts[s:s + 2]).to(DEV))
            tile_embs += [e[j].float() for j in range(e.shape[0])]

        # one bag of patches from every tile; and that bag plus the full page
        Punion = torch.cat(tile_embs, dim=0)
        Pboth = torch.cat([Pfull, Punion], dim=0)

        for qi, q in enumerate(Q):
            per_tile = [maxsim(q, T) for T in tile_embs]
            full = maxsim(q, Pfull)
            best = max(per_tile)
            S["full_page"][qi, pi] = full
            S["published_max"][qi, pi] = best
            S["patch_union"][qi, pi] = maxsim(q, Punion)
            S["page_plus_tiles"][qi, pi] = maxsim(q, Pboth)
            S["top3_sum"][qi, pi] = sum(sorted(per_tile, reverse=True)[:3])
            S["max_page_tile"][qi, pi] = max(full, best)
            S["blend"][qi, pi] = 0.7 * full + 0.3 * best

        del Pfull, Punion, Pboth, tile_embs
        if pi % 200 == 0:
            torch.cuda.empty_cache()
        if pi % 100 == 0 or pi == npg - 1:
            np.savez_compressed(CKPT, done=pi + 1, ntiles=ntiles, **S)

    base, _, _, _ = recall_at_k(S["full_page"], test, pages)
    print(f"\n  full-page baseline           R@5 = {base:.2f}%", flush=True)
    results = {"full_page": round(base, 2)}
    for v in VARIANTS:
        ov, pa, pc, hits = recall_at_k(S[v], test, pages)
        results[v] = round(ov, 2)
        save_report(f"colnomic_tiling_v2_{v}_{a.grid}", ov, pa, pc, hits, npg, time.time() - t0,
                    extra={"model_id": COLNOMIC, "aggregation": v, "grid": a.grid,
                           "tiles_per_page": round(ntiles / npg, 1),
                           "full_page_baseline": round(base, 2),
                           "delta_vs_full_page_pp": round(ov - base, 2),
                           "note": "Tiles are crops of the 200 DPI render (3400x4400); "
                                   "they add no resolution. This tests AGGREGATION, not DPI."})
        print(f"  {v:<28} R@5 = {ov:5.2f}%   ({ov-base:+.2f} pp vs full page)", flush=True)

    if os.path.exists(CKPT):
        os.remove(CKPT)
    json.dump({"grid": a.grid, "tiles_per_page": round(ntiles / npg, 1),
               "results": results, "elapsed_min": round((time.time() - t0) / 60, 1)},
              open(f"{REPORTS}/_colnomic_tiling_v2_{a.grid}.json", "w"), indent=1)
    print(f"\n[v2] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
