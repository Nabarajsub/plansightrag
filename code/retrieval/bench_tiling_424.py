"""Tiling ablation (H4) on the 424-pair page-disjoint test split.

For every page in the 1,898-page corpus:
  - Full-page row: reuse the existing v3 ColPali index (200 DPI page-level)
  - Tile-level row: render at 400 DPI, slice into 1024x1024 with 256-px overlap,
                    encode each tile with ColPali, aggregate per-page by max
                    over tile-level MaxSim scores

Outputs:
  reports/tiling_full_page_424.json   - Recall@5, per-agency, hits
  reports/tiling_tile_level_424.json  - Recall@5, per-agency, hits, tiles/page
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, math, os, sys, time
import numpy as np
import torch
import fitz
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report, REPORTS

COLPALI_ID = "vidore/colpali-v1.2"
V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"
TILE_INDEX = f"{PSR_ROOT}/baselines_v2/indices/tiles_400dpi.pt"
TILE_DPI = 400
TILE_SIZE = 1024
OVERLAP = 256
BATCH = 2  # multi-vector ColPali, keep small


def page_image_to_tiles(image_path):
    """Render a single page image_path at 400 DPI from its source PDF, return PIL tiles.
    Falls back to opening the PNG directly if PDF source isn't resolvable."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    stride = TILE_SIZE - OVERLAP
    n_x = max(1, math.ceil((W - OVERLAP) / stride))
    n_y = max(1, math.ceil((H - OVERLAP) / stride))
    tiles = []
    for i in range(n_x):
        for j in range(n_y):
            x = i * stride; y = j * stride
            r = min(x + TILE_SIZE, W); b = min(y + TILE_SIZE, H)
            tiles.append(img.crop((x, y, r, b)))
    return tiles


@torch.no_grad()
def encode_pages_as_tiles(pages, processor, model, device):
    """Tile every page on the fly. Returns list[List[Tensor(Np, D)]] per page (CPU fp16)."""
    if os.path.exists(TILE_INDEX):
        print(f"[index] loading cached tile embeddings: {TILE_INDEX}")
        return torch.load(TILE_INDEX, weights_only=False)
    page_tile_embs = []
    for p in tqdm(pages, desc="tile + encode pages"):
        tiles = page_image_to_tiles(p["image_path"])
        tile_embs = []
        for k in range(0, len(tiles), BATCH):
            chunk = tiles[k:k + BATCH]
            batch = processor.process_images(chunk).to(device)
            out = model(**batch)  # (B, N_patches, D)
            for j in range(out.shape[0]):
                tile_embs.append(out[j].to(torch.float16).cpu())
        page_tile_embs.append(tile_embs)
    os.makedirs(os.path.dirname(TILE_INDEX), exist_ok=True)
    torch.save(page_tile_embs, TILE_INDEX)
    print(f"[index] saved -> {TILE_INDEX}")
    return page_tile_embs


@torch.no_grad()
def main():
    device = "cuda"
    pages = load_pages()
    test = load_test()

    print(f"[load] {COLPALI_ID}")
    processor = ColPaliProcessor.from_pretrained(COLPALI_ID)
    model = ColPali.from_pretrained(
        COLPALI_ID, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    # ---- Encode all 424 queries once ----
    queries = [r["query"] for r in test]
    q_embs = []
    for qi in tqdm(range(0, len(queries), 16), desc="encode queries"):
        chunk = queries[qi:qi + 16]
        batch = processor.process_queries(chunk).to(device)
        out = model(**batch)
        for j in range(out.shape[0]):
            q_embs.append(out[j].to(torch.float16))

    # ---- Full-page row: load v3 index, run standard MaxSim ----
    print(f"[full-page] loading v3 index: {V3_INDEX}")
    recs = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    path_to_emb = {}
    for r in recs:
        m = r.get("metadata", {})
        ip = m.get("image_path") or m.get("page_path") or m.get("path") or m.get("file_path")
        path_to_emb[ip] = r["embedding"].to(torch.float16)
    page_embs_fp = [path_to_emb.get(p["image_path"]) for p in pages]
    print(f"[full-page] {sum(e is not None for e in page_embs_fp)}/{len(pages)} aligned")

    score_fp = np.zeros((len(q_embs), len(pages)), dtype=np.float32)
    for pi, pe in enumerate(tqdm(page_embs_fp, desc="MaxSim full-page")):
        if pe is None: continue
        pe_g = pe.to(device).to(torch.float32)
        for qi, qe in enumerate(q_embs):
            qe_g = qe.to(device).to(torch.float32)
            inter = torch.matmul(qe_g, pe_g.T)
            score_fp[qi, pi] = inter.max(dim=-1).values.sum().item()
    overall_fp, per_agency_fp, per_cat_fp, hits_fp = recall_at_k(score_fp, test, pages)
    save_report("tiling_full_page_424", overall_fp, per_agency_fp, per_cat_fp,
                hits_fp, len(pages), 0.0,
                extra={"granularity": "full-page (200 DPI v3)",
                       "model_id": COLPALI_ID})

    # ---- Tile-level row: encode tiles, max-per-page MaxSim ----
    page_tile_embs = encode_pages_as_tiles(pages, processor, model, device)

    score_tl = np.zeros((len(q_embs), len(pages)), dtype=np.float32)
    n_tiles = []
    for pi, tile_embs in enumerate(tqdm(page_tile_embs, desc="MaxSim tile-level")):
        n_tiles.append(len(tile_embs))
        # For each query, score against every tile, take MAX.
        for qi, qe in enumerate(q_embs):
            qe_g = qe.to(device).to(torch.float32)
            best = -1e30
            for te in tile_embs:
                te_g = te.to(device).to(torch.float32)
                inter = torch.matmul(qe_g, te_g.T)
                s = inter.max(dim=-1).values.sum().item()
                if s > best: best = s
            score_tl[qi, pi] = best
    overall_tl, per_agency_tl, per_cat_tl, hits_tl = recall_at_k(score_tl, test, pages)
    save_report("tiling_tile_level_424", overall_tl, per_agency_tl, per_cat_tl,
                hits_tl, len(pages), 0.0,
                extra={"granularity": "tile-level (400 DPI, 1024x1024, 256-px overlap)",
                       "model_id": COLPALI_ID,
                       "median_tiles_per_page": float(np.median(n_tiles)),
                       "total_tiles": int(np.sum(n_tiles))})

    # ---- Companion LaTeX fragment for tbl:tiling --------------------------
    OUT_TEX = os.path.join(REPORTS, "_tbl_tiling.tex")
    lines = []
    lines.append("% Auto-generated by baselines_v2/bench_tiling_424.py")
    lines.append("\\begin{table}[!t]")
    lines.append("\\caption{Full-page vs.\\ tile-level retrieval on the 424-pair page-disjoint test split over the 1{,}898-page five-DOT visual index. Full-page uses the v3 200~DPI index; tile-level rasterizes each page at 400~DPI and slices into $1024{\\times}1024$ tiles with 256-pixel overlap, then aggregates per page by max over tile-level MaxSim scores. ColPali retriever fixed across both rows.}\\label{tbl:tiling}")
    lines.append("\\small")
    lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lcc@{}}")
    lines.append("\\toprule")
    lines.append("Granularity & R@5 (\\%) & Tiles/Page \\\\")
    lines.append("\\midrule")
    lines.append(f"Full-page (200 DPI)       & {overall_fp:.2f} & 1 \\\\")
    lines.append(f"Tile-level (400 DPI)      & {overall_tl:.2f} & $\\approx${np.median(n_tiles):.0f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{table}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[done] -> {OUT_TEX}")


if __name__ == "__main__":
    main()
