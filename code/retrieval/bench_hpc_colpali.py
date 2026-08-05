"""Binary-quantized ColPali (HPC-ColPali) on the 424-pair / 1,898-page eval.

Re-measures the binary-quantization variant under the same protocol as every
other baseline in `baselines_v2/`. Reports Recall@5, compression ratio,
and median retrieval latency.

Quantization: each 128-dim ColPali patch embedding is binarized by sign
(+1 if >0 else 0) and packed 8 bits / byte -> 16 bytes per patch
(vs. 64 bytes at fp16, 256 at fp32). At score time we unpack to {-1, +1}
floats and run the same MaxSim late-interaction as full-precision ColPali.
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
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report, REPORTS

COLPALI_ID = "vidore/colpali-v1.2"
V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"
TAG = "hpc_colpali_bq"


def pack_bits(emb: torch.Tensor) -> np.ndarray:
    """emb: (N_patches, D=128) float -> (N_patches, D/8) uint8."""
    mask = (emb > 0).cpu().numpy()
    return np.packbits(mask, axis=-1)


def unpack_bits(packed: np.ndarray, dim: int) -> np.ndarray:
    """(N, D/8) uint8 -> (N, D) {-1, +1} float32."""
    bits = np.unpackbits(packed, axis=-1)[:, :dim].astype(np.float32)
    return bits * 2.0 - 1.0


@torch.no_grad()
def main():
    device = "cuda"
    pages = load_pages()
    test = load_test()

    # ---- Load full-precision v3 ColPali index, then binary-pack ----
    print(f"[load] {V3_INDEX}")
    recs = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    print(f"[load] {len(recs)} page embeddings")

    # Build path -> packed-embedding map so we can order against pages.json
    path_to_packed = {}
    fp_bytes = 0
    bq_bytes = 0
    D = 128
    for r in tqdm(recs, desc="binary quantize"):
        emb = r["embedding"].float()  # (N_patches, 128)
        fp_bytes += emb.numel() * 2  # fp16 reference
        packed = pack_bits(emb)
        bq_bytes += packed.size  # uint8
        # Use the page image path stored in metadata.
        meta = r.get("metadata", {})
        ipath = meta.get("image_path") or meta.get("page_path") or meta.get("path")
        if ipath is None:
            # Reconstruct: many v3 records have plan_id / page_number metadata
            ipath = meta.get("file_path")
        path_to_packed[ipath] = packed

    # Order the index to match pages.json
    page_packed = []
    missing = 0
    for p in pages:
        pk = path_to_packed.get(p["image_path"])
        if pk is None:
            missing += 1
            page_packed.append(None)
        else:
            page_packed.append(pk)
    print(f"[index] {len(page_packed)} pages aligned ({missing} missing)")

    fp_mb = fp_bytes / (1024 * 1024)
    bq_mb = bq_bytes / (1024 * 1024)
    compression = fp_mb / bq_mb if bq_mb else 0.0
    print(f"[size] FP16 reference {fp_mb:.2f} MB -> BQ {bq_mb:.2f} MB ({compression:.1f}x)")

    # ---- Encode queries with the full-precision ColPali model -----------
    print(f"[load] {COLPALI_ID}")
    processor = ColPaliProcessor.from_pretrained(COLPALI_ID)
    model = ColPali.from_pretrained(
        COLPALI_ID, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    queries = [r["query"] for r in test]
    q_embs = []
    for qi in tqdm(range(0, len(queries), 16), desc="encode queries"):
        chunk = queries[qi:qi + 16]
        batch = processor.process_queries(chunk).to(device)
        out = model(**batch)  # (B, Lq, D)
        for j in range(out.shape[0]):
            q_embs.append(out[j].to(torch.float32).cpu().numpy())

    # ---- Score: MaxSim against unpacked {-1, +1} page embeddings --------
    score_matrix = np.zeros((len(q_embs), len(page_packed)), dtype=np.float32)
    timings = []
    for pi, packed in enumerate(tqdm(page_packed, desc="score (BQ MaxSim)")):
        if packed is None:
            continue
        pe = unpack_bits(packed, D)  # (Np, D)
        for qi, qe in enumerate(q_embs):
            t0 = time.perf_counter()
            inter = qe @ pe.T  # (Lq, Np)
            s = inter.max(axis=-1).sum()
            score_matrix[qi, pi] = s
            timings.append((time.perf_counter() - t0) * 1000)

    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    t = np.asarray(timings)
    save_report(
        TAG, overall, per_agency, per_cat, hits, len(pages),
        elapsed=float(t.sum() / 1000),
        extra={
            "model_id": COLPALI_ID + " (binary quantized)",
            "compression_ratio": float(compression),
            "fp16_mb": float(fp_mb),
            "bq_mb": float(bq_mb),
            "p50_ms_per_page_score": float(np.percentile(t, 50)),
            "p95_ms_per_page_score": float(np.percentile(t, 95)),
            "missing_pages": missing,
        },
    )


if __name__ == "__main__":
    main()
