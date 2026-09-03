"""Per-baseline implementation settings table.

Describes every baseline implementation in the eight fields needed to reproduce
it: OCR engine and settings, image resolution, chunking, pooling, metadata use,
query formulation, hyperparameter selection, and whether the method was applied
in its intended task setting. This compiles them, preferring
what the report JSON recorded at run time and falling back to a curated mapping
read off the generating scripts. Every cell is sourced; nothing is inferred from
the method name.

    python baseline_settings.py
"""

from __future__ import annotations

# --- release path resolution (release copy; the run-time original under
# baselines_v2/ is unchanged) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---


import json
import os
import glob

REPORTS = _os.path.join(PSR_ROOT, "reports", "analysis")

# Curated from the generating scripts. Field order follows R2's list.
# "intended?" = was the method applied in the task setting its authors designed
# it for; where it was not, that is stated in the paper rather than left implicit.
CURATED = {
 "colnomic_3b":      ("—", "dynamic (native)", "none — full page", "MaxSim late interaction", "no", "raw question", "none", "yes"),
 "colnomic_7b":      ("—", "dynamic (native)", "none — full page", "MaxSim late interaction", "no", "raw question", "none", "yes"),
 "colpali_v13":      ("—", "448x448 fixed", "none — full page", "MaxSim late interaction", "no", "raw question", "none", "yes"),
 "colqwen25_v0_2":   ("—", "dynamic (native)", "none — full page", "MaxSim late interaction", "no", "raw question", "none", "yes"),
 "nemotron_colembed_4b": ("—", "dynamic (native)", "none — full page", "MaxSim late interaction", "no", "raw question", "none", "yes"),
 "nemotron_colembed_8b": ("—", "dynamic (native)", "none — full page", "MaxSim late interaction", "no", "raw question", "none", "yes"),
 "visrag_ret":       ("—", "dynamic (native)", "none — full page", "single-vector dense", "no", "raw question", "none", "yes"),
 "dse_qwen2_2b":     ("—", "dynamic (native)", "none — full page", "single-vector dense", "no", "raw question", "none", "yes"),
 "ocr_bm25":         ("Tesseract 5, default psm", "200 DPI PNG", "whole page, one doc", "lexical (no pooling)", "no", "raw question", "k1=1.5, b=0.75 (Okapi defaults)", "yes"),
 "ocr_bm25_meta":    ("Tesseract 5, default psm", "200 DPI PNG", "whole page, one doc", "lexical (no pooling)", "YES — agency, plan_id, sheet_title appended", "raw question", "k1=1.5, b=0.75 (Okapi defaults)", "yes"),
 "bge_m3_ocr":       ("Tesseract 5, default psm", "200 DPI PNG", "none — full page, 8192-tok limit", "CLS dense", "no", "raw question", "defaults", "yes"),
 "bge_m3_ocr_tuned": ("Tesseract 5, default psm", "200 DPI PNG", "512-token windows, 128 overlap", "max over chunks + sparse fusion", "no", "raw question", "chunk/overlap swept on dev", "yes"),
 "ocr_minilm":       ("Tesseract 5, default psm", "200 DPI PNG", "none — TRUNCATES at 256 tokens", "mean", "no", "raw question", "defaults", "yes — but truncation defeats it"),
 "ocr_minilm_chunked": ("Tesseract 5, default psm", "200 DPI PNG", "256-token windows", "max over chunks", "no", "raw question", "defaults", "yes"),
 "clip_vit_b32":     ("—", "224x224 (whole page downsampled)", "none", "image CLS", "no", "raw question", "defaults", "NO — page-level retrieval is out of CLIP's caption-matching setting"),
 "clip_l14_336_tiled": ("—", "336x336 tiles", "3x3 tiling", "max over tiles", "no", "raw question", "grid swept on dev", "NO — same caveat, tiling mitigates"),
 "nougat_decode_minilm": ("Nougat-base decode (not OCR)", "896x672 native", "none", "MiniLM mean over decoded text", "no", "raw question", "defaults", "NO — Nougat targets academic PDFs"),
 "pix2struct_decode_minilm": ("Pix2Struct decode", "1024 patches", "none", "MiniLM mean over decoded text", "no", "raw question", "defaults", "NO — screenshot/VQA setting, not retrieval"),
 "layoutlmv3":       ("Tesseract 5 (for layout boxes)", "224x224", "none", "CLS / mean (both reported)", "no", "raw question", "defaults", "NO — token classification model, not a retriever"),
 "udop_encoder_meanpool": ("Tesseract 5 (for layout boxes)", "224x224", "none", "encoder mean over min(mask,h) tokens", "no", "raw question", "defaults", "NO — seq2seq doc model, not a retriever"),
 "visionrag_pyramid": ("Tesseract 5", "200 DPI, 3-level pyramid", "per-level, RRF fused", "MiniLM mean — TRUNCATES at 256", "no", "raw question", "RRF k=60", "yes — but truncation defeats it"),
 "visionrag_pyramid_fixed": ("Tesseract 5", "200 DPI, 3-level pyramid", "256-token windows per level, RRF fused", "max over chunks", "no", "raw question", "RRF k=60", "yes"),
}
COLS = ("OCR engine / settings", "image resolution", "chunking", "pooling",
        "metadata used", "query formulation", "hyperparameters", "intended task setting?")


def main():
    rows = {}
    for f in sorted(glob.glob(os.path.join(REPORTS, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "recall@5_overall" not in d:
            continue
        tag = d.get("tag") or os.path.basename(f)[:-5]
        rows[tag] = {"recall@5": round(d["recall@5_overall"], 2),
                     "model_id": d.get("model_id") or d.get("retriever") or "—",
                     "runtime_recorded": {k: v for k, v in d.items()
                                          if k in ("retriever", "text_source", "tokeniser",
                                                   "indexes_metadata", "embedder", "n_chunks",
                                                   "pages_with_empty_text", "note")}}
    out, missing = {}, []
    for tag, r in rows.items():
        cur = CURATED.get(tag)
        if cur:
            out[tag] = dict(zip(COLS, cur), **{"recall@5": r["recall@5"],
                                               "model_id": r["model_id"],
                                               "source": "script + report"})
        else:
            missing.append(tag)

    W = 150
    print("=" * W)
    print("PER-BASELINE IMPLEMENTATION SETTINGS")
    print("=" * W)
    print(f"\n  {'baseline':<26}{'R@5':>7}  {'OCR engine':<30}{'resolution':<26}"
          f"{'chunking':<32}{'intended?':<10}")
    print("  " + "-" * (W - 4))
    for tag in sorted(out, key=lambda t: -out[t]["recall@5"]):
        v = out[tag]
        intended = "yes" if v[COLS[7]].startswith("yes") else "NO"
        print(f"  {tag:<26}{v['recall@5']:>6.2f}%  {v[COLS[0]][:29]:<30}"
              f"{v[COLS[1]][:25]:<26}{v[COLS[2]][:31]:<32}{intended:<10}")
    print("\n  " + "-" * (W - 4))
    n_out = sum(1 for v in out.values() if not v[COLS[7]].startswith("yes"))
    print(f"  {len(out)} baselines documented across all 8 fields · "
          f"{n_out} applied OUTSIDE their intended task setting (stated in the paper)")
    if missing:
        print(f"\n  {len(missing)} report(s) with no curated entry "
              f"(variants of a documented baseline): {', '.join(sorted(missing)[:12])}"
              f"{' ...' if len(missing) > 12 else ''}")

    dst = os.path.join(REPORTS, "_baseline_settings.json")
    json.dump({"note": __doc__.strip().split("\n")[0], "fields": list(COLS),
               "baselines": out, "uncurated_variants": sorted(missing)},
              open(dst, "w"), indent=1)
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
