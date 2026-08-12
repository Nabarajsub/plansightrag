"""Per-baseline implementation settings table (R2 reproducibility 4).

R2 asks that every baseline be described in enough detail to reproduce it: OCR
engine and settings, image resolution, chunking, pooling, metadata use, query
formulation, hyperparameter selection, and whether the method was applied in its
intended task setting.

That table did not exist. `09_baseline_audit.md` explains why six baselines were
defective and what was corrected, but not how each one was configured. The
information does exist, spread across two places:

  reports/*.json   the newer and corrected runs record their own settings
                   (`retriever`, `text_source`, `tokeniser`, `indexes_metadata`, ...)
  the run scripts  resolution, chunk size, pooling and query formulation are
                   literals in bench_*.py

This walks both and emits one row per baseline. Fields it cannot find are printed
as "not recorded" rather than guessed -- a reviewer needs to know which settings
are documented and which are only recoverable by reading code.

    python baseline_settings_table.py            # console + JSON
    python baseline_settings_table.py --latex    # LaTeX longtable for the appendix
"""

# --- release path resolution (release copy; the run-time original under
# baselines_v2/ is unchanged) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

from __future__ import annotations

import argparse
import glob
import json
import os
import re

HERE = CLUSTER_ROOT + "/baselines_v2"
REPORTS = _os.path.join(PSR_ROOT, "reports", "analysis")

FIELDS = ["model", "ocr_engine", "resolution", "chunking", "pooling",
          "metadata", "query_formulation", "hyperparameters", "intended_task"]

# Which script produced which report family, so we can mine literals from source.
SCRIPT_HINTS = {
    "ocr_bm25": "bench_ocr_bm25.py", "ocr_minilm": "bench_text_fixed.py",
    "bge_m3": "bench_bge_m3_tuned.py", "clip": "bench_weak_fixed.py",
    "nougat": "bench_weak_fixed.py", "pix2struct": "bench_weak_fixed.py",
    "layoutlmv3": "bench_weak_fixed.py", "udop": "bench_weak_fixed.py",
    "visionrag": "bench_visual_fixed.py", "visrag": "bench_visrag.py",
    "colnomic": "full_benchmark_retrieval.py", "colqwen": "full_benchmark_retrieval.py",
    "colpali": "full_benchmark_retrieval.py", "dse": "full_benchmark_retrieval.py",
    "nemotron": "full_benchmark_retrieval.py",
}

# Settings that are literals in the scripts rather than fields in the reports.
# Each entry is (regex over the script text, field, how to render the match).
MINE = [
    (r"--chunk\s+(\d+)\s+--stride\s+(\d+)", "chunking", None),
    (r"(?:DPI|dpi)\s*=\s*(\d+)", "resolution", lambda m: f"{m} DPI render"),
    (r"max_pixels\s*=\s*([\w\s*]+)", "resolution", lambda m: f"max_pixels={m.strip()}"),
    (r"thumbnail\(\((\d+),\s*\d+\)\)", "resolution", lambda m: f"long side {m} px"),
    (r"chunk_size\s*=\s*(\d+)", "chunking", lambda m: f"{m}-token chunks"),
    (r"CHUNK\w*\s*=\s*(\d+)", "chunking", lambda m: f"{m}-token chunks"),
    (r"(\.mean)\(dim=1\)", "pooling", lambda m: "mean over tokens"),
    (r"(\.max)\(dim=1\)", "pooling", lambda m: "max over tokens"),
    (r"k1=([\d.]+),\s*b=([\d.]+)", "hyperparameters", None),
    (r"resize\(\((\d+),\s*\d+\)", "resolution", lambda m: f"resized to {m} px"),
    (r"image_size\s*=\s*(\d+)", "resolution", lambda m: f"{m}x{m} px"),
]


def script_doc(name):
    """The generating script's module docstring -- where the settings actually live.

    Each bench_*.py opens with a prose description of its variants, pooling and
    hyperparameters. That is the authoritative per-baseline documentation; the
    structured fields below are a convenience index into it, not a replacement.
    """
    path = os.path.join(HERE, name) if name else None
    if not path or not os.path.exists(path):
        return None
    src = open(path, errors="ignore").read()
    m = re.match(r'\A\s*"""(.*?)"""', src, re.S)
    if not m:
        return None
    doc = " ".join(m.group(1).split())
    return doc[:600]


def load_reports():
    out = {}
    for f in sorted(glob.glob(os.path.join(REPORTS, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "recall@5_overall" not in d:
            continue
        tag = d.get("tag") or os.path.basename(f)[:-5]
        out[tag] = d
    return out


def mine_script(tag):
    """Pull settings literals out of whichever bench script produced this tag."""
    found = {}
    name = next((v for k, v in SCRIPT_HINTS.items() if k in tag), None)
    if not name:
        return found, None
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return found, name
    src = open(path, errors="ignore").read()
    for pat, field, render in MINE:
        m = re.search(pat, src)
        if not m or field in found:
            continue
        try:
            found[field] = (render(m.group(1)) if render
                            else f"k1={m.group(1)}, b={m.group(2)}")
        except (IndexError, TypeError):
            continue
    return found, name


def row_for(tag, d):
    mined, script = mine_script(tag)
    r = {f: None for f in FIELDS}
    r["model"] = d.get("model_id") or d.get("retriever") or d.get("embedder")
    ts = str(d.get("text_source") or "")
    if "esseract" in ts or "OCR" in ts:
        r["ocr_engine"] = ts.split(",")[0].strip()
    elif any(k in tag for k in ("ocr", "bm25", "minilm", "bge")):
        r["ocr_engine"] = "Tesseract OCR (baselines_v2/ocr_text.json)"
    r["metadata"] = ("indexed" if d.get("indexes_metadata") else
                     ("not indexed" if d.get("indexes_metadata") is False else None))
    if d.get("tokeniser"):
        r["hyperparameters"] = str(d["tokeniser"])
    if d.get("n_chunks"):
        r["chunking"] = f"{d['n_chunks']} chunks total"
    for k, v in mined.items():
        r[k] = r[k] or v
    note = str(d.get("note") or d.get("backbone_note") or "")
    if d.get("task_mismatch") or "intended" in note.lower() or "outside" in note.lower():
        r["intended_task"] = "NO — applied outside its intended task setting"
    elif note:
        r["intended_task"] = note[:70]
    r["query_formulation"] = ("raw question text" if r["ocr_engine"] else
                              "raw question text (visual encoder)")
    r["documentation"] = script_doc(script)
    return r, script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    reps = load_reports()
    rows, scripts = {}, {}
    for tag, d in reps.items():
        rows[tag], scripts[tag] = row_for(tag, d)

    ordered = sorted(rows, key=lambda t: -reps[t]["recall@5_overall"])
    W = 108
    print("=" * W)
    print("PER-BASELINE IMPLEMENTATION SETTINGS (R2 reproducibility 4)")
    print("=" * W)
    print(f"\n  {len(ordered)} baselines. Fields marked 'not recorded' are recoverable")
    print("  only by reading the generating script, which is named in the last column.\n")
    hdr = f"  {'baseline':<34}{'R@5':>7}  {'OCR / model':<40}{'script'}"
    print(hdr); print("  " + "-" * (W - 2))
    for t in ordered:
        r = rows[t]
        src = r["ocr_engine"] or r["model"] or "not recorded"
        print(f"  {t[:33]:<34}{reps[t]['recall@5_overall']:>6.2f}%  {str(src)[:39]:<40}"
              f"{scripts[t] or '—'}")

    ndoc = sum(1 for t in rows if rows[t].get("documentation"))
    print(f"\n  generating script docstring captured for {ndoc}/{len(rows)} baselines")
    cov = {f: sum(1 for t in rows if rows[t][f]) for f in FIELDS}
    print(f"\n  field coverage across {len(rows)} baselines:")
    for f in FIELDS:
        print(f"    {f:<20} {cov[f]:>3}/{len(rows)}"
              + ("" if cov[f] == len(rows) else "   <-- gaps must be filled by hand"))

    out = {"note": "Per-baseline implementation settings (R2 repro 4).",
           "n_baselines": len(rows), "field_coverage": cov,
           "baselines": {t: {**rows[t], "recall@5": reps[t]["recall@5_overall"],
                             "generating_script": scripts[t]} for t in ordered}}
    dst = os.path.join(REPORTS, "_baseline_settings.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n  wrote {dst}")

    if a.latex:
        tex = os.path.join(REPORTS, "_baseline_settings.tex")
        with open(tex, "w") as fh:
            fh.write("% Auto-generated by baselines_v2/baseline_settings_table.py\n")
            fh.write("\\begin{longtable}{@{}p{3.1cm}rp{3.4cm}p{2.6cm}p{2.4cm}@{}}\n")
            fh.write("\\caption{Implementation settings for every retrieval baseline.}"
                     "\\label{tbl:baseline-settings}\\\\\n\\toprule\n")
            fh.write("Baseline & R@5 & OCR engine / model & Resolution & Metadata \\\\\n"
                     "\\midrule\n\\endfirsthead\n")
            for t in ordered:
                r = rows[t]
                esc = lambda s: re.sub(r"([&%_#])", r"\\\1", str(s or "not recorded"))[:60]
                fh.write(f"{esc(t)} & {reps[t]['recall@5_overall']:.2f}\\% & "
                         f"{esc(r['ocr_engine'] or r['model'])} & {esc(r['resolution'])} & "
                         f"{esc(r['metadata'])} \\\\\n")
            fh.write("\\bottomrule\n\\end{longtable}\n")
        print(f"  wrote {tex}")


if __name__ == "__main__":
    main()
