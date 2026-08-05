"""OCR + threshold-compare baseline on the 500-drawing single-doc compliance set.

Motivation (construct validity of the 100% verdict claim): the agentic VLM judge
reaches 100% on scale500 *when supplied the pre-resolved rule threshold*. This
baseline tests how much of that is genuine visual reasoning vs. simply reading a
printed dimension. We OCR each drawing, extract the governing dimension via a
per-archetype regex, and compare to the same pre-resolved threshold the VLM gets.
If this non-VLM pipeline also scores high, the synthetic task is largely
OCR-trivial; the gap (if any) localizes where the VLM adds value.

Two modes per drawing:
  A. label-guided: regex the value next to the known governing label.
  B. self-resolved threshold (rebar d/2 computed from OCR'd beam height).

CPU-only (tesseract). Run with OMP_THREAD_LIMIT=1.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, os, re, sys, time
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "$HOME/envs/wydot_copali/bin/tesseract"
ROOT = f"{PSR_ROOT}"
MANIFEST = f"{ROOT}/compliance/scale500/mockups/manifest.json"
OUT = f"{ROOT}/compliance/scale500/ocr_baseline_report.json"

NUM = r"(\d+(?:\.\d+)?)"


def ocr_at(img, ang):
    rot = img.rotate(ang, expand=True, fillcolor="white") if ang else img
    return pytesseract.image_to_string(rot, config="--oem 1 --psm 6")


def find(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def predict(archetype, text):
    """Return (pred_compliant, governing_value, threshold, found)."""
    t = text.replace("\n", " ")
    if archetype == "draw_culvert":
        v = find(rf"COVER\s*=?\s*{NUM}", t)
        return (v is not None and v >= 2.0), v, 2.0, v is not None
    if archetype == "draw_guardrail":
        # dimension label is "{depth}"  FOOTING DEPTH" (value BEFORE label);
        # avoid the note "MIN FOOTING DEPTH PER {agency} = 30"" (value AFTER).
        v = find(rf"{NUM}\s*[\"']?\s*FOOTING\s*DEPTH", t)
        return (v is not None and v >= 30.0), v, 30.0, v is not None
    if archetype == "draw_inlet":
        v = find(rf"GRATE\s*OPENING\s*=?\s*{NUM}", t)
        return (v is not None and v <= 4.0), v, 4.0, v is not None
    if archetype == "draw_sign_post":
        # dimension label "{embed}"  ANCHOR EMBED"; avoid note "MIN EMBEDMENT = 12"".
        v = find(rf"{NUM}\s*[\"']?\s*ANCHOR\s*EMBED", t)
        return (v is not None and v >= 12.0), v, 12.0, v is not None
    if archetype == "draw_rebar":
        stirrup = find(rf"STIRRUPS?\s*@\s*{NUM}", t)
        height = find(rf"{NUM}\s*[\"']?\s*HEIGHT", t)
        thr = (height // 2) if height else None
        ok = (stirrup is not None and thr is not None and stirrup <= thr)
        return ok, stirrup, thr, (stirrup is not None and height is not None)
    return None, None, None, False


def main():
    man = json.load(open(MANIFEST))
    print(f"[ocr] {len(man)} drawings")
    t0 = time.time()
    per_arch = {}
    n_ok = n_found = 0
    failures = []
    for i, m in enumerate(man):
        # adaptive multi-orientation OCR: many dimension labels are drawn rotated
        # 90deg (_dim_v) and unreadable at 0deg; only rotate when the value is not
        # yet found, keeping the baseline fast but giving OCR its fair best shot.
        img = Image.open(m["image_path"]).convert("RGB")
        pred, val, thr, found = (None, None, None, False)
        for ang in (0, 270, 90):  # try each orientation independently (no cross-contamination)
            pred, val, thr, found = predict(m["archetype"], ocr_at(img, ang))
            if found:
                break
        gt = m["ground_truth_compliant"]
        correct = (pred == gt)
        a = per_arch.setdefault(m["archetype"], {"correct": 0, "n": 0, "found": 0})
        a["correct"] += int(correct); a["n"] += 1; a["found"] += int(found)
        n_ok += int(correct); n_found += int(found)
        if not correct:
            failures.append({"name": m["name"], "archetype": m["archetype"], "gt_compliant": gt,
                             "pred_compliant": pred, "ocr_value": val, "threshold": thr,
                             "value_found": found, "violated_rule": m["violated_rule"]})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(man)}  acc={100*n_ok/(i+1):.1f}%  found={100*n_found/(i+1):.1f}%")
    acc = round(100.0 * n_ok / len(man), 2)
    rep = {
        "experiment": "OCR + threshold-compare baseline on scale500 single-doc compliance",
        "n": len(man),
        "ocr_verdict_accuracy": acc,
        "value_extraction_rate": round(100.0 * n_found / len(man), 2),
        "vlm_reference_accuracy": 100.0,
        "per_archetype": {k: {"accuracy": round(100.0 * v["correct"] / v["n"], 2),
                              "value_found_rate": round(100.0 * v["found"] / v["n"], 2),
                              "n": v["n"]} for k, v in sorted(per_arch.items())},
        "n_failures": len(failures),
        "failures": failures[:40],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ["ocr_verdict_accuracy", "value_extraction_rate", "per_archetype", "n_failures"]}, indent=2))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
