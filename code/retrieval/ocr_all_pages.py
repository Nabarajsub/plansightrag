"""OCR all 1,898 pages with pytesseract -- one-time shared cost for the
text-baseline strong baselines (BGE-M3, NV-Embed-v2).
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import os
# tesseract is OpenMP-multithreaded by default; with many worker processes that
# causes catastrophic core oversubscription. Pin each tesseract call to 1 thread
# so the ProcessPoolExecutor workers actually parallelize. Must be set before the
# tesseract subprocess is spawned (children inherit this env).
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
import json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import pytesseract

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages

OUT = f"{PSR_ROOT}/baselines_v2/ocr_text.json"
N_WORKERS = int(os.environ.get("OCR_WORKERS", "32"))


OCR_MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", "2600"))


def ocr_one(idx_path):
    idx, path = idx_path
    try:
        img = Image.open(path).convert("RGB")
        # 400-DPI plan sheets are enormous (>10k px); tesseract on them takes
        # minutes each. Downscale so OCR is tractable while staying legible.
        if max(img.size) > OCR_MAX_SIDE:
            s = OCR_MAX_SIDE / max(img.size)
            img = img.resize((int(img.size[0] * s), int(img.size[1] * s)), Image.LANCZOS)
        txt = pytesseract.image_to_string(img, config="--oem 1 --psm 6")
        return idx, " ".join(txt.split())
    except Exception as e:
        return idx, f"<OCR_ERROR: {type(e).__name__}>"


def _flush(pages, results):
    out_records = [{"image_path": p["image_path"], "agency": p["agency"],
                    "ocr_text": results[i]}
                   for i, p in enumerate(pages) if results[i] is not None]
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out_records, f)
    os.replace(tmp, OUT)
    return len(out_records)


def main():
    pages = load_pages()
    results = [None] * len(pages)

    # Resume: reuse any OCR text already written for these image paths.
    if os.path.exists(OUT):
        try:
            prev = {r["image_path"]: r["ocr_text"] for r in json.load(open(OUT))}
            for i, p in enumerate(pages):
                if p["image_path"] in prev and not str(prev[p["image_path"]]).startswith("<OCR_ERROR"):
                    results[i] = prev[p["image_path"]]
            print(f"[resume] reused {sum(r is not None for r in results)} cached OCR records")
        except Exception as e:
            print(f"[resume] could not read existing OUT: {e}")

    todo = [(i, pages[i]["image_path"]) for i in range(len(pages)) if results[i] is None]
    print(f"[ocr] {len(pages)} pages, {len(todo)} to do, {N_WORKERS} workers")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for fut in as_completed([pool.submit(ocr_one, a) for a in todo]):
            idx, txt = fut.result()
            results[idx] = txt
            done += 1
            if done % 200 == 0:
                n = _flush(pages, results)
                print(f"  {done}/{len(todo)}  ({n} saved)  elapsed {time.time()-t0:.0f}s")
    n = _flush(pages, results)
    avg_chars = sum(len(str(r)) for r in results if r is not None) / max(n, 1)
    print(f"[done] wrote {n} OCR records -> {OUT}")
    print(f"  avg chars/page: {avg_chars:.0f}, total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
