#!/usr/bin/env python3
"""Rebuild the page corpus from the public source PDFs.

The plan images are not redistributed (agency licensing differs). This script
reproduces them: give it a directory containing the source PDFs named as in
data/sources.json (download them from the URLs there), and it renders every page
with the same library, scale and file naming that produced the pages referenced
in data/page_index.json -- so a rendered corpus lines up with the benchmark
page identifiers one-to-one.

    python code/render_pages.py --pdf-dir /path/to/pdfs --out /path/to/pages

Then set PLANS_ROOT=/path/to/pages. Scale and naming per document come from
data/render_map.json (WYDOT at 200 DPI with unpadded page numbers; Caltrans at
400 DPI; AZ, CO, FL and MI at PyMuPDF Matrix(3,3), i.e. 216 DPI; the latter two
groups zero-padded).
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PSR_ROOT = os.environ.get("PSR_ROOT") or os.path.dirname(HERE)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def find_pdf(pdf_dir, name):
    for root, _, files in os.walk(pdf_dir):
        if name in files:
            return os.path.join(root, name)
        if root.count(os.sep) - pdf_dir.count(os.sep) >= 2:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True, help="directory holding the source PDFs (subdirectories allowed)")
    ap.add_argument("--out", default=os.path.join(PSR_ROOT, "data", "pages"), help="output directory for the PNG pages")
    ap.add_argument("--only", help="render only this document (filename as in sources.json)")
    ap.add_argument("--max-pages", type=int, help="render at most this many pages per document (testing)")
    ap.add_argument("--used-only", action="store_true", help="only documents that feed the reported index")
    ap.add_argument("--skip-sha", action="store_true", help="do not warn on sha256 mismatch (revised editions)")
    a = ap.parse_args()
    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("PyMuPDF is required: pip install pymupdf")

    rmap = json.load(open(os.path.join(PSR_ROOT, "data", "render_map.json")))["documents"]
    expect = {e["source_file"] for e in json.load(open(os.path.join(PSR_ROOT, "data", "page_index.json")))["pages"].values()}
    os.makedirs(a.out, exist_ok=True)
    produced, hit = 0, 0
    for d in rmap:
        if a.only and d["document"] != a.only:
            continue
        if a.used_only and not d["used_in_reported_index"]:
            continue
        pdf = find_pdf(a.pdf_dir, d["document"])
        if not pdf:
            print(f"[missing] {d['document']}  <- download from {d['url']}")
            continue
        if not a.skip_sha and d["sha256"] and sha256(pdf) != d["sha256"]:
            print(f"[warn] {d['document']}: sha256 differs from the manifest; the agency may have revised this edition")
        doc = fitz.open(pdf)
        n = len(doc)
        if d["pages"] and n != d["pages"]:
            print(f"[warn] {d['document']}: {n} pages, manifest says {d['pages']}")
        scale = 3.0 if d["dpi"] == 216 else d["dpi"] / 72.0
        mat = fitz.Matrix(scale, scale)
        w = d["page_number_width"]
        for i in range(n if not a.max_pages else min(n, a.max_pages)):
            num = f"{i+1:0{w}d}" if w else str(i + 1)
            name = f"{d['prefix']}_page_{num}.png"
            outp = os.path.join(a.out, name)
            if not os.path.exists(outp):
                doc.load_page(i).get_pixmap(matrix=mat).save(outp)
            produced += 1
            hit += name in expect
        print(f"[ok] {d['document']}: {n} pages -> {d['prefix']}_page_*.png @ {d['dpi']} DPI")
    print(f"\nrendered {produced} pages; {hit} of them are pages referenced by page_index.json ({len(expect)} total)")


if __name__ == "__main__":
    main()
