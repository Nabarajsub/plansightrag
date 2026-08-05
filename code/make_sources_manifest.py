"""Record exactly which source documents the corpus was built from.

Reviewer #2 (major 9, repro 1 and 7) asks for page identifiers and for the plan
images to be referenced rather than redistributed. Referencing only works if the
reader can confirm they hold the *same* document: DOT standard plans are revised,
so "download it from the agency website" is not by itself reproducible.

This writes data/sources.json with, per source PDF: agency, document title,
edition, page count, byte size, SHA-256, and the publisher URL. The checksum is
what makes the reference verifiable.

    python code/make_sources_manifest.py --corpus /path/to/corpus/root

`url` is left empty where it is not already recorded — fill those in before
publishing; do not guess them.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "sources.json")

# Which directory belongs to which agency, and how it is used in the paper.
CORPUS = {
    "wydot_2026":         ("WYDOT", "Wyoming DOT Standard Plans", "retrieval index"),
    "california_2025":    ("Caltrans", "California DOT Standard Plans", "retrieval index"),
    "arizona_2025":       ("AZDOT", "Arizona DOT Standard Drawings", "retrieval index"),
    "colorado_2025":      ("CDOT", "Colorado DOT Standard Bridge Drawings", "retrieval index"),
    "florida_2026":       ("FDOT", "Florida DOT Design Standards", "retrieval index"),
    "michigan_2025":      ("MDOT", "Michigan DOT Standard Plans", "held-out zero-shot transfer"),
    "wydot_design_manual": ("WYDOT", "Wyoming DOT Road Design Manual",
                            "real-corpus rule grounding (Section 5.9)"),
}
# Documents that sit at the corpus root rather than in an agency directory.
ROOT_DOCS = {
    "Wyoming 2021 Standard Specifications for Road and Bridge Construction":
        ("WYDOT", "Wyoming Standard Specifications for Road and Bridge Construction, 2021",
         "real-corpus rule grounding (Section 5.9)"),
    "Michian standard book":
        ("MDOT", "Michigan DOT Standard Plans", "held-out zero-shot transfer"),
}

# Known publisher landing pages. Leave blank rather than guessing.
URLS = {
    "WYDOT": "", "Caltrans": "", "AZDOT": "", "CDOT": "", "FDOT": "", "MDOT": "",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pdf_pages(path):
    """Page count without a PDF library: count /Type /Page objects."""
    try:
        import fitz  # PyMuPDF, already a dependency
        with fitz.open(path) as d:
            return d.page_count
    except Exception:
        return None


def attribute_pages():
    """page_index.json -> {pdf stem or collection key: number of benchmark pages}.

    Rendered page filenames encode their origin, so each benchmark page can be
    traced to the exact PDF it came from.
    """
    pi_path = os.path.join(ROOT, "data", "page_index.json")
    if not os.path.exists(pi_path):
        return {}
    pages = json.load(open(pi_path)).get("pages", {})
    hits = {}
    for v in pages.values():
        f = v.get("source_file", "")
        key = None
        m = re.match(r"Caltrans_california drawings_(.+?)_page_\d+\.png$", f)
        if m:
            key = m.group(1)                                  # PDF stem
        elif f.startswith("Wyoming_DOT_"):
            key = re.sub(r"_page_\d+\.png$", "", f).replace("Wyoming_DOT_", "", 1)
        else:
            m = re.match(r"([a-z_]+?_\d{4})_page_\d+\.png$", f)
            if m:
                key = m.group(1)                              # collection dir
        if key:
            hits[key] = hits.get(key, 0) + 1
    return hits


# Rendered-page prefix -> the document that produced it, where the PDF does not
# sit inside the directory named after the collection.
PREFIX_ALIAS = {"Michian standard book": "michigan_2025"}


def used_count(entry, hits, top):
    stem = os.path.splitext(entry["document"])[0]
    return hits.get(stem, hits.get(top, hits.get(PREFIX_ALIAS.get(stem, ""), 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    help="directory holding the agency subdirectories of source PDFs")
    ap.add_argument("--images", default="",
                    help="optional: directory of rasterized pages, to record the render count")
    a = ap.parse_args()

    hits = attribute_pages()
    entries = []
    for dirpath, dirs, files in os.walk(a.corpus):
        dirs[:] = [d for d in dirs if d != "images"]
        for fn in sorted(files):
            if not fn.lower().endswith(".pdf"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, a.corpus)
            top = rel.split(os.sep)[0]
            stem = os.path.splitext(fn)[0]
            if top in CORPUS:
                agency, title, use = CORPUS[top]
            elif stem in ROOT_DOCS:
                agency, title, use = ROOT_DOCS[stem]
            else:
                matched = next((k for k in ROOT_DOCS if stem.startswith(k[:30])), None)
                if matched:
                    agency, title, use = ROOT_DOCS[matched]
                else:
                    agency, title, use = "UNASSIGNED", stem, "confirm before release"
            st = os.stat(path)
            entries.append({
                "agency": agency,
                "collection": title,
                "document": fn,
                "used_for": use,
                "pages": pdf_pages(path),
                "bytes": st.st_size,
                "sha256": sha256(path),
                "file_mtime": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).strftime("%Y-%m-%d"),
                "url": URLS.get(agency, ""),
            })
            entries[-1]["_top"] = top

    for e in entries:
        n = used_count(e, hits, e.pop("_top"))
        e["benchmark_pages_from_this_document"] = n
        e["used_in_reported_index"] = n > 0

    by_agency = {}
    for e in entries:
        if not e["used_in_reported_index"]:
            continue
        b = by_agency.setdefault(e["agency"], {"documents": 0, "pages": 0})
        b["documents"] += 1
        b["pages"] += e["pages"] or 0

    rendered = {}
    if a.images and os.path.isdir(a.images):
        for d in sorted(os.listdir(a.images)):
            p = os.path.join(a.images, d, "images")
            if os.path.isdir(p):
                rendered[d] = sum(1 for f in os.listdir(p)
                                  if f.lower().endswith((".png", ".jpg", ".jpeg")))

    unused = [e for e in entries if not e["used_in_reported_index"]]
    doc = {
        "_note": "Source documents for the five-DOT index, the Michigan transfer set and "
                 "the real-standards rule-grounding corpus. These PDFs are public records "
                 "published by the issuing agencies and are NOT redistributed with this "
                 "package. Verify you hold the same edition by checking sha256, then "
                 "rasterize at 200 DPI (400 DPI for the tiling study) and set PLANS_ROOT.",
        "_verify": "sha256sum <file>   # compare against the sha256 field below",
        "_url_note": "Empty url fields must be filled with the publisher's download page "
                     "before publication.",
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "totals": {
            "documents_on_record": len(entries),
            "documents_used_in_reported_index": sum(1 for e in entries if e["used_in_reported_index"]),
            "source_pages_of_used_documents": sum(e["pages"] or 0 for e in entries if e["used_in_reported_index"]),
            "benchmark_pages_attributed": sum(e["benchmark_pages_from_this_document"] for e in entries),
        },
        "_scope_note": "by_agency counts only documents that contribute at least one "
                       "benchmark page. Documents listed with used_in_reported_index=false "
                       "are present in the working corpus but do NOT back any reported "
                       "result — for example the superseded Caltrans editions. The "
                       "retrieval index (1,898 pages) is larger than the set of pages "
                       "carrying benchmark questions (1,854), so per-document benchmark "
                       "page counts are a lower bound on what was indexed.",
        "unused_documents": [e["document"] for e in unused],
        "by_agency": by_agency,
        "rendered_pages_by_collection": rendered,
        "documents": sorted(entries, key=lambda e: (e["agency"], e["document"])),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1)

    t = doc["totals"]
    print(f"data/sources.json  {t['documents_on_record']} documents on record, "
          f"{t['documents_used_in_reported_index']} used, "
          f"{t['benchmark_pages_attributed']} benchmark pages attributed")
    for ag, b in sorted(by_agency.items()):
        print(f"   {ag:<12} {b['documents']:>4} docs  {b['pages']:>5} pages")
    miss = [e["document"] for e in entries if not e["url"]]
    if miss:
        print(f"   !! {len(miss)} documents have no url — fill before publishing")
    un = [e["document"] for e in entries if e["agency"] == "UNASSIGNED"]
    if un:
        print(f"   !! unassigned: {un[:5]}")
    if unused:
        print(f"   {len(unused)} documents on disk back no reported result "
              f"(flagged used_in_reported_index=false)")


if __name__ == "__main__":
    main()
