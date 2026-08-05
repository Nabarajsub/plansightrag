"""Page-level train/dev/test split of the 4,056-pair scale_v3 benchmark.

Splits by IMAGE PATH (page), not by question, so no page leaks across splits.
Stratified by agency. Michigan stays fully held-out as a 6th-DOT zero-shot set.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---


import json
import random
from collections import defaultdict

SRC = f"{PSR_ROOT}/qna_expansion/scale_v3_final.jsonl"
OUT_DIR = f"{PSR_ROOT}/lora_finetune"
SEED = 2026
TRAIN_FRAC, DEV_FRAC = 0.80, 0.10  # test gets the remaining 0.10


def main():
    random.seed(SEED)
    rows = [json.loads(l) for l in open(SRC)]
    print(f"[split] {len(rows)} total QnA pairs")

    # Group by (agency, page) so we can stratify the split by agency
    pages_by_agency = defaultdict(set)
    for r in rows:
        pages_by_agency[r["agency"]].add(r["image_path"])

    train_pages, dev_pages, test_pages = set(), set(), set()
    for agency, pages in pages_by_agency.items():
        plist = sorted(pages)
        random.shuffle(plist)
        n = len(plist)
        n_tr = int(n * TRAIN_FRAC)
        n_dv = int(n * DEV_FRAC)
        train_pages.update(plist[:n_tr])
        dev_pages.update(plist[n_tr:n_tr + n_dv])
        test_pages.update(plist[n_tr + n_dv:])
        print(f"  {agency:10s} pages: {n}  -> train {n_tr} / dev {n_dv} / test {n - n_tr - n_dv}")

    splits = {"train": [], "dev": [], "test": []}
    for r in rows:
        p = r["image_path"]
        if p in train_pages:
            splits["train"].append(r)
        elif p in dev_pages:
            splits["dev"].append(r)
        else:
            splits["test"].append(r)

    # Sanity: no page overlap
    assert not (train_pages & dev_pages), "train/dev page leak"
    assert not (train_pages & test_pages), "train/test page leak"
    assert not (dev_pages & test_pages), "dev/test page leak"

    for name, data in splits.items():
        out = f"{OUT_DIR}/split_{name}.jsonl"
        with open(out, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        n_hit = sum(int(r.get("final_hit", False)) for r in data)
        print(f"[split] {name}: {len(data)} pairs  ({n_hit} hits, {len(data)-n_hit} misses)  -> {out}")

    print(f"\n[split] page-disjoint splits confirmed. No leakage.")
    print(f"[split] LoRA trains on split_train.jsonl (hits + misses both).")


if __name__ == "__main__":
    main()
