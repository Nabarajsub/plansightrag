"""Benchmark-validity analysis: does the refine loop make the benchmark
retrievable-by-construction?

Reports the number of rephrased questions with original and rephrased questions
evaluated separately, anchor usage with results with and without each anchor
type, and how pages, plans and question patterns overlap across splits.

Everything here is a join over data already on disk: the stored per-query hit vector
for the adopted retriever and the refine-loop attempt chain carried by every
benchmark record. No model is re-run, so no published number can move.

Three analyses:

  1. Recall@5 partitioned by refine-loop path
        never rephrased        n_attempts == 1  - the loop never intervened
        rephrased and hit      1 < n_attempts < cap, curation retriever succeeded
        rephrased at cap       n_attempts == cap, curation retriever NEVER succeeded
     If the never-rephrased subset scores as well as the whole, the benchmark is not
     retrievable-by-construction and the circularity objection does not hold.

  2. Anchor composition of the FINAL question wording - plan IDs, quoted note text,
     unique numerics - with Recall@5 computed with and without each anchor type, and
     a comparison against the attempt-1 wording to show what the loop actually added.

  3. Split integrity - unique pages, questions per page, and whether plan FAMILIES
     (e.g. 511-1A and 511-1B) straddle the train/dev/test boundary.

    python benchmark_validity.py
"""
from __future__ import annotations
import collections, json, math, os, re, sys

import numpy as np

# Release path resolution: run from a clone, the shipped splits and reports are
# used; the authors' cluster paths remain as fallbacks for the original runs.
PSR_ROOT = os.environ.get("PSR_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
def _pick(release, cluster):
    return release if os.path.exists(release) else cluster
REPORTS = f"{PSR_ROOT}/reports/retrieval"
SPLITS = {s: _pick(f"{PSR_ROOT}/data/splits/split_{s}.jsonl",
                   f"{ROOT}/lora_finetune/split_{s}.jsonl")
          for s in ("train", "dev", "test")}
HITS = _pick(f"{PSR_ROOT}/reports/retrieval/colnomic_3b.json",
             f"{ROOT}/plansightrag/reports/retrieval/colnomic_3b.json")
OUT = f"{REPORTS}/_benchmark_validity.json"

CAP = {"CDOT": 8}          # CDOT used a stricter two-anchor rephrase with 8 attempts
DEFAULT_CAP = 5

PLAN_ID = re.compile(r"\b\d{3}-\d+[A-Za-z]?\b")           # e.g. 511-1A, 606-4B
QUOTED = re.compile(r"[\"“”'']([^\"“”'']{6,})[\"“”'']")   # quoted note text
UNIQ_NUM = re.compile(r"\b\d+(?:\.\d+)?\s*(?:\"|''|in\b|ft\b|mm\b|m\b|%|ga\b)", re.I)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (c - s) / d, 2), round(100 * (c + s) / d, 2))


def rate(k, n):
    return {"n": n, "hits": k, "recall@5": round(100 * k / n, 2) if n else None,
            "ci95_wilson": wilson(k, n)}


def cap_for(agency):
    return CAP.get(agency, DEFAULT_CAP)


def anchors(q):
    return {"plan_id": bool(PLAN_ID.search(q)),
            "quoted": bool(QUOTED.search(q)),
            "unique_numeric": bool(UNIQ_NUM.search(q))}


def main():
    test = [json.loads(l) for l in open(SPLITS["test"])]
    hits = json.load(open(HITS))["hits"]
    assert len(hits) == len(test), f"hit vector {len(hits)} != test split {len(test)}"
    rep = {"note": "Join over stored per-query hits and the refine-loop attempt chain. "
                   "No model was re-run.", "retriever": "ColNomic-3B",
           "split": "424-pair page-disjoint test split over the 1,898-page index"}

    # ---- 1. refine-loop path ------------------------------------------------
    buckets = collections.defaultdict(lambda: [0, 0])
    for r, h in zip(test, hits):
        na = r.get("n_attempts") or 1
        c = cap_for(r.get("agency", ""))
        b = ("never_rephrased" if na == 1 else
             "rephrased_at_cap" if na >= c else "rephrased_and_hit")
        buckets[b][0] += h
        buckets[b][1] += 1
    rep["by_refine_path"] = {k: rate(v[0], v[1]) for k, v in buckets.items()}
    rep["by_refine_path"]["ALL"] = rate(sum(hits), len(hits))

    # ---- 2. anchor composition ---------------------------------------------
    comp = collections.Counter()
    by_anchor = collections.defaultdict(lambda: [0, 0])
    added = collections.Counter()
    for r, h in zip(test, hits):
        a = anchors(r["query"])
        for k, v in a.items():
            if v:
                comp[k] += 1
            g = f"{k}_{'yes' if v else 'no'}"
            by_anchor[g][0] += h
            by_anchor[g][1] += 1
        if not any(a.values()):
            comp["none"] += 1
            by_anchor["no_anchor_at_all"][0] += h
            by_anchor["no_anchor_at_all"][1] += 1
        # what did the loop ADD relative to the first attempt?
        ch = r.get("attempts") or []
        if ch:
            a0 = anchors(ch[0].get("question", ""))
            for k in a:
                if a[k] and not a0[k]:
                    added[k] += 1
    n = len(test)
    rep["anchor_composition_final_wording"] = {
        k: {"n": c, "pct": round(100 * c / n, 2)} for k, c in comp.items()}
    rep["anchor_added_by_refine_loop"] = {
        k: {"n": c, "pct_of_all": round(100 * c / n, 2)} for k, c in added.items()}
    rep["recall_by_anchor"] = {k: rate(v[0], v[1]) for k, v in by_anchor.items()}

    # ---- 3. split integrity -------------------------------------------------
    fam = lambda pid: re.sub(r"[A-Za-z]$", "", str(pid or "")).strip()
    pages, plans, fams = {}, {}, {}
    per_split_pages = {}
    for s, p in SPLITS.items():
        rows = [json.loads(l) for l in open(p)]
        per_split_pages[s] = len({(os.path.basename(r["image_path"]) if r.get("image_path") else r["page_id"]) for r in rows})
        for r in rows:
            pages.setdefault((os.path.basename(r["image_path"]) if r.get("image_path") else r["page_id"]), set()).add(s)
            plans.setdefault(str(r.get("plan_id")), set()).add(s)
            fams.setdefault(fam(r.get("plan_id")), set()).add(s)
    straddle = lambda d: sum(1 for v in d.values() if len(v) > 1)
    rep["split_integrity"] = {
        "unique_pages_per_split": per_split_pages,
        "pages_in_more_than_one_split": straddle(pages),
        "plan_ids_total": len(plans), "plan_ids_straddling_splits": straddle(plans),
        "plan_families_total": len(fams), "plan_families_straddling_splits": straddle(fams),
        "family_rule": "trailing letter stripped, e.g. 511-1A and 511-1B -> 511-1",
    }
    rep["split_integrity"]["family_disjoint_needed"] = straddle(fams) > 0

    json.dump(rep, open(OUT, "w"), indent=1)

    # ---- console ------------------------------------------------------------
    print(f"=== 1. Recall@5 by refine-loop path (n={n}) ===")
    for k in ("never_rephrased", "rephrased_and_hit", "rephrased_at_cap", "ALL"):
        d = rep["by_refine_path"].get(k)
        if d:
            print(f"  {k:<20} n={d['n']:>4}  R@5 {d['recall@5']:>6.2f}%  {d['ci95_wilson']}")

    print(f"\n=== 2. anchor composition of the final wording ===")
    for k, d in sorted(rep["anchor_composition_final_wording"].items(), key=lambda x: -x[1]["n"]):
        print(f"  {k:<16} {d['n']:>4}  ({d['pct']:>5.2f}% of questions)")
    print("  added BY the refine loop (absent in attempt 1, present in the final wording):")
    for k, d in sorted(rep["anchor_added_by_refine_loop"].items(), key=lambda x: -x[1]["n"]):
        print(f"    {k:<14} {d['n']:>4}  ({d['pct_of_all']:>5.2f}% of all questions)")
    print("  Recall@5 with / without each anchor:")
    for k in ("plan_id", "quoted", "unique_numeric"):
        y, nn = rep["recall_by_anchor"].get(f"{k}_yes"), rep["recall_by_anchor"].get(f"{k}_no")
        if y and nn:
            print(f"    {k:<14} with {y['recall@5']:>6.2f}% (n={y['n']:>3})   "
                  f"without {nn['recall@5']:>6.2f}% (n={nn['n']:>3})")
    na = rep["recall_by_anchor"].get("no_anchor_at_all")
    if na:
        print(f"    {'NO anchors':<14} {na['recall@5']:>6.2f}% (n={na['n']})")

    si = rep["split_integrity"]
    print(f"\n=== 3. split integrity ===")
    print(f"  unique pages per split: {si['unique_pages_per_split']}")
    print(f"  pages straddling splits:          {si['pages_in_more_than_one_split']}")
    print(f"  plan IDs straddling splits:       {si['plan_ids_straddling_splits']} / {si['plan_ids_total']}")
    print(f"  plan FAMILIES straddling splits:  {si['plan_families_straddling_splits']} / {si['plan_families_total']}")
    print(f"  -> family-disjoint split needed:  {si['family_disjoint_needed']}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
