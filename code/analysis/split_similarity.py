"""Cross-split similarity and duplicate/near-duplicate rates (R2 reproducibility 2).

R2 asks for the exact construction of the full, page-disjoint and Michigan
evaluation sets: unique pages, questions per page, duplicate and near-duplicate
handling, and cross-split similarity. The first two were already reported (A5,
A6); the last two were flagged as owed in `07_release_artifacts.md` and are
computed here.

Two questions, kept separate because they can fail independently:

  PAGE-LEVEL   do the same or near-identical *pages* appear in more than one
               split? Exact duplicates are caught by content hash; near-duplicates
               by plan-family identity, since a standard-plan family (511-1A,
               511-1B, ...) is a set of sheets that differ in one detail.

  QUESTION-LEVEL  do the same or near-identical *questions* straddle splits? Two
               questions can be textually near-identical while sitting on
               different pages, which would leak the answer pattern even under a
               page-disjoint split. Measured with character 5-gram Jaccard, which
               needs no model and is reproducible anywhere.

No embedding model is used on purpose: a lexical measure is auditable by a
reviewer with a text editor, and the claim we need to support is about
construction, not semantics.

    python split_similarity.py
"""

# --- release path resolution (release copy; the run-time original under
# baselines_v2/ is unchanged) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

ROOT = CLUSTER_ROOT
REPORTS = _os.path.join(PSR_ROOT, "reports", "analysis")
SPLITS = sorted(glob.glob(f"{ROOT}/lora_finetune/split_*.jsonl"))
NEAR_T = 0.80          # Jaccard at or above this counts as a near-duplicate
NGRAM = 5


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())


def grams(s, n=NGRAM):
    s = re.sub(r"\s+", " ", norm(s)).strip()
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))} or {s}


def jac(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


def family(path_or_id):
    """511-1A and 511-1B belong to family 511-1; strip a trailing sheet letter."""
    base = os.path.basename(str(path_or_id or ""))
    m = re.search(r"(\d{3,4}[-_]\d{1,3})[A-Za-z]?", base)
    return m.group(1).replace("_", "-") if m else base[:24]


def main():
    splits = {}
    for f in SPLITS:
        name = re.search(r"split_(\w+)\.jsonl", f).group(1)
        splits[name] = [json.loads(l) for l in open(f)]
    order = [k for k in ("train", "dev", "test") if k in splits] or list(splits)
    out = {"note": __doc__.strip().split("\n")[0], "near_duplicate_threshold": NEAR_T,
           "ngram": NGRAM, "splits": {}}

    W = 78
    print("=" * W)
    print("SPLIT CONSTRUCTION — duplicates, near-duplicates, cross-split similarity")
    print("=" * W)

    # ---------------------------------------------------------------- basics
    print(f"\n  {'split':<8}{'items':>8}{'unique pages':>15}{'q/page':>9}"
          f"{'plan families':>16}")
    for s in order:
        rows = splits[s]
        pages = {r.get("image_path") for r in rows}
        fams = {family(r.get("plan_id") or r.get("image_path")) for r in rows}
        out["splits"][s] = {"n_items": len(rows), "n_unique_pages": len(pages),
                            "questions_per_page": round(len(rows) / len(pages), 2),
                            "n_plan_families": len(fams)}
        print(f"  {s:<8}{len(rows):>8}{len(pages):>15}{len(rows)/len(pages):>9.2f}"
              f"{len(fams):>16}")

    # ------------------------------------------------------ page-level overlap
    print(f"\n  PAGE-LEVEL OVERLAP\n  {'-'*(W-4)}")
    page_sets = {s: {r.get("image_path") for r in splits[s]} for s in order}
    fam_sets = {s: {family(r.get("plan_id") or r.get("image_path")) for r in splits[s]}
                for s in order}
    # exact content duplicates across the whole corpus
    hashes = defaultdict(set)
    for s in order:
        for r in splits[s]:
            p = r.get("image_path")
            if p and os.path.exists(p):
                try:
                    h = hashlib.md5(open(p, "rb").read()).hexdigest()
                except OSError:
                    continue
                hashes[h].add(p)
    dup_groups = {h: v for h, v in hashes.items() if len(v) > 1}
    n_dup_pages = sum(len(v) for v in dup_groups.values())
    print(f"    byte-identical page images: {len(dup_groups)} groups covering "
          f"{n_dup_pages} files")
    out["exact_duplicate_page_groups"] = len(dup_groups)
    out["exact_duplicate_pages"] = n_dup_pages

    pair = {}
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            pg = len(page_sets[a] & page_sets[b])
            fm = len(fam_sets[a] & fam_sets[b])
            pair[f"{a}|{b}"] = {"shared_pages": pg, "shared_plan_families": fm,
                                "family_overlap_pct": round(
                                    fm / min(len(fam_sets[a]), len(fam_sets[b])) * 100, 2)}
            print(f"    {a:>5} vs {b:<5}  shared pages {pg:>4}   "
                  f"shared plan families {fm:>4}  "
                  f"({pair[f'{a}|{b}']['family_overlap_pct']:.1f}% of the smaller split)")
    out["pairwise_page_overlap"] = pair

    # -------------------------------------------------- question-level overlap
    print(f"\n  QUESTION-LEVEL SIMILARITY (character {NGRAM}-gram Jaccard)\n  {'-'*(W-4)}")
    if "test" in splits:
        test = splits["test"]
        tg = [(r, grams(r.get("query"))) for r in test]
        others = [(s, r, grams(r.get("query"))) for s in order if s != "test"
                  for r in splits[s]]
        near = []
        exact = 0
        seen_test = Counter(norm(r.get("query")) for r in test)
        for r, g in tg:
            best, bestrow, bests = 0.0, None, None
            for s, r2, g2 in others:
                j = jac(g, g2)
                if j > best:
                    best, bestrow, bests = j, r2, s
            if best >= 0.999:
                exact += 1
            if best >= NEAR_T:
                near.append({"test_query": (r.get("query") or "")[:90],
                             "match_split": bests,
                             "match_query": (bestrow.get("query") or "")[:90],
                             "jaccard": round(best, 3),
                             "same_page": r.get("image_path") == bestrow.get("image_path")})
        print(f"    test questions: {len(test)}")
        print(f"    with a train/dev question at Jaccard >= {NEAR_T}: "
              f"{len(near)}  ({len(near)/len(test)*100:.2f}%)")
        print(f"    exact textual duplicates across splits: {exact}")
        dupe_in_test = sum(v - 1 for v in seen_test.values() if v > 1)
        print(f"    duplicate questions *within* the test split: {dupe_in_test}")
        out["question_level"] = {
            "n_test": len(test), "n_near_duplicate_cross_split": len(near),
            "pct_near_duplicate": round(len(near) / len(test) * 100, 2),
            "n_exact_cross_split": exact,
            "n_duplicate_within_test": dupe_in_test,
            "examples": near[:10]}
        for e in near[:4]:
            print(f"      J={e['jaccard']:.2f} [{e['match_split']}] "
                  f"{e['test_query'][:58]!r}")

    dst = os.path.join(REPORTS, "_split_similarity.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
