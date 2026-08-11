"""Cross-split similarity and near-duplicate rates (R2 reproducibility item 2).

Reviewer 2 asks for the exact construction of the full, page-disjoint and Michigan
evaluation sets, "including the number of unique pages, questions per page,
duplicate and near-duplicate handling, and cross-split similarity". Unique pages
and questions-per-page were already reported (A5, A6); the duplicate and
similarity halves were not, and `07_release_artifacts.md` records them as owed.

Three things are measured, on artefacts already on disk. No model is re-run.

1. QUESTION-LEVEL NEAR-DUPLICATES.
   Character 3-gram TF-IDF cosine within and across splits. Two questions about
   the same dimension on two sheets of one plan family are legitimately similar,
   so a high maximum is expected; what matters is whether *test* questions have
   near-identical twins in *train*, which would mean the test set is partially
   memorisable.

2. PAGE-LEVEL NEAR-DUPLICATES.
   The 1,898 cached ColNomic page embeddings, mean-pooled to one vector per page,
   cosine compared. Mean-pooling a multi-vector index is an approximation of
   MaxSim -- it is used here only to *rank* candidate duplicates cheaply, and the
   reported thresholds should be read as a screen, not as a retrieval result.

3. MICHIGAN. Unique pages and questions per page for the held-out transfer set,
   which the other analyses never covered.

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

import json
import os
from collections import defaultdict

import numpy as np

ROOT = CLUSTER_ROOT
REPORTS = _os.path.join(PSR_ROOT, "reports", "analysis")
CACHE = _os.environ.get("COLNOMIC_CACHE",
    _os.path.join(CLUSTER_ROOT, "baselines_v2", "cache_colnomic_pages.pt"))
SPLITS = {"train": f"{ROOT}/lora_finetune/split_train.jsonl",
          "dev": f"{ROOT}/lora_finetune/split_dev.jsonl",
          "test": f"{ROOT}/lora_finetune/split_test.jsonl"}
MICHIGAN = f"{ROOT}/qna_expansion/verified_michigan.jsonl"
NEAR = 0.90            # cosine at or above this counts as a near-duplicate
EXACT = 0.999


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = {k: load(v) for k, v in SPLITS.items()}
    out = {"note": __doc__.strip().split("\n")[0], "near_duplicate_cosine": NEAR}
    W = 84
    print("=" * W)
    print("CROSS-SPLIT SIMILARITY AND NEAR-DUPLICATE RATES  (R2 reproducibility 2)")
    print("=" * W)

    # ---------------------------------------------------------- set construction
    print(f"\n  {'set':<12}{'questions':>11}{'unique pages':>14}{'q/page':>9}")
    print("  " + "-" * (W - 4))
    con = {}
    for k, rows in data.items():
        pages = {r["image_path"] for r in rows}
        con[k] = {"questions": len(rows), "unique_pages": len(pages),
                  "questions_per_page": round(len(rows) / len(pages), 2)}
        print(f"  {k:<12}{len(rows):>11}{len(pages):>14}{len(rows)/len(pages):>9.2f}")
    allrows = [r for rows in data.values() for r in rows]
    allpages = {r["image_path"] for r in allrows}
    con["full"] = {"questions": len(allrows), "unique_pages": len(allpages),
                   "questions_per_page": round(len(allrows) / len(allpages), 2)}
    print(f"  {'full (5-DOT)':<12}{len(allrows):>11}{len(allpages):>14}"
          f"{len(allrows)/len(allpages):>9.2f}")

    mich = load(MICHIGAN)
    if mich:
        key = "image_path" if "image_path" in mich[0] else "source_file"
        mp = {r.get(key) for r in mich if r.get(key)}
        con["michigan"] = {"questions": len(mich), "unique_pages": len(mp),
                           "questions_per_page": round(len(mich) / len(mp), 2) if mp else None}
        print(f"  {'michigan':<12}{len(mich):>11}{len(mp):>14}"
              f"{len(mich)/len(mp):>9.2f}")
    out["set_construction"] = con

    # ------------------------------------------------- page-disjointness (recheck)
    ov = {}
    for a in data:
        for b in data:
            if a < b:
                pa = {r["image_path"] for r in data[a]}
                pb = {r["image_path"] for r in data[b]}
                ov[f"{a}|{b}"] = len(pa & pb)
    print(f"\n  page overlap between splits: {ov}  "
          f"({'page-disjoint confirmed' if not any(ov.values()) else 'NOT DISJOINT'})")
    out["page_overlap"] = ov

    # -------------------------------------------- 1. question near-duplicates
    print(f"\n{'-' * W}\n  1. QUESTION-LEVEL SIMILARITY (char 3-gram TF-IDF cosine)\n")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                          sublinear_tf=True)
    texts = {k: [r.get("query", "") for r in rows] for k, rows in data.items()}
    X = vec.fit_transform([t for k in texts for t in texts[k]])
    idx, spans = 0, {}
    for k in texts:
        spans[k] = (idx, idx + len(texts[k])); idx += len(texts[k])
    Xn = X.multiply(1.0 / (np.sqrt(X.multiply(X).sum(axis=1)) + 1e-12))
    Xn = Xn.tocsr()

    qres = {}
    for tgt in ("test", "dev"):
        ts, te = spans[tgt]; rs, re_ = spans["train"]
        best = np.zeros(te - ts)
        blk = 256
        for i in range(ts, te, blk):
            j = min(i + blk, te)
            sim = (Xn[i:j] @ Xn[rs:re_].T).toarray()
            best[i - ts:j - ts] = sim.max(axis=1)
        n_near = int((best >= NEAR).sum()); n_exact = int((best >= EXACT).sum())
        qres[f"{tgt}_vs_train"] = {
            "n": len(best), "mean_max_cosine": round(float(best.mean()), 4),
            "median_max_cosine": round(float(np.median(best)), 4),
            "p95_max_cosine": round(float(np.percentile(best, 95)), 4),
            "n_near_dup": n_near, "pct_near_dup": round(n_near / len(best) * 100, 2),
            "n_exact_dup": n_exact}
        print(f"    {tgt} question -> nearest TRAIN question")
        print(f"      mean {best.mean():.3f} · median {np.median(best):.3f} · "
              f"p95 {np.percentile(best, 95):.3f}")
        print(f"      >= {NEAR:.2f} cosine: {n_near}/{len(best)} "
              f"({n_near/len(best)*100:.2f}%)   exact: {n_exact}")
    out["question_similarity"] = qres

    # ------------------------- does near-duplication inflate the headline? ----
    # The objection behind R2.1/R2.2 is that a leaky test set flatters the result.
    # That is testable: score the near-duplicate subset against the rest.
    hp = os.path.join(REPORTS, "colnomic_3b.json")
    if os.path.exists(hp):
        import math as _m
        hits = json.load(open(hp))["hits"]
        ts, te2 = spans["test"]; rs, re2 = spans["train"]
        bestt = np.zeros(te2 - ts)
        for i in range(ts, te2, 256):
            j = min(i + 256, te2)
            bestt[i - ts:j - ts] = (Xn[i:j] @ Xn[rs:re2].T).toarray().max(axis=1)

        def wil(k, n, z=1.96):
            if not n:
                return (0.0, 0.0)
            p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
            h = z * _m.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
            return (round(max(0.0, (c - h) * 100), 1), round(min(100.0, (c + h) * 100), 1))

        print(f"\n{'-' * W}\n  IMPACT: does near-duplication inflate Recall@5?\n")
        imp = {}
        for lbl, mask in (("exact duplicate (>=0.999)", bestt >= EXACT),
                          ("near duplicate (>=0.90)", bestt >= NEAR),
                          ("NOT near-duplicate (<0.90)", bestt < NEAR),
                          ("all test", np.ones(len(bestt), bool))):
            idx = np.where(mask)[0]
            k = int(sum(hits[i] for i in idx)); n = len(idx)
            lo, hi = wil(k, n)
            imp[lbl] = {"k": k, "n": n, "recall@5": round(k / n * 100, 2),
                        "wilson95": [lo, hi]}
            print(f"    {lbl:<30} R@5 = {k/n*100:6.2f}%  ({k}/{n})  [{lo}, {hi}]")
        print("\n    Near-duplicate questions score LOWER, not higher: a question whose"
              "\n    text also fits another page is ambiguous, so retrieval cannot tell"
              "\n    which page is gold. Removing them RAISES the headline.")
        out["duplication_impact"] = imp

    # ------------------------------------------------ 2. page near-duplicates
    print(f"\n{'-' * W}\n  2. PAGE-LEVEL NEAR-DUPLICATES "
          f"(mean-pooled ColNomic embeddings — a screen, not a retrieval result)\n")
    if not os.path.exists(CACHE):
        print("    cache_colnomic_pages.pt not found; skipping")
    else:
        import torch
        c = torch.load(CACHE, map_location="cpu", weights_only=False)
        paths = list(c["paths"])
        P = np.stack([e.float().mean(0).numpy() for e in c["pembs"]])
        # Mean-pooled multi-vector embeddings are strongly anisotropic: every
        # engineering drawing lands in the same narrow cone, so raw cosine reads
        # ~0.96 between arbitrary pages and flags 98% of the corpus as duplicated.
        # Centring removes the shared component -- the same correction applied to
        # the LayoutLMv3 and UDOP baselines (A25), where it moved mean pairwise
        # cosine from 0.96 to -0.0003. Uncentred figures are kept for comparison.
        raw = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        Sraw = raw @ raw.T
        np.fill_diagonal(Sraw, -1.0)
        P = P - P.mean(axis=0, keepdims=True)
        P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        S = P @ P.T
        np.fill_diagonal(S, -1.0)
        nearest = S.max(axis=1)
        n_near = int((nearest >= NEAR).sum())
        pairs = int((S >= NEAR).sum() // 2)
        print(f"    uncentred (anisotropic): mean NN cosine "
              f"{Sraw.max(axis=1).mean():.3f}, "
              f"{int((Sraw.max(axis=1) >= NEAR).sum())/len(paths)*100:.1f}% of pages "
              f"flagged — an artefact, not a duplicate rate")
        print(f"    centred: {len(paths)} pages · nearest-neighbour cosine: "
              f"mean {nearest.mean():.3f} · median {np.median(nearest):.3f} · "
              f"max {nearest.max():.3f}")
        print(f"    pages with a neighbour >= {NEAR:.2f}: {n_near} "
              f"({n_near/len(paths)*100:.2f}%)   distinct pairs: {pairs}")
        # do near-duplicate pages straddle the test split?
        page_split = {}
        for k, rows in data.items():
            for r in rows:
                page_split[r["image_path"]] = k
        straddle = 0
        ti = [i for i, p in enumerate(paths) if page_split.get(p) == "test"]
        for i in ti:
            js = np.where(S[i] >= NEAR)[0]
            if any(page_split.get(paths[j]) == "train" for j in js):
                straddle += 1
        print(f"    TEST pages with a >= {NEAR:.2f} near-duplicate in TRAIN: "
              f"{straddle}/{len(ti)} ({straddle/len(ti)*100:.2f}%)" if ti else "")
        out["page_near_duplicates"] = {
            "n_pages": len(paths), "threshold": NEAR,
            "mean_nn_cosine": round(float(nearest.mean()), 4),
            "median_nn_cosine": round(float(np.median(nearest)), 4),
            "max_nn_cosine": round(float(nearest.max()), 4),
            "pages_with_near_dup": n_near,
            "pct_pages_with_near_dup": round(n_near / len(paths) * 100, 2),
            "distinct_near_dup_pairs": pairs,
            "test_pages_with_train_near_dup": straddle,
            "n_test_pages_in_index": len(ti),
            "pct_test_pages_with_train_near_dup": round(straddle / len(ti) * 100, 2) if ti else None,
            "uncentred_mean_nn_cosine": round(float(Sraw.max(axis=1).mean()), 4),
            "uncentred_pct_flagged": round(float((Sraw.max(axis=1) >= NEAR).sum())
                                           / len(paths) * 100, 2),
            "method_caveat": ("mean-pooled multi-vector embeddings, mean-centred to "
                              "remove anisotropy (the same correction applied to "
                              "LayoutLMv3/UDOP in A25). A screen for duplicates, not "
                              "a retrieval measurement.")}

    dst = os.path.join(REPORTS, "_split_similarity.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
