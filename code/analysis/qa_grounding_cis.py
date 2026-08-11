"""Confidence intervals for the QA and grounding headline numbers (R2 statistics 3).

Reviewer 2 asks for intervals on *all* headline results, not selected analyses.
Retrieval and compliance already carry them; QA and grounding did not --
`vqa_results.json` stored only n and accuracy, and the real-corpus grounding
report stored bare percentages.

Nothing is re-run. Both families keep per-item outcomes on disk:

  QA         vqa_results.json -> coverage_matrix[qid][model::technique] = bool,
             422 questions x 18 model-technique cells
  grounding  real_plan_scaleup/scaled_72b_top15_report.json -> results[] with
             relevant@1 / relevant@15 / extracted_any per task

Three intervals per cell, because they answer different questions:

  Wilson            binomial interval, treats questions as independent
  query bootstrap   resamples questions, distribution-free
  page-clustered    resamples *pages* and takes all their questions together

The third matters for the same reason R2 raises it in statistics item 5:
questions drawn from one page are correlated, so treating 424 questions as 424
independent observations overstates precision. Where the page-clustered interval
is wider than the Wilson interval, that gap is the correlation.

    python qa_grounding_cis.py
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
import math
import os
import random
from collections import defaultdict

ROOT = CLUSTER_ROOT
REPORTS = _os.path.join(PSR_ROOT, "reports", "analysis")
B = 10000
SEED = 42


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, (c - h) * 100), 2), round(min(100.0, (c + h) * 100), 2))


def boot(vals, rng, n_boot=B):
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    d = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n * 100
               for _ in range(n_boot))
    return (round(d[int(0.025 * n_boot)], 2), round(d[int(0.975 * n_boot)], 2))


def boot_clustered(groups, rng, n_boot=B):
    """Resample whole pages (clusters), keeping each page's questions together."""
    keys = list(groups)
    if not keys:
        return (0.0, 0.0)
    out = []
    for _ in range(n_boot):
        k = t = 0
        for _ in range(len(keys)):
            g = groups[keys[rng.randrange(len(keys))]]
            k += sum(g); t += len(g)
        out.append(k / t * 100 if t else 0.0)
    out.sort()
    return (round(out[int(0.025 * n_boot)], 2), round(out[int(0.975 * n_boot)], 2))


def page_of(qid):
    """The coverage_matrix key is '<image path>|<category>|<question prefix>'."""
    return str(qid).split("|")[0]


def main():
    rng = random.Random(SEED)
    out = {"note": __doc__.strip().split("\n")[0], "n_boot": B, "seed": SEED}

    # ------------------------------------------------------------------ QA
    cov = json.load(open(f"{ROOT}/vqa_eval/vqa_results.json"))["coverage_matrix"]
    cells = defaultdict(dict)
    for qid, row in cov.items():
        for cell, ok in row.items():
            cells[cell][qid] = bool(ok)

    W = 92
    print("=" * W)
    print("QA — confidence intervals on the 424-pair page-disjoint test split")
    print("=" * W)
    print(f"\n  {'model::technique':<24}{'acc':>8}{'Wilson 95%':>18}{'query boot':>18}"
          f"{'page-clustered':>20}")
    print("  " + "-" * (W - 2))
    qa = {}
    for cell in sorted(cells):
        d = cells[cell]
        vals = [int(v) for v in d.values()]
        k, n = sum(vals), len(vals)
        groups = defaultdict(list)
        for qid, v in d.items():
            groups[page_of(qid)].append(int(v))
        w = wilson(k, n); qb = boot(vals, rng); pc = boot_clustered(groups, rng)
        qa[cell] = {"k": k, "n": n, "accuracy": round(k / n * 100, 2),
                    "n_unique_pages": len(groups),
                    "ci_wilson": w, "ci_query_bootstrap": qb, "ci_page_clustered": pc}
        print(f"  {cell:<24}{k/n*100:7.2f}%   [{w[0]:5.1f},{w[1]:5.1f}]"
              f"   [{qb[0]:5.1f},{qb[1]:5.1f}]    [{pc[0]:5.1f},{pc[1]:5.1f}]")
    npg = len(set(page_of(q) for q in cov))
    print(f"\n  {len(cov)} questions over {npg} unique pages "
          f"({len(cov)/npg:.2f} per page)")
    widest = max(qa.values(), key=lambda v: v["ci_page_clustered"][1] - v["ci_page_clustered"][0])
    print(f"  page-clustered intervals run "
          f"{max(v['ci_page_clustered'][1]-v['ci_page_clustered'][0] - (v['ci_wilson'][1]-v['ci_wilson'][0]) for v in qa.values()):.1f} pp "
          f"wider than Wilson at most — that gap is the within-page correlation")
    out["qa"] = {"n_questions": len(cov), "n_unique_pages": npg, "cells": qa}

    # ------------------------------------------------------------ grounding
    g = json.load(open(f"{ROOT}/real_plan_scaleup/scaled_72b_top15_report.json"))
    res = g["results"]
    print("\n" + "=" * W)
    print("GROUNDING — real 931-page WYDOT corpus, 130 rule-grounding tasks")
    print("=" * W)
    print(f"\n  {'metric':<26}{'rate':>8}{'Wilson 95%':>18}{'bootstrap 95%':>18}")
    print("  " + "-" * (W - 2))
    gr = {}
    for label, key in (("relevant@1", "relevant@1"),
                       ("relevant@15", "relevant@15"),
                       ("extracted_any", "extracted_any")):
        vals = [int(bool(r.get(key))) for r in res]
        k, n = sum(vals), len(vals)
        w = wilson(k, n); bb = boot(vals, rng)
        gr[label] = {"k": k, "n": n, "rate": round(k / n * 100, 2),
                     "ci_wilson": w, "ci_bootstrap": bb}
        print(f"  {label:<26}{k/n*100:7.2f}%   [{w[0]:5.1f},{w[1]:5.1f}]"
              f"   [{bb[0]:5.1f},{bb[1]:5.1f}]")

    print(f"\n  by rule family (Wilson 95%):")
    fam = defaultdict(list)
    for r in res:
        fam[r.get("family", "?")].append(r)
    gr_fam = {}
    for f, rs in sorted(fam.items(), key=lambda x: -len(x[1])):
        k = sum(1 for r in rs if r.get("relevant@15")); n = len(rs)
        ke = sum(1 for r in rs if r.get("extracted_any"))
        w = wilson(k, n); we = wilson(ke, n)
        gr_fam[f] = {"n": n, "relevant@15": round(k / n * 100, 2), "ci_relevant@15": w,
                     "extracted_any": round(ke / n * 100, 2), "ci_extracted_any": we}
        print(f"    {f[:38]:<40} n={n:<4} rel@15 {k/n*100:5.1f}% [{w[0]:4.1f},{w[1]:5.1f}]"
              f"   extracted {ke/n*100:5.1f}% [{we[0]:4.1f},{we[1]:5.1f}]")
    out["grounding"] = {"corpus_pages": g["corpus_pages"], "n_tasks": len(res),
                        "caveat": g.get("CAVEAT"), "overall": gr, "by_family": gr_fam}

    dst = os.path.join(REPORTS, "_qa_grounding_cis.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
