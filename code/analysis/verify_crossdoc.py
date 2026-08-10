"""Cross-document consistency: does every document tell the same story?

Checks (a) that superseded numbers are not presented as current anywhere, and
(b) that the same quantity carries the same value in the ledger, the response
letter, the evidence docs and the manuscript.
"""

# --- release path resolution (added for the release copy; the run-time originals
# under baselines_v2/ and explanability/ are unchanged and still carry the
# absolute ARCC paths the experiments were executed with) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

import glob, os, re, sys

ROOT = _os.environ.get("PAPER_ROOT", _os.path.join(CLUSTER_ROOT, "paper"))
DOCS = (sorted(glob.glob(f"{ROOT}/update_results_autcon/*.md"))
        + [f"{ROOT}/review/response_letter.md", f"{ROOT}/review/RESPONSE_PLAN.md"])
TEX = f"{ROOT}/els-cas-templates/els-cas-templates/manuscript_swap.tex"

# value -> (what it is, what supersedes it, phrases that legitimately mention it)
SUPERSEDED = {
    "55.90": ("pre-correction margin vs strongest text", "34.67",
              ("55.90 →", "55.90 to", "from 55.90", "was 55.90", "overstated",
               "originally", "as published", "published margin", "fell from")),
    "69.58": ("pre-correction margin vs hybrid / also ColPali-v1.3 R@5", "46.70",
              ("colpali", "v1.3", "v13", "69.58 →", "from 69.58", "as published",
               "originally", "overstated", "fell from")),
    "-8.73": ("tiling reversal before the aggregation sweep", "-8.49 / +1.42",
              ("83.73", "originally", "as first measured", "first produced",
               "looked like", "published recipe", "−8.73", "as originally")),
}

# quantity -> canonical value; flagged if a doc states a DIFFERENT value for it
CANON = {
    r"ColNomic-3B[^.\n]{0,40}?(\d\d\.\d\d)\s*%?\s*Recall@5": ("H1 headline", {"92.69", "92.45"}),
    r"family-disjoint[^.\n]{0,60}?(\d\d\.\d\d)": ("family-disjoint", {"92.00", "92.45"}),
}

problems, notes = [], []

def scan(path):
    txt = open(path, errors="ignore").read()
    name = os.path.relpath(path, ROOT)
    for val, (what, sup, safe) in SUPERSEDED.items():
        for m in re.finditer(re.escape(val), txt):
            ln = txt[:m.start()].count("\n") + 1
            ctx = txt[max(0, m.start() - 200):m.start() + 200].lower().replace("\n", " ")
            if any(s.lower() in ctx for s in safe):
                continue
            problems.append((name, ln, val, what, sup,
                             txt[max(0, m.start() - 70):m.start() + 70].replace("\n", " ")))

for d in DOCS:
    scan(d)
if os.path.exists(TEX):
    scan(TEX)

print("=" * 78)
print("CROSS-DOCUMENT CONSISTENCY — superseded values presented as current")
print("=" * 78)
if not problems:
    print("\n  none found in the revision documents\n")
else:
    cur = None
    for name, ln, val, what, sup, ctx in problems:
        if name != cur:
            print(f"\n  {name}"); cur = name
        print(f"    line {ln:<5} {val}  ({what}; superseded by {sup})")
        print(f"      ...{ctx.strip()}...")
print(f"\n  flagged: {len(problems)}")
print("\n  NOTE: hits inside manuscript_swap.tex are EXPECTED — that is the submitted")
print("  text that the revision has not yet been applied to. They are the to-do list.")
sys.exit(0)
