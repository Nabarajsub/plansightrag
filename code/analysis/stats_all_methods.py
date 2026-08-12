"""Pooled statistics for EVERY 424-query run, not just the 25 headline methods.

R2 statistics 3 asks for confidence intervals on all headline results. The
existing `_stats_full.json` covers 25 methods -- the main comparison table -- but
leaves out every run added afterwards, several of which are cited as headline
numbers in their own right:

  colnomic_rerank_monoqwen_424   94.58%   the re-ranking ablation
  colnomic_binary_quantized      90.33%   the quantization ablation
  bge_m3_ocr_tuned               50.00%   the strongest dense text baseline
  clip_l14_336_tiled             36.79%   the corrected CLIP
  pix2struct_textcaps_fixed      16.27%   the corrected Pix2Struct
  nougat_base_fixed               2.36%   the corrected Nougat

Every one of those has a stored per-query hit vector, so the intervals are pure
post-processing. Three intervals per method, matching `_stats_full.json`:

  Wilson           binomial, treats the 424 questions as independent
  query bootstrap  resamples questions
  page-clustered   resamples pages, keeping each page's questions together

Duplicate tags (the same aggregation written under both a grid-qualified and an
unqualified filename) are collapsed to the grid-qualified copy.

    python stats_all_methods.py
"""

# --- release path resolution (release copy; the run-time original under
# baselines_v2/ is unchanged) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

from __future__ import annotations
import glob, json, math, os, random, re
from collections import defaultdict

HERE = CLUSTER_ROOT + "/baselines_v2"
REPORTS = _os.path.join(PSR_ROOT, "reports", "analysis")
B, SEED = 10000, 42


def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, (c - h) * 100), 2), round(min(100.0, (c + h) * 100), 2)]


def boot(v, rng):
    n = len(v)
    d = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(B))
    return [round(d[int(.025 * B)], 2), round(d[int(.975 * B)], 2)]


def boot_clustered(groups, rng):
    keys = list(groups); out = []
    for _ in range(B):
        k = t = 0
        for _ in range(len(keys)):
            g = groups[keys[rng.randrange(len(keys))]]
            k += sum(g); t += len(g)
        out.append(k / t * 100 if t else 0.0)
    out.sort()
    return [round(out[int(.025 * B)], 2), round(out[int(.975 * B)], 2)]


def main():
    rng = random.Random(SEED)
    test = [json.loads(l) for l in
            open(CLUSTER_ROOT + "/lora_finetune/split_test.jsonl")]
    pages = [r.get("image_path") for r in test]

    runs = {}
    for f in sorted(glob.glob(os.path.join(REPORTS, "*.json"))):
        try: d = json.load(open(f))
        except Exception: continue
        if not isinstance(d, dict) or not isinstance(d.get("hits"), list): continue
        if len(d["hits"]) != len(test): continue
        tag = d.get("tag") or os.path.basename(f)[:-5]
        # prefer the grid-qualified copy when the same aggregation exists twice
        if tag in runs and not os.path.basename(f).endswith(("_coarse.json", "_published.json")):
            continue
        runs[tag] = d["hits"]

    prev = json.load(open(os.path.join(REPORTS, "_stats_full.json")))["per_method"]
    out = {"note": __doc__.strip().split("\n")[0], "n_boot": B, "seed": SEED,
           "n_queries": len(test), "n_unique_pages": len(set(pages)),
           "n_methods": len(runs), "per_method": {}}

    print("=" * 96)
    print("POOLED STATISTICS — every 424-query run")
    print("=" * 96)
    print(f"\n  {'method':<40}{'R@5':>8}{'Wilson':>16}{'query boot':>16}{'page-clustered':>17}")
    print("  " + "-" * 94)
    for tag in sorted(runs, key=lambda t: -sum(runs[t])):
        h = [int(x) for x in runs[tag]]
        k, n = sum(h), len(h)
        g = defaultdict(list)
        for p, v in zip(pages, h): g[p].append(v)
        rec = {"k": k, "n": n, "recall@5": round(k / n * 100, 2),
               "ci_wilson": wilson(k, n), "ci_query_bootstrap": boot(h, rng),
               "ci_page_clustered": boot_clustered(g, rng),
               "in_stats_full": tag in prev}
        out["per_method"][tag] = rec
        flag = "" if tag in prev else "  NEW"
        print(f"  {tag[:39]:<40}{rec['recall@5']:7.2f}%{str(rec['ci_wilson']):>16}"
              f"{str(rec['ci_query_bootstrap']):>16}{str(rec['ci_page_clustered']):>17}{flag}")
    new = sum(1 for t in runs if t not in prev)
    print(f"\n  {len(runs)} methods total — {new} that had no pooled statistics before")
    dst = os.path.join(REPORTS, "_stats_all_methods.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"  wrote {dst}")


if __name__ == "__main__":
    main()
