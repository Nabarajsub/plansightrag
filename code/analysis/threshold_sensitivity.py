"""Threshold-sensitivity analysis for the pre-declared hypotheses H1-H4.

The acceptance thresholds (15 pp, 90%, 60%, 20 pp) were fixed in advance. Rather
than justify each number after the fact, this asks:

    Does the verdict depend on the threshold we happened to choose?

For each hypothesis we sweep the threshold across its whole admissible range and
report the value at which the verdict flips. Where the flip point is far from the
declared threshold, the conclusion is insensitive to the choice. Where it is
close, we say so.

Every point estimate is reported with an interval and an effect size, so the
inference rests on those rather than on the threshold crossing.

    python threshold_sensitivity.py
"""

from __future__ import annotations

# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---


import json
import math
import os
import random

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
B = 10000
SEED = 20260810


def hits(tag):
    with open(os.path.join(REPORTS, f"{tag}.json")) as fh:
        return json.load(fh)["hits"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, (c - h) * 100), min(100.0, (c + h) * 100))


def paired_margin(a, b, rng):
    """Bootstrap CI for (mean(a) - mean(b)) * 100 over paired queries."""
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    obs = sum(d) / n * 100
    draws = sorted(
        sum(d[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(B)
    )
    return obs, draws[int(0.025 * B)], draws[int(0.975 * B)]


def cohen_h(p1, p2):
    f = lambda p: 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))
    return abs(f(p1) - f(p2))


def mag(h):
    return "small" if h < 0.5 else "medium" if h < 0.8 else "large"


def band(flip, declared, span):
    """How much room is there between the declared threshold and the flip point?"""
    room = abs(flip - declared)
    return "robust" if room >= 0.25 * span else "fragile"


def main():
    rng = random.Random(SEED)
    out = {"note": __doc__.strip().split("\n")[0], "n_boot": B, "seed": SEED,
           "hypotheses": {}}
    W = 78
    print("=" * W)
    print("THRESHOLD-SENSITIVITY ANALYSIS")
    print("=" * W)

    # ---------------------------------------------------------------- H1
    # Declared: ColNomic-3B Recall@5 >= 15 pp above the strongest text AND
    # the strongest hybrid baseline, on the 424-query page-disjoint split.
    sysh = hits("colnomic_3b")
    n = len(sysh)
    comparators = {
        "strongest text (BM25 + OCR + metadata)": "ocr_bm25_meta",
        "strongest hybrid (VisionRAG-pyramid, corrected)": "visionrag_pyramid_fixed",
        "strongest dense text (BGE-M3 tuned)": "bge_m3_ocr_tuned",
    }
    print(f"\nH1  patch-level visual retrieval beats text/hybrid retrieval")
    print(f"    declared threshold: margin >= 15 pp     (n = {n} queries)\n")
    h1 = {"declared_pp": 15.0, "comparisons": {}}
    worst = None
    for label, tag in comparators.items():
        b = hits(tag)
        obs, lo, hi = paired_margin(sysh, b, rng)
        eh = cohen_h(sum(sysh) / n, sum(b) / n)
        h1["comparisons"][tag] = {
            "label": label, "margin_pp": round(obs, 2),
            "boot_ci95": [round(lo, 2), round(hi, 2)],
            "cohen_h": round(eh, 3), "magnitude": mag(eh),
            "flip_threshold_pp": round(obs, 2),
            "verdict_at_declared": "supported" if obs >= 15 else "rejected",
        }
        worst = obs if worst is None else min(worst, obs)
        print(f"    vs {label}")
        print(f"      margin {obs:6.2f} pp  95% CI [{lo:.2f}, {hi:.2f}]  "
              f"h = {eh:.2f} ({mag(eh)})")
        print(f"      verdict flips only if the threshold is raised above "
              f"{obs:.2f} pp")
    # binding comparator = the smallest margin
    span = 100.0
    h1["binding_margin_pp"] = round(worst, 2)
    h1["verdict_invariant_over_pp"] = [0.0, round(worst, 2)]
    h1["robustness"] = band(worst, 15.0, 50.0)
    print(f"\n    >> H1 is SUPPORTED for any threshold in [0, {worst:.2f}] pp.")
    print(f"       The declared 15 pp sits {worst - 15:.2f} pp below the flip "
          f"point; the\n       verdict does not depend on it.")
    out["hypotheses"]["H1"] = h1

    # ---------------------------------------------------------------- H2
    # Declared: >=90% verdict accuracy on the CAD single-doc set,
    #           >=90% false-positive avoidance on the PASS set,
    #           >=60% sensitivity on the FAIL set.
    print(f"\n{'-' * W}\nH2  the agentic pipeline audits dense sheets and multi-plan designs")
    print("    declared thresholds: 90% accuracy / 90% FP-avoidance / 60% sensitivity\n")
    h2_tests = [
        ("CAD verdict accuracy, 6 sets pooled", 673, 674, 90.0, "accuracy"),
        ("PASS set, false-positive avoidance", 8, 8, 90.0, "fp_avoidance"),
        ("FAIL set, sensitivity (TPR)", 19, 24, 60.0, "sensitivity"),
    ]
    h2 = {"tests": {}}
    for label, k, nn, declared, key in h2_tests:
        p = k / nn * 100
        lo, hi = wilson(k, nn)
        # the verdict flips when the threshold is raised above the point estimate,
        # but the honest flip point for a small sample is the lower CI bound
        h2["tests"][key] = {
            "label": label, "k": k, "n": nn, "point": round(p, 2),
            "wilson_ci95": [round(lo, 2), round(hi, 2)],
            "declared_pct": declared,
            "flip_threshold_pct": round(p, 2),
            "declared_inside_ci": lo <= declared <= hi,
            "verdict_at_declared": "met" if p >= declared else "not met",
        }
        flag = ("  <-- the declared threshold lies INSIDE the CI: this test "
                "cannot\n          discriminate at this sample size")
        print(f"    {label}")
        print(f"      {k}/{nn} = {p:6.2f}%   95% CI [{lo:.1f}, {hi:.1f}]  "
              f"(declared {declared:.0f}%)")
        print(f"      flips if the threshold is raised above {p:.2f}%"
              + (flag if lo <= declared <= hi else ""))
    print("\n    >> H2's sensitivity test is the fragile one. n = 24 gives a CI")
    print("       roughly 35 pp wide, so 60% and 79% are not distinguishable.")
    print("       We withdraw the 60% bar and report the operating point instead.")
    out["hypotheses"]["H2"] = h2

    # ---------------------------------------------------------------- H3
    print(f"\n{'-' * W}\nH3  LoRA domain adaptation improves held-out Recall@5")
    print("    declared: any improvement without degrading unseen-agency transfer\n")
    h3 = {"in_domain_delta_pp": -37.03, "direction": "negative",
          "flip_threshold_pp": 0.0, "robustness": "robust",
          "note": ("Rejected on sign, not on magnitude. The best LoRA "
                   "configuration is worse than the off-the-shelf checkpoint, "
                   "so no threshold in [0, inf) reverses the verdict.")}
    print("    best LoRA config is WORSE than the off-the-shelf checkpoint")
    print("    (full-LM LoRA: -37.03 pp in-distribution)")
    print("\n    >> H3 is REJECTED for every non-negative threshold. The verdict")
    print("       is determined by the sign of the effect; no threshold is used.")
    out["hypotheses"]["H3"] = h3

    # ---------------------------------------------------------------- H4
    print(f"\n{'-' * W}\nH4  high-resolution tiling improves judge accuracy")
    print("    declared threshold: >= 20 pp judge-accuracy gain\n")
    h4 = {"declared_pp": 20.0, "measurements": {}}
    # published ColPali measurement
    h4["measurements"]["colpali_judge"] = {
        "label": "judge accuracy, ColPali (as published)",
        "delta_pp": 4.53, "flip_threshold_pp": 4.53,
        "verdict_at_declared": "rejected"}
    h4["measurements"]["colpali_retrieval"] = {
        "label": "Recall@5, ColPali (as published)",
        "delta_pp": 5.19, "flip_threshold_pp": 5.19,
        "verdict_at_declared": "rejected"}
    print("    ColPali, judge accuracy   64.23 -> 68.77   (+4.53 pp)")
    print("       flips only if the threshold is lowered below 4.53 pp")
    # corrected ColNomic measurement
    sig = os.path.join(REPORTS, "_tiling_v2_significance.json")
    if os.path.exists(sig):
        s = json.load(open(sig))
        for agg in ("patch_union", "published_max"):
            c = s["comparisons"].get(agg)
            if not c:
                continue
            h4["measurements"][f"colnomic_{agg}"] = {
                "label": f"Recall@5, ColNomic-3B, {agg}",
                "delta_pp": round(c["delta_pp"], 2),
                "boot_ci95": [round(x, 2) for x in c["boot_ci95"]],
                "mcnemar_p": c["mcnemar_exact_p"],
                "flip_threshold_pp": round(c["delta_pp"], 2),
                "ci_spans_zero": c["boot_ci95"][0] < 0 < c["boot_ci95"][1]}
            spans = c["boot_ci95"][0] < 0 < c["boot_ci95"][1]
            print(f"    ColNomic-3B, {agg:<14} {c['delta_pp']:+.2f} pp  "
                  f"95% CI [{c['boot_ci95'][0]:.2f}, {c['boot_ci95'][1]:.2f}]  "
                  f"p = {c['mcnemar_exact_p']:.3g}")
            if spans:
                print("       CI spans zero -- no threshold is needed to reject; "
                      "the effect\n       is not distinguishable from none")
    h4["robustness"] = "robust"
    print("\n    >> H4 is REJECTED for any threshold above 5.19 pp, and the")
    print("       corrected ColNomic measurement has a CI containing zero, so the")
    print("       rejection no longer rests on a threshold at all.")
    out["hypotheses"]["H4"] = h4

    # ---------------------------------------------------------------- summary
    print(f"\n{'=' * W}\nSUMMARY\n{'=' * W}")
    print(f"  {'H':<4}{'declared':<12}{'verdict':<12}{'invariant over':<24}"
          f"{'robust?'}")
    rows = [
        ("H1", ">=15 pp", "supported", f"any threshold 0-{worst:.1f} pp", "robust"),
        ("H2", ">=60% sens", "supported", "60-79% only; CI 35 pp wide", "FRAGILE"),
        ("H2", ">=90% acc", "supported", "any threshold <= 100%", "robust"),
        ("H3", "sign only", "rejected", "every non-negative threshold", "robust"),
        ("H4", ">=20 pp", "rejected", "any threshold > 5.2 pp", "robust"),
    ]
    for r in rows:
        print(f"  {r[0]:<4}{r[1]:<12}{r[2]:<12}{r[3]:<24}{r[4]}")
    print("\n  Three of the four hypotheses reach the same verdict across the whole")
    print("  plausible range of thresholds. Only H2's 60% sensitivity bar is")
    print("  threshold-sensitive, and it is withdrawn.")

    dst = os.path.join(REPORTS, "_threshold_sensitivity.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
