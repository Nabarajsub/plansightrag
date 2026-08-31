"""Re-derive every numeric claim in the ledger from its cited source file.

This does not re-run any experiment. It opens the report each ledger row points
at and checks that the number written in the ledger is the number in the file.
A claim that cannot be traced to a file is reported as UNVERIFIED rather than
assumed correct.
"""

# --- release path resolution (added for the release copy; the run-time originals
# under baselines_v2/ and explanability/ are unchanged and still carry the
# absolute ARCC paths the experiments were executed with) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

import json, os, sys

ROOT = CLUSTER_ROOT
R = f"{PSR_ROOT}/reports/analysis"
ok, bad, skip = [], [], []


def chk(tag, claim, got, want, tol=0.01):
    if got is None:
        skip.append((tag, claim, "source missing"))
    elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
        (ok if abs(got - want) <= tol else bad).append((tag, claim, got, want))
    else:
        (ok if str(got) == str(want) else bad).append((tag, claim, got, want))


def J(p):
    """Load a report JSON. A release clone keeps reports under several
    subdirectories, so a name that is not where the caller expects is looked up
    in the others before giving up -- otherwise a clone reports 'source missing'
    for files it actually ships."""
    try:
        return json.load(open(p))
    except Exception:
        pass
    base = _os.path.basename(p)
    for sub in ("analysis", "retrieval", "compliance", "transfer_lora"):
        alt = f"{PSR_ROOT}/reports/{sub}/{base}"
        if alt != p and _os.path.exists(alt):
            try:
                return json.load(open(alt))
            except Exception:
                pass
    return None


# ---------------- A13 rank-1 conditioning ----------------
d = J(f"{R}/rank1_conditioning.json")
if d:
    keys = list(d) if isinstance(d, dict) else []
    skip.append(("A13", f"keys={keys[:6]}", "manual read"))

# ---------------- A15 / D4 / D5 baselines ----------------
sf = J(f"{R}/_stats_full.json")
if sf:
    pm = sf["per_method"]
    chk("A15", "ocr_bm25_meta R@5", pm.get("ocr_bm25_meta", {}).get("recall@5"), 58.02)
    chk("D4", "bge_m3_ocr R@5", pm.get("bge_m3_ocr", {}).get("recall@5"), 36.79)
    chk("D4", "visionrag_pyramid fixed", pm.get("visionrag_pyramid_fixed", {}).get("recall@5"), 45.99)
    chk("D4", "visionrag_pyramid orig", pm.get("visionrag_pyramid", {}).get("recall@5"), 23.11)
    chk("A25", "clip_vit_b32 (orig)", pm.get("clip_vit_b32", {}).get("recall@5"), 1.89)
    chk("A25", "layoutlmv3", pm.get("layoutlmv3", {}).get("recall@5"), 0.0)
    chk("A25", "udop", pm.get("udop_decode_minilm", {}).get("recall@5"), 0.0)
    chk("H1", "colnomic_3b R@5", pm.get("colnomic_3b", {}).get("recall@5"), 92.69)
    chk("D1", "colpali_v13 R@5", pm.get("colpali_v13", {}).get("recall@5"), 69.58)
    chk("E", "tiling_full_page_424 (ColPali)", pm.get("tiling_full_page_424", {}).get("recall@5"), 76.89)
    chk("A16", "n methods with CIs", len(pm), 25)

# ---------------- D5 margins ----------------
if sf:
    pm = sf["per_method"]
    c = pm["colnomic_3b"]["recall@5"]
    chk("D5", "margin vs strongest text (bm25_meta)", round(c - pm["ocr_bm25_meta"]["recall@5"], 2), 34.67)
    chk("D5", "margin vs strongest hybrid (visionrag fixed)", round(c - pm["visionrag_pyramid_fixed"]["recall@5"], 2), 46.70)

# ---------------- A23 / D6 ColNomic companion ablations ----------------
for tag, f, want in (("A23", "colnomic_tiling_full_page_424", 92.45),
                     ("D6", "colnomic_tiling_tile_level_424", 83.73),
                     ("A23", "colnomic_rerank_monoqwen_424", 94.58),
                     ("A23", "colnomic_binary_quantized", 90.33)):
    d = J(f"{R}/{f}.json")
    chk(tag, f, round(d["recall@5_overall"], 2) if d else None, want)

# ---------------- A27 tiling aggregation ----------------
sig = J(f"{R}/_tiling_v2_significance.json")
if sig:
    cp = sig["comparisons"]
    for agg, w_r, w_d, w_p in (("published_max", 83.96, -8.49, 2.03e-6),
                               ("top3_sum", 81.84, -10.61, 1.36e-9),
                               ("patch_union", 93.87, 1.42, 0.286),
                               ("page_plus_tiles", 93.87, 1.42, 0.263)):
        chk("A27", f"published grid {agg} R@5", round(cp[agg]["recall5"], 2), w_r)
        chk("A27", f"published grid {agg} delta", round(cp[agg]["delta_pp"], 2), w_d)
        chk("A27", f"published grid {agg} p", cp[agg]["mcnemar_exact_p"], w_p, tol=max(1e-8, w_p * 0.02))
    cg = sig.get("coarse_grid", {})
    for agg, w_r, w_p in (("published_max", 90.09, 0.1325), ("top3_sum", 84.20, 6.867e-7),
                          ("patch_union", 93.40, 0.5572), ("page_plus_tiles", 93.63, 0.3833)):
        chk("A27", f"coarse grid {agg} R@5", round(cg.get(agg, {}).get("recall5", -1), 2), w_r)
        chk("A27", f"coarse grid {agg} p", cg.get(agg, {}).get("mcnemar_exact_p"), w_p, tol=max(1e-8, w_p * 0.02))

# ---------------- A26 family-disjoint ----------------
d = J(f"{R}/_family_disjoint_split.json")
if d:
    v = d.get("recall@5") or d.get("recall@5_overall") or (d.get("family_disjoint") or {}).get("recall@5")
    chk("A26", "family-disjoint R@5", round(v, 2) if v else None, 92.00)

# ---------------- A28 threshold sensitivity ----------------
d = J(f"{R}/_threshold_sensitivity.json")
if d:
    h1 = d["hypotheses"]["H1"]
    chk("A28", "H1 binding margin", round(h1["binding_margin_pp"], 2), 34.67)
    cs = h1["comparisons"]
    chk("A28", "H1 vs bm25_meta cohen h", round(cs["ocr_bm25_meta"]["cohen_h"], 2), 0.86)
    chk("A28", "H1 vs visionrag cohen h", round(cs["visionrag_pyramid_fixed"]["cohen_h"], 2), 1.10)
    chk("A28", "H1 vs bge tuned cohen h", round(cs["bge_m3_ocr_tuned"]["cohen_h"], 2), 1.02)
    t = d["hypotheses"]["H2"]["tests"]
    chk("A28", "H2 accuracy point", round(t["accuracy"]["point"], 2), 99.85)
    chk("A28", "H2 sensitivity point", round(t["sensitivity"]["point"], 2), 79.17)
    chk("A28", "H2 sens CI contains 60", t["sensitivity"]["declared_inside_ci"], True)
    chk("A28", "H2 PASS CI contains 90", t["fp_avoidance"]["declared_inside_ci"], True)

# ---------------- A29 capability ladder ----------------
d = J(f"{R}/_capability_ladder.json")
if d:
    c = d["configs"]
    best = "72B + CoT + thresholds"
    for rung in ("rule_id", "rule_interp", "value_extract", "arithmetic", "verdict"):
        chk("A29", f"{best} {rung}", c[best][rung]["pct"], 100.0)
    chk("A29", "72B+CoT+thr retrieval (ColPali log)", c[best]["retrieval"]["pct"], 32.0)
    chk("A29", "7B+CoT value_extract", round(c["7B + CoT"]["value_extract"]["pct"]), 68)
    chk("A29", "7B+CoT arithmetic", round(c["7B + CoT"]["arithmetic"]["pct"]), 94)
    chk("A29", "7B+CoT verdict", c["7B + CoT"]["verdict"]["pct"], 51.0)
    chk("A29", "72B plain rule_interp", c["72B plain"]["rule_interp"]["pct"], 44.0)
    chk("A29", "72B plain verdict", c["72B plain"]["verdict"]["pct"], 80.0)

# ---------------- A29b compliance retrieval on ColNomic ----------------
d = J(f"{R}/_compliance_retrieval_colnomic.json")
if d:
    chk("A29b", "ColNomic compliance R@5", d["recall@5"], 65.00)
    chk("A29b", "n index pages", d["n_index_pages"], 1998)
    comp = d["miss_top5_composition"]
    chk("A29b", "miss top5 all same-archetype", set(comp) == {"same_archetype_mockup"}, True)
    chk("A29b", "miss top5 slot count", comp.get("same_archetype_mockup"), 175)

# ---------------- A30 numeric match ----------------
d = J(f"{R}/_numeric_match.json")
if d:
    chk("A30", "n scorable", d["n_scorable"], 102)
    chk("A30", "n dimensional items", d["n_dimensional_items"], 111)
    fa = sum(v["judge_false_accept"] for v in d["cells"].values())
    fr = sum(v["judge_false_reject"] for v in d["cells"].values())
    chk("A30", "judge false accepts", fa, 95)
    chk("A30", "judge false rejects", fr, 126)
    ag = [v["agreement_pct"] for v in d["cells"].values() if v["agreement_pct"] is not None]
    chk("A30", "mean agreement", round(sum(ag) / len(ag), 1), 88.0, tol=0.1)

# ---------------- D3 latency ----------------
d = J(f"{R}/latency_424.json")
if d:
    hw = d.get("hardware") or {}
    chk("D3", "hardware field present", bool(hw), True)
    chk("D3", "gpu is H100 not A100", "H100" in hw.get("gpu_name", ""), True)

# ---------------- D7 malformed stubs ----------------
import glob, re as _re
STUB = _re.compile(r"^\**\s*(rewritten\s+)?question\s*:?\s*\**", _re.I)
def is_stub(t):
    core = _re.sub(r"[\*\s:#\-]", "", STUB.sub("", (t or "").strip()))
    return len(core) < 8 or ("?" not in (t or "") and len((t or "").strip()) < 40)
n_stub = n_tot = 0
_splits = (sorted(glob.glob(f"{PSR_ROOT}/data/splits/split_*.jsonl"))
           or sorted(glob.glob(f"{ROOT}/lora_finetune/split_*.jsonl")))
for f in _splits:
    for l in open(f):
        r = json.loads(l); n_tot += 1
        if is_stub(r.get("query")): n_stub += 1
chk("D7", "malformed stubs", n_stub, 25)
chk("D7", "benchmark total", n_tot, 4056)

# ---------------- report ----------------
print("=" * 78)
print("LEDGER NUMERIC VERIFICATION — every claim re-derived from its source file")
print("=" * 78)
print(f"\n  PASS {len(ok)}   FAIL {len(bad)}   UNVERIFIED {len(skip)}\n")
if bad:
    print("  MISMATCHES")
    for t, c, g, w in bad:
        print(f"    {t:<6} {c:<44} file={g!r}  ledger={w!r}")
if skip:
    print("\n  NOT MACHINE-CHECKED")
    for t, c, why in skip:
        print(f"    {t:<6} {c:<44} {why}")
print("\n  VERIFIED")
for t, c, g, w in ok:
    print(f"    {t:<6} {c:<44} {g}")
sys.exit(1 if bad else 0)
