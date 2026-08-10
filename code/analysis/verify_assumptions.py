"""Test the assumptions the new analyses rest on, rather than trusting them.

Each block states an assumption in words, then checks it against data. An
assumption that cannot be checked is reported as such.
"""

# --- release path resolution (added for the release copy; the run-time originals
# under baselines_v2/ and explanability/ are unchanged and still carry the
# absolute ARCC paths the experiments were executed with) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

import json, os, re, sys
from collections import Counter

ROOT = CLUSTER_ROOT
res = []


def A(name, holds, detail):
    res.append((name, holds, detail))


# --- 1. capability_ladder: the check-name -> design_facts field mapping ------
# ASSUMPTION: "Concrete cover" is drawn as design_facts["cover_in"], etc.
CHECK_FIELD = {"Concrete cover": "cover_in", "Footing depth": "footing_depth_in",
               "Grate opening": "grate_open_in", "Stirrup spacing": "stirrup_spacing_in",
               "Anchor bolt embedment": "anchor_embed_in"}
man = {d["name"]: d for d in json.load(open(f"{ROOT}/compliance/table9_n100/manifest.json"))}
src = json.load(open(f"{ROOT}/compliance/compliance_n10_report_n100_72b_cot_thresh.json"))
src = src if isinstance(src, list) else (src.get("results") or src.get("cases"))
NUM = re.compile(r"-?\d+(?:\.\d+)?")
good = bad = unchecked = 0
mismatch = []
for c in src:
    ci = (c.get("checks_injected") or [None])[0]
    m = man.get(c.get("name"))
    if not ci or not m:
        unchecked += 1; continue
    fld = CHECK_FIELD.get(ci["name"])
    vr = m.get("violated_rule") or ""
    nums = [float(x) for x in NUM.findall(vr)]
    drawn = m["design_facts"].get(fld)
    if drawn is None:
        unchecked += 1; continue
    # for a violated case the rule string states "<drawn> < <threshold>"
    if nums and m.get("ground_truth_compliant") is False:
        if abs(nums[0] - float(drawn)) <= 0.051:
            good += 1
        else:
            bad += 1; mismatch.append((c["name"], ci["name"], fld, drawn, nums[0]))
    else:
        unchecked += 1
A("ladder: check-name -> design_facts field mapping",
  bad == 0 and good > 0,
  f"{good} non-compliant cases cross-checked against their violated_rule string, "
  f"{bad} mismatched, {unchecked} not applicable (compliant cases state no value)"
  + (f" | examples: {mismatch[:3]}" if mismatch else ""))

# --- 2. capability_ladder: injected threshold matches the rule string --------
gt = bt = 0
for c in src:
    ci = (c.get("checks_injected") or [None])[0]
    m = man.get(c.get("name"))
    if not ci or not m or m.get("ground_truth_compliant") is not False:
        continue
    nums = [float(x) for x in NUM.findall(m.get("violated_rule") or "")]
    if len(nums) >= 2:
        (gt if abs(nums[1] - float(ci["threshold"])) <= 0.051 else bt).__int__()
        if abs(nums[1] - float(ci["threshold"])) <= 0.051: gt += 1
        else: bt += 1
A("ladder: injected threshold == threshold in the violated_rule string",
  bt == 0 and gt > 0, f"{gt} agree, {bt} disagree")

# --- 3. ladder: all five configs answer the SAME 100 cases ------------------
names = {}
for lbl, f in (("7B plain", "7b_plain"), ("7B + CoT", "7b_cot"), ("72B plain", "72b_plain"),
               ("72B+CoT+thr", "72b_cot_thresh"), ("agentic", "agentic_72b")):
    p = f"{ROOT}/compliance/compliance_n10_report_n100_{f}.json"
    if not os.path.exists(p): continue
    d = json.load(open(p)); d = d if isinstance(d, list) else (d.get("results") or d.get("cases"))
    names[lbl] = {c.get("name") for c in d}
base = next(iter(names.values()))
A("ladder: all five configs cover an identical case set",
  all(v == base for v in names.values()),
  f"set sizes {[len(v) for v in names.values()]}, "
  f"pairwise identical: {all(v == base for v in names.values())}")

# --- 4. compliance_retrieval: questions identical across configs ------------
qs = {}
for lbl, f in (("72b_plain", "72b_plain"), ("7b_plain", "7b_plain"), ("72b_cot_thresh", "72b_cot_thresh")):
    p = f"{ROOT}/compliance/compliance_n10_report_n100_{f}.json"
    d = json.load(open(p)); d = d if isinstance(d, list) else (d.get("results") or d.get("cases"))
    qs[lbl] = {c.get("name"): (c.get("question") or c.get("query")) for c in d}
b = qs["72b_plain"]
same = all(qs[k][n] == b[n] for k in qs for n in b if n in qs[k])
A("compliance retrieval: the question text is the same across configs "
  "(so re-running retrieval on 72b_plain's questions is protocol-faithful)",
  same, f"compared {len(b)} cases across {len(qs)} configs; identical={same}")

# --- 5. numeric_match: bracketed value really is millimetres ----------------
rows = [json.loads(l) for l in open(f"{ROOT}/vqa_eval/answers_qwen72b.jsonl")]
FTIN = re.compile(r"(\d+(?:\.\d+)?)\s*'(?:\s*-?\s*(\d+(?:\.\d+)?)\s*\")?")
BR = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*\]")
agree = disagree = 0
for r in rows:
    ref = r.get("reference") or ""
    mb, mf = BR.search(ref), FTIN.search(ref)
    if not (mb and mf): continue
    ft = float(mf.group(1)); inch = float(mf.group(2)) if mf.group(2) else 0.0
    mm_imp = ft * 304.8 + inch * 25.4
    mm_br = float(mb.group(1))
    (agree, disagree) = (agree + 1, disagree) if abs(mm_imp - mm_br) <= max(3.0, 0.02 * mm_imp) else (agree, disagree + 1)
A("numeric_match: bracketed [NNNN] in a reference is the millimetre equivalent",
  disagree == 0 and agree > 0,
  f"{agree} references where imperial and bracketed value agree within 2%, {disagree} disagree")

# --- 6. numeric_match: unscorable items really are unscorable ---------------
d = json.load(open(f"{PSR_ROOT}/reports/analysis/_numeric_match.json"))
A("numeric_match: coverage of Dimensional Accuracy",
  d["n_scorable"] + d["n_unscorable"] == d["n_dimensional_items"],
  f"{d['n_scorable']} scorable + {d['n_unscorable']} unscorable = {d['n_dimensional_items']}")

# --- 7. faithfulness probe: was the shape-matched control actually used? ----
p = f"{PSR_ROOT}/reports/analysis/_faithfulness_probe.json"
if os.path.exists(p):
    fp = json.load(open(p))
    fired = sum(1 for r in fp["records"] if r.get("control_shape_matched"))
    A("faithfulness: the shape-matched control fired",
      False, f"fired on {fired}/{len(fp['records'])} items -- the fallback "
             f"(random scatter) ran instead, so masked and control differ in "
             f"dispersion (median {fp['records'][0]['geometry']['masked_dispersion']} "
             f"vs {fp['records'][0]['geometry']['control_dispersion']})")

# --- 8. threshold_sensitivity: H2 accuracy denominator ----------------------
d = json.load(open(f"{PSR_ROOT}/reports/analysis/_threshold_sensitivity.json"))
t = d["hypotheses"]["H2"]["tests"]["accuracy"]
A("threshold sweep: H2 accuracy uses the pooled 6-set CAD total (673/674)",
  (t["k"], t["n"]) == (673, 674), f"k={t['k']} n={t['n']} (manuscript Table: 673/674 = 99.85%)")

# --- report -----------------------------------------------------------------
print("=" * 78)
print("ASSUMPTION VERIFICATION")
print("=" * 78)
nfail = 0
for name, holds, detail in res:
    mark = "HOLDS " if holds else "FAILS "
    if not holds: nfail += 1
    print(f"\n  [{mark}] {name}")
    print(f"           {detail}")
print(f"\n{'-' * 78}\n  {len(res) - nfail} hold, {nfail} do not\n")
