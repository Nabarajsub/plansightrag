"""Capability ladder: decompose compliance verdict accuracy into its stages (R2.4).

Reviewer 2 major 4 asks us to separate retrieval, rule identification, rule
interpretation, value extraction, arithmetic and the final verdict, because a
single "verdict accuracy" number hides which capability is actually carrying the
result -- and hides which one fails first when the task gets harder.

Nothing here needs a GPU. Every stage is recoverable from logs already on disk:

  reports          compliance/compliance_n10_report_n100_*.json
                     retrieval_top5_hit, judge_verdict{checks[], verdict},
                     judge_correct, checks_injected[]
  ground truth     compliance/table9_n100/manifest.json
                     design_facts (the value actually drawn on the sheet),
                     violated_rule, ground_truth_compliant

The six rungs, and what each one isolates:

  1 retrieval          did the gold page make top-5
  2 rule_id            did the judge check the rule that was actually injected
  3 rule_interp        did it use the correct numeric threshold for that rule
  4 value_extract      did it read the correct value off the drawing
  5 arithmetic         given ITS OWN extracted value and threshold, is its
                       PASS/FAIL for that check logically correct
  6 verdict            was the overall COMPLIANT/NON_COMPLIANT verdict right

Rung 5 is deliberately scored against the judge's own inputs rather than against
ground truth. That separates "cannot compare two numbers" from "read the wrong
number" -- which is exactly the distinction the reviewer is asking for, and the
two failure modes carry very different engineering implications.

    python capability_ladder.py
"""

# --- release path resolution (added for the release copy; the run-time originals
# under baselines_v2/ and explanability/ are unchanged and still carry the
# absolute ARCC paths the experiments were executed with) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

from __future__ import annotations

import glob
import json
import math
import os
import re

ROOT = CLUSTER_ROOT
COMP = os.path.join(ROOT, "compliance")
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# The five injected checks map one-to-one onto a design_facts field: that field
# holds the value the CAD generator actually drew on the sheet.
CHECK_FIELD = {
    "Concrete cover": "cover_in",
    "Footing depth": "footing_depth_in",
    "Grate opening": "grate_open_in",
    "Stirrup spacing": "stirrup_spacing_in",
    "Anchor bolt embedment": "anchor_embed_in",
}

CONFIGS = {
    "7B plain": "compliance_n10_report_n100_7b_plain.json",
    "7B + CoT": "compliance_n10_report_n100_7b_cot.json",
    "72B plain": "compliance_n10_report_n100_72b_plain.json",
    "72B + CoT + thresholds": "compliance_n10_report_n100_72b_cot_thresh.json",
    "72B agentic": "compliance_n10_report_n100_agentic_72b.json",
}

RUNGS = [
    ("retrieval", "gold page in top-5"),
    ("rule_id", "checked the injected rule"),
    ("rule_interp", "used the correct threshold"),
    ("value_extract", "read the correct value off the sheet"),
    ("arithmetic", "compared its own two numbers correctly"),
    ("verdict", "final COMPLIANT / NON_COMPLIANT correct"),
]

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def num(x):
    """First number in a string like '0.5 inches' or '4\\'-10\"'."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = NUM.search(str(x).replace(",", ""))
    return float(m.group()) or 0.0 if m else None


def _mentions(blob, val):
    """Does this free text state `val` as a number (0.5 matches '0.5' or '.5')?"""
    for m in NUM.finditer(blob):
        if close(float(m.group()), val):
            return True
    return False


def close(a, b, tol=0.051):
    return a is not None and b is not None and abs(a - b) <= tol


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, (c - h) * 100), min(100.0, (c + h) * 100))


def load_cases(path):
    d = json.load(open(path))
    r = d if isinstance(d, list) else (d.get("results") or d.get("cases") or [])
    return r


def _norm(c):
    """One check shape from four log dialects.

    The configs were written at different times and disagree on field names:
    name / check_name / rule, and actual_value / value. The threshold is
    sometimes its own field and sometimes only the second number of a comparison
    string such as "0.5 vs 2.0".
    """
    name = c.get("name") or c.get("check_name") or c.get("rule") or ""
    val = c.get("actual_value", c.get("value"))
    thr = c.get("threshold")
    if thr is None:
        nums = NUM.findall(str(c.get("compare") or ""))
        thr = float(nums[1]) if len(nums) >= 2 else None
    return {"name": name, "actual_value": val, "threshold": thr,
            "compare": c.get("compare"), "result": c.get("result")}


def judge_checks(case):
    """The judge's per-check findings, normalised across the three log shapes.

    Returns (checks, mode) where mode is:
      "structured" -- the config emitted a checks[] array with name /
                      actual_value / threshold / result. Every rung is scored
                      exactly.
      "agentic"    -- per-step audit_results with check_name + finding_json.
                      Same fields, different nesting; also scored exactly.
      "text"       -- the plain configs emit only a free-text rationale. We can
                      still ask whether the correct rule, value and threshold
                      appear in it, but this is weaker evidence than a parsed
                      field and is labelled as such in the output. Arithmetic
                      cannot be isolated from free text and is left unscored.
    """
    jv = case.get("judge_verdict")
    if isinstance(jv, dict) and isinstance(jv.get("checks"), list):
        return [_norm(c) for c in jv["checks"] if isinstance(c, dict)], "structured"

    au = case.get("audit_results")
    if isinstance(au, list) and au:
        out = []
        for a in au:
            if not isinstance(a, dict):
                continue
            fj = a.get("finding_json") or {}
            out.append(_norm({**fj, "name": a.get("check_name")}))
        return out, "agentic"

    if isinstance(jv, dict):
        blob = " ".join(str(jv.get(k) or "") for k in
                        ("rationale", "violated_rule", "verdict"))
        return [{"_text": blob}], "text"
    return [], "none"


def canonical_injected():
    """The injected check for every case, from the one log that records it.

    The plain configs do not store `checks_injected`, only a free-text
    `expected_violated_rule`. Scoring each config against a different notion of
    ground truth would make the columns incomparable, so we take the injected
    check for all five configs from the single report that carries it and key it
    by case name.
    """
    src = os.path.join(COMP, "compliance_n10_report_n100_72b_cot_thresh.json")
    out = {}
    for c in load_cases(src):
        ci = (c.get("checks_injected") or [None])[0]
        if ci:
            out[c.get("name")] = ci
    return out


def score_case(case, mf, injected_by_name):
    """Return {rung: True/False/None}; None = not applicable / not recoverable."""
    out = {r: None for r, _ in RUNGS}

    # ---- 1 retrieval -------------------------------------------------------
    if case.get("retrieval_top5_hit") is not None:
        out["retrieval"] = bool(case["retrieval_top5_hit"])

    # ---- 6 verdict ---------------------------------------------------------
    if case.get("judge_correct") is not None:
        out["verdict"] = bool(case["judge_correct"])

    inj = (case.get("checks_injected") or [None])[0] \
        or injected_by_name.get(case.get("name"))
    if inj is None or mf is None:
        return out
    inj_name = inj.get("name")
    inj_thr = num(inj.get("threshold"))
    op = (inj.get("operator") or ">=").strip()

    facts = mf.get("design_facts") or {}
    field = CHECK_FIELD.get(inj_name)
    true_val = num(facts.get(field)) if field else None

    checks, mode = judge_checks(case)
    out["_mode"] = mode
    if not checks:
        out["rule_id"] = False
        return out

    # ---- free-text configs: score rungs 2-4 by presence, skip arithmetic ----
    if mode == "text":
        blob = (checks[0].get("_text") or "").lower()
        key = (inj_name or "").lower().split()[-1]        # cover / depth / opening
        out["rule_id"] = key in blob
        if inj_thr is not None:
            out["rule_interp"] = _mentions(blob, inj_thr)
        if true_val is not None:
            out["value_extract"] = _mentions(blob, true_val)
        return out

    # ---- 2 rule identification --------------------------------------------
    key = (inj_name or "").lower().split()[-1]   # cover/depth/opening/spacing/embedment

    def same_rule(c):
        n = (c.get("name") or "").strip().lower()
        if not n:
            return False
        return (n == (inj_name or "").lower() or key in n
                or n in (inj_name or "").lower())

    match = next((c for c in checks if same_rule(c)), None)
    out["rule_id"] = match is not None
    if match is None:
        return out

    # ---- 3 rule interpretation --------------------------------------------
    j_thr = num(match.get("threshold"))
    out["rule_interp"] = close(j_thr, inj_thr)

    # ---- 4 value extraction ------------------------------------------------
    j_val = num(match.get("actual_value"))
    if true_val is not None:
        out["value_extract"] = close(j_val, true_val)

    # ---- 5 arithmetic ------------------------------------------------------
    # Does the judge's own PASS/FAIL follow from ITS OWN two numbers?
    if j_val is not None and j_thr is not None:
        ok = {">=": j_val >= j_thr - 1e-9, ">": j_val > j_thr,
              "<=": j_val <= j_thr + 1e-9, "<": j_val < j_thr}.get(op)
        if ok is not None:
            res = str(match.get("result") or "").strip().upper()
            if res in ("PASS", "FAIL"):
                out["arithmetic"] = (res == "PASS") == ok
    return out


def main():
    manifest = {d["name"]: d for d in
                json.load(open(os.path.join(COMP, "table9_n100", "manifest.json")))}

    injected_by_name = canonical_injected()
    table, derivation = {}, {}
    for label, fname in CONFIGS.items():
        path = os.path.join(COMP, fname)
        if not os.path.exists(path):
            print(f"  [skip] {label}: {fname} not found")
            continue
        cases = load_cases(path)
        agg = {r: [0, 0] for r, _ in RUNGS}   # [k, n]
        modes = set()
        for c in cases:
            sc = score_case(c, manifest.get(c.get("name")), injected_by_name)
            modes.add(sc.pop("_mode", None))
            for r, v in sc.items():
                if v is None:
                    continue
                agg[r][1] += 1
                agg[r][0] += int(v)
        table[label] = agg
        modes.discard(None)
        derivation[label] = "+".join(sorted(modes)) or "none"

    W = 96
    print("=" * W)
    print("CAPABILITY LADDER  (R2.4 / R2-interp 3 / R2-fig 4)")
    print("n = 100 CAD compliance cases, five judge configurations")
    print("=" * W)
    print(f"\n{'stage':<16}" + "".join(f"{l:>16}" for l in table))
    print("-" * W)
    for r, desc in RUNGS:
        row = f"{r:<16}"
        for label in table:
            k, n = table[label][r]
            row += f"{(f'{k/n*100:.0f}% ({k}/{n})' if n else '--'):>16}"
        print(row)
    print("-" * W)
    print(f"{'derived from':<16}" + "".join(f"{derivation.get(l,'-'):>16}" for l in table))
    print("-" * W)
    print("\nstage definitions")
    for r, desc in RUNGS:
        print(f"  {r:<16} {desc}")

    # detail for the deployed recipe
    best = "72B + CoT + thresholds"
    if best in table:
        print(f"\n{'-' * W}\ndeployed recipe: {best}   (Wilson 95% CI)\n")
        for r, _ in RUNGS:
            k, n = table[best][r]
            if not n:
                continue
            lo, hi = wilson(k, n)
            print(f"  {r:<16} {k/n*100:6.2f}%  ({k}/{n})   [{lo:.1f}, {hi:.1f}]")

    out = {"note": "Capability ladder for R2.4. CPU-only; derived from existing logs.",
           "n_cases": 100, "rungs": {r: d for r, d in RUNGS},
           "derivation": derivation,
           "configs": {lbl: {r: {"k": v[0], "n": v[1],
                                 "pct": round(v[0] / v[1] * 100, 2) if v[1] else None,
                                 "wilson95": [round(x, 2) for x in wilson(*v)] if v[1] else None}
                             for r, v in agg.items()}
                       for lbl, agg in table.items()}}
    dst = os.path.join(REPORTS, "_capability_ladder.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
