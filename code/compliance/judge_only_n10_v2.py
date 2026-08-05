"""72B + CoT + per-drawing pre-computed thresholds.

Injects archetype-specific checks (with thresholds resolved per drawing — e.g.,
for a beam with H=30", max stirrup spacing = 15") so the judge does not need to
do arithmetic on derived rules or guess which value applies.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---


import argparse
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from compliance_smoke_n10 import parse_json, vlm_gen

OUT_DIR = f"{PSR_ROOT}/compliance"
MANIFEST = f"{OUT_DIR}/mockups_n10/manifest.json"
RETRIEVAL_SOURCE = f"{OUT_DIR}/compliance_n10_report.json"
TOP_K = 5


def derive_checks(d: dict) -> list[dict]:
    facts = d["design_facts"]
    arch = d["archetype"]
    if arch == "draw_culvert":
        return [{
            "name": "Concrete cover", "rule": "ACI 318 / AASHTO LRFD",
            "operator": ">=", "threshold": 2.0, "unit": "inches",
            "hint": "labeled 'COVER = X\" MIN' near the wall hatching",
        }]
    if arch == "draw_guardrail":
        return [{
            "name": "Footing depth", "rule": "WYDOT 606.05",
            "operator": ">=", "threshold": 30.0, "unit": "inches",
            "hint": "labeled 'FOOTING DEPTH' on the vertical dimension below ground line",
        }]
    if arch == "draw_rebar":
        beam_h = facts.get("beam_h_in", 24)
        return [
            {"name": "Stirrup spacing", "rule": f"ACI 318 max = d/2 = {beam_h//2}\"",
             "operator": "<=", "threshold": float(beam_h) / 2, "unit": "inches",
             "hint": "labeled '#3 STIRRUPS @ X\" OC' on the leader callout"},
            {"name": "Concrete cover", "rule": "ACI 318 / AASHTO LRFD",
             "operator": ">=", "threshold": 2.0, "unit": "inches",
             "hint": "labeled 'COVER = X\"' on the leader callout"},
        ]
    if arch == "draw_inlet":
        return [{
            "name": "Grate opening", "rule": "PROWAG / ADA (vehicular surfaces)",
            "operator": "<=", "threshold": 4.0, "unit": "inches",
            "hint": "labeled 'GRATE OPENING = X\"' on the leader callout",
        }]
    if arch == "draw_sign_post":
        return [{
            "name": "Anchor bolt embedment", "rule": "AASHTO / std spec",
            "operator": ">=", "threshold": 12.0, "unit": "inches",
            "hint": "labeled 'ANCHOR EMBED' on the vertical dimension below ground line",
        }]
    return []


def build_judge_prompt(d, top5, checks):
    summary = (f"plan_id={d['plan_id']}  agency={d['agency']}\n"
               f"sheet_title={d.get('sheet_title', '?')}\n"
               f"design_facts={json.dumps(d['design_facts'])}")
    refs = "\n".join(f"  ref{i+1}: {x['agency']:8s} {x.get('plan_id','?'):14s}"
                     for i, x in enumerate(top5))
    checks_text = "\n".join(
        f"  Check {i+1}: {c['name']} (rule: {c['rule']})\n"
        f"    - Look at the drawing for the value, hint: {c['hint']}\n"
        f"    - The value must be {c['operator']} {c['threshold']} {c['unit']}"
        for i, c in enumerate(checks)
    )
    return f"""You are a compliance reviewer. Reason STEP BY STEP.

DRAWING SUMMARY:
{summary}

RETRIEVED STANDARD REFERENCES (top {TOP_K}):
{refs}

YOU MUST PERFORM THESE SPECIFIC CHECKS (each is pre-resolved for THIS drawing):
{checks_text}

Procedure:
  Step 1. For EACH check, EXTRACT the actual value visible on the drawing (read the label).
  Step 2. Perform the numeric comparison explicitly: state "actual X {{op}} threshold Y -> PASS or FAIL".
  Step 3. If ANY check is FAIL, the overall verdict is NON_COMPLIANT. Otherwise COMPLIANT.

Output STRICT JSON:
{{
  "checks": [
    {{"name": "...", "actual_value": "...", "threshold": "...", "compare": "X vs Y", "result": "PASS or FAIL"}}
  ],
  "verdict": "COMPLIANT" or "NON_COMPLIANT",
  "rationale": "one short sentence",
  "violated_rule": "the violated rule or 'none'"
}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model_id", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--report_suffix", default="72b_cot_thresh")
    args = ap.parse_args()

    REPORT = f"{OUT_DIR}/compliance_n10_report_{args.report_suffix}.json"
    FAILURES = f"{OUT_DIR}/compliance_n10_failures_{args.report_suffix}.jsonl"

    t0 = time.time()
    manifest = {d["name"]: d for d in json.load(open(MANIFEST))}
    src = json.load(open(RETRIEVAL_SOURCE))
    src_by_name = {r["name"]: r for r in src["results"]}

    print(f"[main] {len(manifest)} drawings  judge={args.judge_model_id}  4bit={args.use_4bit}")
    print(f"[step1] loading judge VLM")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.judge_model_id, **kwargs).eval()
    qp = AutoProcessor.from_pretrained(args.judge_model_id, trust_remote_code=True)

    results = []
    fail_f = open(FAILURES, "w")
    for name, d in manifest.items():
        prev = src_by_name[name]
        checks = derive_checks(d)
        print(f"\n=== {name}  gt={'COMPLIANT' if d['ground_truth_compliant'] else 'NON_COMPLIANT'} ===")
        img = Image.open(d["image_path"]).convert("RGB")
        prompt = build_judge_prompt(d, prev["top5"], checks)
        verdict_raw = vlm_gen(qm, qp, img, prompt, max_new=700)
        verdict = parse_json(verdict_raw)
        agent_v = (verdict or {}).get("verdict", "?")
        expected_v = "COMPLIANT" if d["ground_truth_compliant"] else "NON_COMPLIANT"
        correct = agent_v == expected_v
        print(f"[judge] agent={agent_v}  expected={expected_v}  correct={correct}")
        if verdict:
            print(f"[checks] {verdict.get('checks','?')}")
        rec = {
            "name": name, "plan_id": d["plan_id"],
            "ground_truth_compliant": d["ground_truth_compliant"],
            "checks_injected": checks,
            "question": prev["question"],
            "retrieval_self_rank": prev["retrieval_self_rank"],
            "retrieval_top5_hit": prev["retrieval_top5_hit"],
            "judge_verdict": verdict, "judge_correct": correct,
            "judge_raw": verdict_raw[:2000],
        }
        results.append(rec)
        if not correct:
            fail_f.write(json.dumps(rec) + "\n")
    fail_f.close()

    n = len(results)
    n_retrieve = sum(int(r["retrieval_top5_hit"]) for r in results)
    n_judge = sum(int(r["judge_correct"]) for r in results)
    n_comp = sum(1 for r in results if r["ground_truth_compliant"])
    n_jc = sum(int(r["judge_correct"]) for r in results if r["ground_truth_compliant"])
    n_jn = sum(int(r["judge_correct"]) for r in results if not r["ground_truth_compliant"])
    summary = {
        "n_total": n,
        "retrieval_top5_pct": 100 * n_retrieve / n,
        "judge_correct_pct": 100 * n_judge / n,
        "judge_correct_on_compliant": f"{n_jc}/{n_comp}",
        "judge_correct_on_noncompliant": f"{n_jn}/{n - n_comp}",
        "elapsed_min": (time.time() - t0) / 60,
        "results": results,
    }
    with open(REPORT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {REPORT}")
    print(f"[summary] judge_correct={n_judge}/{n}  (compliant {n_jc}/{n_comp}, noncompliant {n_jn}/{n - n_comp})")


if __name__ == "__main__":
    main()
