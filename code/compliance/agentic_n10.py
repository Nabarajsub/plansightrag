"""Path B: Multi-agent compliance pipeline on n=10 with Qwen-VL-72B.

Reuses the existing Planner / Synthesizer pattern from agentic_compliance/agents.py.
Adds a new AuditorWithDesign that sees BOTH the design drawing AND a retrieved
standard image, so it can compare a specific value on the design to the rule
in the standard.

Retrieval is taken from compliance_n10_report.json (the 7B run) — we only need
the top-5 metadata + look up image_paths via the v3 index. Avoids needing
ColPali loaded alongside the 72B.
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
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

# Reuse derive_checks + parse_json + vlm_gen from existing modules
sys.path.insert(0, f"{PSR_ROOT}/compliance")
from compliance_smoke_n10 import parse_json  # noqa
from judge_only_n10_v2 import derive_checks  # noqa

OUT_DIR = f"{PSR_ROOT}/compliance"
MANIFEST = f"{OUT_DIR}/mockups_n10/manifest.json"
RETRIEVAL_SOURCE = f"{OUT_DIR}/compliance_n10_report.json"
V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"


# ---------- Per-drawing query composition ----------
ARCH_LABEL = {
    "draw_culvert":   "reinforced concrete box culvert section",
    "draw_guardrail": "steel post guardrail foundation",
    "draw_rebar":     "reinforced concrete beam rebar detail",
    "draw_inlet":     "curb drainage inlet",
    "draw_sign_post": "breakaway sign post foundation",
}


def compose_query(d):
    label = ARCH_LABEL.get(d["archetype"], "engineering detail")
    checks = derive_checks(d)
    rule_names = [c["name"] for c in checks]
    return (f"Verify whether the {label} in plan {d['plan_id']} (agency {d['agency']}) "
            f"meets the following compliance requirements: " + ", ".join(rule_names) + ".")


# ---------- Agents ----------
def vlm_gen_text(model, processor, instruction, max_new=512):
    msgs = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def resize_for_vlm(img, max_side=1280):
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side; new_h = int(h * max_side / w)
    else:
        new_h = max_side; new_w = int(w * max_side / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def vlm_gen_one_image(model, processor, img, instruction, max_new=400):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": instruction},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def planner_plan(model, processor, query, checks):
    """Decompose compliance query into ordered sub-checks. Each check
    becomes one audit step with a specific value to extract and a threshold."""
    checks_text = "\n".join(
        f"  - {c['name']} (rule: {c['rule']}, must be {c['operator']} {c['threshold']} {c['unit']})"
        for c in checks
    )
    prompt = f"""You are a Compliance Planner. Decompose this query into one verification step PER required check.

Query: {query}

Required checks (these are PRE-RESOLVED for this drawing):
{checks_text}

For each check, produce one step. Output STRICT JSON list:
[
  {{"step": 1, "check_name": "...", "search_query": "<a short query describing the relevant standard plan to retrieve>", "operator": "...", "threshold": "...", "unit": "..."}},
  ...
]
Return ONLY the JSON list, no preamble."""
    raw = vlm_gen_text(model, processor, prompt, max_new=400)
    parsed = parse_json("[" + raw.split("[", 1)[-1] if "[" in raw else raw)
    # parse_json expects a {...} object. Try list parsing manually:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Fallback: one generic step per check
    return [
        {"step": i + 1, "check_name": c["name"], "search_query": query,
         "operator": c["operator"], "threshold": c["threshold"], "unit": c["unit"]}
        for i, c in enumerate(checks)
    ]


def auditor_with_design(model, processor, design_img, ref_meta, step, drawing_facts):
    """Single-image audit: see DESIGN visually, get retrieved standard reference
    via TEXT (avoids OOM from passing 2 images to 4-bit 72B on 48GB)."""
    ref_text = (f"Agency: {ref_meta.get('agency','?')}, Plan: {ref_meta.get('plan_id','?')}, "
                f"Sheet: {ref_meta.get('sheet_title','?')}")
    prompt = f"""You see the DESIGN drawing under review.

Retrieved STANDARD reference (text context): {ref_text}

Audit step: {step['check_name']}
Required: the value of "{step['check_name']}" in the DESIGN must be {step['operator']} {step['threshold']} {step['unit']}.

DESIGN context (already extracted facts): {json.dumps(drawing_facts)}

Procedure:
  1. EXTRACT the actual numeric value for "{step['check_name']}" from the DESIGN drawing (read the label).
  2. Compare to the threshold: state "X {step['operator']} {step['threshold']}{step['unit']} -> PASS or FAIL".

Output STRICT JSON:
{{
  "actual_value": "...",
  "compare": "X vs Y",
  "result": "PASS or FAIL",
  "evidence_source": "DESIGN",
  "notes": "one short sentence"
}}"""
    return vlm_gen_one_image(model, processor, design_img, prompt, max_new=350)


def synthesizer_synthesize(model, processor, query, audit_results):
    findings = "\n".join(
        f"  - Step {r['step']} ({r['check_name']}): {r['finding_json']}"
        for r in audit_results
    )
    prompt = f"""You are a Senior Compliance Auditor. Synthesize the audit findings into a final verdict.

Original Query: {query}

Audit findings:
{findings}

Rule: if ANY step result is "FAIL", the overall verdict is NON_COMPLIANT. Otherwise COMPLIANT.

Output STRICT JSON:
{{
  "verdict": "COMPLIANT" or "NON_COMPLIANT",
  "rationale": "one sentence citing which check(s) drove the verdict",
  "violated_rules": ["..."]
}}"""
    return vlm_gen_text(model, processor, prompt, max_new=300)


# ---------- v3 image_path lookup ----------
def build_v3_lookup():
    print(f"[init] loading v3 index for image_path lookup")
    recs = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    lookup = {}
    for r in recs:
        m = r["metadata"]
        lookup[(m["agency"], m.get("plan_id", ""))] = m["image_path"]
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model_id", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--report_suffix", default="agentic_72b")
    args = ap.parse_args()

    REPORT = f"{OUT_DIR}/compliance_n10_report_{args.report_suffix}.json"
    FAILURES = f"{OUT_DIR}/compliance_n10_failures_{args.report_suffix}.jsonl"
    REPORTS_MD_DIR = f"{OUT_DIR}/agentic_reports_{args.report_suffix}"
    os.makedirs(REPORTS_MD_DIR, exist_ok=True)

    t0 = time.time()
    manifest = {d["name"]: d for d in json.load(open(MANIFEST))}
    src = json.load(open(RETRIEVAL_SOURCE))
    src_by_name = {r["name"]: r for r in src["results"]}
    v3_lookup = build_v3_lookup()

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
        print(f"\n=== {name}  gt={'COMPLIANT' if d['ground_truth_compliant'] else 'NON_COMPLIANT'} ===")
        query = compose_query(d)
        checks = derive_checks(d)
        print(f"[planner] query: {query[:120]}")

        # Plan
        steps = planner_plan(qm, qp, query, checks)
        if not isinstance(steps, list):
            steps = [{"step": 1, "check_name": checks[0]["name"], "search_query": query,
                      "operator": checks[0]["operator"], "threshold": checks[0]["threshold"],
                      "unit": checks[0]["unit"]}]
        print(f"[planner] {len(steps)} steps")

        # Audit each step (single-image: design only; standard ref as text)
        design_img = resize_for_vlm(Image.open(d["image_path"]).convert("RGB"))
        top5 = src_by_name[name]["top5"]
        audit_results = []
        for i, step in enumerate(steps):
            ref = top5[min(i, len(top5) - 1)] if top5 else {}
            ref_path = v3_lookup.get((ref.get("agency"), ref.get("plan_id"))) if ref else None
            finding_raw = auditor_with_design(qm, qp, design_img, ref, step, d["design_facts"])
            finding = parse_json(finding_raw) or {"result": "?", "raw": finding_raw[:300]}
            audit_results.append({
                "step": step.get("step", i + 1),
                "check_name": step.get("check_name", "?"),
                "ref_plan_id": ref.get("plan_id", "?") if ref else "?",
                "ref_agency":  ref.get("agency",  "?") if ref else "?",
                "ref_image_path": ref_path,
                "finding_json": finding,
            })
            print(f"  [audit step {i+1}] {step.get('check_name','?')}: {finding.get('result','?')}")

        # Synthesize
        synth_raw = synthesizer_synthesize(qm, qp, query, audit_results)
        synth = parse_json(synth_raw) or {"verdict": "?", "raw": synth_raw[:300]}
        agent_v = synth.get("verdict", "?")
        expected_v = "COMPLIANT" if d["ground_truth_compliant"] else "NON_COMPLIANT"
        correct = agent_v == expected_v
        print(f"[synth] agent={agent_v}  expected={expected_v}  correct={correct}")

        rec = {
            "name": name, "plan_id": d["plan_id"],
            "ground_truth_compliant": d["ground_truth_compliant"],
            "query": query, "checks_injected": checks,
            "planner_steps": steps,
            "audit_results": audit_results,
            "synthesis": synth, "synth_raw": synth_raw[:800],
            "judge_correct": correct,
        }
        results.append(rec)
        if not correct:
            fail_f.write(json.dumps(rec) + "\n")
        # also write a per-drawing markdown
        with open(f"{REPORTS_MD_DIR}/{name}.md", "w") as f:
            f.write(f"# Agentic Compliance Report: {name}\n\n")
            f.write(f"**Plan**: {d['plan_id']}  **Agency**: {d['agency']}\n\n")
            f.write(f"**Query**: {query}\n\n")
            f.write(f"## Plan ({len(steps)} steps)\n\n")
            for s in steps:
                f.write(f"- Step {s.get('step','?')}: {s.get('check_name','?')} "
                        f"({s.get('operator','?')} {s.get('threshold','?')} {s.get('unit','?')})\n")
            f.write(f"\n## Audit findings\n\n")
            for r in audit_results:
                f.write(f"### Step {r['step']} — {r['check_name']}\n")
                f.write(f"Reference: {r['ref_agency']} / {r['ref_plan_id']}\n\n")
                f.write(f"```json\n{json.dumps(r['finding_json'], indent=2)}\n```\n\n")
            f.write(f"## Synthesis\n\n```json\n{json.dumps(synth, indent=2)}\n```\n\n")
            f.write(f"**Ground truth**: {expected_v}  **Agent verdict**: {agent_v}  **Correct**: {correct}\n")

    fail_f.close()

    n = len(results)
    n_judge = sum(int(r["judge_correct"]) for r in results)
    n_comp = sum(1 for r in results if r["ground_truth_compliant"])
    n_jc = sum(int(r["judge_correct"]) for r in results if r["ground_truth_compliant"])
    n_jn = sum(int(r["judge_correct"]) for r in results if not r["ground_truth_compliant"])

    summary = {
        "n_total": n,
        "judge_correct_pct": 100 * n_judge / n,
        "judge_correct_on_compliant": f"{n_jc}/{n_comp}",
        "judge_correct_on_noncompliant": f"{n_jn}/{n - n_comp}",
        "elapsed_min": (time.time() - t0) / 60,
        "results": results,
    }
    with open(REPORT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {REPORT}  per-drawing md -> {REPORTS_MD_DIR}/")
    print(f"[summary] judge_correct={n_judge}/{n}  (compliant {n_jc}/{n_comp}, noncompliant {n_jn}/{n - n_comp})")


if __name__ == "__main__":
    main()
