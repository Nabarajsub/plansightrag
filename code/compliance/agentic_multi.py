"""Multi-plan compliance pipeline: Planner -> Retriever -> Auditor -> Synthesizer.

For each design (with N components):
  1. Planner: looks at the design + a rules cheatsheet, identifies N components,
     emits one audit step per component (with the applicable plan's search query).
  2. Retriever: for each step, pulls top-1 standard plan from v3 1898-page index.
  3. Auditor: sees DESIGN image only + uses the step's rule + threshold, extracts
     the design value, computes pass/fail. Single-image (avoids OOM on 48GB).
  4. Synthesizer: aggregates step verdicts -> COMPLIANT iff ALL pass.

Memory pattern: ColPali on cuda:1, Qwen-VL-72B 4-bit spread across both GPUs with
max_memory budget. Image resize to 1280px max.
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
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules_registry import COMPONENT_RULES  # noqa

V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"
TOP_K = 5


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def resize_for_vlm(img, max_side=1280):
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side; new_h = int(h * max_side / w)
    else:
        new_h = max_side; new_w = int(w * max_side / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


# ---------- VLM helpers ----------
def vlm_text(model, processor, instruction, max_new=400):
    msgs = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def vlm_image(model, processor, img, instruction, max_new=350):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


# ---------- Retrieval ----------
def stage1_retrieve(manifest, cp_device=None):
    if cp_device is None:
        cp_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    """For each component in each design, retrieve top-1 standard via ColPali."""
    print(f"[step1] loading ColPali on {cp_device}")
    cp = ColPali.from_pretrained(
        "vidore/colpali-v1.2", torch_dtype=torch.bfloat16, device_map=cp_device).eval()
    cpp = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    print(f"[step1] loading v3 index")
    existing = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    db = []
    for r in existing:
        db.append({"embedding": r["embedding"].to(cp_device).to(torch.bfloat16),
                   "metadata": r["metadata"]})

    retrievals = {}
    for d in manifest:
        per_comp = []
        for c in d["components"]:
            rule = COMPONENT_RULES[c["key"]]
            q = rule["search_queries"][0]
            inputs = cpp.process_queries([q]).to(cp_device)
            with torch.no_grad():
                q_emb = cp(**inputs)
            scores = []
            for doc in db:
                inter = torch.matmul(q_emb, doc["embedding"].T)
                scores.append(inter.max(dim=-1).values.sum(dim=-1).item())
            top_idx = np.argsort(scores)[::-1][:TOP_K]
            top_meta = [db[i]["metadata"] for i in top_idx]
            per_comp.append({
                "component": c["key"], "search_query": q,
                "applicable_plans_hint": rule["applicable_plans_hint"],
                "top1": {"agency": top_meta[0]["agency"], "plan_id": top_meta[0].get("plan_id"),
                         "sheet_title": top_meta[0].get("sheet_title", "")},
                "top5": [{"agency": m["agency"], "plan_id": m.get("plan_id"),
                          "sheet_title": m.get("sheet_title", "")[:50]} for m in top_meta],
            })
        retrievals[d["name"]] = per_comp
        print(f"  [retrieve] {d['name']:18s} N={len(per_comp)}")

    del cp, cpp, db, existing
    gc.collect(); torch.cuda.empty_cache()
    return retrievals


# ---------- Pipeline stages ----------
def planner_step(model, processor, design_img, components_for_planner):
    """Planner sees the drawing + the list of components and applicable rules.
    Outputs an ordered list of audit steps (one per component)."""
    cheat = "\n".join(
        f"  - {c['key']}: threshold {c['operator']} {c['threshold']} {c['unit']}  ({c['rule_name']})"
        for c in components_for_planner
    )
    prompt = f"""You are a compliance Planner reviewing a multi-component engineering design.

The design under review depicts {len(components_for_planner)} labeled component values.
Each component is governed by a separate engineering standard plan.

Applicable compliance rules (from registry):
{cheat}

Task: Produce one audit step per component. Output STRICT JSON list:
[
  {{"step": 1, "component": "<key>", "check": "<what to verify>"}},
  ...
]
Return ONLY the JSON list."""
    raw = vlm_image(model, processor, design_img, prompt, max_new=500)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # Fallback: one step per component
    return [{"step": i + 1, "component": c["key"], "check": c["rule_name"]}
            for i, c in enumerate(components_for_planner)]


def auditor_step(model, processor, design_img, component_info, retrieval_info):
    rule = COMPONENT_RULES[component_info["key"]]
    ref = retrieval_info["top1"]
    # Use per-drawing threshold from manifest (already pre-resolved)
    thresh = component_info.get("threshold", rule["threshold"])
    prompt = f"""You are a compliance Auditor. Look at the DESIGN drawing.

Component under audit: {component_info["key"]} ({rule["rule_name"]})
Applicable standard threshold: value must be {rule["operator"]} {thresh} {rule["unit"]}
Retrieved reference standard (top-1): {ref["agency"]} / {ref["plan_id"]} / {ref["sheet_title"]}

Procedure:
  1. EXTRACT the actual numeric value for this component from the DESIGN drawing.
  2. Compare to threshold: state "X {rule["operator"]} {thresh}{rule["unit"]} -> PASS or FAIL".

Output STRICT JSON:
{{"actual_value": "...", "compare": "X vs Y", "result": "PASS or FAIL", "notes": "..."}}"""
    return vlm_image(model, processor, design_img, prompt, max_new=200)


def synthesizer_step(model, processor, query, audit_results):
    findings = "\n".join(
        f"  - {r['component']}: {r['finding_json']}" for r in audit_results)
    prompt = f"""You are the Senior Compliance Auditor. Aggregate the per-step findings.

Original query: {query}

Audit findings:
{findings}

Rule: if ANY step result is "FAIL", overall verdict = NON_COMPLIANT. Otherwise COMPLIANT.

Output STRICT JSON:
{{"verdict": "COMPLIANT" or "NON_COMPLIANT",
  "rationale": "...",
  "violated_components": ["..."]}}"""
    return vlm_text(model, processor, prompt, max_new=250)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--judge_model_id", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--report_path", required=True)
    args = ap.parse_args()

    t0 = time.time()
    manifest = json.load(open(args.manifest))
    print(f"[main] {len(manifest)} multi-plan drawings  judge={args.judge_model_id}")

    retrievals = stage1_retrieve(manifest)

    print(f"[step2] loading judge VLM ({args.judge_model_id})")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.judge_model_id, **kwargs).eval()
    qp = AutoProcessor.from_pretrained(args.judge_model_id, trust_remote_code=True)

    results = []
    out_fail = args.report_path.replace(".json", "_failures.jsonl")
    fail_f = open(out_fail, "w")
    for d in manifest:
        name = d["name"]
        n_plans = d["n_plans"]
        gt = d["ground_truth_compliant"]
        print(f"\n=== {name}  N={n_plans}  gt={'COMP' if gt else 'NON_COMP'} ===")

        design_img = resize_for_vlm(Image.open(d["image_path"]).convert("RGB"))
        # Build planner input (component + rule cheatsheet)
        comp_for_planner = []
        for c in d["components"]:
            rule = COMPONENT_RULES[c["key"]]
            # Prefer per-drawing threshold from manifest (already resolved, e.g.,
            # stirrup_spacing d/2 -> 12 for a 24" column). Fall back to registry
            # default if the manifest didn't pre-resolve.
            thresh = c.get("threshold", rule["threshold"])
            comp_for_planner.append({
                "key": c["key"], "threshold": thresh, "operator": rule["operator"],
                "unit": rule["unit"], "rule_name": rule["rule_name"],
            })
        steps = planner_step(qm, qp, design_img, comp_for_planner)
        print(f"  [planner] {len(steps)} steps")

        # Audit each step
        retr = retrievals[name]
        audit_results = []
        for i, step in enumerate(steps):
            comp_key = step.get("component", d["components"][i]["key"] if i < len(d["components"]) else "?")
            comp = next((c for c in d["components"] if c["key"] == comp_key),
                        d["components"][i] if i < len(d["components"]) else None)
            ret = next((r for r in retr if r["component"] == comp_key),
                       retr[i] if i < len(retr) else retr[0])
            if not comp or not ret:
                continue
            finding_raw = auditor_step(qm, qp, design_img, comp, ret)
            finding = parse_json(finding_raw) or {"result": "?", "raw": finding_raw[:300]}
            print(f"  [audit step {i+1}] {comp_key}: {finding.get('result','?')}")
            audit_results.append({
                "step": step.get("step", i + 1), "component": comp_key,
                "ref_plan": f"{ret['top1']['agency']}/{ret['top1']['plan_id']}",
                "finding_json": finding,
            })

        # Synthesize
        synth_raw = synthesizer_step(qm, qp, f"Multi-component compliance check: {d['name']}", audit_results)
        synth = parse_json(synth_raw) or {"verdict": "?", "raw": synth_raw[:300]}
        agent_v = synth.get("verdict", "?")
        expected_v = "COMPLIANT" if gt else "NON_COMPLIANT"
        correct = agent_v == expected_v
        print(f"  [synth] agent={agent_v}  expected={expected_v}  correct={correct}")
        if synth.get("violated_components"):
            print(f"  cited violations: {synth.get('violated_components')}")

        rec = {
            "name": name, "n_plans": n_plans, "plan_id": d["plan_id"],
            "ground_truth_compliant": gt,
            "expected_violated_key": d.get("violated_key"),
            "expected_violated_rule": d.get("violated_rule"),
            "planner_steps": steps,
            "retrievals_per_component": retr,
            "audit_results": audit_results,
            "synthesis": synth, "judge_correct": correct,
        }
        results.append(rec)
        if not correct:
            fail_f.write(json.dumps(rec) + "\n")
    fail_f.close()

    n = len(results)
    n_correct = sum(int(r["judge_correct"]) for r in results)
    # Per-N breakdown
    by_n = {}
    for r in results:
        by_n.setdefault(r["n_plans"], []).append(int(r["judge_correct"]))
    summary = {
        "n_total": n,
        "judge_correct_pct": 100 * n_correct / n,
        "judge_correct_on_compliant": f"{sum(int(r['judge_correct']) for r in results if r['ground_truth_compliant'])}/{sum(1 for r in results if r['ground_truth_compliant'])}",
        "judge_correct_on_noncompliant": f"{sum(int(r['judge_correct']) for r in results if not r['ground_truth_compliant'])}/{sum(1 for r in results if not r['ground_truth_compliant'])}",
        "by_n_plans": {n: f"{sum(v)}/{len(v)} ({100*sum(v)/len(v):.0f}%)" for n, v in by_n.items()},
        "elapsed_min": (time.time() - t0) / 60,
        "results": results,
    }
    with open(args.report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {args.report_path}")
    print(f"[summary] {n_correct}/{n} correct  by_N={summary['by_n_plans']}")


if __name__ == "__main__":
    main()
