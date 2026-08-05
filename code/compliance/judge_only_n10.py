"""Run a powerful judge VLM (Qwen-VL-72B) on pre-computed retrieval data.

Splits retrieval (cheap, done in the 7B run) from judge (heavy, needs 72B).
Reads {drawing, question, top5_refs} from the 7B report, calls the judge on
each, scores against ground truth.
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

# Reuse prompts from the smoke module
from compliance_smoke_n10 import JUDGE_PROMPT, JUDGE_PROMPT_COT, parse_json, vlm_gen


OUT_DIR = f"{PSR_ROOT}/compliance"
MANIFEST = f"{OUT_DIR}/mockups_n10/manifest.json"
RETRIEVAL_SOURCE = f"{OUT_DIR}/compliance_n10_report.json"  # the original 7B run, has top5 + questions
TOP_K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model_id", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--use_cot", action="store_true")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--report_suffix", default="72b_judge_only")
    args = ap.parse_args()

    REPORT = f"{OUT_DIR}/compliance_n10_report_{args.report_suffix}.json"
    FAILURES = f"{OUT_DIR}/compliance_n10_failures_{args.report_suffix}.jsonl"

    t0 = time.time()
    manifest = {d["name"]: d for d in json.load(open(MANIFEST))}
    src = json.load(open(RETRIEVAL_SOURCE))
    src_by_name = {r["name"]: r for r in src["results"]}
    print(f"[main] {len(manifest)} drawings  judge={args.judge_model_id}  cot={args.use_cot}  4bit={args.use_4bit}")

    print(f"[step1] loading judge VLM ({args.judge_model_id})")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.judge_model_id, **kwargs).eval()
    qp = AutoProcessor.from_pretrained(args.judge_model_id, trust_remote_code=True)

    judge_template = JUDGE_PROMPT_COT if args.use_cot else JUDGE_PROMPT

    results = []
    fail_f = open(FAILURES, "w")
    for name, d in manifest.items():
        prev = src_by_name[name]
        question = prev["question"]
        top5 = prev["top5"]
        print(f"\n=== {name}  gt={'COMPLIANT' if d['ground_truth_compliant'] else 'NON_COMPLIANT'} ===")
        img = Image.open(d["image_path"]).convert("RGB")
        summary = (f"plan_id={d['plan_id']}  agency={d['agency']}\n"
                   f"design_facts={json.dumps(d['design_facts'])}")
        refs = "\n".join(f"  ref{i+1}: {x['agency']:8s} {x.get('plan_id','?'):14s}"
                         for i, x in enumerate(top5))
        prompt = judge_template.format(summary=summary, refs=refs, k=TOP_K)
        verdict_raw = vlm_gen(qm, qp, img, prompt, max_new=600 if args.use_cot else 400)
        verdict = parse_json(verdict_raw)
        agent_v = (verdict or {}).get("verdict", "?")
        expected_v = "COMPLIANT" if d["ground_truth_compliant"] else "NON_COMPLIANT"
        correct = agent_v == expected_v
        print(f"[judge] agent={agent_v}  expected={expected_v}  correct={correct}")
        if verdict:
            print(f"[rationale] {verdict.get('rationale','?')[:200]}")
        rec = {
            "name": name, "plan_id": d["plan_id"],
            "ground_truth_compliant": d["ground_truth_compliant"],
            "expected_violated_rule": d.get("violated_rule"),
            "question": question,
            "retrieval_self_rank": prev["retrieval_self_rank"],
            "retrieval_top5_hit": prev["retrieval_top5_hit"],
            "top5": top5,
            "judge_verdict": verdict, "judge_correct": correct,
            "judge_raw": verdict_raw[:1500],
        }
        results.append(rec)
        if not correct:
            fail_f.write(json.dumps(rec) + "\n")
    fail_f.close()

    n = len(results)
    n_retrieve = sum(int(r["retrieval_top5_hit"]) for r in results)
    n_judge = sum(int(r["judge_correct"]) for r in results)
    n_comp = sum(1 for r in results if r["ground_truth_compliant"])
    n_noncomp = n - n_comp
    n_jc = sum(int(r["judge_correct"]) for r in results if r["ground_truth_compliant"])
    n_jn = sum(int(r["judge_correct"]) for r in results if not r["ground_truth_compliant"])
    summary = {
        "n_total": n,
        "retrieval_top5_pct": 100 * n_retrieve / n,
        "judge_correct_pct": 100 * n_judge / n,
        "judge_correct_on_compliant": f"{n_jc}/{n_comp}",
        "judge_correct_on_noncompliant": f"{n_jn}/{n_noncomp}",
        "elapsed_min": (time.time() - t0) / 60,
        "results": results,
    }
    with open(REPORT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {REPORT}  failures -> {FAILURES}")
    print(f"[summary] retrieve@5={n_retrieve}/{n}  judge_correct={n_judge}/{n}  "
          f"(compliant {n_jc}/{n_comp}, noncompliant {n_jn}/{n_noncomp})")


if __name__ == "__main__":
    main()
