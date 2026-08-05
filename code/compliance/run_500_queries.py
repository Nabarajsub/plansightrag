"""Run 500 NL compliance queries through the agentic Planner-Retriever-Auditor-
Synthesizer pipeline (Qwen2.5-VL-72B 4-bit backbone, v3 1898-page index).

Mirrors the existing batch runner pattern but reads queries from JSON and
processes all 500 in one process to amortize model load.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import argparse, gc, json, os, re, sys, time
from collections import Counter, defaultdict

import torch
from PIL import Image

V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"
QUERIES = f"{PSR_ROOT}/compliance/query500/queries_500.json"
OUT_DIR = f"{PSR_ROOT}/compliance/query500"

from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def resize_for_vlm(img, max_side=1024):
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        return img.resize((max_side, int(h * max_side / w)), Image.LANCZOS)
    return img.resize((int(w * max_side / h), max_side), Image.LANCZOS)


def vlm_text(model, processor, prompt, max_new=350):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def vlm_image(model, processor, img, prompt, max_new=350):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def parse_json_obj(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None


def parse_json_list(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None


def planner(qm, qp, query):
    prompt = f"""You are a Compliance Planner. Decompose this query into 2-3 verification steps.

Query: {query}

Output STRICT JSON list of objects, each with keys (step, description, search_query, expected_plan_id_hint).
Return ONLY the JSON list, no preamble."""
    raw = vlm_text(qm, qp, prompt, max_new=400)
    steps = parse_json_list(raw)
    if not steps:
        return [{"step": 1, "description": "verify the query directly", "search_query": query, "expected_plan_id_hint": None}]
    return steps


def auditor(qm, qp, img, step_desc, original_query, ref_meta):
    ref_txt = f"Retrieved: {ref_meta.get('agency','?')}/{ref_meta.get('plan_id','?')}/{ref_meta.get('sheet_title','?')[:80]}"
    prompt = f"""You are a Compliance Auditor. Look at the retrieved standard plan.

Step Goal: {step_desc}
Original Query: {original_query}
Reference: {ref_txt}

Report:
- Findings: what specific values, notes, or rules are visible on this page?
- Verdict: PASS, FAIL, or INSUFFICIENT_EVIDENCE for this step.
Return JSON: {{"findings": "...", "verdict": "PASS or FAIL or INSUFFICIENT_EVIDENCE"}}"""
    return vlm_image(qm, qp, img, prompt, max_new=300)


def synthesizer(qm, qp, query, audits):
    fbody = "\n".join(f"  - Step {a['step']} ({a['ref_plan']}): {a.get('finding_json',{}).get('verdict','?')} - {a.get('finding_json',{}).get('findings','?')[:200]}"
                      for a in audits)
    prompt = f"""You are the Senior Compliance Auditor. Aggregate the per-step findings into a final verdict.

Original Query: {query}

Audit findings:
{fbody}

Rule: if ANY step verdict is FAIL, overall = FAIL. If all PASS, overall = PASS. If any INSUFFICIENT_EVIDENCE and no FAIL, overall = INSUFFICIENT_EVIDENCE (treated as PASS for binary scoring).

Output STRICT JSON: {{"verdict": "PASS or FAIL or INSUFFICIENT_EVIDENCE", "rationale": "..."}}"""
    return vlm_text(qm, qp, prompt, max_new=200)


def stage1_retrieve_all(queries, cp_device=None):
    if cp_device is None:
        cp_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    print(f"[s1] loading ColPali on {cp_device}")
    cp = ColPali.from_pretrained("vidore/colpali-v1.2", torch_dtype=torch.bfloat16, device_map=cp_device).eval()
    cpp = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    print(f"[s1] loading v3 index")
    db_raw = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    db = []
    for r in db_raw:
        db.append({"embedding": r["embedding"].to(cp_device).to(torch.bfloat16), "metadata": r["metadata"]})
    print(f"[s1] index = {len(db)}; running planner+retrieval pre-pass for {len(queries)} queries")

    # PRE-PASS: for each query, run a single retrieval using the query text (top-3).
    # We do NOT yet run the planner (planner needs the judge model). We use query-text
    # retrieval to populate evidence pools; planner runs in stage 2 and refines.
    import numpy as np
    retrievals = {}
    for q in queries:
        inputs = cpp.process_queries([q["query"]]).to(cp_device)
        with torch.no_grad():
            q_emb = cp(**inputs)
        scores = []
        for doc in db:
            inter = torch.matmul(q_emb, doc["embedding"].T)
            scores.append(inter.max(dim=-1).values.sum(dim=-1).item())
        top_idx = np.argsort(scores)[::-1][:3]
        retrievals[q["id"]] = [db[i]["metadata"] for i in top_idx]
    del cp, cpp, db, db_raw; gc.collect(); torch.cuda.empty_cache()
    return retrievals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model_id", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="limit number of queries for debugging")
    ap.add_argument("--report_path", default=f"{OUT_DIR}/query500_report.json")
    args = ap.parse_args()

    t0 = time.time()
    queries = json.load(open(QUERIES))
    if args.limit > 0:
        queries = queries[: args.limit]
    print(f"[main] {len(queries)} queries  judge={args.judge_model_id}")

    retrievals = stage1_retrieve_all(queries)

    print(f"[s2] loading judge VLM")
    kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.use_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.judge_model_id, **kw).eval()
    qp = AutoProcessor.from_pretrained(args.judge_model_id, trust_remote_code=True)

    results = []
    for i, q in enumerate(queries):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"\n[progress] {i+1}/{len(queries)}  elapsed={(time.time()-t0)/60:.1f} min")
        steps = planner(qm, qp, q["query"])
        # Audit each step against retrieved top-1 (cycle through retrievals if more steps than refs)
        refs = retrievals[q["id"]]
        audits = []
        for j, step in enumerate(steps):
            ref = refs[min(j, len(refs) - 1)]
            try:
                img = resize_for_vlm(Image.open(ref["image_path"]).convert("RGB"))
            except Exception:
                continue
            finding_raw = auditor(qm, qp, img, step.get("description", "verify step"), q["query"], ref)
            audits.append({
                "step": step.get("step", j + 1),
                "ref_plan": f"{ref['agency']}/{ref.get('plan_id','?')}",
                "ref_image_path": ref["image_path"],
                "finding_raw": finding_raw[:600],
                "finding_json": parse_json_obj(finding_raw) or {},
            })
        synth_raw = synthesizer(qm, qp, q["query"], audits)
        synth = parse_json_obj(synth_raw) or {"verdict": "?", "raw": synth_raw[:300]}
        agent_v = synth.get("verdict", "?")
        # Map INSUFFICIENT_EVIDENCE -> PASS for binary scoring (per the prompt rule)
        agent_v_binary = "FAIL" if agent_v == "FAIL" else "PASS"
        gt = q["ground_truth"]
        correct = agent_v_binary == gt
        results.append({
            "id": q["id"], "query": q["query"],
            "ground_truth": gt, "mutation": q.get("mutation"),
            "planner_steps": steps, "audit_results": audits,
            "synthesis": synth, "agent_verdict_raw": agent_v,
            "agent_verdict_binary": agent_v_binary,
            "correct": correct,
        })

    # Aggregate
    n = len(results)
    n_correct = sum(int(r["correct"]) for r in results)
    n_pass = sum(1 for r in results if r["ground_truth"] == "PASS")
    n_fail = sum(1 for r in results if r["ground_truth"] == "FAIL")
    n_correct_pass = sum(int(r["correct"]) for r in results if r["ground_truth"] == "PASS")
    n_correct_fail = sum(int(r["correct"]) for r in results if r["ground_truth"] == "FAIL")
    # Per-mutation TPR (sensitivity)
    per_mut = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in results:
        m = r.get("mutation")
        if m:
            per_mut[m][1] += 1
            per_mut[m][0] += int(r["correct"])

    summary = {
        "n_total": n, "overall_acc_pct": 100 * n_correct / n,
        "pass_acc": f"{n_correct_pass}/{n_pass} ({100*n_correct_pass/max(n_pass,1):.1f}%)",
        "fail_tpr":  f"{n_correct_fail}/{n_fail} ({100*n_correct_fail/max(n_fail,1):.1f}%)",
        "per_mutation_tpr": {m: f"{v[0]}/{v[1]} ({100*v[0]/max(v[1],1):.1f}%)" for m, v in per_mut.items()},
        "elapsed_min": (time.time() - t0) / 60,
        "results": results,
    }
    with open(args.report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] {args.report_path}")
    print(f"[summary] {n_correct}/{n} overall ({100*n_correct/n:.1f}%)")
    print(f"  PASS: {summary['pass_acc']}  FAIL TPR: {summary['fail_tpr']}")
    print(f"  per-mutation: {summary['per_mutation_tpr']}")


if __name__ == "__main__":
    main()
