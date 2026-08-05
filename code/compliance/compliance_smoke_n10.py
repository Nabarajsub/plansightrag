"""Compliance smoke on n=10 drawings (5 archetypes x compliant/non-compliant).

Reads manifest.json, embeds drawings into v3 index, runs question-generation +
retrieval + Qwen-VL judge for each, scores against ground truth, writes report.
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

sys.path.insert(0, f"{PSR_ROOT}/qna_scale1k")
from _retriever import ColPaliRetriever, V3_INDEX  # noqa: E402

from colpali_engine.models import ColPali, ColPaliProcessor

OUT_DIR = f"{PSR_ROOT}/compliance"
MANIFEST = f"{OUT_DIR}/mockups_n10/manifest.json"

TOP_K = 5

# ---- prompts ----
COMPLIANCE_Q = (
    "Looking at this engineering standard plan, write ONE specific compliance "
    "question an engineer would ask to verify whether this design meets the "
    "required minimums/maximums (e.g., minimum cover, maximum spacing, minimum "
    "embedment). Anchor the question with the specific labeled value visible on "
    "the page. Return ONLY the question."
)

JUDGE_RULES = """Engineering compliance rules to consider:
- ACI 318 / AASHTO LRFD: minimum concrete cover for embedded reinforcement >= 2 inches for cast-in-place exposed surfaces.
- ACI 318: stirrup spacing in beams should not exceed d/2 (half the effective depth).
- WYDOT 606.05: guardrail post footing min depth 30 inches.
- FDOT: utility pole / guardrail post foundation min footing depth 42 inches.
- AASHTO / standard practice: sign post anchor bolt minimum embedment 12 inches.
- PROWAG / ADA: drainage grate openings <= 4 inches for vehicular surfaces to prevent bicycle/wheelchair wheel entrapment."""

JUDGE_PROMPT = """You are a compliance reviewer.

DRAWING SUMMARY:
{summary}

RETRIEVED STANDARD REFERENCES (top {k}):
{refs}

""" + JUDGE_RULES + """

Task: Decide if the DRAWING is COMPLIANT or NON_COMPLIANT.

Output STRICT JSON:
{{
  "verdict": "COMPLIANT" or "NON_COMPLIANT",
  "rationale": "one short sentence citing the specific spec value(s) checked",
  "violated_rule": "the violated rule, or 'none'"
}}"""

JUDGE_PROMPT_COT = """You are a compliance reviewer. Reason STEP BY STEP.

DRAWING SUMMARY:
{summary}

RETRIEVED STANDARD REFERENCES (top {k}):
{refs}

""" + JUDGE_RULES + """

Follow this procedure EXACTLY:
  Step 1. List every numeric value visible (dimension, spacing, depth, embedment, cover, grade).
  Step 2. For each value, identify the applicable compliance rule from the list above.
  Step 3. For each (value, rule), perform the numeric comparison explicitly. State "value X vs requirement Y -> PASS" or "FAIL".
          Be careful with inequality direction (minimum vs maximum).
  Step 4. If ANY check is FAIL, the verdict is NON_COMPLIANT. Otherwise COMPLIANT.

Output STRICT JSON:
{{
  "extracted_values": [{{"name": "...", "value": "..."}}],
  "checks": [{{"value": "...", "rule": "...", "compare": "X vs Y", "result": "PASS or FAIL"}}],
  "verdict": "COMPLIANT" or "NON_COMPLIANT",
  "rationale": "one short sentence",
  "violated_rule": "the violated rule, or 'none'"
}}"""


def vlm_gen(model, processor, image, instruction, max_new=400):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": instruction},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_index_and_keep_colpali(manifest, cp_device="cuda:1"):
    """Load ColPali on cp_device, embed new drawings, return (cp_model, cp_proc, records).
    Keeps ColPali alive so retrieval doesn't need a second load (which would OOM after
    the big judge model has filled cuda:0).
    """
    import numpy as np
    print(f"[step1] loading ColPali on {cp_device}")
    cp = ColPali.from_pretrained(
        "vidore/colpali-v1.2", torch_dtype=torch.bfloat16, device_map=cp_device
    ).eval()
    cpp = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    print(f"[step1] embedding {len(manifest)} drawings")
    new_recs = []
    for d in manifest:
        img = Image.open(d["image_path"]).convert("RGB")
        with torch.no_grad():
            batch = cpp.process_images([img]).to(cp_device)
            emb = cp(**batch)[0].to(torch.bfloat16)
        new_recs.append({"embedding": emb, "metadata": {
            "image_path": d["image_path"],
            "filename": os.path.basename(d["image_path"]),
            "agency": d["agency"], "publication_year": "2025",
            "plan_id": d["plan_id"],
            "sheet_title": d.get("design_facts", {}).get("sheet_title", "COMPLIANCE TEST"),
            "category": "Compliance Test", "keywords": [],
            "unique_id": d["plan_id"],
        }})
    print(f"[step1] loading v3 index")
    existing = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    # Move existing embeddings to cp_device for fast MaxSim
    db = []
    for r in existing:
        db.append({"embedding": r["embedding"].to(cp_device).to(torch.bfloat16), "metadata": r["metadata"]})
    db.extend(new_recs)
    print(f"[step1] in-mem index = {len(db)} pages")
    return cp, cpp, db


def inline_retrieve(cp_model, cp_proc, db, query, k=10):
    import numpy as np
    inputs = cp_proc.process_queries([query]).to(cp_model.device)
    with torch.no_grad():
        q_emb = cp_model(**inputs)
    scores = []
    for doc in db:
        inter = torch.matmul(q_emb, doc["embedding"].T)
        scores.append(inter.max(dim=-1).values.sum(dim=-1).item())
    top_idx = np.argsort(scores)[::-1][:k]
    return [db[i]["metadata"] for i in top_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--use_cot", action="store_true")
    ap.add_argument("--use_4bit", action="store_true", help="load judge in 4-bit NF4")
    ap.add_argument("--report_suffix", default="default")
    args = ap.parse_args()

    REPORT = f"{OUT_DIR}/compliance_n10_report_{args.report_suffix}.json"
    FAILURES = f"{OUT_DIR}/compliance_n10_failures_{args.report_suffix}.jsonl"

    t0 = time.time()
    manifest = json.load(open(MANIFEST))
    print(f"[main] {len(manifest)} drawings  judge={args.judge_model_id}  cot={args.use_cot}  4bit={args.use_4bit}")

    # ColPali pinned to cuda:1 (small ~6GB). Judge VLM gets cuda:0 mostly.
    cp_model, cp_proc, db = build_index_and_keep_colpali(manifest, cp_device="cuda:1")

    print(f"[step2] loading judge VLM ({args.judge_model_id})")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto",
                  max_memory={0: "22GiB", 1: "15GiB", "cpu": "60GiB"})
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
    for d in manifest:
        print(f"\n=== {d['name']}  gt={'COMPLIANT' if d['ground_truth_compliant'] else 'NON_COMPLIANT'} ===")
        img = Image.open(d["image_path"]).convert("RGB")

        question = vlm_gen(qm, qp, img, COMPLIANCE_Q, max_new=128).split("\n")[0].strip().strip('"')
        print(f"[q] {question}")

        top = inline_retrieve(cp_model, cp_proc, db, question, k=10)
        target_rank = next((i + 1 for i, m in enumerate(top) if m["image_path"] == d["image_path"]), None)
        hit5 = target_rank is not None and target_rank <= TOP_K
        print(f"[retrieve] self_rank={target_rank}  top{TOP_K}_hit={hit5}")
        top5_meta = [{"rank": i+1, "agency": m["agency"], "plan_id": m.get("plan_id"),
                      "is_self": m["image_path"] == d["image_path"]} for i, m in enumerate(top[:TOP_K])]

        summary = (f"plan_id={d['plan_id']}  agency={d['agency']}\n"
                   f"design_facts={json.dumps(d['design_facts'])}")
        refs = "\n".join(f"  ref{i+1}: {m['agency']:8s} {m.get('plan_id','?'):14s} {m.get('sheet_title','?')[:60]}"
                         for i, m in enumerate(top[:TOP_K]))
        verdict_raw = vlm_gen(qm, qp, img, judge_template.format(summary=summary, refs=refs, k=TOP_K),
                              max_new=600 if args.use_cot else 400)
        verdict = parse_json(verdict_raw)
        agent_v = (verdict or {}).get("verdict", "?")
        expected_v = "COMPLIANT" if d["ground_truth_compliant"] else "NON_COMPLIANT"
        correct = agent_v == expected_v
        print(f"[judge] agent={agent_v}  expected={expected_v}  correct={correct}")
        if verdict:
            print(f"[rationale] {verdict.get('rationale','?')}")

        rec = {
            "name": d["name"], "plan_id": d["plan_id"],
            "ground_truth_compliant": d["ground_truth_compliant"],
            "expected_violated_rule": d.get("violated_rule"),
            "question": question,
            "retrieval_self_rank": target_rank, "retrieval_top5_hit": hit5,
            "top5": top5_meta,
            "judge_verdict": verdict, "judge_correct": correct,
        }
        results.append(rec)
        if not correct:
            fail_f.write(json.dumps(rec) + "\n")
    fail_f.close()

    # Aggregate
    n = len(results)
    n_retrieve = sum(int(r["retrieval_top5_hit"]) for r in results)
    n_judge = sum(int(r["judge_correct"]) for r in results)
    n_comp = sum(1 for r in results if r["ground_truth_compliant"])
    n_noncomp = n - n_comp
    n_judge_on_comp = sum(int(r["judge_correct"]) for r in results if r["ground_truth_compliant"])
    n_judge_on_noncomp = sum(int(r["judge_correct"]) for r in results if not r["ground_truth_compliant"])

    summary = {
        "n_total": n,
        "retrieval_top5_pct": 100 * n_retrieve / n,
        "judge_correct_pct": 100 * n_judge / n,
        "judge_correct_on_compliant": f"{n_judge_on_comp}/{n_comp}",
        "judge_correct_on_noncompliant": f"{n_judge_on_noncomp}/{n_noncomp}",
        "elapsed_min": (time.time() - t0) / 60,
        "results": results,
    }
    with open(REPORT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {REPORT}  failures -> {FAILURES}")
    print(f"[summary] retrieve@5={n_retrieve}/{n}  judge_correct={n_judge}/{n}  "
          f"(compliant {n_judge_on_comp}/{n_comp}, noncompliant {n_judge_on_noncomp}/{n_noncomp})")


if __name__ == "__main__":
    main()
