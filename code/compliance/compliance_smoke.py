"""Compliance pipeline smoke test on n=2 CAD-generated drawings.

Validates:
  1. ColPali + Qwen-VL can be loaded together on one L40S (48GB).
  2. The 2 new drawings get embedded + indexed alongside v3 (1898+2 pages).
  3. ColPali retrieves the new drawings when asked compliance-flavored questions.
  4. Qwen-VL judge:
       - culvert (single-doc):  must rule COMPLIANT with WYDOT spec
       - guardrail (cross-doc): must rule COMPLIANT with WYDOT, NON-COMPLIANT with FDOT
  5. Logs every misjudgment with the violated rule for paper failure analysis.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---


import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, f"{PSR_ROOT}/qna_scale1k")
from _retriever import ColPaliRetriever, V3_INDEX  # noqa: E402

from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------- Config ----------
OUT_DIR = f"{PSR_ROOT}/compliance"
NEW_DRAWINGS = [
    {
        "image_path": f"{OUT_DIR}/mockups/sample_culvert.png",
        "agency":     "WYDOT",
        "plan_id":    "B-505 (M)",
        "sheet_title":"REINFORCED CONCRETE BOX CULVERT - SECTION",
        "compliance_case": "single_doc",
        "ground_truth": {"WYDOT": "compliant"},
        "violated_rule": None,
        "design_facts": {
            "concrete_class": "Class A-A",
            "rebar":          "#5 @ 9\" OC, Grade 60",
            "cover_in":       2.5,
            "span_ft":        8.0,
            "wall_in":        12,
        },
    },
    {
        "image_path": f"{OUT_DIR}/mockups/sample_guardrail_crossdoc.png",
        "agency":     "WYDOT",
        "plan_id":    "606-7B",
        "sheet_title":"STEEL POST GUARDRAIL FOUNDATION - DETAIL",
        "compliance_case": "cross_doc",
        "ground_truth": {"WYDOT": "compliant", "FDOT": "non_compliant"},
        "violated_rule": "FDOT requires 42\" minimum footing depth; drawing shows 36\".",
        "design_facts": {
            "post":           "W6x9 steel",
            "footing_depth_in": 36,
            "footing_dia_in":  12,
            "concrete_class":  "Class A",
        },
    },
]
QWEN_VL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"   # smoke uses 7B; production uses 72B
TOP_K = 5
TEMP_INDEX_PATH = f"{OUT_DIR}/temp_index_with_drawings.pt"


# ---------- Step 1: embed the 2 new drawings, build temp index ----------
def build_temp_index():
    print(f"[step1] embedding {len(NEW_DRAWINGS)} new drawings with ColPali")
    cp = ColPali.from_pretrained(
        "vidore/colpali-v1.2", torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    cpp = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    new_recs = []
    for d in NEW_DRAWINGS:
        img = Image.open(d["image_path"]).convert("RGB")
        with torch.no_grad():
            batch = cpp.process_images([img]).to(cp.device)
            emb = cp(**batch)[0].to(torch.bfloat16).cpu()
        new_recs.append({"embedding": emb, "metadata": {
            "image_path": d["image_path"], "filename": os.path.basename(d["image_path"]),
            "agency": d["agency"], "publication_year": "2025",
            "plan_id": d["plan_id"], "sheet_title": d["sheet_title"],
            "category": "Compliance Test", "keywords": [], "unique_id": d["plan_id"],
        }})
        print(f"  embedded {d['plan_id']}")

    print(f"[step1] loading v3 index ({V3_INDEX})")
    existing = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    combined = existing + new_recs
    torch.save(combined, TEMP_INDEX_PATH)
    print(f"[step1] temp index = {len(combined)} pages -> {TEMP_INDEX_PATH}")

    # Free ColPali GPU before loading Qwen-VL
    del cp
    torch.cuda.empty_cache()


# ---------- Step 2: retrieval test (refine-loop style, but using Qwen-VL-7B drafter) ----------
COMPLIANCE_PROMPTS = {
    "single_doc": (
        "Looking at this engineering standard plan, write ONE specific question "
        "an engineer would ask to check whether this design is compliant with "
        "the agency's standard plan requirements. Anchor the question with a "
        "specific labeled value visible on the page (a dimension, a material "
        "class, a rebar spec). Return ONLY the question."
    ),
    "cross_doc": (
        "Looking at this engineering standard plan, write ONE specific compliance "
        "question that would require comparing this design to MULTIPLE state DOT "
        "standards (e.g., would this footing depth meet both WYDOT and FDOT minimums?). "
        "Anchor with a specific labeled value visible on the page. Return ONLY the question."
    ),
}


def vlm_gen(model, processor, image, instruction, max_new=200):
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


def retrieve_top_k(retriever, query, k=TOP_K):
    return retriever.retrieve(query, k=k)


# ---------- Step 3: compliance judge ----------
JUDGE_PROMPT_SINGLE = """You are a compliance reviewer.

DRAWING SUMMARY: {summary}

RETRIEVED STANDARD REFERENCES (top {k}):
{refs}

Task: Decide if the DRAWING is compliant with the agency standard.

Output STRICT JSON:
{{
  "verdict": "COMPLIANT" or "NON_COMPLIANT",
  "rationale": "one short sentence citing the spec value(s) checked",
  "violated_rule": "the violated rule, or 'none'"
}}"""

JUDGE_PROMPT_CROSS = """You are a compliance reviewer evaluating a design against MULTIPLE state DOT standards.

DRAWING SUMMARY: {summary}

RETRIEVED STANDARD REFERENCES (top {k}):
{refs}

Task: For EACH agency referenced (WYDOT, FDOT, etc.), decide if the DRAWING meets that agency's minimum requirements.

Output STRICT JSON:
{{
  "per_agency": {{
    "WYDOT": {{"verdict": "COMPLIANT" or "NON_COMPLIANT", "rationale": "..."}},
    "FDOT":  {{"verdict": "COMPLIANT" or "NON_COMPLIANT", "rationale": "..."}}
  }},
  "violated_rule": "the violated rule(s), or 'none'"
}}"""


def parse_json(text):
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    t0 = time.time()
    # Step 1: embed + index
    build_temp_index()

    # Step 2: load Qwen-VL + retriever (now ColPali reloads inside ColPaliRetriever)
    print(f"[step2] loading Qwen-VL ({QWEN_VL_ID})")
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_VL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    qp = AutoProcessor.from_pretrained(QWEN_VL_ID, trust_remote_code=True)

    retriever = ColPaliRetriever(index_path=TEMP_INDEX_PATH)

    results = []
    for d in NEW_DRAWINGS:
        print(f"\n=== {d['plan_id']}  ({d['compliance_case']}) ===")
        img = Image.open(d["image_path"]).convert("RGB")

        # 2a. Generate a compliance-flavored question
        prompt = COMPLIANCE_PROMPTS[d["compliance_case"]]
        question = vlm_gen(qm, qp, img, prompt, max_new=120)
        question = question.split("\n")[0].strip().strip('"')
        print(f"[q] {question}")

        # 2b. Retrieve top-K and check whether the drawing itself is in top-K
        top = retrieve_top_k(retriever, question, k=10)
        target_rank = next(
            (i + 1 for i, m in enumerate(top) if m["image_path"] == d["image_path"]),
            None,
        )
        hit = target_rank is not None and target_rank <= TOP_K
        print(f"[retrieve] drawing-self rank={target_rank}  hit_top{TOP_K}={hit}")
        print("[retrieve] top5:")
        for i, m in enumerate(top[:TOP_K]):
            tag = "**SELF**" if m["image_path"] == d["image_path"] else ""
            print(f"  {i+1}. {m['agency']:8s} {m.get('plan_id','?'):14s} {m.get('sheet_title','?')[:50]}  {tag}")

        # 2c. Build judge prompt with retrieved refs + drawing facts summary
        summary = (
            f"plan_id={d['plan_id']}  agency={d['agency']}  "
            f"title={d['sheet_title']}\n  facts={json.dumps(d['design_facts'])}"
        )
        refs_text = "\n".join(
            f"  ref{i+1}: {m['agency']:8s} {m.get('plan_id','?'):14s} {m.get('sheet_title','?')[:60]}"
            for i, m in enumerate(top[:TOP_K])
        )
        judge_prompt = (JUDGE_PROMPT_CROSS if d["compliance_case"] == "cross_doc" else JUDGE_PROMPT_SINGLE).format(
            summary=summary, refs=refs_text, k=TOP_K
        )
        verdict_raw = vlm_gen(qm, qp, img, judge_prompt, max_new=400)
        verdict = parse_json(verdict_raw)
        print(f"[judge raw] {verdict_raw[:400]}")
        print(f"[judge parsed] {verdict}")

        # 2d. Compare against ground truth
        gt = d["ground_truth"]
        if d["compliance_case"] == "single_doc":
            agent_v = (verdict or {}).get("verdict", "?")
            expected = gt.get("WYDOT", "?")
            correct = (agent_v == "COMPLIANT" and expected == "compliant") or \
                      (agent_v == "NON_COMPLIANT" and expected == "non_compliant")
        else:
            per = (verdict or {}).get("per_agency", {})
            correct_per = []
            for a, exp in gt.items():
                v = per.get(a, {}).get("verdict", "?")
                correct_per.append(
                    (v == "COMPLIANT" and exp == "compliant") or
                    (v == "NON_COMPLIANT" and exp == "non_compliant")
                )
            correct = all(correct_per)
        print(f"[score] judge_correct={correct}")

        results.append({
            "plan_id": d["plan_id"],
            "case": d["compliance_case"],
            "question": question,
            "retrieval_rank": target_rank,
            "retrieval_hit_top5": hit,
            "top5": [
                {"rank": i+1, "agency": m["agency"], "plan_id": m.get("plan_id"),
                 "sheet_title": m.get("sheet_title"), "is_self": m["image_path"] == d["image_path"]}
                for i, m in enumerate(top[:TOP_K])
            ],
            "judge_verdict": verdict,
            "judge_correct": correct,
            "ground_truth": gt,
            "violated_rule": d.get("violated_rule"),
        })

    out_path = f"{OUT_DIR}/compliance_smoke_report.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "elapsed_sec": time.time() - t0}, f, indent=2)
    print(f"\n[done] wrote {out_path}  elapsed={(time.time()-t0)/60:.1f} min")
    n_hit = sum(int(r["retrieval_hit_top5"]) for r in results)
    n_correct = sum(int(r["judge_correct"]) for r in results)
    print(f"[summary] retrieval_hit={n_hit}/{len(results)}  judge_correct={n_correct}/{len(results)}")


if __name__ == "__main__":
    main()
