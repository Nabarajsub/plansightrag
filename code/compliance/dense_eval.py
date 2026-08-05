"""Path A (72B + CoT + per-drawing thresholds) on the dense4 manifest.

Same prompt template as judge_only_n10_v2.py, but builds its own retrieval
in-process (compliance_n10_report.json doesn't cover dense4 drawings).

Memory strategy (avoids the earlier OOM):
  1. Load ColPali on cuda:1.
  2. Embed dense drawings + load existing index in-mem (1898+4 = 1902 records).
  3. Generate a generic compliance retrieval query PER archetype (no VLM needed).
  4. Retrieve top-5 per drawing.
  5. FREE ColPali + empty cache.
  6. Load Qwen-VL-72B 4-bit on full GPU.
  7. For each drawing: build CoT+thresholds prompt + judge.
  8. Score, write report.
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
import sys
import time

import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, f"{PSR_ROOT}/compliance")
from compliance_smoke_n10 import parse_json, vlm_gen  # noqa
from judge_only_n10_v2 import derive_checks, build_judge_prompt  # noqa

V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"
TOP_K = 5

# Per-archetype retrieval queries
RETRIEVAL_Q = {
    "draw_culvert":   "concrete box culvert section reinforcement cover requirements",
    "draw_guardrail": "steel post guardrail foundation footing depth requirements",
    "draw_rebar":     "reinforced concrete beam stirrup spacing reinforcement detail",
    "draw_inlet":     "curb drainage inlet grate opening size requirements",
    "draw_sign_post": "breakaway sign post foundation anchor bolt embedment",
}


def stage1_retrieve(manifest, cp_device="cuda"):
    print(f"[step1] loading ColPali on {cp_device}")
    cp = ColPali.from_pretrained(
        "vidore/colpali-v1.2", torch_dtype=torch.bfloat16, device_map=cp_device
    ).eval()
    cpp = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    new_recs = []
    print(f"[step1] embedding {len(manifest)} drawings")
    for d in manifest:
        img = Image.open(d["image_path"]).convert("RGB")
        with torch.no_grad():
            batch = cpp.process_images([img]).to(cp_device)
            emb = cp(**batch)[0].to(torch.bfloat16)
        new_recs.append({"embedding": emb, "metadata": {
            "image_path": d["image_path"], "filename": os.path.basename(d["image_path"]),
            "agency": d["agency"], "plan_id": d["plan_id"],
            "sheet_title": d.get("design_facts", {}).get("sheet_title", "Compliance Test"),
            "category": "Compliance", "keywords": [], "unique_id": d["plan_id"],
        }})

    print(f"[step1] loading v3 index records (cpu, then push to {cp_device})")
    existing = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    db = []
    for r in existing:
        db.append({"embedding": r["embedding"].to(cp_device).to(torch.bfloat16),
                   "metadata": r["metadata"]})
    db.extend(new_recs)
    print(f"[step1] in-mem index: {len(db)}")

    # Retrieve per drawing
    retrievals = {}
    for d in manifest:
        q = RETRIEVAL_Q.get(d["archetype"], "engineering compliance standard plan")
        inputs = cpp.process_queries([q]).to(cp_device)
        with torch.no_grad():
            q_emb = cp(**inputs)
        scores = []
        for doc in db:
            inter = torch.matmul(q_emb, doc["embedding"].T)
            scores.append(inter.max(dim=-1).values.sum(dim=-1).item())
        top_idx = np.argsort(scores)[::-1][:max(TOP_K, 10)]
        top = [db[i]["metadata"] for i in top_idx]
        target_rank = next((i + 1 for i, m in enumerate(top) if m["image_path"] == d["image_path"]), None)
        retrievals[d["name"]] = {
            "question": q,
            "retrieval_self_rank": target_rank,
            "retrieval_top5_hit": bool(target_rank is not None and target_rank <= TOP_K),
            "top5": [{"rank": i+1, "agency": m["agency"], "plan_id": m.get("plan_id"),
                      "is_self": m["image_path"] == d["image_path"]}
                     for i, m in enumerate(top[:TOP_K])],
        }
        print(f"  [retrieve] {d['name']:24s} rank={target_rank} top5_hit={retrievals[d['name']]['retrieval_top5_hit']}")

    # Free ColPali + index from GPU
    print(f"[step1] freeing ColPali GPU memory")
    del cp, cpp, db, existing
    gc.collect()
    torch.cuda.empty_cache()
    return retrievals


def resize_for_vlm(img, max_side=1280):
    """Cap longest side to limit visual tokens; keeps aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    if w >= h:
        new_w = max_side; new_h = int(h * max_side / w)
    else:
        new_h = max_side; new_w = int(w * max_side / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _single_gpu():
    return torch.cuda.is_available() and torch.cuda.device_count() == 1


def stage2_judge(manifest, retrievals, judge_model_id, use_4bit, out_report, out_fail):
    print(f"[step2] loading judge VLM {judge_model_id}")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(judge_model_id, **kwargs).eval()
    qp = AutoProcessor.from_pretrained(judge_model_id, trust_remote_code=True)

    results = []
    fail_f = open(out_fail, "w")
    for d in manifest:
        name = d["name"]
        ret = retrievals[name]
        print(f"\n=== {name}  gt={'COMPLIANT' if d['ground_truth_compliant'] else 'NON_COMPLIANT'} ===")
        img = resize_for_vlm(Image.open(d["image_path"]).convert("RGB"))
        checks = derive_checks(d)
        prompt = build_judge_prompt(d, ret["top5"], checks)
        verdict_raw = vlm_gen(qm, qp, img, prompt, max_new=500)
        verdict = parse_json(verdict_raw)
        agent_v = (verdict or {}).get("verdict", "?")
        expected_v = "COMPLIANT" if d["ground_truth_compliant"] else "NON_COMPLIANT"
        correct = agent_v == expected_v
        print(f"[judge] {agent_v} (expected {expected_v}) correct={correct}")
        if verdict and verdict.get("checks"):
            for c in verdict["checks"]:
                print(f'    {c.get("name","?")[:30]:30s} {c.get("compare","?")[:50]:50s} -> {c.get("result","?")}')
        rec = {
            "name": name, "plan_id": d["plan_id"], "archetype": d["archetype"],
            "ground_truth_compliant": d["ground_truth_compliant"],
            "expected_violated_rule": d.get("violated_rule"),
            "checks_injected": checks,
            "retrieval_self_rank": ret["retrieval_self_rank"],
            "retrieval_top5_hit": ret["retrieval_top5_hit"],
            "top5": ret["top5"],
            "judge_verdict": verdict, "judge_correct": correct,
            "judge_raw": verdict_raw[:1800],
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
        "results": results,
    }
    with open(out_report, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {out_report}")
    print(f"[summary] retrieval@5={n_retrieve}/{n}  judge={n_judge}/{n}  "
          f"(comp {n_jc}/{n_comp}, noncomp {n_jn}/{n - n_comp})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--judge_model_id", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--report_path", required=True)
    args = ap.parse_args()

    t0 = time.time()
    manifest = json.load(open(args.manifest))
    print(f"[main] {len(manifest)} drawings  judge={args.judge_model_id}")

    retrievals = stage1_retrieve(manifest)
    out_fail = args.report_path.replace(".json", "_failures.jsonl")
    stage2_judge(manifest, retrievals, args.judge_model_id, args.use_4bit, args.report_path, out_fail)

    print(f"[main] total elapsed = {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
