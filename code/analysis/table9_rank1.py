"""Verdict accuracy conditioned on rank-1 retrieval (R1.7).

Same frozen question set, same shared ColNomic-3B retrieval pass and same
deployed judge recipe as table9_frozen.py, but the auditor is shown the RANK-1
RETRIEVED page instead of the supplied target sheet, with design values withheld.
On cases where rank 1 is the target sheet the input is identical to the withheld
arm; on the others the judge sees a different sheet. Verdict accuracy is then
reported split by whether rank 1 was the target sheet.

    python table9_rank1.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

import torch
from PIL import Image

# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---
ROOT = CLUSTER_ROOT
COMP = f"{ROOT}/compliance"
MANIFEST = f"{COMP}/table9_n100/manifest.json"
CACHE = f"{ROOT}/baselines_v2/cache_colnomic_pages.pt"
OUT = f"{ROOT}/baselines_v2/reports/_table9_rank1.json"
COLNOMIC = "nomic-ai/colnomic-embed-multimodal-3b"
TOP_K = 5
Image.MAX_IMAGE_PIXELS = None

MODELS = {"7b": "Qwen/Qwen2.5-VL-7B-Instruct", "72b": "Qwen/Qwen2.5-VL-72B-Instruct"}

JUDGE_RULES = """Engineering compliance rules to consider:
- ACI 318 / AASHTO LRFD: minimum concrete cover for embedded reinforcement >= 2 inches for cast-in-place exposed surfaces.
- ACI 318: stirrup spacing in beams should not exceed d/2 (half the effective depth).
- WYDOT 606.05: guardrail post footing min depth 30 inches.
- FDOT: utility pole / guardrail post foundation min footing depth 42 inches.
- AASHTO / standard practice: sign post anchor bolt minimum embedment 12 inches.
- PROWAG / ADA: drainage grate openings <= 4 inches for vehicular surfaces to prevent bicycle/wheelchair wheel entrapment."""

PLAIN = """You are a compliance reviewer.

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

COT = """You are a compliance reviewer. Reason STEP BY STEP.

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


def derive_checks(d):
    """Verbatim from judge_only_n10_v2.derive_checks -- per-drawing resolved thresholds."""
    facts, arch = d["design_facts"], d["archetype"]
    if arch == "draw_culvert":
        return [{"name": "Concrete cover", "rule": "ACI 318 / AASHTO LRFD", "operator": ">=",
                 "threshold": 2.0, "unit": "inches",
                 "hint": "labeled 'COVER = X\" MIN' near the wall hatching"}]
    if arch == "draw_guardrail":
        return [{"name": "Footing depth", "rule": "WYDOT 606.05", "operator": ">=",
                 "threshold": 30.0, "unit": "inches",
                 "hint": "labeled 'FOOTING DEPTH' on the vertical dimension below ground line"}]
    if arch == "draw_rebar":
        bh = facts.get("beam_h_in", 24)
        return [{"name": "Stirrup spacing", "rule": f"ACI 318 max = d/2 = {bh//2}\"",
                 "operator": "<=", "threshold": float(bh) / 2, "unit": "inches",
                 "hint": "labeled '#3 STIRRUPS @ X\" OC' on the leader callout"},
                {"name": "Concrete cover", "rule": "ACI 318 / AASHTO LRFD", "operator": ">=",
                 "threshold": 2.0, "unit": "inches",
                 "hint": "labeled 'COVER = X\"' on the leader callout"}]
    if arch == "draw_inlet":
        return [{"name": "Grate opening", "rule": "PROWAG / ADA (vehicular surfaces)",
                 "operator": "<=", "threshold": 4.0, "unit": "inches",
                 "hint": "labeled 'GRATE OPENING = X\"' on the leader callout"}]
    if arch == "draw_sign_post":
        return [{"name": "Anchor bolt embedment", "rule": "AASHTO / std spec", "operator": ">=",
                 "threshold": 12.0, "unit": "inches",
                 "hint": "labeled 'ANCHOR EMBED' on the vertical dimension below ground line"}]
    return []


def thresh_prompt(summary, refs, checks):
    checks_text = "\n".join(
        f"  Check {i+1}: {c['name']} (rule: {c['rule']})\n"
        f"    - Look at the drawing for the value, hint: {c['hint']}\n"
        f"    - The value must be {c['operator']} {c['threshold']} {c['unit']}"
        for i, c in enumerate(checks))
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


# verbatim from agentic_n10.ARCH_LABEL -- the frozen queries must be byte-identical
# to the agentic configuration's, which is the set already known to be 100% aligned
ARCH_LABEL = {"draw_culvert": "reinforced concrete box culvert section",
              "draw_guardrail": "steel post guardrail foundation",
              "draw_rebar": "reinforced concrete beam rebar detail",
              "draw_inlet": "curb drainage inlet",
              "draw_sign_post": "breakaway sign post foundation"}


def compose_query(d, checks):
    """The agentic configuration's deterministic query -- the frozen question set."""
    label = ARCH_LABEL.get(d["archetype"], "engineering detail")
    names = ", ".join(c["name"] for c in checks)
    return (f"Verify whether the {label} in plan {d['plan_id']} (agency {d['agency']}) "
            f"meets the following compliance requirements: {names}.")


def parse_json(t):
    m = re.search(r"\{.*\}", t or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, (c - h) * 100), 2), round(min(100.0, (c + h) * 100), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="72b_cot_thresh")
    ap.add_argument("--arms", default="rank1")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    cfgs = a.configs.split(",")
    arms = a.arms.split(",")

    manifest = json.load(open(MANIFEST))
    for d in manifest:
        d["_checks"] = derive_checks(d)
        d["_query"] = compose_query(d, d["_checks"])

    # ---------- retrieval once, on the adopted backbone, shared by all ----------
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    print("[t9] retrieval pass on ColNomic-3B (shared by every configuration)", flush=True)
    ret = ColQwen2_5.from_pretrained(COLNOMIC, torch_dtype=torch.bfloat16,
                                     device_map="cuda").eval()
    rp = ColQwen2_5_Processor.from_pretrained(COLNOMIC)
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    paths, pembs = list(cache["paths"]), list(cache["pembs"])
    meta = [{"agency": "?", "plan_id": os.path.basename(p)[:20]} for p in paths]
    for d in manifest:
        img = Image.open(d["image_path"]).convert("RGB"); img.thumbnail((1400, 1400))
        with torch.no_grad():
            e = ret(**rp.process_images([img]).to("cuda"))[0].to(torch.float16).cpu()
        pembs.append(e); paths.append(d["image_path"])
        meta.append({"agency": d["agency"], "plan_id": d["plan_id"], "_self": d["name"]})
    mock_start = len(paths) - len(manifest)
    for qi, d in enumerate(manifest):
        with torch.no_grad():
            Q = ret(**rp.process_queries([d["_query"]]).to("cuda"))[0].float()
        sc = torch.empty(len(pembs))
        for i, P in enumerate(pembs):
            sc[i] = (Q @ P.to("cuda").float().T).max(dim=1).values.sum().item()
        top = torch.topk(sc, TOP_K).indices.tolist()
        gold = mock_start + qi
        d["_top5"] = [meta[t] for t in top]
        d["_rank1_path"] = paths[top[0]]
        d["_rank"] = (top.index(gold) + 1) if gold in top else None
        d["_hit5"] = gold in top
        if (qi + 1) % 25 == 0:
            print(f"    retrieval {qi+1}/{len(manifest)}", flush=True)
    n_hit = sum(1 for d in manifest if d["_hit5"])
    print(f"[t9] frozen-question retrieval R@5 = {n_hit}/{len(manifest)} "
          f"({n_hit/len(manifest)*100:.2f}%)", flush=True)
    del ret, pembs
    torch.cuda.empty_cache()

    # ---------- judge arms ----------
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    results = {}
    loaded = None
    for cfg in cfgs:
        size = "72b" if cfg.startswith("72b") else "7b"
        if loaded != size:
            if loaded:
                del jm, jp
                torch.cuda.empty_cache()
            print(f"[t9] loading {MODELS[size]}", flush=True)
            jm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODELS[size], quantization_config=bnb, device_map="auto",
                torch_dtype=torch.bfloat16).eval()
            jp = AutoProcessor.from_pretrained(MODELS[size], trust_remote_code=True)
            loaded = size

        for arm in arms:
            recs = []
            for k, d in enumerate(manifest, 1):
                summary = f"plan_id={d['plan_id']}  agency={d['agency']}\n"
                if arm == "supplied":
                    summary += f"design_facts={json.dumps(d['design_facts'])}"
                elif arm in ("withheld", "rank1"):
                    summary += ("design_facts=<withheld: read every value from the "
                                "drawing itself>")
                refs = "\n".join(f"  ref{i+1}: {x.get('agency','?'):8s} "
                                 f"{str(x.get('plan_id','?')):14s}"
                                 for i, x in enumerate(d["_top5"]))
                if cfg.endswith("cot_thresh"):
                    prompt = thresh_prompt(summary, refs, d["_checks"])
                elif cfg.endswith("cot"):
                    prompt = COT.format(summary=summary, refs=refs, k=TOP_K)
                else:
                    prompt = PLAIN.format(summary=summary, refs=refs, k=TOP_K)

                img = Image.open(d["_rank1_path"] if arm == "rank1" else d["image_path"]).convert("RGB")
                img.thumbnail((1400, 1400))
                msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                     {"type": "text", "text": prompt}]}]
                txt = jp.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = jp(text=[txt], images=[img], padding=True,
                         return_tensors="pt").to(jm.device)
                with torch.no_grad():
                    o = jm.generate(**inp, max_new_tokens=600 if "cot" in cfg else 400,
                                    do_sample=False)
                raw = jp.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()
                v = parse_json(raw)
                exp = "COMPLIANT" if d["ground_truth_compliant"] else "NON_COMPLIANT"
                recs.append({"name": d["name"], "plan_id": d["plan_id"],
                             "ground_truth_compliant": d["ground_truth_compliant"],
                             "expected_violated_rule": d.get("violated_rule"),
                             "question": d["_query"], "checks_injected": d["_checks"],
                             "retrieval_self_rank": d["_rank"],
                             "rank1_is_target": d["_rank"] == 1,
                             "image_shown": d["_rank1_path"] if arm == "rank1" else d["image_path"],
                             "retrieval_top5_hit": d["_hit5"], "top5": d["_top5"],
                             "judge_verdict": v, "judge_raw": raw,
                             "judge_correct": (v or {}).get("verdict") == exp})
                if k % 20 == 0:
                    nc = sum(r["judge_correct"] for r in recs)
                    print(f"    {cfg}/{arm} {k}/{len(manifest)} correct={nc}", flush=True)
                    torch.cuda.empty_cache()
            kk = sum(r["judge_correct"] for r in recs)
            results[f"{cfg}::{arm}"] = {
                "config": cfg, "arm": arm, "k": kk, "n": len(recs),
                "accuracy": round(kk / len(recs) * 100, 2),
                "wilson95": wilson(kk, len(recs)), "records": recs}
            print(f"  [done] {cfg}/{arm}: {kk}/{len(recs)} = {kk/len(recs)*100:.2f}%",
                  flush=True)
            json.dump({"note": __doc__.split("\n")[0], "frozen_question_set": True,
                       "retrieval_backbone": COLNOMIC,
                       "retrieval_recall5": round(n_hit / len(manifest) * 100, 2),
                       "elapsed_min": round((time.time() - t0) / 60, 1),
                       "arms": results}, open(a.out, "w"), indent=1)

    print(f"\n{'=' * 76}\nTABLE 9, FROZEN QUESTION SET\n{'=' * 76}")
    print(f"  retrieval (shared, ColNomic-3B, frozen queries): "
          f"{n_hit/len(manifest)*100:.2f}%\n")
    print(f"  {'config':<18}{'facts supplied':>18}{'facts withheld':>18}{'gap':>10}")
    print("  " + "-" * 62)
    for cfg in cfgs:
        s = results.get(f"{cfg}::supplied"); w = results.get(f"{cfg}::withheld")
        fs = f"{s['accuracy']:.0f}% ({s['k']}/{s['n']})" if s else "—"
        fw = f"{w['accuracy']:.0f}% ({w['k']}/{w['n']})" if w else "—"
        gap = f"{s['accuracy'] - w['accuracy']:+.0f} pp" if (s and w) else "—"
        print(f"  {cfg:<18}{fs:>18}{fw:>18}{gap:>10}")
    print(f"\n  wrote {a.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
