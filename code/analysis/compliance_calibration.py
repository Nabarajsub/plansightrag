"""Calibration and abstention for the compliance judge (R2 statistics 6).

Reviewer 2 asks for calibration alongside sensitivity, specificity, precision and
recall. The other five are recoverable from the existing logs; calibration is not,
because every prompt in the study asks only for

    {"verdict": ..., "rationale": ..., "violated_rule": ...}

with no confidence and no abstention option. A model that is always forced to
commit cannot be calibrated, and reporting a reliability curve from post-hoc
guesses would be worse than reporting none.

So this re-runs the judge with two additions to the output schema and nothing
else changed:

    "confidence": integer 0-100
    "verdict":    COMPLIANT | NON_COMPLIANT | UNCERTAIN

Everything else -- the frozen question set, the shared ColNomic-3B retrieval, the
prompts, the decoding settings -- is identical to `table9_frozen.py`, so the
verdict column is directly comparable and any drift is attributable to the schema
change alone.

It runs on the **facts-withheld** arm. Calibration is only meaningful where the
model actually makes mistakes, and in the supplied arm the deployed recipe scores
100%, which would give a degenerate curve.

Reported:
  ECE     expected calibration error, 10 equal-width bins
  MCE     maximum calibration error over non-empty bins
  Brier   mean squared error of the confidence-as-probability
  reliability table: per-bin confidence vs observed accuracy
  abstention rate, and accuracy on the cases the model did commit to

    python compliance_calibration.py --configs 7b_cot,72b_plain,72b_cot_thresh
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

# --- release path resolution ---------------------------------------------
# PSR_ROOT: this repository's root (defaults to two levels up from this file).
# CLUSTER_ROOT: the full evaluation tree, for artefacts too large to ship
# (the page-embedding cache and the drawing manifest).
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---
ROOT = CLUSTER_ROOT
COMP = f"{ROOT}/compliance"
MANIFEST = f"{COMP}/table9_n100/manifest.json"
CACHE = f"{ROOT}/baselines_v2/cache_colnomic_pages.pt"
OUT = f"{PSR_ROOT}/reports/analysis/_compliance_calibration.json"
COLNOMIC = "nomic-ai/colnomic-embed-multimodal-3b"
TOP_K = 5
N_BINS = 10
Image.MAX_IMAGE_PIXELS = None
MODELS = {"7b": "Qwen/Qwen2.5-VL-7B-Instruct", "72b": "Qwen/Qwen2.5-VL-72B-Instruct"}

# reuse the frozen-run definitions verbatim so the two studies stay comparable
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("t9f", f"{COMP}/table9_frozen.py")
_t9 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_t9)
derive_checks, compose_query, parse_json, wilson = (
    _t9.derive_checks, _t9.compose_query, _t9.parse_json, _t9.wilson)
JUDGE_RULES = _t9.JUDGE_RULES

CONF_CLAUSE = """
Also report how confident you are, and say so honestly:
  "confidence": an integer 0-100. 100 means certain; 50 means a coin flip.
  If the drawing does not let you determine the value, set "verdict" to
  "UNCERTAIN" rather than guessing.
"""

PLAIN = """You are a compliance reviewer.

DRAWING SUMMARY:
{summary}

RETRIEVED STANDARD REFERENCES (top {k}):
{refs}

""" + JUDGE_RULES + CONF_CLAUSE + """
Task: Decide if the DRAWING is COMPLIANT, NON_COMPLIANT, or UNCERTAIN.

Output STRICT JSON:
{{
  "verdict": "COMPLIANT" or "NON_COMPLIANT" or "UNCERTAIN",
  "confidence": 0-100,
  "rationale": "one short sentence citing the specific spec value(s) checked",
  "violated_rule": "the violated rule, or 'none'"
}}"""

COT = """You are a compliance reviewer. Reason STEP BY STEP.

DRAWING SUMMARY:
{summary}

RETRIEVED STANDARD REFERENCES (top {k}):
{refs}

""" + JUDGE_RULES + CONF_CLAUSE + """
Follow this procedure EXACTLY:
  Step 1. List every numeric value visible (dimension, spacing, depth, embedment, cover, grade).
  Step 2. For each value, identify the applicable compliance rule from the list above.
  Step 3. For each (value, rule), perform the numeric comparison explicitly.
  Step 4. If ANY check is FAIL, the verdict is NON_COMPLIANT. If a required value
          cannot be read from the drawing, the verdict is UNCERTAIN.

Output STRICT JSON:
{{
  "checks": [{{"value": "...", "rule": "...", "compare": "X vs Y", "result": "PASS or FAIL"}}],
  "verdict": "COMPLIANT" or "NON_COMPLIANT" or "UNCERTAIN",
  "confidence": 0-100,
  "rationale": "one short sentence",
  "violated_rule": "the violated rule, or 'none'"
}}"""


def thresh_prompt(summary, refs, checks):
    ct = "\n".join(
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
{ct}
{CONF_CLAUSE}
Procedure:
  Step 1. For EACH check, EXTRACT the actual value visible on the drawing (read the label).
  Step 2. Perform the numeric comparison explicitly.
  Step 3. If ANY check is FAIL the verdict is NON_COMPLIANT; if a value cannot be
          read from the drawing the verdict is UNCERTAIN; otherwise COMPLIANT.

Output STRICT JSON:
{{
  "checks": [{{"name": "...", "actual_value": "...", "threshold": "...", "compare": "X vs Y", "result": "PASS or FAIL"}}],
  "verdict": "COMPLIANT" or "NON_COMPLIANT" or "UNCERTAIN",
  "confidence": 0-100,
  "rationale": "one short sentence",
  "violated_rule": "the violated rule or 'none'"
}}"""


def calib(pairs):
    """pairs: [(confidence 0-1, correct bool)] -> ECE, MCE, Brier, bin table."""
    bins = [[] for _ in range(N_BINS)]
    for c, ok in pairs:
        b = min(N_BINS - 1, int(c * N_BINS))
        bins[b].append((c, ok))
    n = len(pairs)
    ece = mce = 0.0
    table = []
    for i, b in enumerate(bins):
        if not b:
            table.append({"bin": f"{i/N_BINS:.1f}-{(i+1)/N_BINS:.1f}", "n": 0,
                          "mean_confidence": None, "accuracy": None, "gap": None})
            continue
        mc = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, ok in b if ok) / len(b)
        gap = abs(mc - acc)
        ece += len(b) / n * gap
        mce = max(mce, gap)
        table.append({"bin": f"{i/N_BINS:.1f}-{(i+1)/N_BINS:.1f}", "n": len(b),
                      "mean_confidence": round(mc * 100, 1),
                      "accuracy": round(acc * 100, 1), "gap": round(gap * 100, 1)})
    brier = sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in pairs) / n
    return round(ece * 100, 2), round(mce * 100, 2), round(brier, 4), table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="7b_cot,72b_plain,72b_cot_thresh")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    cfgs = a.configs.split(",")

    manifest = json.load(open(MANIFEST))
    for d in manifest:
        d["_checks"] = derive_checks(d)
        d["_query"] = compose_query(d, d["_checks"])

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    print("[cal] shared retrieval pass on ColNomic-3B", flush=True)
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
        meta.append({"agency": d["agency"], "plan_id": d["plan_id"]})
    for qi, d in enumerate(manifest):
        with torch.no_grad():
            Q = ret(**rp.process_queries([d["_query"]]).to("cuda"))[0].float()
        sc = torch.empty(len(pembs))
        for i, P in enumerate(pembs):
            sc[i] = (Q @ P.to("cuda").float().T).max(dim=1).values.sum().item()
        d["_top5"] = [meta[t] for t in torch.topk(sc, TOP_K).indices.tolist()]
    del ret, pembs
    torch.cuda.empty_cache()

    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    results, loaded = {}, None
    for cfg in cfgs:
        size = "72b" if cfg.startswith("72b") else "7b"
        if loaded != size:
            if loaded:
                del jm, jp; torch.cuda.empty_cache()
            print(f"[cal] loading {MODELS[size]}", flush=True)
            jm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODELS[size], quantization_config=bnb, device_map="auto",
                torch_dtype=torch.bfloat16).eval()
            jp = AutoProcessor.from_pretrained(MODELS[size], trust_remote_code=True)
            loaded = size

        recs = []
        for k, d in enumerate(manifest, 1):
            summary = (f"plan_id={d['plan_id']}  agency={d['agency']}\n"
                       "design_facts=<withheld: read every value from the drawing itself>")
            refs = "\n".join(f"  ref{i+1}: {x.get('agency','?'):8s} {str(x.get('plan_id','?')):14s}"
                             for i, x in enumerate(d["_top5"]))
            if cfg.endswith("cot_thresh"):
                prompt = thresh_prompt(summary, refs, d["_checks"])
            elif cfg.endswith("cot"):
                prompt = COT.format(summary=summary, refs=refs, k=TOP_K)
            else:
                prompt = PLAIN.format(summary=summary, refs=refs, k=TOP_K)

            img = Image.open(d["image_path"]).convert("RGB"); img.thumbnail((1400, 1400))
            m = [{"role": "user", "content": [{"type": "image", "image": img},
                                              {"type": "text", "text": prompt}]}]
            t = jp.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            inp = jp(text=[t], images=[img], padding=True, return_tensors="pt").to(jm.device)
            with torch.no_grad():
                o = jm.generate(**inp, max_new_tokens=600 if "cot" in cfg else 400,
                                do_sample=False)
            raw = jp.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
            v = parse_json(raw) or {}
            verdict = str(v.get("verdict", "")).upper()
            conf = v.get("confidence")
            try:
                conf = max(0.0, min(1.0, float(conf) / 100.0))
            except (TypeError, ValueError):
                conf = None
            exp = "COMPLIANT" if d["ground_truth_compliant"] else "NON_COMPLIANT"
            recs.append({"name": d["name"], "expected": exp, "verdict": verdict,
                         "confidence": conf, "abstained": verdict == "UNCERTAIN",
                         "correct": verdict == exp, "raw": raw[:400]})
            if k % 20 == 0:
                print(f"    {cfg} {k}/{len(manifest)} "
                      f"correct={sum(r['correct'] for r in recs)} "
                      f"abstain={sum(r['abstained'] for r in recs)}", flush=True)
                torch.cuda.empty_cache()

        committed = [r for r in recs if not r["abstained"] and r["confidence"] is not None]
        pairs = [(r["confidence"], r["correct"]) for r in committed]
        ece, mce, brier, table = calib(pairs) if pairs else (None, None, None, [])
        nk = sum(r["correct"] for r in recs)
        results[cfg] = {
            "n": len(recs), "accuracy_all": round(nk / len(recs) * 100, 2),
            "wilson_all": wilson(nk, len(recs)),
            "abstention_rate": round(sum(r["abstained"] for r in recs) / len(recs) * 100, 2),
            "n_committed": len(committed),
            "accuracy_committed": (round(sum(1 for r in committed if r["correct"])
                                         / len(committed) * 100, 2) if committed else None),
            "ECE": ece, "MCE": mce, "Brier": brier, "reliability": table,
            "records": recs}
        print(f"  [done] {cfg}: acc {nk}/{len(recs)}  abstain "
              f"{results[cfg]['abstention_rate']}%  ECE {ece}", flush=True)
        json.dump({"note": "Calibration and abstention (R2 stats 6).",
                   "arm": "facts_withheld", "n_bins": N_BINS,
                   "elapsed_min": round((time.time() - t0) / 60, 1),
                   "configs": results}, open(a.out, "w"), indent=1)

    W = 74
    print(f"\n{'=' * W}\nCALIBRATION AND ABSTENTION (facts-withheld arm)\n{'=' * W}")
    print(f"  {'config':<18}{'acc':>8}{'abstain':>10}{'acc|commit':>13}{'ECE':>8}{'Brier':>9}")
    for cfg, r in results.items():
        print(f"  {cfg:<18}{r['accuracy_all']:7.1f}%{r['abstention_rate']:9.1f}%"
              f"{(r['accuracy_committed'] or 0):12.1f}%{(r['ECE'] or 0):8.2f}"
              f"{(r['Brier'] or 0):9.4f}")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
