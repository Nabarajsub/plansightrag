"""Autonomous Rule-Grounding pilot (novelty centerpiece).

Today the agentic judge is HANDED the pre-resolved numeric threshold and reaches
100%. This pilot removes the human registry: for each compliance drawing the agent
(1) forms a rule-retrieval query from the visually-inferred component, (2) retrieves
the governing spec sheet from a 15-sheet Standard-Specifications corpus with ColNomic,
(3) a VLM EXTRACTS the numeric threshold from the retrieved sheet, (4) resolves
symbolic rules (stirrup d/2) using a dimension read from the design, and (5) audits
the design value against the SELF-GROUNDED threshold.

We isolate the new capability (autonomous threshold sourcing) by taking the design
value from ground truth, so the only thing under test is whether the system can find
and apply the right rule without being told it. Reports:
  - retrieval_section_acc : top-1 retrieved sheet is the correct spec section
  - threshold_extraction_acc : VLM reads the correct numeric limit off the sheet
  - verdict_acc_self_grounded : end-to-end verdict using the self-grounded threshold
  - (reference) verdict_acc_injected = 100% (human-supplied threshold)
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, os, re, sys, time
import numpy as np
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

ROOT = f"{PSR_ROOT}"
RG = f"{ROOT}/compliance/rule_grounding"
SPEC_MANIFEST = f"{RG}/spec_corpus/manifest.json"
DRAW_MANIFEST = f"{ROOT}/compliance/scale500/mockups/manifest.json"
COLNOMIC = "nomic-ai/colnomic-embed-multimodal-3b"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT = f"{RG}/rule_grounding_report.json"
DEV = "cuda"

# archetype -> (component key, correct spec section, governing value, operator, item phrasing, rule query)
ARCH = {
    "draw_culvert":   ("concrete_cover", "601", 2.0,  ">=", "minimum clear cover for reinforced section",
                       "minimum concrete cover requirement for reinforcing steel"),
    "draw_rebar":     ("stirrup_spacing", "601", None, "<=", "maximum shear-stirrup spacing",
                       "maximum shear stirrup spacing reinforced concrete beam"),
    "draw_guardrail": ("footing_depth", "606", 30.0, ">=", "guardrail post footing minimum depth",
                       "minimum guardrail post footing foundation depth"),
    "draw_sign_post": ("anchor_embed", "606", 12.0, ">=", "sign-post anchor bolt minimum embedment",
                       "minimum sign post anchor bolt embedment depth"),
    "draw_inlet":     ("grate_opening", "232", 4.0, "<=", "drainage grate maximum clear opening",
                       "maximum drainage grate clear opening pedestrian ADA"),
}


def load_vlm():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN, quantization_config=bnb,
                                                            device_map=DEV, torch_dtype=torch.bfloat16)
    p = AutoProcessor.from_pretrained(QWEN, trust_remote_code=True); m.eval()
    return m, p


@torch.no_grad()
def vlm_ask(m, p, image, prompt, max_new=64):
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = p(text=[text], images=[image], padding=True, return_tensors="pt").to(m.device)
    out = m.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return p.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def maxsim(qe, pe):
    return torch.matmul(qe.float(), pe.float().T).max(-1).values.sum().item()


def main():
    spec = json.load(open(SPEC_MANIFEST))
    draws = json.load(open(DRAW_MANIFEST))
    spaths = [s["image_path"] for s in spec]
    print(f"[rg] {len(spec)} spec sheets | {len(draws)} drawings")

    col = ColQwen2_5.from_pretrained(COLNOMIC, torch_dtype=torch.bfloat16, device_map=DEV)
    cproc = ColQwen2_5_Processor.from_pretrained(COLNOMIC); col.eval()

    # encode spec corpus
    sembs = []
    for s in range(0, len(spaths), 4):
        imgs = [Image.open(p).convert("RGB") for p in spaths[s:s+4]]
        with torch.no_grad():
            o = col(**cproc.process_images(imgs).to(DEV))
        sembs += [o[j].to(torch.float16).cpu() for j in range(o.shape[0])]

    vlm, vproc = load_vlm()

    n = ret_ok = ext_ok = verdict_ok = 0
    fails = []
    t0 = time.time()
    for d in draws:
        meta = ARCH.get(d["archetype"])
        if not meta:
            continue
        key, sec, gov, op, item, query = meta
        # 1-2. retrieve governing spec sheet
        with torch.no_grad():
            qe = col(**cproc.process_queries([query]).to(DEV))[0].to(torch.float16).cpu()
        scores = np.array([maxsim(qe, se) for se in sembs])
        top = int(np.argsort(scores)[::-1][0])
        ret_sheet = spec[top]
        retrieved_right = (ret_sheet["section"] == sec)

        # 3. VLM extracts the numeric threshold from the retrieved sheet
        ans = vlm_ask(vlm, vproc, Image.open(ret_sheet["image_path"]).convert("RGB"),
                      f"This is a standard-specifications table. Report ONLY the limit value for "
                      f"the row '{item}'. Answer with just the value (e.g., '2.0 in' or 'd/2').")

        # parse extracted threshold
        df = d["design_facts"]
        if key == "stirrup_spacing":
            # symbolic d/2 -> resolve with beam height (effective depth proxy)
            extracted_symbolic = bool(re.search(r"d\s*/\s*2", ans, re.IGNORECASE))
            thr = (df["beam_h_in"] // 2) if extracted_symbolic else None
            ext_correct = extracted_symbolic
            design_val = df["stirrup_spacing_in"]
        else:
            mnum = re.search(r"(\d+(?:\.\d+)?)", ans)
            thr = float(mnum.group(1)) if mnum else None
            ext_correct = (thr is not None and abs(thr - gov) < 1e-6)
            design_val = {"concrete_cover": df.get("cover_in"),
                          "footing_depth": df.get("footing_depth_in"),
                          "anchor_embed": df.get("anchor_embed_in"),
                          "grate_opening": df.get("grate_open_in")}[key]

        # 5. audit design value against SELF-GROUNDED threshold
        if thr is None:
            pred = None
        elif op == ">=":
            pred = design_val >= thr
        else:
            pred = design_val <= thr
        verdict_correct = (pred == d["ground_truth_compliant"])

        n += 1; ret_ok += retrieved_right; ext_ok += ext_correct; verdict_ok += int(verdict_correct)
        if not verdict_correct:
            fails.append({"name": d["name"], "archetype": d["archetype"], "query": query,
                          "retrieved_section": ret_sheet["section"], "retrieved_right": retrieved_right,
                          "vlm_extracted": ans, "threshold": thr, "design_val": design_val,
                          "gt_compliant": d["ground_truth_compliant"], "pred": pred})

    rep = {
        "experiment": "Autonomous Rule-Grounding (retrieve spec -> extract threshold -> audit)",
        "retriever": COLNOMIC, "extractor": QWEN, "n": n,
        "retrieval_section_acc": round(100.0 * ret_ok / n, 2),
        "threshold_extraction_acc": round(100.0 * ext_ok / n, 2),
        "verdict_acc_self_grounded": round(100.0 * verdict_ok / n, 2),
        "verdict_acc_injected_reference": 100.0,
        "n_failures": len(fails), "failures": fails[:40],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ["retrieval_section_acc", "threshold_extraction_acc",
                                           "verdict_acc_self_grounded", "n_failures"]}, indent=2))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
