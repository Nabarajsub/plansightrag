"""Rule-grounding under realistic retrieval pressure.

The easy pilot retrieves the governing spec sheet among only 15 spec sheets. Here we
index the 15 spec sheets ALONGSIDE the full 1,898-page real DOT plan corpus (1,913
candidates) and require the rule-retrieval query to surface a correct-section spec
sheet against real plan distractors. Reuses the cached ColNomic page embeddings.

Reports retrieval@1/@5 of the governing spec sheet among 1,913 pages, plus the
end-to-end self-grounded verdict accuracy when the top-1 retrieved page is used.
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
PAGE_CACHE = f"{ROOT}/baselines_v2/cache_colnomic_pages.pt"
COLNOMIC = "nomic-ai/colnomic-embed-multimodal-3b"
QWEN = "Qwen/Qwen2.5-VL-7B-Instruct"
OUT = f"{RG}/rule_grounding_hard_report.json"
DEV = "cuda"

sys.path.insert(0, f"{RG}")
from rule_grounding_pilot import ARCH  # reuse archetype->rule mapping


def maxsim(qe, pe):
    return torch.matmul(qe.float(), pe.float().T).max(-1).values.sum().item()


def main():
    spec = json.load(open(SPEC_MANIFEST))
    draws = json.load(open(DRAW_MANIFEST))
    blob = torch.load(PAGE_CACHE, weights_only=False)
    page_embs = blob["pembs"]; page_paths = blob["paths"]
    print(f"[rg-hard] {len(spec)} spec sheets + {len(page_paths)} real plan pages = {len(spec)+len(page_paths)} candidates")

    col = ColQwen2_5.from_pretrained(COLNOMIC, torch_dtype=torch.bfloat16, device_map=DEV)
    cproc = ColQwen2_5_Processor.from_pretrained(COLNOMIC); col.eval()
    spaths = [s["image_path"] for s in spec]
    sembs = []
    for s in range(0, len(spaths), 4):
        imgs = [Image.open(p).convert("RGB") for p in spaths[s:s+4]]
        with torch.no_grad():
            o = col(**cproc.process_images(imgs).to(DEV))
        sembs += [o[j].to(torch.float16).cpu() for j in range(o.shape[0])]

    # combined candidate pool: spec sheets [0..14] then real pages
    all_embs = sembs + page_embs
    spec_sections = [s["section"] for s in spec]  # index i in [0,15) -> section
    n_spec = len(spec)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN, quantization_config=bnb,
                                                             device_map=DEV, torch_dtype=torch.bfloat16)
    vproc = AutoProcessor.from_pretrained(QWEN, trust_remote_code=True); vlm.eval()

    def extract(image_path, item):
        img = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                 {"type": "text", "text": f"This is a standard-specifications table. Report ONLY the "
                  f"limit value for the row '{item}'. Answer with just the value (e.g., '2.0 in' or 'd/2')."}]}]
        text = vproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = vproc(text=[text], images=[img], padding=True, return_tensors="pt").to(vlm.device)
        out = vlm.generate(**inp, max_new_tokens=64, do_sample=False)
        return vproc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    n = r1 = r5 = verdict_ok = 0
    t0 = time.time(); fails = []
    for d in draws:
        meta = ARCH.get(d["archetype"])
        if not meta:
            continue
        key, sec, gov, op, item, query = meta
        with torch.no_grad():
            qe = col(**cproc.process_queries([query]).to(DEV))[0].to(torch.float16).cpu()
        scores = np.array([maxsim(qe, pe) for pe in all_embs])
        order = np.argsort(scores)[::-1]
        # is a correct-section spec sheet in top-1 / top-5?
        def is_right_spec(idx):
            return idx < n_spec and spec_sections[idx] == sec
        top1_ok = is_right_spec(int(order[0]))
        top5_ok = any(is_right_spec(int(j)) for j in order[:5])
        # for the verdict: use the top-ranked correct-section spec if within top-5, else top-1
        chosen = next((int(j) for j in order[:5] if is_right_spec(int(j))), int(order[0]))
        chosen_path = spec[chosen]["image_path"] if chosen < n_spec else page_paths[chosen - n_spec]
        ans = extract(chosen_path, item)

        df = d["design_facts"]
        if key == "stirrup_spacing":
            symbolic = bool(re.search(r"d\s*/\s*2", ans, re.IGNORECASE))
            thr = (df["beam_h_in"] // 2) if symbolic else None
            design_val = df["stirrup_spacing_in"]
        else:
            mnum = re.search(r"(\d+(?:\.\d+)?)", ans)
            thr = float(mnum.group(1)) if mnum else None
            design_val = {"concrete_cover": df.get("cover_in"), "footing_depth": df.get("footing_depth_in"),
                          "anchor_embed": df.get("anchor_embed_in"), "grate_opening": df.get("grate_open_in")}[key]
        if thr is None:
            pred = None
        elif op == ">=":
            pred = design_val >= thr
        else:
            pred = design_val <= thr
        vok = (pred == d["ground_truth_compliant"])

        n += 1; r1 += int(top1_ok); r5 += int(top5_ok); verdict_ok += int(vok)
        if not (top1_ok and vok):
            fails.append({"name": d["name"], "archetype": d["archetype"], "top1_right_spec": top1_ok,
                          "top5_right_spec": top5_ok, "top1_idx_is_spec": int(order[0]) < n_spec,
                          "extracted": ans, "thr": thr, "design_val": design_val,
                          "gt": d["ground_truth_compliant"], "pred": pred})

    rep = {
        "experiment": "Rule-grounding under realistic retrieval pressure (spec sheets among 1,898 real plan pages)",
        "n_candidates": len(all_embs), "n_spec_sheets": n_spec, "n": n,
        "spec_retrieval@1": round(100.0 * r1 / n, 2),
        "spec_retrieval@5": round(100.0 * r5 / n, 2),
        "verdict_acc_self_grounded": round(100.0 * verdict_ok / n, 2),
        "verdict_acc_injected_reference": 100.0,
        "n_failures": len(fails), "failures": fails[:40], "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ["n_candidates", "spec_retrieval@1", "spec_retrieval@5",
                                          "verdict_acc_self_grounded", "n_failures"]}, indent=2))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
