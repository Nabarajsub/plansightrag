"""Model-as-auditor verdict on REAL WYDOT plan imagery (genuine model-driven verdict).

Uses the SAME recipe that reaches 100% on the synthetic CAD set -- Qwen2.5-VL-72B
(4-bit) + CoT + the governing rule supplied in the prompt -- so this is a fair
synthetic-vs-real comparison. The VLM must READ the proposed value off a real
cross-section sheet and judge it against the stated WYDOT rule; the verdict is the
model's, not a coded comparison.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, re, time
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

RPE = f"{PSR_ROOT}/real_plan_eval"
SET = f"{RPE}/auditor_set"
QWEN = "Qwen/Qwen2.5-VL-72B-Instruct"
OUT = f"{RPE}/real_auditor_report.json"
DEV = "cuda"


def main():
    man = json.load(open(f"{SET}/manifest.json"))
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    print(f"[auditor] loading {QWEN} (4-bit) | {len(man)} real-imagery drawings")
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
    # cap vision tokens: real landscape sheets are huge and OOM the 72B on 2x A30
    vproc = AutoProcessor.from_pretrained(QWEN, trust_remote_code=True,
                                          min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

    def fit(image, longest=1400):
        w, h = image.size
        if max(w, h) > longest:
            s = longest / max(w, h)
            image = image.resize((int(w * s), int(h * s)), Image.BICUBIC)
        return image

    @torch.no_grad()
    def ask(image, prompt, mx=320):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        t = vproc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = vproc(text=[t], images=[image], padding=True, return_tensors="pt").to(vlm.device)
        out = vlm.generate(**inp, max_new_tokens=mx, do_sample=False)
        return vproc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    n = ok = 0
    results = []
    t0 = time.time()
    for m in man:
        img = fit(Image.open(m["image"]).convert("RGB"))
        prompt = (
            "You are a WYDOT engineering compliance auditor. This is a real standard cross-section "
            "sheet with a 'PROPOSED DESIGN SUBMITTAL' callout. "
            f"Governing rule: {m['rule']}. "
            f"Read the value stated in the PROPOSED DESIGN SUBMITTAL callout for '{m['component']}', "
            "compare it to the governing rule, and decide compliance. Reason briefly, then end with "
            "exactly one line: 'VERDICT: COMPLIANT' or 'VERDICT: NON_COMPLIANT'.")
        resp = ask(img, prompt)
        mobj = re.search(r"VERDICT:\s*(NON[_\s-]?COMPLIANT|COMPLIANT)", resp, re.IGNORECASE)
        verdict = mobj.group(1).upper().replace(" ", "_").replace("-", "_") if mobj else "PARSE_FAIL"
        pred_compliant = (verdict == "COMPLIANT")
        correct = (pred_compliant == m["gt_compliant"]) and verdict != "PARSE_FAIL"
        n += 1; ok += int(correct)
        results.append({"id": m["id"], "component": m["component"], "submitted_value": m["submitted_value"],
                        "rule": m["rule"], "gt_compliant": m["gt_compliant"],
                        "verdict": verdict, "correct": correct, "response_tail": resp[-160:]})
        print(f"  {m['id']:16s} {m['component'][:24]:24s} val={m['submitted_value']:14s} "
              f"-> {verdict:14s} gt={m['gt_compliant']} {'OK' if correct else 'XX'}")

    rep = {
        "experiment": "Model-as-auditor verdict on real WYDOT plan imagery (Qwen2.5-VL-72B + CoT + given rule)",
        "model": QWEN, "n": n,
        "verdict_accuracy": round(100.0 * ok / n, 2),
        "n_compliant": sum(m["gt_compliant"] for m in man),
        "n_noncompliant": sum(not m["gt_compliant"] for m in man),
        "correct_on_compliant": sum(1 for r in results if r["gt_compliant"] and r["correct"]),
        "correct_on_noncompliant": sum(1 for r in results if not r["gt_compliant"] and r["correct"]),
        "results": results, "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ["n", "verdict_accuracy", "correct_on_compliant", "correct_on_noncompliant"]}, indent=2))
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
