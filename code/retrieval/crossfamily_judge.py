"""Cross-family LLM-as-judge check (Major #4).

The benchmark uses Qwen2.5-VL throughout (drafter/verifier/answerer/judge), so the
judge-accuracy numbers carry a self-preference risk. We re-judge the SAME 424
Qwen2.5-VL-7B zero-shot answers with a NON-Qwen judge (InternVL2.5-8B) and a Qwen
judge (Qwen2.5-VL-72B) on identical inputs, then report each judge's accuracy,
their pairwise agreement, and Cohen's kappa. Both judges see the gold page image
+ question + reference + model answer and return YES/NO.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import ast, json, re, time
import torch
from PIL import Image

ROOT = f"{PSR_ROOT}"
ANS = f"{ROOT}/vqa_eval/answers_qwen7b.jsonl"
QWEN = "Qwen/Qwen2.5-VL-72B-Instruct"
INTERNVL = "OpenGVLab/InternVL2_5-8B-MPO-hf"
OUT = f"{ROOT}/baselines_v2/reports/crossfamily_judge.json"
DEV = "cuda"

PROMPT = ("You are grading an answer to a question about an engineering drawing.\n\n"
          "Question: {q}\nReference answer: {ref}\nModel answer: {ans}\n\n"
          "Is the model answer correct and consistent with the reference (allowing for "
          "paraphrase, units, and rounding)? Reply with exactly one word: YES or NO.")


def load_items():
    rows = [json.loads(l) for l in open(ANS)]
    out = []
    for r in rows:
        a = r["answers"]
        a = ast.literal_eval(a) if isinstance(a, str) else a
        ans = a.get("zeroshot", "") if isinstance(a, dict) else str(a)
        img = r["id"].split("|")[0].strip()   # id is 'path|category|query'
        out.append({"img": img, "q": r["query"], "ref": r["reference"],
                    "ans": ans, "cat": r.get("category", "?")})
    return out


def yn(text):
    t = text.strip().upper()
    if re.search(r"\bYES\b", t): return 1
    if re.search(r"\bNO\b", t): return 0
    return 1 if t.startswith("Y") else 0


def judge_qwen(items):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16).eval()
    p = AutoProcessor.from_pretrained(QWEN, trust_remote_code=True, max_pixels=1280*28*28)
    v = []
    for it in items:
        img = Image.open(it["img"]).convert("RGB")
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": PROMPT.format(q=it["q"], ref=it["ref"], ans=it["ans"])}]}]
        t = p.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = p(text=[t], images=[img], return_tensors="pt").to(m.device)
        with torch.no_grad():
            o = m.generate(**inp, max_new_tokens=4, do_sample=False)
        v.append(yn(p.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]))
    del m; torch.cuda.empty_cache()
    return v


def judge_internvl(items):
    from transformers import AutoProcessor, InternVLForConditionalGeneration, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    m = InternVLForConditionalGeneration.from_pretrained(INTERNVL, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.bfloat16).eval()
    p = AutoProcessor.from_pretrained(INTERNVL, trust_remote_code=True)
    v = []
    for it in items:
        img = Image.open(it["img"]).convert("RGB")
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                {"type": "text", "text": PROMPT.format(q=it["q"], ref=it["ref"], ans=it["ans"])}]}]
        inp = p.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                    return_dict=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            o = m.generate(**inp, max_new_tokens=4, do_sample=False)
        v.append(yn(p.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]))
    del m; torch.cuda.empty_cache()
    return v


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a)/n; pb1 = sum(b)/n
    pe = pa1*pb1 + (1-pa1)*(1-pb1)
    return round((po-pe)/(1-pe), 4) if pe < 1 else 1.0, round(100*po, 2)


def main():
    items = load_items()
    print(f"[judge] {len(items)} Qwen-7B zero-shot answers; judges: Qwen2.5-VL-72B vs InternVL2.5-8B")
    t0 = time.time()
    qv = judge_qwen(items)
    iv = judge_internvl(items)
    k, agree = kappa(qv, iv)
    rep = {
        "n": len(items),
        "qwen72b_judge_accuracy": round(100.0*sum(qv)/len(qv), 2),
        "internvl8b_judge_accuracy": round(100.0*sum(iv)/len(iv), 2),
        "pairwise_agreement_pct": agree,
        "cohens_kappa": k,
        "interpretation": "kappa>0.6 substantial; >0.4 moderate",
        "elapsed_sec": round(time.time()-t0, 1),
    }
    json.dump(rep, open(OUT, "w"), indent=2)
    print(json.dumps(rep, indent=2)); print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
