"""Verify accepted QnA pairs with Qwen2.5-VL-7B as an independent judge.

For each accepted (page, question, answer) tuple, ask Qwen-7B "is this question
answerable from this page?" and "is the proposed answer consistent with the
page?". Drops failures into a quarantine file.

Usage:
    python verify_qwen7b.py --in qwen_drafts_wydot.jsonl \
                            --out verified_qwen_wydot.jsonl \
                            --quarantine quarantine_qwen_wydot.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

VERIFIER_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

VERIFY_PROMPT = (
    "You are verifying a synthetic QnA pair for a retrieval benchmark.\n"
    "QUESTION: {q}\n"
    "PROPOSED ANSWER: {a}\n\n"
    "Look at the image. Answer with exactly one of:\n"
    "  YES  -- the question is well-formed and the answer is supported by the page\n"
    "         (or, for a 'Not specified' answer, the page genuinely lacks the asked info).\n"
    "  NO   -- the question is malformed, off-page, or the answer contradicts the page.\n"
    "Return ONLY YES or NO."
)


def load_verifier():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VERIFIER_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    )
    proc = AutoProcessor.from_pretrained(VERIFIER_ID, trust_remote_code=True)
    model.eval()
    return model, proc


@torch.no_grad()
def verify(model, proc, image: Image.Image, q: str, a: str) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": VERIFY_PROMPT.format(q=q, a=a)},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(gen, skip_special_tokens=True)[0].strip().upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quarantine", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    print(f"[verify] {len(rows)} input records")
    model, proc = load_verifier()

    f_ok = open(args.out, "w")
    f_q = open(args.quarantine, "w")
    n_ok = n_bad = 0
    for i, r in enumerate(rows):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except Exception:
            r["verdict"] = "ERROR_IMAGE"
            f_q.write(json.dumps(r) + "\n"); n_bad += 1; continue
        vd = verify(model, proc, img, r["query"], r["answer"])
        r["verdict"] = vd
        if vd.startswith("YES"):
            f_ok.write(json.dumps(r) + "\n"); n_ok += 1
        else:
            f_q.write(json.dumps(r) + "\n"); n_bad += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  ok={n_ok}  bad={n_bad}")
    f_ok.close(); f_q.close()
    print(f"[verify done] ok={n_ok}  quarantine={n_bad}")


if __name__ == "__main__":
    main()
