"""Llama-3.2-90B-Vision drafter with ColPali retrieve+refine (max_attempts=5,
category-aware rephrase). Uses the 5-DOT v3 index.

Shares CATEGORIES / CATEGORY_PROMPT / rephrase_prompt_for() with generate_qwen.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, MllamaForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _retriever import ColPaliRetriever  # noqa: E402
from generate_qwen import (  # noqa: E402
    CATEGORIES,
    CATEGORY_PROMPT,
    rephrase_prompt_for,
)

DEFAULT_DRAFTER_ID = "meta-llama/Llama-3.2-90B-Vision-Instruct"


def load_drafter(model_id: str = DEFAULT_DRAFTER_ID):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[drafter] loading {model_id} (4-bit)")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor


@torch.no_grad()
def vlm_generate(model, processor, image: Image.Image, instruction: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def draft_question(m, p, image, category):
    prompt = (f"{CATEGORY_PROMPT[category]}\n\n"
              "Return ONLY the question text, no preamble, no answer.")
    q = vlm_generate(m, p, image, prompt, max_new_tokens=128)
    for prefix in ("Question:", "Q:", "**Question**:"):
        if q.lower().startswith(prefix.lower()):
            q = q[len(prefix):].strip()
    return q.split("\n")[0].strip().strip('"')


def rephrase_question(m, p, image, prev_q, category):
    prompt = f"Previous question: {prev_q}\n\n{rephrase_prompt_for(category)}"
    q = vlm_generate(m, p, image, prompt, max_new_tokens=160)
    return q.split("\n")[0].strip().strip('"')


def answer_question(m, p, image, question):
    prompt = (f"{question}\n\n"
              "Answer in one short sentence (or 'Not specified' if not on the page). "
              "Do not add preamble.")
    return vlm_generate(m, p, image, prompt, max_new_tokens=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages_json", required=True)
    ap.add_argument("--max_attempts", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model_id", default=DEFAULT_DRAFTER_ID)
    args = ap.parse_args()

    random.seed(args.seed)
    pages = json.load(open(args.pages_json))
    print(f"[main] target pages: {len(pages)}")

    done_keys: set[str] = set()
    if args.resume and os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done_keys.add(f"{r['image_path']}|{r['category']}")
            except Exception:
                pass

    retriever = ColPaliRetriever()
    drafter, processor = load_drafter(args.model_id)
    drafter_label = args.model_id.split("/")[-1]

    out_f = open(args.out, "a")
    t0 = time.time()
    n_total = n_hit = 0

    for page_idx, page in enumerate(pages):
        try:
            image = Image.open(page["image_path"]).convert("RGB")
        except Exception as exc:
            print(f"[skip] {page['image_path']}: {exc}")
            continue
        for cat in CATEGORIES:
            key = f"{page['image_path']}|{cat}"
            if key in done_keys:
                continue
            attempts = []
            q = draft_question(drafter, processor, image, cat)
            for att in range(1, args.max_attempts + 1):
                rank = retriever.rank_of(q, page["image_path"], k_check=max(args.top_k, 10))
                hit = bool(rank is not None and rank <= args.top_k)
                attempts.append({"attempt": att, "question": q, "rank": rank, "hit": hit})
                if hit:
                    break
                if att < args.max_attempts:
                    q = rephrase_question(drafter, processor, image, q, cat)
            final = attempts[-1]
            answer = answer_question(drafter, processor, image, final["question"])
            out_f.write(json.dumps({
                "image_path": page["image_path"],
                "agency": page["agency"],
                "plan_id": page.get("plan_id", ""),
                "sheet_title": page.get("sheet_title", ""),
                "category": cat,
                "drafter": drafter_label,
                "query": final["question"],
                "answer": answer,
                "attempts": attempts,
                "final_rank": final["rank"],
                "final_hit": final["hit"],
                "n_attempts": len(attempts),
            }) + "\n")
            out_f.flush()
            n_total += 1
            n_hit += int(final["hit"])
        if (page_idx + 1) % 5 == 0 or page_idx == len(pages) - 1:
            dt = time.time() - t0
            print(f"[progress] page {page_idx+1}/{len(pages)}  Q={n_total}  "
                  f"hit={n_hit} ({n_hit/max(n_total,1)*100:.1f}%)  elapsed={dt/60:.1f} min")

    out_f.close()
    print(f"[done] {n_total} entries  hit-rate={n_hit/max(n_total,1)*100:.2f}%")


if __name__ == "__main__":
    main()
