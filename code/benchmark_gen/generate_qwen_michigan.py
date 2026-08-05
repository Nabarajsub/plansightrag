"""Qwen2.5-VL-72B drafter with ColPali retrieve+refine loop (max_attempts=5,
category-aware rephrase). Uses the 5-DOT v3 index.

Pipeline per target page (4 categories each):
  1. Draft a question for the category.
  2. Check ColPali retrieval against the 1898-page v3 index.
  3. If the target page is not in Top-K, rephrase with a category-specific
     anchor strategy. Up to MAX_ATTEMPTS = 5 attempts.
  4. Write the final question + answer + attempt history to JSONL.

Usage:
    python generate_qwen.py --pages_json pages_scale_wydot.json \\
                            --max_attempts 5 --out qwen_drafts_wydot.jsonl
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
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _retriever_michigan import ColPaliRetriever  # noqa: E402

DEFAULT_DRAFTER_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

CATEGORIES = [
    "Dimensional Accuracy",
    "Visual Interpretation",
    "Logical Reasoning",
    "Hallucination Rate",
]

CATEGORY_PROMPT = {
    "Dimensional Accuracy":
        "Write ONE specific question about a numeric dimension, distance, "
        "spacing, or measurement shown on this engineering standard plan. "
        "The answer must be a specific number with units visible on the page.",
    "Visual Interpretation":
        "Write ONE specific question about a visual element of this engineering "
        "standard plan: a section view, callout, label, symbol, or hatching "
        "pattern. The answer must be a short phrase visible on the page.",
    "Logical Reasoning":
        "Write ONE specific question that requires reading a note, table, or "
        "caption on this engineering standard plan to infer a condition or "
        "rule (e.g., 'When does X apply?', 'What does Y require?'). The answer "
        "must be supported by text on the page. Anchor the question with "
        "EXACT wording from the relevant note or table (e.g., 'per Note 3', "
        "'in Table A', 'where Condition X applies').",
    "Hallucination Rate":
        "Write ONE specific question whose answer is NOT visible on this page "
        "but sounds plausible (e.g., asking for a value or detail the page "
        "does not actually contain). The correct answer should be 'Not "
        "specified' with a brief explanation.",
}

REPHRASE_BASE = (
    "Your previous question for this page did not retrieve the page from a "
    "ColPali visual index. Rewrite it so that the wording uses ONE distinctive "
    "anchor (a plan ID, a specific dimension, a label text, a section title) "
    "that uniquely points to THIS page. Do NOT pile on multiple anchors. Keep "
    "it natural-sounding. Return ONLY the rewritten question."
)

REPHRASE_LOGICAL = (
    "Your previous question for this page did not retrieve the page from a "
    "ColPali visual index. Logical-reasoning questions miss when they lack "
    "structural anchors. Rewrite it so that it QUOTES 3-6 exact words from a "
    "visible NOTE, TABLE HEADER, CALLOUT, or LABEL on the page (e.g., 'per "
    "Note 3:', 'in Table A column', 'where \"Condition X applies\"'). The "
    "rewritten question MUST contain that exact quoted phrase. Keep it natural. "
    "Return ONLY the rewritten question."
)


def rephrase_prompt_for(category: str) -> str:
    return REPHRASE_LOGICAL if category == "Logical Reasoning" else REPHRASE_BASE


def load_drafter(model_id: str = DEFAULT_DRAFTER_ID):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[drafter] loading {model_id} (4-bit)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor


@torch.no_grad()
def vlm_generate(model, processor, image: Image.Image, instruction: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": instruction},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def draft_question(model, processor, image, category):
    prompt = (f"{CATEGORY_PROMPT[category]}\n\n"
              "Return ONLY the question text, no preamble, no answer.")
    q = vlm_generate(model, processor, image, prompt, max_new_tokens=128)
    for prefix in ("Question:", "Q:", "**Question**:"):
        if q.lower().startswith(prefix.lower()):
            q = q[len(prefix):].strip()
    return q.split("\n")[0].strip().strip('"')


def rephrase_question(model, processor, image, prev_q, category):
    prompt = f"Previous question: {prev_q}\n\n{rephrase_prompt_for(category)}"
    q = vlm_generate(model, processor, image, prompt, max_new_tokens=160)
    return q.split("\n")[0].strip().strip('"')


def answer_question(model, processor, image, question):
    prompt = (f"{question}\n\n"
              "Answer in one short sentence (or 'Not specified' if not on the page). "
              "Do not add preamble.")
    return vlm_generate(model, processor, image, prompt, max_new_tokens=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages_json", required=True)
    ap.add_argument("--max_attempts", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model_id", default=DEFAULT_DRAFTER_ID)
    args = ap.parse_args()

    random.seed(args.seed)
    pages = json.load(open(args.pages_json))
    print(f"[main] target pages: {len(pages)}  categories: {len(CATEGORIES)}  expected Q: {len(pages)*len(CATEGORIES)}")

    done_keys: set[str] = set()
    if args.resume and os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done_keys.add(f"{r['image_path']}|{r['category']}")
            except Exception:
                pass
        print(f"[resume] {len(done_keys)} entries already in {args.out}")

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
            rate = n_total / max(dt, 1e-3) * 60
            print(f"[progress] page {page_idx+1}/{len(pages)}  Q={n_total}  "
                  f"hit={n_hit} ({n_hit/max(n_total,1)*100:.1f}%)  "
                  f"{rate:.1f} Q/min  elapsed={dt/60:.1f} min")

    out_f.close()
    print(f"[done] {n_total} entries  hit-rate={n_hit/max(n_total,1)*100:.2f}%")


if __name__ == "__main__":
    main()
