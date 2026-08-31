"""Regenerate the 25 malformed question stubs with the Qwen drafter.

The Llama drafter occasionally emitted a markdown header ('**Question:**') and no
question. This redrafts those pages with Qwen2.5-VL-72B - which produced no
malformed output anywhere in the benchmark - using the same category prompts and
the same Qwen2.5-VL-7B verifier as the original pipeline.

The regenerated question is written to a NEW field, `query_regenerated`, and the
original `query` is left untouched. That is deliberate: `query` is what every
published metric was computed on, so the released splits must keep it verbatim or
the results stop reproducing from the shipped data. Consumers who want the clean
set read `query_regenerated` where present.

    python regenerate_malformed.py            # all flagged records
    python regenerate_malformed.py --dry-run  # list what would be redrafted
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import argparse, glob, json, os, re, sys, time

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

Image.MAX_IMAGE_PIXELS = None
DRAFTER = "Qwen/Qwen2.5-VL-72B-Instruct"
VERIFIER = "Qwen/Qwen2.5-VL-7B-Instruct"
SPLITS = sorted(glob.glob(f"{PSR_ROOT}/data/splits/split_*.jsonl"))
# Resolve page images on this machine. Point PLANS_ROOT (or PSR_IMAGE_DIRS, a
# colon-separated list) at wherever the rendered plan pages live; the remaining
# entries are the authors' own paths and are simply skipped when absent.
_env_dirs = [d for d in os.environ.get("PSR_IMAGE_DIRS", "").split(":") if d]
LOCAL_DIRS = [d for d in (_env_dirs + [
                          PLANS_ROOT,
                          os.path.expanduser("~/floorplan_qa/extracted_images"),
                          "/gscratch/nsubedi1/floorplan_qa/extracted_images",
                          "/project/gr-wydot-chatbot/copalirag/data/colorado_2025/images",
                          "/project/gr-wydot-chatbot/copalirag/data/arizona_2025/images",
                          "/project/gr-wydot-chatbot/copalirag/data/florida_2026/images",
                          ]) if d and os.path.isdir(d)]

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
        "must be supported by text on the page.",
    "Hallucination Rate":
        "Write ONE specific question whose answer is NOT visible on this page "
        "but sounds plausible. The correct answer should be 'Not specified'.",
}
ASK_Q = "{cat}\n\nReturn ONLY the question text. No preamble, no heading, no answer."
ASK_A = ("Question: {q}\n\nAnswer it from this page in one short phrase. "
         "If the page does not state it, reply exactly 'Not specified'.")
VERIFY = ("You are verifying a synthetic QnA pair for a retrieval benchmark.\n"
          "QUESTION: {q}\nPROPOSED ANSWER: {a}\n\nLook at the image. Answer with exactly one of:\n"
          "  YES  -- the question is well-formed and the answer is supported by the page\n"
          "         (or, for a 'Not specified' answer, the page genuinely lacks the asked info).\n"
          "  NO   -- the question is malformed, off-page, or the answer contradicts the page.\n"
          "Return ONLY YES or NO.")

STUB = re.compile(r"^\**\s*(rewritten\s+)?question\s*:?\s*\**", re.I)


def is_stub(t):
    core = re.sub(r"[\*\s:#\-]", "", STUB.sub("", (t or "").strip()))
    return len(core) < 8 or ("?" not in (t or "") and len((t or "").strip()) < 40)


def find_image(basename):
    for d in LOCAL_DIRS:
        p = os.path.join(d, basename)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    targets = []
    for f in SPLITS:
        for i, l in enumerate(open(f)):
            r = json.loads(l)
            if r.get("malformed") or is_stub(r.get("query")):
                targets.append((f, i, r))
    print(f"[regen] {len(targets)} malformed records across {len(SPLITS)} splits")
    for f, i, r in targets:
        img = find_image(r.get("source_file", ""))
        print(f"   {os.path.basename(f):<18} {r.get('category','?'):<22} "
              f"{'IMG OK' if img else 'IMG MISSING'}  {r.get('query','')[:28]!r}")
    if a.dry_run:
        return
    if not all(find_image(r.get("source_file", "")) for _, _, r in targets):
        print("[regen] some page images are unavailable on this machine; aborting")
        return

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True)
    dm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        DRAFTER, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
    dp = AutoProcessor.from_pretrained(DRAFTER, trust_remote_code=True,
                                       min_pixels=256 * 28 * 28, max_pixels=1400 * 28 * 28)

    def ask(model, proc, img, text, mx=160):
        m = [{"role": "user", "content": [{"type": "image", "image": img},
                                          {"type": "text", "text": text}]}]
        t = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[t], images=[img], padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=mx, do_sample=False)
        return proc.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()

    drafted = []
    for k, (f, i, r) in enumerate(targets, 1):
        img = Image.open(find_image(r["source_file"])).convert("RGB")
        img.thumbnail((1500, 1500))
        cat = r.get("category", "Dimensional Accuracy")
        q = STUB.sub("", ask(dm, dp, img, ASK_Q.format(cat=CATEGORY_PROMPT.get(cat, "")))).strip()
        q = q.split("\n")[0].strip().strip('"')
        ans = ask(dm, dp, img, ASK_A.format(q=q), mx=80)
        drafted.append({"file": f, "idx": i, "query": q, "answer": ans, "img": img})
        print(f"  [{k}/{len(targets)}] {cat:<22} {q[:66]}", flush=True)
    del dm
    torch.cuda.empty_cache()

    vm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VERIFIER, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
    vp = AutoProcessor.from_pretrained(VERIFIER, trust_remote_code=True)
    for d in drafted:
        v = ask(vm, vp, d["img"], VERIFY.format(q=d["query"], a=d["answer"]), mx=6)
        d["verdict"] = "YES" if re.search(r"\bYES\b", v, re.I) else "NO"
    n_ok = sum(1 for d in drafted if d["verdict"] == "YES")
    print(f"[regen] verifier passed {n_ok}/{len(drafted)}")

    # write back: NEW fields only, `query` and `answer` stay byte-identical
    by_file = {}
    for d in drafted:
        by_file.setdefault(d["file"], {})[d["idx"]] = d
    for f, edits in by_file.items():
        recs = [json.loads(l) for l in open(f)]
        for i, d in edits.items():
            recs[i]["query_regenerated"] = d["query"]
            recs[i]["answer_regenerated"] = d["answer"]
            recs[i]["regenerated_by"] = DRAFTER
            recs[i]["regenerated_verifier_verdict"] = d["verdict"]
        with open(f, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  wrote {len(edits)} regenerated questions -> {f}")
    print("[regen] done. `query` is unchanged, so all published metrics still reproduce.")


if __name__ == "__main__":
    main()
