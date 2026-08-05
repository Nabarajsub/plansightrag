"""Eval a trained ColNomic adapter (head or full-LoRA) vs zero-shot ColNomic on the
424-pair test split (over the 1,898-page index) + Michigan transfer -- the H3
ablation numbers for manuscript_swap.

Usage:
  python eval_colnomic.py --mode head --adapter lora_finetune/colnomic_head_standard --tag head_standard
  python eval_colnomic.py --mode full --adapter lora_finetune/colnomic_full_lm --tag full_lm
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import argparse, glob, json, os, sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

ROOT = f"{PSR_ROOT}"
MODEL_ID = "nomic-ai/colnomic-embed-multimodal-3b"
TEST = f"{ROOT}/lora_finetune/split_test.jsonl"
MICH_PAGES_DIR = f"{ROOT}/data/michigan_2025/images"
MICH_QA = f"{ROOT}/qna_expansion/verified_michigan.jsonl"
OUT = f"{ROOT}/baselines_v2/reports"


def encode(model, proc, paths, is_query=False, queries=None):
    dev = "cuda"; embs = []
    if is_query:
        for s in range(0, len(queries), 16):
            with torch.no_grad():
                out = model(**proc.process_queries(queries[s:s+16]).to(dev))
            embs += [out[j].to(torch.float16).cpu() for j in range(out.shape[0])]
    else:
        for s in tqdm(range(0, len(paths), 4), desc="enc img"):
            imgs = [Image.open(p).convert("RGB") for p in paths[s:s+4]]
            with torch.no_grad():
                out = model(**proc.process_images(imgs).to(dev))
            embs += [out[j].to(torch.float16).cpu() for j in range(out.shape[0])]
    return embs


def recall5(qembs, pembs, qrows, ppaths):
    dev = "cuda"; bn = [os.path.basename(p) for p in ppaths]; hit = 0
    for qi, r in enumerate(qrows):
        qe = qembs[qi].to(dev).float()
        sc = np.array([torch.matmul(qe, pembs[pi].to(dev).float().T).max(-1).values.sum().item() for pi in range(len(pembs))])
        top = {bn[j] for j in np.argsort(sc)[::-1][:5]}
        hit += int(os.path.basename(r["image_path"]) in top)
    return 100.0 * hit / len(qrows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["head", "full", "zeroshot"], required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = "cuda"

    model = ColQwen2_5.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev)
    proc = ColQwen2_5_Processor.from_pretrained(MODEL_ID)
    if args.mode == "head" and args.adapter:
        sd = torch.load(f"{args.adapter}/custom_text_proj.pt", weights_only=False)
        model.custom_text_proj.load_state_dict(sd); print(f"[adapter] head loaded from {args.adapter}")
    elif args.mode == "full" and args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter); print(f"[adapter] LoRA loaded from {args.adapter}")
    model.eval()

    # 5-DOT test split over the 1,898-page index
    sys.path.insert(0, f"{ROOT}/baselines_v2")
    from _eval_lib import load_pages, load_test
    pages = load_pages(); test = load_test()
    ppaths = [p["image_path"] for p in pages]
    pembs = encode(model, proc, ppaths)
    qembs = encode(model, proc, None, is_query=True, queries=[r["query"] for r in test])
    test_r5 = recall5(qembs, pembs, test, ppaths)

    # Michigan transfer
    mp = sorted(glob.glob(f"{MICH_PAGES_DIR}/*.png"))
    mq = [json.loads(l) for l in open(MICH_QA)]
    mpembs = encode(model, proc, mp)
    mqembs = encode(model, proc, None, is_query=True, queries=[r["query"] for r in mq])
    mich_r5 = recall5(mqembs, mpembs, mq, mp)

    out = {"tag": args.tag, "mode": args.mode, "adapter": args.adapter,
           "test_recall@5": round(test_r5, 2), "michigan_recall@5": round(mich_r5, 2)}
    json.dump(out, open(f"{OUT}/colnomic_lora_{args.tag}.json", "w"), indent=2)
    print(f"\n[done] {args.tag}: test R@5={test_r5:.2f}%  Michigan R@5={mich_r5:.2f}%")


if __name__ == "__main__":
    main()
