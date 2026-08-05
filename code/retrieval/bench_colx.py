"""Generic late-interaction visual-retriever benchmark on the 424-pair test split.

Tests any colpali_engine Col* model (ColPali / ColQwen2.5 / ColNomic, which loads
via the ColQwen2.5 class) under the identical protocol used for ColQwen2.5-v0.2:
encode all 1,898 pages + 424 queries, MaxSim late-interaction, Recall@5.

Usage:
  python bench_colx.py --model vidore/colpali-v1.3 --klass ColPali --tag colpali_v13
  python bench_colx.py --model nomic-ai/colnomic-embed-multimodal-3b --klass ColQwen2_5 --tag colnomic_3b
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import argparse, sys, time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

# torch 2.5.1 (<2.6) trips transformers' CVE guard on .bin checkpoints; these are
# official published retrievers loaded weights_only=True, so disable the guard.
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None


def get_classes(klass):
    import colpali_engine.models as M
    model = getattr(M, klass)
    # processor naming is inconsistent: ColPaliProcessor vs ColQwen2_5_Processor
    for name in (klass + "Processor", klass + "_Processor"):
        if hasattr(M, name):
            return model, getattr(M, name)
    raise AttributeError(f"no processor found for {klass}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--klass", default="ColQwen2_5", help="ColPali | ColQwen2_5 | ColQwen2")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    dev = "cuda"

    Model, Proc = get_classes(args.klass)
    print(f"[load] {args.model} as {args.klass}")
    t0 = time.time()
    proc = Proc.from_pretrained(args.model)
    model = Model.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=dev).eval()
    print(f"[load] {time.time()-t0:.1f}s")

    pages = load_pages()
    test = load_test()
    print(f"[data] {len(pages)} pages, {len(test)} queries")

    # encode pages
    page_embs = []
    for s in tqdm(range(0, len(pages), 4), desc="encode pages"):
        imgs = [Image.open(p["image_path"]).convert("RGB") for p in pages[s:s + 4]]
        batch = proc.process_images(imgs).to(dev)
        with torch.no_grad():
            out = model(**batch)
        for j in range(out.shape[0]):
            page_embs.append(out[j].to(torch.float16).cpu())

    # encode queries
    q_embs = []
    qs = [r["query"] for r in test]
    for s in tqdm(range(0, len(qs), 16), desc="encode queries"):
        batch = proc.process_queries(qs[s:s + 16]).to(dev)
        with torch.no_grad():
            out = model(**batch)
        for j in range(out.shape[0]):
            q_embs.append(out[j].to(torch.float16).cpu())

    # MaxSim score matrix
    scores = np.zeros((len(q_embs), len(page_embs)), dtype=np.float32)
    for pi, pe in enumerate(tqdm(page_embs, desc="MaxSim")):
        peg = pe.to(dev).to(torch.float32)
        for qi, qe in enumerate(q_embs):
            qeg = qe.to(dev).to(torch.float32)
            scores[qi, pi] = torch.matmul(qeg, peg.T).max(dim=-1).values.sum().item()

    overall, per_agency, per_cat, hits = recall_at_k(scores, test, pages)
    save_report(args.tag, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": args.model, "klass": args.klass})
    print(f"\n[done] {args.tag}: Recall@5 = {overall:.2f}%")
    for a, v in sorted(per_agency.items()):
        print(f"    {a:10s} R@5={v['recall@5']:.2f}%")


if __name__ == "__main__":
    main()
