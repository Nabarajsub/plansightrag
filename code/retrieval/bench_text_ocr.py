"""Text + OCR strong baselines: BGE-M3 and NV-Embed-v2.

Both consume the shared OCR text dump from ocr_all_pages.py and produce
single-vector dense representations of pages and queries.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# NV-Embed-v2's vendored modeling code calls DynamicCache.get_usable_length(),
# which transformers renamed to get_seq_length(). Add a back-compat alias so the
# remote code runs unmodified (same drift that hit VisRAG-Ret).
from transformers.cache_utils import DynamicCache as _DynamicCache
if not hasattr(_DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length=0, layer_idx=0):
        return self.get_seq_length(layer_idx)
    _DynamicCache.get_usable_length = _get_usable_length

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

OCR_PATH = f"{PSR_ROOT}/baselines_v2/ocr_text.json"


def load_ocr():
    return json.load(open(OCR_PATH))


def run_bge_m3():
    from FlagEmbedding import BGEM3FlagModel
    TAG = "bge_m3_ocr"
    INDEX = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"
    t0 = time.time()
    print(f"[bge-m3] loading model")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    ocr = load_ocr()
    pages = load_pages()
    assert [r["image_path"] for r in ocr] == [p["image_path"] for p in pages]
    if os.path.exists(INDEX):
        page_embs = torch.load(INDEX, weights_only=False)
        print(f"[bge-m3] cached -> {tuple(page_embs.shape)}")
    else:
        out = model.encode([r["ocr_text"] or " " for r in ocr], batch_size=16, max_length=8192,
                           return_dense=True, return_sparse=False, return_colbert_vecs=False)
        page_embs = torch.tensor(out["dense_vecs"], dtype=torch.float32)
        os.makedirs(os.path.dirname(INDEX), exist_ok=True)
        torch.save(page_embs, INDEX)
        print(f"[bge-m3] saved {tuple(page_embs.shape)}")
    t_index = time.time() - t0
    test = load_test()
    qs = [r["query"] for r in test]
    out = model.encode(qs, batch_size=64, max_length=512,
                       return_dense=True, return_sparse=False, return_colbert_vecs=False)
    q_embs = torch.tensor(out["dense_vecs"], dtype=torch.float32)
    page_embs = F.normalize(page_embs.float(), dim=-1)
    q_embs = F.normalize(q_embs, dim=-1)
    score_matrix = (q_embs @ page_embs.T).numpy()
    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": "BAAI/bge-m3", "t_index_sec": t_index})


def run_nv_embed():
    from transformers import AutoModel
    TAG = "nv_embed_v2_ocr"
    INDEX = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"
    t0 = time.time()
    print(f"[nv-embed] loading model")
    model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda").eval()
    ocr = load_ocr()
    pages = load_pages()
    task = "Given a question about an engineering standard plan, retrieve the relevant plan sheet."

    def _encode(texts, instr=None, batch=4, max_len=4096):
        embs = []
        for i in tqdm(range(0, len(texts), batch)):
            chunk = texts[i:i + batch]
            with torch.no_grad():
                # NV-Embed-v2 uses .encode() with optional instruction
                e = model.encode(chunk, instruction=instr or "", max_length=max_len)
            embs.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(embs, dim=0)

    if os.path.exists(INDEX):
        page_embs = torch.load(INDEX, weights_only=False)
        print(f"[nv-embed] cached -> {tuple(page_embs.shape)}")
    else:
        doc_texts = [r["ocr_text"] or " " for r in ocr]
        page_embs = _encode(doc_texts, instr="", batch=2, max_len=4096)
        os.makedirs(os.path.dirname(INDEX), exist_ok=True)
        torch.save(page_embs, INDEX)
        print(f"[nv-embed] saved {tuple(page_embs.shape)}")
    t_index = time.time() - t0
    test = load_test()
    qs = [r["query"] for r in test]
    q_instr = f"Instruct: {task}\nQuery: "
    q_embs = _encode([q_instr + q for q in qs], batch=8, max_len=512)
    score_matrix = (q_embs.float() @ page_embs.float().T).numpy()
    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": "nvidia/NV-Embed-v2", "t_index_sec": t_index})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["bge_m3", "nv_embed"], required=True)
    args = ap.parse_args()
    if args.model == "bge_m3":
        run_bge_m3()
    else:
        run_nv_embed()
