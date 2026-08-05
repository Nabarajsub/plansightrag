"""Weak OCR-free document VLM baselines (re-run): Nougat / Pix2Struct / UDOP.

Each model decodes the page image to text; we then embed that text with MiniLM
and do dense retrieval (matching the original tbl:comparison protocol).
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
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoProcessor, AutoTokenizer,
    NougatProcessor, VisionEncoderDecoderModel,
    Pix2StructProcessor, Pix2StructForConditionalGeneration,
    UdopProcessor, UdopForConditionalGeneration,
)

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def decode_nougat(pages, device):
    processor = NougatProcessor.from_pretrained("facebook/nougat-small")
    model = VisionEncoderDecoderModel.from_pretrained(
        "facebook/nougat-small", torch_dtype=torch.float16).to(device).eval()
    texts = []
    for p in tqdm(pages, desc="nougat decode"):
        try:
            pil = Image.open(p["image_path"]).convert("RGB")
            pix = processor(pil, return_tensors="pt").pixel_values.to(device, torch.float16)
            with torch.no_grad():
                out = model.generate(pix, max_new_tokens=256, do_sample=False)
            t = processor.batch_decode(out, skip_special_tokens=True)[0]
            texts.append(t or " ")
        except Exception as e:
            texts.append(" ")
    return texts


def decode_pix2struct(pages, device):
    processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
    model = Pix2StructForConditionalGeneration.from_pretrained(
        "google/pix2struct-base", torch_dtype=torch.float16).to(device).eval()
    texts = []
    for p in tqdm(pages, desc="pix2struct decode"):
        try:
            pil = Image.open(p["image_path"]).convert("RGB")
            inp = processor(images=pil, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256)
            t = processor.batch_decode(out, skip_special_tokens=True)[0]
            texts.append(t or " ")
        except Exception:
            texts.append(" ")
    return texts


def decode_udop(pages, device):
    processor = UdopProcessor.from_pretrained("microsoft/udop-large")
    model = UdopForConditionalGeneration.from_pretrained(
        "microsoft/udop-large", torch_dtype=torch.float16).to(device).eval()
    texts = []
    for p in tqdm(pages, desc="udop decode"):
        try:
            pil = Image.open(p["image_path"]).convert("RGB")
            inp = processor(pil, text="Describe this document.", return_tensors="pt",
                            truncation=True, max_length=512).to(device, torch.float16)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=256)
            t = processor.batch_decode(out, skip_special_tokens=True)[0]
            texts.append(t or " ")
        except Exception:
            texts.append(" ")
    return texts


DECODERS = {"nougat": decode_nougat, "pix2struct": decode_pix2struct, "udop": decode_udop}


def run(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TAG = f"{model_name}_decode_minilm"
    INDEX = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"
    DECODES = f"{PSR_ROOT}/baselines_v2/indices/{TAG}_decoded.json"
    t0 = time.time()
    pages = load_pages()

    if os.path.exists(DECODES):
        texts = json.load(open(DECODES))
    else:
        texts = DECODERS[model_name](pages, device)
        os.makedirs(os.path.dirname(DECODES), exist_ok=True)
        with open(DECODES, "w") as f:
            json.dump(texts, f)

    print(f"[{model_name}] decoded {len(texts)} pages; embedding")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    if os.path.exists(INDEX):
        page_embs = torch.load(INDEX, weights_only=False)
    else:
        page_embs = torch.tensor(embedder.encode(texts, batch_size=128, normalize_embeddings=True))
        torch.save(page_embs, INDEX)
    t_index = time.time() - t0
    test = load_test()
    q_embs = torch.tensor(embedder.encode([r["query"] for r in test], batch_size=128,
                                           normalize_embeddings=True))
    score_matrix = (q_embs @ page_embs.T).numpy()
    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"decoder": model_name, "embedder": EMBED_MODEL,
                                          "t_index_sec": t_index})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(DECODERS), required=True)
    args = ap.parse_args()
    run(args.model)
