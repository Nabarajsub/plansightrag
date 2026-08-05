"""Strong baseline: DSE-Qwen2-2B-MRL (Document Screenshot Embedding).

Single-vector visual retrieval. Index 1,898 pages, evaluate Recall@5
on the 424-pair page-disjoint test split. Cosine similarity.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# MrLight/dse-qwen2-2b-mrl-v1 ships a .bin checkpoint; torch 2.5.1 (<2.6) trips
# transformers' CVE-2025-32434 guard. The checkpoint is the official published
# release and is still loaded with weights_only=True, so disable the guard.
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

MODEL_ID = "MrLight/dse-qwen2-2b-mrl-v1"
TAG = "dse_qwen2_2b"
INDEX_PATH = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"
BATCH = 1
# DSE pools the last hidden state, so the lm_head logits are never used. We replace
# lm_head with Identity (below) to avoid the 18GB (vocab x seq) projection that OOMs
# the 24GB A30 -- this lets us keep a usable resolution instead of crushing it.
MAX_PIXELS = 1280 * 28 * 28
MIN_PIXELS = 4 * 28 * 28

# Recommended DSE prompt templates (per model card)
DOC_PROMPT_PREFIX = "What is shown in this image?"
QUERY_PROMPT_PREFIX = "Query: {q}"


# DSE-Qwen2 extracts the embedding from an explicit <|endoftext|> token appended
# after the chat template, and -- critically -- encodes *queries* with a dummy
# image placeholder too (the model is trained that way). Both are required for
# the embeddings to be meaningful; without them retrieval collapses to ~chance.
DUMMY_IMG = Image.new("RGB", (28, 28))


def _last_token_emb(out, attention_mask):
    last = out.hidden_states[-1]  # (B, T, D)
    seq_lens = attention_mask.sum(dim=1) - 1
    emb = last[torch.arange(last.shape[0]), seq_lens]
    return F.normalize(emb, dim=-1).to(torch.float16).cpu()


@torch.no_grad()
def encode_image(model, processor, pil_images, device):
    """Encode a batch of images. Returns L2-normalized (B, D) tensor on CPU."""
    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": DOC_PROMPT_PREFIX}]}]
        for img in pil_images]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
             for m in msgs]
    inputs = processor(text=texts, images=pil_images, return_tensors="pt", padding=True).to(device)
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    return _last_token_emb(out, inputs["attention_mask"])


@torch.no_grad()
def encode_text(model, processor, queries, device):
    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": DUMMY_IMG},
        {"type": "text", "text": QUERY_PROMPT_PREFIX.format(q=q)}]}]
        for q in queries]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
             for m in msgs]
    inputs = processor(text=texts, images=[DUMMY_IMG] * len(queries),
                       return_tensors="pt", padding=True).to(device)
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    return _last_token_emb(out, inputs["attention_mask"])


def main():
    device = "cuda"
    t0 = time.time()
    print(f"[load] {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True,
                                              min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa"
    ).eval()
    # We only consume hidden states; drop the lm_head vocab projection to avoid OOM.
    model.lm_head = torch.nn.Identity()
    print(f"[load] {time.time() - t0:.1f}s")

    pages = load_pages()

    if os.path.exists(INDEX_PATH):
        page_embs = torch.load(INDEX_PATH, weights_only=False)
        print(f"[index] cached -> {len(page_embs)} pages")
    else:
        page_embs_list = []
        for i in tqdm(range(0, len(pages), BATCH), desc="encode pages"):
            chunk = pages[i:i + BATCH]
            imgs = [Image.open(p["image_path"]).convert("RGB") for p in chunk]
            embs = encode_image(model, processor, imgs, device)
            page_embs_list.append(embs)
        page_embs = torch.cat(page_embs_list, dim=0)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        torch.save(page_embs, INDEX_PATH)
        print(f"[index] saved {tuple(page_embs.shape)} -> {INDEX_PATH}")
    t_index = time.time() - t0

    test = load_test()
    queries = [r["query"] for r in test]
    print(f"[query] encoding {len(queries)} queries")
    q_embs_list = []
    for i in tqdm(range(0, len(queries), BATCH * 4), desc="encode queries"):
        chunk = queries[i:i + BATCH * 4]
        embs = encode_text(model, processor, chunk, device)
        q_embs_list.append(embs)
    q_embs = torch.cat(q_embs_list, dim=0).to(torch.float32)  # (Nq, D)
    page_embs = page_embs.to(torch.float32)                    # (Np, D)
    score_matrix = (q_embs @ page_embs.T).numpy()              # cosine since both L2-normed

    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": MODEL_ID, "t_index_sec": t_index})


if __name__ == "__main__":
    main()
