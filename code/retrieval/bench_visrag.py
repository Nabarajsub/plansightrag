"""Strong baseline: VisRAG-Ret (openbmb/VisRAG-Ret).

Single-vector dual-stream visual retrieval (MiniCPM-V backbone).
Index 1,898 pages, evaluate Recall@5 on the 424-pair test split.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# VisRAG-Ret's vendored modeling_minicpm.py was written for an older transformers
# where DynamicCache exposed get_usable_length(); it was renamed get_seq_length().
# Add a back-compat alias so the remote code runs unmodified.
from transformers.cache_utils import DynamicCache as _DynamicCache
if not hasattr(_DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length=0, layer_idx=0):
        return self.get_seq_length(layer_idx)
    _DynamicCache.get_usable_length = _get_usable_length

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, recall_at_k, save_report

MODEL_ID = "openbmb/VisRAG-Ret"
TAG = "visrag_ret"
INDEX_PATH = f"{PSR_ROOT}/baselines_v2/indices/{TAG}.pt"
BATCH = 4


def _weighted_mean_pool(last_hidden, attention_mask):
    """Position-weighted mean pooling, exactly per the openbmb/VisRAG-Ret model
    card: weights increase toward the end of the sequence via the cumulative
    sum of the attention mask, so padding contributes zero."""
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(last_hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float().clamp(min=1e-9)
    return s / d


@torch.no_grad()
def encode(model, tokenizer, text_list, image_list):
    """VisRAG-Ret has no `.encode`; the model is called directly and returns an
    object with `.last_hidden_state` and `.attention_mask`. For documents pass
    image_list with empty text; for queries pass text with image_list=None.
    Matches the model card's `encode()` example."""
    if image_list is None:
        inputs = {"text": text_list, "image": [None] * len(text_list), "tokenizer": tokenizer}
    else:
        inputs = {"text": text_list, "image": image_list, "tokenizer": tokenizer}
    outputs = model(**inputs)
    reps = _weighted_mean_pool(outputs.last_hidden_state, outputs.attention_mask)
    return F.normalize(reps, p=2, dim=-1).to(torch.float16).cpu()


def main():
    """Lazy bridge: the VisRAG-Ret model exposes a clean .encode() in its
    custom code. Use that to avoid reimplementing tokenization."""
    device = "cuda"
    t0 = time.time()
    print(f"[load] {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    print(f"[load] {time.time() - t0:.1f}s")

    pages = load_pages()

    if os.path.exists(INDEX_PATH):
        page_embs = torch.load(INDEX_PATH, weights_only=False)
        print(f"[index] cached -> {tuple(page_embs.shape)}")
    else:
        # Use the model's documented .encode() with INSTRUCTION + image inputs
        page_embs_list = []
        for i in tqdm(range(0, len(pages), BATCH), desc="encode pages"):
            chunk = pages[i:i + BATCH]
            imgs = [Image.open(p["image_path"]).convert("RGB") for p in chunk]
            # For docs, instruction is empty; the image carries the content.
            embs = encode(model, tokenizer, ["" for _ in imgs], imgs)
            page_embs_list.append(embs)
        page_embs = torch.cat(page_embs_list, dim=0)
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        torch.save(page_embs, INDEX_PATH)
        print(f"[index] saved {tuple(page_embs.shape)} -> {INDEX_PATH}")
    t_index = time.time() - t0

    test = load_test()
    queries = [r["query"] for r in test]
    INSTR = "Represent this query for retrieving relevant documents: "
    print(f"[query] encoding {len(queries)} queries")
    q_embs_list = []
    for i in tqdm(range(0, len(queries), BATCH * 4), desc="encode queries"):
        chunk = queries[i:i + BATCH * 4]
        embs = encode(model, tokenizer, [INSTR + q for q in chunk], None)
        q_embs_list.append(embs)
    q_embs = torch.cat(q_embs_list, dim=0).to(torch.float32)
    page_embs = page_embs.to(torch.float32)
    score_matrix = (q_embs @ page_embs.T).numpy()

    overall, per_agency, per_cat, hits = recall_at_k(score_matrix, test, pages)
    save_report(TAG, overall, per_agency, per_cat, hits, len(pages),
                time.time() - t0, extra={"model_id": MODEL_ID, "t_index_sec": t_index})


if __name__ == "__main__":
    main()
