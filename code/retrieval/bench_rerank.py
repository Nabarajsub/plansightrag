"""Two-stage retrieve -> rerank on the 424-pair test split.

Stage 1: ColPali (our backbone) retrieves the top-K candidate pages.
Stage 2: MonoQwen2-VL-v0.1 (lightonai, Apache-2.0 pointwise visual reranker)
         re-scores each (query, candidate page image) by P("True") and reorders.

Reports base Recall@5, the Recall@K ceiling (max a reranker could reach), and the
reranked Recall@5 -- i.e., how much a visual reranker lifts the proposed pipeline.

Output: reports/rerank_monoqwen_424.json, reports/_tbl_rerank.tex
Apache-2.0 reranker (deployable). No Gemini.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import gc, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, f"{PSR_ROOT}/baselines_v2")
from _eval_lib import load_pages, load_test, REPORTS

import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *a, **k: None

ROOT = f"{PSR_ROOT}"
V3_INDEX = f"{ROOT}/build_v3/all_dot_index_v3.pt"
COLPALI_ID = "vidore/colpali-v1.2"
RERANK_ID = "lightonai/MonoQwen2-VL-v0.1"
RERANK_BASE_PROC = "Qwen/Qwen2-VL-2B-Instruct"
TOPK = 20
OUT_JSON = os.path.join(REPORTS, "rerank_monoqwen_424.json")
OUT_TEX = os.path.join(REPORTS, "_tbl_rerank.tex")


def free(*o):
    for x in o:
        del x
    gc.collect(); torch.cuda.empty_cache()


def base(p):
    return os.path.basename(p)


# -------------------------------------------------- stage 1: ColPali top-K
def colpali_topk(pages, test):
    from colpali_engine.models import ColPali, ColPaliProcessor
    dev = "cuda"
    proc = ColPaliProcessor.from_pretrained(COLPALI_ID)
    cp = ColPali.from_pretrained(COLPALI_ID, torch_dtype=torch.bfloat16, device_map=dev).eval()
    qs = [r["query"] for r in test]
    qemb = []
    for s in range(0, len(qs), 16):
        b = proc.process_queries(qs[s:s + 16]).to(dev)
        with torch.no_grad():
            out = cp(**b)
        for j in range(out.shape[0]):
            qemb.append(out[j].to(torch.float32))
    recs = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    p2e = {r["metadata"]["image_path"]: r["embedding"].to(torch.float32) for r in recs}
    paths = [p["image_path"] for p in pages]
    embs = [p2e.get(pp) for pp in paths]
    score = np.full((len(qemb), len(paths)), -1e30, dtype=np.float32)
    for pi, e in enumerate(tqdm(embs, desc="ColPali MaxSim")):
        if e is None:
            continue
        eg = e.to(dev)
        for qi, qe in enumerate(qemb):
            score[qi, pi] = torch.matmul(qe.to(dev), eg.T).max(dim=-1).values.sum().item()
    free(cp)
    topk = [[paths[j] for j in np.argsort(score[qi])[::-1][:TOPK]] for qi in range(len(qs))]
    return topk


# -------------------------------------------------- stage 2: MonoQwen rerank
def rerank(test, topk):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(RERANK_BASE_PROC)
    m = Qwen2VLForConditionalGeneration.from_pretrained(
        RERANK_ID, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa").eval()
    true_id = proc.tokenizer.convert_tokens_to_ids("True")
    false_id = proc.tokenizer.convert_tokens_to_ids("False")

    def score(query, img):
        prompt = ("Assert the relevance of the previous image document to the following "
                  "query, answer True or False. The query: " + query)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=text, images=img, return_tensors="pt").to(m.device)
        with torch.no_grad():
            logit = m(**inp).logits[:, -1, :]
        pr = torch.softmax(logit[:, [true_id, false_id]].float(), dim=-1)
        return pr[0, 0].item()

    reranked = []
    cache = {}
    for n, (r, cands) in enumerate(zip(test, topk)):
        scored = []
        for pth in cands:
            if pth not in cache:
                try:
                    im = Image.open(pth).convert("RGB")
                    # downscale: full-res plan sheets explode the reranker's vision
                    # token count and caused the previous run to time out.
                    if max(im.size) > 1024:
                        s = 1024 / max(im.size)
                        im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
                    cache[pth] = im
                except Exception:
                    cache[pth] = None
            img = cache[pth]
            scored.append((pth, score(r["query"], img) if img is not None else -1.0))
        scored.sort(key=lambda x: -x[1])
        reranked.append([pth for pth, _ in scored])
        cache.clear()
        if (n + 1) % 25 == 0:
            print(f"  [rerank] {n+1}/{len(test)}")
    free(m)
    return reranked


def recall_at(test, lists, k):
    hits = [int(base(test[i]["image_path"]) in {base(p) for p in lists[i][:k]})
            for i in range(len(test))]
    return 100.0 * sum(hits) / len(hits), hits


def main():
    pages = load_pages()
    test = load_test()
    print(f"[data] {len(pages)} pages, {len(test)} queries; TOPK={TOPK}")

    topk = colpali_topk(pages, test)
    base_r5, _ = recall_at(test, topk, 5)
    ceil_rk, _ = recall_at(test, topk, TOPK)
    print(f"[stage1] ColPali R@5={base_r5:.2f}%  R@{TOPK}(ceiling)={ceil_rk:.2f}%")

    reranked = rerank(test, topk)
    rr_r5, _ = recall_at(test, reranked, 5)
    print(f"[stage2] reranked R@5={rr_r5:.2f}%")

    out = {"topk": TOPK, "base_recall@5": base_r5, "ceiling_recall@k": ceil_rk,
           "reranked_recall@5": rr_r5, "lift_pp": rr_r5 - base_r5,
           "reranker": RERANK_ID, "base_retriever": COLPALI_ID}
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    L = ["% Auto-generated by baselines_v2/bench_rerank.py",
         "\\begin{table}[!t]",
         f"\\caption{{Two-stage retrieve--rerank on the 424-pair test split. ColPali retrieves "
         f"the top-{TOPK}; MonoQwen2-VL-v0.1 (Apache-2.0 pointwise visual reranker) reorders them. "
         f"The top-{TOPK} recall is the reranker's ceiling.}}\\label{{tbl:rerank}}",
         "\\centering", "\\small", "\\begin{tabular}{lc}", "\\toprule",
         "Configuration & Recall@5 (\\%) \\\\", "\\midrule",
         f"ColPali (Stage 1 only) & {base_r5:.2f} \\\\",
         f"ColPali + MonoQwen2-VL rerank & {rr_r5:.2f} \\\\",
         "\\midrule",
         f"\\emph{{Top-{TOPK} ceiling}} & \\emph{{{ceil_rk:.2f}}} \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(OUT_TEX, "w").write("\n".join(L) + "\n")
    print(f"\n[done] base R@5={base_r5:.2f}  reranked R@5={rr_r5:.2f}  "
          f"lift={rr_r5-base_r5:+.2f}pp  (ceiling {ceil_rk:.2f})")


if __name__ == "__main__":
    main()
