"""H4 (proper test): tile-level vs full-page JUDGE ACCURACY on the dense-page
subset of the 424-pair test split.

H4 asks whether high-resolution tiling lifts *judge accuracy* (not just Recall@5)
on dense pages. We:
  1. Define the dense-page subset = test queries whose target page yields the most
     400-DPI tiles (top tertile by tile count) -- the large multi-view / schedule
     sheets where full-page retrieval loses fine detail.
  2. For each dense query, retrieve top-3 pages two ways with the SAME adopted
     ColNomic-3B retriever: full-page (cached 200-DPI index) and tile-level
     (1024px tiles with 256px overlap, encoded on the fly).
  3. Generate an answer with Qwen2.5-VL-7B from each condition's top-3 pages.
  4. Judge both answers against the reference with a local Qwen2.5-VL-72B judge.
  5. Report judge accuracy per condition and the tile-minus-full gain vs the 20-pp
     H4 threshold.

Outputs: reports/tiling_judge_colnomic_424.json and its LaTeX table.
Models loaded/freed in sequence (ColNomic-3B -> Qwen-7B -> Qwen-72B-4bit).
"""
from __future__ import annotations
import gc, json, os, sys
import numpy as np
import torch
from PIL import Image

# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _eval_lib import load_pages, load_test, REPORTS

ROOT = CLUSTER_ROOT
V3_INDEX = f"{ROOT}/build_v3/all_dot_index_v3.pt"
TILE_INDEX = f"{ROOT}/baselines_v2/indices/tiles_400dpi.pt"
COLNOMIC_ID = "nomic-ai/colnomic-embed-multimodal-3b"
PAGE_CACHE = f"{ROOT}/baselines_v2/cache_colnomic_pages.pt"
TILE_SIZE, TILE_OVERLAP, TILE_DPI = 1024, 256, 400
GEN_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
JUDGE_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
OUT_JSON = os.path.join(REPORTS, "tiling_judge_colnomic_424.json")
OUT_TEX = os.path.join(REPORTS, "_tbl_tiling_judge_colnomic.tex")

DENSE_FRACTION = 0.34   # top-tertile of pages by tile count = "dense"
TOPK = 3
H4_THRESHOLD = 20.0
N_BOOT, SEED = 10000, 42


def free(*objs):
    for o in objs:
        del o
    gc.collect(); torch.cuda.empty_cache()


def bootstrap(flags):
    rng = np.random.default_rng(SEED)
    a = np.asarray(flags, dtype=np.float32)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, a.size, size=(N_BOOT, a.size))
    s = a[idx].mean(axis=1) * 100.0
    return float(a.mean() * 100), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ---------------------------------------------------------------- retrieval
def _tiles_of(path):
    """400-DPI tiles for one page, matching colnomic_ablations.py exactly."""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    scale = TILE_DPI / 200.0
    img = img.resize((int(W * scale), int(H * scale)), Image.BICUBIC)
    W, H = img.size
    stride = TILE_SIZE - TILE_OVERLAP
    out = []
    for y in range(0, max(1, H - TILE_OVERLAP), stride):
        for x in range(0, max(1, W - TILE_OVERLAP), stride):
            out.append(img.crop((x, y, min(x + TILE_SIZE, W), min(y + TILE_SIZE, H))))
    return out or [img]


def retrieve_topk(pages, test, dense_q_idx):
    """Return {qi: (full_top3_paths, tile_top3_paths)} using the ADOPTED backbone."""
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    dev = "cuda"
    proc = ColQwen2_5_Processor.from_pretrained(COLNOMIC_ID)
    model = ColQwen2_5.from_pretrained(COLNOMIC_ID, torch_dtype=torch.bfloat16,
                                       device_map=dev).eval()

    qlist = [test[i]["query"] for i in dense_q_idx]
    qemb = []
    for s0 in range(0, len(qlist), 16):
        batch = proc.process_queries(qlist[s0:s0 + 16]).to(dev)
        with torch.no_grad():
            out = model(**batch)
        for j in range(out.shape[0]):
            qemb.append(out[j].to(torch.float32))
    print(f"[retr] {len(qemb)} dense queries encoded", flush=True)

    n = len(pages)
    paths = [p["image_path"] for p in pages]

    # ---- full page: cached ColNomic index -------------------------------
    cache = torch.load(PAGE_CACHE, map_location="cpu", weights_only=False)
    pembs = cache["pembs"] if isinstance(cache, dict) else cache
    score_fp = np.full((len(qemb), n), -1e30, dtype=np.float32)
    for pi in range(n):
        pe = pembs[pi]
        if pe is None:
            continue
        peg = pe.to(dev).to(torch.float32)
        for k, qe in enumerate(qemb):
            score_fp[k, pi] = torch.matmul(qe.to(dev), peg.T).max(dim=-1).values.sum().item()
        del peg
    del pembs, cache
    gc.collect(); torch.cuda.empty_cache()
    print("[retr] full-page scores done", flush=True)

    # ---- tile level: encode tiles on the fly, page score = max over tiles
    score_tl = np.full((len(qemb), n), -1e30, dtype=np.float32)
    for pi in range(n):
        try:
            crops = _tiles_of(paths[pi])
        except Exception as e:
            print(f"  [tile] skip {paths[pi]}: {e}", flush=True)
            continue
        best = np.full(len(qemb), -1e30, dtype=np.float32)
        for s0 in range(0, len(crops), 4):
            bt = proc.process_images(crops[s0:s0 + 4]).to(dev)
            with torch.no_grad():
                te = model(**bt)
            for t in range(te.shape[0]):
                teg = te[t].to(torch.float32)
                for k, qe in enumerate(qemb):
                    v = torch.matmul(qe.to(dev), teg.T).max(dim=-1).values.sum().item()
                    if v > best[k]:
                        best[k] = v
            del bt, te
        score_tl[:, pi] = best
        if (pi + 1) % 100 == 0:
            print(f"  [tile] {pi + 1}/{n} pages", flush=True)
            torch.cuda.empty_cache()
    free(model)

    qmap = {}
    for k, qi in enumerate(dense_q_idx):
        ftop = [paths[j] for j in np.argsort(score_fp[k])[::-1][:TOPK]]
        ttop = [paths[j] for j in np.argsort(score_tl[k])[::-1][:TOPK]]
        qmap[qi] = (ftop, ttop)
    return qmap


# ---------------------------------------------------------------- VQA
def vqa_answers(test, dense_q_idx, qmap):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        GEN_ID, torch_dtype=torch.bfloat16, device_map="auto").eval()
    p = AutoProcessor.from_pretrained(GEN_ID, trust_remote_code=True)

    def gen(query, paths):
        imgs = []
        for pth in paths[:TOPK]:
            try:
                imgs.append(Image.open(pth).convert("RGB"))
            except Exception:
                pass
        if not imgs:
            return "[ERROR: no image]"
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text":
            "You are an expert civil engineer. Using ONLY the provided standard-plan "
            "images, answer precisely (give values with units).\n\nQuestion: " + query})
        msgs = [{"role": "user", "content": content}]
        text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = p(text=[text], images=imgs, padding=True, return_tensors="pt")
        inp = {k: v.to(m.device) for k, v in inp.items()}
        with torch.no_grad():
            o = m.generate(**inp, max_new_tokens=256, do_sample=False)
        return p.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    ans = {}
    for n, qi in enumerate(dense_q_idx):
        ftop, ttop = qmap[qi]
        ans[qi] = {"full": gen(test[qi]["query"], ftop),
                   "tile": gen(test[qi]["query"], ttop)}
        if (n + 1) % 20 == 0:
            print(f"  [vqa] {n+1}/{len(dense_q_idx)}")
    free(m, p)
    return ans


# ---------------------------------------------------------------- judge
def judge_all(test, dense_q_idx, ans):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        JUDGE_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
    p = AutoProcessor.from_pretrained(JUDGE_ID, trust_remote_code=True)

    def judge_one(img, q, ref, a):
        if isinstance(a, str) and a.startswith("[ERROR"):
            return 0
        prompt = (f"You are grading an answer to a question about an engineering drawing.\n\n"
                  f"Question: {q}\nReference answer: {ref}\nSubmitted answer: {a}\n\n"
                  f"Does the submitted answer convey the same essential information as the "
                  f"reference (allowing for paraphrase, units, rounding)? Reply exactly YES or NO.")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = p(text=[text], images=[img], padding=True, return_tensors="pt")
        inp = {k: v.to(m.device) for k, v in inp.items()}
        with torch.no_grad():
            o = m.generate(**inp, max_new_tokens=8, do_sample=False)
        r = p.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return int("yes" in r.strip().lower()[:5])

    full_flags, tile_flags = [], []
    for n, qi in enumerate(dense_q_idx):
        r = test[qi]
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            w, h = img.size
            if max(w, h) > 1100:
                s = 1100 / max(w, h); img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        except Exception:
            continue
        full_flags.append(judge_one(img, r["query"], r.get("answer", ""), ans[qi]["full"]))
        tile_flags.append(judge_one(img, r["query"], r.get("answer", ""), ans[qi]["tile"]))
        if (n + 1) % 20 == 0:
            print(f"  [judge] {n+1}/{len(dense_q_idx)}")
    free(m, p)
    return full_flags, tile_flags


def main():
    pages = load_pages()
    test = load_test()

    # dense subset by tile count (no GPU needed -- just lengths)
    page_tiles = torch.load(TILE_INDEX, weights_only=False)
    n_tiles = np.array([len(t) for t in page_tiles])
    del page_tiles; gc.collect()
    thr = np.quantile(n_tiles, 1 - DENSE_FRACTION)
    dense_pages = {pages[i]["image_path"] for i in range(len(pages)) if n_tiles[i] >= thr}
    dense_q_idx = [i for i, r in enumerate(test) if r["image_path"] in dense_pages]
    print(f"[dense] tile-count threshold >= {thr:.0f}; {len(dense_pages)} dense pages; "
          f"{len(dense_q_idx)} dense queries")

    qmap = retrieve_topk(pages, test, dense_q_idx)
    print("[retr] done; running VQA")
    ans = vqa_answers(test, dense_q_idx, qmap)
    print("[vqa] done; judging")
    full_flags, tile_flags = judge_all(test, dense_q_idx, ans)

    fp_pt, fp_lo, fp_hi = bootstrap(full_flags)
    tl_pt, tl_lo, tl_hi = bootstrap(tile_flags)
    gain = tl_pt - fp_pt
    verdict = "supported" if gain >= H4_THRESHOLD else "rejected"

    out = {"n_dense": len(dense_q_idx), "tile_threshold": float(thr),
           "retriever": COLNOMIC_ID, "backbone_note": "adopted ColNomic-3B backbone (H4 judge endpoint)",
           "full_page": {"judge_acc": fp_pt, "ci": [fp_lo, fp_hi]},
           "tile_level": {"judge_acc": tl_pt, "ci": [tl_lo, tl_hi]},
           "gain_pp": gain, "threshold_pp": H4_THRESHOLD, "H4_verdict": verdict,
           "full_flags": full_flags, "tile_flags": tile_flags}
    os.makedirs(REPORTS, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    L = ["% Auto-generated by baselines_v2/bench_tiling_judge_424.py",
         "\\begin{table}[!t]",
         f"\\caption{{Judge accuracy of full-page vs.\\ tile-level retrieval on the dense-page "
         f"subset ($N={len(dense_q_idx)}$ queries whose target page is in the top tertile by "
         f"400~DPI tile count) of the 424-pair test split. Answers generated by Qwen2.5-VL-7B from "
         f"the top-{TOPK} retrieved pages and scored by a local Qwen2.5-VL-72B judge; 95\\% bootstrap "
         f"CIs. H4 requires a $\\ge${H4_THRESHOLD:.0f}~pp tile-over-full gain.}}\\label{{tbl:tiling-judge}}",
         "\\centering", "\\small",
         "\\begin{tabular}{lc}", "\\toprule",
         "Retrieval granularity & Judge accuracy (\\%) \\\\", "\\midrule",
         f"Full-page (200 DPI) & {fp_pt:.2f} [{fp_lo:.1f}, {fp_hi:.1f}] \\\\",
         f"Tile-level (400 DPI) & {tl_pt:.2f} [{tl_lo:.1f}, {tl_hi:.1f}] \\\\",
         "\\midrule",
         f"\\textbf{{Gain (tile $-$ full)}} & \\textbf{{{gain:+.2f}~pp}} \\\\",
         "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(OUT_TEX, "w").write("\n".join(L) + "\n")

    print(f"\n[done] dense N={len(dense_q_idx)}  full={fp_pt:.2f}%  tile={tl_pt:.2f}%  "
          f"gain={gain:+.2f}pp  -> H4 {verdict}")
    print(f"  -> {OUT_JSON}\n  -> {OUT_TEX}")


if __name__ == "__main__":
    main()
