"""Full-data per-category appendix table on the 424-pair page-disjoint test split.

Replaces the legacy 78-question WYDOT subset table (tbl:bench-stats-primary) with
the whole-dataset equivalent, mirroring the main paper's scale:

  Per category (Dimensional / Visual / Logical / Hallucination), report
    - Recall@5 (Hit): clean ColPali zero-shot over the 1,898-page five-DOT index
      (generator-independent), with 95% bootstrap CIs.
    - Judge Accuracy (Judge): per open generator, with 95% bootstrap CIs, scored
      by a local Qwen2.5-VL-72B judge (no proprietary API).

Generators: Qwen2.5-VL-7B, Qwen2.5-VL-72B (per-query verdicts reused from the
existing 424-split judge run, vqa_eval/vqa_results.json), and InternVL2_5-8B
(generated fresh here with the top-3 ColPali-retrieved pages, then judged by the
same local Qwen-72B judge). No Gemini anywhere.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import gc, json, os
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

ROOT = f"{PSR_ROOT}"
TEST = f"{ROOT}/lora_finetune/split_test.jsonl"
V3_INDEX = f"{ROOT}/build_v3/all_dot_index_v3.pt"
RETR = f"{ROOT}/vqa_eval/retrieval_test.json"
VQA_RESULTS = f"{ROOT}/vqa_eval/vqa_results.json"   # has Qwen-7B/72B per-query verdicts
OUT_JSON = f"{ROOT}/baselines_v2/reports/bench_stats_424.json"
OUT_TEX = f"{ROOT}/baselines_v2/reports/_tbl_bench_stats_424.tex"

INTERNVL = "OpenGVLab/InternVL2_5-8B"
JUDGE_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
N_BOOT, SEED = 10000, 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# (full category name in split_test, short display label)
CAT_ORDER = [
    ("Dimensional Accuracy", "Dimensional"),
    ("Visual Interpretation", "Visual"),
    ("Logical Reasoning", "Logical"),
    ("Hallucination Rate", "Hallucination"),
]
INSTR = ("You are an expert civil engineer reading state DOT standard plans. "
         "Using ONLY the provided plan images, answer the question concisely. "
         "If a numeric value is asked, give the value with units.\n\nQuestion: ")


def load_test():
    return [json.loads(l) for l in open(TEST)]


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def bootstrap_acc(flags, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(flags, dtype=np.float32)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    s = a[idx].mean(axis=1) * 100.0
    return (float(a.mean() * 100.0),
            float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


# ----------------------------------------------------------------------------
# 1. Per-category Recall@5 (clean ColPali over the 1,898-page index)
# ----------------------------------------------------------------------------
def retrieval_per_category(rows):
    dev = "cuda"
    print("[retr] ColPali vidore/colpali-v1.2")
    cp = ColPali.from_pretrained("vidore/colpali-v1.2", torch_dtype=torch.bfloat16,
                                 device_map=dev).eval()
    cpp = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    recs = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    db = [{"emb": r["embedding"].to(dev).to(torch.bfloat16),
           "base": os.path.basename(r["metadata"]["image_path"])} for r in recs]
    print(f"[retr] {len(db)} index pages, {len(rows)} queries")
    hits_by_cat = {c: [] for c, _ in CAT_ORDER}
    for i, r in enumerate(rows):
        inp = cpp.process_queries([r["query"]]).to(dev)
        with torch.no_grad():
            q = cp(**inp)
        scores = np.array([torch.matmul(q, d["emb"].T).max(dim=-1).values.sum(dim=-1).item()
                           for d in db])
        top5 = {db[j]["base"] for j in np.argsort(scores)[::-1][:5]}
        hit = int(os.path.basename(r["image_path"]) in top5)
        if r["category"] in hits_by_cat:
            hits_by_cat[r["category"]].append(hit)
        if (i + 1) % 100 == 0:
            print(f"  [retr] {i+1}/{len(rows)}")
    free(cp)
    out = {}
    allhits = []
    for cat, label in CAT_ORDER:
        out[label] = bootstrap_acc(hits_by_cat[cat])
        allhits += hits_by_cat[cat]
    out["Overall"] = bootstrap_acc(allhits)
    return out, hits_by_cat


# ----------------------------------------------------------------------------
# 2. Reuse Qwen-7B / Qwen-72B per-query verdicts (retrieval technique)
# ----------------------------------------------------------------------------
def reuse_qwen_verdicts(rows):
    # coverage_matrix keys are the qid "image_path|category|query[:40]", so the
    # category is the second pipe-delimited field (no join with split_test needed).
    cov = json.load(open(VQA_RESULTS)).get("coverage_matrix", {})
    out = {}  # model -> {label: [flags]}
    for tag, label in [("qwen7b", "Qwen2.5-VL-7B"), ("qwen72b", "Qwen2.5-VL-72B")]:
        by_cat = {c: [] for c, _ in CAT_ORDER}
        for qid, verds in cov.items():
            parts = qid.split("|")
            cat = parts[1] if len(parts) > 1 else None
            key = f"{tag}::retrieval"
            if cat in by_cat and key in verds:
                by_cat[cat].append(int(bool(verds[key])))
        out[label] = by_cat
    return out


# ----------------------------------------------------------------------------
# 3. InternVL2_5-8B generation (top-3 retrieved pages) + Qwen-72B judge
# ----------------------------------------------------------------------------
def _ivl_transform(size=448):
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def _closest_ratio(ar, ratios):
    best, bd = (1, 1), float("inf")
    for r in ratios:
        d = abs(ar - r[0] / r[1])
        if d < bd:
            bd, best = d, r
    return best


def _ivl_preprocess(image, size=448, max_num=8):
    w, h = image.size
    ratios = sorted({(i, j) for i in range(1, max_num + 1) for j in range(1, max_num + 1)
                     if i * j <= max_num}, key=lambda x: x[0] * x[1])
    tr = _closest_ratio(w / h, ratios)
    img = image.resize((size * tr[0], size * tr[1]))
    tiles = [img.crop(((i % tr[0]) * size, (i // tr[0]) * size,
                       (i % tr[0] + 1) * size, (i // tr[0] + 1) * size))
             for i in range(tr[0] * tr[1])]
    if len(tiles) != 1:
        tiles.append(image.resize((size, size)))
    return tiles


def _ivl_pixels(path, device, max_num=8):
    tfm = _ivl_transform(448)
    tiles = _ivl_preprocess(Image.open(path).convert("RGB"), 448, max_num)
    return torch.stack([tfm(t) for t in tiles]).to(torch.bfloat16).to(device)


def gen_internvl(rows, retr):
    # Use the native HF integration (transformers 4.45+) instead of the vendored
    # remote code: the -hf checkpoint's class properly inherits GenerationMixin and
    # uses the standard messages API, so model.generate() works directly.
    from transformers import AutoProcessor, InternVLForConditionalGeneration
    # Native-HF build of InternVL2.5-8B (the -hf variant the transformers InternVL
    # integration loads cleanly; plain InternVL2_5-8B ships only remote code).
    mid = "OpenGVLab/InternVL2_5-8B-MPO-hf"
    m = InternVLForConditionalGeneration.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map="auto").eval()
    p = AutoProcessor.from_pretrained(mid)
    out = {}
    for i, r in enumerate(rows):
        paths = retr.get(r["query"], [r["image_path"]])[:3] or [r["image_path"]]
        imgs = []
        for pth in paths:
            try:
                imgs.append(Image.open(pth).convert("RGB"))
            except Exception:
                pass
        if not imgs:
            out[r["query"]] = "[ERROR: no image]"
            continue
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": INSTR + r["query"]})
        msgs = [{"role": "user", "content": content}]
        try:
            inp = p.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        return_dict=True, return_tensors="pt").to(m.device)
            with torch.no_grad():
                o = m.generate(**inp, max_new_tokens=256, do_sample=False)
            resp = p.decode(o[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as exc:
            resp = f"[ERROR: {exc}]"
        out[r["query"]] = resp.strip() if isinstance(resp, str) else str(resp)
        if (i + 1) % 50 == 0:
            print(f"  [internvl] {i+1}/{len(rows)}")
    free(m, p)
    return out


def judge_internvl(rows, answers):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        JUDGE_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
    p = AutoProcessor.from_pretrained(JUDGE_ID, trust_remote_code=True)

    def judge_one(img, q, ref, ans):
        if isinstance(ans, str) and ans.startswith("[ERROR"):
            return False
        prompt = (f"You are grading an answer to a question about an engineering drawing.\n\n"
                  f"Question: {q}\nReference answer: {ref}\nSubmitted answer: {ans}\n\n"
                  f"Does the submitted answer convey the same essential information as the "
                  f"reference (allowing for paraphrase, units, and rounding)? "
                  f"Reply with exactly one word: YES or NO.")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = p(text=[text], images=[img], padding=True, return_tensors="pt")
        inp = {k: v.to(m.device) for k, v in inp.items()}
        with torch.no_grad():
            o = m.generate(**inp, max_new_tokens=8, do_sample=False)
        resp = p.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return "yes" in resp.strip().lower()[:5]

    by_cat = {c: [] for c, _ in CAT_ORDER}
    for i, r in enumerate(rows):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            w, h = img.size
            if max(w, h) > 1100:
                s = 1100 / max(w, h)
                img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        except Exception:
            continue
        if r["category"] in by_cat:
            by_cat[r["category"]].append(int(judge_one(img, r["query"], r.get("answer", ""),
                                                       answers.get(r["query"], ""))))
        if (i + 1) % 50 == 0:
            print(f"  [judge] {i+1}/{len(rows)}")
    free(m, p)
    return by_cat


def cat_summary(by_cat):
    d = {}
    allf = []
    for cat, label in CAT_ORDER:
        d[label] = bootstrap_acc(by_cat[cat])
        allf += by_cat[cat]
    d["Overall"] = bootstrap_acc(allf)
    return d


def main():
    rows = load_test()
    retr = json.load(open(RETR)) if os.path.exists(RETR) else {}
    print(f"[424] {len(rows)} test questions")

    hit, _ = retrieval_per_category(rows)
    qwen = reuse_qwen_verdicts(rows)
    judge_tbl = {m: cat_summary(bc) for m, bc in qwen.items()}

    print("[gen] InternVL2_5-8B on 424")
    ivl_ans = gen_internvl(rows, retr)
    print("[judge] Qwen2.5-VL-72B on InternVL answers")
    ivl_bycat = judge_internvl(rows, ivl_ans)
    judge_tbl["InternVL-2.5-8B"] = cat_summary(ivl_bycat)

    out = {"n": len(rows), "n_boot": N_BOOT, "seed": SEED,
           "hit": {k: list(v) for k, v in hit.items()},
           "judge": {m: {k: list(v) for k, v in d.items()} for m, d in judge_tbl.items()}}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    # ---- LaTeX --------------------------------------------------------------
    def hc(label):
        t = hit[label]
        return f"{t[0]:.2f} [{t[1]:.1f}, {t[2]:.1f}]"

    def jc(t):
        return f"{t[0]:.2f} [{t[1]:.1f}, {t[2]:.1f}]"

    L = ["% Auto-generated by baselines_v2/eval_424_judge.py",
         "\\begin{table*}[!t]",
         "\\caption{Per-category Recall@5 (Hit) and Judge Accuracy (Judge) on the full "
         "424-pair page-disjoint test split over the 1{,}898-page five-DOT index, with 95\\% "
         "bootstrap CIs (10{,}000 resamples, seed~42). Recall@5 is clean zero-shot ColPali and is "
         "generator-independent. Judge Accuracy is scored by a local Qwen2.5-VL-72B judge (no "
         "proprietary API). Per-category $N$: Dim$=$112, Vis$=$145, Log$=$120, Hal$=$47.}"
         "\\label{tbl:bench-stats-primary}",
         "\\centering", "\\small",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcc@{}}",
         "\\toprule",
         "Model & Category & Hit (\\%) & Judge (\\%) \\\\",
         "\\midrule"]

    def block(model_label, src):
        cats = [("Dimensional", "Dimensional"), ("Visual", "Visual"),
                ("Logical", "Logical"), ("Hallucination", "Hallucination")]
        lines = []
        for j, (label, disp) in enumerate(cats):
            lead = f"\\multirow{{5}}{{*}}{{{model_label}}}" if j == 0 else ""
            lines.append(f"  {lead} & {disp:<12s} & {hc(label)} & {jc(src[label])} \\\\")
        lines.append(f"  & \\textbf{{Overall}} & \\textbf{{{hc('Overall')}}} & \\textbf{{{jc(src['Overall'])}}} \\\\")
        return "\n".join(lines)

    for i, mdl in enumerate(["Qwen2.5-VL-72B", "Qwen2.5-VL-7B", "InternVL-2.5-8B"]):
        if i:
            L.append("\\midrule")
        L.append(block(mdl, judge_tbl[mdl]))
    L += ["\\bottomrule", "\\end{tabular*}", "\\end{table*}"]
    open(OUT_TEX, "w").write("\n".join(L) + "\n")

    print(f"\n[done] -> {OUT_JSON}\n[done] -> {OUT_TEX}")
    print(f"  Recall@5 overall = {hit['Overall'][0]:.2f}%")
    for mdl, d in judge_tbl.items():
        print(f"  {mdl:16s} Judge overall = {d['Overall'][0]:.2f}% "
              f"[{d['Overall'][1]:.1f}, {d['Overall'][2]:.1f}]")


if __name__ == "__main__":
    main()
