"""Legacy 78-Q WYDOT comparison -- local-model VQA + judge (gemini-free).

Sequential, single-job pipeline:
  1. Generate retrieval-augmented answers on the 78-Q WYDOT subset with the two
     OPEN local generators -- Qwen2.5-VL-7B and InternVL2_5-8B -- each given the
     top-3 ColPali-retrieved pages (from eval_78q_legacy.py's retrieval_78q.json).
  2. Judge every answer with Qwen2.5-VL-72B (4-bit, local) against the reference.
  3. Per-category Judge Accuracy + 95% bootstrap CIs (10k, seed 42).
  4. Emit the refreshed legacy table fragment, splicing in the recomputed ColPali
     Recall@5 (Hit) column from legacy78q_retrieval.json and keeping the existing
     Gemini-3-Pro-Preview judge numbers as a STATIC published row (no API calls).

No Gemini is ever called. Models are loaded, used, then freed in sequence so the
72B judge fits after the generators are released.
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

ROOT = f"{PSR_ROOT}"
Q_FILES = [f"{ROOT}/iclr_2027_research/data_cache/wydot78_train.jsonl",
           f"{ROOT}/iclr_2027_research/data_cache/wydot78_dev.jsonl"]
RETR = f"{ROOT}/baselines_v2/retrieval_78q.json"
RETR_REPORT = f"{ROOT}/baselines_v2/reports/legacy78q_retrieval.json"
OUT_JSON = f"{ROOT}/baselines_v2/reports/legacy78q_judge.json"
OUT_TEX = f"{ROOT}/baselines_v2/reports/_tbl_legacy_78q.tex"

QWEN7B = "Qwen/Qwen2.5-VL-7B-Instruct"
INTERNVL = "OpenGVLab/InternVL2_5-8B"
JUDGE_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

N_BOOT, SEED = 10000, 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CAT_ORDER = [
    ("Dimensional Accuracy", "Dimensional"),
    ("Visual Interpretation", "Visual"),
    ("Logical Reasoning", "Logical"),
    ("Hallucination Rate", "Hallucination"),
]
# Existing published Gemini-3-Pro-Preview judge numbers (kept static; no new call).
GEMINI_STATIC = {  # category-label -> (point, lo, hi)
    "Dimensional":   (82.41, 68.3, 95.9),
    "Visual":        (71.25, 48.8, 90.0),
    "Logical":       (100.00, 100.0, 100.0),
    "Hallucination": (93.75, 81.3, 100.0),
    "Overall":       (86.28, 78.3, 93.3),
}
INSTR = ("You are an expert civil engineer reading WYDOT standard plans. "
         "Using ONLY the provided plan images, answer the question concisely. "
         "If a numeric value is asked, give the value with units.\n\nQuestion: ")


def load_questions():
    rows = []
    for f in Q_FILES:
        rows += [json.loads(l) for l in open(f)]
    return rows


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
# Qwen2.5-VL generation
# ----------------------------------------------------------------------------
def gen_qwen(rows, retr):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN7B, torch_dtype=torch.bfloat16, device_map="auto").eval()
    p = AutoProcessor.from_pretrained(QWEN7B, trust_remote_code=True)
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
        text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = p(text=[text], images=imgs, padding=True, return_tensors="pt")
        inp = {k: v.to(m.device) for k, v in inp.items()}
        with torch.no_grad():
            o = m.generate(**inp, max_new_tokens=256, do_sample=False)
        out[r["query"]] = p.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0].strip()
        if (i + 1) % 20 == 0:
            print(f"  [qwen7b] {i+1}/{len(rows)}")
    free(m, p)
    return out


# ----------------------------------------------------------------------------
# InternVL2_5-8B generation
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


def _ivl_preprocess(image, size=448, max_num=12):
    w, h = image.size
    ar = w / h
    ratios = sorted({(i, j) for i in range(1, max_num + 1) for j in range(1, max_num + 1)
                     if i * j <= max_num}, key=lambda x: x[0] * x[1])
    tr = _closest_ratio(ar, ratios)
    tw, th = size * tr[0], size * tr[1]
    img = image.resize((tw, th))
    tiles = [img.crop(((i % tr[0]) * size, (i // tr[0]) * size,
                       (i % tr[0] + 1) * size, (i // tr[0] + 1) * size))
             for i in range(tr[0] * tr[1])]
    if len(tiles) != 1:
        tiles.append(image.resize((size, size)))
    return tiles


def _ivl_pixels(path, device, max_num=12):
    tfm = _ivl_transform(448)
    tiles = _ivl_preprocess(Image.open(path).convert("RGB"), 448, max_num)
    return torch.stack([tfm(t) for t in tiles]).to(torch.bfloat16).to(device)


def gen_internvl(rows, retr):
    from transformers import AutoModel, AutoTokenizer
    from transformers.generation.utils import GenerationMixin
    m = AutoModel.from_pretrained(INTERNVL, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                  low_cpu_mem_usage=True, device_map="auto").eval()
    tok = AutoTokenizer.from_pretrained(INTERNVL, trust_remote_code=True, use_fast=False)
    # transformers v4.50+ no longer auto-inherits GenerationMixin for custom remote
    # LMs, so InternVL's language_model loses .generate() (which model.chat needs).
    lm_cls = type(m.language_model)
    if not issubclass(lm_cls, GenerationMixin):
        lm_cls.__bases__ = (GenerationMixin,) + lm_cls.__bases__
        print("[fix] injected GenerationMixin into", lm_cls.__name__)
    out = {}
    for i, r in enumerate(rows):
        paths = retr.get(r["query"], [r["image_path"]])[:3] or [r["image_path"]]
        pv_list, nptch = [], []
        for pth in paths:
            try:
                pv = _ivl_pixels(pth, m.device, max_num=8)
                pv_list.append(pv)
                nptch.append(pv.shape[0])
            except Exception:
                pass
        if not pv_list:
            out[r["query"]] = "[ERROR: no image]"
            continue
        pixel_values = torch.cat(pv_list, dim=0)
        prefix = "".join(f"Image-{k+1}: <image>\n" for k in range(len(pv_list)))
        question = prefix + INSTR + r["query"]
        try:
            resp = m.chat(tok, pixel_values, question,
                          dict(max_new_tokens=256, do_sample=False),
                          num_patches_list=nptch)
        except Exception as exc:
            resp = f"[ERROR: {exc}]"
        out[r["query"]] = resp.strip() if isinstance(resp, str) else str(resp)
        if (i + 1) % 20 == 0:
            print(f"  [internvl] {i+1}/{len(rows)}")
    free(m, tok)
    return out


# ----------------------------------------------------------------------------
# Qwen2.5-VL-72B judge
# ----------------------------------------------------------------------------
def judge_all(rows, ans_by_model):
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

    verdicts = {mdl: [] for mdl in ans_by_model}  # list of (correct, category)
    for i, r in enumerate(rows):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
            w, h = img.size
            if max(w, h) > 1100:
                s = 1100 / max(w, h)
                img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        except Exception:
            continue
        for mdl, ans in ans_by_model.items():
            verdicts[mdl].append((judge_one(img, r["query"], r.get("answer", ""),
                                            ans.get(r["query"], "")), r["category"]))
        if (i + 1) % 20 == 0:
            print(f"  [judge] {i+1}/{len(rows)}")
    free(m, p)
    return verdicts


def main():
    rows = load_questions()
    retr = json.load(open(RETR)) if os.path.exists(RETR) else {}
    retr_report = json.load(open(RETR_REPORT)) if os.path.exists(RETR_REPORT) else {}
    print(f"[78q] {len(rows)} questions; retrieval cached for {len(retr)} queries")

    print("[gen] Qwen2.5-VL-7B")
    qwen = gen_qwen(rows, retr)
    print("[gen] InternVL2_5-8B")
    ivl = gen_internvl(rows, retr)

    ans_by_model = {"Qwen2.5-VL-7B": qwen, "InternVL-2.5-8B": ivl}
    print("[judge] Qwen2.5-VL-72B")
    verdicts = judge_all(rows, ans_by_model)

    # ---- per-category Judge Accuracy + CIs ----------------------------------
    judge_tbl = {}
    for mdl, vs in verdicts.items():
        judge_tbl[mdl] = {}
        by_cat = {c: [] for c, _ in CAT_ORDER}
        for correct, cat in vs:
            if cat in by_cat:
                by_cat[cat].append(int(correct))
        for cat, label in CAT_ORDER:
            judge_tbl[mdl][label] = bootstrap_acc(by_cat[cat])
        judge_tbl[mdl]["Overall"] = bootstrap_acc([int(c) for c, _ in vs])

    out = {"n": len(rows), "n_boot": N_BOOT, "seed": SEED,
           "answers": {k: v for k, v in ans_by_model.items()},
           "judge": {m: {k: list(v) for k, v in d.items()} for m, d in judge_tbl.items()}}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)

    # ---- LaTeX table --------------------------------------------------------
    # Hit (Recall@5) column comes from the recomputed retrieval report; it is
    # generator-independent, so the same value applies to every model row.
    def hit_cell(label):
        pc = retr_report.get("per_category", {}).get(label) if label != "Overall" \
            else retr_report.get("overall")
        if not pc:
            return "---"
        return f"{pc['point']:.2f} [{pc['lo']:.1f}, {pc['hi']:.1f}]"

    def jc(t):
        return f"{t[0]:.2f} [{t[1]:.1f}, {t[2]:.1f}]"

    L = []
    L.append("% Auto-generated by baselines_v2/eval_78q_judge.py (+ eval_78q_legacy.py)")
    L.append("\\begin{table*}[!t]")
    L.append("\\caption{Recomputed per-category Recall@5 (Hit) and Judge Accuracy (Judge) on the "
             "legacy 78-question WYDOT subset, with 95\\% bootstrap CIs (10{,}000 resamples, seed~42). "
             "Recall@5 is recomputed with ColPali over the 237-page WYDOT-only index and is "
             "generator-independent. Judge Accuracy for the open generators is recomputed with a "
             "local Qwen2.5-VL-72B judge (no proprietary API); the Gemini-3-Pro-Preview row reproduces "
             "the previously published judge numbers for reference. Per-category $N$: Dim$=$29, "
             "Vis$=$16, Log$=$17, Hal$=$16.}\\label{tbl:bench-stats-primary}")
    L.append("\\centering")
    L.append("\\small")
    L.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llcc@{}}")
    L.append("\\toprule")
    L.append("Model & Category & Hit (\\%) & Judge (\\%) \\\\")
    L.append("\\midrule")

    def block(model_label, judge_src, is_static=False):
        rows_tex = [f"\\multirow{{5}}{{*}}{{{model_label}}}"]
        cats = [("Dimensional", "Dimensional"), ("Visual", "Visual"),
                ("Logical", "Logical"), ("Hallucination", "Hallucination")]
        for j, (label, disp) in enumerate(cats):
            jcell = jc(judge_src[label]) if not is_static else \
                f"{judge_src[label][0]:.2f} [{judge_src[label][1]:.1f}, {judge_src[label][2]:.1f}]"
            lead = rows_tex.pop(0) if j == 0 else ""
            rows_tex.append(f"  {lead} & {disp:<12s} & {hit_cell(label)} & {jcell} \\\\")
        ov = judge_src["Overall"]
        ovc = f"{ov[0]:.2f} [{ov[1]:.1f}, {ov[2]:.1f}]"
        rows_tex.append(f"  & \\textbf{{Overall}} & \\textbf{{{hit_cell('Overall')}}} & \\textbf{{{ovc}}} \\\\")
        return "\n".join(rows_tex)

    L.append(block("Gemini-3-Pro-Preview", GEMINI_STATIC, is_static=True))
    L.append("\\midrule")
    L.append(block("Qwen2.5-VL-7B", judge_tbl["Qwen2.5-VL-7B"]))
    L.append("\\midrule")
    L.append(block("InternVL-2.5-8B", judge_tbl["InternVL-2.5-8B"]))
    L.append("\\bottomrule")
    L.append("\\end{tabular*}")
    L.append("\\end{table*}")
    open(OUT_TEX, "w").write("\n".join(L) + "\n")

    print(f"\n[done] -> {OUT_JSON}")
    print(f"[done] -> {OUT_TEX}")
    for mdl, d in judge_tbl.items():
        print(f"  {mdl:16s} Overall Judge = {d['Overall'][0]:.2f}% "
              f"[{d['Overall'][1]:.1f}, {d['Overall'][2]:.1f}]")


if __name__ == "__main__":
    main()
