"""Causal faithfulness probe for the MaxSim grounding heatmaps (Cluster F, R2.6).

Reviewer 2 objects that we present MaxSim heatmaps as if they explained the
answer, when they only show where the *retriever* matched the query. That is
correct as written, and the safe fix is to rename them "retrieval attribution"
and drop every causal word.

This asks whether a weaker causal claim survives: if the highlighted region is
where the evidence actually lives, then hiding it should change the answer, and
hiding an equal amount of unrelated page should not.

Design -- each item is answered three times by the same VLM:

  base      the untouched page
  masked    the top-5% MaxSim patches painted out
  control   an equal-area, equal-shape block painted out in the LEAST salient
            region of the same page

The control must match the masked condition in occlusion *geometry*, not only in
area. An earlier version scattered the control patches at random across the page;
that hid the same number of cells but punched ~60 separate holes into the drawing,
while the top-5% patches are spatially clustered and blank one contiguous block.
Scattered occlusion is far more disruptive to a VLM reading a dense sheet, and the
probe duly reported that hiding the salient region perturbs the answer LESS than
the control -- an artefact of shape, not evidence about saliency. The control is
now the same bounding-box shape as the masked region, slid to the lowest-salience
position on the page.

The comparison that matters is masked-vs-control, not masked-vs-base. Painting
out any part of a dense engineering sheet degrades an answer somewhat; the
control absorbs that nuisance effect. What we report is the gap.

We report two metrics:

  change rate  did the answer text change at all. This needs no correctness
               judgement, so it is immune to scorer bias and is the PRIMARY
               metric.
  flip rate    did a correct answer stop being correct. Secondary, and scored by
               the same Qwen judge the benchmark itself uses -- an earlier run of
               this probe scored correctness with a string/numeric matcher, which
               kept only 39 of 100 items and biased the surviving set towards
               short numeric answers.

A large gap supports "the highlighted region carries the evidence." A gap near
zero means the heatmap is decorative, and we will say so and keep only the
retrieval-attribution reading. Either outcome is publishable; the point is to
stop asserting it without evidence.

    python faithfulness_probe.py --n 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "baselines_v2"))
from numeric_match import quantities, matches          # noqa: E402  (reuse the scorer)


# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

ROOT = CLUSTER_ROOT
OUT = f"{ROOT}/baselines_v2/reports/_faithfulness_probe.json"
RETRIEVER = "nomic-ai/colnomic-embed-multimodal-3b"
VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
TOP_PCT = 0.05        # fraction of patches to hide
SEED = 20260810
Image.MAX_IMAGE_PIXELS = None

ASK = ("Answer the question from this engineering drawing in one short phrase. "
       "If the drawing does not state it, reply exactly 'Not specified'.\n\n"
       "Question: {q}")


def norm(s):
    return " ".join(str(s or "").lower().replace('"', " ").replace("'", " ").split())


JUDGE = ("You are grading an answer to a question about an engineering drawing.\n\n"
         "Question: {q}\nReference answer: {ref}\nSubmitted answer: {ans}\n\n"
         "Does the submitted answer convey the same essential information as the "
         "reference (allowing for paraphrase, units, and rounding)? "
         "Reply with exactly one word: YES or NO.")


def changed(a, b):
    """Did the answer text materially change? Numeric answers compare by value."""
    qa, qb = quantities(a), quantities(b)
    if qa and qb:
        return not matches(qa, qb)
    return norm(a) != norm(b)



STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "the", "a", "an", "of", "for", "to", "in", "on", "at", "by", "with",
    "from", "as", "and", "or", "if", "it", "its", "this", "that", "these",
    "those", "there", "shown", "indicated", "used", "per", "into", "about",
}


def content_token_idx(tokens):
    keep = []
    for i, t in enumerate(tokens):
        if t == "<|endoftext|>":
            continue
        c = t.replace("\u0120", "").replace("Ġ", "").strip()
        if not c or not any(ch.isalnum() for ch in c) or c.lower() in STOPWORDS:
            continue
        keep.append(i)
    return keep or list(range(len(tokens)))


def paint(img, heat, n_patches, idx):
    """Blank the patch cells listed in `idx` (flat indices into the grid).

    get_n_patches / get_similarity_maps_from_embeddings work in (n_patches_X,
    n_patches_Y) order -- axis 0 spans the image WIDTH. The map is ravelled
    row-major, so a flat index f decomposes as xi = f // n_y, yi = f % n_y.
    The previous version read the axes as (rows, cols) and scaled each by the
    other dimension, so it blanked transposed cells -- i.e. it masked regions
    that were NOT the salient ones, which would make this probe meaningless.
    """
    n_x, n_y = n_patches
    W, H = img.size
    out = np.array(img).copy()
    cw, chh = W / n_x, H / n_y          # cell width spans x, cell height spans y
    for f in idx:
        xi, yi = divmod(int(f), n_y)
        x0, x1 = int(xi * cw), int(math.ceil((xi + 1) * cw))
        y0, y1 = int(yi * chh), int(math.ceil((yi + 1) * chh))
        out[y0:y1, x0:x1] = 255            # paper white, not black: less OOD
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    rng = random.Random(SEED)

    rows = [json.loads(l) for l in open(f"{ROOT}/lora_finetune/split_test.jsonl")]
    # answerable items only: a Hallucination-Rate item has nothing to hide
    rows = [r for r in rows
            if r.get("category") != "Hallucination Rate"
            and os.path.exists(r.get("image_path", ""))
            and norm(r.get("answer")) not in ("", "not specified")]
    rng.shuffle(rows)
    rows = rows[:a.n]
    print(f"[probe] {len(rows)} items | hide top {TOP_PCT:.0%} of patches", flush=True)

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from colpali_engine.interpretability import get_similarity_maps_from_embeddings
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)

    dev = "cuda"
    print("[probe] loading retriever", flush=True)
    ret = ColQwen2_5.from_pretrained(RETRIEVER, torch_dtype=torch.bfloat16,
                                     device_map=dev).eval()
    rp = ColQwen2_5_Processor.from_pretrained(RETRIEVER)

    print("[probe] loading VLM", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16).eval()
    vp = AutoProcessor.from_pretrained(VLM, trust_remote_code=True,
                                       min_pixels=256 * 28 * 28,
                                       max_pixels=1400 * 28 * 28)

    def ask(img, q):
        m = [{"role": "user", "content": [{"type": "image", "image": img},
                                          {"type": "text", "text": ASK.format(q=q)}]}]
        t = vp.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = vp(text=[t], images=[img], padding=True, return_tensors="pt").to(vlm.device)
        with torch.no_grad():
            o = vlm.generate(**inp, max_new_tokens=48, do_sample=False)
        return vp.batch_decode(o[:, inp["input_ids"].shape[1]:],
                               skip_special_tokens=True)[0].strip()

    def judge(img, q, ref, ans):
        m = [{"role": "user", "content": [{"type": "image", "image": img},
              {"type": "text", "text": JUDGE.format(q=q, ref=ref, ans=ans)}]}]
        t = vp.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = vp(text=[t], images=[img], padding=True, return_tensors="pt").to(vlm.device)
        with torch.no_grad():
            o = vlm.generate(**inp, max_new_tokens=8, do_sample=False)
        return "yes" in vp.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0].strip().lower()[:5]

    recs = []
    for k, r in enumerate(rows, 1):
        img = Image.open(r["image_path"]).convert("RGB")
        small = img.copy()
        small.thumbnail((1400, 1400))

        bimg = rp.process_images([small]).to(dev)
        bq = rp.process_queries([r["query"]]).to(dev)
        with torch.no_grad():
            ie, qe = ret(**bimg), ret(**bq)
        n_patches = (rp.get_n_patches(small.size,
                                      spatial_merge_size=ret.spatial_merge_size)
                     if hasattr(ret, "spatial_merge_size")
                     else rp.get_n_patches(small.size))
        maps = get_similarity_maps_from_embeddings(
            image_embeddings=ie, query_embeddings=qe,
            n_patches=n_patches, image_mask=rp.get_image_mask(bimg))[0]
        # mask the SAME map the manuscript figures display: mean over content
        # tokens (padding/stopwords excluded), so the probe tests the published
        # explanation rather than a different pooling of the same embeddings.
        _toks = rp.tokenizer.convert_ids_to_tokens(bq["input_ids"][0].tolist())
        heat = maps.float()[content_token_idx(_toks)].mean(dim=0).cpu().numpy().ravel()

        ph, pw = n_patches
        n_hide = max(1, int(round(TOP_PCT * heat.size)))
        top_idx = np.argsort(-heat)[:n_hide].tolist()

        # Geometry-matched control: translate the EXACT mask pattern to the
        # least-salient position on the page. Sliding a bounding box does not
        # work here -- the top-5% cells are scattered, so their bbox covers most
        # of the page and no disjoint window of that size exists. Translating the
        # pattern itself (on a torus, so no cell is lost at an edge) preserves
        # the number of hidden cells AND their spatial arrangement, which is the
        # nuisance variable: scattered occlusion disrupts a dense drawing more
        # than one contiguous blank, independently of what is hidden.
        H2 = heat.reshape(ph, pw)
        cells = [(i // pw, i % pw) for i in top_idx]
        masked_set = set(top_idx)
        best, best_idx = None, None
        for dr in range(ph):
            for dc in range(pw):
                if dr == 0 and dc == 0:
                    continue
                idx = [((r + dr) % ph) * pw + ((c + dc) % pw) for r, c in cells]
                if masked_set.intersection(idx):
                    continue
                v = float(np.mean([heat[i] for i in idx]))
                if best is None or v < best:
                    best, best_idx = v, idx
        if best_idx is None:      # every translation overlaps: fall back
            cold = np.argsort(heat)[: max(n_hide, heat.size // 2)].tolist()
            best_idx = rng.sample(cold, n_hide)
        ctl_idx = best_idx

        def dispersion(idx):
            """Mean nearest-neighbour Chebyshev distance between hidden cells.
            ~1 means a solid contiguous block; larger means scattered holes."""
            pts = [(i // pw, i % pw) for i in idx]
            if len(pts) < 2:
                return 0.0
            tot = 0.0
            for a_i, (ra, ca) in enumerate(pts):
                tot += min(max(abs(ra - rb), abs(ca - cb))
                           for b_i, (rb, cb) in enumerate(pts) if b_i != a_i)
            return tot / len(pts)

        geom = {"grid": [ph, pw],
                "masked_dispersion": round(dispersion(top_idx), 3),
                "control_dispersion": round(dispersion(ctl_idx), 3),
                "hidden_frac": round(n_hide / heat.size, 3)}

        base = ask(small, r["query"])
        mask = ask(paint(small, heat, n_patches, top_idx), r["query"])
        ctrl = ask(paint(small, heat, n_patches, ctl_idx), r["query"])
        j_base = judge(small, r["query"], r["answer"], base)
        j_mask = judge(small, r["query"], r["answer"], mask)
        j_ctrl = judge(small, r["query"], r["answer"], ctrl)

        rec = {"query": r["query"], "reference": r["answer"],
               "category": r.get("category"), "agency": r.get("agency"),
               "image": os.path.basename(r["image_path"]),
               "n_patches": list(n_patches), "n_hidden": n_hide,
               "n_control_hidden": len(ctl_idx),
               "control_shape_matched": best is not None, "geometry": geom,
               "base": base, "masked": mask, "control": ctrl,
               "base_ok": j_base, "masked_ok": j_mask, "control_ok": j_ctrl,
               "masked_changed": changed(base, mask),
               "control_changed": changed(base, ctrl)}
        recs.append(rec)
        if k % 10 == 0 or k == len(rows):
            nb = sum(x["base_ok"] for x in recs)
            print(f"  [{k}/{len(rows)}] base_ok={nb}", flush=True)
        if k % 25 == 0:
            torch.cuda.empty_cache()

    # Only items the model got right unmasked can inform a flip rate.
    live = [x for x in recs if x["base_ok"]]
    n = len(live)
    flip_m = sum(1 for x in live if not x["masked_ok"])
    flip_c = sum(1 for x in live if not x["control_ok"])

    def wilson(k, nn, z=1.96):
        if nn == 0:
            return (0.0, 0.0)
        p = k / nn
        d = 1 + z * z / nn
        c = (p + z * z / (2 * nn)) / d
        h = z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / d
        return (max(0.0, (c - h) * 100), min(100.0, (c + h) * 100))

    def mcnemar(pairs):
        """pairs: list of (masked_bad, control_bad) booleans."""
        b01 = sum(1 for m, c in pairs if m and not c)
        b10 = sum(1 for m, c in pairs if c and not m)
        if b01 + b10 == 0:
            return b01, b10, 1.0
        k = min(b01, b10)
        return b01, b10, min(1.0, 2 * sum(math.comb(b01 + b10, i)
                                          for i in range(k + 1)) / 2 ** (b01 + b10))

    # PRIMARY: change rate over ALL items -- no correctness judgement involved
    N = len(recs)
    chg_m = sum(1 for x in recs if x["masked_changed"])
    chg_c = sum(1 for x in recs if x["control_changed"])
    cb01, cb10, cp = mcnemar([(x["masked_changed"], x["control_changed"]) for x in recs])

    # SECONDARY: flip of correctness, on items the model got right unmasked
    b01, b10, p = mcnemar([(not x["masked_ok"], not x["control_ok"]) for x in live])

    out = {"note": "Causal faithfulness probe for MaxSim heatmaps (R2.6).",
           "retriever": RETRIEVER, "vlm": VLM, "top_pct_hidden": TOP_PCT,
           "seed": SEED, "n_items": N, "n_base_correct": n,
           "primary_change_rate": {
               "masked_pct": round(chg_m / N * 100, 2) if N else None,
               "control_pct": round(chg_c / N * 100, 2) if N else None,
               "gap_pp": round((chg_m - chg_c) / N * 100, 2) if N else None,
               "ci_masked": [round(x, 2) for x in wilson(chg_m, N)],
               "ci_control": [round(x, 2) for x in wilson(chg_c, N)],
               "mcnemar": {"masked_only": cb01, "control_only": cb10, "exact_p": cp}},
           "flip_rate_masked_pct": round(flip_m / n * 100, 2) if n else None,
           "flip_rate_control_pct": round(flip_c / n * 100, 2) if n else None,
           "flip_ci_masked": [round(x, 2) for x in wilson(flip_m, n)],
           "flip_ci_control": [round(x, 2) for x in wilson(flip_c, n)],
           "gap_pp": round((flip_m - flip_c) / n * 100, 2) if n else None,
           "mcnemar": {"masked_only": b01, "control_only": b10, "exact_p": p},
           "records": recs}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n{'=' * 70}\nFAITHFULNESS PROBE\n{'=' * 70}")
    print(f"  items                       {N}")
    print(f"\n  PRIMARY -- answer changed at all (no correctness judgement)")
    print(f"    top-5% hidden             {chg_m/N*100:6.2f}%  "
          f"[{wilson(chg_m,N)[0]:.1f}, {wilson(chg_c,N)[1]:.1f}]" if N else "")
    print(f"    control (equal area)      {chg_c/N*100:6.2f}%  "
          f"[{wilson(chg_c,N)[0]:.1f}, {wilson(chg_c,N)[1]:.1f}]" if N else "")
    print(f"    gap                       {(chg_m-chg_c)/N*100:+6.2f} pp   "
          f"McNemar exact p = {cp:.4g}   ({cb01} vs {cb10} discordant)" if N else "")
    print(f"\n  SECONDARY -- correct answer became incorrect")
    print(f"    answered correctly unmasked {n}")
    if n:
        print(f"  flip rate, top-5% hidden  {flip_m/n*100:6.2f}%  "
              f"[{wilson(flip_m,n)[0]:.1f}, {wilson(flip_m,n)[1]:.1f}]")
        print(f"  flip rate, control        {flip_c/n*100:6.2f}%  "
              f"[{wilson(flip_c,n)[0]:.1f}, {wilson(flip_c,n)[1]:.1f}]")
        print(f"  gap                       {(flip_m-flip_c)/n*100:+6.2f} pp   "
              f"McNemar exact p = {p:.4g}")
        print(f"\n  {'supports' if p < 0.05 and flip_m > flip_c else 'does NOT support'}"
              f" a causal reading of the highlighted region")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
