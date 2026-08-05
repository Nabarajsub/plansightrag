"""Adversarial near-threshold stress-test set for compliance checking.

Distinct from scale500: violations land within 5-10% of the rule cutoff
(not 50%+), so the judge must read the dimensioned value precisely instead
of category-classifying gross mismatches.

Same 5 archetypes, same drawing API, same compliant/non-compliant balance,
same seed-locked agencies for reproducibility -- only the parameter ranges
change. 50 drawings = 5 archetypes x 10 (5 compliant + 5 non-compliant).
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, os, random, sys
sys.path.insert(0, f"{PSR_ROOT}/compliance")
import archetypes as A

OUT = f"{PSR_ROOT}/compliance/stress50/mockups"
random.seed(7777)

AGENCIES = ["WYDOT", "Caltrans", "AZDOT", "CDOT", "FDOT"]

def sample_choice(lst): return random.choice(lst)
def sample_int(lo, hi): return random.randint(lo, hi)


def gen_culvert(idx, compliant):
    """Cover >= 2.0\" (ACI 318); near-threshold compliant = 2.00-2.25\", non = 1.75-1.9375\"."""
    span = sample_choice([6.0, 8.0, 10.0, 12.0])
    rise = sample_choice([4.0, 5.0, 6.0])
    wall = sample_choice([10, 12, 14])
    top = sample_choice([12, 14, 16, 18])
    rebar_sp = sample_choice([8, 9, 10, 12])
    cover = sample_choice([2.0, 2.125, 2.25]) if compliant else sample_choice([1.75, 1.875, 1.9375])
    violation = None if compliant else f"Cover {cover}\" < 2.0\" min (ACI 318)"
    return dict(
        agency=sample_choice(AGENCIES), plan_id=f"B-{700+idx:03d}",
        span_ft=span, rise_ft=rise, wall_in=wall, top_slab_in=top,
        cover_in=cover, rebar_spacing_in=rebar_sp,
        concrete_class="Class A-A", rebar_grade="Grade 60",
    ), violation


def gen_guardrail(idx, compliant):
    """Footing >= 30\"; compliant = 30-32\", non = 27-29\"."""
    post_h = sample_choice([27, 28, 29])
    foot_dia = sample_choice([10, 12, 14])
    foot_depth = sample_choice([30, 31, 32]) if compliant else sample_choice([27, 28, 29])
    violation = None if compliant else f"Footing {foot_depth}\" < 30\" min (WYDOT 606.05)"
    return dict(
        agency=sample_choice(AGENCIES), plan_id=f"606-{700+idx:03d}",
        post_height_in=post_h, footing_depth_in=foot_depth, footing_dia_in=foot_dia,
        post_section=sample_choice(["W6x9", "W6x12", "W8x10"]),
        concrete_class="Class A", min_footing_note_in=30,
    ), violation


def gen_rebar(idx, compliant):
    """Stirrup <= d/2; compliant = d/2 - 0..1\", non = d/2 + 1..2\" (single-inch over)."""
    bw = sample_choice([12, 16, 18, 20])
    bh = sample_choice([24, 28, 30, 32])
    nt = sample_choice([2, 3, 4])
    nb = sample_choice([3, 4, 5])
    max_stirrup = bh // 2
    if compliant:
        stirrup = sample_choice([max_stirrup, max(4, max_stirrup - 1)])
    else:
        stirrup = sample_choice([max_stirrup + 1, max_stirrup + 2])
    violation = None if compliant else f"Stirrup {stirrup}\" > d/2={max_stirrup}\" max (ACI 318)"
    return dict(
        agency=sample_choice(AGENCIES), plan_id=f"A-{700+idx:03d}",
        beam_w_in=bw, beam_h_in=bh, bar_size=sample_choice(["#6","#7","#8","#9"]),
        n_top=nt, n_bot=nb, stirrup_spacing_in=stirrup,
        concrete_class=sample_choice(["Class S2","Class A-A","Class IV"]), cover_in=2.0,
    ), violation


def gen_inlet(idx, compliant):
    """Grate <= 4.0\"; compliant = 3.5-4.0\", non = 4.125-4.5\"."""
    tw = sample_choice([18, 24, 30, 36])
    th = sample_choice([4, 6, 8])
    grate = sample_choice([3.5, 3.75, 4.0]) if compliant else sample_choice([4.125, 4.25, 4.5])
    violation = None if compliant else f"Grate opening {grate}\" > 4\" max (PROWAG/ADA)"
    return dict(
        agency=sample_choice(AGENCIES), plan_id=f"232-{700+idx:03d}",
        throat_w_in=tw, throat_h_in=th, grate_open_in=grate,
        concrete_class=sample_choice(["Class IV", "Class A-A"]),
        frame_grade="ASTM A48 CL 35B",
        reinforcing_note="REINFORCING #4 @ 6\" OC EW.",
    ), violation


def gen_signpost(idx, compliant):
    """Anchor embed >= 12\"; compliant = 12-13\", non = 10.5-11.5\"."""
    pd = sample_choice([3.0, 4.0, 4.5])
    fd = sample_choice([18, 24, 30, 36, 42])
    fdep = sample_choice([36, 42, 48])
    anchor_d = sample_choice([0.625, 0.75, 0.875, 1.0])
    embed = sample_choice([12, 12.5, 13]) if compliant else sample_choice([10.5, 11, 11.5])
    violation = None if compliant else f"Anchor embed {embed}\" < 12\" min (AASHTO)"
    return dict(
        agency=sample_choice(AGENCIES), plan_id=f"C-{700+idx:03d}",
        post_dia_in=pd, embed_depth_in=18, footing_dia_in=fd,
        footing_depth_in=fdep, anchor_bolt_dia_in=anchor_d, anchor_embed_in=embed,
        concrete_class="Class P",
    ), violation


GENS = {
    "culvert":   ("draw_culvert",    gen_culvert),
    "guardrail": ("draw_guardrail",  gen_guardrail),
    "rebar":     ("draw_rebar",      gen_rebar),
    "inlet":     ("draw_inlet",      gen_inlet),
    "signpost":  ("draw_sign_post",  gen_signpost),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for arch_name, (fn, gen) in GENS.items():
        for i in range(10):
            compliant = (i % 2 == 0)
            kwargs, violation = gen(i, compliant)
            name = f"{arch_name}_adv_{'ok' if compliant else 'bad'}_{i:02d}"
            out_path = f"{OUT}/{name}.png"
            getattr(A, fn)(out_path, **kwargs)
            manifest.append({
                "name": name, "archetype": fn, "image_path": out_path,
                "agency": kwargs["agency"], "plan_id": kwargs["plan_id"],
                "design_facts": kwargs,
                "ground_truth_compliant": compliant,
                "violated_rule": violation,
            })
    random.shuffle(manifest)
    mpath = f"{OUT}/manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    n_c = sum(1 for m in manifest if m["ground_truth_compliant"])
    print(f"saved {len(manifest)} drawings ({n_c} compliant + {len(manifest)-n_c} non-compliant)")
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
