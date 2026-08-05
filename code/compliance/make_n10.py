"""Generate the n=10 validation set: 5 archetypes x (compliant, non-compliant).

Writes 10 PNGs + a manifest JSON describing ground truth and the rule violated
in each non-compliant variant.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---


import json
import os

import archetypes as A

OUT = f"{PSR_ROOT}/compliance/mockups_n10"
os.makedirs(OUT, exist_ok=True)

# (archetype_name, compliant_kwargs, non_compliant_kwargs, violated_rule)
SET = [
    # 1. CULVERT
    ("culvert_ok", "draw_culvert", dict(
        agency="WYDOT", plan_id="B-505",
        span_ft=8.0, rise_ft=5.0, wall_in=12, top_slab_in=14,
        cover_in=2.5, rebar_spacing_in=9.0,
        concrete_class="Class A-A", rebar_grade="Grade 60",
    ), None),
    ("culvert_bad", "draw_culvert", dict(
        agency="WYDOT", plan_id="B-505V",
        span_ft=8.0, rise_ft=5.0, wall_in=12, top_slab_in=14,
        cover_in=1.0,           # <-- VIOLATES 2" MIN COVER (ACI/AASHTO)
        rebar_spacing_in=9.0,
        concrete_class="Class A-A", rebar_grade="Grade 60",
    ), "Concrete cover 1.0\" is below the 2\" minimum required by ACI 318 / AASHTO LRFD."),

    # 2. GUARDRAIL POST
    ("guardrail_ok", "draw_guardrail", dict(
        agency="WYDOT", plan_id="606-7B",
        post_height_in=27, footing_depth_in=36, footing_dia_in=12,
        post_section="W6x9", concrete_class="Class A",
        min_footing_note_in=30,
    ), None),
    ("guardrail_bad", "draw_guardrail", dict(
        agency="WYDOT", plan_id="606-7C",
        post_height_in=27, footing_depth_in=24,  # <-- BELOW 30" WYDOT MIN
        footing_dia_in=12, post_section="W6x9",
        concrete_class="Class A", min_footing_note_in=30,
    ), "Footing depth 24\" is below the 30\" minimum required by WYDOT 606.05."),

    # 3. REBAR DETAIL
    ("rebar_ok", "draw_rebar", dict(
        agency="Caltrans", plan_id="A77B",
        beam_w_in=18, beam_h_in=30, bar_size="#8",
        n_top=3, n_bot=4, stirrup_spacing_in=8,
        concrete_class="Class S2", cover_in=2.0,
    ), None),
    ("rebar_bad", "draw_rebar", dict(
        agency="Caltrans", plan_id="A77C",
        beam_w_in=18, beam_h_in=30, bar_size="#8",
        n_top=3, n_bot=4,
        stirrup_spacing_in=18,    # <-- EXCEEDS ACI MAX d/2 ~ 15" FOR 30" BEAM
        concrete_class="Class S2", cover_in=2.0,
    ), "Stirrup spacing 18\" exceeds ACI 318 max of d/2 (~15\" for a 30\" beam)."),

    # 4. DRAINAGE INLET
    ("inlet_ok", "draw_inlet", dict(
        agency="FDOT", plan_id="232-001",
        throat_w_in=24, throat_h_in=6, grate_open_in=4.0,
        concrete_class="Class IV", frame_grade="ASTM A48 CL 35B",
        reinforcing_note="REINFORCING #4 @ 6\" OC EW.",
    ), None),
    ("inlet_bad", "draw_inlet", dict(
        agency="FDOT", plan_id="232-002",
        throat_w_in=24, throat_h_in=6,
        grate_open_in=9.0,        # <-- EXCEEDS 4" MAX FOR ADA / BICYCLE SAFETY
        concrete_class="Class IV", frame_grade="ASTM A48 CL 35B",
        reinforcing_note="REINFORCING #4 @ 6\" OC EW.",
    ), "Grate opening 9\" exceeds the 4\" maximum required by ADA and bicycle-safety guidance (PROWAG)."),

    # 5. SIGN POST FOUNDATION
    ("signpost_ok", "draw_sign_post", dict(
        agency="AZDOT", plan_id="C-25.10",
        post_dia_in=4.0, embed_depth_in=18, footing_dia_in=18,
        footing_depth_in=42, anchor_bolt_dia_in=0.75, anchor_embed_in=12,
        concrete_class="Class P",
    ), None),
    ("signpost_bad", "draw_sign_post", dict(
        agency="AZDOT", plan_id="C-25.11",
        post_dia_in=4.0, embed_depth_in=18, footing_dia_in=18,
        footing_depth_in=42, anchor_bolt_dia_in=0.75,
        anchor_embed_in=4,        # <-- BELOW 12" MIN EMBEDMENT
        concrete_class="Class P",
    ), "Anchor bolt embedment 4\" is below the 12\" minimum required by AASHTO / std spec."),
]


def main():
    manifest = []
    for name, func_name, kwargs, violation in SET:
        out = f"{OUT}/{name}.png"
        getattr(A, func_name)(out, **kwargs)
        compliant = violation is None
        manifest.append({
            "name": name,
            "archetype": func_name,
            "image_path": out,
            "agency": kwargs["agency"],
            "plan_id": kwargs["plan_id"],
            "design_facts": kwargs,
            "ground_truth_compliant": compliant,
            "violated_rule": violation,
        })
        print(f"  saved {out}  compliant={compliant}")
    mpath = f"{OUT}/manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest -> {mpath}  ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
