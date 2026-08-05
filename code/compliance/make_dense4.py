"""Generate 4 DENSE drawings (2 compliant, 2 non-compliant) for density-robustness test.

Each drawing has multiple views + schedule table + many notes. The violation
in non-compliant cases lives in ONE schedule row (Type B / Type C cell), so the
judge must locate it within a busy multi-view sheet.
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

import archetypes_dense as AD

OUT = f"{PSR_ROOT}/compliance/mockups_dense4"
os.makedirs(OUT, exist_ok=True)

SET = [
    # 1. Culvert dense, compliant (all rows have cover >= 2")
    ("culvert_dense_ok", "draw_culvert_dense", dict(
        agency="WYDOT", plan_id="B-505 (DENSE)",
        schedule=[
            {"span_ft": 6.0, "rise_ft": 4.0, "wall_in": 10, "top_in": 12, "cover_in": 2.0, "stirrup_in": 8},
            {"span_ft": 8.0, "rise_ft": 5.0, "wall_in": 12, "top_in": 14, "cover_in": 2.5, "stirrup_in": 8},
            {"span_ft":10.0, "rise_ft": 6.0, "wall_in": 14, "top_in": 16, "cover_in": 2.5, "stirrup_in": 9},
            {"span_ft":12.0, "rise_ft": 7.0, "wall_in": 16, "top_in": 18, "cover_in": 3.0, "stirrup_in": 9},
        ],
        hero_idx=1, cover_in=2.0, rebar_grade="Grade 60", violated=False,
    ), None),
    # 2. Culvert dense, non-compliant (Type B has cover = 1.0" -> violates 2" min)
    ("culvert_dense_bad", "draw_culvert_dense", dict(
        agency="WYDOT", plan_id="B-505V (DENSE)",
        schedule=[
            {"span_ft": 6.0, "rise_ft": 4.0, "wall_in": 10, "top_in": 12, "cover_in": 2.0, "stirrup_in": 8},
            {"span_ft": 8.0, "rise_ft": 5.0, "wall_in": 12, "top_in": 14, "cover_in": 1.0, "stirrup_in": 8},   # <-- VIOLATES
            {"span_ft":10.0, "rise_ft": 6.0, "wall_in": 14, "top_in": 16, "cover_in": 2.5, "stirrup_in": 9},
            {"span_ft":12.0, "rise_ft": 7.0, "wall_in": 16, "top_in": 18, "cover_in": 3.0, "stirrup_in": 9},
        ],
        hero_idx=1, cover_in=2.0, rebar_grade="Grade 60", violated=True,
    ), "Type B row has concrete cover = 1.0\" which violates ACI 2\" min."),
    # 3. Beam rebar dense, compliant (all stirrup spacings <= height/2)
    ("rebar_dense_ok", "draw_rebar_dense", dict(
        agency="Caltrans", plan_id="A77B (DENSE)",
        schedule=[
            {"width_in": 12, "height_in": 24, "n_top": 2, "n_bot": 3, "bar_size": 6, "stirrup_in":10, "cover_in": 2.0},
            {"width_in": 16, "height_in": 28, "n_top": 3, "n_bot": 4, "bar_size": 7, "stirrup_in":12, "cover_in": 2.0},
            {"width_in": 18, "height_in": 30, "n_top": 3, "n_bot": 4, "bar_size": 8, "stirrup_in":14, "cover_in": 2.0},
            {"width_in": 20, "height_in": 36, "n_top": 4, "n_bot": 5, "bar_size": 8, "stirrup_in":16, "cover_in": 2.0},
        ],
        hero_idx=2, violated=False,
    ), None),
    # 4. Beam rebar dense, non-compliant (Type C: 30" beam with 18" stirrup spacing -> 18 > 15)
    ("rebar_dense_bad", "draw_rebar_dense", dict(
        agency="Caltrans", plan_id="A77C (DENSE)",
        schedule=[
            {"width_in": 12, "height_in": 24, "n_top": 2, "n_bot": 3, "bar_size": 6, "stirrup_in":10, "cover_in": 2.0},
            {"width_in": 16, "height_in": 28, "n_top": 3, "n_bot": 4, "bar_size": 7, "stirrup_in":12, "cover_in": 2.0},
            {"width_in": 18, "height_in": 30, "n_top": 3, "n_bot": 4, "bar_size": 8, "stirrup_in":18, "cover_in": 2.0},  # <-- VIOLATES (18 > 30/2=15)
            {"width_in": 20, "height_in": 36, "n_top": 4, "n_bot": 5, "bar_size": 8, "stirrup_in":16, "cover_in": 2.0},
        ],
        hero_idx=2, violated=True,
    ), "Type C row has stirrup spacing 18\" but beam height 30\" requires max d/2 = 15\"."),
]


def main():
    manifest = []
    for name, fn, kwargs, violation in SET:
        out = f"{OUT}/{name}.png"
        getattr(AD, fn)(out, **kwargs)
        compliant = violation is None
        # For derive_checks() compatibility, expose the HERO row's parameters in design_facts
        hero = kwargs["schedule"][kwargs["hero_idx"]]
        facts_for_compat = {**hero, **{
            "schedule": kwargs["schedule"],
            "hero_idx": kwargs["hero_idx"],
            "agency": kwargs["agency"],
        }}
        if fn == "draw_culvert_dense":
            arch_compat = "draw_culvert"
            facts_for_compat["cover_in"] = hero["cover_in"]
            facts_for_compat["rebar_spacing_in"] = hero["stirrup_in"]
        elif fn == "draw_rebar_dense":
            arch_compat = "draw_rebar"
            facts_for_compat["beam_h_in"] = hero["height_in"]
            facts_for_compat["beam_w_in"] = hero["width_in"]
            facts_for_compat["stirrup_spacing_in"] = hero["stirrup_in"]
        else:
            arch_compat = fn
        manifest.append({
            "name": name,
            "archetype": arch_compat,         # map dense -> base archetype for derive_checks
            "archetype_dense": fn,
            "image_path": out,
            "agency": kwargs["agency"],
            "plan_id": kwargs["plan_id"],
            "design_facts": facts_for_compat,
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
