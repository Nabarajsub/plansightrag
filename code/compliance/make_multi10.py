"""Generate 10 multi-plan compliance pilot drawings.

Distribution:
  N=2 plans:  4 drawings (2 compliant + 2 non-compliant)  -- beam section
  N=3 plans:  2 drawings (1 compliant + 1 non-compliant)  -- bridge bearing
  N=4 plans:  2 drawings (1 compliant + 1 non-compliant)  -- approach slab
  N=5 plans:  2 drawings (1 compliant + 1 non-compliant)  -- bridge pier

For each non-compliant drawing, EXACTLY ONE component violates its applicable
standard; the other N-1 components pass. Ground truth records which component
and which rule.
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

import archetypes_multi as A
from rules_registry import COMPONENT_RULES

OUT = f"{PSR_ROOT}/compliance/multi_plan/mockups"


def make_comp(key, value, unit_display=None):
    rule = COMPONENT_RULES[key]
    return {
        "key": key,
        "label": rule["rule_name"].split(" (")[0],
        "value": value,
        "value_display": f'{value}{unit_display or " "+rule["unit"]}',
        "threshold": rule["threshold"],
        "operator": rule["operator"],
    }


SET = [
    # N=2 (4 drawings)
    ("n2_beam_ok_a", "draw_n2_beam", 2, dict(agency="Caltrans", plan_id="DSGN-N2-01"),
     [make_comp("concrete_cover", 2.5, "\""), make_comp("rebar_grade", 60, " ksi")], None),
    ("n2_beam_ok_b", "draw_n2_beam", 2, dict(agency="WYDOT", plan_id="DSGN-N2-02"),
     [make_comp("concrete_cover", 2.0, "\""), make_comp("rebar_grade", 75, " ksi")], None),
    ("n2_beam_bad_a", "draw_n2_beam", 2, dict(agency="Caltrans", plan_id="DSGN-N2-03"),
     [make_comp("concrete_cover", 1.0, "\""), make_comp("rebar_grade", 60, " ksi")],
     {"failed_key": "concrete_cover", "msg": "Cover 1.0\" < 2.0\" min (ACI 318/AASHTO)"}),
    ("n2_beam_bad_b", "draw_n2_beam", 2, dict(agency="WYDOT", plan_id="DSGN-N2-04"),
     [make_comp("concrete_cover", 2.5, "\""), make_comp("rebar_grade", 40, " ksi")],
     {"failed_key": "rebar_grade", "msg": "Rebar Grade 40 < Grade 60 min (ASTM A615)"}),

    # N=3 (2 drawings)
    ("n3_bearing_ok", "draw_n3_bearing", 3, dict(agency="FDOT", plan_id="DSGN-N3-01"),
     [make_comp("bearing_pad_thickness", 1.25, "\""), make_comp("concrete_class", 4500, " psi"),
      make_comp("anchor_bolt_embed", 14, "\"")], None),
    ("n3_bearing_bad", "draw_n3_bearing", 3, dict(agency="FDOT", plan_id="DSGN-N3-02"),
     [make_comp("bearing_pad_thickness", 1.5, "\""), make_comp("concrete_class", 4500, " psi"),
      make_comp("anchor_bolt_embed", 6, "\"")],
     {"failed_key": "anchor_bolt_embed", "msg": "Anchor embed 6\" < 12\" min (AASHTO)"}),

    # N=4 (2 drawings)
    ("n4_approach_ok", "draw_n4_approach", 4, dict(agency="Caltrans", plan_id="DSGN-N4-01"),
     [make_comp("pavement_thickness", 8.0, "\""), make_comp("concrete_cover", 2.5, "\""),
      make_comp("joint_seal_width", 0.75, "\""), make_comp("rebar_grade", 60, " ksi")], None),
    ("n4_approach_bad", "draw_n4_approach", 4, dict(agency="Caltrans", plan_id="DSGN-N4-02"),
     [make_comp("pavement_thickness", 4.0, "\""), make_comp("concrete_cover", 2.5, "\""),
      make_comp("joint_seal_width", 0.75, "\""), make_comp("rebar_grade", 60, " ksi")],
     {"failed_key": "pavement_thickness", "msg": "Pavement 4\" < 6\" min (AASHTO)"}),

    # N=5 (2 drawings)
    ("n5_pier_ok", "draw_n5_pier", 5, dict(agency="WYDOT", plan_id="DSGN-N5-01"),
     [make_comp("footing_depth", 48, "\""), make_comp("concrete_cover", 2.5, "\""),
      make_comp("stirrup_spacing", 10, "\""), make_comp("bearing_pad_thickness", 1.25, "\""),
      make_comp("anchor_bolt_embed", 14, "\"")], None),
    ("n5_pier_bad", "draw_n5_pier", 5, dict(agency="WYDOT", plan_id="DSGN-N5-02"),
     [make_comp("footing_depth", 48, "\""), make_comp("concrete_cover", 2.5, "\""),
      make_comp("stirrup_spacing", 28, "\""), make_comp("bearing_pad_thickness", 1.25, "\""),
      make_comp("anchor_bolt_embed", 14, "\"")],
     {"failed_key": "stirrup_spacing", "msg": "Stirrup 28\" > d/2 max (ACI 318) for column (typ. < 12\")"}),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for name, fn, n_plans, kwargs, components, violation in SET:
        out = f"{OUT}/{name}.png"
        getattr(A, fn)(out, components=components, **kwargs)
        compliant = violation is None
        manifest.append({
            "name": name, "archetype": fn, "n_plans": n_plans,
            "image_path": out, "agency": kwargs["agency"], "plan_id": kwargs["plan_id"],
            "components": [{"key": c["key"], "value": c["value"], "value_display": c["value_display"],
                            "threshold": c["threshold"], "operator": c["operator"]}
                           for c in components],
            "ground_truth_compliant": compliant,
            "violated_rule": violation["msg"] if violation else None,
            "violated_key":  violation["failed_key"] if violation else None,
        })
        print(f"  saved {out}  N={n_plans}  compliant={compliant}")
    mpath = f"{OUT}/manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nmanifest -> {mpath}  ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
