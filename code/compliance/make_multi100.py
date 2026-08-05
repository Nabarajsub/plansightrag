"""Generate 100 multi-plan compliance drawings: 25 each at N=2/3/4/5.

50 compliant + 50 non-compliant. For each non-compliant, ONE component violates
its applicable standard. The stirrup_spacing rule now stores the pre-resolved
numeric threshold (d/2 resolved per-drawing) to avoid the n5_pier_ok miss.

NO plan IDs cited on the drawing (pure visual inference).
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
sys.path.insert(0, f"{PSR_ROOT}/compliance/multi_plan")
import archetypes_multi as A
from rules_registry import COMPONENT_RULES

OUT = f"{PSR_ROOT}/compliance/multi_plan100/mockups"
random.seed(2026)

AGENCIES = ["WYDOT", "Caltrans", "AZDOT", "CDOT", "FDOT"]


def make_comp(key, value, threshold_override=None, unit_display=None):
    rule = COMPONENT_RULES[key]
    thresh = threshold_override if threshold_override is not None else rule["threshold"]
    return {
        "key": key, "label": rule["rule_name"].split(" (")[0],
        "value": value, "value_display": f'{value}{unit_display or " "+rule["unit"]}',
        "threshold": thresh, "operator": rule["operator"], "unit": rule["unit"],
    }


# ---- N=2: beam (cover + rebar grade) ----
def gen_n2(idx, compliant):
    cover = random.choice([2.0, 2.5, 3.0]) if compliant else random.choice([1.0, 1.5])
    grade = random.choice([60, 75, 80]) if compliant else random.choice([40, 50])
    violation = None
    if not compliant:
        if cover < 2.0:
            violation = {"failed_key": "concrete_cover", "msg": f"Cover {cover}\" < 2.0\" min (ACI 318)"}
        else:
            violation = {"failed_key": "rebar_grade", "msg": f"Rebar Grade {grade} < 60 min"}
            cover = random.choice([2.0, 2.5])
            grade = random.choice([40, 50])
    comps = [make_comp("concrete_cover", cover, unit_display="\""),
             make_comp("rebar_grade", grade, unit_display=" ksi")]
    return comps, violation, "draw_n2_beam", 2

# ---- N=3: bearing (pad thickness + concrete + anchor embed) ----
def gen_n3(idx, compliant):
    pad = random.choice([1.0, 1.25, 1.5, 2.0]) if compliant else random.choice([0.5, 0.75])
    conc = random.choice([4000, 4500, 5000]) if compliant else random.choice([2500, 3000])
    embed = random.choice([12, 14, 16, 18]) if compliant else random.choice([4, 6, 8])
    violation = None
    if not compliant:
        failure = random.choice(["pad", "conc", "embed"])
        if failure == "pad":
            violation = {"failed_key": "bearing_pad_thickness", "msg": f"Pad {pad}\" < 1.0\" min"}
            conc = random.choice([4000,4500,5000]); embed = random.choice([12,14,16,18])
        elif failure == "conc":
            violation = {"failed_key": "concrete_class", "msg": f"Concrete {conc} psi < 4000 psi min"}
            pad = random.choice([1.0,1.25,1.5]); embed = random.choice([12,14,16,18])
        else:
            violation = {"failed_key": "anchor_bolt_embed", "msg": f"Anchor {embed}\" < 12\" min (AASHTO)"}
            pad = random.choice([1.0,1.25,1.5]); conc = random.choice([4000,4500,5000])
    comps = [make_comp("bearing_pad_thickness", pad, unit_display="\""),
             make_comp("concrete_class", conc, unit_display=" psi"),
             make_comp("anchor_bolt_embed", embed, unit_display="\"")]
    return comps, violation, "draw_n3_bearing", 3

# ---- N=4: approach slab (pavement + cover + joint + rebar grade) ----
def gen_n4(idx, compliant):
    pave = random.choice([6.0, 8.0, 10.0]) if compliant else random.choice([3.0, 4.0, 5.0])
    cover = random.choice([2.0, 2.5, 3.0]) if compliant else random.choice([1.0, 1.5])
    joint = random.choice([0.5, 0.75, 1.0]) if compliant else random.choice([0.25, 0.375])
    grade = random.choice([60, 75]) if compliant else random.choice([40, 50])
    violation = None
    if not compliant:
        failure = random.choice(["pave", "cover", "joint", "grade"])
        if failure == "pave":
            violation = {"failed_key": "pavement_thickness", "msg": f"Pavement {pave}\" < 6.0\" min"}
            cover = random.choice([2.0,2.5]); joint = random.choice([0.5,0.75]); grade = random.choice([60,75])
        elif failure == "cover":
            violation = {"failed_key": "concrete_cover", "msg": f"Cover {cover}\" < 2.0\" min"}
            pave = random.choice([6.0,8.0]); joint = random.choice([0.5,0.75]); grade = random.choice([60,75])
        elif failure == "joint":
            violation = {"failed_key": "joint_seal_width", "msg": f"Joint seal {joint}\" < 0.5\" min"}
            pave = random.choice([6.0,8.0]); cover = random.choice([2.0,2.5]); grade = random.choice([60,75])
        else:
            violation = {"failed_key": "rebar_grade", "msg": f"Rebar Grade {grade} < 60 min"}
            pave = random.choice([6.0,8.0]); cover = random.choice([2.0,2.5]); joint = random.choice([0.5,0.75])
    comps = [make_comp("pavement_thickness", pave, unit_display="\""),
             make_comp("concrete_cover", cover, unit_display="\""),
             make_comp("joint_seal_width", joint, unit_display="\""),
             make_comp("rebar_grade", grade, unit_display=" ksi")]
    return comps, violation, "draw_n4_approach", 4

# ---- N=5: pier (footing + cover + stirrup + bearing pad + anchor) ----
# CRITICAL FIX: stirrup_spacing threshold pre-resolved (d/2 for typical column = 12" max)
COLUMN_HEIGHT_IN = 24  # use a fixed 24" column for the stirrup_spacing rule check
STIRRUP_MAX = COLUMN_HEIGHT_IN // 2  # = 12

def gen_n5(idx, compliant):
    foot = random.choice([36, 42, 48, 54]) if compliant else random.choice([18, 22, 26])
    cover = random.choice([2.0, 2.5, 3.0]) if compliant else random.choice([1.0, 1.5])
    stirrup = random.choice([6, 8, 10, 12]) if compliant else random.choice([16, 20, 24, 28])
    pad = random.choice([1.0, 1.25, 1.5]) if compliant else random.choice([0.5, 0.75])
    embed = random.choice([12, 14, 16]) if compliant else random.choice([4, 6, 8])
    violation = None
    if not compliant:
        failure = random.choice(["foot", "cover", "stirrup", "pad", "embed"])
        if failure == "foot":
            violation = {"failed_key": "footing_depth", "msg": f"Footing {foot}\" < 30\" min"}
            cover = random.choice([2.0,2.5]); stirrup = random.choice([6,8,10]); pad = random.choice([1.0,1.25]); embed = random.choice([12,14])
        elif failure == "cover":
            violation = {"failed_key": "concrete_cover", "msg": f"Cover {cover}\" < 2.0\" min"}
            foot = random.choice([36,42]); stirrup = random.choice([6,8,10]); pad = random.choice([1.0,1.25]); embed = random.choice([12,14])
        elif failure == "stirrup":
            violation = {"failed_key": "stirrup_spacing", "msg": f"Stirrup {stirrup}\" > d/2={STIRRUP_MAX}\" max (ACI 318)"}
            foot = random.choice([36,42]); cover = random.choice([2.0,2.5]); pad = random.choice([1.0,1.25]); embed = random.choice([12,14])
        elif failure == "pad":
            violation = {"failed_key": "bearing_pad_thickness", "msg": f"Pad {pad}\" < 1.0\" min"}
            foot = random.choice([36,42]); cover = random.choice([2.0,2.5]); stirrup = random.choice([6,8,10]); embed = random.choice([12,14])
        else:
            violation = {"failed_key": "anchor_bolt_embed", "msg": f"Anchor {embed}\" < 12\" min"}
            foot = random.choice([36,42]); cover = random.choice([2.0,2.5]); stirrup = random.choice([6,8,10]); pad = random.choice([1.0,1.25])
    comps = [make_comp("footing_depth", foot, unit_display="\""),
             make_comp("concrete_cover", cover, unit_display="\""),
             # Pre-resolve d/2 to a numeric threshold per-drawing (fix for the n5_pier_ok miss)
             make_comp("stirrup_spacing", stirrup, threshold_override=STIRRUP_MAX, unit_display="\""),
             make_comp("bearing_pad_thickness", pad, unit_display="\""),
             make_comp("anchor_bolt_embed", embed, unit_display="\"")]
    return comps, violation, "draw_n5_pier", 5


GENS = {2: gen_n2, 3: gen_n3, 4: gen_n4, 5: gen_n5}


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for n in [2, 3, 4, 5]:
        for i in range(25):  # 25 per N-level
            compliant = (i % 2 == 0)  # alternate
            # Force exact 12-13 / 12-13 split per N
            if i >= 24: compliant = False
            comps, violation, fn, n_plans = GENS[n](i, compliant)
            name = f"n{n}_{i:03d}_{'ok' if compliant else 'bad'}"
            kwargs = dict(agency=random.choice(AGENCIES), plan_id=f"DSGN-N{n}-{i:03d}")
            out_path = f"{OUT}/{name}.png"
            getattr(A, fn)(out_path, components=comps, **kwargs)
            manifest.append({
                "name": name, "archetype": fn, "n_plans": n_plans,
                "image_path": out_path,
                "agency": kwargs["agency"], "plan_id": kwargs["plan_id"],
                "components": [{"key": c["key"], "value": c["value"],
                                "value_display": c["value_display"],
                                "threshold": c["threshold"], "operator": c["operator"]}
                               for c in comps],
                "ground_truth_compliant": compliant,
                "violated_rule": violation["msg"] if violation else None,
                "violated_key":  violation["failed_key"] if violation else None,
            })
    random.shuffle(manifest)
    mpath = f"{OUT}/manifest.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    n_c = sum(1 for m in manifest if m["ground_truth_compliant"])
    print(f"saved {len(manifest)} multi-plan drawings ({n_c} compliant + {len(manifest)-n_c} non-compliant)")
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
