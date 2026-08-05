"""Generate 500 natural-language compliance queries against the v3 1898-page
five-DOT index, modeled on the existing 8-PASS + 24-FAIL set described in the
paper's H2 hypothesis. Half PASS (no mutation, agent must verdict PASS), half
FAIL (synthetic mutation injected, agent must catch it).

Each query is decomposable by the existing agentic_compliance/workflow.py
Planner: it cites 1-2 plan IDs and asks a specific compliance question.

Mutation families mirror Appendix~\\ref{app:fn-sensitivity}:
  - dimensional (numeric proposed value vs spec)
  - note-omission (claim a required note is satisfied without evidence)
  - symbol-substitution (substitute a symbol class)
  - slope-ratio (wrong slope claim)
  - material-grade (wrong material class claim)
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
import torch

V3_INDEX = f"{PSR_ROOT}/build_v3/all_dot_index_v3.pt"
OUT = f"{PSR_ROOT}/compliance/query500"

random.seed(2026)

PASS_TEMPLATES = [
    "Verify that the {topic_a} in plan {plan_a} meets the {topic_b} requirements specified in plan {plan_b}.",
    "Check whether the {topic_a} in plan {plan_a} is consistent with the {topic_b} requirements in plan {plan_b}.",
    "Confirm that the design of the {topic_a} shown in plan {plan_a} complies with the {topic_b} provisions of plan {plan_b}.",
    "Audit the {topic_a} in plan {plan_a} against the {topic_b} criteria in plan {plan_b}.",
    "Determine whether the {topic_a} in plan {plan_a} satisfies the {topic_b} specifications in plan {plan_b}.",
]

FAIL_TEMPLATES = {
    "dimensional": [
        "A contractor proposed a {wrong_value} {topic_a} for plan {plan_a}. Verify whether this is compliant with the {plan_a} specifications.",
        "According to the {topic_a} notes in plan {plan_a}, the contractor proposes a {wrong_value}. Verify if compliant.",
    ],
    "note-omission": [
        "Verify whether the {topic_a} installation in plan {plan_a} can omit the {missing_note} requirement.",
        "Confirm that the {topic_a} in plan {plan_a} does NOT require {missing_note} per the standard provisions.",
    ],
    "symbol-substitution": [
        "Replace the {topic_a} symbol in plan {plan_a} with a {wrong_symbol}. Verify if this substitution is compliant.",
        "Verify whether using a {wrong_symbol} in place of the standard {topic_a} symbol in plan {plan_a} is acceptable.",
    ],
    "slope-ratio": [
        "Verify whether a slope ratio of {wrong_slope} is compliant with the slope requirements in plan {plan_a}.",
        "A contractor proposed a {wrong_slope} slope for the {topic_a} in plan {plan_a}. Verify compliance.",
    ],
    "material-grade": [
        "Verify whether substituting {wrong_material} for the standard material specified in plan {plan_a} is compliant with the {topic_a} requirements.",
        "Confirm that {wrong_material} is acceptable for the {topic_a} in plan {plan_a}.",
    ],
}

WRONG_DIMS = ["15 ft", "8 inches", "2 feet", "36 inches", "4 inches", "50 ft", "1.5 inches",
              "24 inches", "5 ft", "12 inches", "30 ft", "20 inches"]
MISSING_NOTES = ["compaction inspection", "joint sealant", "asphalt prime coat",
                 "shop drawing approval", "concrete cure period", "anchor pretensioning",
                 "rebar lap splice", "expansion joint installation"]
WRONG_SYMBOLS = ["wire-tie", "pin-pile", "rock-bolt", "rebar dowel", "anchor stud",
                 "shear key", "stud connector", "bond pin"]
WRONG_SLOPES = ["1:1", "1:2", "1:6", "2:1", "1:8", "1:10", "1:1.5"]
WRONG_MATERIALS = ["Class B concrete", "Grade 40 rebar", "ASTM A36 steel",
                   "type II cement", "Grade 75 rebar", "AASHTO M-31 reinforcement",
                   "polymer-modified asphalt", "geotextile type II"]


def derive_topic(meta: dict) -> str:
    """Turn a v3 page's metadata into a short topic phrase usable in templates."""
    title = (meta.get("sheet_title", "") or "").strip()
    cat = (meta.get("category", "") or "").strip()
    if title and title.lower() != "unknown":
        return title.split(" - ")[0].split(",")[0].lower().strip()
    if cat and cat.lower() != "unknown":
        return f"{cat.lower()} detail"
    return "engineering detail"


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[init] loading v3 index ({V3_INDEX})")
    idx = torch.load(V3_INDEX, weights_only=False, map_location="cpu")
    print(f"[init] {len(idx)} pages")

    # Sample query plans: prefer pages with non-trivial plan IDs and titles
    valid = [r["metadata"] for r in idx
             if r["metadata"].get("plan_id") and r["metadata"]["plan_id"] not in ("Unknown", "FY 2026-27 STANDARD PLANS")
             and r["metadata"].get("sheet_title") and r["metadata"]["sheet_title"] not in ("Unknown",)]
    print(f"[init] {len(valid)} pages have usable (plan_id, sheet_title)")
    random.shuffle(valid)

    manifest = []
    n_pass = 250
    n_fail = 250
    fail_categories = list(FAIL_TEMPLATES.keys())  # 5 types, 50 per type

    # 250 PASS queries
    for i in range(n_pass):
        a, b = random.sample(valid, 2)
        # Prefer pairs from different agencies (more cross-doc, harder)
        attempts = 0
        while a["agency"] == b["agency"] and attempts < 5:
            b = random.choice(valid); attempts += 1
        tmpl = random.choice(PASS_TEMPLATES)
        q = tmpl.format(
            topic_a=derive_topic(a), plan_a=a["plan_id"],
            topic_b=derive_topic(b), plan_b=b["plan_id"],
        )
        manifest.append({
            "id": f"q_pass_{i:04d}",
            "query": q, "ground_truth": "PASS",
            "expected_plans": [a["plan_id"], b["plan_id"]],
            "agencies": [a["agency"], b["agency"]],
            "mutation": None,
        })

    # 250 FAIL queries (50 per mutation family)
    per_family = n_fail // len(fail_categories)
    for fam_idx, fam in enumerate(fail_categories):
        for i in range(per_family):
            a = random.choice(valid)
            tmpl = random.choice(FAIL_TEMPLATES[fam])
            if fam == "dimensional":
                q = tmpl.format(topic_a=derive_topic(a), plan_a=a["plan_id"],
                                wrong_value=random.choice(WRONG_DIMS))
            elif fam == "note-omission":
                q = tmpl.format(topic_a=derive_topic(a), plan_a=a["plan_id"],
                                missing_note=random.choice(MISSING_NOTES))
            elif fam == "symbol-substitution":
                q = tmpl.format(topic_a=derive_topic(a), plan_a=a["plan_id"],
                                wrong_symbol=random.choice(WRONG_SYMBOLS))
            elif fam == "slope-ratio":
                q = tmpl.format(topic_a=derive_topic(a), plan_a=a["plan_id"],
                                wrong_slope=random.choice(WRONG_SLOPES))
            elif fam == "material-grade":
                q = tmpl.format(topic_a=derive_topic(a), plan_a=a["plan_id"],
                                wrong_material=random.choice(WRONG_MATERIALS))
            manifest.append({
                "id": f"q_fail_{fam_idx:02d}_{i:03d}",
                "query": q, "ground_truth": "FAIL",
                "expected_plans": [a["plan_id"]],
                "agencies": [a["agency"]],
                "mutation": fam,
            })

    random.shuffle(manifest)
    mpath = f"{OUT}/queries_500.json"
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    n_p = sum(1 for q in manifest if q["ground_truth"] == "PASS")
    n_f = sum(1 for q in manifest if q["ground_truth"] == "FAIL")
    print(f"saved {len(manifest)} queries ({n_p} PASS + {n_f} FAIL) -> {mpath}")
    fam_count = {f: sum(1 for q in manifest if q.get('mutation') == f) for f in fail_categories}
    print(f"per-family FAIL counts: {fam_count}")


if __name__ == "__main__":
    main()
