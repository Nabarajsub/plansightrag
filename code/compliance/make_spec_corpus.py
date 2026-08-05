"""Render a realistic 'Standard Specifications' reference corpus (matplotlib).

DOT spec books genuinely publish minimum/maximum requirement TABLES; this corpus
mimics them so the agent can RETRIEVE the governing rule and EXTRACT its numeric
threshold visually -- instead of being handed the threshold. Each sheet carries
10-18 requirement rows (the 7 governing rules for our archetypes + distractors)
so retrieval and extraction are non-trivial. Multiple agencies + sections.

The 7 governing requirements (must be discoverable by retrieve+extract):
  concrete_cover        >= 2.0 in   (ACI 318 / AASHTO LRFD)
  rebar_grade           >= 60 ksi   (ASTM A615)
  concrete_class        >= 4000 psi
  stirrup_spacing       <= d/2      (ACI 318)  [symbolic -> agent must resolve]
  footing_depth (rail)  >= 30 in    (WYDOT 606.05)
  anchor_embed (sign)   >= 12 in    (AASHTO)
  grate_opening         <= 4.0 in   (PROWAG/ADA)
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---

import json, os
import matplotlib.pyplot as plt

OUT = f"{PSR_ROOT}/compliance/rule_grounding/spec_corpus"
os.makedirs(OUT, exist_ok=True)
AGENCIES = ["WYDOT", "Caltrans", "AZDOT", "CDOT", "FDOT"]

# Each sheet: (section_no, title, [ (item, requirement, value, reference) ... ])
# The governing rows are tagged via the manifest 'governs' key.
SECTIONS = {
    "concrete_reinf": {
        "no": "601", "title": "STRUCTURAL CONCRETE & REINFORCEMENT",
        "rows": [
            ("Min. clear cover, reinforced section", "minimum", "2.0 in", "ACI 318 Table 20.5.1.3", "concrete_cover"),
            ("Reinforcing steel, min. yield grade", "minimum", "60 ksi", "ASTM A615", "rebar_grade"),
            ("Structural concrete, min. 28-day strength", "minimum", "4000 psi", "Sec. 601.02", "concrete_class"),
            ("Max. shear-stirrup spacing", "maximum", "d/2", "ACI 318 Sec. 9.7.6.2.2", "stirrup_spacing"),
            ("Max. aggregate size", "maximum", "1.5 in", "Sec. 601.03", None),
            ("Min. lap splice length, #5 bar", "minimum", "24 in", "ACI 318", None),
            ("Air entrainment, exposed concrete", "range", "5-7 %", "Sec. 601.04", None),
            ("Min. curing period, moist", "minimum", "7 days", "Sec. 601.07", None),
            ("Max. water-cement ratio", "maximum", "0.45", "Sec. 601.02", None),
        ],
    },
    "foundations": {
        "no": "606", "title": "POSTS, FOUNDATIONS & ANCHORAGE",
        "rows": [
            ("Guardrail post footing, min. depth", "minimum", "30 in", "Sec. 606.05", "footing_depth"),
            ("Sign-post anchor bolt, min. embedment", "minimum", "12 in", "AASHTO LRFD 6.x", "anchor_embed"),
            ("Bearing pad, min. thickness", "minimum", "1.0 in", "Sec. 606.11", "bearing_pad"),
            ("Footing concrete, min. class", "minimum", "Class A", "Sec. 606.03", None),
            ("Backfill compaction", "minimum", "95 % Proctor", "Sec. 606.08", None),
            ("Anchor bolt, min. diameter", "minimum", "0.625 in", "Sec. 606.06", None),
            ("Post embedment, breakaway base", "minimum", "18 in", "Sec. 606.07", None),
            ("Min. footing diameter, drilled", "minimum", "10 in", "Sec. 606.04", None),
        ],
    },
    "drainage": {
        "no": "232", "title": "DRAINAGE STRUCTURES & INLETS",
        "rows": [
            ("Grate clear opening, pedestrian areas", "maximum", "4.0 in", "PROWAG R302 / ADA", "grate_opening"),
            ("Inlet throat, min. width", "minimum", "18 in", "Sec. 232.04", None),
            ("Catch-basin wall, min. thickness", "minimum", "6 in", "Sec. 232.03", None),
            ("Grate frame, casting grade", "minimum", "ASTM A48 CL 35B", "Sec. 232.06", None),
            ("Gravel bedding, min. depth", "minimum", "6 in", "Sec. 232.05", None),
            ("Pipe cover, min. over crown", "minimum", "12 in", "Sec. 232.09", None),
            ("Grate bearing bars, max. spacing", "maximum", "1.25 in", "Sec. 232.06", None),
        ],
    },
}


def render(agency, sec_key, sec):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=130)
    fig.patch.set_facecolor("white")
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.02, 0.965, agency, fontsize=20, family="monospace", weight="bold")
    ax.text(0.02, 0.93, "STANDARD SPECIFICATIONS", fontsize=11, family="monospace")
    ax.text(0.98, 0.965, f"SECTION {sec['no']}", fontsize=13, family="monospace",
            weight="bold", ha="right")
    ax.text(0.98, 0.93, sec["title"], fontsize=10, family="monospace", ha="right")
    ax.add_line(plt.Line2D([0.02, 0.98], [0.915, 0.915], color="black", lw=1.5))
    # table header
    y = 0.86
    ax.text(0.02, y, "ITEM", fontsize=9.5, family="monospace", weight="bold")
    ax.text(0.58, y, "LIMIT", fontsize=9.5, family="monospace", weight="bold")
    ax.text(0.70, y, "VALUE", fontsize=9.5, family="monospace", weight="bold")
    ax.text(0.86, y, "REFERENCE", fontsize=9.5, family="monospace", weight="bold")
    ax.add_line(plt.Line2D([0.02, 0.98], [y - 0.015, y - 0.015], color="black", lw=0.8))
    y -= 0.05
    for item, limit, value, ref, _ in sec["rows"]:
        ax.text(0.02, y, item[:46], fontsize=8.6, family="monospace")
        ax.text(0.58, y, limit, fontsize=8.6, family="monospace")
        ax.text(0.70, y, value, fontsize=8.6, family="monospace", weight="bold")
        ax.text(0.86, y, ref[:18], fontsize=7.6, family="monospace")
        ax.add_line(plt.Line2D([0.02, 0.98], [y - 0.012, y - 0.012], color="0.7", lw=0.3))
        y -= 0.042
    ax.text(0.02, 0.03, f"{agency} STD SPEC  |  SECTION {sec['no']}  |  SHEET 1 OF 1",
            fontsize=7.5, family="monospace", color="0.3")
    out = f"{OUT}/{agency}_{sec['no']}_{sec_key}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return out


def main():
    manifest = []
    for agency in AGENCIES:
        for sec_key, sec in SECTIONS.items():
            path = render(agency, sec_key, sec)
            governs = {g: v for (it, lim, v, ref, g) in sec["rows"] if g}
            manifest.append({"image_path": path, "agency": agency,
                             "section": sec["no"], "title": sec["title"],
                             "governs": governs})
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
    print(f"[spec] rendered {len(manifest)} spec sheets ({len(AGENCIES)} agencies x {len(SECTIONS)} sections) -> {OUT}")


if __name__ == "__main__":
    main()
