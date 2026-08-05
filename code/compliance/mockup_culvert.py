"""Mockup: single concrete-box-culvert cross-section drawing rendered as PNG.
Demonstrates the visual style for the 500 D1 + 100 D3 CAD drawings:
  - Engineering plan aesthetic: thick black lines, hatching, dimension arrows.
  - Title block + numbered notes + material callouts.
  - Parameterized so 500 unique instances (different spans, covers, rebar) are trivial.

This is just one archetype. The full library will include: guardrail elevation,
rebar detail, drainage inlet, sign-post foundation, retaining wall, etc.
"""
from __future__ import annotations
# --- release path resolution ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# Rasterized plan pages are NOT redistributed. Rebuild them from the public DOT
# PDFs (see README) and point PLANS_ROOT at the output directory.
PLANS_ROOT = _os.environ.get("PLANS_ROOT") or _os.path.join(PSR_ROOT, "data", "pages")
# --- end release path resolution ---


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

# ---------- Drawing parameters (parameterize these to make 500 variants) ----------
SPAN_FT       = 8.0       # clear span
RISE_FT       = 5.0       # clear height
WALL_IN       = 12.0      # wall thickness
TOP_SLAB_IN   = 14.0      # top slab thickness
BOT_SLAB_IN   = 14.0      # bottom slab thickness
COVER_IN      = 2.5       # required concrete cover
REBAR_SPACING_IN = 9.0    # main reinforcing bar spacing
CONCRETE_CLASS = "Class A-A"
REBAR_GRADE    = "Grade 60"
PLAN_ID  = "B-505 (M)"
SHEET_TITLE = "REINFORCED CONCRETE BOX CULVERT - SECTION"
AGENCY = "WYDOT"

# ---------- Figure setup: 11x17 landscape (typical engineering sheet, scaled) ----------
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=120)
ax.set_xlim(0, 22)
ax.set_ylim(0, 17)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# Border (sheet boundary) + title block
ax.add_patch(Rectangle((0.4, 0.4), 21.2, 16.2, fill=False, lw=2.0, edgecolor="black"))
ax.add_patch(Rectangle((0.4, 0.4), 21.2, 1.4, fill=False, lw=1.5, edgecolor="black"))
ax.add_patch(Rectangle((15.5, 0.4), 6.1, 1.4, fill=False, lw=1.5, edgecolor="black"))

ax.text(0.7, 1.4, AGENCY, fontsize=16, family="monospace", weight="bold")
ax.text(0.7, 0.95, "STANDARD PLAN", fontsize=8, family="monospace")
ax.text(0.7, 0.65, SHEET_TITLE, fontsize=11, family="monospace", weight="bold")
ax.text(15.7, 1.4, "PLAN NO.", fontsize=8, family="monospace")
ax.text(15.7, 0.95, PLAN_ID, fontsize=14, family="monospace", weight="bold")
ax.text(15.7, 0.65, "SHEET 1 OF 1", fontsize=8, family="monospace")

# ---------- Coordinate transform: drawing area is x in [4,16], y in [3,14] ----------
DRAW_X0, DRAW_Y0 = 4.5, 3.5
DRAW_W,  DRAW_H  = 11.0, 9.0

def ft2u(ft, axis="x"):
    """Project feet -> figure units inside the drawing area."""
    if axis == "x":
        return DRAW_X0 + (ft / (SPAN_FT + WALL_IN / 6.0 + 2)) * DRAW_W
    return DRAW_Y0 + (ft / (RISE_FT + (TOP_SLAB_IN + BOT_SLAB_IN) / 12.0 + 2)) * DRAW_H

# Box culvert outer dimensions (in feet)
wall_ft   = WALL_IN / 12.0
top_ft    = TOP_SLAB_IN / 12.0
bot_ft    = BOT_SLAB_IN / 12.0
outer_w   = SPAN_FT + 2 * wall_ft
outer_h   = RISE_FT + top_ft + bot_ft

# Anchor box origin (left wall outer, bottom slab outer)
x0 = ft2u(1.0)
y0 = ft2u(1.0)
x_outer_w = ft2u(1.0 + outer_w) - x0
y_outer_h = ft2u(1.0 + outer_h) - y0

# Outer box
ax.add_patch(Rectangle((x0, y0), x_outer_w, y_outer_h, fill=False, lw=2.5, edgecolor="black"))

# Inner opening (clear span x rise)
x1 = ft2u(1.0 + wall_ft)
y1 = ft2u(1.0 + bot_ft)
x_inner_w = ft2u(1.0 + wall_ft + SPAN_FT) - x1
y_inner_h = ft2u(1.0 + bot_ft + RISE_FT) - y1
ax.add_patch(Rectangle((x1, y1), x_inner_w, y_inner_h, fill=True, facecolor="white", lw=2.5, edgecolor="black"))

# Concrete hatching: diagonal lines inside walls/slabs
def hatch_band(x_lo, y_lo, x_hi, y_hi, spacing=0.15):
    pts = []
    diag = (x_hi - x_lo) + (y_hi - y_lo)
    n = int(diag / spacing) + 1
    for i in range(-n, n + 1):
        x_start = x_lo + i * spacing
        x_end = x_start + (y_hi - y_lo)
        # clip to band
        seg_x = [max(x_start, x_lo), min(x_end, x_hi)]
        seg_y = [y_lo + (max(x_start, x_lo) - x_start), y_lo + (min(x_end, x_hi) - x_start)]
        if seg_x[0] < seg_x[1]:
            ax.add_line(Line2D(seg_x, seg_y, color="black", lw=0.4))

# Left wall, right wall, top slab, bottom slab hatching
hatch_band(x0, y0, x1, y0 + y_outer_h)                      # left wall
hatch_band(x1 + x_inner_w, y0, x0 + x_outer_w, y0 + y_outer_h)  # right wall
hatch_band(x1, y1 + y_inner_h, x1 + x_inner_w, y0 + y_outer_h)  # top slab
hatch_band(x1, y0, x1 + x_inner_w, y1)                          # bottom slab

# Rebar dots (top mat in top slab)
n_bars_top = int((x_inner_w) * (REBAR_SPACING_IN / 12.0) ** -1) + 1
for i in range(n_bars_top):
    bx = x1 + (i + 0.5) * x_inner_w / max(n_bars_top, 1)
    by = y1 + y_inner_h + (top_ft / 2) * (DRAW_H / (RISE_FT + top_ft + bot_ft + 2))
    ax.plot(bx, by, "ko", markersize=3.0)

# ---------- Dimensioning ----------
def dim_horizontal(xa, xb, y, label, offset=0.5):
    ax.add_patch(FancyArrowPatch((xa, y), (xb, y),
        arrowstyle="<->", mutation_scale=10, lw=1.2, color="black"))
    ax.add_line(Line2D([xa, xa], [y - 0.15, y + 0.15], color="black", lw=1.0))
    ax.add_line(Line2D([xb, xb], [y - 0.15, y + 0.15], color="black", lw=1.0))
    ax.text((xa + xb) / 2, y + 0.18, label, ha="center", fontsize=10, family="monospace")

def dim_vertical(ya, yb, x, label, offset=0.5):
    ax.add_patch(FancyArrowPatch((x, ya), (x, yb),
        arrowstyle="<->", mutation_scale=10, lw=1.2, color="black"))
    ax.add_line(Line2D([x - 0.15, x + 0.15], [ya, ya], color="black", lw=1.0))
    ax.add_line(Line2D([x - 0.15, x + 0.15], [yb, yb], color="black", lw=1.0))
    ax.text(x + 0.20, (ya + yb) / 2, label, va="center", fontsize=10, family="monospace", rotation=90)

# Span dimension (between inner walls)
dim_horizontal(x1, x1 + x_inner_w, y0 - 0.7, f"{SPAN_FT:.1f}' CLEAR SPAN")
# Rise dimension (inner walls)
dim_vertical(y1, y1 + y_inner_h, x0 - 0.7, f"{RISE_FT:.1f}' CLEAR RISE")
# Wall thickness on right wall
dim_horizontal(x1 + x_inner_w, x0 + x_outer_w, y1 + y_inner_h + 0.35, f'{WALL_IN:.0f}"')
# Top slab thickness
dim_vertical(y1 + y_inner_h, y0 + y_outer_h, x0 + x_outer_w + 0.35, f'{TOP_SLAB_IN:.0f}"')

# ---------- Material callout leader lines ----------
def leader(x_text, y_text, x_tip, y_tip, text):
    ax.add_line(Line2D([x_text + 0.1, x_tip], [y_text, y_tip], color="black", lw=0.8))
    ax.plot(x_tip, y_tip, "ko", markersize=4)
    ax.text(x_text, y_text + 0.05, text, fontsize=9, family="monospace")

leader(16.5, 13.0, x1 + x_inner_w / 2, y1 + y_inner_h + top_ft * (DRAW_H / (RISE_FT + top_ft + bot_ft + 2)) * 0.5,
       f"#5 BARS @ {REBAR_SPACING_IN:.0f}\" O.C. - {REBAR_GRADE}")
leader(16.5, 11.5, x0 + 0.3, y0 + y_outer_h / 2,
       f"{CONCRETE_CLASS} CONCRETE")
leader(16.5, 10.0, x0 + wall_ft * (DRAW_W / (SPAN_FT + wall_ft + 2)) * 0.5 + x0, y0 + y_outer_h * 0.3,
       f'COVER = {COVER_IN:.1f}" MIN')

# ---------- Notes block ----------
notes = [
    "NOTES:",
    "1. ALL CONCRETE SHALL BE CLASS A-A.",
    "2. REINFORCING STEEL: ASTM A615 GR 60.",
    "3. MIN COVER 2\" ALL SURFACES UNLESS NOTED.",
    "4. PROVIDE 1\" CHAMFER ALL EXPOSED CORNERS.",
    "5. JOINT SEALANT PER WYDOT 706.04.",
]
for i, n in enumerate(notes):
    ax.text(16.4, 9.0 - i * 0.4, n, fontsize=8.5, family="monospace",
            weight="bold" if i == 0 else "normal")

# Section label
ax.text((x0 + x_outer_w / 2), y0 + y_outer_h + 0.7, "SECTION A-A",
        ha="center", fontsize=12, family="monospace", weight="bold")

# Save
out = f"{PSR_ROOT}/compliance/mockups/sample_culvert.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved -> {out}")
