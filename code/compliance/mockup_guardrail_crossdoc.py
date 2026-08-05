"""Cross-doc compliance archetype: GUARDRAIL POST FOUNDATION DETAIL.

Why cross-doc: footing depth required by each DOT differs.
  - WYDOT min footing depth: 30"
  - FDOT  min footing depth: 42"
  - This drawing shows 36" -> compliant with WYDOT, non-compliant with FDOT.

The agent must (a) identify the detail, (b) retrieve both DOT specs,
(c) compare against both, (d) issue per-DOT compliance verdict.
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
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
from matplotlib.lines import Line2D

POST_HEIGHT_IN   = 27.0   # above-grade post
FOOTING_DEPTH_IN = 36.0   # below-grade footing
FOOTING_DIA_IN   = 12.0
POST_SECTION     = "W6x9"
CONCRETE_CLASS   = "Class A"
PLAN_ID          = "606-7B"
AGENCY           = "WYDOT"
SHEET_TITLE      = "STEEL POST GUARDRAIL FOUNDATION - DETAIL"

fig, ax = plt.subplots(figsize=(11, 8.5), dpi=120)
ax.set_xlim(0, 22); ax.set_ylim(0, 17); ax.set_aspect("equal"); ax.axis("off")
fig.patch.set_facecolor("white")

# Sheet border + title block
ax.add_patch(Rectangle((0.4, 0.4), 21.2, 16.2, fill=False, lw=2.0))
ax.add_patch(Rectangle((0.4, 0.4), 21.2, 1.4, fill=False, lw=1.5))
ax.add_patch(Rectangle((15.5, 0.4), 6.1, 1.4, fill=False, lw=1.5))
ax.text(0.7, 1.4, AGENCY, fontsize=16, family="monospace", weight="bold")
ax.text(0.7, 0.95, "STANDARD PLAN", fontsize=8, family="monospace")
ax.text(0.7, 0.65, SHEET_TITLE, fontsize=11, family="monospace", weight="bold")
ax.text(15.7, 1.4, "PLAN NO.", fontsize=8, family="monospace")
ax.text(15.7, 0.95, PLAN_ID, fontsize=14, family="monospace", weight="bold")
ax.text(15.7, 0.65, "SHEET 1 OF 1", fontsize=8, family="monospace")

# Drawing area: ground line at y=10; post above, footing below
GROUND_Y = 10.0
ax.add_line(Line2D([4, 15], [GROUND_Y, GROUND_Y], color="black", lw=1.8))
# Ground hatching (above-line stipple-ish)
for x in [i * 0.4 + 4.1 for i in range(28)]:
    ax.add_line(Line2D([x, x + 0.25], [GROUND_Y, GROUND_Y + 0.25], color="black", lw=0.6))
ax.text(15.2, GROUND_Y, "GROUND LINE", fontsize=9, family="monospace", va="center")

# Post (above-grade): W6x9 schematic, ~6" wide
post_x_center = 9.5
post_w = 0.7
post_top = GROUND_Y + (POST_HEIGHT_IN / 12.0) * 1.4  # scale: 1ft -> 1.4 units
ax.add_patch(Rectangle((post_x_center - post_w / 2, GROUND_Y), post_w, post_top - GROUND_Y,
                       fill=False, lw=2.2, edgecolor="black"))
# H-section flange marks
ax.add_line(Line2D([post_x_center - post_w / 2 - 0.3, post_x_center + post_w / 2 + 0.3],
                   [GROUND_Y + 0.2, GROUND_Y + 0.2], color="black", lw=1.6))
ax.add_line(Line2D([post_x_center - post_w / 2 - 0.3, post_x_center + post_w / 2 + 0.3],
                   [post_top - 0.2, post_top - 0.2], color="black", lw=1.6))

# Footing (below-grade): concrete cylinder
footing_top = GROUND_Y
footing_bot = GROUND_Y - (FOOTING_DEPTH_IN / 12.0) * 1.4
footing_half_w = (FOOTING_DIA_IN / 12.0) * 0.7
fx0 = post_x_center - footing_half_w
fx1 = post_x_center + footing_half_w
ax.add_patch(Rectangle((fx0, footing_bot), fx1 - fx0, footing_top - footing_bot,
                       fill=False, lw=2.2))
# Concrete hatching inside footing
for i in range(-15, 25):
    x_start = fx0 + i * 0.18
    x_end = x_start + (footing_top - footing_bot)
    seg_x = [max(x_start, fx0), min(x_end, fx1)]
    seg_y = [footing_bot + (max(x_start, fx0) - x_start), footing_bot + (min(x_end, fx1) - x_start)]
    if seg_x[0] < seg_x[1]:
        ax.add_line(Line2D(seg_x, seg_y, color="black", lw=0.4))

# Dimensions
def dim_v(y0, y1, x, label, side=1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
        arrowstyle="<->", mutation_scale=10, lw=1.2, color="black"))
    ax.add_line(Line2D([x - 0.15, x + 0.15], [y0, y0], color="black", lw=1.0))
    ax.add_line(Line2D([x - 0.15, x + 0.15], [y1, y1], color="black", lw=1.0))
    ax.text(x + side * 0.20, (y0 + y1) / 2, label, va="center",
            fontsize=10, family="monospace", rotation=90)

def dim_h(x0, x1, y, label):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
        arrowstyle="<->", mutation_scale=10, lw=1.2, color="black"))
    ax.add_line(Line2D([x0, x0], [y - 0.15, y + 0.15], color="black", lw=1.0))
    ax.add_line(Line2D([x1, x1], [y - 0.15, y + 0.15], color="black", lw=1.0))
    ax.text((x0 + x1) / 2, y + 0.20, label, ha="center", fontsize=10, family="monospace")

dim_v(GROUND_Y, post_top, 12.5, f'{POST_HEIGHT_IN:.0f}"  POST HT.', side=1)
dim_v(footing_bot, GROUND_Y, 12.5, f'{FOOTING_DEPTH_IN:.0f}"  FOOTING DEPTH', side=1)
dim_h(fx0, fx1, footing_bot - 0.8, f'{FOOTING_DIA_IN:.0f}" DIA.')

# Material leaders
def leader(xt, yt, xp, yp, text):
    ax.add_line(Line2D([xt + 0.1, xp], [yt, yp], color="black", lw=0.8))
    ax.plot(xp, yp, "ko", markersize=4)
    ax.text(xt, yt + 0.05, text, fontsize=9, family="monospace")

leader(16.0, 14.0, post_x_center + post_w / 2, post_top - 0.5, f"POST: {POST_SECTION} STEEL")
leader(16.0, 12.5, fx1, (footing_top + footing_bot) / 2, f"{CONCRETE_CLASS} CONCRETE FOOTING")

# Notes
notes = [
    "NOTES:",
    "1. POST: W6x9 STEEL, ASTM A992.",
    "2. FOOTING: CLASS A CONCRETE.",
    f"3. MIN FOOTING DEPTH PER {AGENCY} 606.05 = 30\".",
    "4. INSTALLATION ON FILL REQUIRES",
    "   COMPACTION TO 95% PROCTOR.",
    "5. SEE SHEET 606-7A FOR PLAN VIEW.",
]
for i, n in enumerate(notes):
    ax.text(16.0, 9.0 - i * 0.45, n, fontsize=8.5, family="monospace",
            weight="bold" if i == 0 else "normal")

# Detail label
ax.text(post_x_center, post_top + 0.8, "TYP. POST FOUNDATION DETAIL",
        ha="center", fontsize=12, family="monospace", weight="bold")

out = f"{PSR_ROOT}/compliance/mockups/sample_guardrail_crossdoc.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved -> {out}")
