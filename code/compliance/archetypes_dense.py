"""Dense engineering-drawing variants: multi-view + schedule tables + many
dimensions, mimicking real DOT standard-plan sheets. Violations are embedded
in one specific row/cell so the judge must locate them within a busy drawing.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


def _frame(ax, agency, plan_id, sheet_title):
    ax.set_xlim(0, 22); ax.set_ylim(0, 17); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0.4, 0.4), 21.2, 16.2, fill=False, lw=2.0))
    ax.add_patch(Rectangle((0.4, 0.4), 21.2, 1.4, fill=False, lw=1.5))
    ax.add_patch(Rectangle((15.5, 0.4), 6.1, 1.4, fill=False, lw=1.5))
    ax.text(0.7, 1.4, agency, fontsize=16, family="monospace", weight="bold")
    ax.text(0.7, 0.95, "STANDARD PLAN", fontsize=8, family="monospace")
    ax.text(0.7, 0.65, sheet_title, fontsize=11, family="monospace", weight="bold")
    ax.text(15.7, 1.4, "PLAN NO.", fontsize=8, family="monospace")
    ax.text(15.7, 0.95, plan_id, fontsize=14, family="monospace", weight="bold")
    ax.text(15.7, 0.65, "SHEET 1 OF 1", fontsize=8, family="monospace")


def _hatch(ax, x0, y0, x1, y1, spacing=0.13):
    n = int(((x1 - x0) + (y1 - y0)) / spacing) + 1
    for i in range(-n, n + 1):
        xs = x0 + i * spacing
        xe = xs + (y1 - y0)
        a = max(xs, x0); b = min(xe, x1)
        if a < b:
            ax.add_line(Line2D([a, b], [y0 + (a - xs), y0 + (b - xs)], color="black", lw=0.4))


def _dim_h(ax, x0, x1, y, label, fs=9):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="<->", mutation_scale=8, lw=1.0))
    ax.add_line(Line2D([x0, x0], [y - 0.1, y + 0.1], color="black", lw=0.9))
    ax.add_line(Line2D([x1, x1], [y - 0.1, y + 0.1], color="black", lw=0.9))
    ax.text((x0 + x1) / 2, y + 0.12, label, ha="center", fontsize=fs, family="monospace")


def _dim_v(ax, y0, y1, x, label, fs=9):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="<->", mutation_scale=8, lw=1.0))
    ax.add_line(Line2D([x - 0.1, x + 0.1], [y0, y0], color="black", lw=0.9))
    ax.add_line(Line2D([x - 0.1, x + 0.1], [y1, y1], color="black", lw=0.9))
    ax.text(x + 0.13, (y0 + y1) / 2, label, va="center", fontsize=fs, family="monospace", rotation=90)


def _table(ax, x, y, rows, col_widths, font_size=8, row_height=0.32):
    n_cols = len(col_widths)
    table_w = sum(col_widths)
    # Header background
    ax.add_patch(Rectangle((x, y - row_height), table_w, row_height, fill=False, lw=1.3))
    cx = x
    for i, w in enumerate(col_widths):
        ax.text(cx + w / 2, y - row_height / 2, str(rows[0][i]), ha="center", va="center",
                fontsize=font_size, family="monospace", weight="bold")
        cx += w
        if i < n_cols - 1:
            ax.add_line(Line2D([cx, cx], [y, y - row_height], color="black", lw=0.9))
    # Rows
    for r_idx, row in enumerate(rows[1:], start=1):
        ry = y - row_height * (r_idx + 1)
        ax.add_patch(Rectangle((x, ry), table_w, row_height, fill=False, lw=0.9))
        cx = x
        for i, w in enumerate(col_widths):
            ax.text(cx + w / 2, ry + row_height / 2, str(row[i]), ha="center", va="center",
                    fontsize=font_size, family="monospace")
            cx += w
            if i < n_cols - 1:
                ax.add_line(Line2D([cx, cx], [ry, ry + row_height], color="black", lw=0.7))


def _save(fig, out):
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- Dense culvert: PLAN + SECTION + ELEVATION + SCHEDULE ----------
def draw_culvert_dense(out, *, agency, plan_id, schedule, hero_idx, cover_in, rebar_grade,
                        violated):
    """Multi-view culvert sheet with a SCHEDULE TABLE of multiple span variants.
    `schedule` is list of dicts: [{span_ft, rise_ft, wall_in, top_in, cover_in, stirrup_in}]
    `hero_idx` picks which row is detailed in the section view.
    `violated`: if True, one cell in the schedule violates a rule (we explicitly set it).
    """
    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=110); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "REINFORCED CONCRETE BOX CULVERT - SCHEDULE & DETAILS")

    # SECTION view (top-left), shows the hero row
    hero = schedule[hero_idx]
    DX0, DY0 = 1.5, 8.5; DW, DH = 8.0, 6.5
    sf = hero["span_ft"]; rf = hero["rise_ft"]
    wi = hero["wall_in"]; ti = hero["top_in"]
    sx = DW / (sf + 2 * wi / 12.0 + 1.5)
    sy = DH / (rf + 2 * ti / 12.0 + 1.5)
    x0 = DX0 + 0.7; y0 = DY0 + 0.7
    wft = wi / 12.0; tft = ti / 12.0
    outer_w = (sf + 2 * wft) * sx; outer_h = (rf + 2 * tft) * sy
    ax.add_patch(Rectangle((x0, y0), outer_w, outer_h, fill=False, lw=2.2))
    x1 = x0 + wft * sx; y1 = y0 + tft * sy
    inner_w = sf * sx; inner_h = rf * sy
    ax.add_patch(Rectangle((x1, y1), inner_w, inner_h, facecolor="white", lw=2.0))
    _hatch(ax, x0, y0, x1, y0 + outer_h)
    _hatch(ax, x1 + inner_w, y0, x0 + outer_w, y0 + outer_h)
    _hatch(ax, x1, y1 + inner_h, x1 + inner_w, y0 + outer_h)
    _hatch(ax, x1, y0, x1 + inner_w, y1)
    _dim_h(ax, x1, x1 + inner_w, y0 - 0.4, f'{sf:.1f}\' SPAN')
    _dim_v(ax, y1, y1 + inner_h, x0 - 0.4, f'{rf:.1f}\' RISE')
    _dim_h(ax, x1 + inner_w, x0 + outer_w, y1 + inner_h + 0.25, f'{wi}"')
    ax.text(x0 + outer_w / 2, y0 + outer_h + 0.4, f"SECTION A-A (TYPE {chr(65 + hero_idx)})",
            ha="center", fontsize=10, family="monospace", weight="bold")

    # ELEVATION view (top-right), schematic outline
    EX0, EY0 = 11.0, 9.5; EW, EH = 9.0, 4.5
    ax.add_patch(Rectangle((EX0 + 0.5, EY0 + 0.5), EW - 1.0, EH - 1.0, fill=False, lw=2.0))
    for i in range(3):
        ax.add_patch(Rectangle((EX0 + 1.0 + i * 2.5, EY0 + 0.7), 2.0, EH - 1.4,
                                fill=False, lw=1.0, linestyle="--"))
    ax.text(EX0 + EW / 2, EY0 + EH - 0.3, "ELEVATION (3-CELL ASSEMBLY)",
            ha="center", fontsize=10, family="monospace", weight="bold")

    # SCHEDULE table (bottom)
    rows = [["TYPE", "SPAN", "RISE", "WALL", "TOP SLAB", "COVER", "STIRRUPS"]]
    for i, s in enumerate(schedule):
        rows.append([
            chr(65 + i),
            f'{s["span_ft"]:.1f}\'',
            f'{s["rise_ft"]:.1f}\'',
            f'{s["wall_in"]}"',
            f'{s["top_in"]}"',
            f'{s["cover_in"]:.1f}"',
            f'{s["stirrup_in"]}" OC',
        ])
    _table(ax, 1.5, 7.0, rows, col_widths=[1.2, 1.4, 1.4, 1.2, 1.8, 1.4, 2.0])

    # PLAN-VIEW box (bottom-right) with bearing pad notes
    PX, PY = 13.5, 2.0
    ax.add_patch(Rectangle((PX, PY), 6.5, 4.0, fill=False, lw=1.8))
    ax.add_patch(Rectangle((PX + 0.4, PY + 0.4), 5.7, 3.2, fill=False, lw=1.4))
    _dim_h(ax, PX + 0.4, PX + 6.1, PY + 0.15, "PLAN W = 28'-0\"", fs=8)
    ax.text(PX + 3.25, PY + 4.1, "PLAN VIEW", ha="center", fontsize=10,
            family="monospace", weight="bold")

    # NOTES block (left below section)
    notes = [
        "NOTES:",
        f"1. CONCRETE: CLASS A-A (f'c = 4500 PSI).",
        f"2. REINFORCEMENT: ASTM A615 {rebar_grade}.",
        f"3. MIN CLEAR COVER: {cover_in:.1f}\" ALL EXPOSED SURFACES.",
        "4. STIRRUP SPACING IS s_max = d/2 PER ACI 318-19.",
        "5. JOINT SEALANT PER STD SPEC 706.04.",
        "6. EARTHWORK PER STD SPEC 203.",
        "7. CAMBER: 1/8\" PER 10' OF SPAN.",
        "8. INSTALL PER MANUFACTURER & STD 511.",
    ]
    for i, n in enumerate(notes):
        ax.text(1.5, 6.4 - i * 0.32, n, fontsize=8, family="monospace",
                weight="bold" if i == 0 else "normal")

    _save(fig, out)


# ---------- Dense beam rebar: SCHEDULE of beam types + section + stirrup detail ----------
def draw_rebar_dense(out, *, agency, plan_id, schedule, hero_idx, violated):
    """Multi-view rebar sheet with a SCHEDULE of multiple beam sizes/configurations."""
    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=110); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "REINFORCED CONCRETE BEAM - SCHEDULE & DETAILS")

    hero = schedule[hero_idx]
    # SECTION view (top-left)
    cx_, cy_ = 4.5, 11.0
    w = hero["width_in"] / 4.0; h = hero["height_in"] / 4.0
    x0 = cx_ - w / 2; y0 = cy_ - h / 2
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=2.0))
    co = 0.4
    # Top + bottom bars
    n_top = hero["n_top"]; n_bot = hero["n_bot"]
    for i in range(n_top):
        bx = x0 + co + (i + 0.5) * (w - 2 * co) / n_top
        ax.plot(bx, y0 + h - co, "ko", markersize=5)
    for i in range(n_bot):
        bx = x0 + co + (i + 0.5) * (w - 2 * co) / n_bot
        ax.plot(bx, y0 + co, "ko", markersize=5)
    # Stirrup outline
    ax.add_patch(Rectangle((x0 + co * 0.7, y0 + co * 0.7), w - 1.4 * co, h - 1.4 * co,
                            fill=False, lw=1.0, linestyle="--"))
    _dim_h(ax, x0, x0 + w, y0 - 0.4, f'{hero["width_in"]}"')
    _dim_v(ax, y0, y0 + h, x0 - 0.5, f'{hero["height_in"]}"')
    ax.text(cx_, y0 + h + 0.4, f"SECTION (TYPE {chr(65 + hero_idx)})", ha="center",
            fontsize=10, family="monospace", weight="bold")

    # ELEVATION (top-right): beam with stirrup spacing marks
    EX, EY = 9.5, 9.5
    ax.add_patch(Rectangle((EX + 0.5, EY + 0.5), 10.0, 3.0, fill=False, lw=2.0))
    s_oc = hero["stirrup_in"]
    n_stir = int(120 / s_oc)  # over a 10ft beam
    for i in range(min(n_stir, 12)):
        sx = EX + 0.7 + i * 9.6 / 12
        ax.add_line(Line2D([sx, sx], [EY + 0.6, EY + 3.4], color="black", lw=0.8))
    ax.text(EX + 5.0, EY + 3.7, "ELEVATION", ha="center", fontsize=10, family="monospace", weight="bold")

    # SCHEDULE table
    rows = [["TYPE", "W", "H", "TOP BARS", "BOT BARS", "STIRRUPS @ OC", "COVER"]]
    for i, s in enumerate(schedule):
        rows.append([
            chr(65 + i),
            f'{s["width_in"]}"',
            f'{s["height_in"]}"',
            f'{s["n_top"]}-#{s["bar_size"]}',
            f'{s["n_bot"]}-#{s["bar_size"]}',
            f'{s["stirrup_in"]}"',
            f'{s["cover_in"]}"',
        ])
    _table(ax, 1.5, 7.5, rows, col_widths=[1.2, 1.4, 1.4, 1.8, 1.8, 2.0, 1.4])

    # NOTES
    notes = [
        "NOTES:",
        "1. CONCRETE: CLASS S2 (f'c = 5000 PSI).",
        "2. MAIN REBAR: ASTM A615 GR 60.",
        "3. STIRRUPS: #3 BARS, MAX SPACING = d/2 PER ACI 318.",
        "4. CLEAR COVER PER TABLE ABOVE (MIN 2\" PER ACI).",
        "5. SPLICE LENGTH PER ACI 318-19 TABLE 25.4.2.2.",
        "6. EPOXY COATING NOT REQUIRED ON INTERIOR SURFACES.",
        "7. NO REBAR SHALL BE SPLICED WITHIN PLASTIC HINGE ZONE.",
        "8. SAW-CUT JOINTS PER STD SPEC.",
    ]
    for i, n in enumerate(notes):
        ax.text(11.5, 7.4 - i * 0.32, n, fontsize=8, family="monospace",
                weight="bold" if i == 0 else "normal")

    _save(fig, out)
