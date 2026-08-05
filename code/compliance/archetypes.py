"""5 reusable engineering-drawing archetype templates (matplotlib PNG).

Each function takes a parameter dict and renders a PNG. Parameters that have a
spec-bound (min/max) version live in `compliance_rules.py` so the same template
can produce compliant and non-compliant instances.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
from matplotlib.lines import Line2D


def _frame(ax, agency: str, plan_id: str, sheet_title: str):
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


def _hatch(ax, x0, y0, x1, y1, spacing=0.15):
    n = int(((x1 - x0) + (y1 - y0)) / spacing) + 1
    for i in range(-n, n + 1):
        xs = x0 + i * spacing
        xe = xs + (y1 - y0)
        a = max(xs, x0); b = min(xe, x1)
        if a < b:
            ax.add_line(Line2D([a, b],
                               [y0 + (a - xs), y0 + (b - xs)],
                               color="black", lw=0.4))


def _dim_h(ax, x0, x1, y, label):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="<->", mutation_scale=10, lw=1.2))
    ax.add_line(Line2D([x0, x0], [y - 0.15, y + 0.15], color="black", lw=1.0))
    ax.add_line(Line2D([x1, x1], [y - 0.15, y + 0.15], color="black", lw=1.0))
    ax.text((x0 + x1) / 2, y + 0.20, label, ha="center", fontsize=10, family="monospace")


def _dim_v(ax, y0, y1, x, label):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="<->", mutation_scale=10, lw=1.2))
    ax.add_line(Line2D([x - 0.15, x + 0.15], [y0, y0], color="black", lw=1.0))
    ax.add_line(Line2D([x - 0.15, x + 0.15], [y1, y1], color="black", lw=1.0))
    ax.text(x + 0.20, (y0 + y1) / 2, label, va="center", fontsize=10, family="monospace", rotation=90)


def _leader(ax, xt, yt, xp, yp, text):
    ax.add_line(Line2D([xt + 0.1, xp], [yt, yp], color="black", lw=0.8))
    ax.plot(xp, yp, "ko", markersize=4)
    ax.text(xt, yt + 0.05, text, fontsize=9, family="monospace")


def _notes(ax, notes, x=16.0, y0=9.0, dy=0.42):
    for i, n in enumerate(notes):
        ax.text(x, y0 - i * dy, n, fontsize=8.5, family="monospace",
                weight="bold" if i == 0 else "normal")


def _save(fig, out):
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------- 1. CULVERT SECTION ----------
def draw_culvert(out, *, agency, plan_id, span_ft, rise_ft, wall_in, top_slab_in,
                 cover_in, rebar_spacing_in, concrete_class, rebar_grade):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=120); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "REINFORCED CONCRETE BOX CULVERT - SECTION")
    DX0, DY0, DW, DH = 4.5, 3.5, 11.0, 9.0
    wall_ft = wall_in / 12.0; top_ft = top_slab_in / 12.0; bot_ft = top_slab_in / 12.0
    sx = DW / (span_ft + 2 * wall_ft + 2); sy = DH / (rise_ft + top_ft + bot_ft + 2)
    x0 = DX0 + sx; y0 = DY0 + sy
    x1 = x0 + wall_ft * sx; y1 = y0 + bot_ft * sy
    inner_w = span_ft * sx; inner_h = rise_ft * sy
    outer_w = (span_ft + 2 * wall_ft) * sx; outer_h = (rise_ft + top_ft + bot_ft) * sy
    ax.add_patch(Rectangle((x0, y0), outer_w, outer_h, fill=False, lw=2.5))
    ax.add_patch(Rectangle((x1, y1), inner_w, inner_h, facecolor="white", lw=2.5))
    # Hatching (4 bands)
    _hatch(ax, x0, y0, x1, y0 + outer_h)
    _hatch(ax, x1 + inner_w, y0, x0 + outer_w, y0 + outer_h)
    _hatch(ax, x1, y1 + inner_h, x1 + inner_w, y0 + outer_h)
    _hatch(ax, x1, y0, x1 + inner_w, y1)
    # Rebar dots
    n = max(int(inner_w * 12.0 / rebar_spacing_in), 2)
    for i in range(n):
        bx = x1 + (i + 0.5) * inner_w / n
        ax.plot(bx, y1 + inner_h + top_ft * sy * 0.5, "ko", markersize=3)
    _dim_h(ax, x1, x1 + inner_w, y0 - 0.7, f"{span_ft:.1f}' CLEAR SPAN")
    _dim_v(ax, y1, y1 + inner_h, x0 - 0.7, f"{rise_ft:.1f}' CLEAR RISE")
    _dim_h(ax, x1 + inner_w, x0 + outer_w, y1 + inner_h + 0.35, f'{wall_in:.0f}"')
    _dim_v(ax, y1 + inner_h, y0 + outer_h, x0 + outer_w + 0.35, f'{top_slab_in:.0f}"')
    _leader(ax, 16.0, 13.0, x1 + inner_w / 2, y1 + inner_h + top_ft * sy * 0.5,
            f'#5 BARS @ {rebar_spacing_in:.0f}" OC - {rebar_grade}')
    _leader(ax, 16.0, 11.5, x0 + 0.2, y0 + outer_h / 2, f"{concrete_class} CONCRETE")
    _leader(ax, 16.0, 10.0, x0 + wall_ft * sx * 0.4, y0 + outer_h * 0.3,
            f'COVER = {cover_in:.1f}" MIN')
    _notes(ax, [
        "NOTES:", f"1. {concrete_class} CONCRETE.",
        f"2. REBAR ASTM A615 {rebar_grade}.",
        f"3. MIN COVER {cover_in:.1f}\" ALL SURFACES.",
        "4. JOINT SEALANT PER STD SPEC.",
    ])
    ax.text(x0 + outer_w / 2, y0 + outer_h + 0.7, "SECTION A-A",
            ha="center", fontsize=12, family="monospace", weight="bold")
    _save(fig, out)


# ---------- 2. GUARDRAIL POST FOUNDATION ----------
def draw_guardrail(out, *, agency, plan_id, post_height_in, footing_depth_in,
                   footing_dia_in, post_section, concrete_class, min_footing_note_in):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=120); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "STEEL POST GUARDRAIL FOUNDATION - DETAIL")
    GY = 10.0
    ax.add_line(Line2D([4, 15], [GY, GY], color="black", lw=1.8))
    for x in [i * 0.4 + 4.1 for i in range(28)]:
        ax.add_line(Line2D([x, x + 0.25], [GY, GY + 0.25], color="black", lw=0.6))
    ax.text(15.2, GY, "GROUND LINE", fontsize=9, family="monospace", va="center")
    pcx = 9.5; pw = 0.7
    pt = GY + (post_height_in / 12.0) * 1.4
    ax.add_patch(Rectangle((pcx - pw / 2, GY), pw, pt - GY, fill=False, lw=2.2))
    ax.add_line(Line2D([pcx - pw / 2 - 0.3, pcx + pw / 2 + 0.3], [GY + 0.2, GY + 0.2], color="black", lw=1.6))
    ax.add_line(Line2D([pcx - pw / 2 - 0.3, pcx + pw / 2 + 0.3], [pt - 0.2, pt - 0.2], color="black", lw=1.6))
    fb = GY - (footing_depth_in / 12.0) * 1.4
    fhw = (footing_dia_in / 12.0) * 0.7
    fx0 = pcx - fhw; fx1 = pcx + fhw
    ax.add_patch(Rectangle((fx0, fb), fx1 - fx0, GY - fb, fill=False, lw=2.2))
    _hatch(ax, fx0, fb, fx1, GY, spacing=0.18)
    _dim_v(ax, GY, pt, 12.5, f'{post_height_in:.0f}"  POST HT.')
    _dim_v(ax, fb, GY, 12.5, f'{footing_depth_in:.0f}"  FOOTING DEPTH')
    _dim_h(ax, fx0, fx1, fb - 0.8, f'{footing_dia_in:.0f}" DIA.')
    _leader(ax, 16.0, 14.0, pcx + pw / 2, pt - 0.5, f"POST: {post_section} STEEL")
    _leader(ax, 16.0, 12.5, fx1, (GY + fb) / 2, f"{concrete_class} CONCRETE FOOTING")
    _notes(ax, [
        "NOTES:", f"1. POST: {post_section} STEEL, ASTM A992.",
        f"2. FOOTING: {concrete_class} CONCRETE.",
        f'3. MIN FOOTING DEPTH PER {agency} = {min_footing_note_in:.0f}".',
        "4. COMPACT FILL TO 95% PROCTOR.",
    ])
    ax.text(pcx, pt + 0.8, "TYP. POST FOUNDATION DETAIL",
            ha="center", fontsize=12, family="monospace", weight="bold")
    _save(fig, out)


# ---------- 3. REBAR DETAIL ----------
def draw_rebar(out, *, agency, plan_id, beam_w_in, beam_h_in, bar_size, n_top, n_bot,
               stirrup_spacing_in, concrete_class, cover_in):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=120); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "REINFORCED CONCRETE BEAM - REBAR DETAIL")
    bcx = 9.5; bcy = 9.0
    w = beam_w_in / 12.0 * 3.0; h = beam_h_in / 12.0 * 3.0
    x0 = bcx - w / 2; y0 = bcy - h / 2
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=2.5))
    # Cover offset
    co = cover_in / 12.0 * 3.0
    # Top bars
    for i in range(n_top):
        bx = x0 + co + (i + 0.5) * (w - 2 * co) / n_top
        ax.plot(bx, y0 + h - co, "ko", markersize=6)
    # Bottom bars
    for i in range(n_bot):
        bx = x0 + co + (i + 0.5) * (w - 2 * co) / n_bot
        ax.plot(bx, y0 + co, "ko", markersize=6)
    # Stirrup outline
    s_co = co * 0.6
    ax.add_patch(Rectangle((x0 + s_co, y0 + s_co), w - 2 * s_co, h - 2 * s_co,
                           fill=False, lw=1.2, linestyle="--"))
    _dim_h(ax, x0, x0 + w, y0 - 0.7, f'{beam_w_in:.0f}" WIDTH')
    _dim_v(ax, y0, y0 + h, x0 - 0.7, f'{beam_h_in:.0f}" HEIGHT')
    _leader(ax, 16.0, 13.0, x0 + co + 0.05, y0 + h - co, f"{n_top} - {bar_size} BARS TOP")
    _leader(ax, 16.0, 11.5, x0 + co + 0.05, y0 + co, f"{n_bot} - {bar_size} BARS BOT")
    _leader(ax, 16.0, 10.0, x0 + s_co, y0 + h / 2,
            f'#3 STIRRUPS @ {stirrup_spacing_in:.0f}" OC')
    _leader(ax, 16.0, 8.5, x0 + co * 0.5, y0 + h - co * 0.5, f'COVER = {cover_in:.1f}"')
    _notes(ax, [
        "NOTES:", f"1. {concrete_class} CONCRETE.",
        f"2. MAIN BARS: {bar_size}, ASTM A615 GR 60.",
        f"3. STIRRUPS @ {stirrup_spacing_in:.0f}\" MAX SPACING.",
        f"4. CLEAR COVER = {cover_in:.1f}\" MIN.",
    ])
    ax.text(bcx, y0 + h + 0.8, "BEAM CROSS-SECTION",
            ha="center", fontsize=12, family="monospace", weight="bold")
    _save(fig, out)


# ---------- 4. DRAINAGE INLET ----------
def draw_inlet(out, *, agency, plan_id, throat_w_in, throat_h_in, grate_open_in,
               concrete_class, frame_grade, reinforcing_note):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=120); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "CURB DRAINAGE INLET - SECTION & PLAN")
    icx = 9.0; iy = 8.0
    w = throat_w_in / 12.0 * 1.8; h = throat_h_in / 12.0 * 4.0
    x0 = icx - w / 2; y0 = iy
    # Curb above
    ax.add_patch(Rectangle((x0 - 0.5, y0 + h), w + 1.0, 0.8, fill=False, lw=2.2))
    _hatch(ax, x0 - 0.5, y0 + h, x0 + w + 0.5, y0 + h + 0.8, spacing=0.13)
    # Throat opening
    ax.add_patch(Rectangle((x0, y0 + h - 0.7), w, 0.7, fill=False, lw=2.2,
                           edgecolor="black"))
    # Catch basin walls
    ax.add_patch(Rectangle((x0 - 0.4, y0), 0.4, h, fill=False, lw=2.2))
    ax.add_patch(Rectangle((x0 + w, y0), 0.4, h, fill=False, lw=2.2))
    _hatch(ax, x0 - 0.4, y0, x0, y0 + h, spacing=0.12)
    _hatch(ax, x0 + w, y0, x0 + w + 0.4, y0 + h, spacing=0.12)
    # Floor
    ax.add_patch(Rectangle((x0 - 0.4, y0 - 0.5), w + 0.8, 0.5, fill=False, lw=2.2))
    _hatch(ax, x0 - 0.4, y0 - 0.5, x0 + w + 0.4, y0, spacing=0.13)
    # Grate (top view bars)
    for i in range(5):
        gx = x0 + 0.15 + i * (w - 0.3) / 4
        ax.add_line(Line2D([gx, gx], [y0 + h, y0 + h + 0.8], color="black", lw=1.2))
    _dim_h(ax, x0, x0 + w, y0 + h - 1.2, f'{throat_w_in:.0f}" THROAT W')
    _dim_v(ax, y0 + h - 0.7, y0 + h, x0 + w + 0.8, f'{throat_h_in:.0f}" H')
    _leader(ax, 16.0, 13.0, x0 + w / 2, y0 + h + 0.4,
            f"CAST IRON GRATE, {frame_grade}")
    _leader(ax, 16.0, 11.5, x0 + 0.2, y0 + h / 2, f"{concrete_class} CONCRETE")
    _leader(ax, 16.0, 10.0, x0 + w / 2, y0 + h - 0.35, f'GRATE OPENING = {grate_open_in:.1f}"')
    _notes(ax, [
        "NOTES:", f"1. {concrete_class} CONCRETE.",
        f"2. GRATE/FRAME: {frame_grade}.",
        f"3. {reinforcing_note}",
        "4. PROVIDE 6\" GRAVEL BEDDING.",
    ])
    ax.text(icx, y0 + h + 1.6, "INLET SECTION VIEW",
            ha="center", fontsize=12, family="monospace", weight="bold")
    _save(fig, out)


# ---------- 5. SIGN POST FOUNDATION ----------
def draw_sign_post(out, *, agency, plan_id, post_dia_in, embed_depth_in, footing_dia_in,
                   footing_depth_in, anchor_bolt_dia_in, anchor_embed_in, concrete_class):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=120); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "BREAKAWAY SIGN POST - FOUNDATION DETAIL")
    GY = 10.5
    ax.add_line(Line2D([3.5, 15.5], [GY, GY], color="black", lw=1.8))
    for x in [i * 0.4 + 3.6 for i in range(30)]:
        ax.add_line(Line2D([x, x + 0.25], [GY, GY + 0.25], color="black", lw=0.6))
    pcx = 9.5
    # Pipe post above
    pw = post_dia_in / 12.0 * 0.5
    ax.add_patch(Rectangle((pcx - pw, GY), 2 * pw, 4.0, fill=False, lw=2.2))
    # Base plate
    ax.add_patch(Rectangle((pcx - pw - 0.4, GY), 2 * pw + 0.8, 0.15, fill=False, lw=1.5))
    # Footing
    fb = GY - (footing_depth_in / 12.0) * 1.3
    fhw = (footing_dia_in / 12.0) * 0.6
    fx0 = pcx - fhw; fx1 = pcx + fhw
    ax.add_patch(Rectangle((fx0, fb), fx1 - fx0, GY - fb, fill=False, lw=2.2))
    _hatch(ax, fx0, fb, fx1, GY, spacing=0.18)
    # Anchor bolts (4 visible, simplified to 2 in section)
    ae_u = (anchor_embed_in / 12.0) * 1.3
    for ax_x in [pcx - pw + 0.05, pcx + pw - 0.05]:
        ax.add_line(Line2D([ax_x, ax_x], [GY + 0.18, GY - ae_u], color="black", lw=1.5))
        ax.add_line(Line2D([ax_x - 0.08, ax_x + 0.08], [GY + 0.18, GY + 0.18], color="black", lw=1.5))
    _dim_v(ax, fb, GY, 13.5, f'{footing_depth_in:.0f}"  FTG DEPTH')
    _dim_h(ax, fx0, fx1, fb - 0.7, f'{footing_dia_in:.0f}" DIA.')
    _dim_v(ax, GY - ae_u, GY, fx1 + 1.4, f'{anchor_embed_in:.0f}"  ANCHOR EMBED')
    _leader(ax, 16.0, 14.0, pcx + pw, GY + 2.0, f"POST: {post_dia_in:.1f}\" PIPE")
    _leader(ax, 16.0, 12.5, fx1, (GY + fb) / 2, f"{concrete_class} CONCRETE FTG")
    _leader(ax, 16.0, 11.0, pcx + pw - 0.05, GY - ae_u / 2, f'{anchor_bolt_dia_in:.2f}" DIA BOLTS')
    _notes(ax, [
        "NOTES:", f"1. {concrete_class} CONCRETE.",
        f"2. ANCHOR BOLTS: {anchor_bolt_dia_in:.2f}\" DIA, ASTM F1554.",
        f"3. MIN EMBEDMENT = 12\" PER STD SPEC.",
        f"4. POST PIPE: ASTM A53 GR B.",
    ])
    ax.text(pcx, GY + 4.5, "SIGN POST FOUNDATION DETAIL",
            ha="center", fontsize=12, family="monospace", weight="bold")
    _save(fig, out)
