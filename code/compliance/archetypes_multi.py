"""Multi-plan compliance archetypes: drawings with N components depicted, each
governed by a separate standard plan. NO plan IDs cited on the drawing — agent
must infer applicable standards from visual content.
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
    ax.text(0.7, 0.95, "STANDARD PLAN - DESIGN UNDER REVIEW", fontsize=8, family="monospace")
    ax.text(0.7, 0.65, sheet_title, fontsize=11, family="monospace", weight="bold")
    ax.text(15.7, 1.4, "DESIGN NO.", fontsize=8, family="monospace")
    ax.text(15.7, 0.95, plan_id, fontsize=14, family="monospace", weight="bold")
    ax.text(15.7, 0.65, "SHEET 1 OF 1", fontsize=8, family="monospace")


def _save(fig, out):
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _hatch(ax, x0, y0, x1, y1, spacing=0.13):
    n = int(((x1 - x0) + (y1 - y0)) / spacing) + 1
    for i in range(-n, n + 1):
        xs = x0 + i * spacing
        xe = xs + (y1 - y0)
        a = max(xs, x0); b = min(xe, x1)
        if a < b:
            ax.add_line(Line2D([a, b], [y0 + (a - xs), y0 + (b - xs)], color="black", lw=0.4))


def _leader(ax, xt, yt, xp, yp, text):
    ax.add_line(Line2D([xt + 0.05, xp], [yt, yp], color="black", lw=0.8))
    ax.plot(xp, yp, "ko", markersize=3)
    ax.text(xt, yt + 0.05, text, fontsize=9, family="monospace")


def _component_callout(ax, x, y, text, value_with_unit):
    """One labeled component on the drawing. No plan ID — agent must infer."""
    ax.text(x, y, text, fontsize=9, family="monospace", weight="bold")
    ax.text(x, y - 0.32, value_with_unit, fontsize=10, family="monospace")


# ---------- N=2: Beam section + rebar ----------
def draw_n2_beam(out, *, agency, plan_id, components):
    """Reinforced concrete beam with N=2 components: cover + rebar grade."""
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=110); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "REINFORCED BEAM - DESIGN DETAIL")
    # Beam cross-section
    cx, cy = 8, 9; w, h = 4, 6
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=2.5))
    co = 0.5
    for i in range(3):
        bx = x0 + co + (i + 0.5) * (w - 2 * co) / 3
        ax.plot(bx, y0 + h - co, "ko", markersize=6)
    for i in range(4):
        bx = x0 + co + (i + 0.5) * (w - 2 * co) / 4
        ax.plot(bx, y0 + co, "ko", markersize=6)
    ax.text(cx, y0 + h + 0.7, "SECTION", ha="center", fontsize=12, family="monospace", weight="bold")
    # Component callouts (no plan IDs)
    cy0 = 14
    for i, c in enumerate(components):
        _component_callout(ax, 14, cy0 - i * 1.5, c["label"].upper(), c["value_display"])
        # Leader line to the relevant region
        target = (x0 + co * 0.5, y0 + h - co * 0.5) if "cover" in c["key"].lower() else (cx, y0 + co)
        ax.add_line(Line2D([14, target[0]], [cy0 - i * 1.5 - 0.15, target[1]], color="black", lw=0.7))
    # Generic notes (no plan ID references)
    notes = ["NOTES:", "1. ALL VALUES PER MANUFACTURER SPEC.", "2. SHOP DRAWINGS REQUIRED.",
             "3. INSPECT REBAR PLACEMENT BEFORE POUR."]
    for i, n in enumerate(notes):
        ax.text(2, 5 - i * 0.4, n, fontsize=8.5, family="monospace",
                weight="bold" if i == 0 else "normal")
    _save(fig, out)


# ---------- N=3: Bridge bearing pad detail ----------
def draw_n3_bearing(out, *, agency, plan_id, components):
    """Bridge bearing pad with N=3 components: pad thickness + concrete + anchor."""
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=110); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "BRIDGE BEARING PAD - DESIGN DETAIL")
    # Bearing pad assembly schematic
    cx, cy = 7, 9
    # Concrete cap (top)
    ax.add_patch(Rectangle((cx - 3, cy + 1), 6, 1.2, fill=False, lw=2.0))
    _hatch(ax, cx - 3, cy + 1, cx + 3, cy + 2.2)
    # Bearing pad
    ax.add_patch(Rectangle((cx - 2, cy + 0.2), 4, 0.8, facecolor="lightgrey", lw=2.0, edgecolor="black"))
    # Steel girder/seat below
    ax.add_patch(Rectangle((cx - 3, cy - 1), 6, 1.2, fill=False, lw=2.0))
    _hatch(ax, cx - 3, cy - 1, cx + 3, cy + 0.2)
    # Anchor bolts
    for ax_x in [cx - 1.5, cx + 1.5]:
        ax.add_line(Line2D([ax_x, ax_x], [cy + 2.2, cy - 1.5], color="black", lw=1.5))
        ax.add_line(Line2D([ax_x - 0.1, ax_x + 0.1], [cy + 2.3, cy + 2.3], color="black", lw=1.5))
    ax.text(cx, cy + 3.0, "BEARING PAD DETAIL", ha="center", fontsize=12, family="monospace", weight="bold")
    # Component callouts
    cy0 = 15
    for i, c in enumerate(components):
        _component_callout(ax, 14, cy0 - i * 1.4, c["label"].upper(), c["value_display"])
    notes = ["NOTES:", "1. INSTALL PER MANUFACTURER.", "2. CHECK LEVELING WITH SHIMS.",
             "3. TEST TORQUE AFTER 24 HRS."]
    for i, n in enumerate(notes):
        ax.text(2, 4.5 - i * 0.4, n, fontsize=8.5, family="monospace",
                weight="bold" if i == 0 else "normal")
    _save(fig, out)


# ---------- N=4: Approach slab section ----------
def draw_n4_approach(out, *, agency, plan_id, components):
    """Bridge approach slab with N=4 components: pavement, rebar, joint, cover."""
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=110); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "BRIDGE APPROACH SLAB - DESIGN DETAIL")
    # Approach slab side view
    GY = 9
    # Slab
    ax.add_patch(Rectangle((4, GY), 10, 1.4, fill=False, lw=2.2))
    _hatch(ax, 4, GY, 14, GY + 1.4)
    # Pavement above
    ax.add_patch(Rectangle((4, GY + 1.4), 10, 0.8, fill=False, lw=1.8))
    # Joint
    ax.add_line(Line2D([9, 9], [GY, GY + 2.2], color="black", lw=2.5))
    ax.text(9, GY + 2.4, "EXP. JOINT", ha="center", fontsize=8, family="monospace")
    # Rebar
    for i in range(5):
        bx = 4.5 + i * 2
        ax.plot(bx, GY + 0.3, "ko", markersize=4)
        ax.plot(bx, GY + 1.1, "ko", markersize=4)
    ax.text(9, GY + 3.5, "APPROACH SLAB SECTION", ha="center", fontsize=12, family="monospace", weight="bold")
    # Component callouts
    cy0 = 15
    for i, c in enumerate(components):
        _component_callout(ax, 15, cy0 - i * 1.2, c["label"].upper(), c["value_display"])
    notes = ["NOTES:", "1. POUR PAVEMENT AFTER SLAB CURED 7 DAYS.",
             "2. SEAL ALL JOINTS PER STD SPEC.",
             "3. PROVIDE DRAINAGE OUTLETS @ JOINTS."]
    for i, n in enumerate(notes):
        ax.text(2, 5 - i * 0.4, n, fontsize=8.5, family="monospace",
                weight="bold" if i == 0 else "normal")
    _save(fig, out)


# ---------- N=5: Bridge pier complete detail ----------
def draw_n5_pier(out, *, agency, plan_id, components):
    """Bridge pier with N=5 components: footing depth, column rebar spacing,
    bearing pad thickness, anchor embed, concrete cover."""
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=110); fig.patch.set_facecolor("white")
    _frame(ax, agency, plan_id, "BRIDGE PIER - DESIGN DETAIL")
    # Pier elevation
    GY = 3.5; pcx = 7
    # Footing below grade
    ax.add_patch(Rectangle((pcx - 2.5, GY - 2), 5, 1.5, fill=False, lw=2.2))
    _hatch(ax, pcx - 2.5, GY - 2, pcx + 2.5, GY - 0.5)
    # Column
    ax.add_patch(Rectangle((pcx - 0.8, GY - 0.5), 1.6, 7, fill=False, lw=2.5))
    _hatch(ax, pcx - 0.8, GY - 0.5, pcx + 0.8, GY + 6.5, spacing=0.16)
    # Stirrups (visual horizontals on column)
    for y in [GY + 0.5, GY + 1.5, GY + 2.5, GY + 3.5, GY + 4.5, GY + 5.5]:
        ax.add_line(Line2D([pcx - 0.7, pcx + 0.7], [y, y], color="black", lw=0.8, linestyle="--"))
    # Cap on top
    ax.add_patch(Rectangle((pcx - 2.5, GY + 6.5), 5, 0.8, fill=False, lw=2.2))
    _hatch(ax, pcx - 2.5, GY + 6.5, pcx + 2.5, GY + 7.3)
    # Bearing pad
    ax.add_patch(Rectangle((pcx - 1, GY + 7.3), 2, 0.4, facecolor="lightgrey", lw=1.8, edgecolor="black"))
    # Anchor (visible bolt)
    ax.add_line(Line2D([pcx, pcx], [GY + 7.7, GY + 6.7], color="black", lw=1.5))
    # Ground line
    ax.add_line(Line2D([3, 11], [GY, GY], color="black", lw=1.5))
    for x in [i * 0.4 + 3.1 for i in range(20)]:
        ax.add_line(Line2D([x, x + 0.2], [GY, GY + 0.2], color="black", lw=0.6))
    ax.text(pcx, GY + 8.5, "PIER ELEVATION", ha="center", fontsize=12, family="monospace", weight="bold")
    # Component callouts (right side)
    cy0 = 15
    for i, c in enumerate(components):
        _component_callout(ax, 13.5, cy0 - i * 1.0, c["label"].upper(), c["value_display"])
    notes = ["NOTES:", "1. ALL POURS TO BE INSPECTED.", "2. FOOTING ON COMPETENT SOIL OR PILE."]
    for i, n in enumerate(notes):
        ax.text(2, 11 - i * 0.4, n, fontsize=8.5, family="monospace",
                weight="bold" if i == 0 else "normal")
    _save(fig, out)
