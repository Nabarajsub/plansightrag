#!/usr/bin/env python3
"""Regenerate docs/supplementary.md from the canonical SUPPLEMENTARY.md.

SUPPLEMENTARY.md at the repository root is the single source of truth and is
readable without GitHub Pages enabled. The site page is derived from it so the
two cannot drift. Run after editing SUPPLEMENTARY.md:

    python3 code/make_docs_supplementary.py
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "SUPPLEMENTARY.md"
DST = ROOT / "docs" / "supplementary.md"
REPO = "https://github.com/Nabarajsub/plansightrag/blob/main"

body = SRC.read_text().split("\n")
assert body[0].startswith("# Supplementary"), "unexpected first line in SUPPLEMENTARY.md"
body = "\n".join(body[1:]).lstrip("\n")
body = body.replace("[`TABLE_PROVENANCE.md`](TABLE_PROVENANCE.md)",
                    f"[`TABLE_PROVENANCE.md`]({REPO}/TABLE_PROVENANCE.md)")

# the site has navigation, so the in-page contents list is redundant
i, j = body.find("## Contents"), body.find("## Table S1.")
if i != -1 and j != -1:
    body = body[:i] + body[j:]

DST.write_text(
    "---\nlayout: default\ntitle: Supplementary\npermalink: /supplementary/\n---\n\n"
    "## Supplementary information\n\n" + body)
print(f"wrote {DST.relative_to(ROOT)} ({len(body.splitlines())} lines)")
