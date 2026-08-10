"""Judge-free numeric scoring for the Dimensional Accuracy category (Cluster C).

All three reviewers object that the benchmark is scored by an LLM judge that was
never validated against people. For the largest answer-bearing category the judge
is avoidable entirely: a dimensional answer is a physical quantity, and two
quantities can be compared arithmetically.

This parses the reference and the model answer into physical lengths, normalises
to millimetres, and scores a match on tolerance. It removes the judge from the
loop for 112 of 424 test items and gives us a second, mechanical estimate to set
against the LLM judge -- the disagreements are themselves the judge-error
analysis the reviewers asked for.

Handled: 4'-10", 21", 4' [1220], 1.5 m, 12 in, 300 mm, 2.5 ft, bare numbers with
a unit named in the question, and multi-quantity answers (any reference quantity
matching any predicted quantity counts, since '4 feet (1220 mm)' is one answer
stated twice).

Deliberately NOT handled, and reported as unscorable rather than guessed: bar
designations (#5), ratios (2:1), percentages, angles, and answers with no numeric
content ('Not specified'). Those still need the judge, and we say so.

    python numeric_match.py
"""

# --- release path resolution (added for the release copy; the run-time originals
# under baselines_v2/ and explanability/ are unchanged and still carry the
# absolute ARCC paths the experiments were executed with) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict

ROOT = CLUSTER_ROOT
VQA = os.path.join(ROOT, "vqa_eval")
REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

TOL_REL = 0.02      # 2% relative
TOL_ABS_MM = 1.5    # or 1.5 mm, whichever is larger

TO_MM = {"mm": 1.0, "millimeter": 1.0, "millimetre": 1.0,
         "cm": 10.0, "centimeter": 10.0,
         "m": 1000.0, "meter": 1000.0, "metre": 1000.0,
         "in": 25.4, "inch": 25.4, "inches": 25.4, '"': 25.4,
         "ft": 304.8, "foot": 304.8, "feet": 304.8, "'": 304.8}

# 4'-10"  /  4' - 10"  /  4'10"
FT_IN = re.compile(r"(\d+(?:\.\d+)?)\s*'\s*-?\s*(\d+(?:\.\d+)?)\s*\"")
# 12.5 mm / 4 feet / 21" / 3'
NUM_UNIT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|cm|millimet(?:er|re)s?|centimeters?|meters?|metres?|m|"
    r"in\b|inch(?:es)?|ft\b|feet|foot|\"|')", re.I)
BRACKET = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*\]")     # 4' [1220]  -> mm
UNSCORABLE = re.compile(r"#\s*\d|:\s*\d|\d\s*%|degree|°", re.I)


def quantities(text):
    """All lengths in `text`, in millimetres."""
    if not text:
        return []
    t = str(text)
    out = []
    for m in FT_IN.finditer(t):
        out.append(float(m.group(1)) * 304.8 + float(m.group(2)) * 25.4)
    # drop the feet-inches spans so their parts are not double counted
    t2 = FT_IN.sub(" ", t)
    for m in NUM_UNIT.finditer(t2):
        u = m.group(2).lower()
        if u in TO_MM:
            out.append(float(m.group(1)) * TO_MM[u])
    for m in BRACKET.finditer(t):
        out.append(float(m.group(1)))          # bracketed values are mm by convention
    return out


def matches(ref_q, pred_q):
    for a in ref_q:
        for b in pred_q:
            if abs(a - b) <= max(TOL_ABS_MM, TOL_REL * abs(a)):
                return True
    return False


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, (c - h) * 100), min(100.0, (c + h) * 100))


def main():
    judge = json.load(open(os.path.join(VQA, "vqa_results.json")))["coverage_matrix"]
    records = {}
    for f, model in (("answers_qwen7b.jsonl", "qwen7b"),
                     ("answers_qwen72b.jsonl", "qwen72b")):
        p = os.path.join(VQA, f)
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("category") != "Dimensional Accuracy":
                continue
            records.setdefault(r["id"], {"ref": r.get("reference"), "by": {}})
            records[r["id"]]["by"][model] = r.get("answers") or {}

    total = len(records)
    # split scorable vs not
    scorable, unscorable = {}, []
    for qid, rec in records.items():
        rq = quantities(rec["ref"])
        if rq and not UNSCORABLE.search(str(rec["ref"])):
            scorable[qid] = rq
        else:
            unscorable.append(qid)

    W = 84
    print("=" * W)
    print("JUDGE-FREE NUMERIC SCORING -- Dimensional Accuracy  (Cluster C, R1.3/R2.3)")
    print("=" * W)
    print(f"\n  Dimensional Accuracy items on the 424 test split : {total}")
    print(f"  parseable as a physical length                   : {len(scorable)} "
          f"({len(scorable)/total*100:.1f}%)")
    print(f"  not numerically scorable (kept with the judge)   : {len(unscorable)}")

    cells = defaultdict(lambda: {"num_k": 0, "num_n": 0,
                                 "agree": 0, "both": 0,
                                 "judge_yes_num_no": 0, "judge_no_num_yes": 0})
    for qid, rq in scorable.items():
        rec = records[qid]
        jrow = judge.get(qid, {})
        for model, answers in rec["by"].items():
            for tech, pred in answers.items():
                key = f"{model}::{tech}"
                num_ok = matches(rq, quantities(pred))
                c = cells[key]
                c["num_k"] += int(num_ok)
                c["num_n"] += 1
                if key in jrow:
                    j = bool(jrow[key])
                    c["both"] += 1
                    c["agree"] += int(j == num_ok)
                    c["judge_yes_num_no"] += int(j and not num_ok)
                    c["judge_no_num_yes"] += int(num_ok and not j)

    print(f"\n{'-' * W}")
    print(f"  {'model::technique':<24}{'numeric':>10}{'judge':>10}{'agree':>10}"
          f"{'J+/N-':>8}{'J-/N+':>8}")
    print("-" * W)
    rows = []
    for key, c in sorted(cells.items()):
        if not c["num_n"]:
            continue
        num = c["num_k"] / c["num_n"] * 100
        jk = c["agree"] + c["judge_yes_num_no"] + c["judge_no_num_yes"]
        jn = c["both"]
        j_acc = ((c["num_k"] - c["judge_no_num_yes"] + c["judge_yes_num_no"]) / jn * 100
                 if jn else float("nan"))
        agree = c["agree"] / jn * 100 if jn else float("nan")
        rows.append((key, num, j_acc, agree, c["judge_yes_num_no"], c["judge_no_num_yes"]))
        print(f"  {key:<24}{num:9.1f}%{j_acc:9.1f}%{agree:9.1f}%"
              f"{c['judge_yes_num_no']:8d}{c['judge_no_num_yes']:8d}")
    print("-" * W)
    print("  J+/N- = judge said correct, arithmetic says no  (candidate judge false accept)")
    print("  J-/N+ = judge said wrong,   arithmetic says yes (candidate judge false reject)")

    if rows:
        ag = [r[3] for r in rows if not math.isnan(r[3])]
        fa = sum(r[4] for r in rows)
        fr = sum(r[5] for r in rows)
        print(f"\n  mean judge/arithmetic agreement : {sum(ag)/len(ag):.1f}%")
        print(f"  total judge false accepts       : {fa}")
        print(f"  total judge false rejects       : {fr}")
        print(f"  net judge bias                  : "
              f"{'lenient' if fa > fr else 'strict'} by {abs(fa - fr)} decisions")

    out = {"note": "Judge-free numeric scoring for Dimensional Accuracy (Cluster C).",
           "tolerance": {"relative": TOL_REL, "absolute_mm": TOL_ABS_MM},
           "n_dimensional_items": total, "n_scorable": len(scorable),
           "n_unscorable": len(unscorable),
           "cells": {k: {"numeric_pct": round(v["num_k"] / v["num_n"] * 100, 2),
                         "numeric_k": v["num_k"], "numeric_n": v["num_n"],
                         "wilson95": [round(x, 2) for x in wilson(v["num_k"], v["num_n"])],
                         "agreement_pct": (round(v["agree"] / v["both"] * 100, 2)
                                           if v["both"] else None),
                         "judge_false_accept": v["judge_yes_num_no"],
                         "judge_false_reject": v["judge_no_num_yes"]}
                     for k, v in sorted(cells.items()) if v["num_n"]}}
    dst = os.path.join(REPORTS, "_numeric_match.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\n  wrote {dst}")


if __name__ == "__main__":
    main()
