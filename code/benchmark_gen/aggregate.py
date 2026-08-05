"""Aggregate verified QnA from Qwen + Llama drafters and report retrieval metrics."""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="verified_*.jsonl files")
    ap.add_argument("--out_jsonl", default="pilot500_final.jsonl")
    ap.add_argument("--out_report", default="pilot500_report.json")
    args = ap.parse_args()

    rows = []
    for pat in args.inputs:
        for p in glob.glob(pat):
            rows.extend(json.loads(l) for l in open(p))
    print(f"[agg] loaded {len(rows)} verified rows from {len(args.inputs)} pattern(s)")

    seen = {}
    for r in rows:
        key = (r["image_path"], r["category"], r["query"])
        if key not in seen:
            seen[key] = r
    rows = list(seen.values())
    print(f"[agg] {len(rows)} unique rows after dedup")

    n_hit = sum(int(r.get("final_hit", False)) for r in rows)
    by_cat = defaultdict(list)
    by_agency = defaultdict(list)
    by_drafter = defaultdict(list)
    by_attempts = Counter()
    for r in rows:
        h = int(r.get("final_hit", False))
        by_cat[r["category"]].append(h)
        by_agency[r["agency"]].append(h)
        by_drafter[r["drafter"]].append(h)
        by_attempts[r.get("n_attempts", 0)] += 1

    def pct(xs):
        return 100.0 * sum(xs) / max(len(xs), 1)

    report = {
        "total": len(rows),
        "recall_at_5_overall_pct": pct([int(r.get("final_hit", False)) for r in rows]),
        "by_category": {k: {"n": len(v), "recall@5": pct(v)} for k, v in by_cat.items()},
        "by_agency": {k: {"n": len(v), "recall@5": pct(v)} for k, v in by_agency.items()},
        "by_drafter": {k: {"n": len(v), "recall@5": pct(v)} for k, v in by_drafter.items()},
        "attempts_distribution": dict(by_attempts),
    }

    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== RETRIEVAL REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
