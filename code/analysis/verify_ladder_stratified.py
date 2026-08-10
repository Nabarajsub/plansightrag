"""Does the capability ladder survive the question/injected-check misalignment?

`compliance_smoke_n10.py` generates the compliance question with a VLM at run
time, so each config produced its own questions. On 25-40% of cases the generated
question asks about a different quantity than the injected violation. Rungs 2-5
score the judge against the INJECTED check, so on those cases the judge is being
marked down for answering the question it was actually asked.

This re-scores the ladder separately on the aligned and misaligned subsets. If
the A29 reading ("weak configs fail at value extraction, not arithmetic") only
holds on the misaligned subset, it is an artefact and must be withdrawn.

Reads only; changes no experiment and no published report.
"""

# --- release path resolution (added for the release copy; the run-time originals
# under baselines_v2/ and explanability/ are unchanged and still carry the
# absolute ARCC paths the experiments were executed with) ---
import os as _os
PSR_ROOT = _os.environ.get("PSR_ROOT") or _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
CLUSTER_ROOT = _os.environ.get("CLUSTER_ROOT", "/project/gr-wydot-chatbot/copalirag")
# --- end release path resolution ---

import json, os, sys
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from capability_ladder import (score_case, canonical_injected, load_cases,
                               CONFIGS, RUNGS, COMP, wilson)

KEY = {"Concrete cover": ("cover",), "Footing depth": ("footing", "depth", "embed"),
       "Grate opening": ("grate", "opening", "throat"), "Stirrup spacing": ("stirrup",),
       "Anchor bolt embedment": ("anchor", "embedment", "embed")}

manifest = {d["name"]: d for d in
            json.load(open(os.path.join(COMP, "table9_n100", "manifest.json")))}
inj = canonical_injected()

print("=" * 96)
print("CAPABILITY LADDER, STRATIFIED BY QUESTION/INJECTED-CHECK ALIGNMENT")
print("=" * 96)

for label, fname in CONFIGS.items():
    p = os.path.join(COMP, fname)
    if not os.path.exists(p):
        continue
    cases = load_cases(p)
    strata = {"aligned": [], "misaligned": []}
    for c in cases:
        q = (c.get("question") or c.get("query") or "").lower()
        k = KEY.get((inj.get(c.get("name")) or {}).get("name"), ())
        if not q or not k:
            continue
        strata["aligned" if any(x in q for x in k) else "misaligned"].append(c)
    if not strata["misaligned"]:
        print(f"\n  {label}: 100% aligned (uses checks_injected directly) — nothing to stratify")
        continue
    print(f"\n  {label}   aligned n={len(strata['aligned'])}  "
          f"misaligned n={len(strata['misaligned'])}")
    print(f"    {'rung':<16}{'aligned':>12}{'misaligned':>14}{'gap':>9}")
    for rung, _ in RUNGS:
        out = {}
        for s, cs in strata.items():
            k = n = 0
            for c in cs:
                sc = score_case(c, manifest.get(c.get("name")), inj)
                v = sc.get(rung)
                if v is None:
                    continue
                n += 1; k += int(v)
            out[s] = (k, n)
        (ka, na), (km, nm) = out["aligned"], out["misaligned"]
        pa = ka / na * 100 if na else None
        pm = km / nm * 100 if nm else None
        gap = f"{pa - pm:+.0f}" if (pa is not None and pm is not None) else "—"
        fa = f"{pa:.0f}% ({ka}/{na})" if pa is not None else "—"
        fm = f"{pm:.0f}% ({km}/{nm})" if pm is not None else "—"
        print(f"    {rung:<16}{fa:>12}{fm:>14}{gap:>9}")
print("\n" + "=" * 96)
