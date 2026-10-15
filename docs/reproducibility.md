---
layout: default
title: Reproducibility
permalink: /reproducibility/
---

## Reproducibility

Every number in every table of the manuscript is produced by a script in `code/`
that writes a JSON report into `reports/`. The mapping from each manuscript table to
its backing report and generating script is in
[`TABLE_PROVENANCE.md`](https://github.com/Nabarajsub/plansightrag/blob/main/TABLE_PROVENANCE.md).

### Source documents

The plan corpus is not redistributed. Every source document is a public record
published by its issuing agency, and
[`data/sources.json`](https://github.com/Nabarajsub/plansightrag/blob/main/data/sources.json)
records the publisher, the page count and a SHA-256 for each one. Verify an edition by
hash rather than by URL, since agencies revise their standard plans over time.

| Agency | Collection | Pages |
|---|---|---|
| Wyoming DOT | Standard Plans | 237 |
| Caltrans | 2025 Standard Plans | 638 |
| Arizona DOT | 2025 Standard Drawings | 181 |
| Arkansas (ARDOT) | Standard bridge drawings | 62 |
| Florida DOT | 2026 Design Standards | 780 |
| Michigan DOT | Standard Plans (held-out transfer set) | 298 |

> **Provenance note.** The 62-page Arkansas set was recorded as Colorado DOT in the
> initial release. Its sheets carry Arkansas State Highway Commission title blocks and
> Arkansas PE seals, so the agency label has been corrected. The SHA-256 is unchanged,
> the reported results are unaffected, and the local collection key `colorado_2025` is
> retained so that the frozen reports still match the run that produced them.

### Environment

Pinned in
[`requirements.txt`](https://github.com/Nabarajsub/plansightrag/blob/main/requirements.txt).
Results were obtained with 4-bit quantized weights on a single 80 GB GPU; quantization,
hardware and checkpoint licence terms all affect exact reproduction, so model revisions
and decoding settings are pinned alongside the code.
