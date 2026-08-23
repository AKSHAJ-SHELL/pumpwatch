# INDICON 2026 submission

**Deadline: 31 August 2026.** Microsoft CMT. Conference ID #72446.

## The two constraints that shape everything

**Double-blind.** No names, no affiliations, no institutional identifiers, and
self-references in the third person. [main.tex](main.tex) is already anonymised —
the author block says so explicitly, and the reproducibility paragraph promises an
anonymised artifact rather than linking a named repository.

⚠️ **This is why the GitHub repo must stay private until acceptance.** A public
`AKSHAJ-SHELL/pumpwatch` is discoverable and carries your name in `CITATION.cff`.
Do not mint an open Zenodo DOI either — see [RELEASE.md](../RELEASE.md), Route B or C.

**Four to six pages, hard cap, including figures and references.** Over six pages
is an automatic reject, and no paid extra pages exist.

## Current state

`main.tex` is ~3400 words of body text with 2 tables and 8 references. In IEEE
two-column A4 10pt that lands around **5 pages**, leaving roughly one page for
figures and the additional references the reference list needs.

## Space budget

| Section | Target | Notes |
|---|---|---|
| Abstract + intro | 0.9 pg | |
| Related work | 0.6 pg | Do **not** cut the Vieira differentiation — it is the closest work |
| System design | 1.1 pg | + Fig. 1 (architecture) |
| Datasets + protocol | 0.9 pg | + Table I |
| Results | 1.6 pg | + Tables II, III + Fig. 2 |
| Limitations + conclusion | 0.5 pg | |
| References | 0.4 pg | ~16–20 entries |

## Figures — pick 2, maybe 3. You have 38.

Recommended, in priority order:

1. **B6** (`figures/summary/B6_pca_class_vs_machine.png`) — the same PCA coloured by
   fault class, then by machine identity. This is the leakage argument in one
   picture and it is the most persuasive thing in the whole set.
2. **D13** (`figures/summary/D13_leakage_across_datasets.png`) — leakage inflation
   across all three datasets. Carries §V-B on its own.
3. **A two-tier architecture diagram** — does not exist yet; you would draw it. Worth
   it only if space survives, since reviewers of a systems paper expect one.

Everything else (calibration, per-machine LOMO, energy breakdown, severity) is
supporting material that the text already summarises. At six pages they cost more
than they return.

## Before you submit

- [ ] Read Vieira et al. 2025 (arXiv:2509.22267) **in full** — §II differentiates it
      from an extract, not a complete read
- [ ] Verify the three flagged citations (Magadán end page; Varejão author order;
      confirm 0.887→0.733 is the same model under both sampling strategies)
- [ ] Add 8–12 references — pump fault physics, MCSA, CUSUM, LoRa energy budgets,
      selective prediction. Eight is thin for INDICON.
- [ ] Compile on Overleaf with the IEEE Conference template and **check the page
      count** before anything else
- [ ] Run IEEE PDF eXpress
- [ ] Confirm no author-identifying metadata survives in the PDF properties
- [ ] Plagiarism screening is mandatory — self-plagiarism from your synopsis counts

## Venue fit

The theme is "Net-Zero Cyber-Physical Intelligence: AI, 6G & Sustainable
Electronics". The energy result is the natural hook and it is genuine rather than
retrofitted: a battery-powered sensing tier where transmission turns out to be 1% of
the budget and continuous sensing is the rest, with a measured multi-year battery
life. Consider working "energy-aware" or "sustainable" into the title if the fit
needs to be more obvious to a programme committee.
