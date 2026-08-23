# Handoff package — pump condition monitoring, INDICON 2026

Everything needed to write the paper. Target venue is **IEEE INDICON 2026**, deadline
**31 August 2026**, Microsoft CMT, **double-blind**, **hard limit 6 pages including
figures and references**.

## Read in this order

1. **[PAPER_DRAFT.md](PAPER_DRAFT.md)** — the full rough draft. Abstract, deviations
   from the original proposal, and §1–§7 in prose. This is deliberately *longer* than
   six pages; it is the superset to cut from, not the submission.
2. **[FIGURES.md](FIGURES.md)** — all 38 figures with captions, what claim each
   supports, and a priority rating.

   **Real data:** `figures/espset/` (11 in-service pumps — every headline result),
   `figures/twente/` (2-motor rig), `figures/summary/` (cross-dataset).
   **Synthetic:** `figures/synthetic/`, quarantined and labelled. A wiring check, not
   a measurement. Every model-performance number in the paper is from real data.
3. **[results/paper_tables.md](results/paper_tables.md)** — every results table,
   generated from the JSONs. Do not hand-copy numbers; regenerate with `make tables`.
4. **[paper/main.tex](paper/main.tex)** — a 6-page IEEE-format cut, already
   anonymised for double-blind. A starting point if it is useful; ignore it if you
   would rather cut from the markdown yourself.

## What has been measured, in one page

| Claim | Number | Where |
|---|---|---|
| Cross-machine, 11 in-service pumps, LOMO | macro-F1 **0.738 ± 0.015** vs tuned GBDT **0.666** | `results_espset_both.json` |
| Same, abstaining variant | **0.753** at 0.81 coverage | " |
| Tuning does not rescue baselines | logistic 0.663→0.638, GBDT 0.666→0.664 | " |
| Recall at ≤1 false alarm/pump/month | **0.203** vs GBDT 0.084, logistic 0.032 | " |
| Commissioning specification | saturates ~**500** windows, regresses at 1000 | " |
| Leakage inflation, ESPset | 0.793 → 0.425 = **1.9×** | " |
| Leakage inflation, Twente rig | 0.853 → 0.352 = **2.4×** | `results_twente_real.json` |
| Leakage inflation, synthetic | 1.000 → 0.930 = 1.1× (smallest!) | `results_full.json` |
| Gateway latency, **on the RK3588 board** | **88 ms**/window; KV cache 7.1×, ensemble 5.6× | `hardware_bench_orangepi.json` |
| Node energy | transmission ≈ **1%** of budget; sensing is the rest | `results_espset_both.json` |
| Gate ceiling, worst pump | **0.52** — bounds the whole system | `gate_feature_ablation.json` |

Every one of these was re-verified against the result files: 23 of 23 numbers in the
draft match to three decimals.

## The three things most likely to be got wrong

1. **Synthetic vs real.** `figures/` (top level) and `results_full.json` are a
   synthetic stand-in. They exist to check the pipeline recovers signatures that were
   planted. Citing them as a result would be the worst error available here, and the
   results file carries an explicit caveat field saying so.
2. **Normalisation strategy.** Two are reported — transductive (per-machine) and
   inductive (train-pooled) — and on real data the gap reaches 0.25 macro-F1, larger
   than the model effect. **Every table must say which it uses.** The leakage ladder
   uses per-machine throughout; the cross-machine table uses train-pooled.
3. **Abstaining vs non-abstaining are different models.** They have different
   coverage (0.81 vs 1.00). Reporting an abstaining accuracy against a full-coverage
   baseline is not a comparison. They are named separately everywhere for this reason.

## Data

Neither dataset is redistributed — both are public and CC BY 4.0:

- **ESPset**, 11 in-service submersible pumps: DOI `10.17632/m268jsw339.3`
  (**version 3**, not the `.1` some prior work cites)
- **Twente/4TU**, 2-motor rig: DOI `10.4121/2b61183e-c14f-4131-829b-cc4822c369d0`

`results/*.json` ships in the repo, so **`make tables` and `make figures-all`
regenerate every number and figure without downloading either dataset.**

## Reproducing anything

```bash
pip install -e ".[tabpfn]"
make tables            # every results table
make figures-all       # all 38 figures
make test              # 232 tests
```

Re-running the experiments themselves needs the datasets downloaded; each loader
prints instructions rather than substituting synthetic data.

## Open items, honestly

- **§2 related work** is drafted with citations verified against the published
  record, but **Vieira et al. 2025 (arXiv:2509.22267) needs reading in full** — it is
  the closest work to the protocol contribution and §2 differentiates it from an
  extract.
- Three citation details flagged in the verification table at the end of §2.
- Reference list has 8 entries; INDICON will expect closer to 20.
- **Repo must stay private until acceptance** — double-blind. See
  [RELEASE.md](RELEASE.md) for the DOI routes.
- No rig data was collected. The acquisition path is implemented and exercisable
  against a simulated backend, but contribution C3 has no data and is not claimed.
