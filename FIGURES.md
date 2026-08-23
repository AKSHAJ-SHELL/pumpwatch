# Figure index — 21 figures, all on real data

## Where the real data is

| Directory | What | Count |
|---|---|---|
| `figures/espset/` | **REAL — 11 in-service submersible pumps**, 5737 records. The cross-machine evidence and every headline result. | 12 |
| `figures/twente/` | **REAL — 2-motor laboratory rig**, 287 records. Current channel and severity grading. | 7 |
| `figures/summary/` | Cross-dataset comparisons spanning all three. | 2 |

**Every figure and every number here is measured on real machines.** The synthetic
stand-in used during development to check the pipeline is excluded from this package
entirely — it is not in the figures, not in the results, and not cited anywhere in the
draft.

Priority column: ⭐⭐⭐ = put in the paper; ⭐⭐ = strong, use if space; ⭐ = supporting,
the text already says it.

---

## `figures/summary/` — cross-dataset. **Both are top picks.**

| File | What it shows | Priority |
|---|---|---|
| `B6_pca_class_vs_machine.png` | The same PCA projection coloured twice: by fault class, then by machine identity. Machine structure is the stronger signal — which is *why* a random split lets a model read machine identity and report it as diagnosis. **The leakage argument in one picture.** | ⭐⭐⭐ |
| `D13_leakage_across_datasets.png` | Leakage inflation measured on all three datasets at once. Carries the whole protocol result. Note the ordering: the effect is *smallest* on synthetic data. | ⭐⭐⭐ |

## `figures/espset/` — REAL, 11 in-service pumps. The headline dataset.

| File | What it shows | Priority |
|---|---|---|
| `D1_leakage_ladder.png` | Macro-F1 against split protocol, levels 0→4. The leakage argument on real machines. | ⭐⭐⭐ |
| `D7_lomo_per_machine.png` | One bar per held-out machine. With 11 machines these *are* the data points, so they are plotted individually rather than hidden behind a mean. Shows the spread that makes machine count the binding constraint. | ⭐⭐⭐ |
| `D12_recall_at_alarm_budget.png` | Fault recall at ≤1 false alarm/pump/month, in the operator's units. The operational result — the gap is wider here than in macro-F1. | ⭐⭐⭐ |
| `D4_context_sweep.png` | Accuracy and latency against reference-set size. Saturates ~500 windows, *regresses* at 1000. This is the commissioning specification. | ⭐⭐ |
| `D11_normalization_gap.png` | Transductive vs inductive normalisation. On real data the gap exceeds the model effect. | ⭐⭐ |
| `D2_accuracy_vs_latency.png` | Does the expensive model earn its compute? | ⭐ |
| `D6_calibration_tabpfn_abstain.png` | Reliability diagram, pooled over LOMO folds — abstaining variant. | ⭐ |
| `D6_calibration_tabpfn_noabstain.png` | Same, non-abstaining. Compare the pair: abstention changes calibration, not just coverage. | ⭐ |
| `D6_calibration_lightgbm.png` | Same, gradient-boosted baseline. | ⭐ |
| `D6_calibration_logistic.png` | Same, logistic. | ⭐ |
| `C5_escalation_vs_battery.png` | Gate escalation rate → uplinks/day → battery years, driven by the **measured** escalation rate on these pumps. | ⭐⭐ |
| `E3_energy_breakdown.png` | Per-phase node energy at the measured escalation rate. **Transmission is ~1% of the budget; continuous sensing is the rest.** Good venue fit (Net-Zero theme). | ⭐⭐ |

## `figures/twente/` — REAL, 2-motor rig. Current channel and severity grading.

| File | What it shows | Priority |
|---|---|---|
| `D1_leakage_ladder.png` | Leakage ladder on the rig. Largest inflation of the three datasets (2.4×). | ⭐⭐ |
| `D14_detection_by_severity_tabpfn_abstain.png` | Detection rate against the dataset's own fault grading. Our honest substitute for a lead-time curve, since there is no run-to-failure data. ⚠️ Severity 4 and 5 have n=5 and n=6 — the caption must give the counts. | ⭐⭐ |
| `D14_detection_by_severity_tabpfn_noabstain.png` | Same, non-abstaining. | ⭐ |
| `D14_detection_by_severity_lightgbm.png` | Same, gradient-boosted. | ⭐ |
| `D14_detection_by_severity_logistic.png` | Same, logistic. | ⭐ |
| `C5_escalation_vs_battery.png` | Escalation → battery, Twente rate. | ⭐ |
| `E3_energy_breakdown.png` | Node energy, Twente escalation rate. | ⭐ |

## If forced to two figures

`summary/B6_pca_class_vs_machine.png` and `summary/D13_leakage_across_datasets.png`.
Between them they carry the protocol contribution, which is the part of this work most
likely to outlast the model.

## If four

Add `espset/D7_lomo_per_machine.png` (shows the spread, and the honesty about it) and
`espset/D12_recall_at_alarm_budget.png` (the operational result).
