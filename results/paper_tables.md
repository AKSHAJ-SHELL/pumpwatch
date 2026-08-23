# Paper tables (generated — do not hand-edit)

## ESPset — 11 in-service submersible pumps

`n_samples=5737, n_features=22, n_machines=11`

**Leakage ladder** — macro-F1, 95% CI over held-out groups in brackets.

> Ladder entries are a **single run**. The normalisation table below is averaged over seeds, so its level-4 figures differ from this table's by roughly the seed spread (~0.005). Same folds, same data - do not read the difference as a discrepancy, and do not mix the two in one table.

| Leakage level | majority | logistic | lightgbm | tabpfn_abstain | tabpfn_noabstain |
|---|---|---|---|---|---|
| 0 random-window (INVALID) — *INVALID* | 0.228 [0.23–0.23] | 0.626 [0.63–0.63] | 0.793 [0.79–0.79] | 0.756 [0.76–0.76] (cov 0.90) | 0.751 [0.75–0.75] |
| 4 leave-one-machine-out — *thesis_test* | 0.228 [0.29–0.40] | 0.463 [0.37–0.52] | 0.425 [0.39–0.52] | 0.469 [0.40–0.49] (cov 0.87) | 0.468 [0.40–0.49] |

**Leakage inflation**

| Model | random-window | strictest valid | inflation |
|---|---|---|---|
| majority | 0.228 | 0.228 | 1.0x |
| logistic | 0.626 | 0.463 | 1.4x |
| lightgbm | 0.793 | 0.425 | 1.9x |
| tabpfn_abstain | 0.756 | 0.469 | 1.6x |
| tabpfn_noabstain | 0.751 | 0.468 | 1.6x |

**Normalisation strategy** (cross-machine)

| Model | unsupervised_per_machine | train_pooled | delta |
|---|---|---|---|
| majority | 0.228 | 0.228 | +0.000 |
| logistic | 0.463 | 0.663 | +0.200 |
| lightgbm | 0.421 | 0.666 | +0.245 |
| tabpfn_abstain | 0.475 (cov 0.87) | 0.753 (cov 0.81) | +0.278 |
| tabpfn_noabstain | 0.473 | 0.738 | +0.265 |

> `unsupervised_per_machine` normalises each machine using its own statistics, including the held-out one - transductive, and legitimate when a node self-commissions on the target pump. `train_pooled` uses training-machine statistics only - inductive, and the stricter reading. **Never quote one for the other.** The leakage-ladder tables above use `unsupervised_per_machine` throughout.

**Stage-1 gate**

| Pump | healthy escalation | faulty escalation | field rate | commissioned |
|---|---|---|---|---|
| esp_00 | 0.05 | 0.78 | 0.057 | NO |
| esp_01 | 0.06 | 0.89 | 0.064 | yes |
| esp_02 | 0.11 | 1.00 | 0.117 | yes |
| esp_03 | 0.05 | 0.98 | 0.061 | yes |
| esp_04 | 0.03 | 0.48 | 0.036 | yes |
| esp_05 | 0.10 | 0.99 | 0.113 | yes |
| esp_06 | 0.18 | 0.98 | 0.192 | yes |
| esp_07 | 0.06 | 1.00 | 0.068 | yes |
| esp_08 | 0.02 | 1.00 | 0.029 | yes |
| esp_09 | 0.08 | 0.53 | 0.082 | yes |
| esp_10 | 0.08 | 0.48 | 0.082 | yes |

Field-weighted escalation **0.082**, gate recall ceiling **0.83** (pooled by fault count 0.79, **worst machine 0.48**), 3.0 uplinks/day, 1.18 yr battery, TX 1.0% of the budget. Adequately commissioned on 10/11 machines.

> Gateway accuracy is an upper bound conditioned on escalation: end-to-end fault recall cannot exceed the gate recall ceiling. The headline ceiling is the **unweighted** mean over machines - one pump, one vote. The pooled figure weights by fault count and is lower whenever a few machines contributed most of the faults. **The worst machine is the number that bounds a deployment guarantee**: an average hides pumps on which the gate discards most faults before the classifier ever sees them. Battery life is driven by the *field* rate, which healthy false-escalation dominates — the test-set rate reflects how many faulty examples were collected, not field prevalence.

## Twente/4TU — 2 motors, 4 operating speeds

`n_records=287, n_features=42`

**Leakage ladder** — macro-F1, 95% CI over held-out groups in brackets.

> Ladder entries are a **single run**. The normalisation table below is averaged over seeds, so its level-4 figures differ from this table's by roughly the seed spread (~0.005). Same folds, same data - do not read the difference as a discrepancy, and do not mix the two in one table.

| Leakage level | majority | logistic | lightgbm | tabpfn_abstain | tabpfn_noabstain |
|---|---|---|---|---|---|
| 0 random-window (INVALID) — *INVALID* | 0.065 [0.06–0.06] | 0.542 [0.54–0.54] | 0.853 [0.85–0.85] | 0.967 [0.97–0.97] | 0.967 [0.97–0.97] |
| 1 record-wise — *weak* | 0.059 [0.07–0.20] | 0.308 [0.16–0.32] | 0.352 [0.23–0.37] | 0.446 [0.32–0.47] (cov 0.84) | 0.411 [0.28–0.47] |
| 2 component-wise — *good* | 0.031 [0.00–0.13] | 0.241 [0.07–0.23] | 0.349 [0.15–0.35] | 0.399 [0.19–0.41] (cov 0.89) | 0.399 [0.19–0.40] |
| 3 cross-operating — *essential* | 0.059 [0.10–0.10] | 0.153 [0.02–0.33] | 0.222 [0.08–0.27] | 0.220 [0.06–0.33] (cov 0.52) | 0.282 [0.08–0.45] |

**Leakage inflation**

| Model | random-window | strictest valid | inflation |
|---|---|---|---|
| majority | 0.065 | 0.059 | 1.1x |
| logistic | 0.542 | 0.153 | 3.5x |
| lightgbm | 0.853 | 0.222 | 3.8x |
| tabpfn_abstain | 0.967 | 0.220 | 4.4x |
| tabpfn_noabstain | 0.967 | 0.282 | 3.4x |

**Normalisation strategy** (cross-machine)

| Model | unsupervised_per_machine | train_pooled | delta |
|---|---|---|---|

> `unsupervised_per_machine` normalises each machine using its own statistics, including the held-out one - transductive, and legitimate when a node self-commissions on the target pump. `train_pooled` uses training-machine statistics only - inductive, and the stricter reading. **Never quote one for the other.** The leakage-ladder tables above use `unsupervised_per_machine` throughout.

**Stage-1 gate**

_No gate stage in this results file._

