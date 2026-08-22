# PAPER.md — draft structure, evidence map, and what is left

**Status: skeleton with evidence attached.** Every number below is cited from a
committed results file; every figure named exists on disk. Sections marked
🔴 have no evidence yet and must either be produced or cut.

Companion documents: [DESIGN.md](DESIGN.md) (system design + 20 findings),
[remediation.md](remediation.md) (code-health design, executed).

---

## 0. The honest headline

> **A two-tier irrigation-pump monitoring architecture in which a new pump is
> commissioned by swapping an in-context reference set rather than retraining,
> validated leave-one-machine-out on 11 in-service pumps — together with an
> evaluation-protocol result showing that the standard random split inflates
> reported accuracy by 1.9–5.8× on the same data.**

Two things to keep out of the abstract, both of which the design already warns
about (DESIGN §0.7, §12):

- ❌ "training-free" — TabPFN removes *gradient training*, not the need for
  labelled examples of every class in the context set
- ❌ any claim the NPU accelerates the classifier — it cannot (variable shapes)

⭐ **The strongest single sentence available is not the F1.** It is that at a
realistic alarm budget the method catches **2.4× as many faults** as a tuned GBDT
(20.3% vs 8.4% at ≤1 false alarm/pump/month). Calibration matters most where the
operating threshold is extreme, and a farmer's alarm budget is extreme.

---

## 1. Contributions, with evidence status

| | Claim | Evidence | Status |
|---|---|---|---|
| **C1** | Two-tier gate/classify with a quantified event-triggered power budget | gate escalation 2.0% field-weighted, 0.73 uplinks/day, recall ceiling 0.996 | ✅ |
| **C2** | Cross-machine adaptation by reference-set substitution, no retraining | ESPset LOMO, 11 machines, TabPFN 0.738 vs LightGBM 0.666 | ✅ |
| **C3** | Labelled dry-run characterisation set | none — blocked on rig hardware | 🔴 **cut or defer** |
| **C4** | Compute-vs-benefit accounting vs tuned baselines | tuning does not rescue baselines; 2.4× at alarm budget | ✅ |
| **C5** | *(new, unplanned)* Evaluation-protocol result: leakage inflation on real data | 1.9× ESPset, 5.8× Twente, 1.1× synthetic | ✅ |

**C5 was not in the original plan and may be the most citable contribution.** It is
methodological, dataset-independent, and demonstrated rather than argued.

**C3 must be cut from this paper.** Without a rig there is no dry-run data, and the
architecture section can still carry the trip path as a *design* with simulated
validation, clearly labelled.

---

## 2. Proposed structure

### Abstract
Lead with C2 + C5. Quote the alarm-budget number, not the F1.

### 1. Introduction
Smallholder irrigation pumps; dry running destroys a mechanical seal in <60 s;
commissioning cost is the barrier to per-pump ML. Contributions C1, C2, C4, C5.

### 2. Related work
TabPFN v1/v2 (Hollmann 2023/2025), the rotating-machinery TabPFN precedent
(IEEE Sensors J. 23(24), 2023 — **v1, not pumps, not embedded**), leakage critiques
in bearing diagnosis (Vieira 2026), Demšar 2006 / Dietterich 1998 for the stats.
⚠️ No PHM Society challenge has used a pump — do not cite one.

### 3. System design — C1
Two tiers; dry-run trip terminates at the node (sub-minute, contactor);
gateway classifies slow faults. Figures **C2**, **C2b**, **C7**, **C5**, **E3**, **E4**.

⭐ **Report the inversion:** with a working gate, LoRa TX is **0.7%** of the node
budget and continuous CUSUM sensing is the rest. The v1.0 assumption that TX
dominates at ~55% was computed for a fixed schedule the same document rejected.
The optimisation target is a cheaper always-on front end, not a smaller payload.

### 4. Datasets and protocol — C5
Two real datasets, deliberately complementary (DESIGN §−2.3):

| | Twente/4TU | ESPset |
|---|---|---|
| Machines | 2, **disjoint fault sets** | **11, shared classes** |
| LOMO | ❌ impossible | ✅ only source |
| Cross-operating | ✅ 50/75/100% | ❌ order-normalised |
| Current channel | ✅ only source | ❌ |

⭐ **Twente cannot support LOMO** — its two motors share only the healthy class, so
every fold would train and test on disjoint labels. Published claims of
cross-machine results on Twente alone are mistaken. Figure **D1**, **D13**.

The leakage ladder (levels 0–4), plus the general rule that a split is only
interpretable when every fold trains on the classes it tests.

### 5. Results

**5.1 C2 — cross-machine (Table 1, Figures D7, D11)**

| Model | Macro-F1 | Acc | Cov | Per-machine CI |
|---|---|---|---|---|
| majority | 0.228 ±0.000 | 0.837 | 1.00 | [0.290, 0.395] |
| logistic | 0.663 ±0.000 | 0.914 | 1.00 | [0.578, 0.787] |
| LightGBM | 0.666 ±0.006 | 0.930 | 1.00 | [0.610, 0.779] |
| **TabPFN** | **0.738 ±0.015** | 0.911 | 1.00 | [0.583, 0.773] |
| TabPFN (abstaining) | 0.753 ±0.015 | 0.937 | 0.81 | [0.642, 0.810] |

Margin +0.072 at matched coverage = 4.4× combined seed std. ⚠️ **But per-machine
CIs overlap** — state that machine count, not seed count, is the binding constraint.

**5.2 C5 — leakage (Figure D13, B6)**

| Dataset | Invalid random split | Honest split | Inflation |
|---|---|---|---|
| synthetic | 1.000 | 0.930 (LOMO) | 1.1× |
| ESPset | 0.793 | 0.421 (LOMO, 11 folds) | **1.9×** |
| Twente | 0.851 | 0.147 (record-wise) | **5.8×** |

The effect grows with how real the data is. **B6** is the visual form: colour the
same PCA by class, then by machine.

**5.3 C4 — does the expensive model earn it (Figures D2, D12, D4)**

- Tuned baselines: logistic 0.663→0.638, LightGBM 0.666→0.664 — **tuning does not
  rescue them**, so the win is not an untuned-comparison artefact
- ⭐ **Alarm budget** (≤1 false alarm/pump/month = FAR 0.00093): majority 0.000,
  logistic 0.032, LightGBM 0.084, **TabPFN 0.203**
- **Commissioning spec:** accuracy saturates at ~500 reference rows (0.739, 0.76 s)
  and *regresses* at 1000 — a bigger reference set is not simply better

**5.4 Secondary — cross-operating (Twente, Motor-2 at 50/75/100%)**
TabPFN 0.459 vs LightGBM 0.410, logistic 0.134, majority 0.126.
⚠️ Abstention's sign flips between datasets: helps on ESPset (+0.015), hurts here
(−0.025). Always report the variant and its coverage.

### 6. Limitations
Machine count is binding; 20% recall at the alarm budget bounds deployability;
Twente is a subset; vane counts unobtainable; burst pairing is an approximation;
no run-to-failure data so no RUL claim — detection-vs-severity (**D14**) is the
honest substitute.

### 7. Conclusion
Reference-set substitution works cross-machine; the protocol result may outlast
the model result.

---

## 3. What is left — ordered checklist

### Must do before submission

- [ ] **Cut C3** from the contributions, or reframe as "architecture + simulated
      validation, dataset deferred"
- [ ] **Write §2 Related work** — 🔴 the only section with no material in the repo
- [ ] **Decide the venue framing** (see §4 below)
- [ ] **Confirm the ESPset licence attribution** — CC BY 4.0 requires citation;
      TabPFN's Prior Labs License §10 requires "Built with PriorLabs-TabPFN" *if
      distributing weights* (not for a paper, but check if you release code+weights)
- [ ] **Regenerate figures at publication DPI** and pick the final ~8

### Should do — cheap, strengthens the paper

- [ ] **Twente wider extraction** (~40 min) — unlocks the component-wise rung, gives
      D14 more than one severity level, and `ch3` may settle the vane count
- [ ] **Run the gate on ESPset** — 4801 healthy records clear the commissioning
      shortfall that the demo cache cannot; turns a caveat into a result
- [ ] **Deduplicate figures** — A3/A6/A7/C2/C5/C7/C8/E3/E4 are dataset-independent
      but currently rendered into all three output directories
- [ ] **Delete `figures/espset/D6_calibration_tabpfn.png`** — stale, pre-rename

### Would strengthen most, but costs the most

- [ ] **More machines.** The single highest-value experiment left. Per-machine CIs
      span ~0.19 and overlap; a twelfth pump sharpens C2 more than any code change.
      Worth checking whether another multi-machine rotating-machinery set (e.g.
      Paderborn, 14 real bearing damages) can test the in-context claim outside pumps.
- [ ] **The rig** — unblocks C3, and it is the only route to a dry-run dataset

### Explicitly not doing
RK3588 NPU port, multi-label, Friedman/CD diagrams at n<5, RUL claims.

---

## 4. Venue

| Option | Fit |
|---|---|
| **IEEE Sensors Journal** | The TabPFN rotating-machinery precedent is there. Natural home for C1+C2+C4. |
| **PHM Society** | If **C5 becomes the headline** — the leakage result is a methodology contribution and this audience cares. |
| **IEEE Sensors Letters** | Short first paper on C1 + the energy inversion alone. Lowest risk, banks a result. |
| **MSSP** | If C5 leads and the framing is "evaluation protocol in machine diagnosis". |

⭐ **Recommendation: lead with C5 and submit the methodology framing.** The
cross-machine result is good but its CIs overlap; the leakage result is
unambiguous, demonstrated on two real datasets, and applies to everyone else's
work as well as yours. It is the more defensible headline *and* the more useful one.

---

## 5. Honest self-assessment

**What a reviewer will accept:** the leakage ladder, the Twente-cannot-do-LOMO
finding, the alarm-budget framing, the energy inversion, the confound audit.

**Where a reviewer will push:**
1. *"11 machines is not many."* True. Answer with the per-machine CIs shown openly
   and the statement that machine count is the binding constraint.
2. *"20% recall is not deployable."* Also true. Answer: it bounds the claim, it is
   reported rather than hidden, and it is 2.4× the GBDT.
3. *"ESPset is not irrigation pumps."* Correct — they are offshore ESPs. The
   architecture targets irrigation; the cross-machine evidence comes from
   submersibles. **Say this plainly rather than blurring it.**
4. *"Where is the dry-run data?"* Cut C3 and the question does not arise.
