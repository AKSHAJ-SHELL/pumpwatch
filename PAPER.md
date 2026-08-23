# PAPER.md — draft structure, evidence map, and what is left

**Status: skeleton with evidence attached.** Every number below is cited from a
committed results file; every figure named exists on disk. Sections marked
🔴 have no evidence yet and must either be produced or cut.

Companion documents: [DESIGN.md](DESIGN.md) (system design + 20 findings),
[remediation.md](remediation.md) (code-health design, executed).

---

## 0. Abstract (draft — reframed against the synopsis)

> Smallholder irrigation pumps fail silently, and the failure that matters most is
> fast: less than a minute of dry running destroys a mechanical seal. Per-pump
> machine learning is impractical at this price point because commissioning cost
> dominates hardware cost. We present a two-tier monitoring architecture in which
> battery-powered MCU nodes perform continuous statistical gating and a shared
> gateway classifies escalated events with a prior-fitted tabular foundation model
> (TabPFN v2), so that **commissioning a new pump requires substituting an
> in-context reference set rather than retraining**.
>
> We evaluate on two public datasets of real machines under a five-rung leakage
> ladder. On **11 in-service submersible pumps under leave-one-machine-out**,
> TabPFN reaches macro-F1 **0.738 ± 0.015** against a nested-tuned gradient-boosted
> baseline at **0.666**; at a deployment-realistic budget of **one false alarm per
> pump per month** it recovers **2.4× as many faults** (20.3% vs 8.4%). Accuracy
> saturates at roughly **500 labelled reference windows**, giving a concrete
> commissioning specification.
>
> We further report a protocol result that applies beyond this system: on identical
> data and models, **random-window splits inflate reported macro-F1 by 1.9× on
> in-service data and 2.4× on rig data** relative to protocols that hold out the
> machine or the recording session — and we show that one widely used pump dataset
> **cannot support cross-machine evaluation at all**, because its two machines share
> no fault class.
>
> We report three negative results that revise our own design: vibration is the
> wrong primary sensor for dry running (motor current is), "training-free"
> overstates what in-context learning provides, and the edge accelerators on our
> deployment hardware cannot take a model whose input shape varies by construction.
> (⚠️ We did **not** attempt the port — state this as a constraint analysis, never as
> a tested negative. See PAPER_DRAFT.md §3.4.)

✅ **FIXED — the two 2.4× are separated in PAPER_DRAFT.md's abstract.** Original note:
⚠️ **Two different 2.4× appeared in the abstract** — the recall multiplier at the
alarm budget, and the Twente leakage inflation after the wider extraction (was 5.8×
on the narrower subset). They are unrelated quantities and a reader will conflate
them. Reword one; the leakage sentence is the easier to recast (e.g. "by a factor of
two on in-service data and rather more on rig data", with the exact figures in §5.2).

**Abstract discipline — do not reintroduce these:**

- ❌ "fully training-free" / "no gradient training at any stage" (DESIGN §0.7)
- ❌ any claim an NPU or TPU accelerates the classifier
- ❌ dry-run as a *classified* class — it is a local trip (DESIGN §0.1)
- ❌ "irrigation pumps" where the evidence is offshore submersibles — say which

---

## 0b. Deviations from the synopsis — state these explicitly

The synopsis was a proposal. Three of its claims were rejected by our own
red-team **before** implementation, and reporting them as findings is stronger than
quietly dropping them.

| Synopsis claim | What we found | Disposition |
|---|---|---|
| "fully training-free", "no gradient training at any stage" | TabPFN removes *gradient* training, not the need for labelled examples of every class in the context | **Reframe**: "no per-deployment retraining" |
| vibration via "contact-mounted accelerometers" | Vibration *decreases* under dry running; motor current drops 30–60% and is unambiguous. A borewell submersible cannot be accelerometer-mounted at all | **Finding**: add a CT; report the `ct_only` profile |
| classify "dry-run, cavitation, bearing wear" | Dry-run moved to a local CT trip (mixing rig-only dry-run with public faults makes rig identity the feature). Cross-machine evidence covers healthy/misalignment/rubbing/unbalance | **Rename the fault set honestly** |
| "each pump's own normal-operation data as in-context examples" | Tested. Using the target pump's own distribution **hurts** on real data (0.46 vs 0.66); the reported result pools *other* machines | **Finding**: report both, §−2.11 |
| "labeled vibration dataset of induced faults on low-cost pumps" | No rig; no data | 🔴 **Cut to future work** |
| "RK3588 gateway" | ✅ RESOLVED — measured on an Orange Pi 5 Plus, 88 ms/window | ✅ was: **Measure or restate as "an ARM-class gateway"** |

---

## 1. Contributions — reframed

Stated as what the evidence supports, in the order they should appear.

**C1 — A two-tier architecture with a measured, not assumed, power budget.**
Continuous statistical gating at the node; classification only on escalated events.
Measured: field-weighted escalation **2.0%**, **0.73 uplinks/day**, gate recall
ceiling **0.996**.
⭐ *Finding that inverts the design's own assumption:* with a working gate, LoRa TX
is **0.7%** of the node budget and continuous sensing is the rest. The optimisation
target is a cheaper always-on front end, not a smaller payload.
Evidence: `results_full.json` → `gate_summary`. Figures C5, E3, E4.

**C2 — Cross-machine adaptation by reference-set substitution, no retraining.**
11 in-service pumps, leave-one-machine-out, 5 seeds. TabPFN **0.738 ± 0.015** vs
nested-tuned LightGBM **0.666**; margin **+0.072**, 4.4× the combined seed std.
Commissioning needs ~**500** labelled reference windows (0.739; regresses at 1000).
⚠️ Per-machine CIs overlap — machine count, not seed count, is the binding
constraint, and we say so.
Evidence: `results_espset_both.json`. Figures D7, D11, D4.

**C3 — Compute-vs-benefit, against baselines given a fair fight.**
Nested, machine-grouped tuning does **not** rescue the baselines (logistic
0.663→0.638, LightGBM 0.666→0.664), so the win is not an untuned-comparison
artefact. ⭐ At **≤1 false alarm/pump/month** (FAR 0.00093): majority 0.000,
logistic 0.032, LightGBM 0.084, **TabPFN 0.203** — a 2.4× margin, far larger than
the 6% relative edge on macro-F1, because calibration matters most where the
operating threshold is extreme.
Evidence: `tuned_baselines`, `recall_at_alarm_budget`. Figures D2, D12.

**C4 — An evaluation-protocol result that applies beyond this system.** *(new)*
Same data, same models, different split protocols:

| Dataset | Random split | Honest split | Inflation |
|---|---|---|---|
| synthetic | 1.000 | 0.930 (LOMO) | 1.1× |
| ESPset (in-service) | 0.793 | 0.421 (LOMO, 11 folds) | **1.9×** |
| Twente (rig) | 0.853 | 0.352 (record-wise) | **2.4×** |

⭐ Plus: **one widely used pump dataset cannot support cross-machine evaluation at
all** — Twente's two motors share only the healthy class, so every LOMO fold would
train and test on disjoint labels. We give a general check (`split_label_coverage`)
that flags this before it produces a number.
Figures D1, D13, B6.

**C5 — Three negative results that revise the design.**
(i) vibration is the wrong primary sensor for dry running; (ii) "training-free"
overstates in-context learning; (iii) neither edge accelerator on the deployment
board — RK3588 NPU, Coral Edge TPU — can take this model without concessions that
defeat the mechanism. ⚠️ Constraint analysis, not an attempted port.

**Deferred to future work:** the induced-fault dataset on low-cost pumps (needs a
rig) and the dry-run characterisation set. The trip path is presented as
architecture with simulated validation, labelled as such.

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

⚠️ Normalisation strategy: **`train_pooled`** (inductive). §5.2 below uses
`unsupervised_per_machine`. Label both in the paper — see the Day 8–9 note.

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

⚠️ Normalisation strategy: **`unsupervised_per_machine`** (transductive), throughout
the ladder. Not comparable with §5.1's numbers.

| Dataset | Invalid random split | Honest split | Inflation |
|---|---|---|---|
| synthetic | 1.000 | 0.930 (LOMO) | 1.1× |
| ESPset | 0.793 | 0.421 (LOMO, 11 folds) | **1.9×** |
| Twente | 0.853 | 0.352 (record-wise) | **2.4×** |

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

## 3. What is left — 9-day plan

Written as a schedule because the deadline is the constraint. Anything not on it
is out of scope for this submission.

### Day 1–2 — Reframe (highest value, nothing blocks it)
- [ ] Paste the abstract and §1 contributions from PAPER_DRAFT.md into the template
- [ ] Rewrite the title: drop "training-free", name the actual fault set
- [x] Deviations subsection **drafted as prose** in PAPER_DRAFT.md, placed before §1
- [ ] State plainly that ESPset is offshore **submersible** pumps while the target
      application is irrigation. A reviewer who spots this unacknowledged will
      distrust everything else; acknowledged, it costs one sentence.

### Day 3 — Hardware, conditional  ← **only a command to run**
- [x] Benchmark script written: `make bench-hardware`. It reads the board identity
      from `/proc/device-tree/model` itself, so the `cat` step is folded in, and it
      warns loudly when there is no device tree — a laptop run of it otherwise looks
      exactly like a board run.
- [x] **DONE — measured on an Orange Pi 5 Plus (RK3588).** 88 ms/window in the
      deployed configuration; KV cache worth 7.1× and the 8-member ensemble costing
      5.6×, both within 0.3 of the laptop ratios. Board is uniformly 21–22× slower
      than the laptop with no configuration-dependent cliff. §3.4 of the draft is
      written from these numbers; `results/hardware_bench_rk3588_opi_5_plus.json`.
- [ ] **If torch will not install on it:** skip. Restate as "an ARM-class gateway";
      do not spend the day.
- [x] ❌ Do **not** attempt the Coral TPU port. ⚠️ But phrase the claim as a
      *constraint analysis*, not "we tested it and it failed" — we tested nothing.
      The honest obstacles are the operator set and INT8 quantisation of a
      prior-fitted model, not the input shape alone (padding would fix the shape).
      §3.4 of the draft has the defensible wording.

### Day 4–6 — Write
- [x] **§2 Related work — DRAFTED** in PAPER_DRAFT.md, citations verified against the
      published record. Key find: the ESPset authors (Varejão et al. 2024) already
      report a leakage effect on this dataset, F-measure 0.887→0.733 from removing
      *similarity bias* — which is our ladder's level 1. Our LOMO rung is two levels
      stricter and the effect is larger (1.9×). §2.3 positions against that explicitly.
      ⚠️ The old anchor "Vieira 2026" matches no real paper and was replaced by
      Wheat et al. 2024, IEEE Access 12:169879–169895.
      ⚠️ Three citation details still need your eyes — see the verification table at
      the end of §2.
      ⚠️ No PHM Society challenge has ever used a pump — do not cite one.
- [x] **Abstract — DRAFTED** against the final measured numbers, in PAPER_DRAFT.md.
- [x] §1, §3, §4, §5, §6, §7 drafted in [PAPER_DRAFT.md](PAPER_DRAFT.md) — read and
      edit for voice, but the material and its caveats are there.
- [ ] Every table comes from `make tables` → `results/paper_tables.md`. Do not
      hand-copy numbers into the template; regenerate after any re-run.

### Day 7 — Two cheap wins that convert limitations into results  ← **done**
- [x] **Twente wider extraction**: 150 CSVs, 395 records on ch1. Component-wise and
      cross-operating rungs now populated; D14 has five severity levels. The `ch3`
      channel did **not** settle the vane count — it yields no candidate Z at any
      speed, refuting the pump-end-sensor hypothesis. That is a reportable negative,
      written into §4.1 of the draft.
- [x] **Gate on ESPset**: commissioned on **11/11** machines with the compact feature
      set — field escalation 5.9%, recall ceiling 0.98, 2.1 uplinks/day. Widening to
      the full feature set *degrades* it to 0.83 / 8.2% / 10-of-11, which is direct
      real-data evidence for DESIGN §−1.3 (the gate is bounded by commissioning
      length, not feature count). Both are in §5.3.

### Artifact — repo is pushed, DOI needs one browser step
- [x] Private repo at `github.com/AKSHAJ-SHELL/pumpwatch`, 56 commits, tagged `v1.0.0`
- [x] `CITATION.cff` and `.zenodo.json` written; `results/*.json` tracked so
      `make tables` / `make figures-all` reproduce every number without the 20.8 GB
      Twente download
- [ ] **Mint the DOI** — see [RELEASE.md](RELEASE.md). ⚠️ Zenodo's GitHub integration
      only archives **public** repos, so the route depends on your venue:
      single-blind → make public and connect Zenodo (Route A); double-blind but you
      want a DOI → *reserve* a DOI on an embargoed Zenodo record (Route B);
      double-blind and no DOI needed → cite "available on acceptance" (Route C).
- [x] Data-and-code availability statement drafted (end of §7); insert the DOI when minted

### Day 8–9 — Finish
- [ ] Pick the final ~8 figures from the 38 available (**yours** — an editorial call)
- [x] All figures regenerate at 300 dpi; none in the tree is below it
- [x] Limitations drafted (§6 of PAPER_DRAFT.md)
- [x] Housekeeping: the eight dataset-independent figures now render into `figures/`
      once, per-dataset targets pass `--no-shared-figures`, and the stale
      `figures/espset/D6_calibration_tabpfn.png` is gone
- [x] Attributions collected in [ATTRIBUTION.md](ATTRIBUTION.md); repo now has a
      LICENSE. Prior Labs §10 is discharged in code (`ATTRIBUTION_NOTICE`, covered by
      a test), in LICENSE and in ATTRIBUTION.md — so it holds whether or not the
      release is judged to distribute weights.
- [ ] ⚠️ **Label the normalisation strategy on every results table.** The outline
      quoted `train_pooled` in §5.1 and `unsupervised_per_machine` in §5.2 unlabelled;
      on real data the gap is up to 0.251 macro-F1, larger than the model effect.

### Explicitly out of scope
Rig data, RKNN/Coral port, multi-label, Friedman/CD diagrams at n<5, RUL claims,
more machines (the thing that would help most, and cannot be done in 9 days).

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
