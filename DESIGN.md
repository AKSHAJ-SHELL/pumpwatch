# DESIGN.md — Two-Tier Irrigation Pump Fault Monitoring (revised)

**System design v1.2** (post red-team, post implementation)  
Akshaj Shandilya

v1.1 was the revised design after an adversarial review of v1.0. v1.2 records what
changed once the code was actually run — several v1.1 assertions turned out to be
wrong, and the corrections are findings in their own right. Every correction traces
to a specific defect. The paper must state them the same way.

---

## −2. What the real data changed about the design

Both public datasets are now downloaded and parsed. Three of the design's
assumptions about them were wrong.

**−2.1 ⛔ Twente cannot support leave-one-machine-out.** v1.0 §4.1 claimed "two
pumps at two speeds also gives you a genuine (if small) leave-one-machine-out
axis". It does not. Motor-2 carries the bearing, impeller and electrical faults;
Motor-4 carries the cavitation, alignment, unbalance and coupling faults; **the only
labels they share are the healthy variants.** Every LOMO fold would train and test
on disjoint label sets. The confound audit fires on this unaided (7 of 8 classes
flagged as single-machine). `twente_raw.lomo_feasible()` reports the evidence and
the experiment refuses the split.

→ **C2 rests entirely on ESPset**, which has 11 real machines.

**−2.2 ESPset is far more important than "realism check" (DESIGN §4).** It is the
only dataset in the project that can test the cross-machine claim: 11 in-service
submersible pumps, 6032 records, field prevalence (~84% healthy). Its licence is
**CC BY 4.0** (verified, not "unverified") and the current version is **DOI
10.17632/m268jsw339.3**, not the `.1` previously cited. Constraints: spectra only
(no time-domain or envelope features), order-normalised (no absolute-frequency
bearing analysis), velocity in in/s (converted to mm/s on load), and **vibration
only — no current channel**, so ESPset cannot speak to `ct_only`, MCSA or dry
running.

**−2.3 The two real datasets are complementary, not redundant.**

| | Twente/4TU | ESPset |
|---|---|---|
| Machines | 2, disjoint fault sets | **11, shared classes** |
| LOMO (C2) | ❌ impossible | ✅ the only source |
| Cross-operating | ✅ Motor-2 at 50/75/100% | ❌ order-normalised |
| Current channel | ✅ | ❌ |
| `ct_only` vs `full` | ✅ only source | ❌ |
| Faults | seeded on a rig | **in service** |

**−2.4 Leakage inflation, measured on real machines.** Same data, same model, two
split protocols:

| Dataset | Invalid random split | Honest split | Inflation |
|---|---|---|---|
| ESPset (LightGBM) | 0.793 | 0.444 (LOMO, 11 folds) | **1.8×** |
| Twente (LightGBM) | 0.851 | 0.147 (record-wise) | **5.8×** |

This is the leakage argument demonstrated rather than asserted, and it is a
stronger result than anything the synthetic data produced.

**−2.5 On real cross-machine data, nothing works well yet.** ESPset LOMO macro-F1:
logistic 0.46, LightGBM 0.43, majority 0.23 — and LightGBM's *accuracy* (0.849) is
**below** the majority baseline's (0.837), because 84% of the data is healthy. This
vindicates macro-F1/PR-AUC as headline over accuracy, and it is the honest starting
point rather than a failure.

**−2.6 TabPFN wins where the design predicted it would.** On Twente's real
cross-operating split (Motor-2, hold out a speed): TabPFN 0.459, LightGBM 0.410,
logistic 0.134, majority 0.126. Small n, modest d, distribution shift — exactly its
claimed regime. On synthetic data with abundant samples, LightGBM won. Both results
belong in the paper.

**−2.7 A split is only interpretable if every fold trains on the classes it
tests.** Generalised from −2.1 into `splits.split_label_coverage`. It immediately
caught two degenerate rungs created by the extraction subset itself. Report it
alongside any ladder result.

**−2.8 ⚠️ The normalisation-strategy result inverts between synthetic and real
data — so §−1.2's framing needs care.** On the synthetic set, transductive
per-machine normalisation *helped* (logistic 0.94 vs 0.61 inductive). On ESPset it
*hurts badly*: logistic 0.46 transductive vs **0.67** inductive, LightGBM 0.43 vs
**0.67**.

The mechanism is the point. Per-machine z-scoring removes each machine's own mean
and scale. That is the right move when the between-machine variation is **nuisance**
(different rpm, sensor gain, mounting) — the synthetic case, where machines differed
hugely in scale. It is the wrong move when the between-machine variation carries
**signal** — ESPset's spectra are already order-normalised and velocity-calibrated,
so the remaining amplitude differences *are* severity, and normalising them away
destroys the thing being classified.

→ **The choice of normalisation is a claim about what varies between machines, and
it must be justified per dataset rather than fixed.** Report both; the gap is
diagnostic.

---

## −1. What the implementation changed about the design

These are conclusions from running the code, not restatements of intent.

**−1.1 The radio is not the energy bottleneck. Continuous sensing is.**
v1.0 §5.4 put LoRa TX at 55% of the node budget and concluded "all optimisation
effort belongs in transmitting less often". That was computed for a fixed 15-minute
schedule the same document declared inadmissible. With the gate working and
event-triggered operation, TX is **~1%** of the budget and continuous CUSUM sampling
during pump runtime is **~98%**. Because dry-run protection requires monitoring
current continuously whenever the pump runs (§0.2), that sampling cost is
irreducible without new hardware. **The optimisation target is a cheaper always-on
front end — a comparator or duty-cycled CUSUM — not a smaller payload.** Figures
E3, C5.

**−1.2 Cross-machine normalisation is a design decision, not preprocessing.**
Under LOMO the held-out pump has no labelled history, so "fit the scaler on training
data" is undefined for it. Two defensible answers, and the gap between them is a
result: `train_pooled` (inductive, never touches the target pump) and
`unsupervised_per_machine` (transductive, uses the target's *unlabelled*
distribution). On the current data logistic regression scores 0.937 transductive vs
0.607 inductive; LightGBM barely moves (0.990 vs 0.977). **Any cross-machine claim
must state which assumption it is making.** Figure D11.

**−1.3 The MCU gate is bounded by commissioning length, not by feature count.**
Mahalanobis needs n > 10p healthy windows, so a 63-feature gate demands ~630 windows
of healthy operation before the node can be armed. The gate therefore runs on an
8-feature physically-chosen subset (`DEFAULT_GATE_FEATURES`). Commissioning adequacy
is checked and reported, not assumed.

**−1.4 Escalation rate must be prevalence-weighted.** A test set's class mix reflects
how many faults were collected; battery life depends on how often a real pump is
faulty. Reported both ways.

**−1.5 TabPFN's OOD abstention can silently destroy the cross-machine claim.** With
the covariance accepted at n > p+2, the context ellipsoid was so ill-conditioned that
every sample from an unseen pump fell outside it: coverage 0.00, macro-F1 0.000 under
LOMO. "Adapts to a new pump" had become "refuses to answer" with no error. Now
requires n > 10p and warns when it disables itself. **Report coverage beside every
selective-prediction score.**

**−1.6 The v2 licence pin was inverted, and package versions ≠ model versions.**
There is no 3.x TabPFN *package*; PyPI goes 2.x then 6.x/7.x/8.x. On the 2.x line
there is no `ModelVersion` and no `create_default_for_version`, so the v1.1
instruction to call it always raised, always fell through to an unverified
constructor, and always tagged results "fallback". **The pin is the package
constraint `tabpfn>=2.0,<3`.** Verified: tabpfn 2.2.1 declares the Prior Labs
License v1.1 (Apache 2.0 + attribution). §10 of that licence requires displaying
**"Built with PriorLabs-TabPFN"** on distribution — an obligation, not a footnote.

**−1.7 The KV cache was never enabled.** The wrapper used the library default
`fit_mode="fit_preprocessors"`, which caches preprocessing but re-encodes the whole
context every call. `fit_with_cache` is the KV cache. Measured on this machine,
single-threaded: **11.03 s → 0.27 s** (41×) on a 416×63 context. Figure E1.

**−1.8 TabPFN does not beat a tuned GBDT here.** Under LOMO: LightGBM 0.990,
TabPFN 0.984, McNemar p=0.73 — no significant difference at 3–5 orders of magnitude
more compute. TabPFN does win on the low-dimensional `ct_only` profile (0.989 vs
0.893) and under inductive normalisation. **That is the honest C4 result and it is
still publishable.** All on synthetic data; see §4.

**−1.9 LightGBM and torch segfault together on macOS.** Both ship an OpenMP runtime;
the process dies (exit 139) when the second one spins up its thread pool — which is
exactly what the C4 comparison requires. Pinned to one OpenMP thread. Reported
latencies are therefore single-threaded and conservative.

---

## 0. Corrections that reshape the system

### 0.1 Dry-run leaves the ML classifier

There is still **no public labelled dry-run dataset**. That does not mean dry-run
becomes a TabPFN class merged with Twente bearings.

**Class–rig confounding:** dry-run labels exist only on the own rig; bearing /
misalignment / unbalance exist in Twente. Merging them makes *rig identity* the
dry-run feature. `audit.py` hard-fails such merges.

**Architecture:** dry-run protection is a **CT under-current CUSUM + trip** at the
MCU — a commodity-relay equivalent, framed as such. Detection terminates at the
node with a contactor trip (see §0.2). ML dry-run experiments run **within
own-rig data only**, as characterization of the trip path — not as cross-source
classifier training.

A $10 under-current relay already solves dry-run. This system's trip path must
justify itself by (a) false-trip analysis against valve-throttle confusers,
(b) integration with the gated feature uplink for slow faults, and (c) cost /
telemetry bundling — not by pretending ML invents under-current detection.

### 0.2 Detection without actuation is theater

A mechanical seal dies in **<60 s**. Escalating a LoRa packet to a farmer does
not protect the seal. The node must **trip a contactor**. That makes this a
safety-relevant actuation path:

- false-trip cost = lost irrigation hours
- miss cost = destroyed seal / pump
- radio path must not be the sole trip authority for dry-run (local CT CUSUM is)
- authenticated baseline updates (see threat model)

`node/trip.py` models detection → actuation latency and false-trip rates against
closed-valve confusers.

**The three mechanisms are ANDed, not ORed.** CUSUM detects the abrupt shift with
minimal delay but fires on *any* load loss; the absolute floor (0.55 × rated, between
dry-run's ~0.45 and closed-valve's ~0.70) is the only mechanism that knows the
difference; persistence rejects transients. Under OR the floor was decorative and the
trip fired on **100% of closed-valve confusers and 23% of healthy runs** — a contactor
that cuts irrigation whenever a farmer throttles a valve. ANDed: detection 1.00,
closed-valve false trip 0.00, healthy 0.00, median delay ~1.5 s against the 60 s seal
budget. The operating point is selected off a measured sweep
(`sweep_trip_operating_points`, figure C7) subject to a false-trip budget, not
hardcoded — it is a safety decision with asymmetric costs.

⚠️ The false-trip rate is more sensitive to assumed healthy-current variability than
to any tuning parameter. It is a parameter (`HEALTHY_CURRENT_NOISE_FRACTION`, 8%) and
**must be measured on the rig**; the original 2% made a 30% closed-valve drop a ~15σ
excursion, which is why CUSUM defeated every confuser.

### 0.3 Two deployment profiles

| Profile | Sensors | Target |
|---|---|---|
| `full` | vibration + CT (+ optional pressure) | surface / monoblock |
| `ct_only` | CT at starter box | **submersible borewell** (dominant Indian smallholder pump) |

You cannot stud-mount an IIS3DWB on a pump 60 m down a borewell. Every
classification experiment runs **both** profiles. CT-only is the honest headline
for the stated user; vibration results are secondary.

### 0.4 Event-triggered energy is primary

The v1.0 "4–5 year battery" number was computed from a **15-minute fixed
schedule the same document declared inadmissible**. That arithmetic is retained
only as a falsified comparison (`fixed_schedule_energy`).

Primary model: wake-on-start → continuous CUSUM while pump runs → gated TX.
Pump runtime hours/day is a required parameter. LoRa airtime uses the Semtech
formula (`node/airtime.py`); payload size comes from the serialized feature
vector — no more "100 B @ SF9 = 0.40 s" fiction.

### 0.5 Power factor is conditional

PF requires a voltage phase reference. Default BOM is **CT-only** (no ZMPT).
Feature schema carries `voltage_available`; PF features are never silently
invented. Open hardware trade: add voltage tap vs drop PF — quantify via ablation.

### 0.6 No stage-1 binary TabPFN; no multi-label TabPFN

The MCU gate **is** stage 1. The gateway runs one multiclass classifier with
abstention/OOD. Multi-label is dropped (TabPFN is single-label). Co-occurring
faults are a stated limitation, surfaced via confusion analysis.

### 0.7 TabPFN: v2 pin, CPU only, no NPU claim

| Version | Commercial use |
|---|---|
| **v2** | yes (Prior Labs License ≈ Apache 2.0 + attribution) |
| 2.5 / 2.6 / **3 (pip default)** | no |

Pin: `TabPFNClassifier.create_default_for_version(ModelVersion.V2)`.
RK3588 NPU cannot accelerate TabPFN (variable shapes, unsupported ops). Budget
CPU on 4× A76. Defensible claim:

> **No per-deployment gradient retraining — new pumps are adapted by swapping
> the in-context reference set (cross-machine generalization under LOMO).**

Not: "fully training-free" / "NPU accelerates the classifier."

### 0.8 Right-sized statistics

LOMO with 2–3 machines is 2–3 data points. No Friedman–Nemenyi / CD diagram
until ≥5 datasets. McNemar exact for model pairs; bootstrap at recording (or
machine) level; report per-machine LOMO results honestly.

---

## 1. Contributions (reframed)

- **C1.** Two-tier gate/classify architecture with a quantified **event-triggered**
  power budget and a local dry-run trip path.
- **C2.** Cross-machine generalization of an in-context reference set, tested
  under **leave-one-machine-out** (the only protocol that tests the claim).
- **C3.** *(Deferred to rig phase.)* Labelled dry-run characterization set for
  the trip path — not merged into cross-source classifiers.
- **C4.** Compute-vs-benefit accounting: TabPFN v2 vs LightGBM vs logistic vs
  majority, with latency. If GBDT wins, that is still a publishable result.

---

## 2. Architecture

```
PER PUMP (battery)
  ADXL372 / CT threshold wake
  IIS3DWB (full profile) + split-core CT
  STM32U5: features + EWMA + CUSUM(current) + Mahalanobis
       │
       ├─ DRY-RUN TRIP ──▶ contactor (local, sub-minute)     ★ fast path
       │
       └─ gated LoRa (IN865) feature vector ──▶ GATEWAY
                                              RK3588 CPU (not NPU)
                                              TabPFN v2 + baselines
                                              abstention / OOD
```

Slow faults (bearing, cavitation, impeller) go to the gateway. Dry-run does not.

---

## 3. Sensors and features

- Severity band ≠ diagnostic band. Do not size the accelerometer to ISO 10816-7 alone.
- Feature vector is **schema-versioned** (currently 1.1.0), not hardcoded to 42.
- Speed estimation from spectrum when nameplate RPM unknown — resolved before any
  profile branch, since MCSA sidebands need it and `ct_only` has no vibration channel.
- Vane count / bearing geometry optional — features degrade to a defined subset.
  Geometry is *nameplate data*, available to `ct_only` too: the profile decides which
  **signals** exist, not which facts about the pump are known.
- Profiles: see `configs/profiles.yaml`.
- **Dual-rate acquisition** (`node/acquire.py`): 26.7 kSPS burst for the cavitation
  band and bearing envelope; decimated 1.67 kSPS long window for order analysis and
  VPF ± 1× sidebands. The decimation is anti-alias filtered — naive `x[::16]` folds
  the 4 kHz bearing carrier onto ~663 Hz, on top of the shaft orders.

### 3.1 MCSA is what makes `ct_only` a profile rather than a null result

A CT at the starter box sees mechanical faults because periodic torque disturbances
amplitude-modulate stator current, placing energy at f_line ± k·f_disturbance. Without
those features `ct_only` reduces to two scalars derived from one number and cannot
separate fault classes at all — which would make the "honest headline profile for the
stated user" (§0.3) vacuous. Unbalance drives 1×, misalignment 2×, looseness
half-order, impeller damage VPF ± 1×; bearings couple weakly, which is the honest
reason `ct_only` should trail the full profile on bearing classes. Sidebands are
reported relative to the fundamental so they are invariant to load and CT scaling —
necessary for a reference set to transfer between pumps of different sizes.

⚠️ Two feature-scaling corrections worth carrying into the paper: `iso_vel_rms_mm_s`
is now genuinely mm/s (summing squared FFT bins is a *power* calculation and needs the
window's power gain, not the coherent gain that corrects a peak amplitude), and the
bearing envelope is band-passed around the resonance before the Hilbert transform —
without that, 1× and vane-pass content dominate and the "bearing" features measure
shaft harmonics.

---

## 4. Data

- **Twente / 4TU** (Kumar et al. 2023, CC BY 4.0): anchor for cavitation,
  impeller, bearing, current+vibration. **No dry-run class.**
- **Own rig:** dry-run / graded cavitation / impeller — with full session
  metadata for leakage-safe splits. Seal-temp hard cutoff mandatory.
- **Never merge** own-rig-only classes with Twente-only classes in one classifier.
  `audit.assert_not_confounded` enforces this.
- Twente lists **15 fault families against TabPFN's hard cap of 10**, so real data
  would fail at `fit_context`. `datasets.twente.collapse_labels` groups families that
  share a mechanism and a maintenance action; it raises on unmapped labels rather
  than truncating, because dropping classes to fit a model limit changes what the
  reported accuracy means.
- Per-pump geometry (vane count, bearing dimensions, rated current) comes from the
  manifest. It used to be hardcoded for every record, which would place every VPF and
  bearing-envelope feature at an invented frequency on real data.

⚠️ **All numbers currently in `results/` are synthetic.** The signatures were written
into the generator by hand, so the scores verify that the feature pipeline and splits
recover signatures known to be present — a wiring check and an upper bound, not
evidence about pumps. The `ct_only` score in particular is high because the generator
encodes clean torque-modulation sidebands; real motor current carries load variation,
supply distortion and VFD switching noise. `results_*.json` carries this caveat inline.
**No claim in the paper may cite these numbers as a result.**

---

## 5. Validation

Leakage ladder 0–4, **all five rungs run** (levels 1–3 were implemented and never
called; only LOMO ever ran). LOMO is mandatory for C2. Context-set construction is
explicit under each protocol (TabPFN leakage hygiene). Normalisation strategy is an
explicit choice, reported both ways — see §−1.2. Metrics: PR-AUC headline, ROC-AUC
secondary, recall@fixed FAR, calibration (ECE), multiclass Brier, coverage, raw
confusion counts, McNemar over every model pair.

⚠️ **The ladder is only meaningful if the data has the nuisance structure it
exposes.** With one independently generated window per record, all five rungs score
identically and the figure demonstrates nothing. The demo cache is therefore
structured as pump → operating point → condition → session → windows, with each
session fixing a sensor gain, mounting resonance and noise floor that its windows
inherit. On that structure the INVALID random-window split reports macro-F1 1.000
while LOMO reports 0.90–0.99. Real recordings have this structure natively; a
synthetic stand-in has to be built with it deliberately.

---

## 6. Economics (finding, not footnote)

Indicative BOM (USD, 2025-ish, qty small):

| Item | full profile | ct_only |
|---|---|---|
| STM32U5 node + SX1262 + PSU | ~40 | ~40 |
| IIS3DWB + mount | ~25 | — |
| ADXL372 wake | ~5 | — |
| Split-core CT | ~5 | ~5 |
| Enclosure / install | ~30 | ~20 |
| **Node subtotal** | **~105** | **~65** |
| RK3588 16 GB + NVMe gateway | ~150 | ~150 |
| Contactor interface | ~15 | ~15 |

Compare: smallholder pump ~$150–400; commercial dry-run relay ~$10–40.
A monitoring system that costs more than the pump is a **result to report**,
not something to hide. ct_only + shared gateway amortised across N pumps is
the only plausible smallholder economics; state N.

---

## 7. Threat model (raw LoRa P2P)

- No LoRaWAN MIC / join. Nodes speak raw LoRa to one gateway.
- **Baseline updates** (μ, Cholesky L) redefine "normal." Must be authenticated
  (symmetric key per node, or out-of-band commissioning). Spoofed baseline =
  attacker-controlled false trips or missed faults.
- Dry-run trip is **local** (CT CUSUM) — must not depend on gateway reachability.
  Rural mains / load-shedding takes the gateway offline during pump hours;
  node autonomy for the trip path is mandatory. Store-and-forward for slow-fault
  uplinks when gateway returns.

Verify IN865 ERP against primary WPC gazette before finalising RF BOM.
**868 MHz is not licence-free in India.**

---

## 8. Scope cuts

Out of this codebase: firmware, PCB, RKNN port, multi-label TabPFN, Friedman
diagrams at n=3, RUL/prognosis claims on pumps, NPU acceleration claims.

In scope: physics + synth, dual-rate acquisition model, gates/trip, energy/airtime
models, features/profiles, Twente/ownrig loaders, confound audit, splits, baselines
+ TabPFN v2 wrapper, evaluation, figures, this design doc.

**Known gaps, stated rather than hidden:**

- ~~No real data.~~ **Done.** Twente/4TU (20.8 GB, MD5-verified) and ESPset are both
  downloaded and parsed; see §−2. `results_espset_*.json` and
  `results_twente_real.json` contain real measurements. `results_full.json` remains
  synthetic and is still labelled as such.
- Twente is loaded from a **subset** of the archive (6 conditions × 2 motors,
  1 vibration + 1 current channel, 8 bursts each) because full extraction needs
  ~320 GB. Extracting more severity levels per fault family would make the
  component-wise rung interpretable; right now it is not (§−2.7).
- Own-rig now has an interlocked acquisition loop and a simulated backend, so the
  whole collection path including the abort branch is exercisable. What remains is
  a real DAQ backend and an actual rig. **C3 still has no collected data.**
- Twente vane counts and bearing geometry are only in PDF datasheets, so VPF and
  envelope features degrade out on real Twente data.
- Twente vibration and current bursts are paired by index, which is an
  approximation (§ script docstring), and it is what the real-data `ct_only`
  comparison rests on.
- The MCU gate's commissioning requirement (n > 10p) is not met by the demo cache; the
  run reports the shortfall rather than pretending otherwise.
- No hyperparameter tuning or nested CV for any baseline, so a "TabPFN wins" result
  would be untuned-baseline-confounded. Currently TabPFN does not win, which makes
  this less urgent — but it must be fixed before any claim that it does.
- No repeated seeds. Every number is a single deterministic run.

---

## 9. Build order

See Makefile. Physics → gates/trip → energy → features → datasets/audit →
baselines → TabPFN → LOMO → figures.

## 10. Attribution obligation

TabPFN is used under the **Prior Labs License v1.1** (Apache 2.0 + attribution).
§10 requires that any distributed product or service containing the weights
prominently display:

> **Built with PriorLabs-TabPFN**

Internal benchmarking and testing do not trigger it. Deploying to a farmer or a
cooperative does. The string is kept in code as
`gateway.tabpfn_clf.ATTRIBUTION_NOTICE` so the obligation travels with the software
rather than living only here.
