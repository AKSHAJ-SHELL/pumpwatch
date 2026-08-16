# DESIGN.md — Two-Tier Irrigation Pump Fault Monitoring (revised)

**System design v1.1** (post red-team)  
Akshaj Shandilya

This document is the revised design after an adversarial review of v1.0. Every
correction below traces to a specific defect. The codebase implements these
corrections; the paper must state them the same way.

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
- Feature vector is **schema-versioned**, not hardcoded to 42.
- Speed estimation from spectrum when nameplate RPM unknown.
- Vane count / bearing geometry optional — features degrade to a defined subset.
- Profiles: see `configs/profiles.yaml`.

---

## 4. Data

- **Twente / 4TU** (Kumar et al. 2023, CC BY 4.0): anchor for cavitation,
  impeller, bearing, current+vibration. **No dry-run class.**
- **Own rig:** dry-run / graded cavitation / impeller — with full session
  metadata for leakage-safe splits. Seal-temp hard cutoff mandatory.
- **Never merge** own-rig-only classes with Twente-only classes in one classifier.
  `audit.assert_not_confounded` enforces this.

---

## 5. Validation

Leakage ladder 0–4; LOMO is mandatory for C2. Context-set construction is
explicit under each protocol (TabPFN leakage hygiene). Per-machine
normalization fitted on training only. Metrics: PR-AUC headline, recall@fixed
FAR, calibration (ECE), raw confusion counts, McNemar.

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

In scope: physics + synth, gates/trip, energy/airtime models, features/profiles,
Twente/ownrig loaders, confound audit, splits, baselines + TabPFN v2 wrapper,
evaluation, figures, this design doc.

---

## 9. Build order

See Makefile. Physics → gates/trip → energy → features → datasets/audit →
baselines → TabPFN → LOMO → figures.
