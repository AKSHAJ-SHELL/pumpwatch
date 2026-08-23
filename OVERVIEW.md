# What this system is, and what the methodology is

`DESIGN.md` grew into a findings log and `PAPER_DRAFT.md` is written to persuade. This
is neither: it explains the system and the method plainly, for someone picking the
project up. Every number here is regenerable by `make tables`.

---

## 1. The problem

A smallholder irrigation pump fails on two timescales, and conflating them is the
design error everything else follows from.

**Fast and destructive.** Dry running — the pump loses suction, the mechanical seal
loses the fluid film that lubricates and cools it, and the faces are destroyed within
tens of seconds. Nothing that sends a measurement to a server and waits for a reply can
prevent this.

**Slow and expensive.** Bearing wear, impeller damage, misalignment, cavitation, a
loosening foot. Weeks to months. Worth *diagnosing* rather than merely detecting,
because the maintenance action differs by fault. Latency is irrelevant here.

**The real obstacle is commissioning cost, not inference cost.** A per-pump supervised
model needs labelled examples of every fault on every pump. For a population of
smallholder pumps that costs more than the pumps.

---

## 2. The architecture

Two tiers, split by **timescale**, not by capability.

```
     ┌──────────────── NODE (battery, per pump) ────────────────┐
     │  accelerometer + current transformer                     │
     │                                                          │
     │  ① run state ──── is the pump doing work at all?         │
     │        │           (if not: everything below is skipped)  │
     │        ▼                                                  │
     │  ② dry-run trip ── CUSUM ∧ floor ∧ persistence            │
     │        │           → trips the contactor locally, <1 min  │
     │        ▼                                                  │
     │  ③ gate ────────── Mahalanobis ∨ EWMA ∨ CUSUM             │
     │                    commissioned on THIS pump's baseline   │
     └──────────────────────────┬───────────────────────────────┘
                                │ escalated windows only (LoRa)
     ┌──────────────────────────▼───────────────────────────────┐
     │  GATEWAY (mains, ARM SBC, shared across a site)          │
     │  ④ classify with a prior-fitted tabular model            │
     │     commissioning a new pump = substituting the           │
     │     reference set, not retraining                         │
     └───────────────────────────────────────────────────────────┘
```

**① Run state** (`node/runstate.py`). Threshold with hysteresis and a dwell on a
load-bearing channel. `UNKNOWN` is representable, because defaulting to "running" is how
a monitor learns an idle baseline. Nothing downstream evaluates while the machine is
off, and the recursive detectors advance no state either — feeding them idle windows
drags their reference into the stopped-machine distribution.

**② Dry-run trip** (`node/trip.py`). Requires CUSUM **and** an absolute floor **and**
persistence, all three. An earlier version ORed the floor with CUSUM and tripped on
essentially every startup transient: an unprimed pump looks exactly like one that has
lost suction, for the few seconds before it fills.

**③ Gate** (`node/gates.py`). Commissioned on *each pump's own healthy baseline* —
a node is installed on a presumed-healthy pump, observes it, and thereafter escalates
what does not look like what it saw. Needs `n > 10p` healthy windows before the
Mahalanobis covariance is usable at all.

**④ Gateway** (`gateway/tabpfn_clf.py`). A prior-fitted tabular model conditions on a
reference set supplied at inference time. The KV cache is warmed once at boot.

### The duty cycle (`duty.py`) — two cadences, deliberately different

| | rate | governs |
|---|---|---|
| **commissioning** | 12 windows/runtime-hour | how many days until a node is usable (~2.9) |
| **decision** | 1/runtime-day | the false-alarm budget |

Holding these equal is what capped end-to-end recall at 0.168. Slowing both together
would push commissioning from 2.9 days to 106. They are separate because commissioning
is a one-time calibration phase — a schedule, not a capability.

---

## 3. The methodology

This is the part that generalises. Four decisions are usually left implicit, and each
of them moved our results more than the choice of model did.

### 3.1 What you hold out — the leakage ladder

Every model is evaluated at five levels of strictness on identical features:

| level | held out | verdict |
|---|---|---|
| 0 | nothing (random windows) | **invalid** — windows from one recording on both sides |
| 1 | the recording | weak |
| 2 | the physical component | good |
| 3 | the operating point | essential |
| 4 | **the whole machine** | the thesis test |

Level 0 is reported as an *artefact*, to quantify what common practice buys. A fold is
scored only if it trains on the classes it tests — a rule that disqualifies one widely
used pump dataset from cross-machine evaluation entirely, because its two motors share
no fault class.

### 3.2 How you normalise — protocol, not preprocessing

Two explicit strategies, and **the gap between them exceeds the gap between models**:

| model | per-machine (transductive) | train-pooled (inductive) |
|---|---|---|
| logistic | 0.463 | 0.663 |
| gradient-boosted | 0.421 | 0.666 |
| foundation model | 0.473 | 0.738 |

0.20–0.28 macro-F1, on 11 of 11 machines, against a 0.072 margin between models.
Replicates on an independent bearing benchmark. **Mechanism unresolved** — our
class-imbalance hypothesis is refuted by our own controlled test.

### 3.3 How often you decide — the operating point

The alarm budget is not physics. It is the operator promise (one false alarm per pump
per month) divided by the number of decisions in that month.

| cadence | FAR budget | GBDT | foundation |
|---|---|---|---|
| every 5 min | 0.00093 | 0.084 | **0.203** |
| hourly | 0.011 | **0.504** | 0.421 |
| **daily** | 0.033 | **0.781** | 0.751 |

**The operating point decides which model wins.** The foundation model leads by 2.4× at
the tightest budget and loses at every deployable one. A model comparison is not a
statement about models unless the operating point accompanies it.

### 3.4 What evidence can support a gate claim

Derived from getting it wrong twice, on our own data:

1. **Establish run state before evaluating anything.**
2. **Count commissioning in observed running windows**, not calendar time.
3. **Report the operating point with every rate.**
4. **Unlabelled operational data can falsify a gate but cannot validate one.**

---

## 4. Data

| dataset | units | labels | role |
|---|---|---|---|
| **ESPset** | 11 in-service submersible pumps | ✅ | the only cross-machine axis; every headline number |
| **Twente/4TU** | 2 motors × 4 speeds | ✅ | the only current channel; cross-operating rung |
| **Paderborn** | 11 bearings | ✅ | independent replication of the normalisation effect |
| **CIRA** | 3 industrial pumps, 1 Hz | ❌ | the only real acquisition clock; methodology only |

All CC BY 4.0. **None is redistributed** — every loader raises with download
instructions. A synthetic generator exists as the test fixture and produces nothing that
reaches the paper.

⚠️ ESPset is offshore submersibles, not irrigation pumps. Shared failure physics,
nothing else.

---

## 5. What it actually achieves

| | |
|---|---|
| Cross-machine macro-F1 (LOMO, 11 pumps) | **0.738** vs tuned GBDT 0.666, majority 0.228 |
| Recall at 1 false alarm/pump/month | **0.781** (GBDT at daily cadence) |
| End-to-end, incl. gate ceiling | **0.647** — was 0.168 at the original cadence |
| Commissioning specification | ~500 reference windows; **regresses** beyond |
| Gateway latency (measured on RK3588) | **88 ms**/window; KV cache 7.1× |
| Node battery | 1.20 yr; transmission ~1% of budget, sensing is the rest |

**Honest reading:** catching roughly two developing faults in three at one false alarm
per pump per month. A maintenance triage aid, **not a safety system** — a third of
developing faults still reach failure unflagged, and on the worst pump the gate's
ceiling alone is 0.48.

---

## 6. What is known to be wrong or untested

- **The gate's escalation rate (0.082) is measured on curated data.** Every record in
  ESPset and Twente is an acquisition a person chose to take on a running machine. The
  one attempt to measure it on continuous plant telemetry gave one usable pump.
- **Machine count binds.** 11 pumps; per-machine CIs overlap. More seeds do not help.
- **The normalisation mechanism is unknown.**
- **Persistence (k-of-n) is untested.** Needs a clock, labels, and a correctly
  commissioned baseline simultaneously; we have never had all three.
- **No rig data.** The collection path is exercisable against a simulated backend only.
- **No accelerator.** Neither the RK3588 NPU nor a Coral Edge TPU can take a model whose
  input shape varies by construction. No port attempted; stated as a constraint.

---

## 7. Running it

```bash
make test              # 263 tests
make experiment-espset # the headline numbers (needs the data downloaded)
make tables            # every results table, generated
make figures-all       # 21 real-data figures
make bench-hardware    # run this ON the gateway board
```

`results/*.json` ships with the code, so `make tables` and `make figures-all` reproduce
every number without downloading a dataset.
