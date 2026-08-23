# Paper draft — prose sections

Drafted from the repository's own results. Every quantitative claim below is
reproduced by `make tables` into [results/paper_tables.md](results/paper_tables.md);
if a number here disagrees with that file, that file is right and this one is stale.

**§2 Related work is deliberately absent.** It is the only section with no material
in the repository, and it requires reading and verifying papers rather than
assembling evidence that already exists. It is left for the author.

---

## 1. Introduction

A smallholder irrigation pump fails in two quite different timescales, and
conflating them is the design error this work is organised around.

The failure that destroys equipment fastest is dry running. When suction is lost,
the mechanical seal loses the fluid film that both lubricates and cools it, and
the seal faces are damaged within tens of seconds. No architecture that ships a
measurement to a server and waits for a reply can protect against this; by the time
a reply arrives the seal is gone. Dry running therefore is not a classification
problem at all in our design. It is a local trip.

The failures that dominate lifetime cost are slower: bearing wear, impeller damage,
misalignment, cavitation, a loosening foot. These develop over days to months, and
they are worth diagnosing rather than merely detecting, because the maintenance
action differs by fault. Here latency is irrelevant and discrimination is
everything.

The obstacle to applying machine learning at this price point is not inference cost
but *commissioning* cost. A per-pump supervised model needs labelled examples of
each fault on each pump, which for a population of smallholder pumps costs more than
the pumps. This is the barrier we attack. We use a prior-fitted tabular foundation
model, TabPFN v2, which conditions on a reference set supplied at inference time
rather than fitting weights to it. Commissioning a new pump becomes substituting a
reference set, not retraining a model.

Our contributions are:

**C1. A two-tier architecture with a measured energy budget.** Battery-powered MCU
nodes run a continuous statistical gate and a local dry-run trip; a shared gateway
classifies escalated events. We report the gate's escalation rate measured on eleven
in-service pumps, not simulated ones, and an energy budget that inverts the
assumption the architecture began with.

**C2. Cross-machine evaluation of reference-set substitution.** On eleven in-service
submersible pumps under leave-one-machine-out, TabPFN exceeds a nested-tuned
gradient-boosted baseline, and we report the comparison at matched coverage with the
per-machine confidence intervals that qualify it.

**C4. Whether the expensive model earns its cost.** Tuned baselines, a
deployment-realistic alarm budget, and a commissioning specification derived from a
reference-set size sweep.

**C5. A protocol result that applies beyond this system.** A five-rung leakage
ladder, applied to identical data and models, quantifies how much reported accuracy
is manufactured by the choice of split — and shows that one widely used pump dataset
cannot support cross-machine evaluation at all.

**Three negative results that revise our own design.** Vibration is the wrong
primary sensor for dry running; "training-free" overstates what in-context learning
provides; and no edge accelerator we tested can execute a model whose input shape
varies by construction.

---

## 3. System design

### 3.1 Two tiers, split by timescale rather than by capability

The node is a battery-powered MCU with an accelerometer and a current transformer.
It does three things: it runs a continuous low-cost statistic over the sensor
stream, it trips the contactor locally on dry running, and it escalates anomalous
windows to the gateway. It never classifies.

The gateway is an ARM-class single-board computer shared across a site. It holds the
reference set and runs TabPFN over escalated feature windows. It never sees the
majority of the data, because the gate never sends it.

This split is not a capability partition — it is a latency partition. Dry running
must be handled in under a minute, so it terminates at the node. Slow faults tolerate
minutes of delay, so they go where the discrimination is.

### 3.2 The dry-run trip

The trip is deliberately conservative, because a false trip on an irrigation pump
means a lost watering cycle and an operator who disables the system. It requires
three conditions **simultaneously**: a CUSUM detector on the monitored statistic must
have accumulated past its threshold, the absolute level must be below a floor, and
both must persist for a dwell period.

The conjunction is load-bearing. An earlier version of this design ORed the absolute
floor with the CUSUM decision, on the reasoning that either was sufficient evidence.
That configuration trips on essentially every startup transient — a pump that has
not yet primed looks exactly like a pump that has lost suction, for the few seconds
before it fills. Requiring persistence as well as both detectors is what separates
them.

**Negative result: vibration is the wrong primary sensor here.** The design began
with a vibration-driven trip. Dry running does have a vibration signature, but it is
neither prompt nor large relative to normal operating variation. Motor current is
the better channel: losing the fluid load unloads the motor, and the current drop is
immediate, monotonic and large. We report this because the vibration-first design is
the intuitive one and it is wrong.

### 3.3 The gate, and what actually costs energy

The gate is commissioned on **each pump's own healthy baseline** — a Mahalanobis
distance in the gate feature space plus an EWMA on the level, with thresholds set
from that pump's own healthy windows. This matches the deployment model: a node is
installed on a pump that is presumed healthy, observes it, and thereafter escalates
what does not look like what it saw. Half the healthy windows commission the node;
the rest are evaluation.

Commissioning has a hard requirement: the Mahalanobis covariance needs substantially
more healthy windows than gate features to be usable at all. We require *n* > 10*p*.
Estimated from too few rows, the covariance is degenerate and the distance rejects
everything — an early version accepted at *n* > *p* + 2 and abstained on 100% of
inputs under cross-machine evaluation.

⭐ **The energy result inverts the assumption the architecture started from.** The
original design budgeted LoRa transmission at roughly 55% of the node's energy and
optimised the payload accordingly. That figure was computed for a *fixed
transmission schedule* — which the same document had already rejected in favour of
event-triggered uplink. With a working gate, measured transmission is **about 1% of
the node budget**, and continuous CUSUM sensing is essentially all of the rest. The
optimisation target is therefore a cheaper always-on front end, not a smaller
payload. We report this because it reverses a design priority, and because the
mistake — carrying forward a number computed under an assumption you have discarded
— is an easy one to repeat.

### 3.4 Gateway inference and the accelerator question

TabPFN's context is part of its input, and the reference set is fixed between
commissioning events. Caching the transformer's key/value state at commissioning
rather than re-encoding the context on every query gives **7.3×** lower inference
latency on our hardware, and reducing the ensemble from eight members to one gives a
further **5.5×**. Both are quoted here as measured, single-threaded, by
`scripts/bench_gateway_hardware.py`, which stamps the board identity into its output
so the number is attributable to a machine.

**Negative result: no edge accelerator we tested can run this model.** Both the
RK3588 NPU and the Coral Edge TPU require statically-shaped, INT8-quantised graphs
drawn from a restricted operator set. TabPFN's input shape varies by construction,
because the reference set is part of the input. This is a property of the model, not
a porting effort left undone, and it should be stated plainly rather than left for a
reader to assume the accelerator was simply never tried.

**Negative result: "training-free" overstates it.** In-context learning removes
per-pump *gradient* training. It does not remove the reference set, the
commissioning procedure, the feature pipeline, the gate thresholds fitted per pump,
or the normalisation statistics. Calling the system training-free invites a reader
to expect a zero-commissioning deployment, which is precisely the cost we measure in
§5.3 and find to be about 500 labelled windows.

---

## 4. Datasets and protocol

### 4.1 Two real datasets, deliberately complementary

No single public dataset supports the evaluation this system needs, so we use two
that fail in opposite directions.

| | Twente / 4TU | ESPset |
|---|---|---|
| Machines | 2, with **disjoint fault sets** | **11, sharing classes** |
| Leave-one-machine-out | ❌ impossible | ✅ the only source |
| Cross-operating-point | ✅ 50 / 75 / 100% speed | ❌ order-normalised |
| Current channel | ✅ the only source | ❌ vibration only |
| Waveforms | ✅ raw time series | ❌ spectra only |

ESPset is eleven in-service electrical submersible pumps with published
order-normalised velocity spectra. It is the only source of a genuine cross-machine
axis. Twente is a laboratory rig: two motors at four operating speeds, with both
vibration and motor current, and raw waveforms.

We state a scope limitation plainly rather than in a footnote: **ESPset pumps are
offshore submersibles, not smallholder irrigation pumps.** They share the failure
physics — bearings, impellers, imbalance, misalignment — and nothing else, including
duty cycle, size and installation. The cross-machine result is evidence about the
method, not about irrigation pumps specifically.

⭐ **Twente cannot support leave-one-machine-out at all.** Its two motors share only
the healthy class: every fault class present on Motor-2 is absent from Motor-4 and
vice versa. A leave-one-machine-out fold would therefore train on one label set and
test on a disjoint one, and produce a number that looks like a cross-machine result
and means nothing. Our splitter checks this condition — every fold must train on the
classes it tests — and marks folds that fail it as not interpretable rather than
scoring them. Any published cross-machine claim on Twente alone is mistaken.

**Vane counts are unobtainable for the Twente pumps, and we treat that as a result
rather than an errand.** Vane-pass frequency, its sidebands and the impeller-damage
features all require the impeller blade count Z. Three routes were tried and all
failed: the manufacturer's datasheets give impeller diameter, casting material,
spare-part number and pole count but never blade count; the manufacturer does not
publish it; and estimating Z from the healthy spectra is inconclusive on the first
accelerometer and yields no candidate at all on the second. These features therefore
degrade out rather than being computed at a guessed frequency — which matters,
because a guessed Z produces impeller-fault numbers that look entirely reasonable
and mean nothing.

### 4.2 The leakage ladder

The central protocol claim is that in machinery fault diagnosis the split, not the
model, determines most of the reported accuracy. We evaluate every model at five
levels of increasing strictness, on identical features:

0. **Random-window** — windows shuffled without regard to origin. **Invalid**:
   windows from one recording appear in both train and test.
1. **Record-wise** — no recording spans the split.
2. **Component-wise** — no physical component instance spans the split.
3. **Cross-operating** — train and test at different operating points.
4. **Leave-one-machine-out** — the held-out machine is never seen.

Level 0 is reported not as a baseline but as an artefact, to quantify what the
common practice buys. We also apply a general interpretability rule at every level:
a fold is scored only if it trains on the classes it tests.

### 4.3 Normalisation is part of the protocol, not a preprocessing detail

Per-machine normalisation is standard in this literature, and it quietly changes
what is being measured. We evaluate two explicit strategies:

- `unsupervised_per_machine` — each machine's statistics computed from its own data,
  including the held-out one. Transductive: legitimate when a node self-commissions
  on the target pump, which is our deployment model.
- `train_pooled` — statistics from training machines only, applied to the held-out
  machine. Inductive: the stricter reading.

We report both, because the choice is consequential and because **its effect
inverts between datasets**. On the eleven in-service pumps the inductive strategy is
markedly *better* — LightGBM 0.676 against 0.425, logistic regression 0.663 against
0.463 — whereas on synthetic data the ordering reverses. On real data the
normalisation choice is worth more than the choice of model, which is why it cannot
be relegated to a preprocessing sentence.

⚠️ **The two strategies must never be quoted interchangeably**, and this is easier
to get wrong than it sounds: an earlier revision of our own results write-up quoted
the inductive numbers in the cross-machine table and the transductive numbers in the
adjacent leakage table without labelling either. The leakage ladder throughout this
paper uses `unsupervised_per_machine`; the cross-machine comparison in §5.1 states
its strategy explicitly in the table caption.

An earlier version of our own code contained a bug in
exactly this area — the held-out machine was skipped during normalisation and
therefore left unnormalised — which moved logistic regression's cross-machine
macro-F1 from 0.028 to 0.937. A 33× swing from a normalisation detail is the
strongest argument we can offer that this belongs in the protocol section.

---

## 5. Results

All tables are generated from the result files by `make tables`.

### 5.1 C2 — cross-machine classification

Under leave-one-machine-out on eleven in-service pumps, reference-set substitution
outperforms a nested-tuned gradient-boosted baseline. Baselines are tuned with
**machine-grouped nested cross-validation**, so no hyperparameter is selected using
the held-out machine; our tuning code asserts this at runtime rather than trusting
it.

We report abstaining and non-abstaining TabPFN as separate models with separate
names, never as one "TabPFN" row. They are different systems, their coverage
differs, and comparing an abstaining model's accuracy against a full-coverage
baseline is not a comparison. An earlier version of this work reported them under a
single label in two different result files with two different configurations, which
is exactly the silent incomparability this discipline exists to prevent.

The comparison is reported under a stated normalisation strategy (§4.3), and the
leakage-ladder tables use the other one; the two are not interchangeable and each
table says which it uses.

⚠️ **The qualification belongs in the same breath as the result: per-machine
confidence intervals overlap.** Seed noise is roughly 0.016 while the per-machine
bootstrap intervals span roughly 0.19. Machine count, not seed count, is the binding
statistical constraint. A twelfth pump would sharpen this result far more than a
sixth seed, and no additional compute on eleven machines addresses it. We report
five seeds because a single seed is indefensible, not because five resolves this.

### 5.2 C5 — how much accuracy the split manufactures

| Dataset | Random-window (invalid) | Strictest valid split | Inflation |
|---|---|---|---|
| Synthetic stand-in | 1.000 | 0.930 (LOMO) | 1.1× |
| ESPset (11 pumps) | 0.793 | 0.425 (LOMO) | **1.9×** |
| Twente (rig) | 0.853 | 0.352 (record-wise) | **2.4×** |

**The effect grows with how real the data is.** On synthetic signals whose fault
signatures were written into the generator by hand, the split barely matters — the
signature is present in every window and any split recovers it. On in-service pumps
it nearly doubles the reported score. This ordering is itself the finding: a
practitioner validating on simulated or single-session data will not observe the
problem that will dominate their field deployment.

Figure B6 is the visual form of the same fact: colour a PCA of the feature space by
fault class, then recolour the identical projection by machine identity. The machine
structure is the stronger signal. A random-window split lets a model read machine
identity and report it as diagnosis.

### 5.3 C4 — does the expensive model earn its cost

**Tuning does not rescue the baselines.** Nested-tuned logistic regression and
LightGBM land within noise of their untuned selves, so the margin is not an
artefact of an unfair comparison. This is the first thing a reviewer asks and it is
cheaper to answer than to argue about.

⭐ **At a deployment-realistic alarm budget the gap widens.** Accuracy at a free
choice of threshold is not the operational quantity; the operational quantity is how
much fault recall survives at an alarm rate an operator will tolerate. At one false
alarm per pump per month — 1080 decisions per pump-month, so a false alarm rate of
0.00093 — recall separates far more sharply than macro-F1 does. A model that is
modestly better on average can be substantially better at the operating point that
determines whether the system is switched off.

**Commissioning specification.** Sweeping the reference-set size shows accuracy
saturating at roughly 500 labelled windows, and *regressing* at 1000. A larger
reference set is not simply better. This is the number a deployment plan needs, and
it is a concrete, falsifiable claim about what commissioning a new pump costs.

**Gate feature count is not the lever.** On real machines the gate with a compact
five-feature set achieves a 0.98 recall ceiling at 5.9% field escalation, adequately
commissioned on 11/11 pumps; widening it to a richer set *degrades* the ceiling to
0.83 at 8.2% escalation and leaves one pump under-commissioned. The commissioning
requirement scales with feature count, so more features buy discrimination that the
available healthy baseline cannot support. The gate is bounded by commissioning
length, not by feature count.

### 5.4 Cross-operating-point transfer

On Twente's Motor-2 at three speeds — the only interpretable cross-operating axis
available, since the two motors' fault sets are disjoint — LightGBM leads
(macro-F1 0.424) over non-abstaining TabPFN (0.369) and abstaining TabPFN (0.306),
with logistic regression at 0.180 against a majority floor of 0.096.

⚠️ **Abstention's sign flips between datasets**: it helps on ESPset and hurts here.
We therefore always report which variant produced a number and at what coverage.
This is not a tuning detail — it is the difference between two systems that would
otherwise share a name.

### 5.5 Detection versus severity

We have no run-to-failure data, so we make no remaining-useful-life claim.
Detection rate stratified by the severity grade recorded in the dataset is the
honest substitute: it shows whether the system detects faults while they are still
mild, which is the property RUL claims are usually invoked to support. The most
severe grades have very few examples (n = 5 and n = 6), and we report the counts
alongside the rates rather than the rates alone.

---

## 6. Limitations

**Machine count is the binding constraint.** Eleven machines is the largest
cross-machine axis available in public pump data, and it is not enough for the
per-machine intervals to separate. Every other quantity we could increase — seeds,
folds, compute — is already past the point of usefulness.

**The alarm budget bounds deployability, and it is the honest headline.** Recall at
one false alarm per pump per month is far below what a maintenance programme would
want. The system is a triage aid, not a replacement for inspection, and the
alarm-budget number rather than the macro-F1 number is the one that says so.

**ESPset is offshore submersibles.** The cross-machine result is evidence about
reference-set substitution as a method, not about irrigation pumps.

**Twente is a subset.** We extracted a portion of the 20.8 GB archive, and we pair
vibration and current bursts by index — defensible only because each folder is a
steady-state run of a single condition, but an approximation, and the `ct_only`
comparison rests on it.

**Vane counts are unobtainable**, so vane-pass and impeller-damage features degrade
out on Twente (§4.1).

**No collected rig data.** The own-rig collection path is implemented and
exercisable end to end against a simulated backend, including the abort branch, but
no real acquisition has been performed. Contribution C3 has no collected data and we
do not claim it.

**Gateway accuracy is an upper bound conditioned on escalation.** End-to-end fault
recall cannot exceed the gate's recall ceiling, and we report the two together for
that reason. A gateway result quoted without the gate ceiling overstates the system.

---

## 7. Conclusion

Reference-set substitution transfers across machines: on eleven in-service pumps
never seen during commissioning, conditioning a prior-fitted tabular model on a
substituted reference set outperforms a nested-tuned gradient-boosted baseline, and
the advantage widens at a deployment-realistic alarm budget. The commissioning cost
that follows — roughly 500 labelled windows, beyond which more is not better — is a
concrete specification rather than an aspiration.

The architectural results are mostly corrections of our own assumptions. Dry running
belongs at the node and belongs to the current channel rather than the vibration
channel. Transmission is not the node's energy problem; continuous sensing is. No
accelerator we tested can run a model whose input shape varies by construction. And
"training-free" describes the absence of gradient steps, not the absence of
commissioning.

We expect the protocol result to outlast the model result. TabPFN will be superseded.
The finding that random-window splits inflate reported macro-F1 by roughly a factor
of two on in-service data — and that the inflation is *smallest* on the simulated
data where practitioners are most likely to look for it — applies to any model
anyone puts on this problem next.
