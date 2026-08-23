# Paper draft — prose sections

Drafted from the repository's own results. Every quantitative claim below is
reproduced by `make tables` into [results/paper_tables.md](results/paper_tables.md);
if a number here disagrees with that file, that file is right and this one is stale.

**Citations in §2 were verified against the published record while drafting** — see
the verification table at the end of that section for what was checked and what still
needs your eyes. One anchor previously recorded in `PAPER.md` ("Vieira 2026") does not
correspond to any paper I could find and has been replaced.

---

## Abstract

Smallholder irrigation pumps fail silently, and the failure that destroys equipment
fastest is fast: less than a minute of dry running ruins a mechanical seal. Per-pump
machine learning is impractical at this price point because commissioning cost
dominates hardware cost. We present a two-tier monitoring architecture in which
battery-powered MCU nodes perform continuous statistical gating and a local dry-run
trip, while a shared gateway classifies escalated events with a prior-fitted tabular
foundation model, so that **commissioning a new pump is a substitution of the
in-context reference set rather than a retraining run**.

We evaluate on two public datasets of real machines under a five-rung leakage ladder.
On **eleven in-service submersible pumps under leave-one-machine-out**, the foundation
model reaches macro-F1 **0.738 ± 0.015** against a nested-tuned gradient-boosted
baseline at **0.666**; at a deployment-realistic budget of one false alarm per pump
per month it recovers **2.4× as many faults** (20.3% against 8.4%). Accuracy saturates
at roughly **500 labelled reference windows** and regresses beyond them, giving a
concrete commissioning specification. On the target gateway — a Rockchip RK3588 —
a classification costs **88 ms**, and the two latency optimisations the design assumes
transfer from workstation to board almost exactly (7.1× against 7.4× for the
key/value cache; 5.6× against 5.5× for ensemble reduction).

We further report a protocol result that applies beyond this system: on identical
data and models, **random-window splits nearly double reported macro-F1 on
in-service pumps (1.9×) and inflate it further still on rig data (2.4×)** relative to
protocols that hold out the machine or the recording. We show that one widely used pump dataset
**cannot support cross-machine evaluation at all**, because its two machines share no
fault class.

Finally we report results that revise our own design. The gate's recall ceiling
bounds the whole system, and it is not uniform: across every deployable gate
configuration the worst pump's ceiling is at most **0.52**, so end-to-end recall on
that machine cannot exceed roughly half however good the classifier. Gate performance
is governed by which features are chosen rather than how many. Vibration is the wrong
primary sensor for dry running — motor current is. "Training-free" overstates what
in-context learning provides. And neither edge accelerator on our deployment board can
take a model whose input shape varies by construction, which is a property of
in-context learning rather than a porting effort left undone.

---

**Abstract discipline — do not reintroduce these:**

- ❌ "fully training-free" / "no gradient training at any stage"
- ❌ any claim an NPU or TPU accelerates the classifier
- ❌ dry-run as a *classified* class — it is a local trip
- ❌ "irrigation pumps" where the evidence is offshore submersibles — say which
- ✅ The two unrelated "2.4×" have been separated: the recall multiplier keeps the
  ratio form, the leakage figure now reads "nearly double … further still". Check any
  version you paste elsewhere carries the same fix.

---

## Deviations from the original proposal

This work began from a project proposal, and several of its claims did not survive
contact with the data. We state them here rather than in a footnote, because a
reviewer who discovers an unacknowledged gap distrusts everything around it, and
because three of these are findings in their own right.

**"Fully training-free" is wrong, and we no longer claim it.** In-context learning
removes *gradient* training per deployment. It does not remove the reference set, the
commissioning procedure, the per-pump gate thresholds or the normalisation statistics.
The accurate claim is **no per-deployment retraining**, which is still the property
that makes the economics work.

**Vibration is the wrong primary channel for dry running.** The proposal specified
contact-mounted accelerometers. Vibration energy *falls* when a pump loses suction,
which is the opposite of the signature a threshold detector wants, while motor current
drops sharply and unambiguously. A borewell submersible also cannot be
accelerometer-mounted at all. We added a current transformer and report a
current-only sensor profile.

**Dry running is not one of the classified faults.** It is a local trip at the node,
because it destroys a seal in under a minute and cannot wait for a round trip to a
gateway. Keeping it in the classification set would also have made rig identity a
usable feature, since dry-run examples exist only on rigs. The cross-machine evidence
covers healthy, misalignment, rubbing and unbalance, and we name that set rather than
the proposal's.

**Using each pump's own data as the reference set hurts.** The proposal assumed a pump
would be commissioned against its own normal-operation history. We tested it: on real
data, pooling *other* machines beats using the target pump's own distribution
(0.66 against 0.46 macro-F1). We report both, and treat the choice as protocol rather
than preprocessing (§4.3).

**The Coral Edge TPU was not used.** The proposal specified it as the gateway
accelerator. We did not attempt the port and make no claim to have proven it
impossible; §3.4 gives the constraint analysis. The gateway is shared and
mains-powered, and the measured 88 ms CPU inference is comfortably inside a budget set
by two to three escalations per pump per day.

**No purpose-built rig data was collected.** The proposal included a labelled dataset
of induced faults on low-cost pumps. The collection pipeline is implemented and
exercisable end to end against a simulated backend, including its abort path, but no
real acquisition was performed. We claim no rig contribution and list it as future
work.

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
provides; and the edge accelerators on our deployment hardware cannot take a model
whose input shape varies by construction without concessions that defeat the
mechanism.

---

## 2. Related work

### 2.1 Tabular foundation models and in-context learning

Prior-data fitted networks invert the usual training relationship: a transformer is
pre-trained once on millions of synthetic tabular tasks, then conditions on a
labelled reference set supplied at inference time, without gradient updates.
Hollmann et al. [1] show that the second version of this model outperforms tuned
gradient-boosted trees on datasets up to roughly ten thousand rows, which is
precisely the regime a per-pump commissioning set occupies. Our use of it is
deliberate rather than incidental: the property we need is not accuracy at small *n*
but that adapting to a new machine is a data substitution rather than a training run.

### 2.2 Machine learning for rotating-machinery and pump diagnosis

The closest precedent for our classifier is Magadán et al. [2], who apply TabPFN to
early fault classification in rotating machinery under limited data and find it
superior to conventional learners when labelled examples are scarce. Three things
separate our work from theirs. They evaluate the **first** version of the model; we
use v2, which is a different architecture with a substantially larger context. Their
machines are motors and bearings rather than pumps, so the fault taxonomy excludes
cavitation, impeller damage and dry running. And their evaluation is offline — there
is no deployment target, no energy budget and no gating tier, so the question of what
it costs to run the model in the field does not arise.

For pumps specifically, Varejão, Pellegrini and colleagues [3] provide both the
dataset we rely on for cross-machine evaluation and an open experimental framework
for it. Their framework is unusual in the field for taking evaluation protocol
seriously, and we discuss it as protocol rather than as a baseline in §2.3.

### 2.3 Evaluation protocol and leakage

Our protocol contribution sits in a small but growing literature arguing that
reported accuracy in vibration-based diagnosis is substantially an artefact of how
data are split.

Varejão et al. [3] name the phenomenon **similarity bias**: when a class is defined
from chunks of a single chopped signal and chunks of the same signal appear in both
training and test sets, test patterns become nearly indistinguishable from training
patterns. On the ESPset data they report the best model falling from an F-measure of
**0.887 to 0.733** once their sampling strategy removes it, and they develop the
argument further in a dedicated treatment [4]. Wheat et al. [5] reach a compatible
conclusion on bearing data, distinguishing *segmentation-level* leakage — non-
overlapping windows from one coherent recording split across the boundary — from
*bearing-level* leakage, where one physical component appears on both sides.

Closest to our protocol contribution is Vieira et al. [6], who survey eighteen papers
published in 2025 and find leakage persists in the majority, then propose a
leakage-free methodology built on **bearing-wise partitioning** — all data from one
physical bearing assigned exclusively to train or test. They evaluate on four
benchmarks (CWRU, Paderborn, Ottawa UORED-VAFCLS, HUST) and report that splits
carrying bearing-level leakage reach almost 100% accuracy on Paderborn, which is a
starker version of the same effect we measure.

**We state our position relative to this work precisely, because the overlap is
real.** Vieira et al.'s taxonomy and ours agree closely: their segment-wise and
repetition-wise splits are our levels 0–1, their condition-wise split is our level 3,
and their bearing-wise split — the strictest they define — is our level 2. The
deployment scenario they motivate it with is a bearing replaced during maintenance on
the same machine.

Our contribution is the rung above that. **Leave-one-machine-out holds out an entire
in-service pump**: different installation, different duty cycle, different
commissioning history, not a swapped component on a shared test rig. That is the
condition a node deployed on a customer's pump actually faces, and it is stricter
than component replacement. It is also, to our knowledge, unavailable in the bearing
benchmarks that literature uses, because their components sit on a common rig — which
is why the pump data matters here and not only as an application.

The magnitudes support treating the rung as the reportable quantity. On ESPset,
Varejão et al. report a 1.21× inflation from removing similarity bias; we measure
**1.9×** on the same data when the held-out unit is the pump rather than the
recording. Same direction, larger effect, stricter rung.

⭐ **Vieira et al. also corroborate our central limitation independently.** They find
that "the number of unique training bearings is a decisive factor for achieving
robust performance" — the same conclusion we reach about machines, arrived at on
different data with a different model family. We report that eleven pumps leaves our
per-machine intervals overlapping and that no amount of additional seeds or compute
addresses it; that this is a general property of the problem rather than a quirk of
our setup is, we think, the more useful reading.

We add a further observation: that a split is only interpretable if every fold trains
on the classes it tests — a condition that the widely used Twente dataset fails for
cross-machine evaluation, because its two motors share no fault class.

### 2.4 Edge deployment and two-tier condition monitoring

Two-tier architectures — a cheap always-on detector escalating to an expensive
classifier — are standard practice in wireless condition monitoring, and we claim no
novelty in the shape. What we contribute is a measured budget on the deployment
hardware rather than a projected one, and an honest accounting of what bounds the
system: end-to-end recall cannot exceed the gate's escalation recall, a ceiling that
is rarely reported alongside classifier accuracy and which in our case binds harder
than the classifier does.

### 2.5 Selective prediction

Abstention is well established as a way to trade coverage for accuracy. Our
contribution is negative and methodological: we find abstention's benefit **changes
sign** between datasets, helping on ESPset and hurting on cross-operating Twente. Any
comparison that reports an abstaining model's accuracy without its coverage, or that
labels abstaining and non-abstaining variants identically, is not a comparison. We
therefore report them as separate models throughout.

### 2.6 Statistical comparison

We follow Dietterich [7] in using an exact McNemar test for paired classifier
comparison on a common test set, and Demšar [8] on comparisons across multiple
datasets. We deliberately **do not** report Friedman tests or critical-difference
diagrams: with two real datasets and eleven machines, the number of independent units
is far below where those procedures have useful power, and presenting them would
imply a rigour the data do not support.

---

### Citation verification status

Checked against the published record while drafting:

| # | Reference | Status |
|---|---|---|
| [1] | Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S.B., Schirrmeister, R.T., Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature* **637**, 319–326. | ✅ verified |
| [2] | Magadán, L., Roldán-Gómez, J., Granda, J.C., Suárez, F.J. (2023). Early fault classification in rotating machinery with limited data using TabPFN. *IEEE Sensors Journal* **23**(24), 30960–. | ✅ verified — ⚠️ confirm end page and DOI |
| [3] | Varejão, I.M.S., Costa, L.G.O., Silva, L.H.P., Rodrigues, A., Ribeiro, M.P., Varejão, F.M., Oliveira-Santos, T. (2024). An open source experimental framework and public dataset for vibration-based fault diagnosis of electrical submersible pumps used on offshore oil exploration. *Knowledge-Based Systems* **289**, 111452. | ✅ verified — ⚠️ confirm author order |
| [4] | The similarity bias problem: what it is and how it impacts vibration based intelligent fault diagnosis. *Mechanical Systems and Signal Processing* (2025). | ⚠️ **volume, article number and author list still needed** |
| [5] | Wheat, L., von Mohrenschildt, M., Habibi, S., Al-Ani, D. (2024). Impact of data leakage in vibration signals used for bearing fault diagnosis. *IEEE Access* **12**, 169879–169895. | ✅ verified |
| [6] | Vieira, J.P., Bauler, V.A., Rosa, R.K., Silva, D. (2025). Towards a more realistic evaluation of machine learning models for bearing fault diagnosis. arXiv:2509.22267. Federal University of Santa Catarina. | ✅ verified — ⚠️ check for a journal version before submitting; it is marked "submitted to Elsevier" |
| [7] | Dietterich, T.G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation* **10**(7), 1895–1923. | ⚠️ standard reference, **not re-verified this session** |
| [8] | Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR* **7**, 1–30. | ⚠️ standard reference, **not re-verified this session** |

⚠️ **Correction to an earlier note in this file.** I previously wrote that the anchor
"Vieira 2026" matched no locatable paper. That was wrong: it is [6], Vieira et al.
2025, and it is the single most important related work for our protocol contribution —
close enough that §2.3 differentiates it explicitly rather than merely citing it. Do
not submit without reading it in full.

⚠️ Do not cite a PHM Society challenge dataset as a pump benchmark — none of them is
a pump.

⚠️ **The 1.21× figure attributed to [3]** is derived from their reported 0.887→0.733
F-measure drop. Confirm from the paper that those two numbers are the same model
under the two sampling strategies before quoting the ratio.

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
commissioning events. Two optimisations follow, and we measured both **on the
deployment board itself** — a Rockchip RK3588 (Orange Pi 5 Plus, 4× Cortex-A76 +
4× Cortex-A55, 16 GB), single-threaded — rather than on a development machine.

**Table 2** — gateway inference, 400-row context, 63 features, 64-window query batch.

| Configuration | RK3588 (ms/window) | Development laptop (ms/window) |
|---|---|---|
| No KV cache, 8-member ensemble | 3401 | 161 |
| No KV cache, 1 member | 629 | 29 |
| **KV cache, 8-member ensemble** | 489 | 22 |
| **KV cache, 1 member** | **88** | **3.9** |

Caching the transformer's key/value state at commissioning, rather than re-encoding
the context on every query, is worth **7.1×** on the board (7.4× on the laptop).
Reducing the ensemble from eight members to one is worth a further **5.6×** (5.5× on
the laptop). **Both ratios transfer almost exactly**, which is the substantive
finding: the design's latency reasoning was derived on a workstation and holds on
hardware roughly twenty times slower. The board is uniformly 21–22× slower than the
laptop across all four configurations, with no configuration-dependent cliff, which
indicates the workload is compute-bound rather than limited by memory bandwidth.

In the deployed configuration a classification costs **88 ms**. The gate escalates
between 2 and 3 windows per pump per day (§3.3), so inference latency is nowhere near
binding — the architecture would tolerate a gateway two orders of magnitude slower.
Warming the KV cache costs **35 s** on the board against 1.6 s on the laptop, but it
is paid once at commissioning or boot, not per query.

We report these as laptop-independent numbers because "an RK3588 gateway" was
otherwise a claim about hardware nobody had benchmarked. `scripts/bench_gateway_hardware.py`
stamps the device-tree board string, CPU cluster, RAM and thread count into its
output and names the file after the machine, so a measurement cannot be silently
attributed to the wrong one.

**The edge accelerators on our own hardware do not fit this model.** We state this
as a constraint analysis, not as an attempted port: **we did not attempt to compile
TabPFN for either accelerator**, and we make no claim to have empirically established
that it is impossible. What we can state is what each toolchain requires and where
this model conflicts with it.

Both the RK3588 NPU (via RKNN) and the Coral Edge TPU compile a graph for one fixed
input shape, from a restricted operator set, fully quantised to INT8. TabPFN's input
shape varies by construction, because the reference set is part of the input: the
tensor entering the model has shape (*n*_context + *n*_query, *n*_features). Across
four ordinary operating conditions — 200 or 500 reference windows, one or 32 queries
batched — that is four distinct input shapes:

| Condition | Input tensor |
|---|---|
| 200 reference windows, 1 query | (201, 63) |
| 200 reference windows, 32 queries | (232, 63) |
| 500 reference windows, 1 query | (501, 63) |
| 500 reference windows, 32 queries | (532, 63) |

The shape varies precisely because of the mechanism this paper is about:
commissioning by substituting a reference set changes the input.

**The shape constraint alone is not decisive**, and we should say so. Padding the
reference set to a fixed maximum would make the graph static, at the cost of paying
attention over the padding on every query — wasteful, but not disqualifying. Two
further obstacles are the substantive ones, and neither is addressed by padding:
the accelerators' supported operator sets do not cover a transformer attention stack,
so unsupported ops fall back to the CPU and the model would largely run there anyway;
and INT8 quantisation of a prior-fitted foundation model, without degrading the
calibration that our abstention mechanism depends on, is an open problem rather than
a build step.

We therefore report this as a limitation of the deployment target rather than a
result about the model, and we do not claim to have proven it impossible. It is not
an argument against the architecture: the gateway is shared and mains-powered, so CPU
inference is affordable, and the accelerator was only ever an optimisation. The
measured CPU latency in §3.4 is what the architecture actually depends on.

A fourth consideration is practical rather than technical. The Edge TPU runtime
itself is packaged for our architecture and installs cleanly (`libedgetpu1-std`,
arm64), so the accelerator is not unreachable in principle. What we observed is
narrower and we state only that: the package available to us is a Debian bullseye
build installed on an Ubuntu 22.04 host, and the Python binding layer is a
substantially older stack than the rest of our toolchain. A monitoring system
intended to run unattended for years inherits the maintenance trajectory of every
dependency it takes on, and that is worth weighing alongside the technical
obstacles — but we did not attempt an installation, so we make no claim about
whether the bindings can be made to work on a current Python.

We note for completeness that the RK3588's own NPU is present and functional on our
board (RKNPU driver v0.9.6) and is actively maintained. It is available for a
fixed-shape quantised model — just not for this one.

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
0.463. The normalisation choice is therefore worth more than the choice of model,
which is why it cannot be relegated to a preprocessing sentence.

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
outperforms a nested-tuned gradient-boosted baseline.

**Table 1** — cross-machine (LOMO, 11 folds), normalisation `train_pooled`,
mean ± s.d. over 5 seeds.

| Model | Macro-F1 | Accuracy | Coverage | Per-machine CI |
|---|---|---|---|---|
| majority | 0.228 ± 0.000 | 0.837 | 1.00 | [0.290, 0.395] |
| logistic | 0.663 ± 0.000 | 0.914 | 1.00 | [0.578, 0.787] |
| LightGBM | 0.666 ± 0.006 | 0.930 | 1.00 | [0.610, 0.779] |
| TabPFN (no abstention) | **0.738 ± 0.015** | 0.911 | 1.00 | [0.583, 0.773] |
| TabPFN (abstaining) | **0.753 ± 0.015** | 0.937 | 0.81 | [0.642, 0.810] |

At matched coverage the margin over LightGBM is +0.072, roughly 4.4× the combined
seed standard deviation. Baselines are tuned with
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
| ESPset — 11 in-service pumps | 0.793 | 0.425 (LOMO, 11 folds) | **1.9×** |
| Twente — 2-motor rig | 0.853 | 0.352 (record-wise) | **2.4×** |

LightGBM shown; the effect holds for every model (logistic 1.4×, both TabPFN
variants 1.6×). Normalisation `unsupervised_per_machine` throughout.

**On both real datasets the invalid split roughly doubles the reported score.** On
eleven in-service pumps, holding out the machine costs 0.368 macro-F1 against a
random-window split of the same data with the same model; on the rig, holding out the
recording costs 0.501. A practitioner who reports the random-window number is
reporting something close to twice what their deployment will deliver, and the
literature that has looked for this effect finds it consistently: Varejão et al. report
1.21× on this same pump data from removing similarity bias alone, and Vieira et al.
report leaked splits reaching almost 100% accuracy on a bearing benchmark.

Figure B6 is the visual form of the same fact: colour a PCA of the feature space by
fault class, then recolour the identical projection by machine identity. The machine
structure is the stronger signal. A random-window split lets a model read machine
identity and report it as diagnosis.

### 5.3 C4 — does the expensive model earn its cost

**Tuning does not rescue the baselines.** Nested-tuned logistic regression moves
from 0.663 to 0.638 and LightGBM from 0.666 to 0.664 — both within noise of their
untuned selves, and logistic slightly worse. The margin is therefore not an artefact
of an unfair comparison. This is the first thing a reviewer asks and it is cheaper to
answer than to argue about. (That tuning can *reduce* the score is expected under
machine-grouped nested selection: the inner folds are chosen to avoid leaking the
held-out machine, so they are a harsher model-selection signal than a leaky one.)

⭐ **At a deployment-realistic alarm budget the gap widens.** Accuracy at a free
choice of threshold is not the operational quantity; the operational quantity is how
much fault recall survives at an alarm rate an operator will tolerate. At one false
alarm per pump per month — 1080 decisions per pump-month, so a false alarm rate of
0.00093 — recall separates far more sharply than macro-F1 does. A model that is
modestly better on average can be substantially better at the operating point that
determines whether the system is switched off.

| Model | Recall at ≤1 false alarm / pump / month |
|---|---|
| majority | 0.000 |
| logistic | 0.032 |
| LightGBM | 0.084 |
| **TabPFN (abstaining)** | **0.203** |

TabPFN recovers **2.4× as many faults as the tuned gradient-boosted baseline** at the
same alarm budget — a wider separation than macro-F1 suggests, and the operationally
meaningful one. All pairwise differences are significant under an exact McNemar test.

**Commissioning specification.** Sweeping the reference-set size shows accuracy
saturating at roughly 500 labelled windows (macro-F1 0.739 at 0.77 s per query,
against 0.712 at 250 and 0.672 at 100) and *regressing* at 1000 (0.719, 1.22 s). A
larger reference set is not simply better: past saturation it costs latency and gives
back accuracy. This is the number a deployment plan needs, and
it is a concrete, falsifiable claim about what commissioning a new pump costs.

**Gate performance is dominated by feature *choice*, not feature count.**

We initially read a comparison between a five-feature and a seven-feature gate — 0.98
against 0.83 recall ceiling — as evidence that the gate is bounded by commissioning
length rather than feature count, since the commissioning requirement scales as
*n* > 10*p*. That comparison was confounded and we report the correction, because it
is the more useful result. The five-feature set was ESPset's own published feature
columns, selected by domain experts for this fleet; the seven-feature set is what our
extractor computes generically from a spectrum. The two differ in provenance as well
as in size, and provenance turned out to be what mattered.

Holding provenance fixed and varying count — every one of the 127 subsets of the seven
deployable order-spectrum features, evaluated per machine — gives:

| Gate features *k* | Subsets | Best ceiling | Median ceiling | Best worst-pump | Commissioned |
|---|---|---|---|---|---|
| 2 | 21 | 0.838 | 0.467 | 0.387 | 11/11 |
| 3 | 35 | 0.868 | 0.506 | 0.440 | 11/11 |
| 4 | 35 | 0.872 | 0.792 | 0.513 | 11/11 |
| 5 | 21 | 0.873 | 0.836 | 0.520 | 11/11 |
| 6 | 7 | 0.868 | 0.858 | 0.480 | 11/11 |
| 7 | 1 | 0.865 | 0.865 | 0.480 | 10/11 |

**The best achievable ceiling is flat from three features upward** (0.865–0.873).
Adding features neither helps nor hurts what a well-chosen gate can do. The median
rises with *k* only because the number of ways to choose badly falls — at *k* = 7
there is a single subset and best, median and worst coincide. The spread at small *k*
is the real signal: at *k* = 2 the best subset reaches 0.838 while the median reaches
0.467, so **which** two features are chosen matters more than any decision about how
many to use.

Commissioning length does bind, but only at the top of this range: seven features is
the first size at which a pump fails the *n* > 10*p* requirement (10/11 rather than
11/11). That is a real constraint on gate width, just a much weaker one than we first
claimed.

⚠️ **Our generic features are the weaker ones, and we should say so.** No subset of
the features our extractor computes reaches the 0.98 ceiling that ESPset's five
published columns achieve. The published set was designed by people who knew these
machines; ours was designed from the physics of centrifugal pumps in general. A
factor-of-eight difference in residual miss rate (0.02 against 0.13–0.16) is a
strong argument that gate feature design deserves per-fleet attention, and a caution
against assuming a generic extractor transfers.

⚠️ **The gate is weak on some pumps under every deployable configuration.** The best
worst-machine ceiling across all 127 subsets is 0.52. Since end-to-end recall cannot
exceed the gate's, the two-tier system as configured cannot exceed roughly 50% recall
on its worst pump regardless of the gateway classifier. We report the worst machine
alongside the mean, and the fault-count-pooled figure alongside both, because
machines contributed between 13 and 162 faulty windows each and the three differ by
more than our seed noise.

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
The binding figure is the *worst* machine's ceiling, not the mean: with the wide gate
feature set that is 0.48, meaning the two-tier system as a whole cannot exceed 48%
recall on that pump however good the gateway classifier is.

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
channel. Transmission is not the node's energy problem; continuous sensing is. The
edge accelerators we provisioned for the gateway turned out not to fit the model we
chose, and CPU inference on a shared mains-powered board is sufficient anyway. And
"training-free" describes the absence of gradient steps, not the absence of
commissioning.

### Data and code availability

Neither dataset is redistributed here. ESPset is available from Mendeley Data under
CC BY 4.0 (DOI 10.17632/m268jsw339.3, **version 3** — not the `.1` some prior work
cites), and the Twente/4TU pump dataset from 4TU.ResearchData under CC BY 4.0
(DOI 10.4121/2b61183e-c14f-4131-829b-cc4822c369d0). Every loader in our code raises
with download instructions rather than substituting synthetic data.

Source code, the evaluation harness and the result files behind every table and figure
in this paper are available at ⟨ARTIFACT DOI — see RELEASE.md⟩ under the MIT licence.
Because the result files are included, `make tables` and `make figures-all` reproduce
every number reported here without re-downloading either dataset. This work is Built
with PriorLabs-TabPFN.

---

We expect the protocol result to outlast the model result. TabPFN will be superseded.
The finding that random-window splits inflate reported macro-F1 by roughly a factor
of two on real machines applies to any model anyone puts on this problem next.
