# remediation.md — system design for the post-audit fixes

**Companion to [DESIGN.md](DESIGN.md).**

> **STATUS: executed.** All nine issues are closed. Tests 168 → 205, pyflakes 33 → 0,
> and the §6 verification passed — every published number reproduces to three
> decimals under its new label, which is what proves the rename was cosmetic rather
> than behavioural. §1.5 was resolved by **demotion**: `node/acquire.py` now carries
> an explicit "design model, not pipeline code" header rather than being wired into
> the feature path, because wiring it would change Twente feature values for no
> scientific reason.
>
> One thing the execution surfaced that the design did not anticipate: abstention's
> *sign flips between datasets* — it helps on ESPset (+0.015) and hurts on Twente
> (−0.025). That makes the naming fix more valuable than it looked here, since a
> bare "TabPFN" number was hiding a variable-sign effect, not just an ambiguity.
> Recorded in DESIGN §−2.7.

The design below is kept as written, for the record.

---

## 0. Context

The conformance audit that opened this project has been executed. The LOMO
normalization collapse, the fabricated figures, the never-run TabPFN, the trip path
that fired on every confuser — all fixed, and the project now has real results on two
real datasets. Current state:

```
src + scripts   8,442 LOC        tests   168 passing (2,133 LOC)
largest file    figures.py, 871 LOC (13% of src)
```

A fresh audit found **nine remaining issues**. None of them is "the science is
broken" — that phase is over. They are structural, and they share a single cause:

> **Scaffolding was copied into three experiment scripts, and the copies have drifted.**

Drift has already produced one genuine correctness defect (§1.9) and one latent trap
(§1.2). It is also why a bug fixed in one script stayed broken in the other two. The
central design decision in this document is therefore not "fix the nine issues" but
**remove the conditions that let them appear**.

Severity throughout is ranked by consequence, in this order:

1. **Misleads a reader** — someone draws a wrong conclusion from the repo as published
2. **Latent trap** — correct today, breaks silently for the next person
3. **Dead weight** — costs attention, no correctness risk

---

## 1. Issue register

Every issue below is reproducible with the command given, from the repo root.

### 1.1 DESIGN's own gap list is stale — SEVERITY: misleads a reader

```bash
grep -n "No hyperparameter tuning\|No repeated seeds" DESIGN.md
#  540:- No hyperparameter tuning or nested CV for any baseline, so a "TabPFN wins" result
#  543:- No repeated seeds. Every number is a single deterministic run.
```

Both are false. `src/pumpwatch/tuning.py` (202 LOC) implements nested,
machine-grouped search, and `--seeds` runs five. A reviewer reading DESIGN would
conclude the headline claim is untuned and single-seed — **precisely the two
objections it now survives**. The document undersells its own evidence.

### 1.2 Seed support exists in one script of three — SEVERITY: latent trap

```bash
for f in scripts/run_*_experiment.py scripts/run_experiment.py; do
  echo "$f  repeated=$(grep -c run_split_repeated $f)  seedaware=$(grep -c 'lambda seed=0' $f)"
done
#  run_espset_experiment.py   repeated=2  seedaware=5
#  run_twente_experiment.py   repeated=0  seedaware=0
#  run_experiment.py          repeated=0  seedaware=0
```

`run_split_repeated` now *raises* on a factory without a `seed` parameter — a guard
added deliberately, because the previous silent fallback burned 5× the compute for a
spread of exactly zero. The guard is correct. But two scripts still build zero-arg
factories, so adding `--seeds` to either fails immediately. The trap is armed and
waiting for whoever extends them.

### 1.3 `tabpfn` names different models in different result files — SEVERITY: misleads a reader

**The most serious issue in this document.** See §1.9 — listed there because it is the
one that motivates the central design change.

### 1.4 Seven genuinely dead functions — SEVERITY: dead weight

Never called from `src/`, `scripts/`, or `tests/`:

| Symbol | Module |
|---|---|
| `mutual_info_class_machine` | `audit` |
| `rpm_from_hz` | `speed` |
| `npsh_margin` | `physics` |
| `normalize_per_machine` | `splits` — back-compat alias, now unused |
| `write_cache_manifest` | `datasets/espset` |
| `battery_life_curve` | `node/energy` |
| `cavitation_band_energy` | `physics` — has 3 test calls, but unreachable from any pipeline |

`cavitation_band_energy` is the interesting one: it is *tested* but unreachable, which
is worse than untested, because green tests imply it is in service.

### 1.5 `node/acquire.py` is a well-tested island — SEVERITY: dead weight

160 LOC, 7 passing tests, imported by nothing:

```bash
grep -rn "node.acquire" src scripts     # (no output — imported nowhere in the pipeline)
grep -rn "node.acquire" tests           # tests/test_acquire.py:8
```

Written to close the "dual-rate acquisition is missing" gap. It closes it *as a model*,
not as production code — nothing in the feature pipeline decimates through it.

### 1.6 Two figures unreachable from any script — SEVERITY: dead weight

`fig_detection_by_severity` and `fig_baseline_lifecycle` are defined in
`src/pumpwatch/figures.py` and referenced only by their own `def` lines. The first has
data waiting for it — `detection_by_severity` already writes into the Twente results.
The second was written specifically to give `simulate_seasonal_drift` a consumer, and
then never called, so that function is dead by proxy.

### 1.7 33 pyflakes findings — SEVERITY: dead weight

```bash
.venv/bin/python -m pyflakes src scripts | wc -l    # 33
```

Mostly unused imports; `scripts/run_experiment.py` alone carries 11, and
`src/pumpwatch/node/trip.py` carries 4 left over from when `evaluate_trip_path` was
rewritten to build signals directly rather than through `generate_record`. One is a
real smell rather than an import: `figures.py:724` assigns `labels` and never uses it.

### 1.8 No tests for `experiment.py` or `figures.py` — SEVERITY: latent trap

```bash
for m in experiment figures; do echo "$m: $(grep -rn "\b$m\b" tests | wc -l)"; done
#  experiment: 0
#  figures: 0
```

`experiment.py` is the shared harness **every result in the project flows through**,
including `run_split_repeated`. The seed bug shipped precisely because nothing tested
it. `figures.py` is the largest file in the repo at 871 LOC.

### 1.9 ⛔ `tabpfn` is not one model — SEVERITY: misleads a reader

```bash
grep -n '"tabpfn"' scripts/run_espset_experiment.py scripts/run_twente_experiment.py
#  run_twente_experiment.py:173:  factories["tabpfn"] = lambda: CachedTabPFN(
#  run_espset_experiment.py:196:  factories["tabpfn"] = lambda seed=0: CachedTabPFN(
```

| Result file | what `tabpfn` is | coverage |
|---|---|---|
| `results_espset_both.json` | abstention **enabled** (config default) | **0.81** |
| `results_twente_real.json` | abstention **explicitly disabled** | **1.00** |

DESIGN §−2.5 quotes ESPset's **0.753** and §−2.7 quotes Twente's **0.459**, both
labelled "TabPFN". They are different configurations: one is scored on the 81% of
samples it chose to answer, the other on all of them. Any cross-dataset reading of
those two numbers is comparing unlike things.

This is the same class of silent incomparability the project has spent its whole
effort eliminating — and it arrived through copy-paste drift, not through anyone
making a modelling decision.

---

## 2. Design: a canonical model registry

**One change resolves §1.2, §1.9 and the duplication that caused both.**

### 2.1 The problem, stated structurally

Three scripts each construct their own `factories` dict. The construction is ~20 lines
of availability checks, config choices and naming. Nothing forces the three to agree,
so:

- a seed fix landed in one (§1.2)
- an abstention setting diverged in another (§1.9)
- the torch-before-LightGBM OpenMP ordering — a segfault guard — is asserted three times

`experiment.py` already extracted the *harness* (`run_split`, `run_split_repeated`,
`build_ladder`). It did not extract *what gets run*, which is where the drift lives.

### 2.2 The design

A new module `src/pumpwatch/models.py`:

```python
def build_model_zoo(
    include_tabpfn: bool = True,
    tabpfn_context_rows: int | None = 1000,
) -> dict[str, Callable[[int], object]]
```

Returning a name → **factory** mapping, where every factory takes a seed:

| Key | Model | Notes |
|---|---|---|
| `majority` | `MajorityClassifier` | deterministic; accepts seed, ignores it |
| `logistic` | `make_logistic(random_state=seed)` | deterministic under lbfgs |
| `lightgbm` | `make_lightgbm(random_state=seed)` | genuinely stochastic (subsampling) |
| `tabpfn_abstain` | `CachedTabPFN`, abstention on | **no bare `tabpfn`** |
| `tabpfn_noabstain` | `CachedTabPFN`, abstention off | the matched-coverage comparator |

Four properties, each of which kills a class of the drift that produced these issues:

1. **Seed-aware by construction.** `run_split_repeated`'s guard becomes unreachable
   from a caller using the registry — the error path stays for third-party factories.
2. **No bare `tabpfn`.** A results key names an unambiguous configuration. §1.9 cannot
   recur, because there is no name whose meaning depends on which script wrote it.
3. **Availability handled once.** The LightGBM import probe and the
   torch-before-LightGBM ordering documented in `gateway/baselines.make_lightgbm`
   (the macOS OpenMP segfault, exit 139) are asserted in one place.
4. **Supersedes `get_baselines()`**, which returns *instances* rather than factories
   and silently ignores its own `seed` argument for logistic. It should be deleted,
   not left as a second way to do the same thing.

### 2.3 Why a registry rather than just fixing the three copies

Patching each script fixes today's divergence and leaves tomorrow's mechanism intact.
The audit found two independent drifts in three copies over roughly two days of work;
the base rate is the argument. A registry makes the scripts *unable* to disagree,
which is a stronger guarantee than three files that currently happen to match.

---

## 3. Design: the remaining fixes

**§1.1 — refresh DESIGN's gap list.** Replace the two stale claims with what is
actually open: the Twente subset, the unobtainable vane count, the burst-pairing
approximation, the unmet gate commissioning, and — the one that matters most
statistically — that machine count, not seed count, is the binding constraint on the
C2 margin.

**§1.4 — delete the seven dead functions**, and their tests where the test only
exercises a function nothing calls. `cavitation_band_energy` is a judgement call: it
encodes real physics (the 1.5 kHz band-sensitivity weighting) that the generator does
not currently use. Either wire it into `synth.generate_vibration`'s cavitation branch
or delete it; keeping a tested-but-unreachable function is the one option to reject.

**§1.5 — `node/acquire.py`: wire or demote.** Wiring means the Twente loader
decimating through `decimate_signal` instead of loading at native rate; demoting means
moving it beside `node/energy.py` and `node/airtime.py` as an explicitly-labelled
*design model* — code that exists to compute and justify a hardware parameter, not to
run in the pipeline. Both are defensible; what is not defensible is leaving it
ambiguous.

**§1.6 — call the two orphan figures.** `fig_detection_by_severity` needs one call in
`run_twente_experiment.py` where the data already exists. `fig_baseline_lifecycle`
needs a `make figures` entry, which also revives `simulate_seasonal_drift`.

**§1.7 — clear pyflakes**, and add it to `make lint`, which currently only runs
`compileall` and therefore catches syntax errors but not an unused import.

**§1.8 — test `experiment.py`.** Priority is the seed guard, `build_ladder`'s
skip-a-degenerate-rung behaviour, and that `run_split` never lets a fold's test
indices reach the model's `fit`. `figures.py` needs smoke tests that each figure
renders from a minimal dict — enough to catch the `AttributeError: 'str' object has
no attribute 'parent'` class of error, which has already happened once.

---

## 4. Blast radius

The section this document exists for: what each fix touches, and what must be redone.

### 4.1 Model registry — the only fix with data consequences

| Touches | Effect |
|---|---|
| `src/pumpwatch/models.py` | new |
| all three `scripts/run_*.py` | factory blocks deleted, replaced by one call |
| `src/pumpwatch/gateway/baselines.py` | `get_baselines()` removed ✅ |
| `scripts/make_figures.py` | key parsing — see below |
| all four `results/*.json` | **stale, must be regenerated** |
| `DESIGN.md` §−2.5, §−2.7 | numbers must be relabelled by variant |

**Key parsing is the non-obvious coupling.** `make_figures.py` reads model identity out
of result *keys* in ten places (lines 58, 119–121, 134–136, 147, 159–161, 189), by
`endswith(f"__{strategy}")`, `split("__", 2)` and `startswith("majority" / "ct_only")`.
Renaming a model therefore changes which figures include it. The McNemar keys embed
model names too — `mcnemar_lightgbm_vs_tabpfn__train_pooled` — so pairwise comparisons
are renamed as well.

**Migration is smaller than it first appears.** Only `tabpfn` → `tabpfn_abstain`
changes; `tabpfn_noabstain`, `majority`, `logistic` and `lightgbm` are already correct.
The Twente results additionally need their `tabpfn` key mapped to `tabpfn_noabstain`,
because that is what the config actually was.

⭐ **Nothing about the measurements changes — only what they are called.** That is the
verification hook: a correct rename reproduces every current number under its new
label. If a number moves, the rename was not cosmetic and something else broke.

**Re-run cost:** `make experiment-espset-full` ≈ 45 min, `make experiment-twente`
≈ 2 min, `make experiment` ≈ 3 min, `make figures-all` ≈ 3 min.

### 4.2 Everything else

| Fix | Touches | Re-run needed |
|---|---|---|
| §1.1 DESIGN refresh | `DESIGN.md` | none |
| §1.4 dead code | 6 modules + their tests | none |
| §1.5 acquire | `node/acquire.py`, possibly `datasets/twente_raw.py` | only if wired |
| §1.6 orphan figures | `figures.py`, `run_twente_experiment.py`, `make_figures.py` | figures only |
| §1.7 pyflakes | ~12 modules, imports only | none |
| §1.8 harness tests | new `tests/test_experiment.py`, `tests/test_figures.py` | none |

Only §1.5, if wired, can change a number — decimation would alter the Twente feature
values. That is a reason to prefer demotion unless the change is separately justified.

---

## 5. Sequencing

Ordered by consequence-per-unit-risk, not by size.

1. **§1.1 DESIGN refresh.** Minutes, no code, and it is the document a reviewer reads.
   Highest consequence-to-cost ratio in this document.
2. **§1.8 harness tests, seed guard first.** Before the registry, so the refactor lands
   against tests rather than hope.
3. **§2 model registry**, then re-run everything and confirm the numbers reproduce
   under new labels. This is the only step with a rollback cost, so it goes after the
   tests that would catch a regression.
4. **§1.6 orphan figures** — small, and benefits from the registry already landing.
5. **§1.4 / §1.5 / §1.7 dead code and imports.** Last: touches the most files, carries
   the least risk, and is pure subtraction once nothing above depends on it.

Rationale for putting the cheap documentation fix first and the large refactor third:
§1.1 is currently *actively misleading* a reader, while §1.2 and §1.9 are misleading
only someone who cross-reads two result files. Fix what is being read now.

---

## 6. Verification

**Per-step:**
- `make test` — 168 tests must stay green throughout; the count should *rise* at step 2
- `.venv/bin/python -m pyflakes src scripts` → 0 findings after §1.7
- `make figures-all` → renders without a results file only under `--physics-only`;
  otherwise it must still refuse, which is existing behaviour worth not regressing

**For the registry specifically** — the assertion that proves it was cosmetic:

```
for every model M and strategy S in the regenerated results:
    new[f"{rename(M)}__{S}"].overall_macro_f1  ==  old[f"{M}__{S}"].overall_macro_f1
```

with the current values as the reference: ESPset `train_pooled` LightGBM 0.666,
TabPFN-abstaining 0.753 at coverage 0.81, TabPFN-no-abstain 0.738 at coverage 1.00.
Any drift means the refactor changed behaviour, not just names.

**A standing guard worth adding:** a test asserting that every model key written into
`results/*.json` is a member of the registry. That makes §1.9 permanently
unrepeatable rather than merely fixed.

---

## 7. Deliberately not changing

These are open, but they are **data and hardware limits, not remediable defects**.
They belong in DESIGN.md's gap list, not in a remediation plan:

- **C3 has no data.** Blocked on a physical rig. The collection path, seal interlock
  and simulated backend are complete and tested.
- **Twente is a 6-condition subset.** The component-wise rung stays
  `NOT INTERPRETABLE` until more severity levels per fault family are extracted.
- **Vane count is unobtainable.** Datasheet, spectra and Grundfos product literature
  all checked; `n_vanes=None` is final short of physically counting them.
- **Burst pairing** is an index-based approximation, and the real-data `ct_only`
  comparison rests on it.
- **Machine count is the binding statistical constraint.** Per-machine CIs span ~0.19
  and overlap; a twelfth pump would sharpen C2 far more than a sixth seed. No amount
  of refactoring addresses this.

---

## 8. What this document does not claim

It does not claim the nine issues affect any published number. They do not: §1.9 makes
two numbers **incomparable to each other**, but each is a correct measurement of the
configuration that produced it. The remediation is about making the repo say what it
means — not about correcting a result.
