# ⚠️ SYNTHETIC — not evidence about real pumps

Every plot in this directory is drawn from a generator whose fault signatures were
written in by hand, or from a design model. They exist to verify that the feature
pipeline and the splits recover signatures **known to be present**, and to illustrate
the physics and the trip logic.

**No claim in the paper may cite a number from this directory as a result.**

The one place synthetic data appears in the paper is the leakage table, as a
deliberate contrast: inflation is *smallest* here (1.1×) and largest on real machines
(1.9× and 2.4×). That ordering is the finding — a practitioner validating on
simulated data will not see the problem that dominates their field deployment.

Real results:

- `../espset/` — 11 in-service submersible pumps. The cross-machine evidence.
- `../twente/` — 2-motor laboratory rig. Current channel and severity grading.
- `../summary/` — cross-dataset comparisons.
