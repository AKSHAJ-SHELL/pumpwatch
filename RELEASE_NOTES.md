Artifact accompanying the pump condition-monitoring paper.

A two-tier architecture for centrifugal pump fault monitoring: battery-powered MCU
nodes run continuous statistical gating and a local dry-run trip, while a shared
gateway classifies escalated events with a prior-fitted tabular foundation model — so
commissioning a new pump substitutes an in-context reference set rather than
retraining a model.

**Contents**

- The five-rung leakage ladder (random-window through leave-one-machine-out) and the
  machine-grouped nested tuning that makes the baselines fair
- Evaluation on two public CC BY 4.0 datasets: ESPset (11 in-service submersible
  pumps) and the Twente/4TU rig
- Gateway latency measured on the deployment board, a Rockchip RK3588
- 232 tests; `make tables` and `make figures-all` regenerate every table and figure in
  the paper from the tracked results

**Not included:** any dataset file. Every loader raises with download instructions for
its public source.

Built with PriorLabs-TabPFN.
