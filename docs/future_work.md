# TumbleScope technical roadmap

The next priority is to establish whether TumbleScope can make useful apparent-period claims on independent real observations. The measured baseline is a strong PDM comparator, an experimental CNN with weak recovery, and a conservative verdict policy with sparse Supported claims. The phases below describe proposed work and the evidence needed to advance. [Day 5 evaluation](results/day5_evaluation.md) · [Current limitations](limitations.md)

| Phase | Priority | Condition for advancing |
| --- | --- | --- |
| 1. Establish independent evidence | Submission acceptance, a frozen protocol, and an object-disjoint pilot | Manual workflow evidence and a reproducible, one-time evaluation of the reserved pilot |
| 2. Improve scientific capability | Diagnose CNN errors, strengthen verification, and test more representative models and data | Paired comparisons on independent inputs that explain changes in recovery, risk, and coverage |
| 3. Extend engineering capacity | Profile Fast mode, evaluate Thorough mode, and consider batch, API, and export | Measured runtime and resource behavior with preserved scientific outputs and error handling |

## Phase 1 · Establish scientific credibility

### Complete submission acceptance

Rehearse the precomputed synthetic and local SDLCD demos, then run a real observation through live Upload, from preview to verdict and JSON export. Record the input and configuration, check the reason codes and candidate details against the exported result, and keep the synthetic cache available as the demo fallback. This establishes that the submitted workflows behave as documented. [Manual acceptance checklist](technical_doc.md#manual-acceptance)

### Freeze the protocol and evaluate the reserved pilot

Use development evidence to review the provisional thresholds, reconciliation policy, decoy rules, and abstention criteria. Record the chosen settings, checkpoint identity, split, and method for counting coverage and failures. Implement or approve the test entry point before accessing the **reserved 11-object pilot**. Run the frozen configuration once and archive the outputs separately from the Day 5 development results.

Report counts and uncertainty by object, preserving the grouping of repeated observations and reporting cohorts with zero claims explicitly. Show classical baselines, fused shortlist recovery, verification completion, Supported coverage, and errors among claims with their denominators. Retain the results even if performance is weak. Any later tuning belongs on development data, with a fresh object-disjoint test cohort for assessing a revised policy. [Current split and results](results/day5_evaluation.md)

## Phase 2 · Improve scientific capability

### Diagnose the CNN before changing capacity

Audit simulator labels against the clean repeat criterion and inspect examples of half- and double-period confusion, wrong-bin predictions, and NRP abstention. Verify that training and inference still use the same preprocessing and shared feature builder, including masking, cadence channels, windowing, and period bins. Examine training curves, per-bin and per-quality prediction distributions, calibration, and errors on held-out synthetic groups.

Use those diagnostics to decide whether to test architecture, target encoding, loss, or training choices. Compare them through controlled ablations under a fixed protocol. More epochs or a larger corpus are hypotheses to test when the diagnostics support them. Compare LS, PDM, CNN, and fusion on the same object-disjoint observations, including cases uniquely recovered by each method. Keep the CNN **experimental** until it demonstrates repeatable predictive value, either through competitive standalone recovery or a measurable contribution to the fused shortlist with an explicit error trade-off. [CNN design](technical_doc.md#synthetic-generation-and-training) · [Current benchmark](results/day5_evaluation.md)

### Improve verification and calibration

Audit empty decoy pools by observation span, alias structure, and harmonic links. Prototype a richer pool with predeclared eligibility checks and comparable fit budgets. Report how many controls are available and how close they are to a candidate, alongside verification completion, separation failures, and false acceptance on labelled controls. Compare against the frozen verifier on the same development inputs.

Build controlled P/2P cases with varying brightness asymmetry, noise, gaps, and usable cycles. Evaluate favorable, adverse, and unresolved comparisons so that improved completeness does not mask mistaken preferences. Use separate development data to study quality-stratified calibration and coverage versus risk, with counts and uncertainty for sparse groups. Evaluate the CNN's NRP calibration separately from any proposed decision-level reliability measure. A change earns further validation when it improves useful coverage or discrimination and quantifies the accompanying change in false-claim risk. [Current verification and verdict rules](technical_doc.md#physics-verification-and-verdicts) · [Decoy and coverage evidence](results/day5_evaluation.md)

### Expand the evidence base

Grow the real cohort around independently identified objects and observing sessions, with variation in cadence, baseline, signal quality, and origin of the reference period. Assign object-level splits before tuning and report per-object uncertainty. Use the same observations for paired comparisons with the classical baselines.

Expand synthetic data by varying waveform families, surface responses, sampling profiles, realistic noise, and missingness. Hold out generator families or observing regimes so that validation challenges the assumptions used in training. Use controlled synthetic cases to localize failure modes, and use independent real observations to assess transfer. [SDLCD data audit](data_audit.md) · [Development limitations](limitations.md#5-evaluation-scope-coverage-and-reliability)

### Test physical extensions incrementally

Start with more realistic geometry, such as a body with solar panels, then isolate the effects of specular reflection and changing illumination and viewing vectors. Evaluate each addition against the current diffuse baseline with controls and held-out lightcurves. Track whether the added parameters improve fit discrimination and harmonic decisions, and record when physically distinct models remain observationally indistinguishable.

Study SGP4 propagation using suitable CelesTrak orbital elements to inform time-varying observation geometry. Assess orbit and geometry errors separately from brightness-model errors. Explore torque-free tumbling once the geometry and reflectance models have independent checks. Stable-spin versus tumbling classification would need defensible state labels or suitable independent observations, explicit confounder controls, and an abstention option. Its success should be evaluated separately from apparent-period recovery. [Current physical model](technical_doc.md#physics-verification-and-verdicts)

## Phase 3 · Extend engineering capacity

These extensions depend on a stable analysis contract and a credible evaluation baseline.

| Proposed extension | Technical work and evidence |
| --- | --- |
| Performance and deployment | Profile preprocessing, estimators, fitting, and serialization by observation size and candidate count. Measure CPU latency, memory, and timeout behavior under realistic workloads. Any caching or parallel execution should preserve candidate order, verdict gates, and reproducibility within documented tolerances. |
| Thorough mode | Test a larger, explicitly bounded search budget on matched inputs. Report added runtime, verification completion, harmonic discrimination, and changes in coverage and error. Keep separate results for Fast and Thorough. |
| Batch processing and FastAPI | Begin with batch execution that records per-observation failures and configuration provenance. Add a FastAPI layer after input validation, timeouts, resource limits, and concurrent execution are measured. Check parity with the existing analysis and export path. |
| ONNX and CPU inference export | Validate shared preprocessing, missing-value masks, and variable window handling. Compare exported probabilities, candidate ranks, NRP decisions, latency, and memory against the PyTorch CPU checkpoint before using the export in evaluation or deployment. |

This order keeps work on model capability and deployment dependent on measured scientific value. The proposed extensions build on the original project roadmap and remain future work. [Original development plan](../TumbleScope%20-%205-Day%20Development%20Plan%20and%20Two-Person%20Work%20Division.md#116-future-development-roadmap-short-and-realistic) · [Current application and reproduction](technical_doc.md#application-and-reproduction)
