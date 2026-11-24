# TumbleScope limitations

TumbleScope provides a traceable analysis of repeating brightness in one optical lightcurve. The Day 5 evidence supports a working scientific prototype. It shows weak CNN period recovery, very low Supported coverage, and open questions about generalization to independent real observations. [Day 5 evaluation](results/day5_evaluation.md)

## 1. What the analysis can establish

The current Fast pipeline checks observation quality, runs Lomb–Scargle (LS), phase dispersion minimization (PDM), and an experimental CNN, reconciles their candidates, and verifies the shortlist with held-out brightness fits. Streamlit displays the verdict and intermediate evidence. Reconciliation can use available classical candidates when the CNN is unavailable, with weights renormalized as specified by the controller contract. [Technical documentation](technical_doc.md) · [Pipeline contract](../CONTRACTS.md#5-pipeline-behaviour-controller-rules)

The reported quantity is the **apparent (synodic) period**, the brightness-repeat interval seen by the observer, in seconds. A physical or sidereal rotation period describes an object's rotation relative to an inertial reference frame. A symmetric body can produce several brightness repeats during one turn, and changing illumination and viewing geometry can alter the observed repetition. A **Supported** verdict means the candidate meets the configured evidence and verification gates and is compatible with the simplified model. The body's true shape, physical rotation period, and tumbling state remain undetermined. [Scientific scope](technical_doc.md#scientific-scope-and-architecture)

## 2. The CNN is experimental

The trained checkpoint runs through the shared feature builder and emits period candidates or a No Reliable Period (NRP) abstention. On both Day 5 cohorts, PDM recovered substantially more reference periods than the CNN. Top-1 counts matches within ±5%, and Recall@3 counts a match anywhere among the first three candidates. [Day 5 evaluation](results/day5_evaluation.md)

| Cohort | CNN top-1 | CNN Recall@3 | PDM top-1 | PDM Recall@3 |
| --- | ---: | ---: | ---: | ---: |
| Clean SDLCD validation | 2/18 | 5/18 | 10/18 | 13/18 |
| Synthetic development, known periods | 7/140 | 19/140 | 108/140 | 129/140 |

The current CNN has yet to demonstrate competitive period recovery. On synthetic development, the fused shortlist reached **99/140** top-1, below PDM's **108/140**, and both reached **129/140** at Recall@3. Fusion includes candidate clustering, adequacy checks, and alias weighting, so this comparison alone cannot isolate the CNN's contribution. The CNN's NRP probability had a Brier score of **0.27270** and an expected calibration error of **0.28074** on 200 clean synthetic observations. Those scores describe the NRP event and provide no calibration estimate for the final verdict. [Training and calibration artifacts](results/cnn_calibration.json) · [Comparison and denominators](results/day5_evaluation.md)

## 3. Simulation and physical-model boundaries

| Assumption or constraint | Consequence for interpretation |
| --- | --- |
| Box and faceted-cylinder geometry with diffuse reflection | The model represents only a limited set of shapes and surface responses. Solar panels, specular glints, and self-shadowing can produce features outside the fitted model family. |
| Single-axis rotation and fixed illumination and observer vectors within a simulation | The model omits changing geometry during a pass and complex motion such as torque-free tumbling. Different combinations of shape, orientation, and period can produce similar brightness curves. |
| Synthetic training curves degraded with modeled cadence, gaps, and noise | The generator varies observing conditions, but its waveform and surface diversity remain bounded by the simulator. Matching the cadence or noise profile does not establish that the resulting brightness patterns represent real debris. |
| Finite, noisy observational records | Short spans, weak modulation, missing samples, and irregular cadence can hide a repeat or favor a sampling alias. Exposure smearing is uncorrected, and correlated noise can violate the classical significance assumptions. |

Synthetic training and verification use a shared simulator engine. The verification-only cohort uses independent simulator variants, which tests generalization within that modeling framework. It leaves the synthetic-to-real gap unresolved. [Simulator and verification methods](technical_doc.md#physics-verification-and-verdicts) · [Correlated-noise control](results/p1_cp2_pdm_correlated_noise.json)

## 4. Verification, decoys, and abstention

- **Decoy availability and separation.** Decoys are alternative periods constrained by the search bounds, frequency resolution, harmonic relationships, and alias offsets. A candidate must fit the held-out data sufficiently well and beat the eligible decoys by the configured margin. Empty eligible pools left required fits incomplete in **two real and 21 synthetic analyses**. An unfinished required fit or comparison yields Ambiguous with `verification_incomplete`. A completed fit that fails the separation gate is a different outcome.
- **P versus 2P.** Symmetry can make a shorter apparent repeat and its double both plausible. TumbleScope compares required linked hypotheses on held-out brightness. A close or adverse comparison can leave the leading period unresolved even when verification completed.
- **ROC-AUC scope.** The separate synthetic verification experiment supplied the reference apparent period directly. Its pooled ROC-AUC of **0.952** compares **50 true-period scores** with **92 eligible decoy scores**. It measures fit-score discrimination in those simulator variants. End-to-end accuracy also depends on candidate discovery, ranking, alias and harmonic resolution, and the verdict gates. [Verification-only results](results/verification_auc.json) · [End-to-end evidence](results/day5_evaluation.md)

## 5. Evaluation scope, coverage, and reliability

The real validation pilot contains **18 clean SDLCD observations from 14 objects** and informed reconciliation weights. Noise, crop, and gap variants reuse those same identities, and three gap requests were inapplicable. Catalogue apparent-period references have measurement and interpretation uncertainty. Evaluation filters reserved catalogue codes and reports harmonic errors separately from direct matches. The synthetic development shard is group-disjoint from the CNN corpus, but it also informed reconciliation tuning. The **reserved 11-object real test pilot remains unevaluated**, pending threshold freeze and an approved test entry point. [Reference audit](data_audit.md) · [Evaluation cohorts and provenance](results/day5_evaluation.md)

| Cohort | Complete-system outcome | Reliability implication |
| --- | --- | --- |
| SDLCD validation and stress variants | All **69 analyzed variants** were Ambiguous. Three of 72 requests were inapplicable. | Accuracy and false-Supported rate among emitted claims are undefined because there were no claims. |
| Synthetic development | **Two Supported claims in 200 analyses**, both within tolerance. Supported coverage was **1%**. | Two correct claims provide too little evidence to establish reliable accuracy among Supported outputs. |

A reliable operating policy needs both low false-claim risk and useful coverage. Evaluations should report the number of emitted claims, errors among claims, and coverage together, with uncertainty on independent observations. A system can avoid false claims simply by abstaining. [Selective results and denominators](results/day5_evaluation.md)

## 6. Runtime and deployment

The current application runs one observation per upload through a local Streamlit interface and a CPU-oriented pipeline. Fast mode limits the shortlist and optimizer budget. Required harmonic partners and eligible decoys can add fits, so latency depends on candidate structure as well as observation length. The application offers cached examples and single-observation JSON export. Batch throughput, concurrent service behavior, and a higher-budget Thorough mode remain unevaluated. [Application and reproduction](technical_doc.md#application-and-reproduction) · [Verification settings](../configs/verification.yaml)

| Clean Day 5 cohort | Analyses | p50 | p95 |
| --- | ---: | ---: | ---: |
| SDLCD validation | 18 | 20.72 s | 31.26 s |
| Synthetic development | 200 | 18.09 s | 28.20 s |

Timing sums the controller stages on an **AMD Ryzen 7 7435HS** with CPU-only Torch. It excludes input loading and report generation, and the local workload included concurrent jobs. These figures characterize the measured setup. Local SDLCD demos require the audited files and manifest, and precomputed caches depend on their recorded environment and source identities. [Timing evidence](results/day5_evaluation.md#timing) · [Local acceptance notes](../DAY5_PROGRESS.md)

## 7. How to interpret the prototype

The strongest demonstration is an inspectable workflow that surfaces candidate periods, competing explanations, and reasons to abstain. The evidence supports using it to explore apparent photometric repeats under documented assumptions. A dependable period claim for new debris observations requires independent real evaluation, better evidence of coverage and reliability, and explicit tests of the domain gap. [Prioritized roadmap](future_work.md)
