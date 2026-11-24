# Limitations

## Scope of the evidence

The real validation pilot contains 18 observations from 14 objects. The same observations helped select reconciliation weights, and the three stress conditions reuse those identities. Treat gains and quality subgroup differences as development evidence. Quality is re-evaluated after each transform, and the suitable subgroups contain as few as one observation. The real test split remains reserved for final evaluation. [CP4 evaluation report](results/p1_cp4_evaluation_report.md) · [Data audit](data_audit.md)

The reported quantity is the apparent photometric period in seconds. Harmonics, aliasing and incomplete temporal coverage can obscure it. The physics comparison tests compatibility with a simplified model and cannot establish a true sidereal period, shape, or full rotation state. [Technical notes](technical_doc.md#cp4-evaluation-and-reconciliation)

## Model, stress, and operating limits

PDM alone still leads tuned fused synthetic top-1, 108/140 against 99/140. Both reached 129/140 at Recall@3. The tuned synthetic evidence stops at the shortlist. The experimental CNN's original synthetic NRP probability was poorly calibrated on this cohort, with Brier 0.27270 and ECE 0.28074 over 200 outcomes. The reliability curve describes that event and cohort. [Aggregate metrics](results/p1_cp4_evaluation_metrics.csv) · [Calibration](results/p1_cp4_calibration.png)

The stress conditions approximate independent measurement noise, shorter baseline, and a larger internal gap. They do not reproduce every observing disturbance. Three gap variants were impossible at the declared settings and remain in the denominator. Equal matched accuracy on applicable gap pairs does not establish general robustness. [Tuned real operations](results/p1_cp4_tuned_real_operations.csv) · [Stress results](results/p1_cp4_evaluation_report.md#real-stress-results)

Across the 72 requested real variants there were 50 completed abstentions, two incomplete verifications, 17 verdict errors and three skipped transforms. The system produced zero Supported claims, so the false-Supported rate among claims is undefined. Tuned clean real results cover the live controller, and finalist synthetic verdicts and a full saved-config tuned replay remain outstanding. [Tuned real operations](results/p1_cp4_tuned_real_operations.csv) · [Provenance](results/p1_cp4_evaluation_provenance.json)
