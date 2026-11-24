# CP4 evaluation report

The tuned real shortlist improved from 6/18 to 10/18 top-1 matches and from 12/18 to 13/18 at Recall@3. On synthetic development, the cached tuned shortlist reached 99/140 top-1 matches. PDM alone remained ahead at 108/140. Every complete-system real variant had zero Supported claims. [Aggregate and quality-group metrics](p1_cp4_evaluation_metrics.csv)

## Evidence and denominators

| Evidence | Cohort | Execution |
|---|---|---|
| Original real | 18 SDLCD validation observations, 14 objects | Clean end-to-end Fast controller |
| Original synthetic | 200 development observations, 140 with a period and 60 explicit No Reliable Period (NRP) | Clean end-to-end Fast controller |
| Tuned real | 18 validation observations × clean, noise, crop and gap = 72 requested variants | Tuned Fast controller on 69 inputs, with three inapplicable transforms retained |
| Tuned synthetic | Same 200 development observations | Cached pre-verification reconciliation replay |

No real test observations were scored. Source manifests record date, commit, config and seed. The tuned real config hash matches the active config. CSV rows and source hashes were verified, and all stored source aggregates were regenerated from the recorded predictions. [Provenance and audit](p1_cp4_evaluation_provenance.json)

Overall accuracy uses the ±5% apparent-period tolerance and all known-period cases. Prediction coverage uses every requested observation or variant. Conditional accuracy and log-period, harmonic and catastrophic errors use known-period cases with emitted predictions. Fused Recall@3 describes the shortlist. Complete-system recall describes emitted Supported claims. An empty value with denominator zero is undefined. [Aggregate and quality-group metrics](p1_cp4_evaluation_metrics.csv)

## Original versus tuned real clean

Each cell shows top-1 / Recall@3 or claim recall / prediction coverage. Each count uses n=18.

| Configuration | Original | Tuned |
|---|---|---|
| Lomb–Scargle | 5 / 10 / 18 | 5 / 10 / 18 |
| PDM | 10 / 13 / 18 | 10 / 13 / 18 |
| CNN | 2 / 5 / 17 | 2 / 5 / 17 |
| CNN + classical shortlist | 6 / 12 / 18 | 10 / 13 / 18 |
| Complete system | 0 / 0 / 0 | 0 / 0 / 0 |

Tuned fusion reduced the median absolute log error from 0.69214 to 0.00712, harmonic errors from 9/18 to 3/18 and catastrophic errors from 6/18 to 4/18. Retention rose from 12/15 to 13/15 among upstream-hit cases. Original clean produced 14 Ambiguous verdicts and four verdict errors. Tuned clean produced 15 Ambiguous verdicts and three verdict errors. Verification completed in all 18 clean observations under each configuration. [Aggregate and quality-group metrics](p1_cp4_evaluation_metrics.csv) · [Tuned real operations](p1_cp4_tuned_real_operations.csv)

## Real stress results

The same 18 validation identities were requested in each condition, with seed 17. Noise uses half the observed brightness standard deviation, crop retains the first half of the valid baseline, and gap removes the middle 30% of the valid baseline where possible. Original labels measure recoverability after degradation. Each cell reports observed top-1 / Recall@3 or claim recall / prediction coverage, out of 18.

| Configuration | Clean | Noise | Crop | Gap |
|---|---|---|---|---|
| Lomb–Scargle | 5 / 10 / 18 | 3 / 8 / 18 | 2 / 8 / 18 | 3 / 5 / 15 |
| PDM | 10 / 13 / 18 | 10 / 13 / 18 | 11 / 14 / 18 | 10 / 12 / 15 |
| CNN | 2 / 5 / 17 | 1 / 6 / 16 | 3 / 5 / 15 | 2 / 5 / 13 |
| CNN + classical shortlist | 10 / 13 / 18 | 10 / 13 / 18 | 10 / 14 / 18 | 9 / 12 / 15 |
| Complete system | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Noise and crop each lost one clean fused top-1 hit and gained one. Compared with clean, noise increased fused catastrophic errors from 4/18 to 5/18, and crop increased them to 5/18 and harmonic errors from 3/18 to 5/18. Crop added one net shortlist hit and produced two incomplete verifications. Gap lost one clean hit to a skipped transform. On the 15 applicable pairs, clean and gap each hit 9/15, with one gain and one loss. The reported overall gap result remains 9/18. [Per-variant readouts](p1_cp4_tuned_real_readouts.csv) · [Aggregate and quality-group metrics](p1_cp4_evaluation_metrics.csv)

### Quality groups

Quality reflects each variant after degradation. The group counts and membership may shift, so these subgroup results are descriptive. Cells show fused top-1 / Recall@3.

| Variant | Suitable | Limited | Unavailable |
|---|---|---|---|
| Clean | 2/3 / 3/3 | 8/15 / 10/15 | — |
| Noise | 1/1 / 1/1 | 9/17 / 12/17 | — |
| Crop | 2/4 / 4/4 | 8/14 / 10/14 | — |
| Gap | — | 9/15 / 12/15 | 0/3 / 0/3 |

### All 72 operational dispositions

| Variant | Completed abstentions | Incomplete verification | Verdict errors | Skipped transforms |
|---|---:|---:|---:|---:|
| Clean | 15 | 0 | 3 | 0 |
| Noise | 14 | 0 | 4 | 0 |
| Crop | 11 | 2 | 5 | 0 |
| Gap | 10 | 0 | 5 | 3 |
| Total | 50 | 2 | 17 | 3 |

The **22 diagnostic variants span 12 observation IDs**. The P2 verdict diagnostics comprise 12 top-vote reason-mapping and five geometry-margin transport errors. P2 verification had two crop cases with no eligible decoy, each leaving a required harmonic comparison unfinished. The P1 gap transform skipped three variants because the proposed gap could not remove valid epochs while retaining valid epochs on both sides. None is dropped from the cohort totals. False-Supported among claims is undefined with zero claims. False-Supported per labeled requested variant is 0/72. [All exact messages and ownership](p1_cp4_tuned_real_operations.csv)

## Synthetic shortlist and calibration

Original synthetic end-to-end results include 130 Ambiguous, 20 Inconclusive and 50 verdict execution errors, with zero Supported claims. Of the 150 nonclaims with a verdict, 16 had incomplete verification. There is no tuned synthetic end-to-end result in this report. [Original synthetic metrics](p1_cp4_evaluation_metrics.csv) · [Original operational counts](p1_cp4_stage2_search.json)

| Pre-verification result | Top-1 | Recall@3 | Catastrophic | Emission coverage |
|---|---:|---:|---:|---:|
| Original fusion | 80/140 | 121/140 | 41/140 | 180/200 |
| Tuned fusion replay | 99/140 | 129/140 | 26/140 | 180/200 |
| PDM only | 108/140 | 129/140 | 21/140 | 180/200 |

Tuned synthetic fusion retained 129/136 upstream hits and emitted a shortlist on 40/60 NRP cases. The limited group had top-1 69/104 and Recall@3 93/104. The suitable group had top-1 30/36 and Recall@3 36/36. [Cached shortlist rows](p1_cp4_cached_synthetic_shortlist.csv) · [Tuning evidence](p1_cp4_stage2_search.json)

The CNN reliability plot uses the explicit NRP probability on the original clean synthetic cohort. Brier/ECE were 0.27270/0.28074 for all 200, 0.21550/0.21596 for 136 limited, and 0.39424/0.41841 for 64 suitable cases. No real or fused calibration estimate is available. [Calibration plot](p1_cp4_calibration.png) · [Reliability bin counts](p1_cp4_calibration_bins.csv)

Risk–coverage curves condition on ranked predictions with evaluable truth. The legend shows eligible n, which can be smaller than cohort n. Complete-system curves are undefined because the system emitted no claims. [Risk–coverage plot](p1_cp4_risk_coverage.png)

## Reproducibility and CP4 gate

The original-config synthetic smoke replay reproduced all ten scientific rows for one clean and one seeded-noise variant after omitting runtime fields. Its aggregate JSON matched byte for byte. The tuning harness separately reproduced the full original reconciliation metrics on both cached cohorts and reproduced selected-weight shortlist metrics. The tuned real manifest configuration and inputs validated, and its saved metrics recomputed exactly from the CSV. A fresh full tuned-controller replay remains outstanding. [Reproducibility check and digests](p1_cp4_evaluation_provenance.json) · [Tuning replay checks](p1_cp4_stage2_search.json)

The tuned real clean run covered all pilot validation observations with no unhandled pipeline crash and three returned verdict-stage errors. Full verdict delivery, tuned synthetic end-to-end metrics, CP4 app workflow, demo cache and threshold freeze remain outstanding. [Day 4 acceptance audit](../technical_doc.md#day-4-acceptance-audit)
