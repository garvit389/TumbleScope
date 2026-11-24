# Day 5 evaluation

Both development protocols use schema v5 Fast analysis with config hash `de1ae4a3a8f63619814f073ad0f254048e4bd2bb801c765961d82467343f1dfa`. Each applicable observation or variant is analyzed once. The five readouts share its inputs and configuration. A direct hit falls within ±5% of the reference apparent period. The complete system emits a period only on Supported. The real test pilot remains reserved until the human threshold freeze.

## SDLCD validation pilot

There were 18 observations from 14 objects, with 18 requested inputs per condition. Each cell gives top-1 hits / Recall@3 hits out of 18. Complete-system recall counts emitted Supported claims. [Predictions](day5_real_validation/runs.csv) · [Headline and quality-group metrics](day5_real_validation/summary.csv) · [Manifest](day5_real_validation/manifest.json) · [Risk–coverage](day5_real_validation/risk_coverage.html)

| Configuration | Clean | Added noise | Cropped baseline | Internal gap |
| --- | --- | --- | --- | --- |
| Lomb–Scargle | 5 / 10 | 3 / 8 | 2 / 8 | 3 / 5 |
| PDM | 10 / 13 | 10 / 13 | 11 / 14 | 10 / 12 |
| CNN | 2 / 5 | 1 / 6 | 3 / 5 | 2 / 5 |
| CNN + classical | 10 / 13 | 10 / 13 | 10 / 14 | 9 / 12 |
| Complete system | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

On clean data, median absolute log-period error was 0.69214 for LS (18 emitted), 0.00247 for PDM (18), 0.71925 for CNN (17), and 0.00712 for fusion (18). Their clean harmonic error counts were 9/18, 4/18, 5/17 and 3/18 respectively. Their catastrophic error counts were 6/18, 3/18, 11/17 and 4/18. Fusion retained 13/15 upstream hits. [Clean metrics and denominators](day5_real_validation/summary.csv)

All 69 analyzed variants returned Ambiguous. Two cropped observations had an empty eligible decoy pool for a required fit and remained Ambiguous with `verification_incomplete`. Three gap transforms were inapplicable and retained in the 72 requested rows. Verdict-stage errors were 0. Supported coverage and false-Supported per labelled request were both 0/72. Accuracy and false-Supported among Supported claims are undefined with zero claims. Complete-system risk–coverage and log-period error also lack emitted predictions. These variants reuse the same 18 observation identities and the weights were selected on this development cohort. [Operational rows](day5_real_validation/runs.csv)

The stress seed is 17. Noise uses half the observed brightness standard deviation. Crop retains the first half of the valid time baseline. The gap removes the middle 30% when valid epochs remain on both sides. The original reference periods measure recoverability after degradation. [Reproduction settings](day5_real_validation/manifest.json)

## Synthetic development holdout

The clean shard contains 200 observations, with 140 known-period and 60 explicitly labelled No Reliable Period (NRP). Its group IDs are disjoint from the CNN corpus. Reconciliation tuning reused this shard. Top-1 and Recall@3 use the 140 known-period rows. Emission coverage uses all 200. Error-rate denominators use emitted predictions with a known period. [Predictions](day5_synthetic_development/runs.csv) · [Headline and quality-group metrics](day5_synthetic_development/summary.csv) · [Manifest](day5_synthetic_development/manifest.json) · [Risk–coverage](day5_synthetic_development/risk_coverage.html)

| Configuration | Top-1 | Recall@3 | Emissions | Median absolute log error | Harmonic errors | Catastrophic errors |
| --- | --- | --- | --- | --- | --- | --- |
| Lomb–Scargle | 71/140 | 97/140 | 177/200 | 0.02896 | 36/137 | 49/137 |
| PDM | 108/140 | 129/140 | 180/200 | 0.00435 | 28/140 | 21/140 |
| CNN | 7/140 | 19/140 | 93/200 | 0.69823 | 15/88 | 47/88 |
| CNN + classical | 99/140 | 129/140 | 180/200 | 0.00533 | 32/140 | 26/140 |
| Complete system | 2/140 | 2/140 | 2/200 | 0.00277 | 0/2 | 0/2 |

Fusion retained 129/136 known-period cases with an upstream candidate hit. The complete system returned 178 Ambiguous, 20 Inconclusive and two Supported verdicts. Both Supported periods matched their labels within 5%. Coverage was 2/200, accuracy among claims was 2/2, and false-Supported was 0/2 per claim and 0/200 per labelled observation. The complete system abstained on all 60 NRP observations. Evidence from two claims is too sparse to estimate reliable Supported accuracy. All 21 incomplete verifications involved an empty eligible decoy pool and yielded Ambiguous with `verification_incomplete`. Verdict-stage errors were 0. [Verdict and candidate readouts](day5_synthetic_development/runs.csv)

On the clean NRP probability event, CNN Brier score was 0.27270 and ten-bin ECE was 0.28074 across all 200 observations. Risk–coverage uses eligible scored predictions with known correctness. Complete-system synthetic risk–coverage has two eligible claims. Both cohort summaries were recomputed from their saved prediction CSVs and matched the published JSON aggregates. [Reliability](day5_synthetic_development/calibration.html) · [Synthetic aggregate JSON](day5_synthetic_development/summary.json) · [Real aggregate JSON](day5_real_validation/summary.json)

## Timing

On Windows with Python 3.11.9, AMD Ryzen 7 7435HS and CPU-only Torch 2.8.0, the summed per-stage timing for the 18 clean real analyses was 20.72 s p50 and 31.26 s p95. For 200 clean synthetic analyses, it was 18.09 s p50 and 28.20 s p95. These are percentiles of controller-recorded stages during the local Day 5 work, which included concurrent jobs, with process-local OpenMP, OpenBLAS and MKL thread counts set to one. They exclude input loading and report generation. [Real timing data](day5_real_validation/runs.csv) · [Synthetic timing data](day5_synthetic_development/runs.csv)

## Verification-only evidence

The archived independent simulator evaluation fitted 50 supplied reference apparent periods and 92 eligible decoys. All 92 matched comparisons favored the supplied period within their curve. Pooled ROC-AUC across the scores was 0.95152. The saved aggregate was independently recomputed. The experiment uses oracle apparent periods and extra decoy restrictions to measure verification discrimination on those simulator variants. The five-configuration evaluation has no independent decoy false-acceptance metric. [Verification results](verification_auc.json)
