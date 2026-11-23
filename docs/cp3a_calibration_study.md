# CP3a synthetic calibration study

## MVP closeout

**CP3a implementation is ready for the MVP with provisional settings. Day 4 can proceed.** The earlier blocked decision applied to establishing scientifically defensible thresholds for wider use. The current closeout accepts the tested implementation and its known operating limits for the proof of concept.

The root Stage 2 guard covers cycle count, neighbour rejection, harmonic exemption and pass/failure consistency. The integration tests cover the real reconciliation output, the controller's shared sampling window, the shortlist handed to verification, and verdict handling when verification is incomplete. The contract and leakage tests cover the public interfaces and reference-label boundary. P2 can use the current output through the controller, verification and verdict pipeline without another P1 implementation change.

Keep all seven reconciliation settings at their current values. The prespecified calibration rule selected those values by its tie-break, and the selected configuration was evaluated once on disjoint validation seeds. The results below show why the available evidence does not justify replacing individual settings. Treat the values as provisional MVP operating points with optimality still unproven. The blocked comment in the frozen input configuration records the status at the time of the study. This closeout preserves that input file and its recorded hash.

The MVP should disclose the low recovery observed for weak and short signals, the risk of combining distinct close hypotheses into one cluster, and the risk of rejecting the lower-ranked hypothesis even when the candidates form separate clusters. The controlled harmonic and alias cases establish behavior under the tested conditions. General alias discrimination, false-merge rates and real-world recovery remain uncertain.

## Reproduction and separation

From the repository root, using the existing environment, run `python -m evaluation.cp3a_calibration_study calibrate` followed by `python -m evaluation.cp3a_calibration_study validate`. The commands refuse to overwrite the separately named JSON files. Move or archive copies outside those names before rerunning; do not overwrite the historical pilot. Calibration records seeds 51000–51029, and validation uses disjoint seeds 61000–61029. The selection file records the calibration artifact's SHA-256, base configuration and code/configuration hashes; validation checks the latter before running. The calibration and validation reports retain individual seed outcomes, configuration, scenario and estimator-to-reconciliation results summarized as metrics. They do not store raw light curves or individual candidate lists; those are reproducible from seed and source. The 216-case pilot and previous 30-seed probe remain separate.

## Design and selection

Each of 12 scenarios has 30 independently seeded realizations in each split, for 360 observations per split. They use a 32 s injected apparent period, with baseline cycles of 1.7, 2.7, 6, 8 or 12; irregular, nearly regular, gapped and sparse sampling; 40% missing values; noise SD 0.25, 0.35 or 1; amplitude 0, 0.45 or 1; sine, sawtooth, double-hump, two-frequency and noise-only waveforms. The nearby frequency is 1.8 nominal baseline resolutions from the first. Actual Lomb–Scargle, PDM, spectral-window and reconciliation implementations are run; the CNN is not exercised. No real observations, reference labels, held-out observations or test-set labels are used.

Each of the seven settings is compared with an alternative; separation and clustering also have two alternatives. Three combined policies vary phase settings, resolution settings, and cycles/width/coverage, respectively. Unchanged scoring weights and other settings come from the recorded base config. The fixed selection metric is equal-scenario mean signal retention and noise-only specificity, weighted equally. This balanced-accuracy-like choice is a methodological choice, not a contract requirement or a documented cost ratio. A 2% relative period match defines retention; both 16 s and 32 s are counted for a double hump. The first-listed existing configuration wins an exact tie. This rule does not score false candidates on signal observations, paired-harmonic preservation, duplicate/false merges, or an explicit safety constraint, so the selected policy cannot be accepted solely from its score.

Thirty seeds per scenario allow binomial Wilson intervals but do not give narrow bounds: 30/30 has a lower 95% bound of 88.6%, and 0/30 has an upper bound of 11.4%. These are conditional on the artificial scenario and assume independent seeds within it; the same seed numbers are reused across scenarios, so scenario results are paired and should not be treated as independent replicates in an overall confidence interval. No broad population accuracy or false-Supported rate is estimated.

## Results

The existing settings and eight alternatives tied at calibration sensitivity 0.7394, noise specificity 0.9667, and aggregate 0.8530; the first-listed existing settings were frozen. `bins_20` and `separation_2` gave 0.8515. The combined phase policy gave 0.7924; combined resolution and combined strict policies tied baseline. These comparisons show the phase-setting interaction in this scenario mix, but do not independently identify optimal values for the settings. The validation score for the frozen baseline was 0.8091 (sensitivity 0.7182, specificity 0.9000); no uncertainty interval is asserted for this aggregate because scenarios share seed numbers and have heterogeneous outcomes.

| Scenario | Calibration any matched truth retained / 30 | Validation / 30 | Interpretation |
|---|---:|---:|---|
| Short, 1.7 cycles | 0 | 0 | A matched candidate existed in all runs; the configured cycle gate rejects it. |
| Borderline, 2.7 cycles | 0 | 0 | No cluster matched within 2%; this is not evidence that Stage 2 rejected a matched candidate. |
| Long, 8 cycles | 30 | 30 | Easy control. |
| Weak | 7 | 2 | Matched candidates existed in exactly those counts; upstream detection and the match criterion limit recovery. |
| Gapped | 30 | 28 | Validation retains 28/30 (95% Wilson 78.7–98.2%). |
| Missing | 30 | 30 | 40% independent missing rows. |
| Sparse | 27 | 28 | Forty timestamps. |
| Nearly regular cadence | 30 | 30 | Does not establish in-band alias discrimination. |
| Sawtooth | 30 | 29 | Non-sinusoidal control. |
| Double hump | 30 | 30 | Both hypotheses were linked and shortlisted in all runs in both splits; 95% Wilson lower bound 88.6%. |
| Two nearby periods | 30 | 30 | Both matched and retained in all runs; one tested separation, not a separation curve. |
| Noise-only | 1 false shortlist | 3 false shortlists | 95% Wilson intervals 0.6–16.7% and 3.5–25.6%, respectively. |

The per-scenario JSON summaries include rejection counts and 95% intervals, as well as records for duplicate extra clusters, truth matches with evidence from two methods, and whether those methods appear together in a cluster. In the nearby case, no cluster matched both truths in either split using the 2% rule. This proxy can miss a false merge when a merged representative matches only one truth. The duplicate proxy is conditional on matching returned candidates and is not a controlled injection of labelled duplicates. No claim about general false-merge or duplicate-merge rates follows from it.

On signal scenarios, `false_shortlist` means a shortlisted period outside the 2% truth windows. It includes harmonic or alias hypotheses and estimator grid error; it is not a ground-truth false-positive or false-Supported measurement. The nearly regular cadence scenario's sampling frequency can lie outside the searched candidate-difference range, and the runner does not record whether an in-band offset actually produced an alias pair. It therefore does **not** establish alias-risk calibration. Most single-setting alternatives tie on the chosen aggregate; the study does not resolve their scientific trade-offs. Rejection counts are candidate counts, not independent observation counts.

## Controlled pair follow-up

`python -m evaluation.cp3a_controlled_pairs` creates two separate reports, `p1_cp3a_controlled_calibration.json` and `p1_cp3a_controlled_validation.json`, with seeds 71000–71029 and 81000–81029. These post-selection stress tests use measured spectral windows from repeated 12-second observation blocks every 64 seconds. They inject labelled candidate periods and supports directly into reconciliation; they do not measure estimator recovery or tune the frozen policy. For every seed the inputs for the labels “duplicate” and “distinct” are numerically identical: those labels represent competing interpretations, so their differing costs require a scientific choice rather than an algorithmic truth oracle.

In **both** splits, the paired candidates at 0.5 and 1.0 resolution were in one cluster in 30/30 cases. At 1.5 resolutions they were in separate clusters in 30/30, but the lower-scoring candidate failed the neighbour gate in 30/30. At 2.0 resolutions they were separate and both passed in 30/30. The measured in-band window-offset candidate received positive alias risk in 30/30 cases in each split; this tests an exactly injected window match, not alias calibration under noisy estimated frequencies. A 10 s/20 s harmonic pair was linked and both passed in 30/30 cases in each split. Each 30/30 proportion has a 95% Wilson lower bound of 88.6%; these controlled labels and seeded sampling patterns are not a population estimate. The per-seed data and rejection counts are in the reports. The chosen amplitude, source supports, exact offsets and period are fixed, so further sensitivity and real-estimator studies are still needed.

## Deferred research for broader scientific use

Broader scientific calibration can proceed separately from Day 4. A future study would define target outcomes and costs with P1 and cover more periods, cadences, baselines, noise and sampling regimes, including estimator-generated in-band aliases. It would measure false merging and duplicate merging across candidate widths and occupancies, quantify uncertainty in paired policy differences, and validate any newly selected policy on fresh independent seeds. The existing validation results apply to the configuration evaluated here.
