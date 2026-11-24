# TumbleScope technical documentation

## Scientific scope and architecture

An optical lightcurve is a time series of measured brightness. The apparent (synodic) period is the interval over which the brightness pattern repeats in the observer's geometry. A physical rotation period instead describes the time to complete a turn about an axis. A symmetric object can repeat its brightness pattern twice per turn, and changing illumination and viewing geometry can alter a measured repeat interval. The analysis returns apparent periods in seconds. The fitted surface and motion give a model for comparison with the lightcurve. Physical shape, true rotation and tumbling state remain undetermined.

`pipeline/controller.py::analyze` takes a validated `Observation` and the injected configuration. It prepares the observation, applies Stage 1, runs three source-labelled estimators, reconciles their candidates with Stage 2, and verifies the required hypotheses before rendering a verdict. `core/schemas.py` defines the shared dataclasses and validates their scopes. Schema v5 serialization preserves arrays, missing values, execution errors and separate P/2P comparisons. `configs/period_bounds.yaml` supplies the shared bounds of 2.912–445.881 s. [Interfaces and invariants](../CONTRACTS.md) · [Controller](../pipeline/controller.py) · [Period bounds](../configs/period_bounds.yaml)

### Data flow and failure boundaries

| Step | Input | Output and consumer |
| --- | --- | --- |
| Ingestion and preparation | File content and observation metadata | Aligned observations and prepared samples for Stage 1 and estimators |
| Stage 1 | Prepared brightness and valid times | Quality class, metrics and warnings. Insufficient observations yield Inconclusive and stop estimation. |
| Candidate search | Eligible prepared observations and shared period bounds | Three method-labelled candidate lists. The timestamp window provides alias offsets for reconciliation and verification. |
| Reconciliation and Stage 2 | Available candidates, baseline and window evidence | Ranked clusters with evidence by method, harmonic links and candidate adequacy. The Fast shortlist seeds verification. |
| Verification | Shortlist, required partners, samples and fit settings | Held-out errors, decoy separation, harmonic comparisons and a completeness flag |
| Verdict and export | Quality, method-labelled estimator evidence, clusters and verification | Verdict with evidence summary and reasons. Streamlit displays and serializes the controller's result. |

The structured statuses distinguish an estimator that finished with no candidate, one that was skipped, and one that failed. Missing verification on a required shortlist prevents a Supported claim. Technical failures preserve usable upstream evidence. [Pipeline rules](../CONTRACTS.md#5-pipeline-behaviour-controller-rules)

## Ingestion, preparation and Stage 1

- `ingestion.load_observation` reads the audited SDLCD six-column format using elapsed time, exposure, instrumental magnitude and magnitude error. The adapter ignores comment metadata for measurement extraction. Generic CSV and whitespace TXT require a labelled time and brightness column. The generic reader supports seconds, JD or MJD converted to elapsed seconds, and optional uncertainty. Bare `time` means seconds. The app accepts `.csv` and `.txt` uploads. [SDLCD audit](data_audit.md) · [Generic parser](../ingestion/generic.py)
- `preprocessing.prepare.prepare` sorts aligned float64 arrays and zeroes the initial time. It retains and flags duplicate timestamps, tracks original row indices, and flags both edges of sampling gaps and high-brightness outliers. It reverses magnitudes and normalizes with the median and MAD. Magnitudes and positive linear flux become relative flux. Missing or invalid brightness is represented by a false mask and NaN in prepared brightness arrays. The baseline and reciprocal frequency resolution use valid timestamps. [Preparation](../preprocessing/prepare.py)
- `preprocessing.quality.assess_stage1` uses valid count and baseline to detect insufficient observations. It flags large gaps, missing data and weak amplitude relative to a robust successive-difference noise proxy. In the checked-in profile, fewer than 21 valid rows or a nonpositive valid time span makes the observation insufficient. When that gate passes, a gap over 25% of the baseline, more than 20% missing brightness, or amplitude-to-noise below 3 produces a Limited assessment. The amplitude is half the 5th-to-95th percentile brightness span. Limited observations still proceed and carry warnings. Cadence is measured from the observation and exposure smearing remains uncorrected. [Stage 1 code](../preprocessing/quality.py) · [Configuration](../configs/preprocess.yaml)

## Estimators

**Lomb–Scargle.** `classical_analysis/lomb_scargle.py` uses Astropy Lomb–Scargle on an oversampled frequency grid with a cadence-based upper bound. The default is one sinusoidal term. Local maxima pass a significance filter and a peak-separation check before being converted to apparent periods and ranked by power. The returned periodogram is downsampled for presentation.

**Phase Dispersion Minimization.** `classical_analysis/pdm.py` uses the shared trial grid and computes the ratio of within-bin to total variance over populated phase bins. A variance-ratio test with a trial correction filters peaks. PDM ranks by `1 - theta`. Each classical estimator uses the strongest retained peak as the normalization reference, carries its frequency width and preserves a successful empty search. The spectral window uses valid timestamps to propose cadence alias offsets. [Classical settings](../configs/classical.yaml) · [CP2 estimator measurements](#cp2-classical-analysis-backend)

**Experimental CNN.** `ml/features.py::build_cnn_windows` supplies the same encoding during training and inference. It forms overlapping 1,024-position windows with brightness, validity and log time-delta channels. It right-pads short series, aligns the last long window to the tail, caps the selected windows and filters low-valid-count windows. The brightness channel encoding uses the mask, and the prepared observation retains missing values. `ml/model.py::PeriodCNN` uses six dilated residual blocks with batch normalization and ReLU, global average pooling and a 129-class head. The checked-in configuration uses a convolution kernel of 7 with dilations 1, 2, 4, 8, 16 and 32. Windows advance by 512 positions. The outputs are 128 logarithmic period bins and one No Reliable Period (NRP) class. `ml/infer.py::run_cnn` checks checkpoint SHA-256, bounds, feature and model settings, and calibration identity. It combines temperature-scaled window probabilities with valid-count weights, ranks period peaks and reports NRP abstention, along with window disagreement. [CNN settings](../configs/cnn.yaml) · [Calibration evidence](results/cnn_calibration.json)

## Synthetic generation and training

`ml/dataset/generate.py` samples simulated bodies and geometry. The shared box/cylinder simulator supplies clean curves. `ml/dataset/labels.py` checks repeat shifts in the clean waveform for apparent-period labels, then generation applies observation degradation such as cadence variation, gaps and noise. The NRP categories cover nonperiodic, multi-frequency and under-covered examples. Corpus generation stores observation-group provenance and uses the shared window builder in `ml/dataset/corpus.py`. Training uses soft period targets, tracks validation metrics by group, and exports the final-epoch checkpoint. The local CP2 corpus has 3,500 train, 750 validation and 750 test groups. The training bundle contains the train and validation shards. Calibration fits a temperature on the selection subset and chooses an NRP threshold. A separate subset checks calibration. Inference applies the saved temperature before pooling window probabilities and abstains when the pooled NRP probability reaches the threshold. The inference artifact remains experimental. [Training report](results/cnn_training_log.json) · [Calibration report](results/cnn_calibration.json) · [Model configuration](../configs/cnn.yaml)

SDLCD observations inform cadence and noise profiles. SDLCD reference periods stay in reference ingestion and evaluation. Offline simulator labels are distinct from these references.

## Reconciliation and Stage 2

`reconciliation/reconcile.py::reconcile` clusters candidate frequencies within a tolerance tied to observation resolution. It preserves distinct hypotheses when their harmonic relationships differ and keeps one period and normalized support per estimator within each cluster. It ranks clusters by weighted within-method supports, normalized over available estimators, with a penalty for alias risk. The current validation-selected weights are 0.05 CNN, 0.15 LS and 0.80 PDM, with alias penalty 0.30. The window-based alias check looks for frequency offsets to stronger candidates. Harmonic relationships track competing period ratios. [Reconciliation settings](../configs/reconcile.yaml) · [Development selection](results/p1_cp4_tuning_provenance.md)

Stage 2 evaluates cycles observed, period uncertainty from peak width, phase-bin coverage and separation from higher-scored neighbors. It exempts harmonic partners from the neighbor-merging criterion. The Fast controller shortlists up to three passing clusters. It verifies all shortlisted IDs and their direct passing P/2P partners, even when these partners are outside the shortlist. [Shortlist and pair rules](../CONTRACTS.md#55-shortlist-and-harmonic-partners)

## Physics verification and verdicts

`physics/simulator.py::simulate_lightcurve` rotates box or cylinder facet normals about one fixed axis. The relative flux sums facet area and reflectivity weighted by the positive projected illumination and viewing cosines. `physics/fit.py` holds each candidate period fixed while searching aspect, axis orientation, phase and Sun–observer angle for uniform-reflectivity families.

`physics/verify.py::verify_candidates` divides valid samples into contiguous time blocks and keeps equal timestamps together. Within each fold it searches the nonlinear parameters on training blocks, using seeded coarse search and differential evolution. It solves brightness scale and offset on the same training blocks and freezes them for the held-out score. The score is RMS error divided by robust observed scale. It uses relative flux where available and normalized brightness otherwise. Decoy periods come from configured ratios and must pass frequency-resolution, bound, harmonic and alias checks. Decoys use the same folds, optimizer and budget as the candidate. The candidate must pass the held-out error gate and beat every eligible decoy by the configured margin. The provisional configuration uses three time blocks, period ratios of 1.37 and 0.61, at least one eligible decoy, and a 30-second timeout per candidate. Its normalized held-out error ceiling is 0.85 and its minimum decoy separation is 0.10. [Verification implementation](../physics/verify.py) · [Verification settings](../configs/verification.yaml)

Each required P/2P pair stores both errors and their signed difference, with positive values favoring P. The configured margin controls close comparisons. Odd/even residual diagnostics help inspection. Diagnostic pairs also record when a partner failed Stage 2. Missing eligible decoys, fit failures, timeouts or missing required comparisons leave verification incomplete. That state yields **Ambiguous** with `verification_incomplete`.

`verdict/decide.py::decide` applies the canonical reason codes. It selects the highest-ranked verified passing candidate and checks top estimator agreement, competing non-harmonic votes, alias risk, harmonic preference, geometry uncertainty and the quality policy before emitting **Supported**. Competing evidence or verification incompatibility yields **Ambiguous**. Insufficient quality, unavailable evidence, a successful empty search or rejection of all candidates can yield **Inconclusive**. The controller retains valid upstream outputs when stages fail. Execution failures and scientific verdicts have distinct UI states. [Reason-code mapping](../CONTRACTS.md#6-reason-code-mapping-what-each-code-means-operationally) · [Verdict settings](../configs/verdict.yaml)

### Interpreting the reason codes

| Code | Operational meaning |
| --- | --- |
| `low_quality` | Stage 1 is insufficient, or limited quality accompanies a verdict other than Supported. |
| `no_periodic_signal` | At least one estimator completed and all candidate lists were empty. |
| `conflicting_methods` | Top estimator candidates fall in different non-harmonic clusters. |
| `harmonic_ambiguity` | Verification completed and a linked harmonic remains unresolved in the lead's favor. |
| `alias_ambiguity` | The lead does not meet the configured alias-risk gate. |
| `candidate_inadequate` | Estimators are unavailable, candidates fail Stage 2, or the leading cluster has insufficient estimator agreement. |
| `model_mismatch` | Verification completed and no required candidate passed both the held-out-error and decoy-separation gates. |
| `geometry_uncertainty` | At least two fitted geometry families have held-out errors closer than the configured margin. |
| `verification_incomplete` | A required verification result or comparison is missing, failed, or otherwise incomplete. |

A verdict can include several applicable reason codes. A Supported claim carries a period and cluster ID and has no reason codes. [Decision contract](../CONTRACTS.md#6-reason-code-mapping-what-each-code-means-operationally)

## Evaluation design and evidence

`evaluation.run_eval` analyzes each available observation or stress variant once with the Fast controller. The five readouts from that result are LS, PDM, CNN, the CNN/classical reconciled shortlist and complete system. Only a Supported verdict emits a complete-system period. Accuracy at ±5% and Recall@3 use the known-period cases. Conditional log-period, harmonic and catastrophic error rates use emitted predictions with a reference. Candidate retention uses cases with an upstream hit. Coverage uses all requested observations or variants. False-Supported per claim and accuracy among Supported require claims. Risk–coverage ranks predictions with evaluable truth by the corresponding estimator or reconciliation score. Brier and equal-width ECE in the system evaluation describe the CNN's binary NRP event on the clean synthetic cohort. [Evaluation implementation](../evaluation/run_eval.py) · [Metrics](../evaluation/metrics.py)

The real development split contains 18 observations from 14 SDLCD objects. Its noise, crop and internal-gap variants use the original reference labels to measure recoverability. The synthetic development set has 200 unique groups disjoint from the CNN train, validation and test group IDs. It was reused for reconciliation tuning. There are 140 apparent-period labels and 60 explicit NRP examples. The object-disjoint real test pilot contains 11 observations from 11 objects. It remains reserved until the human threshold freeze and approval of a test entry point. [Day 5 metrics and figures](results/day5_evaluation.md) · [Historical CP4 results](results/p1_cp4_evaluation_report.md) · [CP4 tuning provenance](results/p1_cp4_tuning_provenance.md)

### Five configurations on clean data

Each cell reports direct hits within ±5% or matching candidates in the first three positions. The synthetic count excludes the 60 NRP observations from period recovery denominators. The tuning reuse makes these development comparisons.

| Configuration | SDLCD top-1 | SDLCD Recall@3 | Synthetic top-1 | Synthetic Recall@3 |
| --- | ---: | ---: | ---: | ---: |
| Lomb–Scargle | 5/18 | 10/18 | 71/140 | 97/140 |
| PDM | 10/18 | 13/18 | 108/140 | 129/140 |
| CNN | 2/18 | 5/18 | 7/140 | 19/140 |
| CNN + classical | 10/18 | 13/18 | 99/140 | 129/140 |
| Complete system | 0/18 | 0/18 | 2/140 | 2/140 |

All 69 analyzed SDLCD variants returned Ambiguous. The original 72 requests included three inapplicable gap transforms. Both Supported claims among 200 synthetic observations were within tolerance, for **1% Supported coverage**. A required fit lacked an eligible decoy in two real and 21 synthetic analyses, which resulted in incomplete verification. The real cohort produced no Supported claims, and the two synthetic claims provide little evidence about precision at higher coverage. The reserved 11-object real test pilot awaits the threshold freeze and an approved test entry point.

### Stress conditions and selective evidence

Each stress condition requested all 18 SDLCD references. The seed was 17. Noise uses half the observed brightness standard deviation, cropping retains the first half of the valid baseline, and the gap removes the middle 30% when valid epochs remain on both sides. The table shows top-1 / Recall@3, out of 18 per condition. The variants test sensitivity on the same observations.

| Configuration | Added noise | Cropped baseline | Internal gap |
| --- | ---: | ---: | ---: |
| Lomb–Scargle | 3 / 8 | 2 / 8 | 3 / 5 |
| PDM | 10 / 13 | 11 / 14 | 10 / 12 |
| CNN | 1 / 6 | 3 / 5 | 2 / 5 |
| CNN + classical | 10 / 13 | 10 / 14 | 9 / 12 |
| Complete system | 0 / 0 | 0 / 0 | 0 / 0 |

On the synthetic development set the verdicts were 178 Ambiguous, 20 Inconclusive and two Supported. All 60 explicit No Reliable Period (NRP) cases received a non-Supported verdict. The NRP probability had a Brier score of 0.27270 and ten-bin ECE of 0.28074 on all 200 clean synthetic cases. These scores describe the CNN's NRP event. The risk–coverage plots rank emitted candidates with evaluable correctness. The complete-system plot has two eligible synthetic claims and no real claims. [Real risk–coverage](results/day5_real_validation/risk_coverage.html) · [Synthetic risk–coverage](results/day5_synthetic_development/risk_coverage.html) · [CNN reliability](results/day5_synthetic_development/calibration.html)

### Verification-only AUC and runtime

A separate verification-only experiment supplied simulator-labelled apparent periods directly to the verifier. Its pooled synthetic ROC-AUC was **0.952** across 50 true-period scores and 92 decoy scores. It measures fit-score discrimination on these simulator variants. End-to-end accuracy also depends on finding the right candidate, ranking it and meeting the verdict gates. All 92 matched comparisons favored the supplied period within their respective curve. [Saved verification-only report](results/verification_auc.json)

On an AMD Ryzen 7 7435HS, summed controller-stage p50/p95 was **20.72/31.26 seconds** for 18 clean real analyses and **18.09/28.20 seconds** for 200 clean synthetic analyses. These CPU measurements include the effects of concurrent local work and exclude input loading and report generation. The recorded run used Windows, Python 3.11.9 and CPU Torch 2.8.0 with one thread each for OpenMP, OpenBLAS and MKL. [Timing measurements](results/day5_evaluation.md#timing)

## Application and reproduction

Start from the repository root with Python 3.11 and the dependencies in `requirements.txt`. The application works through `app/main.py` and the plots in `app/plots.py`. Upload previews the observation and Stage 1 report. **Run Fast analysis** reports progress and exposes estimator candidates, spectra, reconciliation, optional phase folds, available verification and an analysis JSON export. A valid completed upload can be reused in the same session. Demo reads validated, source-labelled precomputed entries. Each cache payload records the schema, config, model artifacts and implementation identity. The portable synthetic demo lives under `outputs/demo_cache/`. The local SDLCD cache lives under `outputs/day4_local/demo_cache/`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
.\.venv\Scripts\python.exe -m scripts.run_demo_cache --check
.\.venv\Scripts\python.exe -m pytest tests/p2 -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m evaluation.run_eval --dataset real-val --stress noise --stress crop --stress gap --output docs/results/replay_real
.\.venv\Scripts\python.exe -m evaluation.run_eval --dataset synthetic-dev --output docs/results/replay_synthetic
```

If a cache is stale or missing, run `python -m scripts.run_demo_cache` in the same environment. Full cache validation requires the two curated SDLCD files and their SHA-256 entries in `data/sdlcd_raw/download_manifest.json`. The builder can publish the synthetic cache independently and reports missing local real entries. On a clean checkout, acquire the audited files locally under SDLCD's terms. `python -m scripts.download_sdlcd` populates the audited inventory from SDLCD when download access is available and a new manifest can be written. The manifests record input IDs and digests, checkpoint and calibration identities, configuration hash, seeds and package versions. The evaluation supports `--replay <manifest> --output <unused-path>` to check the recorded input and checkpoint identities and reuse the configuration. Runtime fields and wall-clock timeouts can vary. [Data audit](data_audit.md) · [Reproduction handoff](../DAY5_PROGRESS.md)

The independent verification allocation's `--auc --check-inputs` verifies recorded generator and inventory digests. The local Windows checkout required the generator's recorded LF bytes for that check. Other inputs matched their archived digests. See the handoff for the diagnostic and scope.

### Manual acceptance

- Run the cache check and launch Streamlit. Choose **Demo** and pick the synthetic example and each available local SDLCD example. Check the **Precomputed** source label, Ambiguous verdict, analysis sections and JSON download. The local SDLCD files are required to rebuild their caches.
- Choose **Upload** and use `tests/fixtures/synthetic_sine.csv` or the local validation file `00080B_1_2_R_20190724_DATA.txt`. Check the preview and Stage 1, select **Run Fast analysis**, and verify progress, source-labelled evidence, verdict and JSON. Run the same valid file again to check reuse. Upload a malformed CSV to check the parsing error. Check that the interface gives no definitive period when the verdict is Ambiguous.
- Human acceptance of both workflows is pending.

## CP2 classical-analysis backend

This appendix preserves the earlier classical-estimator measurements. At CP2 the controller already ran both classical estimators and computed the separate sampling window. That checkpoint used an early reconciliation skeleton. The integrated Stage 2 and application workflows are described above. [Checkpoint plan](../TumbleScope%20-%205-Day%20Development%20Plan%20and%20Two-Person%20Work%20Division.md)

### Methods, ranking and outputs

- **Lomb–Scargle** fits a floating mean and one, two or three sinusoidal terms to the prepared brightness. Astropy's standard-normalized power measures the fraction of variance removed by the periodic fit. Significant local power maxima become apparent-period candidates. A half-period peak from a double-humped lightcurve is retained for later harmonic comparison. [Implementation](../classical_analysis/lomb_scargle.py) · [Method](https://docs.astropy.org/en/stable/timeseries/lombscargle.html)
- **PDM** folds observations at each trial period and pools the within-bin sample variance. Bins below minimum occupancy are omitted from both the pooled and total variance calculations. Their ratio is Stellingwerf's theta. Significant local minima are ranked by `1 - theta`. The folding model can detect repeating shapes with multiple harmonics. [Implementation](../classical_analysis/pdm.py) · [Method](https://doi.org/10.1086/156444)
- **Spectral window** uses the squared magnitude of the mean unit phasor over valid, distinct times. Zero-frequency power is exactly 1. The strongest nonzero local maxima describe possible alias offsets. A window peak at offset $f_a$ identifies sampling structure that can move signal power to $f \pm f_a$. The height cutoff is a heuristic relative to zero-frequency power. [Implementation](../classical_analysis/window.py) · [Sampling-window reference](https://doi.org/10.3847/1538-4365/AAB766)

LS and PDM average brightness measurements at duplicate valid timestamps, then weight each distinct epoch equally. They share an evenly spaced frequency grid with target spacing `freq_resolution / frequency_oversampling`. The grid starts at the inverse of the maximum period and stops at the smaller of the inverse minimum period and the cadence cap. The window uses the same target spacing and starts at zero. Its upper limit is the larger of the configured median-cadence harmonic and the difference between the reciprocal period bounds. Reaching the window grid cap increases the spacing and can hide narrow alias peaks on long baselines. [Period bounds](../configs/period_bounds.yaml) · [Grid helper](../classical_analysis/lomb_scargle.py)

Both estimators keep significant peaks separated by at least one frequency-resolution element. Each reports `period_s` in seconds, `raw_score` as LS power or `1 - theta`, and within-estimator `support` as the raw score divided by the strongest retained raw score. The best candidate has support 1. Candidate metadata records `freq_hz`, `rank`, and `freq_width_hz`, measured at half maximum with a fallback to the baseline-derived resolution. Successful results include the downsampled periodogram and frequency resolution. All estimator results include runtime. The window returns its frequency grid, power, ranked offsets and frequency resolution. [Frozen output and status contract](../CONTRACTS.md#7-generic-dictionaries-required-and-optional-keys)

A completed search with no significant peaks returns `ok` with an empty list. Under the frozen error rules, too few usable samples returns `skipped` with `too_few_valid_samples`. Technical failures return `failed`. This applies the frozen rule where the development plan's LS task describes too few points as a failure. [Error handling](../CONTRACTS.md#8-error-handling)

### Configuration defaults

| Scope | Settings and defaults |
|---|---|
| Shared | `frequency_oversampling: 5`, `cadence_frequency_factor: 0.5` |
| Lomb–Scargle | `ls_nterms: 1`, `ls_max_candidates: 5`, `ls_periodogram_max_points: 1000`, `ls_false_alarm_alpha: 0.025` |
| PDM | `pdm_phase_bins: 10`, `pdm_min_bin_occupancy: 3`, `pdm_max_candidates: 5`, `pdm_periodogram_max_points: 1000`, `pdm_false_alarm_alpha: 0.05` |
| Spectral window | `n_peaks: 5`, `window_cadence_harmonics: 3`, `window_max_grid_points: 16385`, `window_min_peak_height: 0.05` |

The controller injects `period_min_s` and `period_max_s` from the shared bounds file. The CP2 verification used 2.912–445.881 seconds. [Classical settings](../configs/classical.yaml) · [Shared bounds](../configs/period_bounds.yaml)

### Synthetic significance calibration

Both estimators use fixed-frequency beta tails under independent, equal-variance Gaussian epoch errors, with a Bonferroni adjustment across the search grid. The settings were selected on seeded irregular and gapped series, then checked on fresh seeds. The measured rates below describe those sampling and noise controls. These controls do not estimate a real SDLCD false-positive rate. The Day 5 pilot evaluates recovery against available real reference periods. [LS calibration data](results/p1_cp2_ls_calibration.json) · [PDM calibration data](results/p1_cp2_pdm_calibration.json) · [PDM null distribution](https://users.camk.edu.pl/alex/papers/pdm.pdf)

**LS.** The selection sweep used 150 series per term count (seeds 1000–1149). The holdout used 250 per term count (seeds 5000–5249), across three cadence regimes, with a 37-second injection. The waveform added second and third harmonics for the corresponding multi-term experiments. Recovery required the highest-ranked candidate to be within 1% of the injected period.

| Fourier terms | White-noise searches with a candidate at 0.025 | Unit-amplitude recoveries at 0.025 |
|---:|---:|---:|
| 1 | 9/250 | 161/250 |
| 2 | 4/250 | 180/250 |
| 3 | 10/250 | 188/250 |

Across term counts, changing alpha from 0.05 to 0.025 reduced the holdout noise detections from 31 to 23, with 531 versus 529 unit-amplitude recoveries. Term counts reused the same seed set.

**PDM.** The selection sweep used 240 series per control (seeds 3100–3339) and compared alpha values 0.01, 0.025, 0.05 and 0.10. The holdout used 300 series per control (seeds 6100–6399), with injected periods of 19, 33 or 47 seconds. Recovery required the highest-ranked candidate to be within 2%.

| PDM holdout at alpha 0.05 | Outcome |
|---|---:|
| White noise | 13/300 produced a candidate |
| Sine, amplitude 0.7× noise SD | 96/300 recovered |
| Sine, amplitude 1.2× noise SD | 188/300 recovered |
| Two-harmonic signal, fundamental amplitude 1.2× noise SD | 227/300 recovered |
| AR(1) correlated noise, coefficient 0.65 | 278/300 produced a candidate |

Serial correlation violates the white-noise null. The correlated-noise result shows that PDM can assign significance to noise structure. The configured threshold should be interpreted with that limitation. [Paired white- and correlated-noise results](results/p1_cp2_pdm_correlated_noise.json)

### Acceptance checks and real SDLCD verification

The P1 acceptance tests cover LS recovery within 1% on a gapped sine, multi-term recovery and preservation of a half-period peak. They check that PDM agrees with LS on a clean sine and runs under 10 seconds for 2,000 observations and 5,000 trial periods. The evidence rerun measured a median of 0.570 seconds over five runs and a top-ranked period of 47.026 seconds for a 47-second injection. [PDM benchmark timings and input recipe](results/p1_cp2_pdm_benchmark.json) The window test recovers the first three cadence harmonics on regular timestamps. [LS tests](../tests/p1/test_ls.py) · [PDM tests](../tests/p1/test_pdm.py) · [Window tests](../tests/p1/test_window.py)

Direct execution ran SDLCD ingestion, preprocessing, Stage 1, LS and PDM on three local files. Controller results matched the direct estimators apart from runtime. Both estimators returned `ok` on all three, including the valid empty search on `00041B`. Periods below are in seconds, in descending support order. These are smoke-test candidate outputs. [SDLCD identities, candidates, quality and window data](results/p1_cp2_sdlcd_outputs.json)

| SDLCD file | Stage 1 and valid rows | LS periods | PDM periods |
|---|---|---|---|
| `00029A_2_3_R_20190322_DATA.txt` | Suitable, 273 | 32.840, 65.569, 10.976, 16.471 | 65.569, 32.840, 131.874, 98.187, 195.378 |
| `00041B_2_3_R_20191118_DATA.txt` | Limited, 248 | None | None |
| `00080B_1_2_R_20190724_DATA.txt` | Limited, 186 | 11.913, 10.862, 10.514 | 11.877, 23.765, 17.846, 35.558, 29.669 |

The two limited observations had weak brightness modulation relative to the Stage 1 noise proxy. All windows passed the frozen schema and JSON checks, started at zero frequency with power 1, and returned five ranked offsets.

| File prefix | Window grid points | Strongest offset (Hz) | Relative height |
|---|---:|---:|---:|
| `00029A` | 4,224 | 0.207062 | 0.261 |
| `00041B` | 4,049 | 0.527171 | 0.251 |
| `00080B` | 3,040 | 0.523290 | 0.314 |

### Reproduce the CP2 evidence

Run from the repository root with the project virtual environment active. Supply the three named local SDLCD observations in the input directory. The runner does the calibration sweeps, benchmarks PDM, and checks that the real-file controller and direct estimators agree apart from runtime. Use a separate output directory to preserve the archived measurements. Generated artifacts include seeds or file hashes, trial counts, configuration, environment, and source hashes. [Evidence runner](../evaluation/cp2_evidence.py)

```text
python -m evaluation.cp2_evidence --sdlcd-dir data/sdlcd_raw --output-dir outputs/cp2_replay
```

### Tests

Run these commands from the repository root with the project virtual environment active. The P1 suite includes the locally downloaded SDLCD fixtures used by the existing ingestion checks.

```text
python -m pytest tests/p1/test_ls.py tests/p1/test_pdm.py tests/p1/test_window.py -v
python -m pytest tests/p1 -q
python -m pytest -q
```
