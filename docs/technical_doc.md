# Technical notes

## CP2 classical-analysis backend

P1-07 Lomb–Scargle, P1-08 Phase Dispersion Minimisation (PDM) and P1-09 Spectral Window are implemented. The controller calls the real Lomb–Scargle and PDM functions after Stage 1 for suitable and limited observations. The window runs on the same prepared observation as a separate sampling diagnostic. The Day 2 reconciliation skeleton emits provisional clusters. Full reconciliation and Stage 2 are planned for CP3a. [Development plan](../TumbleScope%20-%205-Day%20Development%20Plan%20and%20Two-Person%20Work%20Division.md) · [Controller](../pipeline/controller.py)

Shared CP2 acceptance still requires the **P2-owned Streamlit upload and display path**. The current app loads a bundled synthetic fixture and shows estimator status and detail. P2 needs to connect SDLCD upload to ingestion, render the actual Stage 1 quality status, and show the Lomb–Scargle and PDM candidate periods separately. The shared acceptance check is to upload two real files and verify those outputs in the app. [Streamlit entry point](../app/main.py)

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

Both estimators use fixed-frequency beta tails under independent, equal-variance Gaussian epoch errors, with a Bonferroni adjustment across the search grid. The settings were selected on seeded irregular and gapped series, then checked on fresh seeds. The measured rates below describe those sampling and noise controls. Real SDLCD false-positive rates and recovery accuracy still need evaluation. [LS calibration data](results/p1_cp2_ls_calibration.json) · [PDM calibration data](results/p1_cp2_pdm_calibration.json) · [PDM null distribution](https://users.camk.edu.pl/alex/papers/pdm.pdf)

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

Run from the repository root with the project virtual environment active. Supply the three named local SDLCD observations in the input directory. The runner does the calibration sweeps, benchmarks PDM, and checks that the real-file controller and direct estimators agree apart from runtime. Generated artifacts include seeds or file hashes, trial counts, configuration, environment, and source hashes. [Evidence runner](../evaluation/cp2_evidence.py)

```text
python -m evaluation.cp2_evidence --sdlcd-dir data/sdlcd_raw --output-dir docs/results
```

### Tests

Run these commands from the repository root with the project virtual environment active. The P1 suite includes the locally downloaded SDLCD fixtures used by the existing ingestion checks.

```text
python -m pytest tests/p1/test_ls.py tests/p1/test_pdm.py tests/p1/test_window.py -v
python -m pytest tests/p1 -q
python -m pytest -q
```
