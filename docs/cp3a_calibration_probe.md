# CP3a independent-seed diagnostic probe

This is a limited diagnostic, not a completed calibration or a basis for changing thresholds. It supplements the preserved 216-case pilot in `.phase0-local/cp3a-audit/cp3a_threshold_sensitivity.json`.

Run `python -m evaluation.cp3a_calibration_probe` from the repository root. The runner refuses to overwrite its output files. It uses a 32-second injected period, 192-second baseline, 128 irregular samples, Gaussian noise SD 0.35, and sine, double-hump and noise-only signals. Real Lomb–Scargle and PDM estimators, spectral window, and reconciliation run for each sample. A signal is retained when the shortlist contains a period within 2% of the injection; for the double-hump case the half-period is also accepted as a competing hypothesis. Noise survival means any shortlist, not a false-Supported verdict. No reference labels or held-out observations are read.

Calibration uses seeds 31000–31029; the separate validation probe uses 41000–41029 and the unchanged existing configuration. The calibration probe compares the existing configuration against one alternative for each of the seven settings. It does not select a winner: all alternatives give identical outcomes on these controls. Validation therefore evaluates the existing configuration as a diagnostic, not a selected and validated threshold policy.

| Split | Scenario | Outcome | 95% Wilson interval |
|---|---|---:|---:|
| Calibration | Sine retention | 30/30 | 88.6–100% |
| Calibration | Double-hump retention, either hypothesis | 30/30 | 88.6–100% |
| Calibration | Noise with shortlist | 1/30 | 0.6–16.7% |
| Validation | Sine retention | 30/30 | 88.6–100% |
| Validation | Double-hump retention, either hypothesis | 30/30 | 88.6–100% |
| Validation | Noise with shortlist | 3/30 | 3.5–25.6% |

Thirty seeds per scenario make the sampling uncertainty visible, but the intervals are wide and do not establish general performance. These are easy, long-baseline, one-noise-level controls; they do not assess gaps, missing observations, variable cadence or amplitude, alias confusion, nearby true signals, distinct-true false merges, duplicate merging, harmonic-pair retention, or interactions between settings. Candidate matching also does not establish that the gate identifies all false candidates. Individual outcomes and rejection counts are in `docs/results/p1_cp3a_probe_calibration.json` and `docs/results/p1_cp3a_probe_validation.json`.

No acceptance target or cost ratio was found in the contract. A predeclared selection method and broader calibration scenarios, with separate validation seeds and uncertainty for the missing measures above, are needed. The seven settings remain unresolved and unchanged; this probe must not be represented as independent validation of a calibrated configuration.
