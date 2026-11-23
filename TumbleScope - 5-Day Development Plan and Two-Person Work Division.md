# TumbleScope - 5-Day Development Plan and Two-Person Work Division

*Team Ebarbs | Prasunex Innovathon 2026 | Planning document only, no code written*

**How to use this document.** P1 = Person 1 (contained workload). P2 = Person 2 (heavier workload, Phase 0, shared infrastructure, integration). Both read Sections 1, 2, 5, 8, 9, 10. Send **Section 3** to P1 and **Section 4** to P2; Sections 5-8 are shared and binding for both.

**Sources used.** The project README (TumbleScope.md), the Round 1 deck and the Round 2 deck. `evaluation-criteria.txt` was not attached, so the criteria listed in your brief were used. Names, numbers and scope (apparent/synodic period, top-5 shortlist, 6-block dilated CNN of about 0.25 M parameters, three verdicts, nine reason codes, Fast mode only) are kept identical to what you already submitted. Small consistency notes: the Round 1 deck says four setups but lists three; this plan uses the five configurations in the README and Round 2 deck. Round 1 mentions 3js for the 3D view; the README says Plotly, so this plan uses Plotly.

---

## 1. Overall development strategy

### 1.1 Architecture in one paragraph

TumbleScope is a chain of **pure functions over frozen dataclasses**. Every module takes contract objects in and returns contract objects out. No module imports another person's internals, only `core.schemas`. Streamlit calls one function, `pipeline.controller.analyze()`, which runs the chain and returns a single `AnalysisResult` containing every intermediate result. That one object also feeds the evaluation harness, so each observation is analysed once and all five evaluation configurations (LS only, PDM only, CNN only, CNN + classical, complete system) are read out of the same result.

```
file -> Observation -> PreparedObservation + QualityReport (Stage 1)
     -> [CNN | Lomb-Scargle | PDM] as three EstimatorResult
     -> ReconciliationResult (clusters, harmonic/alias links, Stage 2, top-K)
     -> VerificationResult (physics fit, held-out blocks, decoys, P vs 2P)
     -> Verdict (Supported | Ambiguous | Inconclusive + reason codes)
     -> AnalysisResult -> dashboard / evaluation
```

### 1.2 Where the work is split, and why

The split follows the natural seam in the data flow:

- **P1 owns the signal side:** everything that turns a raw lightcurve into a quality report and a ranked, harmonic- and alias-aware candidate shortlist, plus the instruments that measure and display it (plots, evaluation metrics, stress tests) and the written documentation. These are pure NumPy/SciPy/Astropy functions with small inputs, no GPU, no optimisation loops and clear unit tests.
- **P2 owns the model side and the infrastructure:** repository, contracts, stubs, physics simulator, synthetic data, CNN training and calibration, physics verification, verdict logic, controller, Streamlit app, integration and release.

**Why this seam works.** Reconciliation consumes a *list of EstimatorResult objects*; the CNN is just one of three. P1 can therefore build and test reconciliation today against a mock CNN fixture that P2 provides in Phase 0, and never needs the real CNN. In the other direction, P2's simulator degrader only needs a JSON file of real sampling statistics from P1 (with a default fallback), and P2's verification only needs the shortlist object that reconciliation returns (a mock is provided).

**Approximate workload.** P1 is about 35-40 percent of the technical effort, P2 about 60-65 percent. This is intentional. Both have genuine scientific implementation work, and P1 additionally owns evaluation and documentation, which are important for judging but are low-risk to the prototype.

### 1.3 The walking skeleton

In Phase 0, P2 creates a **stub for every module**. Each stub has the final function signature and returns a valid, clearly marked dummy object (`is_stub=True`). From the first hour of Day 1, `streamlit run app/main.py` already runs end to end on fake data. Each owner then *replaces* their stubs with real code. Integration becomes swapping stubs for real modules one at a time rather than a big-bang merge on Day 5.

### 1.4 Non-negotiable scientific rules (both people)

1. The target quantity is the **apparent (synodic) period**, the quantity SDLCD reports. Never call it the sidereal or physical rotation period anywhere in code, UI text or slides.
2. CNN, Lomb-Scargle and PDM outputs stay **separate and labelled by source** all the way to the dashboard.
3. **P and 2P are hypotheses; decoys are not.** Harmonic partners are kept and compared; decoys are deliberately non-harmonic, non-alias periods.
4. A good physics fit means *compatible with a simplified model*. It never proves the real object's shape or spin state. UI text must say so.
5. If verification is incomplete, the verdict is **Ambiguous (verification\_incomplete)**, never Supported. If observation quality or evidence is insufficient, the verdict is **Inconclusive**.
6. **SDLCD reference periods never enter inference or training.** They are used only in evaluation.
7. Nothing appears in the README, docs or slides as a result unless a file in `docs/results/` produced it. Everything else is labelled planned.

### 1.5 How to use AI coding agents without clashing

Every task in Sections 3 and 4 is a self-contained block (ID, files, output, acceptance criteria). Give the agent: (a) `CONTRACTS.md`, (b) the one task block, (c) this fence rule: *You may only create or edit the files listed under Files. Do not edit core/, configs you do not own, requirements.txt or any other folder. If you think a contract must change, stop and say so.* Then run the acceptance criteria as tests before committing. Tasks are sized for 1-3 hours.

---

## 2. Phase 0: Repository and environment setup

**Time-box: 3 hours for P2, 45 minutes for P1, completed before Day 1 starts** (for example the evening before). Phase 0 is not one of the five building days. If P2 runs over, ship what exists, and finish only the items marked *deferrable* in the first 30 minutes of Day 1. No real algorithm code is written in Phase 0.

### 2.1 P2 tasks (owner: P2)

| # | Task | Min |
| --- | --- | --- |
| 1 | Create the GitHub repository, add P1 as collaborator, create branches `main`, `p1/dev`, `p2/dev` | 15 |
| 2 | Folder structure (Section 6) with `__init__.py` files, `.gitignore`, `requirements.txt`, install test | 30 |
| 3 | `core/schemas.py`, `core/errors.py`, `core/config.py` (Section 5) | 45 |
| 4 | Stubs for every module, fixtures, empty per-owner YAML configs | 45 |
| 5 | Stub pipeline controller, minimal Streamlit page, contract tests, stub end-to-end test | 30 |
| 6 | `CONTRACTS.md`, README status update, tag `v0-skeleton`, handoff message | 15 |

Details:

- **Repository.** Initialise the repo with the existing README. Default branch `main`. `p1/dev` and `p2/dev` are branched from main after the skeleton is committed (do step 1 first, then create the branches after step 6 if you prefer, but both must exist before handoff).
- **`.gitignore`.** Python (`__pycache__/`, `*.pyc`, `.venv/`, `.pytest_cache/`), `.ipynb_checkpoints/`, OS files, `data/sdlcd_raw/`, `data/synthetic/` (large), `models/*.pt` except the final checkpoint (force-added later), `outputs/` except `outputs/demo_cache/`, `*.npz`, `.env`.
- **`requirements.txt`.** One file, owned by P2 only: numpy, pandas, scipy, astropy, scikit-learn, torch (CPU build is fine locally; Colab uses its own GPU build), streamlit, plotly, matplotlib, pyyaml, pytest, tqdm. Install in a fresh venv on Python 3.10 or 3.11, then pin the top-level packages to the versions that worked. Nobody else edits this file.
- **Shared interface files** (frozen after Phase 0): `core/schemas.py`, `core/errors.py`, `core/config.py`, `CONTRACTS.md`, `tests/test_contracts.py`, `tests/fixtures/*`.
- **Configuration.** `configs/default.yaml` lists which per-owner YAML files to load. Per-owner files: P1 owns `preprocess.yaml`, `classical.yaml`, `reconcile.yaml`; P2 owns `simulation.yaml`, `cnn.yaml`, `verification.yaml`, `verdict.yaml`. Config sections are plain dictionaries, so adding a new numeric key never requires editing `core/`.
- **Stubs.** Each stub returns a valid dummy contract object with `is_stub=True` and a deterministic fake result so the dashboard can render something. Provide fixtures: `tests/fixtures/synthetic_sine.csv` (clean sine with gaps), `tests/fixtures/mock_cnn_result.json` (an EstimatorResult with three peaks including a 2P partner), `tests/fixtures/mock_prepared.npz`, `tests/fixtures/mock_reconciliation.json` (for P2's verification development), `data/profiles/default_profiles.json` (hand-written fallback sampling profiles).
- **Tests that must pass at handoff:** `test_contracts.py` (dataclasses construct and round-trip to JSON; function signatures match the frozen list), `test_stub_pipeline.py` (stub `analyze()` returns a full `AnalysisResult`), `test_no_reference_leak.py` (inference modules never import the reference-label module).

**Checkpoint-specific fixtures and guards.** The Phase 0 handoff includes all CP0 fixtures and guards required by `CONTRACTS.md` §§13–14. P2 supplies the golden CNN archive at CP2 with the numerical feature and generator checks. P1 delivers real Stage 2 at CP3a, and P2 runs its root guard. P2 verifies real inference’s shared-builder call and checkpoint compatibility at CP3b. Configured period bounds and simulator-side checks are due at CP1. Contract §13 governs activation and pass requirements.
- **Handoff message to P1** (send when tagged `v0-skeleton`): repo URL, branch name `p1/dev`, Python version, the five install commands, the commands to run tests and the app, and the instruction to read `CONTRACTS.md` and Section 3.

### 2.2 P1 tasks (owner: P1, minimal)

1. Accept the collaborator invitation and clone the repository.
2. `git checkout p1/dev`.
3. Create a virtual environment and run `pip install -r requirements.txt`.
4. Run `pytest -q` (expect all green) and `streamlit run app/main.py` (expect a page showing a clearly marked stub result).
5. Read `CONTRACTS.md` and Section 3. Reply to P2: Phase 0 OK, plus any contract question.
6. Do **not**: create another repo, add dependencies, change folder structure or edit `core/`. Request any dependency by message to P2.

**Phase 0 exit gate:** both machines run the stub app and tests; `v0-skeleton` is tagged; P1 has confirmed.

---

## 3. Person 1: Signal side, evaluation and documentation

### 3.1 Role summary

P1 turns raw lightcurve files into analysis-ready signals and an honest, ranked candidate shortlist, measures how well the system performs, and writes the documentation. P1 does not need a GPU, does not train models and never touches the simulator, CNN, verification, verdict or app entry point.

**P1 owns (only edit these):** `ingestion/`, `preprocessing/`, `classical_analysis/`, `reconciliation/`, `evaluation/`, `app/plots.py`, `configs/preprocess.yaml`, `configs/classical.yaml`, `configs/reconcile.yaml`, `data/samples/`, `data/reference/`, `data/profiles/sdlcd_profiles.json`, `docs/` (except `docs/results/verification_*`), `README.md` after `v0-skeleton`, `tests/p1/`.

**P1 must NOT touch:** `core/`, `physics/`, `ml/`, `verdict/`, `pipeline/`, `app/main.py`, `requirements.txt`, `.gitignore`, any P2 config file, `tests/fixtures/`, `tests/test_contracts.py`.

**P1 needs from P2:** the contracts and stubs (Phase 0); `mock_cnn_result.json` (Phase 0); the real `analyze()` and a working app (from CP2); CNN outputs and a synthetic holdout set `data/synthetic_holdout/holdout_small.npz` for evaluation (Day 3).

**P1 must deliver before integration:** CP1: ingestion + preprocessing + Stage 1 + sampling profiles. CP2: LS + PDM. CP3a: reconciliation + Stage 2. CP3b: plots. CP4: evaluation outputs and documentation drafts.

### 3.2 Tasks (each is one AI-agent prompt)

**P1-01 SDLCD data audit (Day 1, up to 2 h).**

- Do by hand first: download 20-50 lightcurve files from SDLCD Data Release 3, including at least 25 objects with a reported period, a few without, and a mix of short and long observations. If bulk download exists, use it; otherwise hand-pick. Keep raw data in `data/sdlcd_raw/` (git-ignored); commit only 3-5 small samples to `data/samples/` after checking the site's terms of use and citation note.
- Files: `docs/data_audit.md`, `data/samples/*`.
- Output: a document stating the real column layout, time convention (and its conversion to seconds), brightness convention (instrumental magnitude, sign), availability of uncertainties, header metadata, which period field values are genuine and which are reserved internal codes, plus a table (file, object id, points, duration, median cadence, reference period or none).
- Accept: every claim in the audit was verified on a real file; P2 can read the typical duration and period range from it.

**P1-02 Ingestion (Day 1).**

- Files: `ingestion/__init__.py` (exposes `load_observation`), `ingestion/sdlcd.py`, `ingestion/generic.py`, `tests/p1/test_ingestion.py`.
- Contract: `load_observation(path, fmt=None) -> Observation`. Detect format, extract timestamps (converted to float seconds since first sample), brightness, uncertainties if present, metadata. Set `brightness_kind` ('mag' for SDLCD). It does **not** sort or clean; that is preprocessing. Ambiguous conventions are recorded in `meta['assumptions']` so the UI can show them.
- Failure: raise `IngestionError` with a human-readable message (file empty, no numeric columns, unknown format).
- Accept: all downloaded samples load; the synthetic CSV fixture loads; a malformed file raises `IngestionError`, never a bare exception; no reference label is ever attached to an Observation.

**P1-03 Reference table (Day 1, evaluation only).**

- Files: `ingestion/reference.py`, `data/reference/reference_table.csv`, `tests/p1/test_reference.py`.
- Contract: `load_reference_table(path) -> dict[str, ReferenceLabel]`. Excludes reserved internal period codes (as found in the audit). Adds `object_id` and a deterministic, seeded **object-level** `split` column (`val` or `test`).
- Accept: no object appears in both splits; counts per split are printed; the sample size is recorded in `docs/data_audit.md` (this is a pilot, so report n).

**P1-04 Preprocessing (Day 1).**

- Files: `preprocessing/prepare.py`, `tests/p1/test_prepare.py`.
- Contract: `prepare(obs, cfg) -> PreparedObservation`. Sort time and brightness together; detect and flag duplicates, gaps and missing values (missing is `valid_mask=False`, never zero); robust normalisation (median/MAD) with the sign flipped for magnitudes so that higher means brighter (`y_norm`); relative linear flux for physics (`flux_rel`, or None when the convention is unknown); `dt`; flags for bright outliers (**flagged, not deleted**); originals preserved; no detrending.
- Accept: shuffled input gives identical output to sorted input; NaNs never enter `y_norm` as numbers where `valid_mask` is False; outlier flags exist but the point count is unchanged; magnitude and flux inputs of the same signal give the same `y_norm` shape.

**P1-05 Stage 1 quality gate (Day 1).**

- Files: `preprocessing/quality.py`, `configs/preprocess.yaml`, `tests/p1/test_quality.py`.
- Contract: `assess_stage1(prep, cfg) -> QualityReport` with status `suitable | limited | insufficient`, metrics (valid count, duration, missing fraction, largest gap, median cadence, noise proxy, amplitude-to-noise proxy, outlier fraction, exposure or cadence note) and plain-language warnings. Thresholds live in `preprocess.yaml`.
- Accept: clean sine is suitable; 20 points is insufficient; heavy gaps or low amplitude-to-noise is limited; every non-suitable status includes at least one reason string.

**P1-06 Sampling profiles for P2 (Day 1, due by end of day).**

- Files: `preprocessing/sampling_profile.py`, `data/profiles/sdlcd_profiles.json`.
- Output: one JSON record per real observation with fields defined in `CONTRACTS.md` (n\_points, duration\_s, median\_dt\_s, gap list as fractions of duration, noise sigma relative to amplitude, missing fraction). No brightness values, no reference periods.
- Accept: at least 20 records, validates against the schema in `CONTRACTS.md`. P2's degrader falls back to `default_profiles.json` if this file is late.

**P1-07 Lomb-Scargle (Day 2).**

- Files: `classical_analysis/lomb_scargle.py`, `configs/classical.yaml`, `tests/p1/test_ls.py`.
- Contract: `run_lomb_scargle(prep, cfg) -> EstimatorResult` (method='lomb\_scargle'). Astropy LombScargle with 1-term and multi-term (2-3 terms) as a config option; frequency grid from the baseline and cadence with oversampling; period bounds from config (set after the audit); top-N peaks with minimum separation of one frequency resolution; each candidate carries `period_s`, `raw_score` (power) and `support` normalised to 0-1. A downsampled periodogram goes into `meta` for plotting.
- Note: on double-humped curves LS peaks at P/2. Do **not** auto-double. The harmonic hypothesis is handled in reconciliation and verification.
- Failure: too few points returns `status='failed'` with an error string, not an exception.
- Accept: synthetic sine with random gaps recovers the period within 1 percent; a two-harmonic signal is recovered by the multi-term option; result is deterministic.

**P1-08 PDM (Day 2).**

- Files: `classical_analysis/pdm.py`, `tests/p1/test_pdm.py`.
- Contract: `run_pdm(prep, cfg) -> EstimatorResult` (method='pdm'). Own NumPy implementation of Stellingwerf's statistic (phase bins, per-bin variance, theta), vectorised over trial periods on the same grid as LS; minima as candidates; `support = 1 - theta` normalised to 0-1.
- Accept: agrees with LS on a clean sine; runtime under 10 s for 2,000 points and 5,000 trial periods; deterministic; failure returns `status='failed'`.

**P1-09 Spectral window (Day 2).**

- Files: `classical_analysis/window.py`, `tests/p1/test_window.py`.
- Contract: `compute_window(prep, cfg) -> dict` giving the strongest sampling-window peaks (alias offsets in frequency and their relative heights). Used only by reconciliation.
- Accept: for regular sampling with a known cadence, the window peaks fall at multiples of the sampling frequency.

**P1-10 Candidate reconciliation and Stage 2 (Day 3).**

- Files: `reconciliation/reconcile.py`, `cluster.py`, `harmonics.py`, `aliases.py`, `score.py`, `stage2.py`, `configs/reconcile.yaml`, `tests/p1/test_reconcile.py`.
- Contract: `reconcile(prep, estimator_results, cfg) -> ReconciliationResult`.
  1. Cluster candidates from all available estimators with tolerance tied to frequency resolution (about one resolution element).
  2. Record harmonic relations (ratios 2, 1/2, 3, 1/3, 3/2, 2/3 within tolerance) as `harmonic_links`. Never discard a harmonic partner.
  3. Alias likelihood A(P) from the spectral window (candidate frequency near another strong peak plus or minus a window offset).
  4. Score S(P) = wC RC + wL RLS + wD RPDM - lambdaA A(P) on normalised support. If an estimator failed or was skipped, renormalise weights over the available ones and note it in `notes`.
  5. **Stage 2** per cluster: cycles observed, frequency resolution versus period, separation from neighbours, phase coverage when folded. Output `passed` and `fail_reasons` per cluster.
  6. Rank, return the top K (default 5) of the clusters that pass Stage 2. Failed clusters are still listed, with reasons. If none pass, `stage2_passed_any=False`.
- Accept (using the mock CNN fixture): a synthetic double-humped curve yields P and 2P as linked clusters, both in the shortlist; a curve covering fewer than the minimum cycles fails Stage 2 with a reason; an estimator marked failed does not crash reconciliation; output is JSON round-trippable.

**P1-11 Plots (Day 3).**

- Files: `app/plots.py`, `tests/p1/test_plots.py`.
- Contract: pure functions returning Plotly figures: `fig_lightcurve(prep)`, `fig_periodograms(estimator_results)`, `fig_phase_fold(prep, period_s)`, `fig_candidates(recon)`, `fig_quality(report)`, `fig_verification(verification)`, `fig_risk_coverage(eval_df)`, `fig_calibration(eval_df)`. Source colour and label are consistent across all figures (CNN, LS, PDM are always distinguishable).
- Accept: each function runs on stub objects and on real objects and can be written to HTML; empty or failed inputs return a figure with an explanatory annotation, never an exception.

**P1-12 Evaluation harness (Days 3-4).**

- Files: `evaluation/metrics.py`, `evaluation/stress.py`, `evaluation/run_eval.py`, `tests/p1/test_metrics.py`.
- `metrics.py`: median log-period error, tolerance accuracy, harmonic error rate (predicted is about 2x or 0.5x of reference), catastrophic error rate, Recall@K, candidate retention, false-Supported rate, coverage, risk-coverage points, Brier score, ECE. Pure functions with hand-computed unit tests.
- `stress.py`: functions that degrade an `Observation` (add noise, crop to fewer cycles, remove larger gaps, lower modulation relative to noise, clustered irregular sampling).
- `run_eval.py`: for each observation on the real pilot split and the synthetic holdout, call `analyze()` once, then read out the five configurations from the one `AnalysisResult`. Writes CSV files and figures to `docs/results/`, with sample sizes in every table.
- Accept: metrics tests pass; harness runs on the stub pipeline from Day 3 start; outputs report n, the split used and the date/commit.

**P1-13 Documentation and demo assets (Days 1-5, small daily slices).**

- Files: `README.md`, `docs/technical_doc.md`, `docs/limitations.md`, `docs/future_work.md`, `docs/demo_script.md`, `docs/screenshots/`.
- Output: see Section 11 for the exact contents. Write each section the day the feature lands, so nothing is left to Day 5.
- Accept: every claim cites a file in `docs/results/` or is labelled planned.

### 3.3 What P1 hands over, and when

| Deliverable | Consumer | Due |
| --- | --- | --- |
| `load_observation`, `prepare`, `assess_stage1` | P2 controller | CP1 |
| `data/profiles/sdlcd_profiles.json` | P2 degrader | CP1 |
| `run_lomb_scargle`, `run_pdm` | P2 controller | CP2 |
| `reconcile` (with Stage 2) | P2 verification, verdict | CP3a |
| `app/plots.py` | P2 app | CP3b |
| Reconciliation weights tuned on the val split | P2 `reconcile.yaml` is P1-owned, so P1 edits it directly | CP4 |
| Evaluation tables and figures | docs, slides | CP4, final run in CP5 |

---

## 4. Person 2: Model side, infrastructure and integration

### 4.1 Role summary

P2 builds everything that needs simulation, training or optimisation, owns the contracts and the repository, assembles the pipeline and the dashboard, and is the only person who merges to `main`. P2 starts from Phase 0 (Section 2.1).

**P2 owns (only P2 edits these):** `core/`, `physics/`, `ml/`, `verdict/`, `pipeline/`, `app/main.py` and `app/pages/` if used, `configs/default.yaml`, `configs/simulation.yaml`, `cnn.yaml`, `verification.yaml`, `verdict.yaml`, `requirements.txt`, `.gitignore`, `scripts/`, `models/`, `tests/fixtures/`, `tests/test_*.py` at the top level, `tests/p2/`, `data/synthetic/`, `data/profiles/default_profiles.json`, `outputs/demo_cache/`, `docs/results/verification_*`.

**P2 must NOT touch** (except via a request message to P1): `ingestion/`, `preprocessing/`, `classical_analysis/`, `reconciliation/`, `evaluation/`, `app/plots.py`, `README.md`, P1 config files, `docs/` other than verification results. If a P1 module misbehaves in integration, P2 sends P1 a failing test or a saved input that reproduces it; P1 fixes it.

**P2 needs from P1:** ingestion, preprocessing and Stage 1 (CP1), sampling profiles (CP1, with fallback), LS/PDM (CP2), reconciliation (CP3a), plots (CP3b), evaluation outputs (CP4).

### 4.2 Tasks (each is one AI-agent prompt; training and optimisation tasks are checked by numbers, not by eye)

**P2-01 Physics simulator (Day 1).**

- Files: `physics/params.py` (`SimParams` dataclass: shape type, dimensions, per-facet reflectivity, spin axis angles, phase offset, period, Sun and observer directions or phase angle), `physics/geometry.py`, `physics/simulator.py`, `configs/simulation.yaml`, `tests/p2/test_simulator.py`.
- Contract: `simulate_lightcurve(params, t_s) -> np.ndarray` of relative brightness. Geometry: box and cylinder (rocket body, modelled as a prism of about 16 side facets plus two end caps). Each facet has area A, normal n and reflectivity rho. Single-axis rotation by Rodrigues rotation; fixed or simplified Sun and observer directions. Lambertian diffuse: B proportional to the sum over facets of A rho max(0, n.s) max(0, n.o). No self-shadowing, no tumbling dynamics, no specular term (cut).
- Accept: B(t + P) equals B(t) to 1e-9; a box with symmetric faces repeats at P/2 (this is the harmonic trap, and the test documents it); a cylinder spinning about its own axis seen side-on gives a flat curve (used for no-period examples); 1,000 timestamps of a box in under 5 ms; vectorised over facets and time.

**P2-02 Labels and pilot dataset (Day 1).**

- Files: `ml/dataset/sample_params.py`, `ml/dataset/labels.py`, `ml/dataset/generate.py`, `tests/p2/test_labels.py`.
- Contract: generate a clean, densely sampled curve covering several cycles; label with the **smallest repeat shift** that matches the pattern within a similarity tolerance (this is the apparent period, and it is independent of LS and PDM). Add *No Reliable Period* examples from three sources: pure noise or constant signals, multi-frequency (two unrelated rotations), and under-covered (fewer than the minimum cycles). Store shards as `.npz` with inputs, label bin, soft label, simulator parameters, seed and `split`. Parameter ranges for period and baseline come from P1's data audit. Hold out whole simulator variants (specific aspect-ratio and axis-orientation ranges, plus one geometry family variation) for the test split.
- Accept: 1,000-curve pilot in under 3 minutes; reproducible from a seed; label histogram plotted; 20 curves inspected by eye; the symmetric box is labelled P/2 and says so in the metadata.

**P2-03 Degrader and shared feature builder (Day 2).**

- Files: `ml/dataset/degrade.py`, `ml/features.py`, `tests/p2/test_features.py`.
- Contract: apply a sampling profile (cadence, gap pattern, noise relative to amplitude, missing fraction) to a clean curve, producing `t_s`, `y`, mask. `features.build_cnn_windows(prep) -> np.ndarray (n_windows, 3, 1024)` with channels normalised brightness, validity mask and time difference. **The same function is used for training data and for inference**, so the two cannot drift. The time axis is never resized; short inputs are right-padded and masked; long ones are split into overlapping windows.
- Accept: a synthetic observation pushed through `prepare()` and `build_cnn_windows()` equals the training path for the same degraded sample; real profiles are used when `sdlcd_profiles.json` exists, otherwise `default_profiles.json`.

**P2-04 CNN and training (Days 1-3).**

- Files: `ml/model.py`, `ml/train.py`, `ml/colab_train.ipynb`, `configs/cnn.yaml`, `models/cnn_v1.pt` (committed once final, about 1 MB), `tests/p2/test_model.py`.
- Design (as in the README): dilated residual 1D CNN, 6 blocks, channels 32-32-64-64-128-128, kernel 7, dilations 1-2-4-8-16-32, global average pooling, 128 log-spaced apparent-period bins plus a *No Reliable Period* class. Soft labels in log-period space with a soft-target cross-entropy loss. Each residual block contains exactly ONE dilated convolution (kernel 7) followed by normalisation and activation, with a 1x1 projection on the skip path when the channel count changes. Under that one-convolution-per-block rule the receptive field is 1 + 6 x (1+2+4+8+16+32) = 379 and the parameter count lands near 0.25 M. A test asserts both against the implemented network. If the architecture ever changes (for example two convolutions per block), the expected numbers and the test are updated together and the README and slides are corrected to match, never the other way round.
- Training plan: first run on 5,000 curves (launched early on Day 2 on one Colab GPU), second run on up to 20,000 only if the first is stable and time allows. Log median log-period error, tolerance accuracy and harmonic error rate on synthetic validation, and compare with a trivial baseline.
- Accept: shape and receptive-field tests pass; training loss falls and validation metrics are logged to `docs/results/cnn_training_log.json`; checkpoint loads on CPU. If results are weak, the CNN is reported as experimental (see Section 12). Do not tune on the held-out simulator variants.

**P2-05 Calibration and inference (Day 3).**

- Files: `ml/calibrate.py`, `ml/infer.py`, `tests/p2/test_infer.py`.
- Contract: `run_cnn(prep, cfg) -> EstimatorResult` (method='cnn'). Window predictions combined by a **weighted average, not a product** (windows overlap and are not independent); strong disagreement between windows is kept as extra candidates and as `meta['window_disagreement']`. Temperature scaling on validation data; report ECE and Brier. Candidates are peaks of the calibrated distribution converted from bin centres to seconds; `meta['p_no_reliable']` carries the abstain probability.
- Failure: missing checkpoint returns `status='skipped'`; any other exception is caught by the controller. The pipeline keeps running on LS and PDM.
- Accept: runs on CPU in under 2 s for a typical observation; ECE after calibration is not worse than before; outputs match the contract on the mock inputs.

**P2-06 Physics verification (Days 2-3, the heaviest task).**

- Files: `physics/fit.py`, `physics/verify.py`, `physics/eval_verification.py`, `configs/verification.yaml`, `tests/p2/test_verify.py`.
- Contract: `verify_candidates(prep, recon, cfg, mode='fast', progress=None) -> VerificationResult`. For each shortlisted cluster:
  1. Fit the free parameters (shape type, aspect ratios, spin-axis angles, phase offset, phase angle). Scale and offset are solved **analytically by least squares**: y\_obs is about a F\_sim + b. Linear flux uses the fit directly; magnitudes use m\_sim = -2.5 log10(F\_sim) + C with a fitted C; unknown convention falls back to a labelled normalised-shape comparison.
  2. Optimise with a coarse random search, then SciPy differential evolution with fixed seeds and restarts.
  3. **Held-out time blocks:** fit on contiguous blocks, score on separate contiguous blocks.
  4. **P versus 2P** as competing interpretations for harmonically linked clusters (odd/even-cycle residual check; AIC/BIC reported when parameter counts differ). If both fit equally, return `prefers='undecided'`.
  5. **Non-harmonic decoys** (for example 1.37 P and 0.61 P), excluding known aliases, harmonic partners and anything within the frequency resolution; equal optimisation budget; Delta E = E\_decoy - E\_candidate on held-out error.
- Fast mode: top-3 shortlist, smaller population, fewer restarts and folds. Fast mode never changes the pass/fail thresholds.
- Failure: a candidate whose fit cannot finish gets `status='failed'`; the whole result has `complete=False`.
- `eval_verification.py` computes ROC-AUC of true-candidate versus decoy separation on held-out simulator variants and writes `docs/results/verification_auc.json` with n. This is synthetic evidence only.
- Accept: on 50 clean synthetic curves the true period beats decoys clearly above chance (the actual AUC is recorded whatever it is); fixed seeds give identical results; Fast mode finishes within about 30 s per candidate on a laptop CPU.

**P2-07 Verdict (Day 3, thresholds Day 4).**

- Files: `verdict/decide.py`, `verdict/reasons.py`, `configs/verdict.yaml`, `tests/p2/test_verdict.py`.
- Contract: `decide(quality, estimators, recon, verification, cfg) -> Verdict`.
  - **Inconclusive:** Stage 1 insufficient, or no candidate passes Stage 2, or no estimator finds a periodic signal.
  - **Supported:** at least one candidate passes Stage 2, its verification completed and passed (held-out error and decoy separation), at least two available estimators agree within tolerance, alias risk is below threshold, and any harmonic partner is resolved in its favour.
  - **Ambiguous:** everything else with plausible candidates (harmonic or alias ambiguity, conflicting methods, model mismatch, incomplete verification).
  - Reason codes are the nine from the README. `best_period_s` is set only for Supported; Ambiguous returns `alternatives`.
  - Confidence is a structured evidence summary plus the measured selective accuracy from validation. It is not the raw CNN probability.
- Thresholds are chosen on the validation split (synthetic validation plus the real pilot `val` split) for a low false-Supported rate, then frozen into `configs/frozen_thresholds.yaml` and tagged before the final test run.
- Accept: table-driven tests cover every branch; verification incomplete can never return Supported; identical input and config gives identical verdict.

**P2-08 Controller (skeleton Day 1, final Day 3).**

- Files: `pipeline/controller.py`, `pipeline/cache.py`, `tests/p2/test_controller.py`.
- Contract: `analyze(obs, cfg, mode='fast', progress=None) -> AnalysisResult`. Runs the chain; wraps each stage so that an exception becomes an entry in `stage_errors` and the pipeline degrades gracefully instead of crashing; records timings; writes a `run_config.json` (config, seeds, versions, commit) so a saved configuration reproduces an analysis; caches by hash of input bytes plus config. `mode='thorough'` is accepted but returns a clear not-implemented message (cut).
- Accept: with every stub, then with each stub replaced by the real module in turn, `analyze()` still returns a full result; killing the CNN or verification produces Ambiguous or a reduced result, never a crash.

**P2-09 Streamlit app (Days 3-4).**

- Files: `app/main.py` (plus `app/pages/` if needed).
- Workflow in the page: upload or demo mode, preview, quality report, Fast run with progress and clear error messages, candidates table with source labels, periodograms, phase-fold view, verification per candidate, verdict card with reason codes, limitation banner, JSON download. Demo mode loads curated real SDLCD examples from `outputs/demo_cache/` and labels them *precomputed*. All text says *apparent (synodic) period*.
- Accept: a first-time user can complete the workflow without instructions; no stack traces are ever shown to the user; the same file gives the same result.

**P2-10 Integration, release and demo support (Days 2-5, continuous).**

- Files: `scripts/run_demo_cache.py`, `scripts/fresh_clone_test.md`, merges to `main`.
- Tasks: merge P1's PRs at each checkpoint (Section 15), replace stubs, resolve contract drift, generate the demo cache, run a fresh-clone install test, maintain the release tag, and drive the live demo.

**P2-11 Optional: 3D fitted-configuration view (Day 4-5, only if ahead).**

- Simple Plotly Mesh3d of the best-fit box or cylinder, titled *plausible fitted configuration, not a reconstruction*. This is cut first if time is short.

---

## 5. Shared interfaces and integration contracts

These are created by P2 in Phase 0 in `core/schemas.py` and described in `CONTRACTS.md`. They are **frozen** once Day 1 starts.

### 5.1 Data structures (final shape, abridged for readability)

```python
# core/schemas.py  (owner P2, FROZEN, SCHEMA_VERSION = 1)
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import numpy as np

BrightnessKind = Literal['mag', 'flux', 'unknown']
GateStatus = Literal['suitable', 'limited', 'insufficient']
StageStatus = Literal['ok', 'failed', 'skipped']
VerdictLabel = Literal['Supported', 'Ambiguous', 'Inconclusive']
REASON_CODES = ('low_quality', 'no_periodic_signal', 'conflicting_methods',
    'harmonic_ambiguity', 'alias_ambiguity', 'candidate_inadequate',
    'model_mismatch', 'geometry_uncertainty', 'verification_incomplete')

@dataclass
class Observation:                  # raw as ingested: no sorting, no cleaning
    obs_id: str
    t_s: np.ndarray                 # float64 seconds since first timestamp
    y: np.ndarray                   # raw brightness, NaN where missing
    y_err: Optional[np.ndarray]
    brightness_kind: BrightnessKind
    meta: dict = field(default_factory=dict)   # object_id, exposure, assumptions

@dataclass
class ReferenceLabel:               # EVALUATION ONLY, never imported by inference code
    obs_id: str
    object_id: str
    period_s: float
    split: Literal['val', 'test']

@dataclass
class PreparedObservation:
    obs_id: str
    t_s: np.ndarray                 # sorted, seconds from first valid sample
    y_norm: np.ndarray              # robust-normalised, higher = brighter
    valid_mask: np.ndarray          # bool; False where missing (never zero-filled)
    dt_s: np.ndarray
    flux_rel: Optional[np.ndarray]  # relative linear flux for physics, None if unknown
    flags: dict                     # bool arrays: bright_outlier, duplicate, gap_edge
    norm_params: dict
    meta: dict = field(default_factory=dict)
    is_stub: bool = False

@dataclass
class QualityReport:               # Stage 1
    status: GateStatus
    metrics: dict
    warnings: list
    reason_codes: list              # subset of REASON_CODES
    is_stub: bool = False

@dataclass
class Candidate:
    period_s: float                 # apparent (synodic) period
    support: float                  # normalised 0-1 within its own estimator
    raw_score: float                # LS power, 1 - theta, or CNN probability
    meta: dict = field(default_factory=dict)

@dataclass
class EstimatorResult:
    method: Literal['cnn', 'lomb_scargle', 'pdm']
    status: StageStatus
    candidates: list                # list[Candidate], best first
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)   # periodogram, p_no_reliable, runtime_s
    is_stub: bool = False

@dataclass
class Cluster:
    cluster_id: int
    period_s: float
    members: dict                   # method -> member period
    evidence: dict                  # method -> normalised support (absent if failed)
    alias_risk: float               # A(P), 0-1
    harmonic_links: list            # [{'other_id': int, 'ratio': 2.0}]
    score: float                    # S(P)
    stage2: dict                    # {'passed': bool, 'cycles': float, 'fail_reasons': [...]}

@dataclass
class ReconciliationResult:
    clusters: list                  # ranked, includes Stage 2 failures
    shortlist: list                 # cluster_ids that passed Stage 2, top K
    stage2_passed_any: bool
    notes: list
    is_stub: bool = False

@dataclass
class CandidateVerification:
    cluster_id: int
    period_s: float
    status: StageStatus
    fit_error: Optional[float]
    heldout_error: Optional[float]
    decoy_periods_s: list
    decoy_delta_e: list             # E_decoy - E_candidate on held-out blocks
    # See CONTRACTS.md §7.6 for the current pair records on VerificationResult.
    passed: bool
    best_params: Optional[dict]     # for the 'plausible fitted configuration' view
    notes: list

@dataclass
class VerificationResult:
    per_candidate: list
    complete: bool                  # False => verdict can never be Supported
    mode: Literal['fast', 'thorough']
    runtime_s: float
    is_stub: bool = False

@dataclass
class Verdict:
    label: VerdictLabel
    best_period_s: Optional[float]  # set only when Supported
    best_cluster_id: Optional[int]
    alternatives: list              # other plausible clusters with reasons
    reason_codes: list
    evidence_summary: dict          # structured evidence, not raw CNN probability
    text: str
    is_stub: bool = False

@dataclass
class AnalysisResult:
    obs_id: str
    prepared: Optional[PreparedObservation]
    quality: Optional[QualityReport]
    estimators: dict                # keys 'cnn', 'lomb_scargle', 'pdm'
    reconciliation: Optional[ReconciliationResult]
    verification: Optional[VerificationResult]
    verdict: Verdict
    stage_errors: dict
    timings_s: dict
    mode: str
    config_hash: str
    schema_version: int = 1
```

### 5.2 Function signatures (exact names, frozen)

```python
# P1
ingestion.load_observation(path: str, fmt: Optional[str] = None) -> Observation   # raises IngestionError
ingestion.reference.load_reference_table(path: str) -> dict   # obs_id -> ReferenceLabel, evaluation only
preprocessing.prepare.prepare(obs, cfg: dict) -> PreparedObservation
preprocessing.quality.assess_stage1(prep, cfg: dict) -> QualityReport
classical_analysis.lomb_scargle.run_lomb_scargle(prep, cfg: dict) -> EstimatorResult
classical_analysis.pdm.run_pdm(prep, cfg: dict) -> EstimatorResult
classical_analysis.window.compute_window(prep, cfg: dict) -> dict
reconciliation.reconcile.reconcile(prep, estimators: list, cfg: dict) -> ReconciliationResult

# P2
ml.infer.run_cnn(prep, cfg: dict) -> EstimatorResult
physics.verify.verify_candidates(prep, recon, cfg: dict, mode='fast', progress=None) -> VerificationResult
verdict.decide.decide(quality, estimators: dict, recon, verification, cfg: dict) -> Verdict
pipeline.controller.analyze(obs, cfg, mode='fast', progress=None) -> AnalysisResult
```

`cfg: dict` is always the module's own YAML section. Each module supplies its own defaults in code, so a missing key never crashes it. Plot functions in `app/plots.py` take contract objects and return Plotly figures.

### 5.3 Rules for errors and failed predictions

| Situation | What happens |
| --- | --- |
| Unreadable or malformed file | `ingestion` raises `IngestionError(message)`. The app shows the message, no traceback. |
| Too little data, no peak found, optimiser failed | The module **returns** a result with `status='failed'` (or `'skipped'`) and an `error` string. It does not raise. |
| An estimator fails | Reconciliation renormalises over the available estimators and records it in `notes`. |
| Unexpected exception in any stage | The controller catches it, writes `stage_errors[stage]`, continues if possible. |
| Verification incomplete | `VerificationResult.complete=False`; verdict is Ambiguous with `verification_incomplete`. |
| Stage 1 insufficient or no candidate passes Stage 2 | Verdict is Inconclusive with the matching reason code. |

### 5.4 Sampling-profile JSON (P1 produces, P2 consumes)

A list of records with keys `profile_id`, `n_points`, `duration_s`, `median_dt_s`, `gap_fractions` (list of gap lengths divided by duration), `noise_rel_amplitude`, `missing_fraction`. Never brightness values or reference periods.

### 5.5 Integration ownership and shared files

- **Integration layer owner: P2** (`pipeline/controller.py`, `app/main.py`).
- **Frozen shared files** (edited only by P2, only after agreement): `core/schemas.py`, `core/errors.py`, `core/config.py`, `CONTRACTS.md`, `tests/fixtures/*`, `tests/test_contracts.py`, `requirements.txt`, `.gitignore`, `configs/default.yaml`.
- **Contract change protocol.** Follow `CONTRACTS.md` §12 for human agreement, P2 classification, review, versioning when required, and synchronization after the change.
- **Automated guards:** `test_contracts.py` checks every function signature in Section 5.2 with `inspect.signature`; `test_no_reference_leak.py` fails if any inference module imports `ingestion.reference` or `ReferenceLabel`.

---

## 6. Repository and folder structure

```text
TumbleScope/
|-- app/
|   |-- main.py                 # P2  Streamlit entry point
|   |-- plots.py                # P1  Plotly figure functions
|-- core/                       # P2  FROZEN: schemas.py, errors.py, config.py
|-- ingestion/                  # P1  sdlcd.py, generic.py, reference.py
|-- preprocessing/              # P1  prepare.py, quality.py, sampling_profile.py
|-- classical_analysis/         # P1  lomb_scargle.py, pdm.py, window.py
|-- reconciliation/             # P1  cluster, harmonics, aliases, score, stage2, reconcile
|-- physics/                    # P2  params, geometry, simulator, fit, verify, eval_verification
|-- ml/                         # P2  dataset/, features.py, model.py, train.py, calibrate.py, infer.py
|-- verdict/                    # P2  decide.py, reasons.py
|-- pipeline/                   # P2  controller.py, cache.py
|-- evaluation/                 # P1  metrics.py, stress.py, run_eval.py
|-- configs/
|   |-- default.yaml            # P2  FROZEN
|   |-- preprocess.yaml         # P1
|   |-- classical.yaml          # P1
|   |-- reconcile.yaml          # P1
|   |-- simulation.yaml         # P2
|   |-- cnn.yaml                # P2
|   |-- verification.yaml       # P2
|   |-- verdict.yaml            # P2
|   |-- frozen_thresholds.yaml  # P2  created Day 4, never edited after tagging
|-- data/
|   |-- sdlcd_raw/              # P1  git-ignored
|   |-- samples/                # P1  3-5 small files, terms checked
|   |-- reference/              # P1  reference_table.csv (evaluation only)
|   |-- profiles/               # P1 sdlcd_profiles.json; P2 default_profiles.json
|   |-- synthetic/              # P2  git-ignored
|-- models/                     # P2  final cnn_v1.pt committed
|-- outputs/demo_cache/         # P2  precomputed demo results
|-- scripts/                    # P2
|-- tests/
|   |-- fixtures/               # P2  FROZEN
|   |-- test_contracts.py, test_stub_pipeline.py, test_no_reference_leak.py   # P2 FROZEN
|   |-- p1/                     # P1
|   |-- p2/                     # P2
|-- docs/                       # P1  (docs/results/verification_* written by P2)
|-- CONTRACTS.md                # P2  FROZEN
|-- README.md                   # P2 creates in Phase 0, P1 owns afterwards
|-- requirements.txt            # P2 only
|-- .gitignore                  # P2 only
```

**Rule of thumb:** a folder or file has exactly one owner. Tests live in `tests/p1` and `tests/p2` so two people never edit the same test file. Only P2 edits `requirements.txt`; P1 asks by message.

---

## 7. Five-day timeline

Assumes about 9-10 productive hours per day with AI coding assistants. Tasks in each day are listed in the suggested order. **Checkpoints** are the only moments when work is merged to `main` (Section 8). The aim is that every day ends with something that runs.

### Phase 0 (before Day 1, 3 h for P2, 45 min for P1)

See Section 2. **Gate:** `v0-skeleton` tagged, stub app and tests run on both machines.

### Day 1 - Real data in, simulator up

**P1 (about 8 h):**

- P1-01 download and audit SDLCD files; write `docs/data_audit.md` (post the typical period range and duration to P2 as soon as known).
- P1-02 ingestion (SDLCD and generic), P1-03 reference table.
- P1-04 preprocessing, P1-05 Stage 1 gate.
- P1-06 sampling profiles JSON.
- Fixture-based tests for each module; start the README skeleton sections (problem, quick start).

**P2 (about 9 h):**

- Finish deferrable Phase 0 items.
- P2-01 simulator and tests.
- P2-02 labelling, No Reliable Period generators, 1,000-curve pilot.
- P2-04 first half: model class, receptive-field and parameter tests, overfit-one-batch check.
- Review and merge P1's PR at CP1.

**Deliverables:** P1: any downloaded real SDLCD file goes through `load_observation` -> `prepare` -> `assess_stage1`; profiles JSON. P2: simulator passing its tests; labelled pilot; model skeleton.

**Tested today:** all module unit tests; a real file loads and gets a quality status.

**Integration:** CP1 (end of Day 1): each person opens a PR to `main`; P2 merges, swaps the real ingestion, preprocessing and Stage 1 into the controller (30 min), and checks the stub app still runs.

**Must be done before Day 2:** real file -> quality report works; profiles file exists (or the default is confirmed); simulator tests green.

### Day 2 - Classical analysis live, CNN training started, verification v0

**P1:**

- P1-07 Lomb-Scargle (1-term and multi-term).
- P1-08 PDM.
- P1-09 spectral window; then begin the reconciliation skeleton against `mock_cnn_result.json`.
- Continue README and `docs/technical_doc.md` outline.

**P2:**

- P2-03 degrader and shared feature builder.
- Generate the 5,000-curve corpus; **launch the first Colab training run early in Day 2, before moving on**.
- While training runs: P2-06 part 1 (fit machinery: analytic scale/offset, coarse search, differential evolution on a single candidate, tested on a synthetic curve).
- Then check the training curve; fix obvious problems; leave a second run going if useful.

**Deliverables:** P1: LS and PDM return valid `EstimatorResult`s on real files. P2: first CNN checkpoint (even if weak); single-candidate fit working.

**Tested today:** LS recovers a known period within 1 percent on gapped synthetic data; PDM agrees with LS on clean data; training loss decreases.

**Integration #1 (CP2, end of Day 2):** P2 plugs LS and PDM into the controller. **Thin slice:** upload a real SDLCD file in Streamlit and see quality status plus LS and PDM candidates. This matches Roadmap phases 0-2.

**Must be done before Day 3:** thin slice runs on at least two real files; a CNN checkpoint exists (quality can be bad).

### Day 3 - Core science complete, first end-to-end run

**P1:**

- P1-10 reconciliation and Stage 2 (clustering, harmonic links, alias risk, score, shortlist). **CP3a, as soon as P1-10 passes its tests:** push so P2 can plug it in.
- P1-11 plots module.
- Start P1-12 `metrics.py` and `stress.py` (pure functions with unit tests).

**P2:**

- P2-05 CNN evaluation, calibration and `run_cnn`. If the 5k run is weak, start the 20k run now.
- P2-06 part 2: held-out blocks, harmonic comparison, decoys, Fast mode budget; develop against `mock_reconciliation.json` until CP3a, then against the real one.
- P2-07 verdict v1 with table-driven tests.
- After CP3a: plug reconciliation into the controller.

**Deliverables:** P1: reconciliation and plots merged. P2: calibrated CNN, verification, verdict v1, full controller.

**Tested today:** reconciliation on the double-humped synthetic case (P and 2P linked); verification true-versus-decoy sanity check on synthetic curves; every verdict branch.

**Integration #2 (CP3b, end of Day 3):** merge everything; run the full pipeline on three real files. **First end-to-end real run.** A wrong or Ambiguous answer is fine at this point; a crash is not.

**Must be done before Day 4:** one real file goes upload -> verdict in the app with no crash. If not, apply the Section 12 fallbacks before Day 4 starts.

### Day 4 - Evaluation, dashboard, calibration, freeze

**P1:**

- P1-12 `run_eval.py`: classical baselines (LS only, PDM only) on the real pilot `val` split and synthetic holdout; first stress tests.
- Tune reconciliation weights and Stage 2 thresholds on the `val` split only; edit `reconcile.yaml`.
- Full-system evaluation with the stable controller; the five configurations; stress tests; tables and figures (risk-coverage, calibration, per-quality-group results) into `docs/results/`.
- Technical documentation draft, limitations, future work, screenshots list started.

**P2:**

- P2-09 Streamlit app (upload and demo mode, quality, candidates, periodograms, verdict card), wiring P1's plots.
- Verdict threshold selection on validation outputs from P1 (low false-Supported rate), write `frozen_thresholds.yaml`, tag `thresholds-frozen`.
- Run `eval_verification.py` for AUC on held-out simulator variants; generate the demo cache with `scripts/run_demo_cache.py`.
- Bug fixing; optional P2-11 3D view only if everything above is green.

**Integration #3 (CP4, end of Day 4):** plots wired into the app; evaluation harness runs on the real pipeline. **Feature freeze at CP4.** After this only bug fixes.

**Tested today:** full pipeline on all pilot `val` observations without crashing; app workflow end to end; reproducibility (the same saved config gives the same result).

**Must be done before Day 5:** thresholds and weights frozen and tagged; demo cache exists; docs drafts exist.

### Day 5 - Final test, polish, demo, submission

**P1:**

- Run `run_eval.py` **once** on the frozen pilot `test` split and the synthetic holdout; write final tables and figures. If a bug forces a re-run, disclose it in the docs.
- Finalise `README.md`, `docs/technical_doc.md`, `docs/limitations.md`, `docs/future_work.md`; every number cites a file in `docs/results/`.
- Capture screenshots; record the demo video together with P2.
- Claim check (Section 11.5).

**P2:**

- Critical bug fixes only; fresh-clone install test on a clean folder; confirm the demo cache and live path both work.
- Measure runtime (p50 and p95 per file, Fast mode) with a short script; give P1 the numbers.
- Demo rehearsal, then video recording (P2 drives the app, P1 captures and edits).
- Final merge, tag `v1.0-submission`, push, verify the repo link from a logged-out browser.
- Last: buffer, no new features.

**Deliverables:** tagged repository, README, technical doc, results, video, screenshots, live-demo plan with fallback.

**Integration:** final merge only. All previous checkpoints mean there should be no new conflicts.

---

## 8. Git and merge-conflict prevention strategy

### 8.1 Branches (keep it simple)

- `main`: always runnable. Only P2 merges into it.
- `p1/dev` and `p2/dev`: each person's long-lived working branch. You only commit to your own.
- Optional short branches (for example `p1/reconcile`) are allowed but not needed.
- Tags: `v0-skeleton` (Phase 0), `thresholds-frozen` (Day 4), `v1.0-submission` (Day 5).

### 8.2 Daily routine (copy and paste)

Start of day, and before each sync:

```bash
git checkout p1/dev            # or p2/dev
git fetch origin
git merge origin/main          # bring in the other person's merged work
pip install -r requirements.txt
pytest -q
```

During the day, after each finished task:

```bash
git status                     # check: only files you own?
git add <your folders>         # avoid blind 'git add .'
git commit -m 'P1-07: Lomb-Scargle with tests'
git push origin p1/dev
```

### 8.3 Checkpoint procedure (see Section 15)

1. Owner runs `pytest -q` on their branch and confirms the acceptance criteria of the finished tasks.
2. Owner opens a pull request from their branch to `main` on GitHub with a one-line description.
3. P2 reviews for 10 minutes (do the tests pass? only owned files changed?) and merges. P2's own work goes through the same PR step.
4. Both run the daily routine above to pull `main` back into their branches.
5. P2 runs the app once as a smoke test.

| Checkpoint | Trigger | Content |
| --- | --- | --- |
| CP1 | End of Day 1 | Ingestion, preprocessing, Stage 1, profiles; simulator, labels |
| CP2 | End of Day 2 | LS, PDM; fitting machinery, model |
| CP3a | When P1-10 passes its tests | Reconciliation and Stage 2 |
| CP3b | End of Day 3 | Plots, CNN, verification, verdict |
| CP4 | End of Day 4 | Evaluation harness, app, frozen thresholds |
| CP5 | Day 5, final | Final fixes, docs, tag |

### 8.4 How we avoid editing the same file

- One owner per folder and file (Section 6). The task blocks list the files for each AI-agent prompt, and the prompt tells the agent not to touch anything else.
- Cross-person needs go by message, never by editing: a new dependency, a new field, a bug in the other person's module (send the failing input), a config tweak.
- Per-owner YAML files mean threshold tuning never collides.
- Each person has their own test folder and their own fixtures, except the frozen shared fixtures.
- Large or generated files (raw data, synthetic corpora, training logs) are git-ignored. Only small samples and the final checkpoint are committed.
- Notebooks (Colab training) live only in P2's folder.

### 8.5 Dependency conflicts and incompatible signatures

- Only P2 edits `requirements.txt`. If P1 needs a package, P1 messages P2, who adds it, pushes to `main`, and both run `pip install -r requirements.txt`.
- Never install a package locally without telling P2 (that creates the works-on-my-machine bug).
- `tests/test_contracts.py` compares every public signature with Section 5.2. If a merge breaks it, the person whose function changed fixes it before the merge is accepted.
- If a contract genuinely must change, follow the change protocol in Section 5.5.

### 8.6 Testing modules independently before merging

- P1 runs `pytest tests/p1 -q`; P2 runs `pytest tests/p2 -q`; both run `pytest -q` before opening a PR.
- P1 develops against fixtures (`mock_cnn_result.json`, `mock_prepared.npz`, `synthetic_sine.csv`) and real sample files. P2 develops against `mock_reconciliation.json` until the real one arrives.
- Each module has a tiny smoke command, for example `python -m classical_analysis.lomb_scargle data/samples/<file>` that prints candidates, so a failing module can be checked in seconds without the app.

### 8.7 If a merge conflict appears anyway

That means someone edited a file they do not own. Do not try to fix it in a panic:

1. Run `git merge --abort`.
2. Message the other person with the file name.
3. Decide which version to keep. The owner's version wins. Restore the other person's accidental change from the owner's branch.
4. Re-run the merge.

---

## 9. Feature priority table

| Priority | Features |
| --- | --- |
| **Must build** | SDLCD + CSV ingestion with real-file audit; preprocessing (flag-not-delete, no detrending); Stage 1 gate; Lomb-Scargle (single and multi-term); custom PDM; reconciliation with harmonic and alias analysis and Stage 2; top-K shortlist (K=5, or 3 in Fast mode); box and cylinder diffuse single-axis simulator; synthetic corpus of 5k+ curves with repeat-shift labels and No Reliable Period examples; compact CNN with calibration (ECE, Brier); physics verification with held-out blocks, decoys and P versus 2P; three-verdict logic with reason codes; Streamlit app with demo mode; evaluation of five configurations on a real pilot and synthetic holdout; at least three stress tests; saved-configuration reproducibility; README, technical document, limitations, future work |
| **Should build** | 20k corpus if training is stable; all five stress tests; ablations that come free from `AnalysisResult` (no physics verification, no alias/harmonic checks); risk-coverage and calibration figures; verification ROC-AUC on held-out simulator variants; runtime measurements (p50 and p95); progress and error messages; weight tuning on the validation split |
| **Optional** | Simple Plotly 3D view of the fitted configuration; timing-channel and augmentation ablations (need retraining); gradient-boosting baseline on periodogram features; ONNX export test |
| **Cut from this build** | Thorough mode; solar-panel geometry; Phong specular term; 50k corpus; torque-free tumbling; SGP4/CelesTrak geometry; stable-spin versus tumbling classification; FastAPI, React, Docker, job queue; batch UI; full object-disjoint SDLCD evaluation (replaced by a small, disclosed pilot); 3D polish |

This ordering protects the evaluation criteria: technical soundness and architecture (reconciliation, verification, honest verdicts), feasibility (cut list and fallbacks), innovation (integration of ML, signal analysis and physics with abstention), applicability (documented extensions), and user experience (a clean Streamlit flow and demo mode).

---

## 10. End-of-Day-5 prototype specification

### 10.1 User workflow

1. Open the Streamlit app. Choose **Upload** (SDLCD, CSV or TXT) or **Demo mode** (curated real SDLCD examples, labelled *precomputed*).
2. See a preview of the lightcurve and any ingestion assumptions (time and brightness conventions).
3. Inspect the Stage 1 quality report (suitable, limited or insufficient, with reasons).
4. Press **Run (Fast mode)** and watch progress.
5. Review the three estimators side by side, clearly labelled CNN, Lomb-Scargle, PDM, with periodograms and the CNN distribution.
6. Review the reconciled candidate table: clusters, harmonic partners, alias risk, Stage 2 result, shortlist.
7. Review physics verification per candidate: held-out error, decoy separation, P versus 2P.
8. Read the verdict card: Supported, Ambiguous or Inconclusive, apparent (synodic) period if Supported, alternatives, reason codes and the limitation banner.
9. Download the analysis JSON; the saved run configuration reproduces the result.

### 10.2 What must genuinely work

Ingestion of real SDLCD files, preprocessing, Stage 1, LS, PDM, reconciliation with harmonic and alias handling and Stage 2, the simulator, labelled synthetic data, a trained and calibrated CNN (reported as experimental if it underperforms), physics verification in Fast mode, verdict logic with reason codes, the app, and the evaluation harness with measured numbers.

### 10.3 What is simplified (and labelled so)

- Physics: box and cylinder, diffuse reflection, single-axis rotation, fixed or simplified geometry. The result shows compatibility with a model, not the object's true shape or spin.
- Real-data evaluation: a small pilot with sample size shown, separate from synthetic results.
- Demo mode: precomputed results, labelled as such.
- Thresholds: tuned on a small validation set; the measured false-Supported rate is reported with its sample size.

### 10.4 What is presented only as future work

Thorough mode, solar panels and specular reflection, torque-free tumbling, stable-spin versus tumbling classification, orbital geometry from SGP4, larger corpus, full object-disjoint SDLCD evaluation, FastAPI and batch processing, ONNX export.

---

## 11. Submission and demonstration plan

### 11.1 Deliverables and owners

| Required item | Owner | When prepared |
| --- | --- | --- |
| Working prototype | P2 (app, integration), P1 (plots, evaluation) | Day 4 feature freeze, polish Day 5 |
| Live demo or video | P2 drives, P1 captures and edits | Rehearsal at CP4, recording on Day 5 after the final test run |
| Repository link | P2 | Tag `v1.0-submission` Day 5 |
| Technical documentation (`docs/technical_doc.md`) | P1, with P2 supplying simulator, CNN and verification facts | Draft Day 4, final Day 5 |
| Implemented features and limitations (README plus `docs/limitations.md`) | P1 | Day 5 midday |
| Future development (`docs/future_work.md`) | P1 | Day 4 |

Documentation is written in daily slices, not on Day 5 night.

### 11.2 Demo structure (about 5-6 minutes)

1. **Problem and idea** (45 s): lightcurves, harmonic ambiguity, the integrated workflow.
2. **Live run** (2.5 min): upload one real SDLCD file and show quality, three estimators, reconciliation, verification and the verdict. Then open the demo cache for two more examples. Aim for three outcomes (Supported, Ambiguous, Inconclusive) **if the real results provide them**. Choose examples from the validation split, state they are curated, and never present a hand-picked success as typical.
3. **Measured results** (1.5 min): the five-configuration table, stress-test summary, false-Supported rate with n, calibration and risk-coverage figures, runtime.
4. **Limitations** (45 s): synthetic-to-real gap, apparent versus physical period, simplified physics, small pilot.
5. **Future work** (30 s).

**Fallback:** record the full demo as a video on Day 5 so that a network, laptop or Colab problem never blocks the live session. The live demo runs entirely on CPU from the cache plus a live Fast run of one pre-tested file.

### 11.3 Screenshots, outputs and measurements to capture

- Screenshots: upload and preview; Stage 1 report; side-by-side periodograms; candidate table with harmonic links; verification view; one verdict card for each verdict type obtained; limitation banner.
- Figures from `docs/results/`: five-configuration comparison, stress tests per quality group, risk-coverage, calibration, verification AUC with n, CNN training curve.
- Measurements: p50 and p95 Fast-mode runtime per file, CNN inference time, simulator speed, corpus size actually used, number of real objects in the pilot.
- Example outputs: one analysis JSON and its `run_config.json`.

### 11.4 Technical document outline

Overview; apparent versus physical period; architecture diagram; each module (inputs, outputs, method); shared simulator; synthetic data; CNN; reconciliation; verification protocol; verdict logic; evaluation design and **measured** results; limitations; reproducibility instructions; future work.

### 11.5 Claim check (Day 5, 30 minutes, both people)

Go through the README, document, slides and video script. For every number or capability ask: which file in `docs/results/` produced it, or is it labelled planned? Remove anything that fails. Confirm that nowhere is apparent period called physical rotation period, that Supported is never described as proof, and that the synthetic-to-real gap is stated.

### 11.6 Future-development roadmap (short and realistic)

1. Thorough mode and larger search budgets.
2. Solar panels and specular reflection; richer geometry library.
3. Larger corpus and a calibration study across observation-quality groups.
4. Full object-disjoint SDLCD evaluation.
5. SGP4/CelesTrak observation geometry; torque-free tumbling; stable-spin versus tumbling classification.
6. Batch processing, FastAPI backend, ONNX/CPU inference export.

---

## 12. Risks, fallbacks and contingency plan

| Component | What could go wrong | Fallback | Drop first |
| --- | --- | --- | --- |
| SDLCD parsing (P1) | Unexpected format or time convention | Generic CSV adapter plus hand-converted samples while the SDLCD adapter is finished; record assumptions in the audit | n/a, this is a must |
| Real-data pilot (P1) | Too few objects with valid periods | Report as a small pilot with n; lean on synthetic holdout and stress tests; no accuracy claims | Object-disjoint test claim |
| LS / PDM (P1) | Slow grid; period bounds wrong | Coarser grid; bounds from the audit; downsample very long files | Multi-term option |
| Reconciliation (P1) | Alias logic too complex | Ship clustering plus harmonic links plus basic cycle-count Stage 2; simplify alias risk to window-peak proximity | Refined alias penalty |
| Simulator (P2) | Slow or wrong geometry | Box only (state it on the slide); vectorise; cache facet matrices | Cylinder |
| Synthetic data (P2) | Labels noisy or distribution too easy | Pilot-check on 20 curves; widen parameter ranges; keep the NRP class | Larger corpus |
| CNN (P2) | Weak accuracy or training too slow | 5k corpus, fewer epochs, smaller network; if still weak, run LS + PDM + reconciliation and report the CNN as experimental with measured results | Second 20k run, ablations |
| Verification (P2) | Differential evolution too slow | Fewer folds and restarts, smaller population, top-3; if not ready by Day 4 noon, verdicts return Ambiguous (verification\_incomplete), never Supported | Decoy variety, AIC/BIC detail |
| Verdict calibration (P2) | Too few validation objects for stable thresholds | Conservative thresholds that favour abstention; report coverage honestly | Fine threshold tuning |
| App (P2) | Integration slips | Minimal Streamlit page showing one cached real example end to end | 3D view, extra plots |
| Evaluation (P1) | Harness not ready | Report baselines and the synthetic holdout only; state the real pilot as pending | Some stress tests |

**Drop order if behind:** 3D view, ablations needing retraining, extra stress tests, 20k corpus, cylinder, weight tuning, multi-term LS. Never drop: honest verdict logic, harmonic and alias handling, quality gates, documentation of limitations.

**Biggest integration risks and prevention**

1. **Contract drift** -> frozen schemas, signature tests, change protocol.
2. **Training and inference feature mismatch** -> one shared `ml/features.py`.
3. **Late discovery that modules do not connect** -> walking skeleton plus Integration #1 on Day 2 and #2 on Day 3.
4. **Reference-label leakage** -> separate type and an automated import test.
5. **Over-claiming** -> claim check; every number sourced from `docs/results/`.
6. **P2 bottleneck** -> P1 tasks are independent of P2 after Phase 0; if P2 is blocked, P1 continues with fixtures and can pre-write documentation and evaluation code.
7. **Works on my machine** -> single requirements file, fresh-clone test on Day 5.

---

## 13. Final task checklist

### Person 1

- [ ] Phase 0: clone, install, run tests and the stub app, read CONTRACTS.md, confirm to P2
- [ ] P1-01 SDLCD audit and `docs/data_audit.md`
- [ ] P1-02 ingestion (SDLCD and generic)
- [ ] P1-03 reference table with object-level `val` and `test` split
- [ ] P1-04 preprocessing
- [ ] P1-05 Stage 1 gate
- [ ] P1-06 sampling profiles JSON
- [ ] P1-07 Lomb-Scargle
- [ ] P1-08 PDM
- [ ] P1-09 spectral window
- [ ] P1-10 reconciliation and Stage 2
- [ ] P1-11 plots
- [ ] P1-12 metrics, stress tests, `run_eval.py`; weights tuned on `val` only; final run once on `test`
- [ ] P1-13 README, technical document, limitations, future work, demo script, screenshots
- [ ] Claim check, with sources for every number
- [ ] Pull requests opened at all five syncs, tests green

### Person 2

- [ ] Phase 0: repo, branches, `.gitignore`, `requirements.txt`, contracts, stubs, fixtures, stub app, tests, `v0-skeleton`, handoff
- [ ] P2-01 simulator
- [ ] P2-02 labels and pilot dataset
- [ ] P2-03 degrader and shared feature builder
- [ ] P2-04 CNN training (5k, then 20k if possible)
- [ ] P2-05 calibration and inference
- [ ] P2-06 physics verification and AUC script
- [ ] P2-07 verdict, thresholds frozen and tagged
- [ ] P2-08 controller with graceful degradation and run-config saving
- [ ] P2-09 Streamlit app and demo mode
- [ ] P2-10 integration, demo cache, fresh-clone test, runtime measurements, tag `v1.0-submission`
- [ ] Optional P2-11 3D view
- [ ] Demo rehearsal and video recording
- [ ] Repository link verified from a logged-out browser

### Both

- [ ] Evaluation criteria checked against the demo (functionality, technical execution, usability, innovation, applicability, demo quality)
- [ ] Submission package complete: prototype, demo or video, repository, technical document, implemented features and limitations, future work

---

## 14. Addendum: gaps found in review (these override earlier sections where they differ)

**A1. Git-ignored files that P1 needs.** `data/synthetic/` and `models/*.pt` are git-ignored, but P1 needs a small synthetic holdout set and a CNN checkpoint for evaluation and weight tuning on Days 3-4. Rule: by **CP3b (end of Day 3)** P2 force-adds (`git add -f`) one small holdout shard `data/synthetic_holdout/holdout_small.npz` (a few hundred curves, under about 5 MB) and the current checkpoint `models/cnn_v1.pt` (about 1 MB). `.gitignore` gets exceptions for exactly these two paths in Phase 0. P1's evaluation code reads from `data/synthetic_holdout/` and must not assume the full corpus exists. If the CNN is still weak, P2 commits it anyway, marked experimental in its file name or metadata, so P1 is never blocked.

**A2. Harmonic partners and the shortlist.** The shortlist holds the top K (5, or 3 in Fast mode) clusters that passed Stage 2. In addition, any cluster linked by `harmonic_links` (ratio 2 or 1/2) to a shortlisted cluster, and that also passed Stage 2, is **always included in verification** for the P versus 2P comparison. Partners do not count toward K. Reconciliation (P1) lists partner ids in `notes` and keeps them in `clusters`; verification (P2) looks them up through `harmonic_links`. If a partner failed Stage 2, verification records a skipped diagnostic pair as defined in [CONTRACTS.md §7.6](CONTRACTS.md#76-verificationresultharmonic_comparisons).

**A3. Period bounds must match.** P1's `classical.yaml` (`min_period_s`, `max_period_s`) and P2's `cnn.yaml` (`p_min_s`, `p_max_s`, 128 log-spaced bins) must use the same range. Single source of truth: the range written at the end of `docs/data_audit.md` (P1-01) and copied into `CONTRACTS.md` by P2 on Day 1 as a documented constant. Both configs then copy those two numbers. A test in `tests/test_contracts.py` (P2) loads both YAML files and asserts the bounds agree.

**A4. Conventions that must be written into `CONTRACTS.md` in Phase 0.** Types match through the contract tests, but meaning does not, so these are stated explicitly:

| Convention | Rule |
| --- | --- |
| Time | All `t_s` arrays are float64 **seconds**. Periods are in seconds everywhere (`period_s`). |
| Brightness direction | `y_norm`: higher means brighter, for magnitudes and flux alike. Normalisation is median and MAD. |
| Missing data | `valid_mask=False`; the value under it is never used and never zero-filled. |
| Support | `Candidate.support` is normalised to 0-1 **within its own estimator**, 1 being that estimator's best peak. It is not comparable across estimators until reconciliation scores it. |
| Candidate order | Candidates are sorted best first. Clusters are sorted by score, best first. |
| Harmonic ratio | `ratio` in `harmonic_links` is other period divided by this period (2.0 means the other is double). |
| IDs | `cluster_id` is unique within one `ReconciliationResult` only. |
| Period meaning | Every period is the **apparent (synodic)** period. |
| Frequency resolution | Defined once as 1 divided by the observation baseline; P1 computes it and stores it in `meta['freq_resolution']` for P2 to reuse in decoy exclusion. |

Add a second integration check: at Integration #1 (CP2) P1 and P2 run one real file together and compare the printed `PreparedObservation` and `EstimatorResult` values by eye (units, direction, ranges) before anything else is plugged in.

**A5. `docs/results/` ownership.** P2 owns every file it generates there: `verification_*`, `cnn_training_log.json`, `cnn_calibration.json`, `runtime_*.json`. P1 owns everything produced by `evaluation/run_eval.py`. The two never write the same filename.

**A6. Test-folder setup in Phase 0.** P2 creates `tests/__init__.py`, `tests/p1/__init__.py`, `tests/p2/__init__.py` and a root `conftest.py` so that both `pytest tests/p1` and `pytest tests/p2` work on a fresh clone, and so `pytest -q` collects both folders without name collisions (test file names inside the two folders are never reused).

**A7. Meaning of a checkpoint.** A checkpoint is the moment when a person's finished work is merged from their own branch into `main` through a pull request and the other person pulls `main` back into their own branch. Between checkpoints each person works only on their own branch and does not need to commit or merge with the other. The checkpoints are listed in Section 15.

---

## 15. Checkpoints (the schedule runs on Day and Phase checkpoints, not clock times)

Work is organised by **Day and Phase checkpoints**, triggered by finished tasks, not by the clock.

### 15.1 What a checkpoint is, and is not

A checkpoint is a merge of **already finished and tested** work into `main`. It is **asynchronous**: each person reaches it independently, opens a pull request when their own tasks for that phase pass their acceptance tests, and P2 merges it. Nobody has to be finished at the same moment, and **P2 never stops working to wait for P1** (or the other way round). Whoever finishes first merges first. Whatever arrives later is merged when it arrives, at any time, and then P2 plugs it in.

### 15.2 Checkpoint list

| Checkpoint | Trigger | P1 contributes | P2 contributes | Integration step after both have merged |
| --- | --- | --- | --- | --- |
| **CP0: end of Phase 0** | Skeleton ready | Confirms setup | Repo, contracts, stubs, fixtures, tag `v0-skeleton`; CP0 fixtures and active guards under CONTRACTS.md §§13–14 | Both run tests and the stub app |
| **CP1: end of Day 1** | Day 1 tasks pass their tests | P1-01 to P1-06 (ingestion, reference table, preprocessing, Stage 1, profiles) | P2-01, P2-02 and the model skeleton from P2-04; audited bounds recorded in period_bounds.yaml and configured-bounds and simulator-side checks passing | P2 swaps real ingestion, preprocessing and Stage 1 into the controller |
| **CP2: end of Day 2** | Day 2 tasks pass their tests | P1-07 to P1-09 (LS, PDM, window) | P2-03, first CNN training run launched, fit machinery from P2-06; golden CNN archive, numerical feature guard, and generator shared-builder check | **Integration #1:** thin slice, real file through to LS and PDM candidates in the app |
| **CP3a: when P1-10 is done (Day 3)** | Reconciliation passes its tests | P1-10 reconciliation and Stage 2 | Whatever P2 has finished; real Stage 2 root guard against P1-10 | P2 plugs real reconciliation into verification and the controller |
| **CP3b: end of Day 3** | Day 3 tasks pass their tests | P1-11 plots, start of P1-12 metrics | P2-05, P2-06, P2-07, P2-08; real CNN inference shared-builder and checkpoint compatibility checks | **Integration #2:** first full real-file run, upload to verdict; P2 also commits the holdout shard and checkpoint (A1) |
| **CP4: end of Day 4, feature freeze** | Evaluation, app and thresholds ready | Evaluation outputs, tuned reconciliation weights | App, frozen thresholds, demo cache, verification AUC | **Integration #3:** plots in the app, evaluation on the real pipeline. After this, bug fixes only |
| **CP5: Day 5, final** | Final test run and docs done | Final results, README, technical document | Fixes, runtime numbers, fresh-clone test, tag `v1.0-submission` | Final merge, no new conflicts expected |

The daily task lists in Section 7 still define what each person does each day. The Gate lines (Must be done before Day N) are checked at the start of the next day, not at a clock time.

### 15.3 Does one person have to finish before the other can continue? No.

Each person's tasks run against fixtures and stubs, so neither waits for the other. The table shows what each side uses until the real module arrives and what happens if it is late.

| Dependency | Who is waiting | Works against in the meantime | If the other side is late |
| --- | --- | --- | --- |
| Real ingestion, preprocessing, Stage 1 | P2 (controller) | Stub modules, `mock_prepared.npz`, `synthetic_sine.csv` | P2 continues on simulator, labels and CNN; integration happens when it lands |
| Sampling profiles | P2 (degrader) | `default_profiles.json` | Train on default profiles; swap in the real ones for the second run if time allows |
| LS, PDM | P2 (controller) | Stub estimators | App runs on stubs; thin slice waits, nothing else does |
| Reconciliation | P2 (verification, verdict) | `mock_reconciliation.json` | P2 finishes verification and verdict on the mock; real reconciliation is plugged in when ready. If still missing at the start of Day 4, use the simplified reconciliation fallback from Section 12 |
| Real `analyze()` and CNN outputs | P1 (evaluation, weight tuning) | Stub `analyze()`, mock CNN fixture, synthetic holdout shard | P1 builds and unit-tests metrics and stress tests and writes documentation; real runs start when the controller lands |
| Plots | P2 (app) | Simple built-in Streamlit tables and charts | App ships with basic charts; P1 plots replace them when ready |
| Frozen thresholds | P1 (final test run) | None needed before freeze | Final test run starts only after `thresholds-frozen` is tagged; if P2 is late, P1 writes documentation first |

**The only real waiting points** are the ones where the later step genuinely needs the earlier one: P2 cannot run the first full end-to-end analysis until reconciliation exists, and P1 cannot run the final test evaluation until thresholds are frozen. In both cases the waiting person has other tasks to do (listed above), so nobody sits idle, and the fallbacks in Section 12 apply if a dependency slips past its Day.

### 15.4 Updated routine at each checkpoint

1. When your phase tasks pass their acceptance tests, run `pytest -q` on your branch.
2. Open a pull request from your branch to `main`.
3. P2 checks tests and file ownership and merges (P2's own work goes through the same step).
4. Both run `git fetch origin`, `git merge origin/main` and `pip install -r requirements.txt` on their own branch, then carry on with the next tasks.

---

## 16. Addendum 2: reason codes, schedule realism and CNN specification (overrides earlier sections where they differ)

### 16.1 Reason codes: one source of truth

**Decision.** The nine top-level reason codes stay exactly as in the README and the submitted Round 2 deck: `low_quality`, `no_periodic_signal`, `conflicting_methods`, `harmonic_ambiguity`, `alias_ambiguity`, `candidate_inadequate`, `model_mismatch`, `geometry_uncertainty`, `verification_incomplete`. Renaming them would make the code disagree with what has already been submitted. A review suggested `insufficient_cycles` and `alias_suspected` as top-level names; they do not appear in the README or either deck, so they are not adopted as top-level codes. `insufficient_cycles` is adopted as a **Stage 2 sub-reason** (below), which gives the same information without renaming anything. If you do want different names, change them everywhere in one pass before Day 1, with only P2 editing `core/schemas.py`.

**Stage 2 sub-reasons (new, fixed list).** Stored in `Cluster.stage2['fail_reasons']`, defined once as `STAGE2_FAIL_REASONS` in `core/schemas.py`:

- `insufficient_cycles`: the baseline covers fewer cycles than the minimum.
- `unresolved_frequency`: period uncertainty from the frequency resolution is too large relative to the period.
- `poor_phase_coverage`: folded data leave too much of the phase range empty.
- `merged_into_neighbour`: indistinguishable from a neighbouring cluster at the available resolution.

**Mapping from situation to verdict and codes (this resolves what happens when no candidate passes Stage 2).** Checks run top to bottom; the first matching Inconclusive row ends the decision.

| Situation | Verdict | Reason code(s) |
| --- | --- | --- |
| Stage 1 status is insufficient | Inconclusive | `low_quality` |
| No estimator finds periodic structure, or the CNN abstains and classical peaks are not significant | Inconclusive | `no_periodic_signal` |
| No candidate passes Stage 2 | Inconclusive | `candidate_inadequate`, with the Stage 2 sub-reasons copied into `evidence_summary['stage2_fail_reasons']` |
| Estimators disagree on the leading period | Ambiguous | `conflicting_methods` |
| P versus 2P not resolved by verification | Ambiguous | `harmonic_ambiguity` |
| Alias risk above threshold | Ambiguous | `alias_ambiguity` |
| Fits are poor for every candidate | Ambiguous | `model_mismatch` |
| Different geometries fit equally well | Ambiguous | `geometry_uncertainty` |
| Verification did not finish | Ambiguous | `verification_incomplete` |
| All requirements met | Supported | none |

Several Ambiguous codes may apply at once; all that apply are returned.

**Keeping the agents from inventing names.**

1. The lists live only in `core/schemas.py` (`REASON_CODES`, `STAGE2_FAIL_REASONS`). Nobody types a code as a free string; code imports the constants.
2. `CONTRACTS.md` contains the table above, copied verbatim.
3. New test `tests/test_reason_codes.py` (P2, frozen): every string assigned to a reason or fail-reason field in `verdict/`, `reconciliation/` and `preprocessing/` must be in the constants; the README's reason-code list must equal `REASON_CODES`.
4. Every task prompt for P1-05, P1-10 and P2-07 includes the constants and this table.

### 16.2 Schedule realism

The plan is a **sequence of priorities, not a guarantee** that every Must build item will be finished. The task estimates of 1-3 hours are best-case figures for generating code. Debugging, integration and checking the science against real data will often double them, and this is especially likely for the first ML pipeline you build. Treat the five days as a budget to spend in priority order.

**Rules for running the schedule**

1. **Time-box each task.** If a task passes about twice its estimate, stop polishing. Apply the fallback from Section 12 and move on.
2. **Daily check, 10 minutes at the start of each day.** Ask: did yesterday's checkpoint happen, and what is the next item in the drop order? If you are more than half a day behind, cut the next item now, not on Day 4.
3. **Protect the order of value.** Anything that makes the verdict honest (quality gates, harmonic and alias handling, verification-incomplete rule, abstention) comes before anything that makes results look better (more training data, extra plots, 3D view).
4. **The pipeline must work without the CNN.** If CNN training eats the schedule, ship the experimental CNN label and keep going.
5. **Report what exists.** The submission describes what was built and measured. A smaller honest prototype beats a larger claimed one.

**Minimum credible prototype if you fall behind**

| Reached | What the demo can honestly show |
| --- | --- |
| CP2 only | Real SDLCD file in, quality report, Lomb-Scargle and PDM candidates, honest Inconclusive or Ambiguous outputs, limitations stated |
| CP3b | Adds reconciliation with harmonic and alias handling, Stage 2, simulator-based verification (possibly Fast mode only), CNN reported as experimental, full verdict logic |
| CP4 | Adds the five-configuration evaluation, stress tests and thresholds frozen on validation data |
| CP5 | Final test run, documentation, video, tag |

If the schedule assumption of 9-10 productive hours a day is not realistic for you, decide the drop order on Day 1 instead of discovering it on Day 4: use the cut order in Section 12 (3D view, ablations needing retraining, extra stress tests, 20k corpus, cylinder, weight tuning, multi-term LS).

### 16.3 CNN specification clarification

The P2-04 text has been edited to state the rule explicitly: **one dilated convolution per residual block**, kernel 7, dilations 1, 2, 4, 8, 16, 32, with a 1x1 skip projection when channels change. That gives a receptive field of 379 and about 0.25 M parameters. The test in `tests/p2/test_model.py` asserts both numbers against the implemented network, so a model with two convolutions per block would fail the test deliberately. If the coding agent proposes a different block design, P2 decides whether to keep the original architecture or change the architecture, the test, the README and the slides together.
