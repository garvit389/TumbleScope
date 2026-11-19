# CONTRACTS.md

**TumbleScope interface contracts. Architecture owner: P2. Status: FROZEN at the end of Phase 0 (tag `v0-skeleton`). `SCHEMA_VERSION = 1`.**

This file is the single source of truth for how modules talk to each other. P1, P2 and every AI coding agent must follow it. If code and this file disagree, this file wins and the code is wrong. Do not change anything here except through Section 12.

> Give this whole file to every AI-agent session, together with the one task block being worked on, and this instruction: *Follow this document exactly. Implement only your assigned modules. Your output must conform to these interfaces. Do not independently change shared contracts. Do not run any git command that writes (commit, push, merge, rebase, reset, branch changes) unless the human developer explicitly tells you to.*

**Reading guide.** Items marked **TBD** are real decisions that need data or validation. Each TBD names an owner and a checkpoint and is listed in Section 21. Items marked **provisional** are the simplest defensible choice, usable as is, and will be checked as described. Nothing marked TBD or provisional may be silently replaced by a hardcoded number in code: it lives in a config file.

---

## 0. Rules of use

1. Modules import contract types only from `core.schemas`. No module imports another owner's internals.
2. Every period is the **apparent (synodic) period**, in **seconds**. Never call it the sidereal or physical rotation period in code, comments, UI text or documentation.
3. CNN, Lomb-Scargle and PDM results stay separate and labelled by source until reconciliation scores them.
4. `ReferenceLabel` (SDLCD reference periods) is **evaluation-only**. It must never reach model training, CNN inference, reconciliation, physics verification, the verdict engine or the controller. Only `evaluation/` and `ingestion/reference.py` may touch it.
5. Reason codes, Stage 2 sub-reasons and skip codes are constants imported from `core.schemas`. Nobody types them as free strings.
6. If verification is incomplete, the verdict is never Supported.
7. A physics fit shows compatibility with a **simplified model**. It never proves the real object's shape or rotation state. UI and documentation text must say so.
8. Git authority is separate from file ownership (Section 11). AI agents never commit, push, merge or rewrite history on their own.

---

## 1. Conventions

| Topic | Rule |
|---|---|
| Time | All `t_s` arrays are float64 **seconds**. Periods are `period_s`; frequencies are `freq_hz` (1/s). |
| Brightness direction | `y_norm`: **higher means brighter**, for magnitudes and flux alike. Magnitudes are sign-flipped before normalisation. Normalisation is median and MAD (`norm_params`). |
| Missing data | `valid_mask=False`. In `PreparedObservation`, `y_norm` and `flux_rel` are **NaN exactly where `valid_mask` is False** and finite everywhere else. Missing data is never zero-filled. |
| Support | `Candidate.support` is normalised to 0-1 **within its own estimator** (1 = that estimator's best peak). Not comparable across estimators until reconciliation scores it. |
| Ordering | Candidates are sorted by `support`, best first. Clusters are sorted by `score`, best first. |
| Harmonic ratio | In `harmonic_links`, `ratio` = other period divided by this period (2.0 means the other is double). |
| IDs | `cluster_id` is unique within one `ReconciliationResult` only. |
| Baseline and resolution | `baseline_s` = last valid time minus first valid time. `freq_resolution` = 1 / `baseline_s`. `prepare()` stores both in `PreparedObservation.meta` and everyone reuses them. If fewer than 2 valid samples: `baseline_s=0.0`, `freq_resolution=None`. |
| Randomness | Every stochastic component takes its seed from its config section, with a fixed default. Same input and same config give the same output. |
| Stubs | Stub objects carry `is_stub=True`. The UI shows a visible banner for any stub in a result. |
| Period bounds | **Not in this file.** They live in `configs/period_bounds.yaml` (Section 10). The controller injects them into each module's `cfg` as `period_min_s` and `period_max_s`. No other config file may define them. |
| Numbers that need a value | Thresholds are never hardcoded. They are config keys, listed in Section 21 with an owner. |

---

## 2. Constants (defined once in `core/schemas.py`)

```python
SCHEMA_VERSION = 1

# Top-level verdict reason codes (nine, as in the README and the submitted Round 2 deck)
REASON_CODES = (
    "low_quality",
    "no_periodic_signal",
    "conflicting_methods",
    "harmonic_ambiguity",
    "alias_ambiguity",            # the one canonical alias code
    "candidate_inadequate",
    "model_mismatch",
    "geometry_uncertainty",
    "verification_incomplete",
)

# Candidate-level Stage 2 failure sub-reasons. NEVER appear in Verdict.reason_codes.
STAGE2_FAIL_REASONS = (
    "insufficient_cycles",
    "unresolved_frequency",
    "poor_phase_coverage",
    "merged_into_neighbour",
)

# Why an estimator was not run (EstimatorResult.status == "skipped")
SKIP_CODES = (
    "stage1_insufficient",
    "period_bounds_not_set",
    "no_checkpoint",
    "too_few_valid_samples",
    "not_run",
)

# Why the verdict is Inconclusive for candidate_inadequate (evidence_summary["inconclusive_cause"])
INCONCLUSIVE_CAUSES = (
    "all_estimators_unavailable",
    "all_candidates_failed_stage2",
)

ESTIMATOR_METHODS = ("cnn", "lomb_scargle", "pdm")
```

**Top-level reason codes versus Stage 2 sub-reasons.** Reason codes describe the *whole analysis* and appear in `Verdict.reason_codes`. Stage 2 sub-reasons describe why one *candidate cluster* was rejected before physics verification and appear only in `Cluster.stage2["fail_reasons"]` and `evidence_summary["stage2_fail_reasons"]`. The two sets are disjoint (tested). If every candidate fails Stage 2, the verdict carries `candidate_inadequate` and the sub-reasons ride along in `evidence_summary`. `alias_suspected` is **not** a valid name anywhere (the README and decks use `alias_ambiguity`); the test suite rejects it.

---

## 3. Data structures

Validation rules for every field are in Section 19. Dataclasses expose `validate()`, `to_dict()` and `from_dict()` (Section 19).

```python
# core/schemas.py  (owner P2, FROZEN)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional
import numpy as np

BrightnessKind = Literal["mag", "flux", "unknown"]
GateStatus = Literal["suitable", "limited", "insufficient"]
StageStatus = Literal["ok", "failed", "skipped"]
VerdictLabel = Literal["Supported", "Ambiguous", "Inconclusive"]
EstimatorMethod = Literal["cnn", "lomb_scargle", "pdm"]
ProgressCallback = Callable[[float, str], None]


@dataclass
class Observation:
    """Raw, as ingested. No sorting, no cleaning. Produced by ingestion."""
    obs_id: str
    t_s: np.ndarray                    # 1-D float64, finite, not necessarily sorted
    y: np.ndarray                      # 1-D float64, NaN where missing, never +/-inf
    y_err: Optional[np.ndarray]        # None, or 1-D float64 same length, >= 0, NaN allowed
    brightness_kind: BrightnessKind
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceLabel:
    """EVALUATION ONLY. See Section 0, rule 4."""
    obs_id: str
    object_id: str
    period_s: float                    # finite, > 0
    split: Literal["val", "test"]      # object-level, seeded, deterministic


@dataclass
class PreparedObservation:
    obs_id: str
    t_s: np.ndarray                    # 1-D float64, finite, non-decreasing, starts at 0.0
    y_norm: np.ndarray                 # 1-D float64, NaN iff ~valid_mask
    valid_mask: np.ndarray             # 1-D bool
    dt_s: np.ndarray                   # 1-D float64, >= 0, dt_s[0] == 0.0
    flux_rel: Optional[np.ndarray]     # None, or 1-D float64, NaN iff ~valid_mask, > 0 where valid
    flags: dict[str, np.ndarray]       # keys: bright_outlier, duplicate, gap_edge (1-D bool)
    norm_params: dict[str, float]      # keys: median, mad, sign
    meta: dict[str, Any] = field(default_factory=dict)
    is_stub: bool = False


@dataclass
class QualityReport:                    # Stage 1 (observation level)
    status: GateStatus
    metrics: dict[str, float]
    warnings: list[str]
    reason_codes: list[str]            # insufficient -> ["low_quality"]; suitable/limited -> []
    is_stub: bool = False


@dataclass
class Candidate:
    period_s: float                    # finite, within period bounds
    support: float                     # 0-1 within its own estimator
    raw_score: float                   # LS power, 1 - theta, or CNN probability; finite
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EstimatorResult:
    method: EstimatorMethod
    status: StageStatus
    candidates: list[Candidate]        # best first; EMPTY unless status == "ok"
    error: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    is_stub: bool = False


@dataclass
class Cluster:
    cluster_id: int
    period_s: float
    members: dict[str, float]          # method -> member period used
    evidence: dict[str, float]         # method -> normalised support, 0-1; absent if unavailable
    alias_risk: float                  # A(P), 0-1
    harmonic_links: list[dict[str, Any]]   # [{"other_id": int, "ratio": float}]
    score: float                       # S(P), finite
    stage2: dict[str, Any]             # required keys in Section 7


@dataclass
class ReconciliationResult:
    clusters: list[Cluster]            # may be empty
    shortlist: list[int]               # cluster_ids; passed Stage 2; top K by score
    stage2_passed_any: bool            # == (len(shortlist) > 0)
    notes: list[str]
    is_stub: bool = False


@dataclass
class CandidateVerification:
    cluster_id: int
    period_s: float
    status: StageStatus
    fit_error: Optional[float]         # Section 18; None unless status == "ok"
    heldout_error: Optional[float]     # Section 18; None unless status == "ok"
    decoy_periods_s: list[float]       # [] if no decoy was run
    decoy_delta_e: list[float]         # aligned with decoy_periods_s; positive favours candidate
    harmonic_comparison: Optional[dict[str, Any]]   # required keys in Section 7
    passed: bool
    best_params: Optional[dict[str, Any]]           # required keys in Section 7
    notes: list[str]


@dataclass
class VerificationResult:
    per_candidate: list[CandidateVerification]
    complete: bool                     # strict definition in Section 18
    mode: Literal["fast", "thorough"]
    runtime_s: float
    is_stub: bool = False


@dataclass
class Verdict:
    label: VerdictLabel
    best_period_s: Optional[float]     # set ONLY when label == "Supported"
    best_cluster_id: Optional[int]     # set ONLY when label == "Supported"
    alternatives: list[dict[str, Any]] # each: {"cluster_id": int, "period_s": float, "why": str}
    reason_codes: list[str]            # subset of REASON_CODES; empty iff Supported
    evidence_summary: dict[str, Any]   # required keys in Section 7
    text: str
    is_stub: bool = False


@dataclass
class AnalysisResult:
    obs_id: str
    prepared: Optional[PreparedObservation]     # None only if ingestion-to-prepare failed
    quality: Optional[QualityReport]            # None only if prepared is None
    estimators: dict[str, EstimatorResult]      # ALWAYS exactly the three ESTIMATOR_METHODS keys
    reconciliation: Optional[ReconciliationResult]
    verification: Optional[VerificationResult]
    verdict: Verdict                            # ALWAYS present
    stage_errors: dict[str, str]                # stage name -> short message
    timings_s: dict[str, float]
    mode: str                                   # "fast" only in this build
    config_hash: str
    schema_version: int = SCHEMA_VERSION
```

---

## 4. Function signatures (exact names, frozen)

```python
# ---------- P1 ----------
# ingestion/__init__.py
def load_observation(path: str, fmt: Optional[str] = None) -> Observation: ...   # raises IngestionError
# ingestion/reference.py   (EVALUATION ONLY)
def load_reference_table(path: str) -> dict[str, ReferenceLabel]: ...
# preprocessing/prepare.py
def prepare(obs: Observation, cfg: dict) -> PreparedObservation: ...
# preprocessing/quality.py
def assess_stage1(prep: PreparedObservation, cfg: dict) -> QualityReport: ...
# classical_analysis/lomb_scargle.py
def run_lomb_scargle(prep: PreparedObservation, cfg: dict) -> EstimatorResult: ...
# classical_analysis/pdm.py
def run_pdm(prep: PreparedObservation, cfg: dict) -> EstimatorResult: ...
# classical_analysis/window.py
def compute_window(prep: PreparedObservation, cfg: dict) -> dict: ...            # keys in Section 7
# reconciliation/reconcile.py
def reconcile(prep: PreparedObservation, estimators: list[EstimatorResult],
              cfg: dict) -> ReconciliationResult: ...

# ---------- P2 ----------
# ml/features.py   (the ONLY implementation; used by training and inference)
def build_cnn_windows(prep: PreparedObservation, cfg: dict) -> np.ndarray: ...   # Section 16
# ml/infer.py
def run_cnn(prep: PreparedObservation, cfg: dict) -> EstimatorResult: ...
# physics/simulator.py
def simulate_lightcurve(params: "SimParams", t_s: np.ndarray) -> np.ndarray: ...
# physics/verify.py
def verify_candidates(prep: PreparedObservation, recon: ReconciliationResult, cfg: dict,
                      mode: str = "fast",
                      progress: Optional[ProgressCallback] = None) -> VerificationResult: ...
# verdict/decide.py
def decide(quality: QualityReport, estimators: dict[str, EstimatorResult],
           recon: Optional[ReconciliationResult], verification: Optional[VerificationResult],
           cfg: dict) -> Verdict: ...
# pipeline/controller.py
def analyze(obs: Observation, cfg: "PipelineConfig", mode: str = "fast",
            progress: Optional[ProgressCallback] = None) -> AnalysisResult: ...

# ---------- P1, app/plots.py: contract objects in, Plotly Figure out ----------
def fig_lightcurve(prep): ...
def fig_periodograms(estimators): ...          # dict[str, EstimatorResult]
def fig_phase_fold(prep, period_s): ...
def fig_candidates(recon): ...
def fig_quality(report): ...
def fig_verification(verification): ...
def fig_risk_coverage(eval_df): ...
def fig_calibration(eval_df): ...
```

`cfg: dict` is always that module's own YAML section, plus the injected keys `period_min_s` and `period_max_s`. Each module supplies defaults in code, so a missing optional key never crashes it. `SimParams` is internal to P2 and never appears in a shared contract. **Change from the previous draft:** `build_cnn_windows` now takes `cfg` so that window length, stride, clip and time-scaling constants come from `configs/cnn.yaml`, not from code.

---

## 5. Pipeline behaviour (controller rules)

Order: ingest, prepare, Stage 1, estimators (CNN, LS, PDM), reconcile (with Stage 2), verify, decide.

### 5.1 Estimator outcome semantics

Three different situations must never be confused:

| Situation | `status` | `candidates` | `error` |
|---|---|---|---|
| Ran, found one or more significant candidates | `ok` | one or more | `None` |
| Ran successfully, found **no significant candidate** (includes CNN abstention) | `ok` | `[]` | `None` |
| **Technical failure** (exception, numerical failure, invalid input) | `failed` | `[]` | short message |
| **Not run** (unavailable or skipped) | `skipped` | `[]` | one of `SKIP_CODES` |

"Significant" is defined by config thresholds (Section 21). Setting `status="ok"` with an empty list is the correct way to say "I looked and found nothing". `failed` and `skipped` mean "no evidence either way".

### 5.2 Stage 1 behaviour for each status

| Stage 1 status | Estimators | Reconciliation | Verification | Verdict ceiling |
|---|---|---|---|---|
| `suitable` | All three run | Normal | Normal | None |
| `limited` | **All three run** (README: proceed with warnings) | Normal; quality does not change weights | Normal | None by default. A config key `verdict.limited_max_label` (`"Supported"` or `"Ambiguous"`, default `"Supported"`) can cap it. **TBD** final value: P2, CP4, based on false-Supported rate by quality group on validation data. |
| `insufficient` | **None run**: all three get `status="skipped"`, `error="stage1_insufficient"`. | `None` | `None` | Inconclusive (`low_quality`) |

Propagation of quality information: `QualityReport.warnings` always copied to `evidence_summary["stage1_warnings"]` and `QualityReport.status` to `evidence_summary["stage1_status"]`. If the final label is not Supported and the status is `limited`, `low_quality` is added to `reason_codes`. If the label is Supported and the status is `limited`, `reason_codes` stays empty, but the warnings remain visible in `evidence_summary` and in `Verdict.text`. There is no fourth status.

### 5.3 All estimators fail, skip, or find nothing

| Case | Reconciliation | Verification | Verdict | `evidence_summary` / `stage_errors` / `notes` |
|---|---|---|---|---|
| At least one estimator is `ok`, and **no estimator has any candidate** | `clusters=[]`, `shortlist=[]`, `stage2_passed_any=False`, `notes=["no_candidates"]` | `None` | Inconclusive, `["no_periodic_signal"]` | `n_estimators_available` counts the `ok` ones; failed or skipped estimators are listed in `estimator_status` and `stage_errors` |
| **No estimator is `ok`** (all failed or skipped; Stage 1 not insufficient) | same empty result, `notes=["no_usable_estimators"]` | `None` | Inconclusive, `["candidate_inadequate"]` | `inconclusive_cause="all_estimators_unavailable"`; each failure in `stage_errors["cnn"]` etc. |
| Estimators produced candidates but **every cluster fails Stage 2** | clusters present, `shortlist=[]`, `stage2_passed_any=False` | `None` | Inconclusive, `["candidate_inadequate"]` | `inconclusive_cause="all_candidates_failed_stage2"`; `stage2_fail_reasons` lists the union of sub-reasons |

An empty candidate list is always valid input for `reconcile()`, and `reconcile()` never raises. The pipeline always returns a valid `AnalysisResult`.

### 5.4 Other rules

| Condition | What the controller does |
|---|---|
| Some (not all) estimators fail or are skipped | Keep each `EstimatorResult`. Reconciliation renormalises weights over estimators with `status="ok"` and records it in `notes`. |
| Verification is incomplete | Verdict is Ambiguous with `verification_incomplete` (never Supported). Definition of "incomplete" is in Section 18. |
| Unexpected exception in any stage | Controller catches it, writes `stage_errors[stage]`, continues if possible. The app never crashes. |
| A module returns an object that fails `validate()` | Treated as that module failing: `stage_errors[stage] = "contract_violation: <message>"`, then the same rules as above. Controlled by `pipeline.validate_outputs` (default `true`). |
| Period bounds are not set (`status: TBD`) | Estimators skipped with `period_bounds_not_set`; verdict follows the "no usable estimators" row. |
| `mode="thorough"` | Returns a clear not-implemented message. Fast mode is the only mode in this build. |

### 5.5 Shortlist and harmonic partners

The shortlist is the top K clusters (K from `reconcile.yaml`, default 5; 3 in Fast mode) that passed Stage 2, ordered by score. The **required verification set** is the union of (a) every `cluster_id` in `recon.shortlist` and (b) every distinct cluster directly linked from a shortlisted cluster by `harmonic_links` with ratio `2` or `1/2` that also passed Stage 2. Deduplicate IDs; partners do not count toward K. Only direct links to a shortlisted cluster are included, not transitive harmonic links. Every ID in this set must have exactly one `CandidateVerification` record, including when its protocol fails or is skipped. A partner that failed Stage 2 is not in the required set; verification records `harmonic_comparison = {"partner_id": ..., "prefers": "undecided", "reason": "partner_failed_stage2", ...}` for the eligible counterpart as applicable.

---

## 6. Reason-code mapping (what each code means operationally)

Section 5.3 gives the Inconclusive paths. Each code has exactly one definition:

| Code | Trigger (all thresholds are config keys, Section 21) |
|---|---|
| `low_quality` | Stage 1 is `insufficient`; or the label is not Supported and Stage 1 is `limited`. |
| `no_periodic_signal` | At least one estimator is `ok` and no estimator produced any candidate (Section 5.3). |
| `candidate_inadequate` | No usable estimator, or every cluster failed Stage 2 (Section 5.3), or fewer than `min_agreeing_estimators` estimators contributed candidates to the leading cluster. |
| `conflicting_methods` | At least two estimators produced candidates, and the top candidates of those estimators fall in different, non-harmonic clusters. |
| `harmonic_ambiguity` | The leading candidate has a harmonic partner and `harmonic_comparison["prefers"] == "undecided"`. |
| `alias_ambiguity` | The leading candidate's `alias_risk` exceeds `verdict.alias_risk_max` (**TBD** P2, CP4). |
| `model_mismatch` | Verification is complete, and no candidate in the required verification set has `status="ok"` and `passed=True`. Every required candidate therefore failed at least one acceptance criterion: `heldout_error > verification.heldout_error_max` and/or `decoy_separation < verification.decoy_separation_min` (**both thresholds TBD** P2, CP4). A decoy-separation failure maps to this existing code as well; no new reason code is introduced. |
| `geometry_uncertainty` | For the leading candidate, at least two geometry families were fitted and their held-out errors differ by less than `verification.geometry_margin` (**TBD** P2, CP4). |
| `verification_incomplete` | `VerificationResult.complete` is `False`, or `verification` is `None` while Stage 2 produced a shortlist. |

**Verdict table.** Checks run top to bottom; the first matching Inconclusive row ends the decision. Several Ambiguous codes may apply at once; return all that apply. `best_period_s` and `best_cluster_id` are set only for Supported. Every Ambiguous or Inconclusive verdict must include at least one applicable code from the nine-code `REASON_CODES` tuple; an empty reason list is valid only for Supported. In particular, when verification is complete but no required candidate passes, `model_mismatch` is mandatory whether the failed gate is held-out error, decoy separation, or both. Do not add `model_mismatch` merely because one candidate failed if another required candidate passed both checks.

| Situation | Verdict | Reason code(s) |
|---|---|---|
| Stage 1 `insufficient` | Inconclusive | `low_quality` |
| Estimators ok, no candidates anywhere | Inconclusive | `no_periodic_signal` |
| No usable estimator | Inconclusive | `candidate_inadequate` |
| Every cluster failed Stage 2 | Inconclusive | `candidate_inadequate` |
| Leading cluster supported by fewer than `min_agreeing_estimators` estimators | Ambiguous | `candidate_inadequate`, `evidence_summary["estimators_agree"]=False` |
| Top candidates of two or more estimators disagree (non-harmonic) | Ambiguous | `conflicting_methods` |
| P versus 2P undecided | Ambiguous | `harmonic_ambiguity` |
| Alias risk above threshold | Ambiguous | `alias_ambiguity` |
| Verification is complete, but no required candidate meets both the held-out-error and decoy-separation criteria | Ambiguous | `model_mismatch` |
| Geometry families fit equally well | Ambiguous | `geometry_uncertainty` |
| Verification incomplete | Ambiguous | `verification_incomplete` |
| All requirements met | Supported | none |

**Supported requires all of:** the leading candidate (highest reconciliation score among candidates with `status="ok"`, `passed=True`) belongs to a cluster that passed Stage 2; `VerificationResult.complete` is True; at least `min_agreeing_estimators` (**TBD** P2, CP4; default 2) estimators have their top candidate in that cluster; `alias_risk` is below threshold; any harmonic partner is resolved in its favour; and none of the Ambiguous conditions above applies. If Stage 1 is `limited` and `verdict.limited_max_label` is `"Ambiguous"`, the label is capped and `low_quality` is added.

---

## 7. Generic dictionaries: required and optional keys

**Required** keys are part of the contract. **Optional** keys may be added or omitted freely and never force a `SCHEMA_VERSION` change unless another module starts to depend on them (then use Section 12).

### 7.1 `PreparedObservation.meta`
- Required: `baseline_s` (float >= 0), `freq_resolution` (float > 0 or `None`), `n_rows` (int), `n_valid` (int).
- Optional: `source_file`, `assumptions` (list[str]), `t0_iso`.

### 7.2 `Candidate.meta`
- Required: none for CNN. For `lomb_scargle` and `pdm` candidates: `freq_width_hz` (float > 0; full width at half maximum of the peak, or `freq_resolution` if it cannot be measured).
- Optional: `freq_hz`, `rank`.

### 7.3 `EstimatorResult.meta`
- Required for every result: `runtime_s` (float).
- Required when `status="ok"`:
  - `lomb_scargle`, `pdm`: `periodogram` = `{"freq_hz": list[float], "power": list[float]}` (downsampled, equal lengths), `freq_resolution`.
  - `cnn`: `p_no_reliable` (float 0-1), `abstained` (bool), `n_windows` (int), `window_disagreement` (float 0-1), `experimental` (bool).
- Optional: `checkpoint`, `period_bounds` (CNN: the bounds the checkpoint was trained with), `windows_subsampled`.

### 7.4 `compute_window(prep, cfg)` return value
```python
{
  "freq_hz": list[float],            # frequency grid, Hz, ascending, starts at 0.0
  "window_power": list[float],       # spectral window power, normalised so the value at 0 Hz is 1.0
  "peaks": [                         # strongest peaks at non-zero frequency, height descending,
      {"freq_hz": float,             # at most cfg["n_peaks"]; freq_hz is the ALIAS OFFSET
       "height": float}              # relative to the 0 Hz peak, in [0, 1]
  ],
  "freq_resolution": float | None,   # copied from prep.meta
}
```
A peak at offset `f_a` means that a true signal at `f` can also show a spurious peak at `f ± f_a`.

### 7.5 `Cluster.stage2`
- Required: `passed` (bool), `cycles` (float), `fail_reasons` (list[str] from `STAGE2_FAIL_REASONS`; empty iff `passed`).
- Optional: `rel_period_uncertainty`, `phase_coverage`, `merged_with`.

### 7.6 `CandidateVerification.harmonic_comparison` (when a partner exists, else `None`)
- Required: `partner_id` (int), `prefers` (`"P"`, `"2P"` or `"undecided"`), `delta_e` (float or `None`), `reason` (str or `None`).
- Here **P is the shorter period and 2P the longer period of the pair**, identically in both records. `delta_e = heldout_error(2P) - heldout_error(P)`; **positive favours P**. `prefers="undecided"` when `abs(delta_e) < verification.harmonic_margin` (**TBD** P2, CP4) or either member did not finish.

### 7.7 `CandidateVerification.best_params` (when `status="ok"`, else `None`)
- Required: `shape` (str, the best geometry family), `geometry_errors` (dict: family name to its `heldout_error`, one entry per family fitted), `fit_space` (`"flux_rel"` or `"y_norm"`), `scale_a` (float), `offset_b` (float).
- Optional: all other fitted parameters (spin-axis angles, aspect ratios, phase), used only for the optional 3D view.

### 7.8 `Verdict.evidence_summary`
- Required: `stage1_status` (str), `stage1_warnings` (list[str]), `estimator_status` (dict: method to `status`), `n_estimators_available` (int, count of `ok`), `estimators_agree` (bool or `None`), `alias_risk` (float or `None`), `verification_complete` (bool or `None` if verification not run), `heldout_error` (float or `None`), `decoy_separation` (float or `None`), `stage2_fail_reasons` (list[str]), `selective_accuracy_validation` (dict `{"value": float, "n": int}` or `None`).
- Optional: `inconclusive_cause` (one of `INCONCLUSIVE_CAUSES`; present for Inconclusive/`candidate_inadequate`), `limited_cap_applied` (bool).
- `heldout_error` and `decoy_separation` must come from the same selected candidate record, using the deterministic selection rule in Section 18.2; they are not independently selected minima or maxima across candidates.
- It is **not** the raw CNN probability.

---

## 8. Error handling

| Situation | What happens |
|---|---|
| Unreadable or malformed file | `ingestion` raises `IngestionError(message)`. The app shows the message, never a traceback. |
| Estimator ran successfully but found no significant peak | Returns `status="ok"`, `candidates=[]`, `error=None` (Section 5.1). A no-peak result is not a failure. |
| Not run because there are too few valid samples or another defined skip condition applies | Returns `status="skipped"`, `candidates=[]`, `error` set to the applicable `SKIP_CODES` value. |
| Technical failure during an attempted run (for example an exception, numerical failure or invalid input) | Returns `status="failed"`, `candidates=[]`, and a short technical error message. It does **not** raise. |
| Unexpected exception in another stage | Caught by the controller (Section 5.4). |
| Missing CNN checkpoint | `run_cnn` returns `status="skipped"`, `error="no_checkpoint"`. |
| Checkpoint trained with different period bounds than `period_bounds.yaml` | `run_cnn` returns `status="failed"`, `error="checkpoint_bounds_mismatch"`. |

`core/errors.py` defines `IngestionError` only. Contract violations use `ValueError` with the prefix `contract:`.

---

## 9. Sampling-profile JSON (P1 produces, P2 consumes)

`data/profiles/sdlcd_profiles.json`: a list of records. P2's fallback `data/profiles/default_profiles.json` has the same schema.

```json
{ "profile_id": "string", "n_points": 0, "duration_s": 0.0, "median_dt_s": 0.0,
  "gap_fractions": [0.0], "noise_rel_amplitude": 0.0, "missing_fraction": 0.0 }
```

`gap_fractions` are gap lengths divided by duration. Records never contain brightness values or reference periods.

---

## 10. Configuration rules

- `core/config.py` defines `PipelineConfig`: one plain `dict` per section, loaded from YAML. Adding a key never needs a contract change.
- `configs/default.yaml` (P2, frozen) lists the files to load.
- **Shared period bounds.** `configs/period_bounds.yaml` is the **only** place the bounds exist:
  ```yaml
  status: TBD            # becomes SET once P1's audit values are entered
  min_period_s: null     # TBD: determined by P1-01 (SDLCD data audit)
  max_period_s: null
  ```
  - **P1 determines** the values in the audit and sends them to P2 by message (end of P1-01, before CP1).
  - **P2 commits** the file (it is a shared architectural file), sets `status: SET`, and ensures every consumer reads it.
  - `core.config.get_period_bounds()` loads the file and returns `PeriodBounds(min_period_s, max_period_s)`; it raises `ValueError("contract: period bounds not set")` if `status` is `TBD`, and the controller converts that into the skip rule in Section 5.4.
  - The controller injects `period_min_s` and `period_max_s` into the `cfg` of every module that needs them (P1's classical modules and reconciliation, P2's CNN, simulator and verification). **`classical.yaml`, `reconcile.yaml`, `cnn.yaml`, `simulation.yaml` and `verification.yaml` must not define these keys** (tested), so no duplicate editable value exists. Unit tests of individual modules pass the two keys in their `cfg` explicitly.
  - Because the CNN's 128 log-spaced bins depend on the bounds, the bounds are stored with every synthetic dataset and checkpoint, and `run_cnn` refuses a checkpoint whose bounds differ (Section 8). A Day 1 pilot dataset generated before the bounds are SET is for pilot use only and is regenerated afterwards.
- **P1 owns:** `preprocess.yaml`, `classical.yaml`, `reconcile.yaml`.
- **P2 owns:** `default.yaml`, `period_bounds.yaml` (values supplied by P1), `simulation.yaml`, `cnn.yaml`, `verification.yaml`, `verdict.yaml`, `frozen_thresholds.yaml` (created at CP4, never edited after the `thresholds-frozen` tag).
- `config_hash` = sha256 of the canonical (sorted-key) JSON of all sections plus `SCHEMA_VERSION`.
- Every run writes `run_config.json` (config, seeds, package versions, commit) so a saved configuration reproduces an analysis.
- Thresholds are calibrated on validation data and stored in config. **No threshold is hardcoded and none is invented in this document.**

---

## 11. Ownership, Git authority and frozen files

### 11.1 File ownership (who may edit which files)

| Owner | Folders and files |
|---|---|
| **P1** | `ingestion/`, `preprocessing/`, `classical_analysis/`, `reconciliation/`, `evaluation/`, `app/plots.py`, `configs/preprocess.yaml`, `classical.yaml`, `reconcile.yaml`, `data/sdlcd_raw/`, `data/samples/`, `data/reference/`, `data/profiles/sdlcd_profiles.json`, `docs/` (except P2 results), `README.md` (after `v0-skeleton`), `tests/p1/` |
| **P2** | `core/`, `physics/`, `ml/`, `verdict/`, `pipeline/`, `app/main.py`, `app/pages/`, `configs/default.yaml`, `period_bounds.yaml`, `simulation.yaml`, `cnn.yaml`, `verification.yaml`, `verdict.yaml`, `frozen_thresholds.yaml`, `models/`, `scripts/`, `outputs/demo_cache/`, `data/synthetic/`, `data/synthetic_holdout/`, `data/profiles/default_profiles.json`, `requirements.txt`, `.gitignore`, `tests/p2/`, root-level `tests/test_*.py`, `tests/fixtures/`, `CONTRACTS.md` |

**Frozen shared files** (changed only through Section 12): `core/schemas.py`, `core/errors.py`, `core/config.py`, `CONTRACTS.md`, `configs/default.yaml`, `requirements.txt`, `.gitignore`, `tests/fixtures/*`, and all root-level tests listed in Section 13.

`docs/results/`: P2 owns `verification_*`, `cnn_training_log.json`, `cnn_calibration.json`, `runtime_*.json`. P1 owns everything produced by `evaluation/run_eval.py`. The two never write the same filename. No file has two owners.

### 11.2 Git authority (who may do what in Git)

Ownership says who edits a file. Git authority says who may change history. They are separate.

| Action | P1 (human) | P2 (human, project lead) | AI coding agents |
|---|---|---|---|
| Edit own files on own branch | Yes | Yes | Yes, only files listed in the task |
| Commit and push to own branch (`p1/dev` or `p2/dev`) | Yes, personally | Yes, personally | **No**, unless that branch's human owner explicitly says so for that action |
| Open a pull request to `main` | Yes | Yes | No |
| Merge anything into `main`, push to `main`, tag | **No** | **Yes, only P2** | **No** |
| Rewrite history (rebase, amend pushed commits, `reset --hard`, force push, delete branches) | No | Only on own branch, deliberately | **Never** |
| Touch another developer's branch or files | No | No, except contract changes via Section 12 | **Never** |

Rules:

1. Only P2 approves and performs merges into `main`.
2. Both developers integrate the latest approved `main` into their own branch (`git fetch origin`, `git merge origin/main`, `pip install -r requirements.txt`, `pytest -q`) **before starting any task that depends on the other person's work**, and at the start of every day.
3. AI agents may read the repository, write files in the assigned task's file list, and run tests. They do not run `git commit`, `git push`, `git merge`, `git rebase`, `git reset`, `git checkout` to another branch, or any history-changing command unless the human owner explicitly instructs it for that action.
4. An agent that wants a file outside its task list changed stops and reports it to its human developer.

---

## 12. Change protocol

1. The person who needs a change messages the other with the exact change and the reason. Agents never start this; humans do.
2. Both agree. P2 decides whether it is a contract change (any dataclass field, function name or signature, reason code, constant, required `meta` key, config-key ownership) or not (a new optional `meta` key, which needs no protocol).
3. P2 edits `CONTRACTS.md` and the code in `core/` on a branch, bumps `SCHEMA_VERSION`, updates the affected tests, and after review merges it into `main` with a commit named `contract-change-N`.
4. P2 **tells P1 explicitly** what changed and which of P1's files are affected.
5. Both developers merge the new `main` into their branch immediately and re-run `pytest -q` before continuing any dependent work.
6. A TBD in Section 21 resolved by entering a value in a config file is **not** a contract change. Resolving a TBD that changes a field, a code or a signature is.

---

## 13. Automated guards (all must pass before P2 merges anything)

Root-level tests are owned by P2 and frozen. They run against whatever is merged: stubs early, real modules as they land. Tests exercise behaviour, not just existence.

| # | Test file | What it checks |
|---|---|---|
| 1 | `test_validation.py` | Each dataclass accepts a valid example and `validate()` rejects each listed invalid case from Section 19 (wrong dtype, length mismatch, NaN where forbidden, status combinations, `Supported` with reason codes, and so on). |
| 2 | `test_serialization.py` | Round trip `from_dict(to_dict(x))` preserves values, shapes, dtypes, NaN positions and `None`; invalid `Literal` values are rejected; `inf` is rejected; comparison uses `assert_contract_equal` (Section 19), never `==` on arrays. |
| 3 | `test_contracts.py` | Every function in Section 4 exists with exactly the listed signature (`inspect.signature`), including the `progress` parameter. |
| 4 | `test_period_bounds.py` | `configs/period_bounds.yaml` exists; if `status: SET` then `0 < min < max`; no other config file defines `min_period_s`, `max_period_s`, `p_min_s` or `p_max_s`; the controller injects identical bounds into the classical, reconciliation, CNN, simulator and verification configs; a CNN checkpoint with different stored bounds is refused. Skips with a warning while `status: TBD`; **must not skip from CP1 onward**. |
| 5 | `test_cnn_features.py` | Golden test on a fixture `PreparedObservation`: output shape `(n_windows, 3, 1024)`, dtype `float32`, finite; channel 0 clipped values, channel 1 equals the validity mask (0 on padding), channel 2 in [0, 1]; missing samples give channel 0 = 0 and channel 1 = 0; short input gives 1 right-padded window; long input gives overlapping windows with the final window aligned to the end; `max_windows` respected; no valid windows gives shape `(0, 3, 1024)`. |
| 6 | `test_cnn_features.py` (second test) | Training and inference share one code path: the dataset generator calls `ml.features.build_cnn_windows` (checked with a call counter) and there is no second feature implementation (a source grep for `def build_cnn_windows`). |
| 7 | `test_reason_codes.py` | Every reason, fail-reason, skip-code or inconclusive-cause string used in `verdict/`, `reconciliation/`, `preprocessing/`, `classical_analysis/` and `pipeline/` is a member of its constant tuple; `REASON_CODES` and `STAGE2_FAIL_REASONS` are disjoint; the README's reason-code list equals `REASON_CODES`; the string `alias_suspected` appears nowhere in code, docs or README. |
| 8 | `test_pipeline_behaviour.py::stage1` | Insufficient Stage 1: all three estimators `skipped` with `stage1_insufficient`, `reconciliation` and `verification` are `None`, verdict Inconclusive with `low_quality`. Limited Stage 1: all three estimators run; if the label is not Supported then `low_quality` is present; warnings reach `evidence_summary`. |
| 9 | `test_pipeline_behaviour.py::all_estimators_fail` | With all three estimators forced to `failed` or `skipped`: returns a valid `AnalysisResult`, empty reconciliation, verification `None`, Inconclusive with `candidate_inadequate` and `inconclusive_cause="all_estimators_unavailable"`. With all three `ok` and empty: Inconclusive with `no_periodic_signal`. |
| 10 | `test_pipeline_behaviour.py::stage2_all_fail` | Every cluster failing Stage 2: `shortlist=[]`, `stage2_passed_any=False`, verification not run, Inconclusive with `candidate_inadequate`, sub-reasons in `evidence_summary["stage2_fail_reasons"]`. |
| 11 | `test_verification_semantics.py` | Only the four allowed `(status, passed)` combinations validate; `passed=True` with `status != "ok"` is rejected; `complete` is False if any required candidate is not `status="ok"`; **a verdict is never Supported when `complete` is False** (property test over random verification results); metric sign test with a hand-computed example: if the decoy error exceeds the candidate error then `decoy_delta_e` is positive. |
| 12 | `test_no_reference_leak.py` | No module in `ml/`, `physics/`, `verdict/`, `reconciliation/`, `classical_analysis/`, `preprocessing/` or `pipeline/` imports `ingestion.reference` or `ReferenceLabel`; `analyze()` has no parameter that accepts a reference. |
| 13 | `test_stub_pipeline.py` | Stub `analyze()` returns a complete, valid `AnalysisResult` (validates and round-trips). |
| 14 | `test_progress_callback.py` | Passing a callback to `verify_candidates` and `analyze` gives calls with `fraction` in [0, 1], non-decreasing, ending at 1.0 on completion, `message` a non-empty `str`; passing `None` works; an exception raised inside the callback does not break the pipeline. |
| 15 | `test_stage2_rules.py` | Against the real `reconcile()`: `cycles = baseline_s / period_s`; a candidate with fewer than `min_cycles` fails with `insufficient_cycles`; a candidate within `min_separation_resolutions` resolutions of a higher-scoring non-harmonic cluster fails with `merged_into_neighbour`; harmonic partners are exempt from that rule; `passed` is True iff `fail_reasons` is empty. |

Module-specific unit tests (LS recovers a known period, PDM matches LS, simulator periodicity, CNN receptive field of 379 and about 0.25 M parameters, and so on) stay in `tests/p1/` and `tests/p2/`.

---

## 14. Fixtures provided in Phase 0

- `tests/fixtures/synthetic_sine.csv`: clean sine with gaps.
- `tests/fixtures/mock_prepared.npz`: a valid `PreparedObservation`, including some invalid (NaN-masked) samples.
- `tests/fixtures/mock_cnn_result.json`: an `EstimatorResult` (method `cnn`) with three candidates, one being the 2P partner of another.
- `tests/fixtures/mock_reconciliation.json`: a `ReconciliationResult` with harmonically linked clusters.
- `tests/fixtures/golden_cnn_windows.npz`: expected tensor for the golden feature test.
- `data/profiles/default_profiles.json`: hand-written fallback sampling profiles.

---

## 15. Who delivers what, by checkpoint

Merging into `main` at each checkpoint is done by P2 only (Section 11).

| Checkpoint | P1 delivers | P2 delivers |
|---|---|---|
| CP0 (end of Phase 0) | Setup confirmed | Repo, this file, schemas, stubs, fixtures, tests, `period_bounds.yaml` (TBD), tag `v0-skeleton` |
| CP1 (end of Day 1) | `load_observation`, `prepare`, `assess_stage1`, reference table, sampling profiles, **period-bound values sent to P2** | Simulator, labels, pilot dataset, model skeleton, **`period_bounds.yaml` set**, feature builder skeleton |
| CP2 (end of Day 2) | `run_lomb_scargle`, `run_pdm`, `compute_window` (with "no significant candidate" thresholds set) | Degrader, `build_cnn_windows`, first CNN run launched, fitting machinery |
| CP3a (when P1-10 passes) | `reconcile` with Stage 2 and its thresholds | Whatever is finished |
| CP3b (end of Day 3) | `app/plots.py`, metrics and stress tests started | Calibrated CNN, verification, verdict, controller, holdout shard and checkpoint committed |
| CP4 (end of Day 4) | Evaluation outputs, tuned reconciliation weights | App, frozen thresholds, demo cache, verification AUC |
| CP5 (Day 5) | Final results, README, technical document | Fixes, runtime numbers, fresh-clone test, tag `v1.0-submission` |

---

## 16. CNN input tensor specification

**Function:** `build_cnn_windows(prep, cfg) -> np.ndarray` in `ml/features.py`. This is the **only** implementation. Training data generation, validation, calibration and inference all call it.

**Input:** a `PreparedObservation` only. Never an `Observation`, never a `ReferenceLabel`.

**Output:** `float32`, C-contiguous, shape `(n_windows, 3, W)` with `W = cfg["window_length"]` (**fixed at 1024**), all values finite. `n_windows` may be 0.

**Rows and irregular sampling.** The rows are the rows of `PreparedObservation`, in time order, **including rows where `valid_mask` is False**. No resampling, no interpolation and no gap filling. Irregular timing is carried by channel 2. The time axis is never resized, so absolute time scale is preserved.

| Channel | Meaning | Value for a real, valid sample | Value for a missing sample | Value for padding |
|---|---|---|---|---|
| 0 | Brightness | `clip(y_norm, -y_clip, +y_clip) / y_clip`, range [-1, 1] | `0.0` | `0.0` |
| 1 | Validity mask | `1.0` | `0.0` | `0.0` |
| 2 | Log time step | `(clip(log10(max(dt_s, dt_floor_s)), L_min, L_max) - L_min) / (L_max - L_min)`, range [0, 1] | same formula (the row has a timestamp) | `0.0` |

Channel order is fixed: **0 brightness, 1 validity mask, 2 log time step.** Bright outliers are clipped, never removed. The first row has `dt_s = 0`, which maps to `0.0`. Constants `y_clip`, `dt_floor_s`, `L_min`, `L_max` live in `configs/cnn.yaml`. **TBD** values: P2, by CP2, using P1's cadence statistics from the data audit.

**Windows**

| Case | Behaviour |
|---|---|
| `n_rows <= W` | One window, right-padded with zeros. |
| `n_rows > W` | Window starts at `0, S, 2S, ...` with stride `S = cfg["window_stride"]` (**provisional: 512**, i.e. 50 percent overlap). If the last start does not reach the end, one extra window starts at `n_rows - W`. |
| More than `cfg["max_windows"]` windows | Keep `max_windows` evenly spaced starts, including the first and last. Record `windows_subsampled=True` in the `EstimatorResult.meta`. (**TBD** value: P2, CP2.) |
| A window with fewer than `cfg["min_valid_per_window"]` valid samples | Dropped. (**TBD** value: P2, CP2.) |
| No window survives | Returns shape `(0, 3, W)`; `run_cnn` returns `status="skipped"`, `error="too_few_valid_samples"`. |

**Same preprocessing in training and inference.** The synthetic generator produces an `Observation`, passes it through P1's `prepare()` and then `build_cnn_windows()`. There is no separate training-time feature path. Test 6 enforces this. Cached training shards store the config constants used and refuse to load if they differ.

**Provisional design decisions and how they are validated.** The choice of these three channels follows the README. The log time-step scaling, the clip value, the 50 percent overlap and the dropping rule are the simplest defensible choices. They are validated by synthetic validation metrics and, if time allows, the README's timing-channel ablation. If a value changes, only `configs/cnn.yaml` changes, and the golden fixture is regenerated by P2 in the same commit.

**Model output and abstention (`run_cnn`)**
- The model outputs 129 classes: 128 log-spaced period bins between the shared period bounds, plus one *No Reliable Period* (NRP) class.
- Window probabilities are calibrated, then combined by a **weighted average** (weights = number of valid samples in the window). Windows are not multiplied.
- `meta["p_no_reliable"]` is the combined NRP probability.
- **Abstention:** if `p_no_reliable >= cnn.abstain_threshold` (**TBD**: P2, CP3b, from validation calibration) then `status="ok"`, `candidates=[]`, `meta["abstained"]=True`. This is "looked and found nothing", not a failure.
- Otherwise candidates are the local maxima of the 128-bin distribution (up to `cnn.n_candidates`, default 5). `period_s` is the bin's geometric centre, `raw_score` is the combined probability of that bin, `support = raw_score / max(raw_score over returned candidates)`.
- `meta["window_disagreement"]` is the mean total-variation distance between each window's distribution (NRP removed, renormalised) and the combined one. It is reported only and does not change the status.

---

## 17. Stage 2 definitions (candidate level, owned by P1 in `reconciliation/stage2.py`)

Stage 2 is evaluated for **every cluster** after clustering and scoring. A cluster **passes** iff `fail_reasons` is empty.

| Sub-reason | Definition | Config key (`reconcile.yaml`) |
|---|---|---|
| `insufficient_cycles` | `cycles = baseline_s / period_s` is below `min_cycles`. This is a hard requirement that the repeating pattern is seen enough times. | `min_cycles` (**TBD** P1, CP3a, from injection-and-recovery on synthetic data, as the README requires) |
| `unresolved_frequency` | The relative period uncertainty is too large: `rel_period_uncertainty = freq_width_hz * period_s` (using the largest `freq_width_hz` among the cluster's members, or `freq_resolution` if absent) exceeds `max_rel_period_uncertainty`. This uses the measured peak width, so it is stricter than, and different from, the cycle count. | `max_rel_period_uncertainty` (**TBD** P1, CP3a) |
| `poor_phase_coverage` | Valid samples are folded at the cluster period into `n_phase_bins` bins; `phase_coverage` = fraction of bins with at least `min_points_per_bin` valid samples. Fails if below `min_phase_coverage`. | `n_phase_bins`, `min_points_per_bin`, `min_phase_coverage` (**TBD** P1, CP3a) |
| `merged_into_neighbour` | Another cluster with a **higher score** that is **not** harmonic-linked lies within `min_separation_resolutions * freq_resolution` in frequency. Harmonic partners are exempt: P and 2P are hypotheses, not duplicates. | `min_separation_resolutions` (**TBD** P1, CP3a) |

Related clustering settings in `reconcile.yaml` (all **TBD** P1, CP3a): `cluster_tol_resolutions` (merge tolerance in units of `freq_resolution`), `top_k` (default 5). Reconciliation weights `w_C`, `w_L`, `w_D` and `lambda_A` are tuned by P1 on the validation split at CP4 and then frozen.

If every cluster fails: Section 5.3 (row 3). Harmonic partners that pass Stage 2 enter verification as in Section 5.5.

---

## 18. Verification semantics and metrics

Everything here is a comparison against a **simplified brightness model**. A good fit means the data are compatible with the model, not that the object has that shape or spin state.

### 18.1 Status, `passed` and `complete`

Allowed `(status, passed)` combinations, enforced by `validate()`:

| `status` | `passed` | Meaning |
|---|---|---|
| `ok` | `True` | Protocol finished; met `heldout_error <= heldout_error_max` **and** `decoy_separation >= decoy_separation_min` |
| `ok` | `False` | Protocol finished; `heldout_error > heldout_error_max`, and/or `decoy_separation < decoy_separation_min` (one or both acceptance criteria failed) |
| `failed` | `False` | Optimiser or protocol failed, **timed out** or was interrupted (reason in `notes`, e.g. `timeout`, `no_valid_decoy`) |
| `skipped` | `False` | Not run |

All other combinations are invalid. For `failed` and `skipped`, `fit_error` and `heldout_error` are `None`. For `status="ok", passed=False`, the failed gate(s) are determined from the recorded metrics: an excessive `heldout_error`, an insufficient `decoy_separation`, or both map to the existing top-level reason code `model_mismatch` under Section 6. The `passed` flag is True only when both comparisons pass; equality at either configured boundary passes because the comparisons are `<=` and `>=`, respectively.

`verify_candidates(prep, recon, cfg, ...)` derives the required verification set from `recon` exactly as specified in Section 5.5. It must return exactly one `CandidateVerification` for every required `cluster_id`, and no record for an ID outside that set. A candidate whose protocol cannot finish remains represented by a record with `status="failed"` or `"skipped"`; it is not omitted.

**`VerificationResult.complete` is True iff** (a) `per_candidate` contains exactly one record for every ID in the required set and no other IDs, and (b) every such record has `status="ok"`. A candidate with `status="ok"` and `passed=False` still counts as complete because its protocol finished; `passed` is a separate scientific acceptance result. A timeout or interruption makes that candidate `failed`, so `complete=False`. Partial results are retained in `per_candidate`, but a partially complete verification **can never produce Supported**, even if the leading candidate itself finished and passed. The required set is non-empty whenever verification is run; when the shortlist is empty, the controller does not call verification and `AnalysisResult.verification` is `None`.

**Validation and data flow (no public signature changes).** `VerificationResult.validate()` checks the result's intrinsic structure, including unique `cluster_id` values, but cannot check membership against `recon` because `VerificationResult` does not contain the reconciliation object. After `verify_candidates()` returns, `pipeline.controller.analyze()` independently derives the required ID set from the same `recon`, checks exact one-to-one membership (no missing, duplicate or extra IDs), and recomputes the `complete` predicate above before calling `decide()`. If membership or the `complete` value is inconsistent, the controller treats the verification output as a contract violation under Section 5.4, records `stage_errors["verification"]`, and passes `verification=None` to `decide()`; with a non-empty shortlist this yields `verification_incomplete`, never Supported. `AnalysisResult.validate()` repeats the cross-object membership and `complete` checks when validating a fully assembled or deserialized result. The existing `verify_candidates`, `decide`, `analyze` and dataclass method signatures remain unchanged.

### 18.2 Metric definitions

| Item | Definition |
|---|---|
| Observed series | `y_obs = flux_rel` if it is not `None` (this covers linear flux and magnitudes converted to relative flux; the magnitude form `m = -2.5 log10(F) + C` is equivalent to a fitted flux scale), otherwise `y_norm` (labelled shape comparison). Recorded as `best_params["fit_space"]`. |
| Blocks | The valid samples are divided into contiguous time blocks (`verification.n_blocks`, **TBD** P2, CP3b). Each fold fits on some blocks and holds out others. |
| Fit | For a fixed candidate period, all other model parameters are optimised **on the fit blocks only**. Scale `a` and offset `b` are solved analytically by least squares on the fit blocks. |
| Held-out evaluation | `a`, `b` and all other parameters are **frozen** and applied to the held-out blocks. They are not re-solved on held-out data. |
| Error | For a set S of valid samples: `E(S) = sqrt(mean over S of (y_obs - (a * F_sim + b))^2) / sigma_obs`, with `sigma_obs = 1.4826 * MAD(y_obs over all valid samples)`. Dimensionless; 0 is a perfect fit; lower is better. |
| `fit_error` | Mean over folds of `E(fit samples)`. |
| `heldout_error` | Mean over folds of `E(held-out samples)`. |
| Decoy periods | Taken from `verification.decoy_ratios` times the candidate period (**provisional default `[1.37, 0.61]`**, the README's examples). A decoy is **excluded** if its frequency lies within `k * freq_resolution` of the candidate, of any harmonic-linked period, or of any alias offset from `compute_window`, or if it falls outside the period bounds. At least `verification.min_decoys` valid decoys are required (**TBD** P2, CP3b); otherwise the candidate is `status="failed"`, `notes` includes `no_valid_decoy`. |
| Decoy fit | Each decoy is fitted with the **same folds, same optimiser and same budget** as the candidate, with the period fixed at the decoy value. |
| `decoy_delta_e[i]` | `heldout_error(decoy_i) - heldout_error(candidate)`. **Positive means the candidate fits better than that decoy (favours the candidate). Negative means the decoy fits better.** Aligned index by index with `decoy_periods_s`. |
| `decoy_separation` | `min(decoy_delta_e)`: the candidate must beat **every** decoy. Computed from that candidate's own list; `None` if the list is empty. The `passed` decision uses this value and `verification.decoy_separation_min`. |
| Harmonic comparison | See Section 7.6. `delta_e = heldout_error(2P) - heldout_error(P)`; positive favours the shorter period. |
| Missing values | `None` for scalars, `[]` for lists. Never NaN or inf. |
| Aggregation across candidates | `Verdict.evidence_summary["heldout_error"]` and `Verdict.evidence_summary["decoy_separation"]` are taken from the **same** `CandidateVerification` record. Select the highest-reconciliation-score candidate among records with `status="ok"` and `passed=True`. If none passed, select the highest-reconciliation-score record with `status="ok"` as the diagnostic candidate. Break exact score ties by the lower `cluster_id`. Copy that record's `heldout_error` and `min(decoy_delta_e)` (the defined `decoy_separation`); use `None` for the corresponding missing metric. If no record has `status="ok"`, both evidence fields are `None`. Selection is independent of `per_candidate` list order. |

**Thresholds** live in `verification.yaml`: `heldout_error_max`, `decoy_separation_min`, `harmonic_margin`, `geometry_margin`, `n_blocks`, `min_decoys`, `decoy_ratios`, `timeout_s_per_candidate`. **All thresholds are TBD**: owner P2; calibrated on validation data (synthetic validation plus the real pilot `val` split) by CP4, then frozen at the `thresholds-frozen` tag. None is hardcoded.

---

## 19. Validation rules and serialization

### 19.1 Validation rules (`validate()` raises `ValueError("contract: ...")`)

| Type | Rules |
|---|---|
| `Observation` | `t_s`, `y` are 1-D float64 of equal length n >= 1. `t_s` finite. `y` may contain NaN but never +/-inf (ingestion converts inf to NaN). `y_err` is `None` or 1-D float64 of length n, values >= 0 or NaN. |
| `ReferenceLabel` | `period_s` finite and > 0. `split` in {`val`, `test`}. |
| `PreparedObservation` | All arrays 1-D with length n. `t_s` float64, finite, non-decreasing, `t_s[0] == 0.0`. `valid_mask` bool. `y_norm` float64 and `isnan(y_norm) == ~valid_mask`. `dt_s` float64, >= 0, `dt_s[0] == 0`. `flux_rel` is `None` or float64 with NaN exactly where invalid and > 0 where valid. Each `flags` array bool length n. `norm_params` has `median`, `mad`, `sign`. `meta` has the keys in Section 7.1. |
| `QualityReport` | `status` valid. `insufficient` implies `reason_codes == ["low_quality"]`. `reason_codes` subset of `REASON_CODES`. Metrics finite. |
| `Candidate` | `period_s` finite, > 0, within the injected period bounds when bounds are supplied. `support` in [0, 1]. `raw_score` finite. |
| `EstimatorResult` | `method` in `ESTIMATOR_METHODS`. If `status != "ok"` then `candidates == []`. `failed` requires a non-empty `error`; `skipped` requires `error` in `SKIP_CODES`; `ok` requires `error is None`. Candidates sorted by `support` descending. Required `meta` keys per Section 7.3. |
| `Cluster` | `evidence` keys in `ESTIMATOR_METHODS`, values in [0, 1]. `alias_risk` in [0, 1]. `score` finite. `stage2` has required keys; `passed` iff `fail_reasons == []`; every fail reason in `STAGE2_FAIL_REASONS`. `harmonic_links` entries have `other_id` (existing cluster id) and `ratio` > 0. |
| `ReconciliationResult` | Clusters sorted by `score` descending, unique `cluster_id`. `shortlist` ids exist, all have `stage2["passed"]`, ordered by score, length <= K. `stage2_passed_any == (len(shortlist) > 0)`. An empty cluster list is valid. |
| `CandidateVerification` | Allowed `(status, passed)` pairs from Section 18.1. If `status != "ok"`: `fit_error`, `heldout_error`, `best_params` are `None`. `len(decoy_delta_e) == len(decoy_periods_s)`. All floats finite. |
| `VerificationResult` | Local `validate()` checks unique `cluster_id`s and all `CandidateVerification` records. It cannot check required-set membership in isolation because the required set is derived from `ReconciliationResult`. `pipeline.controller.analyze()` derives the Section 5.5 required set, requires exactly one record per required ID and no extra IDs, and checks `complete == (IDs match exactly and every required record has status="ok")` before calling `decide()`. A mismatch is handled as a verification contract violation under Section 5.4. `AnalysisResult.validate()` repeats this cross-object check using its own `reconciliation` and `verification` fields. |
| `Verdict` | `label` valid. `Supported` implies `reason_codes == []`, `best_period_s` and `best_cluster_id` not `None`. Not `Supported` implies non-empty `reason_codes` (all in `REASON_CODES`) and both best fields `None`. `evidence_summary` has the required keys. Every `alternatives` entry has `cluster_id`, `period_s`, `why`. |
| `AnalysisResult` | `estimators` has exactly the three `ESTIMATOR_METHODS` keys. `verdict` is not `None`. When both `reconciliation` and `verification` are present, cross-check the exact Section 5.5 required ID set and the `complete` predicate as described above. If `verdict.label == "Supported"` then `verification` is not `None`, `verification.complete` is True and the best cluster's `CandidateVerification` has `status="ok"`, `passed=True`. |

NaN and infinity: NaN is allowed only in `Observation.y`, `Observation.y_err`, and in `PreparedObservation.y_norm` and `flux_rel` at invalid positions. Infinity is allowed nowhere. Scalar metrics use `None`, never NaN.

### 19.2 Serialization (`to_dict` / `from_dict`)

| Item | Encoding |
|---|---|
| NumPy array | `{"__ndarray__": true, "dtype": "<name>", "shape": [..], "data": [..]}` flat list in C order. |
| NaN in a float array | `null` inside `data`; restored as NaN on load (only for float dtypes). |
| Infinity | Not allowed: serialisation raises `ValueError("contract: non-finite value")`. |
| NumPy scalar (`np.float64`, `np.int64`, `np.bool_`) | Converted to the plain Python `float`, `int`, `bool`. |
| `None` | JSON `null`. |
| `Literal` / status strings | Validated on `from_dict`; an unknown value raises `ValueError`. |
| Nested dataclasses and lists | Recursed. Dict keys are strings. |
| Unknown extra keys on load | Ignored in `meta`/`evidence_summary`-style dicts; rejected in top-level dataclass fields. |
| `schema_version` | Stored in `AnalysisResult`; `from_dict` rejects a different version. |

**Equality in tests.** Ordinary `==` is never used on objects containing arrays. Tests use `core.schemas.assert_contract_equal(a, b, rtol=1e-12, atol=0.0)`: arrays must have identical shape and dtype, identical NaN positions, and `np.allclose(..., equal_nan=True)`; floats are compared with `rtol`; bools, ints, strings and `None` exactly; lists, dicts and dataclasses recursively. `test_serialization.py` checks value preservation, shape preservation, dtype preservation and optional-field behaviour.

---

## 20. Progress callback convention

```python
ProgressCallback = Callable[[float, str], None]   # (fraction, message)
```

- **Optional.** `None` means no reporting; modules then do nothing extra.
- `fraction` is a float in [0, 1] representing **the fraction of that function's own work completed**. Within one call it is non-decreasing. A function calls the callback with `0.0` when it starts, with increasing values during work, and with `1.0` when it ends normally.
- `message` is a short, non-empty, human-readable `str` (for example `"fitting candidate 2 of 3"`). No newlines.
- Only `verify_candidates` and `analyze` accept a callback. P1's modules are fast and do not take one. `analyze` maps its stages onto overall fractions (it may call `verify_candidates` with a wrapper that rescales its own 0-1 range into the verification share).
- The callback is called from the same thread. Exceptions raised inside the callback are caught and ignored by the caller; progress reporting must never break an analysis.
- Interrupting an analysis is not supported in this build; a timeout makes a candidate `failed` (Section 18.1).

---

## 21. Register of TBD decisions

| Decision | Config location | Owner | By |
|---|---|---|---|
| `min_period_s`, `max_period_s` | `period_bounds.yaml` | P1 determines (data audit); P2 enters | CP1 |
| Stage 1 thresholds (suitable/limited/insufficient) | `preprocess.yaml` | P1 | CP1 |
| "No significant candidate" thresholds for LS and PDM | `classical.yaml` | P1 | CP2 |
| `y_clip`, `dt_floor_s`, `L_min`, `L_max`, `max_windows`, `min_valid_per_window` | `cnn.yaml` | P2 | CP2 |
| Provisional `window_stride = 512` validated | `cnn.yaml` | P2 | CP3b |
| `cnn.abstain_threshold`, `n_candidates` | `cnn.yaml` | P2 | CP3b |
| `min_cycles`, `max_rel_period_uncertainty`, phase-coverage settings, `min_separation_resolutions`, `cluster_tol_resolutions` | `reconcile.yaml` | P1 | CP3a |
| `w_C`, `w_L`, `w_D`, `lambda_A` | `reconcile.yaml` | P1 (tuned on validation) | CP4 |
| `n_blocks`, `min_decoys`, `timeout_s_per_candidate`, decoy ratios confirmed | `verification.yaml` | P2 | CP3b |
| `heldout_error_max`, `decoy_separation_min`, `harmonic_margin`, `geometry_margin` | `verification.yaml` | P2 (calibrated on validation) | CP4 |
| `alias_risk_max`, `min_agreeing_estimators` (default 2), `limited_max_label` | `verdict.yaml` | P2 | CP4 |

Nothing in this table has an invented value. A default shown in this file (for example `window_stride = 512`, `top_k = 5`, `decoy_ratios = [1.37, 0.61]`, `min_agreeing_estimators = 2`) is either taken from the README or marked provisional.