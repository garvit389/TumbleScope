from importlib import import_module
from inspect import Parameter, Signature, signature
from typing import Optional

import numpy as np
import pytest

from core.schemas import (
    AnalysisResult, EstimatorResult, Observation, PreparedObservation, ProgressCallback,
    QualityReport, ReconciliationResult, ReferenceLabel, VerificationResult, Verdict,
)


def argument(name, annotation=Parameter.empty, default=Parameter.empty):
    return Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, default=default, annotation=annotation)


PREP = argument("prep", PreparedObservation)
CFG = argument("cfg", dict)
MODE = argument("mode", str, "fast")
PROGRESS = argument("progress", Optional[ProgressCallback], None)

FUNCTIONS = [
    ("ingestion", "load_observation", [argument("path", str), argument("fmt", Optional[str], None)],
     Observation),
    ("ingestion.reference", "load_reference_table", [argument("path", str)],
     dict[str, ReferenceLabel]),
    ("preprocessing.prepare", "prepare", [argument("obs", Observation), CFG], PreparedObservation),
    ("preprocessing.quality", "assess_stage1", [PREP, CFG], QualityReport),
    ("classical_analysis.lomb_scargle", "run_lomb_scargle", [PREP, CFG], EstimatorResult),
    ("classical_analysis.pdm", "run_pdm", [PREP, CFG], EstimatorResult),
    ("classical_analysis.window", "compute_window", [PREP, CFG], dict),
    ("reconciliation.reconcile", "reconcile", [PREP, argument("estimators", list[EstimatorResult]), CFG],
     ReconciliationResult),
    ("ml.features", "build_cnn_windows", [PREP, CFG], np.ndarray),
    ("ml.infer", "run_cnn", [PREP, CFG], EstimatorResult),
    ("physics.simulator", "simulate_lightcurve", [argument("params", "SimParams"),
                                                argument("t_s", np.ndarray)], np.ndarray),
    ("physics.verify", "verify_candidates", [PREP, argument("recon", ReconciliationResult),
                                             CFG, MODE, PROGRESS], VerificationResult),
    ("verdict.decide", "decide", [argument("quality", QualityReport),
                                 argument("estimators", dict[str, EstimatorResult]),
                                 argument("recon", Optional[ReconciliationResult]),
                                 argument("verification", Optional[VerificationResult]), CFG], Verdict),
    ("pipeline.controller", "analyze", [argument("obs", Observation), argument("cfg", "PipelineConfig"),
                                        MODE, PROGRESS], AnalysisResult),
]

FUNCTIONS += [
    ("app.plots", name, [argument(part) for part in arguments], Signature.empty)
    for name, arguments in (
        ("fig_lightcurve", ["prep"]),
        ("fig_periodograms", ["estimators"]),
        ("fig_phase_fold", ["prep", "period_s"]),
        ("fig_candidates", ["recon"]),
        ("fig_quality", ["report"]),
        ("fig_verification", ["verification"]),
        ("fig_risk_coverage", ["eval_df"]),
        ("fig_calibration", ["eval_df"]),
    )
]


@pytest.mark.parametrize("module,name,parameters,returns", FUNCTIONS, ids=[item[1] for item in FUNCTIONS])
def test_frozen_signatures(module, name, parameters, returns):
    function = getattr(import_module(module), name)
    assert signature(function) == Signature(parameters, return_annotation=returns)
