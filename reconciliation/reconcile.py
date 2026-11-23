"""Reconcile apparent-period candidates and shortlist those passing Stage 2."""

from math import isfinite

from core.schemas import EstimatorResult, PreparedObservation, ReconciliationResult
from reconciliation.aliases import alias_risks
from reconciliation.cluster import cluster_candidates
from reconciliation.harmonics import link_harmonics
from reconciliation.score import score_clusters
from reconciliation.stage2 import assess_stage2


def _validate_settings(cfg):
    for key in ("top_k", "n_phase_bins", "min_points_per_bin"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "cluster_tol_resolutions", "harmonic_ratio_tolerance", "alias_tol_resolutions",
        "alias_min_source_support", "w_C", "w_L", "w_D", "lambda_A", "min_cycles",
        "max_rel_period_uncertainty", "min_phase_coverage", "min_separation_resolutions",
    ):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("alias_min_source_support", "min_phase_coverage"):
        if cfg[key] > 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if cfg["harmonic_ratio_tolerance"] >= 1 / 3:
        raise ValueError("harmonic_ratio_tolerance must keep equal periods non-harmonic")


def reconcile(prep: PreparedObservation, estimators: list[EstimatorResult],
              cfg: dict) -> ReconciliationResult:
    prep.validate()
    available = {result.method for result in estimators if result.status == "ok"}
    entries = [(result.method, candidate) for result in estimators if result.status == "ok"
               for candidate in result.candidates]
    if not entries:
        note = "no_candidates" if available else "no_usable_estimators"
        return ReconciliationResult([], [], False, [note])

    cfg = {"top_k": 5, **cfg}
    _validate_settings(cfg)
    lower = cfg.get("period_min_s", 0.0)
    upper = cfg.get("period_max_s", float("inf"))
    if any(not lower <= candidate.period_s <= upper for _, candidate in entries):
        raise ValueError("contract: candidate outside injected period bounds")

    notes = []
    clusters, widths = cluster_candidates(entries, prep.meta["freq_resolution"], cfg)
    link_harmonics(clusters, cfg)
    score_clusters(clusters, available, alias_risks(prep, clusters, cfg), cfg, notes)
    clusters.sort(key=lambda cluster: (-cluster.score, cluster.period_s, cluster.cluster_id))
    assess_stage2(prep, clusters, widths, cfg)
    shortlist = [cluster.cluster_id for cluster in clusters if cluster.stage2["passed"]][:cfg["top_k"]]
    shortlisted = set(shortlist)
    partners = sorted({link["other_id"] for cluster in clusters if cluster.cluster_id in shortlisted
                       for link in cluster.harmonic_links if link["ratio"] in (2.0, 0.5)})
    if partners:
        notes.append(f"Direct 2:1 harmonic partner IDs of shortlisted clusters: {partners}.")
    result = ReconciliationResult(clusters, shortlist, bool(shortlist), notes)
    result.validate()
    return result
