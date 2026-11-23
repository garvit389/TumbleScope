"""Assess cycles, resolution, phase coverage, and separation for each cluster."""

import numpy as np

from core.schemas import STAGE2_FAIL_REASONS
from reconciliation.cluster import frequencies_within


INSUFFICIENT_CYCLES, UNRESOLVED_FREQUENCY, POOR_PHASE_COVERAGE, MERGED_INTO_NEIGHBOUR = (
    STAGE2_FAIL_REASONS
)


def assess_stage2(prep, clusters, widths, cfg):
    times = prep.t_s[prep.valid_mask]
    baseline = prep.meta["baseline_s"]
    resolution = prep.meta["freq_resolution"]
    bins = cfg["n_phase_bins"]
    frequencies = {cluster.cluster_id: 1 / cluster.period_s for cluster in clusters}

    for cluster in clusters:
        period = cluster.period_s
        cycles = float(baseline / period)
        phases = np.remainder(times, period) / period
        indices = np.minimum((phases * bins).astype(int), bins - 1)
        counts = np.bincount(indices, minlength=bins)
        coverage = float(np.count_nonzero(counts >= cfg["min_points_per_bin"]) / bins)
        width = widths[cluster.cluster_id]
        failures = []
        diagnostics = {"cycles": cycles, "phase_coverage": coverage, "fail_reasons": failures}
        if cycles < cfg["min_cycles"]:
            failures.append(INSUFFICIENT_CYCLES)
        if width is not None:
            diagnostics["rel_period_uncertainty"] = float(width * period)
        if width is None or diagnostics["rel_period_uncertainty"] > cfg["max_rel_period_uncertainty"]:
            failures.append(UNRESOLVED_FREQUENCY)
        if coverage < cfg["min_phase_coverage"]:
            failures.append(POOR_PHASE_COVERAGE)
        if resolution is not None:
            harmonic_ids = {link["other_id"] for link in cluster.harmonic_links}
            for other in clusters:
                if other.score <= cluster.score:
                    break
                if (other.cluster_id not in harmonic_ids
                        and frequencies_within(frequencies[cluster.cluster_id],
                                               frequencies[other.cluster_id],
                                               cfg["min_separation_resolutions"] * resolution)):
                    failures.append(MERGED_INTO_NEIGHBOUR)
                    diagnostics["merged_with"] = other.cluster_id
                    break
        cluster.stage2 = {**diagnostics, "passed": not failures}
