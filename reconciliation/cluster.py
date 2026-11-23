"""Group nearby frequencies while retaining distinct harmonic hypotheses."""

from math import ulp

from core.schemas import Cluster
from reconciliation.harmonics import harmonic_ratio


def frequencies_within(first, second, tolerance):
    """Inclusive frequency comparison, allowing only arithmetic roundoff."""
    separation = abs(first - second)
    if tolerance == 0:
        return separation == 0
    roundoff = 4 * max(ulp(first), ulp(second), ulp(tolerance))
    return separation <= tolerance or abs(separation - tolerance) <= roundoff


def cluster_candidates(entries, resolution, cfg):
    """Build bounded-span groups with one strongest representative per method."""
    tolerance = cfg["cluster_tol_resolutions"] * (resolution or 0.0)
    ordered = sorted(entries, key=lambda entry: (
        -entry[1].period_s, entry[0], -entry[1].support, -entry[1].raw_score,
        -entry[1].meta.get("freq_width_hz", 0.0),
    ))
    # A representative must not erase a member's relation to another hypothesis.
    # Near duplicates with different harmonic neighbours remain separate clusters.
    periods = sorted({candidate.period_s for _, candidate in ordered})
    contexts = {period: tuple(harmonic_ratio(period, other, cfg) for other in periods)
                for period in periods}
    groups = []
    for method, candidate in ordered:
        if (not groups
                or not frequencies_within(1 / candidate.period_s,
                                          1 / groups[-1][0][1].period_s, tolerance)
                or contexts[candidate.period_s] != contexts[groups[-1][0][1].period_s]
                or any(harmonic_ratio(candidate.period_s, other.period_s, cfg) is not None
                       for _, other in groups[-1])):
            groups.append([])
        groups[-1].append((method, candidate))

    clusters = []
    widths = {}
    for cluster_id, group in enumerate(groups, start=1):
        ranked = sorted(group, key=lambda entry: (
            -entry[1].support, entry[1].period_s, entry[0], -entry[1].raw_score,
            -entry[1].meta.get("freq_width_hz", 0.0),
        ))
        selected = {}
        for method, candidate in ranked:
            selected.setdefault(method, candidate)
        # Keep the strongest observed period instead of interpolating a new peak.
        period = ranked[0][1].period_s
        widths[cluster_id] = max(
            (candidate.meta.get("freq_width_hz", 0.0) for candidate in selected.values()),
            default=0.0,
        ) or resolution
        clusters.append(Cluster(
            cluster_id, period,
            {method: selected[method].period_s for method in sorted(selected)},
            {method: selected[method].support for method in sorted(selected)},
            0.0, [], 0.0, {},
        ))
    return clusters, widths
