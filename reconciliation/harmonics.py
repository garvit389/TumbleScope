"""Identify reciprocal links between distinct apparent-period hypotheses."""

from itertools import combinations


_RATIOS = (1.5, 2.0, 3.0)


def harmonic_ratio(period, other_period, cfg):
    """Return the nominal other-period / period ratio when it matches."""
    short, long = sorted((period, other_period))
    measured = long / short
    nominal = min(_RATIOS, key=lambda ratio: abs(measured / ratio - 1))
    if abs(measured / nominal - 1) <= cfg["harmonic_ratio_tolerance"]:
        return nominal if other_period > period else 1.0 / nominal
    return None


def link_harmonics(clusters, cfg):
    for first, second in combinations(clusters, 2):
        ratio = harmonic_ratio(first.period_s, second.period_s, cfg)
        if ratio is not None:
            first.harmonic_links.append({"other_id": second.cluster_id, "ratio": ratio})
            second.harmonic_links.append({"other_id": first.cluster_id, "ratio": 1.0 / ratio})
