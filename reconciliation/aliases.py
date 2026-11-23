"""Compare candidate frequency differences with sampling-window offsets."""

from classical_analysis.window import compute_window


def alias_risks(prep, clusters, cfg):
    risks = dict.fromkeys((cluster.cluster_id for cluster in clusters), 0.0)
    resolution = prep.meta["freq_resolution"]
    if resolution is None or len(clusters) < 2:
        return risks

    peaks = compute_window(prep, cfg)["peaks"]
    tolerance = cfg["alias_tol_resolutions"] * resolution
    frequencies = {cluster.cluster_id: 1 / cluster.period_s for cluster in clusters}
    for cluster in clusters:
        frequency = frequencies[cluster.cluster_id]
        for other in clusters:
            if other.cluster_id == cluster.cluster_id:
                continue
            strength = max(other.evidence.values())
            if strength < cfg["alias_min_source_support"]:
                continue
            offset = abs(frequency - frequencies[other.cluster_id])
            for peak in peaks:
                if abs(offset - peak["freq_hz"]) <= tolerance:
                    # Keep the strongest match. Window peaks and supports are in [0, 1].
                    risks[cluster.cluster_id] = max(
                        risks[cluster.cluster_id], peak["height"] * strength,
                    )
    return risks
