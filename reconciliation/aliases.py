"""Compare candidate frequency differences with sampling-window offsets."""

from math import isfinite
from numbers import Real

from classical_analysis.window import compute_window
from reconciliation.cluster import frequencies_within


def _window_peaks(window, resolution):
    """Validate the shared Section 7.4 payload before using its offsets."""
    def number(value):
        return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)

    if not isinstance(window, dict):
        raise ValueError("contract: invalid window_result")
    frequencies, powers, peaks = (window.get(key) for key in ("freq_hz", "window_power", "peaks"))
    if (not all(isinstance(values, list) for values in (frequencies, powers, peaks))
            or not frequencies or len(frequencies) != len(powers)):
        raise ValueError("contract: invalid window_result arrays")
    if (not all(number(value) for value in frequencies + powers)
            or frequencies[0] != 0 or powers[0] != 1
            or any(second <= first for first, second in zip(frequencies, frequencies[1:]))
            or any(not 0 <= power <= 1 for power in powers)):
        raise ValueError("contract: invalid window_result values")
    if "freq_resolution" not in window or window["freq_resolution"] != resolution:
        raise ValueError("contract: window_result frequency resolution mismatch")
    grid_power = dict(zip(frequencies, powers))
    previous_height = 1.0
    for peak in peaks:
        if (not isinstance(peak, dict) or not {"freq_hz", "height"} <= peak.keys()
                or not number(peak["freq_hz"]) or not 0 < peak["freq_hz"] <= frequencies[-1]
                or not number(peak["height"]) or not 0 <= peak["height"] <= previous_height
                or grid_power.get(peak["freq_hz"]) != peak["height"]):
            raise ValueError("contract: invalid window_result peak")
        previous_height = peak["height"]
    return peaks


def alias_risks(prep, clusters, cfg):
    risks = dict.fromkeys((cluster.cluster_id for cluster in clusters), 0.0)
    resolution = prep.meta["freq_resolution"]
    # Direct callers retain standalone window computation. The controller's
    # shared result always takes precedence, including an explicitly empty peak list.
    window = cfg["window_result"] if "window_result" in cfg else compute_window(prep, cfg)
    peaks = _window_peaks(window, resolution)
    if resolution is None or len(clusters) < 2:
        return risks

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
                if frequencies_within(offset, peak["freq_hz"], tolerance):
                    # Keep the strongest match. Window peaks and supports are in [0, 1].
                    risks[cluster.cluster_id] = max(
                        risks[cluster.cluster_id], peak["height"] * strength,
                    )
    return risks
