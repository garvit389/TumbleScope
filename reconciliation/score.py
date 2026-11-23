"""Combine per-estimator support on the shared normalized scale."""

from math import isfinite

from core.schemas import CNN, ESTIMATOR_METHODS, LOMB_SCARGLE, PDM


_WEIGHT_KEYS = {CNN: "w_C", LOMB_SCARGLE: "w_L", PDM: "w_D"}


def score_clusters(clusters, available, risks, cfg, notes):
    weights = {method: cfg[_WEIGHT_KEYS[method]] for method in ESTIMATOR_METHODS
               if method in available}
    total = sum(weights.values())
    if not isfinite(total) or total <= 0:
        raise ValueError("contract: reconciliation requires finite positive total available weight")
    unavailable = [method for method in ESTIMATOR_METHODS if method not in available]
    if unavailable:
        notes.append(f"Renormalized weights over available estimators. Unavailable: {', '.join(unavailable)}.")

        weights = {method: weight / total for method, weight in weights.items()}

    notes.append("Effective estimator weights: " + ", ".join(
        f"{method}={weight:.17g}" for method, weight in weights.items()) + ".")
    for cluster in clusters:
        cluster.alias_risk = risks[cluster.cluster_id]
        support = sum(weight * cluster.evidence.get(method, 0.0)
                      for method, weight in weights.items())
        cluster.score = support - cfg["lambda_A"] * cluster.alias_risk
