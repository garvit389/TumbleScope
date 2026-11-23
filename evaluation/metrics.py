"""Pure metrics for apparent-period evaluation and binary reliability.

Use None for missing individual measurements and undefined aggregate values.
An explicit No Reliable Period label is independent of a missing reference.
The caller chooses the tolerance, cohorts, and binary probability event.
"""

from bisect import bisect_right
from math import isfinite, log
from numbers import Integral, Real
from statistics import median
from typing import Literal, Sequence

import numpy as np


_Number = float | None
Disposition = Literal["supported", "abstained", "pending", "execution_error", "missing"]
_DISPOSITIONS = ("supported", "abstained", "pending", "execution_error", "missing")


def _finite(value):
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        try:
            number = float(value)
        except OverflowError:
            return None
        if isfinite(number):
            return number
    return None


def _period(value):
    value = _finite(value)
    return value if value is not None and value > 0 else None


def _tolerance(value):
    tolerance = _finite(value)
    if tolerance is None or not 0 <= tolerance < 1:
        raise ValueError("tolerance must be finite and in [0, 1)")
    return tolerance


def _integer(value, name):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _hit(prediction, reference, tolerance, ratio=1.0):
    relative = prediction / reference
    return ratio * (1 - tolerance) <= relative <= ratio * (1 + tolerance)


def _rate(numerator, denominator):
    return {"value": numerator / denominator if denominator else None,
            "numerator": numerator, "denominator": denominator}


def _period_pairs(reference_periods_s, predicted_periods_s):
    pairs = [(_period(reference), _period(prediction))
             for reference, prediction in zip(reference_periods_s, predicted_periods_s, strict=True)]
    known = [(reference, prediction) for reference, prediction in pairs if reference is not None]
    matched = [(reference, prediction) for reference, prediction in known if prediction is not None]
    counts = {"n_input": len(pairs), "n_known_period": len(known), "n_predicted": len(matched),
              "n_missing_reference": len(pairs) - len(known),
              "n_missing_prediction": len(known) - len(matched)}
    return matched, counts


def median_log_period_error(reference_periods_s: Sequence[_Number], predicted_periods_s: Sequence[_Number]):
    """Median absolute natural-log error over valid reference/prediction pairs."""
    pairs, counts = _period_pairs(reference_periods_s, predicted_periods_s)
    errors = [abs(log(prediction) - log(reference)) for reference, prediction in pairs]
    return {"value": median(errors) if errors else None, "denominator": len(errors), **counts}


def tolerance_accuracy(reference_periods_s: Sequence[_Number], predicted_periods_s: Sequence[_Number], *,
                       tolerance: float):
    """Direct hits over known periods and, separately, over emitted valid predictions."""
    tolerance = _tolerance(tolerance)
    pairs, counts = _period_pairs(reference_periods_s, predicted_periods_s)
    hits = sum(_hit(prediction, reference, tolerance) for reference, prediction in pairs)
    return {"overall": _rate(hits, counts["n_known_period"]),
            "conditional": _rate(hits, len(pairs)), **counts}


def harmonic_error_rate(reference_periods_s: Sequence[_Number], predicted_periods_s: Sequence[_Number], *,
                        tolerance: float, harmonic_tolerance: float):
    """Half/double-period mistakes over valid pairs, excluding direct hits."""
    tolerance = _tolerance(tolerance)
    harmonic_tolerance = _tolerance(harmonic_tolerance)
    pairs, counts = _period_pairs(reference_periods_s, predicted_periods_s)
    errors = sum(
        not _hit(prediction, reference, tolerance)
        and any(_hit(prediction, reference, harmonic_tolerance, ratio) for ratio in (0.5, 2.0))
        for reference, prediction in pairs
    )
    return {**_rate(errors, len(pairs)), **counts}


def catastrophic_error_rate(reference_periods_s: Sequence[_Number], predicted_periods_s: Sequence[_Number], *,
                            factor: float):
    """Fraction of valid pairs strictly beyond the multiplicative error factor."""
    factor = _finite(factor)
    if factor is None or factor <= 1:
        raise ValueError("factor must be finite and greater than one")
    pairs, counts = _period_pairs(reference_periods_s, predicted_periods_s)
    errors = sum(max(prediction / reference, reference / prediction) > factor
                 for reference, prediction in pairs)
    return {**_rate(errors, len(pairs)), **counts}


def _candidate_hit(reference, candidates, tolerance):
    return any(_hit(period, reference, tolerance)
               for value in (() if candidates is None else candidates)
               if (period := _period(value)) is not None)


def recall_at_k(reference_periods_s: Sequence[_Number],
                candidate_periods_s: Sequence[Sequence[_Number] | None], *, k: int, tolerance: float):
    """Direct top-K recall over known periods. Invalid entries keep their ranks."""
    k = _integer(k, "k")
    tolerance = _tolerance(tolerance)
    rows = [(_period(reference), candidates)
            for reference, candidates in zip(reference_periods_s, candidate_periods_s, strict=True)]
    known = [(reference, candidates) for reference, candidates in rows if reference is not None]
    hits = sum(_candidate_hit(reference, None if candidates is None else candidates[:k], tolerance)
               for reference, candidates in known)
    return {**_rate(hits, len(known)), "n_input": len(rows), "k": k}


def candidate_retention(reference_periods_s: Sequence[_Number],
                        upstream_periods_s: Sequence[Sequence[_Number] | None],
                        downstream_periods_s: Sequence[Sequence[_Number] | None], *, tolerance: float):
    """Downstream direct hits among known-period rows that had an upstream hit."""
    tolerance = _tolerance(tolerance)
    rows = [(_period(reference), upstream, downstream) for reference, upstream, downstream in
            zip(reference_periods_s, upstream_periods_s, downstream_periods_s, strict=True)]
    known = [(reference, upstream, downstream) for reference, upstream, downstream in rows
             if reference is not None]
    retained_pool = [(reference, downstream) for reference, upstream, downstream in known
                     if _candidate_hit(reference, upstream, tolerance)]
    hits = sum(_candidate_hit(reference, downstream, tolerance) for reference, downstream in retained_pool)
    return {**_rate(hits, len(retained_pool)), "n_input": len(rows), "n_known_period": len(known)}


def supported_metrics(reference_periods_s: Sequence[_Number], predicted_periods_s: Sequence[_Number],
                      dispositions: Sequence[Disposition], *, no_reliable_period: Sequence[bool],
                      tolerance: float):
    """Summarize a caller-selected cohort of eligible attempts.

    Coverage uses all attempts. False-Supported rates use evaluable Supported
    claims and all labeled attempts, respectively. An explicit No Reliable
    Period label makes a Supported claim false even without a usable period.
    An invalid claimed period on known-period truth is tallied separately.
    """
    tolerance = _tolerance(tolerance)
    counts = dict.fromkeys(_DISPOSITIONS, 0)
    labeled = evaluated = incorrect = nrp_count = correct_abstentions = 0
    supported_unknown = supported_missing = missing_prediction_count = 0
    for reference, prediction, disposition, nrp in zip(
        reference_periods_s, predicted_periods_s, dispositions, no_reliable_period, strict=True,
    ):
        reference, prediction = _period(reference), _period(prediction)
        if disposition not in counts:
            raise ValueError(f"unknown disposition {disposition!r}")
        if not isinstance(nrp, (bool, np.bool_)) or (nrp and reference is not None):
            raise ValueError("No Reliable Period labels must be boolean and cannot have period truth")
        nrp = bool(nrp)
        counts[disposition] += 1
        labeled += reference is not None or nrp
        nrp_count += nrp
        missing_prediction_count += prediction is None
        correct_abstentions += nrp and disposition == "abstained"
        if disposition != "supported":
            continue
        if nrp:
            evaluated += 1
            incorrect += 1
        elif reference is None:
            supported_unknown += 1
        elif prediction is None:
            supported_missing += 1
        else:
            evaluated += 1
            incorrect += not _hit(prediction, reference, tolerance)

    eligible = sum(counts.values())
    return {
        "coverage": _rate(counts["supported"], eligible),
        "false_supported": _rate(incorrect, evaluated),
        "false_supported_per_labeled": _rate(incorrect, labeled),
        "supported_accuracy": _rate(evaluated - incorrect, evaluated),
        "correct_abstention": _rate(correct_abstentions, nrp_count),
        "disposition_counts": counts, "n_eligible": eligible, "n_labeled": int(labeled),
        "n_unknown_truth": eligible - int(labeled), "n_no_reliable_period": int(nrp_count),
        "n_supported_unknown_truth": supported_unknown,
        "n_supported_missing_prediction": supported_missing,
        "n_missing_prediction": missing_prediction_count,
    }


def _binary_pairs(scores, outcomes, *, probabilities):
    pairs = []
    n_input = 0
    for score, outcome in zip(scores, outcomes, strict=True):
        n_input += 1
        score = _finite(score)
        outcome = int(outcome) if isinstance(outcome, (bool, np.bool_)) else _finite(outcome)
        if score is not None and outcome in (0, 1) and (not probabilities or 0 <= score <= 1):
            pairs.append((score, int(outcome)))
    return pairs, n_input


def risk_coverage_points(scores: Sequence[_Number], correctness: Sequence[bool | _Number], *,
                         thresholds: Sequence[float] | None = None):
    """Return rows for score >= threshold; default cutoffs are distinct scores.

    Denominator for coverage is the cohort with both a finite ranking score and
    an explicit binary correctness outcome. Higher scores rank first. These
    scores need not be probabilities and are never used as such here.
    """
    pairs, n_input = _binary_pairs(scores, correctness, probabilities=False)
    if thresholds is None:
        cutoffs = sorted({score for score, _ in pairs}, reverse=True)
    else:
        cutoffs = [_finite(threshold) for threshold in thresholds]
        if any(cutoff is None for cutoff in cutoffs):
            raise ValueError("thresholds must be finite")
        cutoffs.sort(reverse=True)
    pairs.sort(reverse=True)
    selected = incorrect = 0
    rows = []
    for cutoff in cutoffs:
        while selected < len(pairs) and pairs[selected][0] >= cutoff:
            incorrect += 1 - pairs[selected][1]
            selected += 1
        rows.append({
            "threshold": cutoff, "coverage": selected / len(pairs) if pairs else None,
            "risk": incorrect / selected if selected else None, "n_selected": selected,
            "n_incorrect": incorrect, "n_eligible": len(pairs), "n_input": n_input,
        })
    return rows


def brier_score(probabilities: Sequence[_Number], outcomes: Sequence[bool | _Number]):
    """Binary mean squared error on valid probability/outcome pairs."""
    pairs, n_input = _binary_pairs(probabilities, outcomes, probabilities=True)
    squared_error = sum(((probability - outcome) ** 2 for probability, outcome in pairs), 0.0)
    return {**_rate(squared_error, len(pairs)), "n_input": n_input}


def expected_calibration_error(probabilities: Sequence[_Number], outcomes: Sequence[bool | _Number], *,
                               n_bins: int):
    """Binary ECE and reliability rows from equal-width probability bins.

    Intervals are left-closed and right-open, except the last includes 1.
    In each bin, observed_accuracy is the event frequency for the given
    probability, such as correctness or No Reliable Period.
    """
    n_bins = _integer(n_bins, "n_bins")
    pairs, n_input = _binary_pairs(probabilities, outcomes, probabilities=True)
    edges = [index / n_bins for index in range(n_bins + 1)]
    bins = {}
    for probability, outcome in pairs:
        index = min(bisect_right(edges, probability) - 1, n_bins - 1)
        count, confidence, positives = bins.get(index, (0, 0.0, 0))
        bins[index] = (count + 1, confidence + probability, positives + outcome)
    rows = [{"mean_confidence": confidence / count, "observed_accuracy": positives / count,
             "n": count, "bin_lower": index / n_bins, "bin_upper": (index + 1) / n_bins}
            for index, (count, confidence, positives) in sorted(bins.items())]
    weighted_error = sum((abs(confidence - positives) for _, confidence, positives in bins.values()), 0.0)
    return {**_rate(weighted_error, len(pairs)), "n_input": n_input, "bins": rows}
