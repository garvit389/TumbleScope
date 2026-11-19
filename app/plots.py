"""Phase 0 figure placeholders."""

import plotly.graph_objects as go


def _placeholder():
    figure = go.Figure()
    figure.add_annotation(text="Phase 0 plot placeholder", showarrow=False)
    return figure


def fig_lightcurve(prep):
    return _placeholder()


def fig_periodograms(estimators):
    return _placeholder()


def fig_phase_fold(prep, period_s):
    return _placeholder()


def fig_candidates(recon):
    return _placeholder()


def fig_quality(report):
    return _placeholder()


def fig_verification(verification):
    return _placeholder()


def fig_risk_coverage(eval_df):
    return _placeholder()


def fig_calibration(eval_df):
    return _placeholder()
