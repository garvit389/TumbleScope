"""Pure Plotly figures for contract results and evaluation summaries."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.schemas import CNN, ESTIMATOR_METHODS, LOMB_SCARGLE, PDM


_SOURCES = {
    CNN: ("CNN", "#0072B2", "diamond"),
    LOMB_SCARGLE: ("Lomb–Scargle (LS)", "#D55E00", "circle"),
    PDM: ("PDM", "#009E73", "square"),
}
_FLAGS = {
    "bright_outlier": ("Brightness outlier", "x"),
    "duplicate": ("Duplicate timestamp", "circle-open"),
    "gap_edge": ("Gap boundary", "triangle-up-open"),
}
_PERIOD = "Apparent period (s)"
_NEUTRAL = "#475569"


def _layout(figure, title):
    figure.update_layout(title=title, template="plotly_white", margin=dict(t=65, b=55))
    return figure


def _message(figure, message, **kwargs):
    figure.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper", text=message,
        showarrow=False, align="center", **kwargs,
    )
    return figure


def _empty(title, message):
    figure = _layout(go.Figure(), title)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _message(figure, message)


def _valid_samples(prep):
    return prep.valid_mask & np.isfinite(prep.t_s) & np.isfinite(prep.y_norm)


def _observations(figure, prep, valid, x):
    figure.add_scatter(
        x=x, y=prep.y_norm[valid], mode="markers", name="Valid samples",
        marker=dict(color=_NEUTRAL, size=5, opacity=0.75),
    )
    for key, (label, symbol) in _FLAGS.items():
        flag = prep.flags.get(key)
        if flag is not None and flag[valid].any():
            marked = flag[valid]
            figure.add_scatter(
                x=x[marked], y=prep.y_norm[valid][marked], mode="markers", name=label,
                marker=dict(color="#7E3F98", symbol=symbol, size=10),
            )


def fig_lightcurve(prep):
    title = "Light curve"
    if prep is None:
        return _empty(title, "Prepared observation unavailable.")
    valid = _valid_samples(prep)
    if not valid.any():
        return _empty(title, "No valid brightness samples.")
    figure = _layout(go.Figure(), title)
    _observations(figure, prep, valid, prep.t_s[valid])
    figure.update_xaxes(title_text="Elapsed time (s)")
    figure.update_yaxes(title_text="Normalized brightness")
    return figure


def _source_notice(result):
    if result is None:
        return "Estimator unavailable."
    if result.status != "ok":
        detail = f" — {result.error}" if result.error else ""
        return f"Estimator {result.status}{detail}."
    if result.method == CNN and result.meta.get("abstained"):
        return "CNN abstained."
    if not result.candidates:
        return "No significant candidate."
    return ""


def fig_periodograms(estimators):
    labels = ("Candidate support", "LS power", "PDM 1 − θ")
    figure = _layout(make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[_SOURCES[method][0] for method in ESTIMATOR_METHODS],
    ), "Estimator evidence")
    figure.update_layout(height=790)
    for row, (method, y_title) in enumerate(zip(ESTIMATOR_METHODS, labels), 1):
        name, color, symbol = _SOURCES[method]
        result = (estimators or {}).get(method)
        notice = _source_notice(result)
        if result is not None and result.status == "ok":
            candidates = result.candidates
            if method == CNN:
                if candidates:
                    figure.add_scatter(
                        x=[item.period_s for item in candidates],
                        y=[item.support for item in candidates],
                        name=name, mode="markers", marker=dict(color=color, symbol=symbol, size=9),
                        row=row, col=1,
                    )
            else:
                spectrum = result.meta.get("periodogram", {})
                frequencies = np.asarray(spectrum.get("freq_hz", []), dtype=float)
                power = np.asarray(spectrum.get("power", []), dtype=float)
                usable = np.isfinite(frequencies) & (frequencies > 0) & np.isfinite(power)
                if usable.any():
                    periods = 1 / frequencies[usable]
                    order = np.argsort(periods)
                    figure.add_scatter(
                        x=periods[order], y=power[usable][order],
                        name=name, mode="lines", line=dict(color=color), row=row, col=1,
                    )
                else:
                    notice = "Periodogram unavailable." + (f" {notice}" if notice else "")
                if candidates:
                    figure.add_scatter(
                        x=[item.period_s for item in candidates],
                        y=[item.raw_score for item in candidates], mode="markers",
                        name=f"{name} candidates", marker=dict(color=color, symbol=symbol, size=8),
                        row=row, col=1,
                    )
        if notice:
            figure.add_annotation(
                x=0.02, y=0.94, xref="x domain", yref="y domain", xanchor="left",
                text=notice, showarrow=False, row=row, col=1,
            )
        figure.update_yaxes(title_text=y_title, row=row, col=1)
    figure.update_xaxes(title_text=_PERIOD, row=3, col=1)
    return figure


def fig_phase_fold(prep, period_s):
    title = "Phase-folded light curve"
    try:
        period = float(period_s)
    except (TypeError, ValueError, OverflowError):
        return _empty(title, "A finite positive apparent period is required.")
    if not np.isfinite(period) or period <= 0:
        return _empty(title, "A finite positive apparent period is required.")
    if prep is None:
        return _empty(title, "Prepared observation unavailable.")
    valid = _valid_samples(prep)
    if not valid.any():
        return _empty(title, "No valid brightness samples to fold.")
    figure = _layout(go.Figure(), f"{title} · {period:g} s")
    phase = np.remainder(prep.t_s[valid], period) / period
    _observations(figure, prep, valid, np.minimum(phase, np.nextafter(1.0, 0.0)))
    figure.update_xaxes(title_text="Phase", range=[0, 1])
    figure.update_yaxes(title_text="Normalized brightness")
    return figure


def fig_candidates(recon):
    title = "Reconciled candidates"
    if recon is None:
        return _empty(title, "Reconciliation unavailable.")
    if not recon.clusters:
        message = ("No usable estimators." if "no_usable_estimators" in recon.notes
                   else "No candidates to reconcile.")
        return _empty(title, message)

    pending = recon.is_stub and "stage2_not_evaluated" in recon.notes
    if pending:
        title += " · Stage 2 assessment pending"
    elif not recon.stage2_passed_any:
        title += " · No candidate passed Stage 2"
    figure = _layout(make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=("Cluster scores", "Estimator support"),
    ), title)
    figure.update_layout(height=640)
    by_id = {cluster.cluster_id: cluster for cluster in recon.clusters}
    link_x, link_y = [], []
    for cluster in recon.clusters:
        for link in cluster.harmonic_links:
            other_id = link["other_id"]
            if other_id > cluster.cluster_id and other_id in by_id:
                other = by_id[other_id]
                link_x.extend((cluster.period_s, other.period_s, None))
                link_y.extend((cluster.score, other.score, None))
    if link_x:
        figure.add_scatter(
            x=link_x, y=link_y, mode="lines", name="Harmonic link",
            line=dict(color="#94A3B8", dash="dot"), hoverinfo="skip", row=1, col=1,
        )

    outcomes = {
        "Shortlisted": ("star", "#475569"),
        "Passed Stage 2": ("circle-open", "#475569"),
        "Rejected by Stage 2": ("x", "#9F1239"),
        "Stage 2 pending": ("diamond-open", "#475569"),
    }
    for outcome, (symbol, color) in outcomes.items():
        clusters = [cluster for cluster in recon.clusters if outcome == (
            "Stage 2 pending" if pending else
            "Shortlisted" if cluster.cluster_id in recon.shortlist else
            "Passed Stage 2" if cluster.stage2["passed"] else "Rejected by Stage 2"
        )]
        if not clusters:
            continue
        details = []
        for cluster in clusters:
            links = ", ".join(f'#{link["other_id"]} (×{link["ratio"]:.3g})'
                              for link in cluster.harmonic_links)
            reasons = ", ".join(cluster.stage2["fail_reasons"])
            details.append([cluster.cluster_id, cluster.alias_risk, cluster.stage2["cycles"],
                            reasons or "—", links or "—"])
        figure.add_scatter(
            x=[cluster.period_s for cluster in clusters], y=[cluster.score for cluster in clusters],
            name=outcome, mode="markers", marker=dict(symbol=symbol, color=color, size=11),
            customdata=details, row=1, col=1,
            hovertemplate=("Cluster #%{customdata[0]} · %{x:.4g} s<br>Score %{y:.3g}"
                           "<br>Alias risk %{customdata[1]:.3f}<br>Cycles %{customdata[2]:.3g}"
                           "<br>Stage 2 reasons %{customdata[3]}"
                           "<br>Harmonics %{customdata[4]}<extra>%{fullData.name}</extra>"),
        )
    for method, (name, color, symbol) in _SOURCES.items():
        members = [cluster for cluster in recon.clusters if method in cluster.evidence]
        if members:
            figure.add_scatter(
                x=[cluster.members[method] for cluster in members],
                y=[cluster.evidence[method] for cluster in members],
                customdata=[cluster.cluster_id for cluster in members],
                hovertemplate="Cluster #%{customdata} · %{x:.4g} s<br>Support %{y:.3f}<extra>%{fullData.name}</extra>",
                mode="markers", name=name, marker=dict(color=color, symbol=symbol, size=8),
                row=2, col=1,
            )
    figure.update_yaxes(title_text="Score", row=1, col=1)
    figure.update_yaxes(title_text="Support", range=[-0.05, 1.05], row=2, col=1)
    figure.update_xaxes(title_text=_PERIOD, row=2, col=1)
    return figure


_QUALITY_METRICS = {
    "n_valid": ("Valid samples", ""),
    "duration_s": ("Duration", " s"),
    "missing_fraction": ("Missing samples", "%"),
    "largest_gap_s": ("Largest gap", " s"),
    "median_dt_s": ("Median cadence", " s"),
    "noise_proxy": ("Noise proxy", " (normalized brightness)"),
    "amplitude_to_noise": ("Amplitude-to-noise ratio", ""),
    "outlier_fraction": ("Flagged outliers", "%"),
}


def fig_quality(report):
    if report is None:
        return _empty("Observation quality", "Stage 1 assessment unavailable.")
    title = f"Observation quality · {report.status}"
    labels, values = [], []
    for name, value in report.metrics.items():
        label, unit = _QUALITY_METRICS.get(name, (name.replace("_", " "), ""))
        labels.append(label)
        values.append(f"{value:.1%}" if unit == "%" else f"{value:.4g}{unit}")
    if not labels:
        labels, values = ["Metrics"], ["No quality metrics reported."]
    for warning in report.warnings:
        labels.append("Warning")
        values.append(warning)
    if report.reason_codes:
        labels.append("Reason codes")
        values.append(", ".join(report.reason_codes))
    figure = go.Figure(go.Table(
        columnwidth=[1, 3], header=dict(values=["Metric or diagnostic", "Value"], align="left"),
        cells=dict(values=[labels, values], align="left", height=29),
    ))
    figure.update_layout(height=110 + 29 * len(labels))
    return _layout(figure, title)


def fig_verification(verification):
    title = "Physics verification"
    if verification is None:
        return _empty(title, "Verification unavailable or not run.")
    if not verification.per_candidate:
        return _empty(title, "No candidate verification records.")
    title += " · Complete" if verification.complete else " · Incomplete"
    figure = _layout(make_subplots(
        rows=3, cols=1, specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("Fit and held-out errors", "Decoy error differences", "Candidate outcomes"),
        row_heights=[0.36, 0.30, 0.34], vertical_spacing=0.10,
    ), title)
    figure.update_layout(height=max(720, 565 + 32 * len(verification.per_candidate)), barmode="group")
    records = verification.per_candidate
    labels = [f"#{entry.cluster_id} · {entry.period_s:g} s" for entry in records]
    for field, name, color in (
        ("fit_error", "Fit error", "#94A3B8"), ("heldout_error", "Held-out error", _NEUTRAL),
    ):
        points = [(label, getattr(entry, field)) for label, entry in zip(labels, records)
                  if entry.status == "ok" and getattr(entry, field) is not None]
        if points:
            x, y = zip(*points)
            figure.add_bar(x=x, y=y, name=name, marker_color=color, row=1, col=1)
    if not figure.data:
        figure.add_annotation(x=0.5, y=0.5, xref="x domain", yref="y domain",
                              text="No completed fits.", showarrow=False, row=1, col=1)
    decoys = [(label, period, delta) for label, entry in zip(labels, records)
              for period, delta in zip(entry.decoy_periods_s, entry.decoy_delta_e)]
    if decoys:
        x, periods, deltas = zip(*decoys)
        figure.add_scatter(
            x=x, y=deltas, customdata=periods, mode="markers", name="Decoy",
            marker=dict(color="#7E3F98", size=8), row=2, col=1,
            hovertemplate="%{x}<br>Decoy %{customdata:.4g} s<br>Δ error %{y:.3g}<extra></extra>",
        )
        figure.add_hline(y=0, line_dash="dot", row=2, col=1)
    else:
        figure.add_annotation(x=0.5, y=0.5, xref="x domain", yref="y domain",
                              text="No decoy comparisons available.", showarrow=False, row=2, col=1)
    outcomes, details = [], []
    fit_ids = {entry.cluster_id for entry in records}
    for entry in records:
        outcomes.append(entry.status if entry.status != "ok" else "Passed" if entry.passed else "Rejected")
        notes = list(entry.notes)
        for comparison in verification.harmonic_comparisons:
            display_id = (comparison.short_cluster_id if comparison.short_cluster_id in fit_ids
                          else comparison.long_cluster_id)
            if entry.cluster_id != display_id:
                continue
            notes.append(f"Harmonic pair {comparison.pair_id} · {comparison.status} · {comparison.prefers}")
            if comparison.delta_e is not None:
                notes.append(f"Δ held-out error {comparison.delta_e:.4g}")
            if comparison.reason:
                notes.append(comparison.reason)
        details.append("; ".join(notes) or "—")
    figure.add_trace(go.Table(
        header=dict(values=["Candidate", "Outcome", "Details"], align="left"),
        cells=dict(values=[labels, outcomes, details], align="left"),
        columnwidth=[1, 1, 3],
    ), row=3, col=1)
    figure.update_yaxes(title_text="Normalized error", row=1, col=1)
    figure.update_yaxes(title_text="Decoy − candidate error", row=2, col=1)
    return figure


def _evaluation_rows(eval_df, columns, title):
    if eval_df is None or eval_df.empty:
        return None, _empty(title, "No evaluation data available.")
    missing = [column for column in columns if column not in eval_df]
    if missing:
        return None, _empty(title, "Missing evaluation columns · " + ", ".join(missing))
    rows = eval_df.copy()
    for column in columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").astype(float)
    x, y, count = columns
    valid = rows[x].between(0, 1) & rows[y].between(0, 1)
    valid &= np.isfinite(rows[count]) & (rows[count] > 0)
    rows = rows.loc[valid]
    if rows.empty:
        return None, _empty(title, "No valid evaluation points with positive sample counts.")
    return rows, None


def _evaluation_traces(figure, rows, x, y, count, default_label, threshold=False):
    groups = (rows.groupby("configuration", dropna=False, sort=False, observed=True)
              if "configuration" in rows else [(default_label, rows)])
    for name, group in groups:
        ordered = group.sort_values(x, kind="stable")
        hover = "%{x:.1%}, %{y:.1%}<br>Samples %{customdata[0]:g}"
        if threshold and "threshold" in ordered:
            customdata = list(zip(ordered[count], ordered["threshold"].astype("string").fillna("—")))
            hover += "<br>Threshold %{customdata[1]}"
        else:
            customdata = [[value] for value in ordered[count]]
        label = "Unspecified configuration" if pd.isna(name) else str(name)
        source = _SOURCES.get(LOMB_SCARGLE if label == "LS" else label)
        if source is None:
            source = next((item for item in _SOURCES.values() if label == item[0]), None)
        style = {} if source is None else {
            "line": dict(color=source[1]), "marker": dict(color=source[1], symbol=source[2]),
        }
        figure.add_scatter(
            x=ordered[x], y=ordered[y], name=source[0] if source else label,
            mode="lines+markers", customdata=customdata, hovertemplate=hover + "<extra>%{fullData.name}</extra>",
            **style,
        )


def fig_risk_coverage(eval_df):
    title = "Risk–coverage"
    rows, empty = _evaluation_rows(eval_df, ("coverage", "risk", "n_selected"), title)
    if empty is not None:
        return empty
    figure = _layout(go.Figure(), title)
    _evaluation_traces(figure, rows, "coverage", "risk", "n_selected", "Risk–coverage", threshold=True)
    figure.update_xaxes(title_text="Coverage", range=[0, 1], tickformat=".0%")
    figure.update_yaxes(title_text="Selective error rate", range=[0, 1], tickformat=".0%")
    return figure


def fig_calibration(eval_df):
    title = "Reliability diagram"
    rows, empty = _evaluation_rows(eval_df, ("mean_confidence", "observed_accuracy", "n"), title)
    if empty is not None:
        return empty
    figure = _layout(go.Figure(), title)
    figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal calibration",
                       line=dict(color="#94A3B8", dash="dot"), hoverinfo="skip")
    _evaluation_traces(figure, rows, "mean_confidence", "observed_accuracy", "n", "Observed calibration")
    figure.update_xaxes(title_text="Mean confidence", range=[0, 1], tickformat=".0%")
    figure.update_yaxes(title_text="Observed accuracy", range=[0, 1], tickformat=".0%")
    return figure
