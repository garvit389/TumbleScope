"""Local Fast analysis and curated precomputed demonstrations."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import config_hash, load_config
from core.errors import IngestionError
from core.schemas import CNN, LOMB_SCARGLE, PDM, Observation, stage2_pending
from ingestion import load_observation
from pipeline import cache
from preprocessing.prepare import prepare
from preprocessing.quality import assess_stage1


_PERIOD = "Apparent (synodic) period (s)"
_NAMES = {CNN: "CNN", LOMB_SCARGLE: "Lomb–Scargle (LS)", PDM: "PDM"}
_STAGES = {
    "prepare": "Preparation", "stage1": "Stage 1 quality", "run_config": "Configuration recording",
    "window": "Sampling window", "period_bounds": "Apparent (synodic) period bounds",
    "reconciliation": "Reconciliation", "verification": "Verification", "verdict": "Verdict",
    **_NAMES,
}
_ASSUMPTIONS = {
    "Unqualified time column assumed to contain seconds.",
    "Brightness convention unknown; increasing values assumed brighter.",
}
_PROGRESS = {
    "Starting analysis": "Preparing observation",
    "Preparation and quality complete": "Stage 1 quality assessed",
    "Period estimation complete": "Estimator results available",
    "Starting verification": "Checking shortlisted candidates",
    "Verification finished": "Verification finished",
    "Analysis complete": "Analysis complete",
    "Loading cached analysis": "Loading saved session result",
    cache._CACHE_HIT_MESSAGE: "Session result loaded",
}


def _basename(name):
    name = PureWindowsPath(name).name
    if name in ("", ".", "..") or ":" in name or Path(name).suffix.lower() not in (".txt", ".csv"):
        raise IngestionError("Invalid upload filename")
    return name


def _load_upload_bytes(name: str, raw: bytes) -> Observation:
    with TemporaryDirectory() as folder:
        path = Path(folder) / _basename(name)
        path.write_bytes(raw)
        return load_observation(str(path))


def load_upload(uploaded):
    return _load_upload_bytes(uploaded.name, uploaded.getvalue())


def _quality_view(report):
    if report is None:
        st.caption("Stage 1 quality is unavailable.")
        return
    st.write(f"Observation quality: {report.status}")
    public = replace(report, warnings=[
        cache._public_quality_warning(warning)
        for warning in report.warnings
    ])
    _plot("fig_quality", public, fallback=lambda g: g.add_table(
        header={"values": ["Quality metric", "Value"]},
        cells={"values": [list(public.metrics), list(public.metrics.values())]},
    ))
    for warning in public.warnings:
        st.caption(warning)
    if public.reason_codes:
        st.caption(", ".join(public.reason_codes))


def _plot(name, *args, fallback):
    try:
        from app import plots

        figure = getattr(plots, name)(*args)
        figure.for_each_xaxis(lambda axis: axis.update(title_text=_PERIOD)
                              if axis.title.text == "Apparent period (s)" else None)
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        figure = fallback(go.Figure())
    if figure is not None and (figure.data or figure.layout.annotations):
        st.plotly_chart(figure, key=name)


def _fold(figure, prep, period=None):
    if prep is None:
        return figure
    times = prep.t_s[prep.valid_mask]
    x = times if period is None else (times % period) / period
    return figure.add_scatter(x=x, y=prep.y_norm[prep.valid_mask], mode="markers").update_layout(
        xaxis_title="Elapsed time (s)" if period is None else "Phase", yaxis_title="Normalized brightness"
    )


def _spectra(figure, estimators):
    from plotly.subplots import make_subplots

    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=list(_NAMES.values()))
    for row, method in enumerate(_NAMES, 1):
        estimate = estimators[method]
        spectrum = estimate.meta.get("periodogram")
        if method == CNN:
            figure.add_scatter(x=[c.period_s for c in estimate.candidates],
                               y=[c.support for c in estimate.candidates],
                               mode="markers", name="CNN candidate support", row=row, col=1)
        elif spectrum:
            points = [(1 / f, p) for f, p in zip(spectrum["freq_hz"], spectrum["power"]) if f > 0]
            periods, powers = zip(*sorted(points)) if points else ([], [])
            figure.add_scatter(x=periods, y=powers, name=_NAMES[method], row=row, col=1)
    return figure.update_xaxes(title_text=_PERIOD, row=3, col=1)


def _candidates(result):
    st.subheader("Estimator evidence")
    choices = {}
    for method, estimate in result.estimators.items():
        detail = f" · {estimate.error}" if estimate.error else ""
        st.caption(f"{_NAMES[method]} · {estimate.status}{detail}")
        if estimate.candidates:
            rows = []
            for rank, candidate in enumerate(estimate.candidates, 1):
                choices[f"{_NAMES[method]} candidate {rank} · {candidate.period_s:g} s"] = candidate.period_s
                row = {"Estimator": method, _PERIOD: candidate.period_s, "Within-method support": candidate.support}
                if method == CNN:
                    row["Peak probability"] = candidate.raw_score
                rows.append(row)
            st.table(rows)
        elif estimate.status == "ok":
            st.caption("No significant candidate.")
        if method == CNN and estimate.status == "ok":
            st.caption(f'No Reliable Period probability {estimate.meta["p_no_reliable"]:.3f} · '
                       f'{"Abstained" if estimate.meta["abstained"] else "Candidates available"}')
    st.caption("The full CNN distribution is unavailable in this result format.")
    _plot("fig_periodograms", result.estimators, fallback=lambda g: _spectra(g, result.estimators))

    recon = result.reconciliation
    if recon is not None and recon.clusters:
        st.subheader("Reconciled candidates")
        for cluster in recon.clusters:
            best = result.verdict is not None and cluster.cluster_id == result.verdict.best_cluster_id
            role = "Supported" if best else "Candidate for inspection"
            choices[f"{role} {cluster.cluster_id} · {cluster.period_s:g} s"] = cluster.period_s
        st.table([{"ID": c.cluster_id, _PERIOD: c.period_s, "Score": c.score, "Alias risk": c.alias_risk,
                   "Sources": ", ".join(_NAMES[method] for method in c.members),
                   "Stage 2": ("Pending" if stage2_pending(recon, result.estimators, result.verification)
                               else "Passed" if c.stage2["passed"] else ", ".join(c.stage2["fail_reasons"])),
                   "Shortlisted": c.cluster_id in recon.shortlist,
                   "Harmonic partners": ", ".join(str(link["other_id"]) for link in c.harmonic_links)}
                  for c in recon.clusters])
        _plot("fig_candidates", recon, fallback=lambda g: g.add_scatter(
            x=[c.period_s for c in recon.clusters], y=[c.score for c in recon.clusters], mode="markers",
        ).update_layout(xaxis_title=_PERIOD, yaxis_title="Reconciliation score"))
    if choices and result.prepared is not None:
        selected = st.selectbox("Apparent (synodic) period for phase folding", list(choices), index=None,
                                key=f"phase_fold_{result.obs_id}_{result.config_hash}")
        if selected is not None:
            _plot("fig_phase_fold", result.prepared, choices[selected],
                  fallback=lambda g: _fold(g, result.prepared, choices[selected]))


def _verification(result):
    report = result.verification
    if report is None:
        return
    st.subheader("Physics verification")
    st.caption("Complete" if report.complete else "Incomplete")
    st.table([{"ID": row.cluster_id, _PERIOD: row.period_s, "Status": row.status,
               "Held-out error": row.heldout_error, "Fit error": row.fit_error,
               "Decoy separation": min(row.decoy_delta_e, default=None), "Passed": row.passed}
              for row in report.per_candidate])
    if report.harmonic_comparisons:
        st.subheader("Harmonic comparisons")
        st.table([{"Pair": row.pair_id, f"Short {_PERIOD.lower()}": row.short_period_s,
                   f"Long {_PERIOD.lower()}": row.long_period_s, "Status": row.status,
                   "Prefers": row.prefers, "Δ held-out error": row.delta_e, "Reason": row.reason}
                  for row in report.harmonic_comparisons])
    _plot("fig_verification", report, fallback=lambda g: g)


def _render(result, snapshot, show_preview=False):
    result = cache._public_result(result)
    if show_preview:
        st.text(result.obs_id)
        _plot("fig_lightcurve", result.prepared, fallback=lambda g: _fold(g, result.prepared))
    _quality_view(result.quality)
    stages = [result.prepared, result.quality, *result.estimators.values(),
              result.reconciliation, result.verification, result.verdict]
    if any(getattr(stage, "is_stub", False) for stage in stages):
        st.warning("Stub result · some analysis stages are placeholders.")
    if result.estimators[CNN].meta.get("experimental"):
        st.warning("Experimental CNN · interpret alongside classical estimators.")
    for stage in result.stage_errors:
        st.caption(f"{_STAGES.get(stage, 'Analysis')}: the stage could not finish.")
    verdict = result.verdict
    if verdict is None:
        if stage2_pending(result.reconciliation, result.estimators, result.verification):
            st.subheader("Stage 2 assessment pending")
            st.write("Stage 2 reconciliation and shortlisting have not yet been evaluated.")
        else:
            st.subheader("Technical execution failure")
            st.write("The analysis could not produce a verdict. Review the retained results below.")
    else:
        st.subheader(verdict.label)
        if verdict.best_period_s is not None:
            st.write(f"{_PERIOD}: {verdict.best_period_s:g}")
        if verdict.reason_codes:
            st.caption("Reason codes · " + ", ".join(verdict.reason_codes))
        if verdict.alternatives:
            st.table([{"Alternative": row["cluster_id"], _PERIOD: row["period_s"], "Evidence": row["why"]}
                      for row in verdict.alternatives])
    _candidates(result)
    _verification(result)
    with st.expander("Reproducibility"):
        st.json({
            "mode": result.mode, "schema_version": result.schema_version, "config_hash": result.config_hash,
            "input_sha256": snapshot["input_sha256"], "revision": snapshot["commit"],
            "model_artifacts": snapshot["artifacts"],
        })
    st.download_button("Download analysis JSON", cache._encoded(result.to_dict()),
                       file_name="tumblescope-analysis.json", mime="application/json")


def _upload(cfg):
    uploaded = st.file_uploader("Upload an SDLCD, CSV or TXT lightcurve", type=["txt", "csv"])
    if uploaded is None:
        st.session_state.pop("displayed_analysis", None)
        return
    try:
        name, raw = _basename(uploaded.name), uploaded.getvalue()
        observation = _load_upload_bytes(name, raw)
    except (IngestionError, OSError, ValueError):
        st.session_state.pop("displayed_analysis", None)
        st.error("The lightcurve could not be read. Check its time and brightness columns.")
        return
    st.text(observation.obs_id)
    st.caption(f"Elapsed time in seconds · brightness kind {observation.brightness_kind}")
    for assumption in observation.meta.get("assumptions", []):
        if assumption in _ASSUMPTIONS:
            st.caption(assumption)
    st.line_chart(pd.DataFrame({"Time (s)": observation.t_s, "Brightness": observation.y}),
                  x="Time (s)", y="Brightness")

    key = (sha256(raw).hexdigest(), name, config_hash(cfg))
    saved = st.session_state.get("displayed_analysis")
    if saved is None or saved[0] != key:
        saved = None
        st.session_state.pop("displayed_analysis", None)
    preview_quality = st.container()
    run = st.button("Run Fast analysis")
    if saved is None and not run:
        try:
            with preview_quality:
                _quality_view(assess_stage1(prepare(observation, cfg.preprocess), cfg.preprocess))
        except Exception:
            st.caption("The Stage 1 preview is unavailable.")
    if run:
        bar = st.progress(0.0, text="Preparing observation")
        reused = False

        def progress(fraction, message):
            nonlocal reused
            reused = message == cache._CACHE_HIT_MESSAGE
            bar.progress(fraction, text=_PROGRESS.get(message, "Checking shortlisted candidates"))

        snapshot = cache._input_snapshot(cfg, raw)
        entry = st.session_state.setdefault("upload_analysis", {})
        result = cache._analyze_upload(raw, name, observation, cfg, entry, progress)
        saved = (key, result, snapshot, reused)
        st.session_state["displayed_analysis"] = saved
    if saved is not None:
        if saved[3]:
            st.caption("Reused cached analysis. Stage timings are from the original run.")
        _render(saved[1], saved[2])


def _demo(cfg):
    entries = {}
    for root, kind in ((cache.LOCAL_DEMOS, "sdlcd"), (cache.PORTABLE_DEMOS, "synthetic")):
        if not (root / "index.json").is_file():
            continue
        try:
            rows = cache._read_demo_index(root, cfg, kind)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            st.warning("A demo cache is stale, incomplete or unreadable.")
            continue
        label = "Local SDLCD" if kind == "sdlcd" else "Synthetic"
        entries.update({f"{label} · {name}": value for name, value in rows.items()})
    if not entries:
        st.info("No precomputed demo is available. Upload a lightcurve to run an analysis.")
        return
    choice = st.selectbox("Precomputed observation", list(entries))
    st.info("Precomputed · " + choice.split(" · ")[0])
    _render(*entries[choice], show_preview=True)


def main():
    st.title("TumbleScope")
    st.info("Apparent (synodic) period analysis uses provisional decision criteria. A physics fit "
            "shows compatibility with a simplified model. The object's shape and rotation state remain unknown.")
    mode = st.radio("Analysis mode", ["Upload", "Demo"], horizontal=True)
    try:
        cfg = load_config()
        if mode == "Upload":
            _upload(cfg)
        else:
            st.session_state.pop("displayed_analysis", None)
            _demo(cfg)
    except Exception:
        st.error("The analysis screen could not finish. Try another observation or restart the session.")


if __name__ == "__main__":
    main()
