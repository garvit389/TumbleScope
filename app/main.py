"""Local upload and bundled synthetic demonstration."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from core.config import load_config
from core.errors import IngestionError
from core.schemas import Observation, stage2_pending
from ingestion import load_observation
from pipeline.cache import _CACHE_HIT_MESSAGE, _analyze_upload
from pipeline.controller import analyze


def _load_upload_bytes(name: str, raw: bytes) -> Observation:
    name = Path(name).name
    if name in ("", ".", ".."):
        raise IngestionError("Invalid upload filename")
    with TemporaryDirectory() as folder:
        path = Path(folder) / name
        path.write_bytes(raw)
        return load_observation(str(path), "sdlcd")


def load_upload(uploaded):
    return _load_upload_bytes(uploaded.name, uploaded.getvalue())


def main():
    st.title("TumbleScope")
    st.caption("Apparent (synodic) period analysis. A physics fit only shows compatibility "
               "with a simplified model.")
    sample = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_sine.csv"
    uploaded = st.file_uploader("Upload a local SDLCD observation", type=["txt"])
    try:
        cfg = load_config()
        if uploaded is None:
            observation = load_observation(str(sample))
            result = analyze(observation, cfg)
        else:
            raw = uploaded.getvalue()
            observation = _load_upload_bytes(uploaded.name, raw)
            reused = False

            def progress(fraction, message):
                nonlocal reused
                reused = message == _CACHE_HIT_MESSAGE

            entry = st.session_state.setdefault("upload_analysis", {})
            result = _analyze_upload(raw, uploaded.name, observation, cfg, entry, progress)
            if reused:
                st.caption("Reused cached analysis. Stage timings are from the original run.")
    except (IngestionError, ValueError) as exc:
        st.error(str(exc))
        return
    stages = [result.prepared, result.quality, *result.estimators.values(),
              result.reconciliation, result.verification, result.verdict]
    if any(getattr(stage, "is_stub", False) for stage in stages):
        st.warning("Synthetic stub result · bundled demo" if uploaded is None else
                   "Some analysis stages are placeholders.")
    st.write(observation.obs_id)
    st.line_chart(pd.DataFrame({"Time (s)": observation.t_s, "Brightness": observation.y}),
                  x="Time (s)", y="Brightness")
    if result.quality is not None:
        st.write(f"Observation quality: {result.quality.status}")
        for warning in result.quality.warnings:
            st.caption(warning)
    for stage, diagnostic in result.stage_errors.items():
        st.caption(f"{stage}: {diagnostic}")
    if result.verdict is None and not stage2_pending(
            result.reconciliation, result.estimators, result.verification):
        st.subheader("Technical execution failure")
        st.write("A required stage failed. Retained results are shown below; no final scientific verdict is available.")
    elif result.verdict is None:
        st.subheader("Stage 2 assessment pending")
        st.write("Estimator candidates are available. Stage 2 reconciliation and shortlisting "
                 "have not yet been evaluated. A final verdict is pending.")
    else:
        st.subheader(result.verdict.label)
        st.write(result.verdict.text)
    st.table([
        {"Estimator": method, "Status": item.status, "Detail": item.error or ""}
        for method, item in result.estimators.items()
    ])
    for method, item in result.estimators.items():
        if item.candidates:
            st.table([{"Estimator": method, "Apparent period (s)": c.period_s,
                       "Within-method support": c.support} for c in item.candidates])
        elif item.status == "ok":
            st.caption(f"{method}: no significant candidate")


if __name__ == "__main__":
    main()
