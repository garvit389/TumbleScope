"""Local upload and bundled synthetic demonstration."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from core.config import load_config
from core.errors import IngestionError
from ingestion import load_observation
from pipeline.controller import analyze


def load_upload(uploaded):
    with TemporaryDirectory() as folder:
        path = Path(folder) / Path(uploaded.name).name
        path.write_bytes(uploaded.getvalue())
        return load_observation(str(path), "sdlcd")


def main():
    st.title("TumbleScope")
    st.caption("Apparent (synodic) period analysis. A physics fit only shows compatibility "
               "with a simplified model.")
    sample = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_sine.csv"
    uploaded = st.file_uploader("Upload a local SDLCD observation", type=["txt"])
    try:
        observation = load_upload(uploaded) if uploaded is not None else load_observation(str(sample))
        result = analyze(observation, load_config())
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
