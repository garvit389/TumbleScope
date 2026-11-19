"""Phase 0 Streamlit demonstration."""

from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import load_config
from core.errors import IngestionError
from ingestion import load_observation
from pipeline.controller import analyze


def main():
    st.title("TumbleScope")
    st.warning("Phase 0 skeleton · Synthetic stub result")
    st.caption("Apparent (synodic) period analysis. A physics fit only shows compatibility "
               "with a simplified model.")
    sample = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_sine.csv"
    try:
        observation = load_observation(str(sample))
        result = analyze(observation, load_config())
    except (IngestionError, ValueError) as exc:
        st.error(str(exc))
        return
    st.line_chart(pd.DataFrame({"Time (s)": observation.t_s, "Synthetic brightness": observation.y}),
                  x="Time (s)", y="Synthetic brightness")
    st.subheader(result.verdict.label)
    st.write(result.verdict.text)
    st.table([
        {"Estimator": method, "Status": item.status, "Detail": item.error or ""}
        for method, item in result.estimators.items()
    ])


if __name__ == "__main__":
    main()
