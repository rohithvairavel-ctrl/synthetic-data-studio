"""Streamlit dashboard: generate synth, compare distributions, show metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import NUMERIC, TARGET, load_adult, train_holdout_split
from src.evaluate import run_single
from src.generators import HAS_SDV, SynthGenerator

st.set_page_config(page_title="Synthetic Data Studio", layout="wide", page_icon="🧬")

st.title("🧬 Synthetic Data Studio")
st.caption(
    "Generate synthetic tabular data, measure ML utility (TSTR), and quantify "
    "privacy risk proxies (DCR + membership inference). "
    "**Synthetic ≠ anonymous** — see README caveats."
)

REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"


@st.cache_data(show_spinner=False)
def load_data(max_rows: int):
    df = load_adult(use_sample=False, max_rows=max_rows)
    train, hold = train_holdout_split(df, test_size=0.25, seed=42)
    return train, hold


def load_cached_metrics():
    csv = REPORTS / "metrics.csv"
    js = REPORTS / "metrics.json"
    metrics_df = pd.read_csv(csv) if csv.exists() else None
    summary = json.loads(js.read_text()) if js.exists() else None
    return metrics_df, summary


with st.sidebar:
    st.header("Controls")
    max_rows = st.slider("Dataset subsample", 500, 5000, 2000, 500)
    method = st.selectbox(
        "Generator",
        ["gaussian_copula", "fallback"] + (["ctgan"] if HAS_SDV else []),
        index=0,
    )
    noise_scale = st.slider("Noise / privacy knob", 0.75, 2.5, 1.0, 0.25)
    n_synth = st.slider("Synthetic rows", 200, 3000, 1000, 100)
    run_btn = st.button("Generate & evaluate", type="primary")
    st.markdown(f"SDV available: **{HAS_SDV}**")

tab_live, tab_cached, tab_about = st.tabs(["Live run", "Cached experiment", "About / caveats"])

with tab_live:
    train, hold = load_data(max_rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Train rows", len(train))
    c2.metric("Holdout rows", len(hold))
    c3.metric("Features", train.shape[1] - 1)

    if run_btn:
        with st.spinner(f"Fitting {method} (noise_scale={noise_scale})…"):
            result = run_single(
                train, hold, method=method, noise_scale=noise_scale, n_synth=n_synth, seed=42, epochs=20
            )
        row = result["row"]
        synth = result["synth"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("TSTR F1", f"{row['utility_f1']:.3f}", delta=f"vs real {row['real_f1']:.3f}")
        m2.metric("TSTR Accuracy", f"{row['utility_accuracy']:.3f}")
        m3.metric("Median DCR ↑", f"{row['dcr_median']:.3f}")
        m4.metric("MIA AUC (~0.5 better)", f"{row['mia_auc']:.3f}")

        st.subheader("Real vs synthetic distributions")
        cols = [c for c in ["age", "hours-per-week", "education-num", "capital-gain"] if c in train.columns]
        fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.hist(train[col].dropna(), bins=20, alpha=0.55, density=True, label="real")
            ax.hist(synth[col].dropna(), bins=20, alpha=0.55, density=True, label="synth")
            ax.set_title(col)
            ax.legend(fontsize=7)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Target balance")
        bal = pd.DataFrame(
            {
                "real": train[TARGET].astype(str).value_counts(normalize=True),
                "synth": synth[TARGET].astype(str).value_counts(normalize=True),
            }
        ).fillna(0)
        st.bar_chart(bal)

        st.subheader("DCR histogram")
        dists = result["dcr_distances"]
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.hist(dists, bins=35, color="#4C72B0", edgecolor="white")
        ax2.axvline(np.median(dists), color="crimson", ls="--", label="median")
        ax2.legend()
        ax2.set_xlabel("Distance to closest real record")
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("Synthetic preview")
        st.dataframe(synth.head(25), use_container_width=True)
        st.download_button(
            "Download synthetic CSV",
            synth.to_csv(index=False),
            file_name="synthetic.csv",
            mime="text/csv",
        )
    else:
        st.info("Configure settings in the sidebar and click **Generate & evaluate**.")

with tab_cached:
    metrics_df, summary = load_cached_metrics()
    if metrics_df is None:
        st.warning("No cached metrics yet. Run `python scripts/run_experiment.py` first.")
    else:
        st.subheader("Experiment metrics (from last batch run)")
        st.dataframe(metrics_df, use_container_width=True)
        if summary:
            st.json({k: v for k, v in summary.items() if k != "runs"})

        fig_dir = ROOT / "figures"
        for name in [
            "utility_privacy_tradeoff.svg",
            "distribution_compare.svg",
            "dcr_distribution.svg",
            "headline_metrics.svg",
            "target_balance.svg",
        ]:
            path = fig_dir / name
            if path.exists():
                st.markdown(f"**{name}**")
                st.image(str(path))

        preview = OUTPUTS / "synthetic_preview.csv"
        if preview.exists():
            st.subheader("Cached synthetic preview")
            st.dataframe(pd.read_csv(preview), use_container_width=True)

with tab_about:
    st.markdown(
        """
### What this measures
- **Utility (TSTR):** train a classifier on synthetic data, score accuracy/F1/AUC on a **real** holdout.
- **DCR:** distance from each synthetic row to the nearest real training row (standardized Euclidean).
  Higher median ⇒ less copying (proxy only).
- **Membership inference AUC:** can an attacker tell train members from holdout non-members by
  proximity to the synthetic set? AUC≈0.5 is ideal.

### Caveats (read before shipping synth to production)
1. **Synthetic ≠ anonymous.** These metrics are *proxies*, not formal privacy guarantees
   (not differential privacy).
2. Near-duplicates and rare categories can still leak.
3. Utility/privacy tradeoffs are dataset- and model-specific.
4. Do not release synthetic data derived from regulated PII without a proper privacy review.

### Backends
- Preferred: **SDV GaussianCopula** (and optional CTGAN).
- Fallback: IterativeImputer + multivariate normal + histogram sampler (`src/generators.py`).
"""
    )
