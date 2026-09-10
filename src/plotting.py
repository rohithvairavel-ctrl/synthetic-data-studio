"""SVG figure generation for utility/privacy tradeoff and distribution comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import NUMERIC, ROOT, TARGET

FIGURES = ROOT / "figures"


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tradeoff(metrics_df: pd.DataFrame) -> Path:
    """Utility (F1) vs privacy (DCR median); marker size ~ MIA AUC."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.set_style("whitegrid")
    for method, g in metrics_df.groupby("method"):
        sizes = 80 + 400 * (g["mia_auc"] - 0.45).clip(lower=0)
        ax.scatter(
            g["dcr_median"],
            g["utility_f1"],
            s=sizes,
            label=method,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
        )
        for _, r in g.iterrows():
            ax.annotate(
                f"ns={r['noise_scale']}",
                (r["dcr_median"], r["utility_f1"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
            )
    if "real_f1" in metrics_df.columns and len(metrics_df):
        ax.axhline(metrics_df["real_f1"].iloc[0], color="gray", ls="--", lw=1.2, label="real F1")
    ax.set_xlabel("Privacy proxy (median DCR)")
    ax.set_ylabel("Utility (TSTR F1 on real holdout)")
    ax.set_title("Utility-Privacy Tradeoff Across Generator Settings")
    ax.legend(frameon=True)
    return _save(fig, "utility_privacy_tradeoff.svg")


def plot_dcr_hist(distances: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(distances, bins=40, color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.axvline(np.median(distances), color="#C44E52", ls="--", label=f"median={np.median(distances):.3f}")
    ax.set_xlabel("Distance to closest real record (standardized)")
    ax.set_ylabel("Count (synthetic rows)")
    ax.set_title("Distance-to-Closest-Record (DCR) Distribution")
    ax.legend()
    return _save(fig, "dcr_distribution.svg")


def plot_distributions(
    real: pd.DataFrame, synth: pd.DataFrame, cols: Optional[list[str]] = None
) -> Path:
    cols = cols or [c for c in ["age", "hours-per-week", "education-num"] if c in real.columns]
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.hist(real[col].dropna(), bins=25, alpha=0.55, label="real", density=True, color="#4C72B0")
        ax.hist(synth[col].dropna(), bins=25, alpha=0.55, label="synth", density=True, color="#55A868")
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Real vs Synthetic Marginal Distributions", y=1.02)
    fig.tight_layout()
    return _save(fig, "distribution_compare.svg")


def plot_target_balance(real: pd.DataFrame, synth: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, df, title in zip(axes, [real, synth], ["Real train", "Synthetic"]):
        vc = df[TARGET].astype(str).value_counts(normalize=True)
        ax.bar(vc.index.astype(str), vc.values, color=["#4C72B0", "#DD8452"][: len(vc)])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("Share")
    fig.suptitle("Target Class Balance", y=1.02)
    fig.tight_layout()
    return _save(fig, "target_balance.svg")


def plot_metrics_bars(metrics_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    picks = []
    for method, g in metrics_df.groupby("method"):
        picks.append(g.iloc[(g["noise_scale"] - 1.0).abs().argmin()])
    sub = pd.DataFrame(picks)
    metrics = [
        ("utility_f1", "TSTR F1"),
        ("dcr_median", "Median DCR"),
        ("mia_auc", "MIA AUC"),
    ]
    colors = sns.color_palette("muted", n_colors=len(sub))
    for ax, (col, title) in zip(axes, metrics):
        ax.bar(sub["method"], sub[col], color=colors, edgecolor="k", linewidth=0.4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        if col == "mia_auc":
            ax.axhline(0.5, color="gray", ls="--", lw=1)
    fig.suptitle("Headline Metrics by Generator (noise_scale~1)", y=1.03)
    fig.tight_layout()
    return _save(fig, "headline_metrics.svg")


def make_all_figures(
    metrics_df: pd.DataFrame,
    real_train: pd.DataFrame,
    synth: pd.DataFrame,
    dcr_distances: Optional[np.ndarray] = None,
) -> list[Path]:
    paths = [
        plot_tradeoff(metrics_df),
        plot_distributions(real_train, synth),
        plot_target_balance(real_train, synth),
        plot_metrics_bars(metrics_df),
    ]
    if dcr_distances is not None:
        paths.append(plot_dcr_hist(dcr_distances))
    return paths
