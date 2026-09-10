"""End-to-end evaluation: sweep generator settings, compute utility/privacy tradeoff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data import PROCESSED_DIR, ROOT, TARGET, load_adult, train_holdout_split, write_sample
from .generators import HAS_SDV, SynthGenerator
from .privacy import privacy_summary
from .utility import tstr_metrics

REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"


def run_single(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    method: str,
    noise_scale: float,
    n_synth: int,
    seed: int = 42,
    epochs: int = 30,
) -> Dict[str, Any]:
    gen = SynthGenerator(method=method, seed=seed, epochs=epochs, noise_scale=noise_scale)
    gen.fit(real_train)
    synth = gen.sample(n_synth)
    if TARGET in synth.columns:
        synth = synth.dropna(subset=[TARGET]).reset_index(drop=True)

    util = tstr_metrics(real_train, synth, real_holdout, seed=seed)
    priv = privacy_summary(synth, real_train, real_holdout, seed=seed)
    dcr_dists = priv.pop("_dcr_distances", None)

    row = {
        "method": method,
        "backend": gen.backend,
        "noise_scale": noise_scale,
        "n_synth": len(synth),
        "utility_f1": util["synth_tstr"]["f1"],
        "utility_accuracy": util["synth_tstr"]["accuracy"],
        "utility_roc_auc": util["synth_tstr"].get("roc_auc"),
        "real_f1": util["real_holdout"]["f1"],
        "real_accuracy": util["real_holdout"]["accuracy"],
        "real_roc_auc": util["real_holdout"].get("roc_auc"),
        "utility_ratio_f1": util["utility_ratio_f1"],
        "utility_ratio_auc": util.get("utility_ratio_auc"),
        "dcr_median": priv["dcr_median"],
        "dcr_mean": priv["dcr_mean"],
        "dcr_p5": priv["dcr_p5"],
        "near_duplicate_rate": priv["near_duplicate_rate"],
        "mia_auc": priv["mia_auc"],
        "privacy_score": priv["privacy_score"],
        "mean_dist_members": priv["mean_dist_members"],
        "mean_dist_nonmembers": priv["mean_dist_nonmembers"],
    }
    return {"row": row, "synth": synth, "dcr_distances": dcr_dists, "util": util, "priv": priv}


def run_tradeoff_sweep(
    max_rows: int = 2500,
    n_synth: int | None = None,
    seed: int = 42,
    include_ctgan: bool = False,
) -> Dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    write_sample(n=200, seed=seed)
    df = load_adult(use_sample=False, max_rows=max_rows)
    real_train, real_holdout = train_holdout_split(df, test_size=0.25, seed=seed)
    real_train.to_csv(PROCESSED_DIR / "real_train.csv", index=False)
    real_holdout.to_csv(PROCESSED_DIR / "real_holdout.csv", index=False)

    if n_synth is None:
        n_synth = len(real_train)

    configs: List[Dict[str, Any]] = []
    for ns in [1.0, 1.25, 1.5, 2.0]:
        configs.append({"method": "gaussian_copula", "noise_scale": ns, "epochs": 30})
    for ns in [0.75, 1.0, 1.5]:
        configs.append({"method": "fallback", "noise_scale": ns, "epochs": 30})
    if include_ctgan and HAS_SDV:
        configs.append({"method": "ctgan", "noise_scale": 1.0, "epochs": 30})

    rows = []
    best_synth = None
    best_key = None
    dcr_for_plot = None

    for cfg in configs:
        print(f">>> Running {cfg['method']} noise_scale={cfg['noise_scale']} ...")
        result = run_single(
            real_train, real_holdout,
            method=cfg["method"], noise_scale=cfg["noise_scale"],
            n_synth=n_synth, seed=seed, epochs=cfg["epochs"],
        )
        rows.append(result["row"])
        print(
            f"    F1={result['row']['utility_f1']:.3f}  "
            f"DCR_med={result['row']['dcr_median']:.3f}  "
            f"MIA_AUC={result['row']['mia_auc']:.3f}  "
            f"backend={result['row']['backend']}"
        )
        key = (cfg["method"], cfg["noise_scale"])
        if key == ("gaussian_copula", 1.0) or (best_synth is None and cfg["method"] == "fallback"):
            best_synth = result["synth"]
            best_key = key
            dcr_for_plot = result["dcr_distances"]

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(REPORTS / "metrics.csv", index=False)

    summary = {
        "dataset": "Adult/Census Income (OpenML adult v2 or UCI)",
        "n_total_used": int(len(df)),
        "n_train": int(len(real_train)),
        "n_holdout": int(len(real_holdout)),
        "n_synth": int(n_synth),
        "has_sdv": HAS_SDV,
        "primary_config": {"method": best_key[0], "noise_scale": best_key[1]} if best_key else None,
        "runs": rows,
        "baseline_real": {
            "f1": rows[0]["real_f1"] if rows else None,
            "accuracy": rows[0]["real_accuracy"] if rows else None,
            "roc_auc": rows[0]["real_roc_auc"] if rows else None,
        },
    }
    with open(REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_synth is not None:
        best_synth.to_csv(OUTPUTS / "synthetic_primary.csv", index=False)
        best_synth.head(50).to_csv(OUTPUTS / "synthetic_preview.csv", index=False)

    if dcr_for_plot is not None:
        np.save(OUTPUTS / "dcr_distances.npy", dcr_for_plot)

    return {
        "metrics_df": metrics_df,
        "summary": summary,
        "real_train": real_train,
        "real_holdout": real_holdout,
        "synth": best_synth,
        "dcr_distances": dcr_for_plot,
    }
