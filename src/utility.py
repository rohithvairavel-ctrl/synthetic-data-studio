"""Train-on-Synthetic-Test-on-Real (TSTR) utility evaluation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLS, TARGET, encode_for_ml


def _clf(seed: int = 42) -> Pipeline:
    """Logistic regression is a standard TSTR probe — stable thresholds."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _clf_rf(seed: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=10,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def prepare_xy(
    train_df: pd.DataFrame, holdout_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_e, hold_e, _, _ = encode_for_ml(train_df, holdout_df)
    cols = [c for c in FEATURE_COLS if c in train_e.columns]
    X_tr = train_e[cols].values.astype(float)
    y_tr = train_e[TARGET].values.astype(int)
    X_te = hold_e[cols].values.astype(float)
    y_te = hold_e[TARGET].values.astype(int)
    return X_tr, y_tr, X_te, y_te, cols


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick threshold maximizing F1 on the training predictions (not holdout)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.15, 0.85, 29):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _scores(model, X, y, threshold: float | None = None) -> Dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    if threshold is None:
        pred = model.predict(X)
    else:
        pred = (proba >= threshold).astype(int)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average="binary", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, proba))
    except ValueError:
        out["roc_auc"] = float("nan")
    if threshold is not None:
        out["threshold"] = float(threshold)
    return out


def tstr_metrics(
    real_train: pd.DataFrame,
    synth: pd.DataFrame,
    real_holdout: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare a classifier trained on real vs synthetic data, both evaluated
    on the same real holdout set (TSTR). Thresholds are calibrated on the
    respective training sets to avoid majority-class collapse on imbalanced labels.
    """
    cols = [c for c in real_train.columns if c in synth.columns]
    synth = synth[cols].copy()
    real_train = real_train[cols].copy()
    hold_cols = [c for c in cols if c in real_holdout.columns]
    real_holdout = real_holdout[hold_cols].copy()

    X_r, y_r, X_h, y_h, feat = prepare_xy(real_train, real_holdout)
    X_s, y_s, X_h_s, y_h_s, _ = prepare_xy(synth, real_holdout)

    model_real = _clf(seed).fit(X_r, y_r)
    model_syn = _clf(seed).fit(X_s, y_s)

    t_real = _best_threshold(y_r, model_real.predict_proba(X_r)[:, 1])
    t_syn = _best_threshold(y_s, model_syn.predict_proba(X_s)[:, 1])

    real_scores = _scores(model_real, X_h, y_h, threshold=t_real)
    syn_scores = _scores(model_syn, X_h_s, y_h_s, threshold=t_syn)

    utility_ratio = syn_scores["f1"] / real_scores["f1"] if real_scores["f1"] > 0 else 0.0
    utility_ratio_auc = (
        syn_scores["roc_auc"] / real_scores["roc_auc"]
        if real_scores.get("roc_auc") and real_scores["roc_auc"] > 0
        else 0.0
    )
    return {
        "real_holdout": real_scores,
        "synth_tstr": syn_scores,
        "utility_ratio_f1": float(utility_ratio),
        "utility_ratio_auc": float(utility_ratio_auc),
        "n_real_train": int(len(real_train)),
        "n_synth": int(len(synth)),
        "n_holdout": int(len(real_holdout)),
        "features": feat,
    }
