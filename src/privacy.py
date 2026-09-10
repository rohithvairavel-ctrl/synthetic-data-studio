"""Privacy risk proxies: Distance-to-Closest-Record (DCR) and simple MIA."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import CATEGORICAL, FEATURE_COLS, NUMERIC, TARGET


def _encode_matrix(df: pd.DataFrame, ref: pd.DataFrame | None = None) -> np.ndarray:
    """Encode features to a numeric matrix for distance / attack features."""
    frames = [df]
    if ref is not None:
        frames.append(ref)
    base = pd.concat(frames, axis=0, ignore_index=True)
    enc = df.copy()
    for col in CATEGORICAL + ([TARGET] if TARGET in df.columns else []):
        if col not in enc.columns:
            continue
        le = LabelEncoder()
        le.fit(base[col].astype(str))
        enc[col] = le.transform(enc[col].astype(str))
    cols = [c for c in FEATURE_COLS if c in enc.columns]
    return enc[cols].astype(float).values


def distance_to_closest_record(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    metric: str = "euclidean",
) -> Dict[str, Any]:
    """
    For each synthetic row, compute distance to nearest real training row.
    Higher median DCR ⇒ less memorization / lower privacy risk (proxy only).
    Also report share of synth rows with near-zero DCR (exact/near copies).
    """
    Xs = _encode_matrix(synthetic, ref=real)
    Xr = _encode_matrix(real, ref=synthetic)
    scaler = StandardScaler()
    Xr_s = scaler.fit_transform(Xr)
    Xs_s = scaler.transform(Xs)

    nn = NearestNeighbors(n_neighbors=1, metric=metric, n_jobs=-1)
    nn.fit(Xr_s)
    dists, _ = nn.kneighbors(Xs_s)
    d = dists.ravel()
    # near-duplicate threshold on standardized space
    near_thresh = 0.05
    return {
        "dcr_mean": float(np.mean(d)),
        "dcr_median": float(np.median(d)),
        "dcr_p5": float(np.percentile(d, 5)),
        "dcr_p95": float(np.percentile(d, 95)),
        "near_duplicate_rate": float(np.mean(d < near_thresh)),
        "n_synth": int(len(synthetic)),
        "n_real": int(len(real)),
        "distances": d,  # for plots; strip before JSON if needed
    }


def membership_inference_auc(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simple shadow-less membership inference proxy:

    Score each candidate record by negative distance to closest synthetic
    neighbor (closer to synth ⇒ more likely 'member' of the training set
    that produced the synthesizer). Members = real_train sample;
    non-members = real_holdout sample.

    AUC ≈ 0.5 ⇒ attack fails (good privacy); AUC ≫ 0.5 ⇒ leakage.
    """
    rng = np.random.default_rng(seed)
    n = min(len(real_train), len(real_holdout), 800)
    members = real_train.sample(n=n, random_state=seed)
    nonmembers = real_holdout.sample(n=n, random_state=seed)

    X_syn = _encode_matrix(synthetic, ref=pd.concat([members, nonmembers], ignore_index=True))
    X_m = _encode_matrix(members, ref=synthetic)
    X_nm = _encode_matrix(nonmembers, ref=synthetic)

    scaler = StandardScaler()
    X_syn_s = scaler.fit_transform(X_syn)
    X_m_s = scaler.transform(X_m)
    X_nm_s = scaler.transform(X_nm)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(X_syn_s)
    d_m, _ = nn.kneighbors(X_m_s)
    d_nm, _ = nn.kneighbors(X_nm_s)

    # Higher score = more member-like
    scores = np.concatenate([-d_m.ravel(), -d_nm.ravel()])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    # shuffle for hygiene
    perm = rng.permutation(len(scores))
    scores, labels = scores[perm], labels[perm]
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")

    return {
        "mia_auc": auc,
        "mean_dist_members": float(d_m.mean()),
        "mean_dist_nonmembers": float(d_nm.mean()),
        "n_attack_samples": int(n),
    }


def privacy_summary(
    synthetic: pd.DataFrame,
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, Any]:
    dcr = distance_to_closest_record(synthetic, real_train)
    mia = membership_inference_auc(real_train, real_holdout, synthetic, seed=seed)
    # Composite: higher is safer (invert MIA AUC around 0.5, reward DCR median)
    privacy_score = float(dcr["dcr_median"] * (1.0 - abs(mia["mia_auc"] - 0.5) * 2))
    out = {k: v for k, v in dcr.items() if k != "distances"}
    out.update(mia)
    out["privacy_score"] = privacy_score
    out["_dcr_distances"] = dcr["distances"]
    return out
