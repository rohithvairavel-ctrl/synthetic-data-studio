"""Dataset loading and preprocessing for Adult / Census Income."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "income"
CATEGORICAL = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
FEATURE_COLS = NUMERIC + CATEGORICAL


def download_adult(force: bool = False) -> Path:
    """Download Adult dataset via OpenML (sklearn fetch) or fall back to UCI URL."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "adult.csv"
    if out.exists() and not force:
        return out

    try:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        # OpenML adult v2 uses 'class' as target
        if "class" in df.columns and TARGET not in df.columns:
            df = df.rename(columns={"class": TARGET})
        df.to_csv(out, index=False)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"OpenML fetch failed ({exc}); trying UCI mirror…")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = FEATURE_COLS + [TARGET]
    df = pd.read_csv(url, names=cols, sep=",", skipinitialspace=True, na_values="?")
    df.to_csv(out, index=False)
    return out


def write_sample(n: int = 200, seed: int = 42) -> Path:
    """Commit a tiny sample for offline demos."""
    path = download_adult()
    df = pd.read_csv(path)
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    sample_path = RAW_DIR / "adult_sample.csv"
    sample.to_csv(sample_path, index=False)
    return sample_path


def load_adult(use_sample: bool = False, max_rows: int | None = 3000) -> pd.DataFrame:
    """Load Adult CSV, clean missing values, optionally subsample for speed."""
    sample_path = RAW_DIR / "adult_sample.csv"
    full_path = RAW_DIR / "adult.csv"
    if use_sample and sample_path.exists():
        df = pd.read_csv(sample_path)
    else:
        if not full_path.exists():
            download_adult()
        df = pd.read_csv(full_path)

    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    # Normalize target to binary strings
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df[TARGET] = df[TARGET].replace({">50K.": ">50K", "<=50K.": "<=50K"})
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


def encode_for_ml(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, dict]:
    """Label-encode categoricals consistently; return X-ready frames + encoders."""
    train = train.copy()
    test = test.copy()
    encoders: dict = {}
    for col in CATEGORICAL:
        if col not in train.columns:
            continue
        le = LabelEncoder()
        vals = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(vals)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le

    y_enc = LabelEncoder()
    y_all = pd.concat([train[TARGET], test[TARGET]], axis=0).astype(str)
    y_enc.fit(y_all)
    train[TARGET] = y_enc.transform(train[TARGET].astype(str))
    test[TARGET] = y_enc.transform(test[TARGET].astype(str))
    return train, test, y_enc, encoders


def train_holdout_split(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
