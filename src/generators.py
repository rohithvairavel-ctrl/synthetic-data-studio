"""Synthetic data generators: SDV GaussianCopula / CTGAN with sklearn fallback."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, TARGET

HAS_SDV = False
try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    HAS_SDV = True
except ImportError:
    HAS_SDV = False


class FallbackHistogramMVN:
    """IterativeImputer + MVN + histogram sampler when SDV is unavailable."""

    def __init__(self, noise_scale: float = 1.0, seed: int = 42):
        self.noise_scale = noise_scale
        self.seed = seed
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.imputer = IterativeImputer(random_state=seed, max_iter=10)
        self.mean_: Optional[np.ndarray] = None
        self.cov_: Optional[np.ndarray] = None
        self.columns_: Optional[list[str]] = None
        self.cat_levels_: dict[str, np.ndarray] = {}
        self.cat_probs_: dict[str, np.ndarray] = {}
        self.num_bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, data: pd.DataFrame) -> "FallbackHistogramMVN":
        df = data.copy()
        self.columns_ = list(df.columns)
        encoded = df.copy()
        for col in list(CATEGORICAL) + [TARGET]:
            if col not in encoded.columns:
                continue
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col].astype(str))
            self.encoders[col] = le
            vals, counts = np.unique(encoded[col], return_counts=True)
            self.cat_levels_[col] = vals
            self.cat_probs_[col] = counts / counts.sum()
        for col in NUMERIC:
            if col in encoded.columns:
                self.num_bounds_[col] = (
                    float(encoded[col].quantile(0.01)),
                    float(encoded[col].quantile(0.99)),
                )
        X = encoded[self.columns_].astype(float).values
        X = self.imputer.fit_transform(X)
        Xs = self.scaler.fit_transform(X)
        self.mean_ = Xs.mean(axis=0)
        cov = np.cov(Xs, rowvar=False)
        cov = cov * (self.noise_scale**2) + np.eye(cov.shape[0]) * 1e-4
        self.cov_ = cov
        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        assert self.mean_ is not None and self.cov_ is not None and self.columns_ is not None
        raw = rng.multivariate_normal(self.mean_, self.cov_, size=num_rows)
        inv = self.scaler.inverse_transform(raw)
        out = pd.DataFrame(inv, columns=self.columns_)
        for col, le in self.encoders.items():
            if col in self.cat_levels_:
                drawn = rng.choice(self.cat_levels_[col], size=num_rows, p=self.cat_probs_[col])
                rounded = np.clip(np.rint(out[col].values), 0, len(le.classes_) - 1).astype(int)
                mix = np.where(rng.random(num_rows) < 0.35, drawn, rounded)
                mix = np.clip(mix, 0, len(le.classes_) - 1).astype(int)
                out[col] = le.inverse_transform(mix)
        for col in NUMERIC:
            if col in out.columns:
                lo, hi = self.num_bounds_.get(col, (0.0, float(out[col].max())))
                out[col] = out[col].clip(lower=max(0.0, lo), upper=hi).round(0)
        return out


def _metadata_from_df(df: pd.DataFrame) -> Any:
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    for col in ("capital-gain", "capital-loss", "fnlwgt", "age", "hours-per-week", "education-num"):
        if col in df.columns:
            try:
                meta.update_column(col, sdtype="numerical")
            except Exception:
                pass
    for col in CATEGORICAL + [TARGET]:
        if col in df.columns:
            try:
                meta.update_column(col, sdtype="categorical")
            except Exception:
                pass
    return meta


def _clip_numeric_to_train(synth: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    out = synth.copy()
    for col in NUMERIC:
        if col in out.columns and col in train.columns:
            lo, hi = train[col].quantile(0.001), train[col].quantile(0.999)
            out[col] = out[col].clip(lower=max(0, lo), upper=hi)
    return out


class SynthGenerator:
    """Unified interface over SDV synthesizers and the MVN fallback."""

    def __init__(self, method: str = "gaussian_copula", seed: int = 42, epochs: int = 50, noise_scale: float = 1.0):
        self.method = method
        self.seed = seed
        self.epochs = epochs
        self.noise_scale = noise_scale
        self._model: Any = None
        self._train_ref: Optional[pd.DataFrame] = None
        self.backend: str = "unknown"

    def fit(self, data: pd.DataFrame) -> "SynthGenerator":
        method = self.method.lower()
        self._train_ref = data.copy()
        if method in {"gaussian_copula", "ctgan"} and HAS_SDV:
            meta = _metadata_from_df(data)
            if method == "gaussian_copula":
                self._model = GaussianCopulaSynthesizer(meta, default_distribution="beta")
                self.backend = "sdv.GaussianCopula"
            else:
                self._model = CTGANSynthesizer(meta, epochs=self.epochs, verbose=False)
                self.backend = "sdv.CTGAN"
            self._model.fit(data)
        else:
            self._model = FallbackHistogramMVN(noise_scale=self.noise_scale, seed=self.seed)
            self._model.fit(data)
            self.backend = "fallback.HistogramMVN"
            if method in {"gaussian_copula", "ctgan"} and not HAS_SDV:
                print(f"[warn] SDV unavailable - using {self.backend}")
        return self

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() before sample().")
        if self.backend.startswith("sdv"):
            syn = self._model.sample(num_rows=num_rows)
            if self._train_ref is not None:
                syn = _clip_numeric_to_train(syn, self._train_ref)
            if self.noise_scale > 1.0:
                rng = np.random.default_rng(self.seed)
                syn = syn.copy()
                extra = self.noise_scale - 1.0
                for col in NUMERIC:
                    if col in syn.columns:
                        scale = float(syn[col].std(ddof=0) or 1.0)
                        syn[col] = syn[col] + rng.normal(0, extra * 0.08 * scale, size=len(syn))
                        syn[col] = syn[col].clip(lower=0)
                if self._train_ref is not None:
                    syn = _clip_numeric_to_train(syn, self._train_ref)
            return syn
        return self._model.sample(num_rows)
