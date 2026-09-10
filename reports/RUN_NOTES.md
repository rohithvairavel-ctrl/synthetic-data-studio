# Experiment run notes

- **Date:** 2026-09-10
- **Environment:** Python 3.13, SDV 1.38.3, scikit-learn, Streamlit
- **Data:** OpenML Adult v2, stratified subsample n=2000 → train 1500 / holdout 500
- **Probe model:** `LogisticRegression(class_weight="balanced")` with F1-optimal threshold fit on train predictions
- **Generators run:** SDV GaussianCopula (`noise_scale` ∈ {1.0, 1.25, 1.5, 2.0}), Fallback Histogram+MVN (`noise_scale` ∈ {0.75, 1.0, 1.5})
- **CTGAN:** Supported in code (`--ctgan` / `method="ctgan"`). A 25-epoch CPU run was started but aborted for wall-clock; use GPU or fewer rows for portfolio demos.
- **Artifacts:** `metrics.csv`, `metrics.json`, SVG figures under `figures/`, `outputs/synthetic_preview.csv`
- **Blockers:** None for GaussianCopula + fallback path. Full `adult.csv` excluded from git (download via script). CTGAN optional/slow on CPU.
