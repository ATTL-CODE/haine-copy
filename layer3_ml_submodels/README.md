# Layer 3 — Physics Coupling, ML Sub-models & Forecast

- `11_extract_taub_from_telemac.py` — tidally-averaged bed shear / Krone deposition probability from the TELEMAC result, co-registered to the SPM grid.
- `12_locate_river_inflow.py`, `13_periyar_segment.py` — river-inlet diagnostics (Periyar).
- `haine_layer3_4_pipeline_v2.py` — the calibrated Sub-model A + UKC + dashboard pipeline (v2, with corrections C1–C4). `haine_layer3_4_pipeline.py` is the legacy version.
- **`haine_build_features.py`** — assembles the real feature table. `build()` = annual mode (runs now); `build_pixel({FY: label_raster})` = per-pixel mode for CPA survey labels (the real-skill path).
- **`haine_submodelA_train.py`** — the ML core: `HistGradientBoostingRegressor` (XGBoost-class), quantile P10/P50/P90 intervals, leave-one-year-out CV, permutation-importance (SHAP proxy). Auto-loads the features CSV.
- **`haine_stochastic_forcing.py`** — Monte-Carlo probabilistic forecast (lognormal normal-year spread from the dredging record; Poisson×Log-Normal flood tail anchored to the Periyar/Neeleeswaram gauge + 2018 flood). `fit_from_series()` fits the flood stats from a real India-WRIS discharge CSV.
- `haine_multiyear_calibrate.py` — inter-annual calibration against the dredging record.

**Data ceiling:** annual labels give ~16 samples (no inter-annual signal); per-pixel survey labels give thousands (real skill). Same architecture either way.
