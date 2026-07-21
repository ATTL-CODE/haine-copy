# 2026-summer-bits-haine-dredging

# HAINE — Hybrid AI–Numerical Engine for Dredging Assessment

**Open, physics-grounded siltation prediction for tropical Indian ports.** Pilot: **Cochin (Kochi) Port.**
BITS Pilani Practice School–I · Maritime Research Centre (MRC) / Foundation for Underwater Domain Awareness (FUDA), Pune.

---

## 1. What problem this solves

Indian major ports dredge tens of millions of m³ of sediment a year — **reactively**, with no predictive picture of *where*, *when*, or *how much* silt accumulates. Cochin (~24 Mm³/yr) is the highest-siltation major port. Temperate sediment models fail here (monsoon supply + cloud blackout, warm-water mud, dammed rivers). **HAINE** fuses hydrodynamics, satellite observation, and machine learning into one open decision-support pipeline built for this tropical context.

## 2. Honest status — read before using

This is a **validated baseline, not a finished predictive model**, and the repo labels which is which:

| Component | Status |
|---|---|
| Layer 1 — tidal hydrodynamics (TELEMAC-2D) | ✅ **validated** (24-h solve, 35,364-node UTM mesh) |
| Layer 2 — satellite SPM cube | ✅ **validated** (independently reproduces the monsoon siltation cycle) |
| Layer 3 — physics coupling + under-keel clearance | ✅ **working** |
| Layer 3 — siltation volume (Sub-model A) | ⚠️ **calibrated, NOT ML-trained** (pinned to the dredging record) |
| Layer 3 — probabilistic forecast (Monte-Carlo) | ✅ **working** (P10/P50/P90 + return periods) |
| Layer 3 — ML training pipeline | ✅ **built & proven on synthetic data (R²=0.50)**; ⏳ awaits real labels |
| Layer 4 — decision dashboard | ✅ **built** |

**The bottleneck is data, not code.** See `docs/PROJECT_STATUS.md` for the full done/to-do breakdown, and `docs/prediction_validity.md` for exactly how far each output can be trusted.

## 3. Architecture

```
Layer 1  Hydrodynamic engine   TELEMAC-2D tidal solve   -> bed shear / deposition gating
Layer 2  Satellite SPM fusion  Sentinel-2 + Landsat      -> 13-month water-only SPM cube
Layer 3  Physics/ML sub-models coupling + Sub-model A    -> siltation volume, UKC, dredger,
                                + probabilistic forecast     post-dredge QC
Layer 4  Decision dashboard    maps + report + JSON      -> per-zone volumes, clearances
```

## 4. Repository layout

```
layer1_hydrodynamics/   mesh reprojection, FES2014 tidal extraction, TELEMAC build scripts
layer2_satellite/       GEE SPM/SDB extraction (04-08) + multi-year extraction + kriging fusion
layer3_ml_submodels/    tau_b extraction, river/segment tools, pipeline, and:
                          haine_build_features.py      real feature assembly (annual + pixel)
                          haine_submodelA_train.py     ML core (gradient-boosted, P10/P50/P90, LOO-CV)
                          haine_stochastic_forcing.py  Monte-Carlo probabilistic forecast
                          haine_multiyear_calibrate.py inter-annual calibration
layer4_dashboard/       dashboard notes (produced by the Layer 3 pipeline)
results/                sample output figures
docs/                   PROJECT_STATUS, CONFIGURATION, validity, corrections, handoff, PS-I report
data/                   expected data layout (data itself NOT committed - see data/README.md)
licenses/third_party/   GPL/LGPL/AGPL for openTELEMAC dependencies
```

## 5. Installation

```bash
# Python environment (Layers 2-4)
conda create -n telemac python=3.11 -y && conda activate telemac
pip install -r requirements.txt

# Layer 1: install openTELEMAC v8p5r0 SEPARATELY
#   conda install -c conda-forge opentelemac   (or build from source - see
#   layer1_hydrodynamics/telemac_build/)
```

## 6. Configuration (no secrets are hard-coded)

Copy `.env.example` to `.env` and fill it in, or export the variables. Full table in `docs/CONFIGURATION.md`.

```bash
export GEE_PROJECT=your-gee-project-id     # your Earth Engine cloud project
export HAINE_DOWNLOAD_DIR=~/Downloads      # where figures/JSON are saved (optional)
earthengine authenticate                   # one-time GEE auth (Layer 2)
```
Other credentialed inputs (AVISO FES2014, ERA5 `~/.cdsapirc`, India-WRIS CSV) are downloaded locally by you — none are committed (`.gitignore` blocks them).

## 7. Usage

```bash
conda activate telemac

# probabilistic siltation forecast (needs no extra data; runs today)
python layer3_ml_submodels/haine_stochastic_forcing.py

# ML training pipeline
python layer3_ml_submodels/haine_build_features.py    # build feature table from your cubes
python layer3_ml_submodels/haine_submodelA_train.py   # train + leave-one-year-out CV + importances
```
Layer-by-layer run order and the multi-year / per-pixel label paths are in `docs/handoff_next_chat.md`.

## 8. Environment gotchas (WSL/Ubuntu)

- Default shell may be **fish** → type `bash` first; then `conda activate telemac` (the `base` env lacks the deps).
- **Strip CRLF** on any file copied from Windows: `sed -i 's/\r$//' file.py`.
- Don't run `find /` unscoped (crawls the Windows C: drive).

## 9. What still needs to be done (summary — full list in docs/PROJECT_STATUS.md)

**Decisive (unlocks a trained model):**
- [ ] Obtain **CPA paired pre/post-dredge survey rasters** → per-pixel labels → train Sub-model A for real (`build_pixel()` is ready).
- [ ] Fuse **multi-year SPM** (FY2016-17, 2017-18, 2019-20) → enables leave-one-year-out CV.

**Accuracy upgrades (need TELEMAC/CWC data, not blockers):**
- [ ] Fit flood stats from the real **India-WRIS Neeleeswaram** discharge record (`fit_from_series()`).
- [ ] Couple **TOMAWAC** waves to Layer 1; run **solver-based Periyar** discharge.
- [ ] Acquire **measured Cochin sediment parameters** (D50, w_s, tau_cr).

**Scale-out:**
- [ ] Extend to **Paradip** and **Kandla** (contrasting siltation regimes).

## 10. Key results (Cochin pilot)

- Validated 24-h TELEMAC-2D tidal solution; satellite SPM reproduces the monsoon cycle with no tuning.
- Calibrated annual siltation ≈ **24 Mm³/yr**, disaggregated by navigation zone.
- Probabilistic forecast: P50 ≈ 24.9, 1-in-10-yr ≈ 32, 1-in-50-yr ≈ 39 Mm³.
- ML core proven at **R²=0.50, 79% interval coverage** on synthetic per-cell data, correct feature importances.

See `results/` and `docs/report/HAINE_PS1_Report.pdf`.

## 11. Author, citation, license

**Pranahv Subramaniam Sankar** (BITS Pilani) · MRC / FUDA, Pune. Mentors: Jay Pinjarkar, Shridhar Prabhuraman.
Cite via `CITATION.cff`. Code: **MIT** (`LICENSE`). Built on openTELEMAC (GPL/LGPL — `licenses/third_party/`), Google Earth Engine, Copernicus Sentinel, USGS/NASA Landsat, ECMWF ERA5.
