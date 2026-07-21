# HAINE — Detailed Project Status Report
### What is done, and what remains
*Companion to the README. This is the honest, technical accounting of the pilot.*

---

## Executive summary

HAINE's **four-layer pipeline is complete and runs end to end** for Cochin Port. The hydrodynamics, the satellite product, the physics coupling, and the under-keel-clearance calculator are **independently correct**. The siltation-volume core is currently a **calibrated physical estimate, not a machine-learned model** — it is pinned to the public dredging record and disaggregated by published fluid-mud structure. A **Monte-Carlo probabilistic forecast** and a **gradient-boosted ML training pipeline** have been added; the ML pipeline is proven on synthetic data and is wired to train on real data the moment labels exist. **The single blocker to a genuinely trained model is data — specifically paired pre/post-dredge survey rasters — not code.**

---

## 1. What is DONE

### Layer 1 — Hydrodynamics (TELEMAC-2D) — VALIDATED
- 24-hour tidal simulation on a **35,364-node / 69,598-element** unstructured mesh of the Cochin estuary, reprojected to **UTM 43N (EPSG:32643)**.
- Free surface oscillates correctly at the open boundary; tide propagates inland with physical damping and phase lag.
- **Resolved:** CFL instability caused by a degree-based mesh (fixed by reprojection to metres); boundary ordering via IPOBO edge-counting; `KENT=5` tidal open boundary.
- Files: `reproject_mesh_to_utm.py`, `09_fes2014_tidal_extraction.py`, TELEMAC model on the working machine (`kochi.slf`, `kochi.cli`, `kochi_t2d_tidal.cas`, `kochi_tidal_r.slf`).

### Layer 2 — Satellite SPM fusion — VALIDATED
- 223 Sentinel-2 / Landsat-9 scenes → mosaicked to 82 sensor-dates → temporal-kriged into **13 gap-free monthly SPM fields** (`port_spm_stack.nc`, 13×112×133, EPSG:32643).
- **Validation:** monthly median independently reproduces Cochin's documented monsoon siltation cycle (winter ~22 mg/L → spring ~60 mg/L) with **no tuning**.
- Files: `04`–`08` (SDB + SPM + fusion), `07b` multi-year extraction.

### Layer 3 — Coupling, sub-models, forecast — WORKING (calibrated)
- **Coupling:** tidally-averaged Krone deposition probability extracted per node from the TELEMAC result and co-registered to the SPM grid (`11_extract_taub_from_telemac.py`). Coordinate-offset mismatch found and fixed (local frame → true UTM via `x0=629505, y0=1079516`).
- **Sub-model A (siltation volume):** calibrated to ~24 Mm³/yr over the dredged footprint; per-zone split anchored to published fluid-mud structure. Corrections C1–C4 applied (`docs/layer3_status_and_corrections.md`).
- **Under-keel clearance:** Barrass squat + 1170 kg/m³ nautical-depth criterion. Independent of the siltation calibration.
- **Probabilistic forecast (NEW):** Monte-Carlo (`haine_stochastic_forcing.py`). Normal-year spread from the real dredging record (σ=0.095); flood tail = Poisson count × Log-Normal magnitude (Kerala LN3/PE3 best-fit) anchored to the Periyar/Neeleeswaram gauge + the 2018 flood. Output: P50 ≈ 24.9, 1-in-10 ≈ 32, 1-in-50 ≈ 39 Mm³, per-zone bands. Weak-validated (typical-year median brackets the record).
- **ML training core (NEW):** `haine_submodelA_train.py` — HistGradientBoosting (XGBoost-class), quantile P10/P50/P90, leave-one-year-out CV, permutation importance. Proven on synthetic per-cell data at **R²=0.50, 79% interval coverage**, correct importances (SPM supply ≫ flood > antecedent).
- **Feature assembly (NEW):** `haine_build_features.py` — reads real cubes/coupling fields into the training table. Annual mode runs now; `build_pixel()` ready for per-pixel survey labels.

### Layer 4 — Decision dashboard — BUILT
- Accretion map, per-zone volumes, under-keel clearances, sensitivity analysis; HTML report + JSON export (from `haine_layer3_4_pipeline_v2.py`).

### Deliverables & housekeeping — DONE
- PS-I implementation report (`docs/report/HAINE_PS1_Report.pdf`); master log, validity analysis, corrections log, handoff doc.
- Code sanitized (no GEE project ID, no personal paths, no secrets); configuration via env vars.

---

## 2. What REMAINS (prioritised)

### P0 — Decisive: unlocks a genuinely trained model
| # | Task | Why it matters | Blocker / owner | Effort |
|---|---|---|---|---|
| 1 | **Obtain CPA paired pre/post-dredge survey rasters** | Per-pixel bed-level-change = true labels → thousands of samples → real ML skill. `build_pixel()` is ready. | Cochin Port Authority data request | external |
| 2 | **Fuse multi-year SPM** (FY2016-17/17-18/19-20) | ≥2 years → leave-one-year-out CV can run; inter-annual signal | download + fuse (scripts exist) | ~1 day |

> **Data-ceiling note:** annual dredging totals give only ~16 samples with near-zero inter-annual signal (empirically R²≈0). Per-pixel survey labels are what convert the proven architecture (R²=0.50 on signal-bearing data) into real skill. This distinction is the crux of the whole project.

### P1 — Accuracy upgrades (need compute/data, not blockers)
| # | Task | Why | Effort |
|---|---|---|---|
| 3 | Fit flood stats from real **India-WRIS Neeleeswaram** discharge | replaces assumed flood tail with fitted (`fit_from_series()`) | ~0.5 day |
| 4 | Couple **TOMAWAC** waves to Layer 1 | Cochin siltation is wave-driven; tidal-only is the biggest physics gap | multi-day (TELEMAC run) |
| 5 | **Solver-based Periyar** discharge run | replaces parametric river term; preserved inlet nodes `[31940,32022,32103,32184,32265,32264,31371]` | multi-day (TELEMAC run) |
| 6 | Measured **Cochin sediment params** (D50, w_s, tau_cr) | dominant absolute-volume uncertainty (currently literature defaults) | external |

### P2 — Scale-out
| # | Task | Why | Effort |
|---|---|---|---|
| 7 | Extend to **Paradip** (riverine) & **Kandla** (macro-tidal) | test cross-regime generalisation; IOR/MAHASAGAR deployment | multi-week |

---

## 3. Data inventory

**Have (on the working machine):** validated TELEMAC tidal result; 1 fused SPM year (FY2024-25) + coupling fields (`port_pdep_tidal.tif`, `port_taub_mean.tif`); dredging totals (22.20/21.00/25.21 Mm³ + 24 contracted); Ferket zone structure.
**Downloaded, not yet fused:** multi-year GEE SPM exports (5 zips).
**Needed:** CPA paired survey rasters (P0); India-WRIS Neeleeswaram CSV; FES2014 (AVISO, registration pending); measured sediment params.

## 4. Known technical constants (do not re-derive)
- Mesh local-frame offset to true UTM: `x0=629505, y0=1079516` (EPSG:32643).
- Zone boxes (UTM): approach `(629300,638200,1099000,1104500)`, ernakulam `(638200,642600,1101200,1105700)`, mattancherry `(637000,641500,1094500,1100200)`, basins `(637500,641800,1100200,1103800)`.
- Dredging record: 22.20 (FY2016-17), 21.00 (FY2017-18), 25.21 (FY2019-20) Mm³; 24 contracted 2020-25. Ferket zone weights: approach 19.1, ernakulam 1.4, mattancherry 2.5, basins 0.9.
- `.cli`: node = column [-2]; boundary rank = line order/IPOBO. `KENT=5` tidal open boundary; `SUPG 2;0;2;2` with negative-depth treatment = 2. Fusion: one global variogram (per-pixel refit → SIGKILL).

## 5. Reproducibility notes
- Env: WSL2/Ubuntu, conda env `telemac` (has numpy/pandas/scipy/sklearn/rasterio/matplotlib); `base` does not. Strip CRLF on Windows-copied files.
- `xgboost`/`shap` optional — code falls back to sklearn HistGradientBoosting + permutation importance.
- Outputs are labelled by validity (independently-correct / anchored / indicative / weak) throughout — preserve this discipline in any extension.
