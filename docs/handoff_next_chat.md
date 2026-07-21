# HAINE — Handoff to Next Chat
*Paste this whole document into a new chat to continue. It is self-contained.*

---

## 0. Who / what this is
- **User:** PS-I intern project at Maritime Research Centre (MRC) / FUDA, Pune. Mentors: Jay Pinjarkar, Shridhar Prabhuraman.
- **Project HAINE** (Hybrid AI–Numerical Engine): 4-layer AI-assisted dredging-assessment framework. Pilot = **Cochin (Kochi) Port**.
- **Honest status of the whole model:** a *validated baseline* — Layer 1 tidal hydrodynamics (TELEMAC-2D), Layer 2 satellite SPM (validated), Layer 3 physics-coupled **calibrated** siltation estimate + UKC, Layer 4 dashboard. The siltation core is **calibrated, NOT yet ML-trained** (blocked on labels). A probabilistic forecast layer and an ML training pipeline were added this session (below).

## 1. Environment (CRITICAL — apply every session)
- WSL2/Ubuntu, machine `DESKTOP-4LOOISI`. Default shell is **fish** → type `bash` first.
- **`conda activate telemac`** — the working env. `(base)` is MISSING pandas/sklearn/matplotlib; only `(telemac)` has the deps. Confirm the prompt shows `(telemac)`.
- Working dir `~/haine_telemac/`. Windows files at `~/Downloads/`.
- **Strip CRLF on every file copied from Windows:** `sed -i 's/\r$//' file.py` (else weird import/syntax errors).
- If a plot step says `No module named 'matplotlib'`: `pip install matplotlib` into the telemac env.
- Never `find /` unscoped (crawls the Windows C: drive and hangs). Big unzips on `/mnt/c/...` are slow — extract to `~/` instead.

## 2. >>> IMMEDIATE NEXT TASK <<<  Fuse the multi-year SPM, get CV running
**Why:** the ML trainer works but only has **1 fused SPM year (FY2024-25) → 4 samples**, so leave-one-year-out cross-validation cannot run (needs ≥2 years). Fusing the other years (FY2016-17, FY2017-18, FY2019-20) → 16 samples → CV runs.

**What's on disk:** five GEE export zips in `~/Downloads/`:
`HAINE_SPM-20260713T115333Z-2-001.zip … -005.zip`

**Step A — inspect the tiles (do this first, it drives everything):**
```bash
cd ~/Downloads
ls -lh HAINE_SPM-20260713T115333Z-2-*.zip          # sizes / not 0-byte
unzip -l HAINE_SPM-20260713T115333Z-2-001.zip | head -30   # <-- tile NAMING (year+sensor)
```
Report back the `unzip -l` listing. The filenames reveal the year/date token and sensor (Sentinel-2 vs Landsat-8 for pre-2021). **The next chat must see this to write the fusion script correctly.**

**Step B — confirm the target format to reproduce:**
```bash
ls ~/haine_layer2/spm_fused_port/ | head -40   # the working FY2024-25 folder
```
The feature builder expects, per year folder, monthly rasters named `port_spm_YYYY-MM.tif` in **EPSG:32643**, plus (in the 2024-25 folder only) the coupling fields `port_pdep_tidal.tif` and `port_taub_mean.tif`. The fused cube is `port_spm_stack.nc` (shape 13×112×133).

**Step C — find the fusion method already used (script 08, kriging):**
```bash
ls ~/haine_telemac/*.py | grep -iE "fus|krig|08|clip|stack"
```

**Step D — next chat writes `haine_fuse_multiyear.py`** that ingests the unzipped tiles, groups by year+sensor, fuses (single global temporal variogram — do NOT refit per pixel, it SIGKILLs), clips to the port AOI, water-masks, and writes `~/haine_layer2/spm_fused_port_FY2016-17/`, `…FY2017-18/`, `…FY2019-20/` in the same `port_spm_YYYY-MM.tif` format.

**Step E — re-run the pipeline:**
```bash
conda activate telemac
python haine_build_features.py         # now finds 4 years -> ~16-row CSV
python haine_submodelA_train.py        # now runs leave-one-year-out CV
```

> Realistic expectation: even at 16 samples the R² will be weak — annual labels have little inter-annual signal. That's the known ceiling, not a bug (see §4). The point of this task is to get CV *running* on real multi-year data.

## 3. What was built THIS session (all in the user's outputs; copy to ~/haine_telemac/)
1. **`haine_stochastic_forcing.py`** — probabilistic siltation forecast (Monte-Carlo). **WORKS / ran successfully.** Output: P50 ≈ 24.9, P90 ≈ 30.9, 1-in-10 ≈ 32, 1-in-50 ≈ 39 Mm³, per-zone bands. Normal-year spread calibrated from the real dredging record (σ=0.095); flood tail = Log-Normal (Kerala LN3/PE3 best-fit) anchored to Periyar/**Neeleeswaram** gauge behaviour + the **2018 Kerala flood** (~1-in-75, +55%). Has a ready `fit_from_series(csv,q_col,date_col)` to fit the flood stats from a real **India-WRIS Neeleeswaram** daily-discharge CSV.
2. **`haine_submodelA_train.py`** — the ML core (replaces the calibrated estimate). `HistGradientBoostingRegressor` (XGBoost-class; xgboost/shap won't pip-install → sklearn used). Quantile P10/P50/P90 intervals, **leave-one-year-out CV**, permutation importance (SHAP proxy). **Patched** to skip CV cleanly when <2 years. Proven at **R²=0.50, 79% interval coverage** on synthetic per-cell data with correct importances (SPM supply ≫ flood > antecedent). Auto-loads `~/haine_layer3/haine_submodelA_features.csv`.
3. **`haine_build_features.py`** — real feature assembly. **Annual mode** (default, runs now): label = FY dredged volume × Ferket zone fraction → accretion rate; reads the year folders + coupling fields. **Pixel mode** `build_pixel({FY:label_raster})`: per-pixel features + per-pixel bed-level-change label from CPA surveys → thousands of samples (the real-skill path).

## 4. The honest data ceiling (the real bottleneck — it's DATA, not code)
- **Annual labels** (dredged volume per year per zone) = at most 4 years × 4 zones = **16 samples**, and the label barely varies year-to-year → **near-zero learnable signal**. Proven twice (R²≈0). Fusing multi-year cubes lets CV *run* but won't give real skill.
- **Per-pixel labels** = **CPA paired pre/post-dredge survey rasters** (bed-level change, m/yr, on the SPM grid) → thousands of samples → **genuine trained skill** (architecture already proven at R²=0.50). **This is THE decisive unlock.** Everything is wired: drop the survey rasters into `build_pixel()` and it trains for real, no code changes.
- Formal data request to Cochin Port Authority (`secretary@cochinport.gov.in`) / NTCPWC Sagar Samriddhi (`ntcpwc@iitm.ac.in`) for these surveys is the single highest-value action.

## 5. Other real model upgrades (need user compute; not blockers)
- **India-WRIS Neeleeswaram CSV** (Periyar daily discharge, 1979–2018, from indiawris.gov.in) → run `fit_from_series()` to replace the stochastic model's *assumed* flood stats with fitted ones.
- **TOMAWAC wave coupling** to Layer 1 — Cochin siltation is wave-driven; tidal-only is the biggest physics gap. Needs a TELEMAC run.
- **Solver-based Periyar river run** — preserved inlet nodes `[31940,32022,32103,32184,32265,32264,31371]`; open the river boundary in the `.cli`, force discharge, re-run, re-extract τ_b. Needs a TELEMAC run.
- **Measured Cochin sediment params** (D50, w_s, τ_cr) — sensitivity analysis shows these dominate absolute-volume uncertainty (currently literature defaults D50=20µm, w_s=0.5mm/s, τ_cr=0.2Pa).

## 6. Key technical constants (continuity)
- **Coordinate offset:** solver mesh is in a LOCAL frame; add `x0=629505, y0=1079516` to recover true UTM (EPSG:32643). The SPM/coupling tifs are already in true UTM.
- **Zone boxes (UTM 43N):** approach `(629300,638200,1099000,1104500)`, ernakulam `(638200,642600,1101200,1105700)`, mattancherry `(637000,641500,1094500,1100200)`, basins `(637500,641800,1100200,1103800)`.
- **Dredging record (HARD):** 22.20 (FY2016-17), 21.00 (FY2017-18), 25.21 (FY2019-20) Mm³; 24 contracted 2020-25. **Ferket zone split weights:** approach 19.1, ernakulam 1.4, mattancherry 2.5, basins 0.9.
- **SPM cube:** 13×112×133, EPSG:32643. Monsoon months = Jun–Sep. Fusion: ONE global variogram (per-pixel refit → SIGKILL).
- **`.cli` file:** node number is column [-2]; boundary rank = line order / IPOBO (NOT the trailing colour tag). Use KENT=5 (prescribed elevation) for tidal open boundary; SUPG `2;0;2;2` when negative-depth treatment = 2. `get_ipobo()` via matplotlib misses boundary node 2 → use raw edge-counting.
- **CRS/CFL bug:** mesh in geographic degrees → CFL instability every timestep; reproject to UTM metres fixes it.

## 7. Deliverables status (deadline: Tue 21 July, morning)
All files exist in outputs (`.md` + most as `.pdf`):
- ✅ **Research/Innovation Note**, ✅ **UDA Digest Article** (both md+pdf)
- ✅ **Midway Presentation** (`HAINE_Midway_Presentation.pptx`, validated)
- ✅ **Final Presentation** (`HAINE_Cochin_Dredging.pptx`)
- ✅ **Final Report – Part 2** (`HAINE_PS1_Report.pdf` + `.tex`). *Overleaf note:* upload the 17 loose PNGs from `HAINE_report_figures.zip` (Overleaf doesn't auto-extract zips); the `.tex` uses extension-less `\includegraphics` + `\graphicspath`.
- ✅ **Final Video Script** (`HAINE_Final_Video_Script.pdf`) — user must **record** the video.
- ⚠️ **E-learning Module** (`HAINE_Elearning_Module.pdf`) — content done; **confirm required format with Rushikesh** (SCORM / video / PDF).

## 8. First things for the next chat to say/do
1. Ask the user to run **§2 Step A** (`unzip -l …-001.zip | head -30`) and paste the listing.
2. From the tile naming, write **`haine_fuse_multiyear.py`** matching the existing `spm_fused_port/` format (§2 Step B).
3. Get to ≥2 fused years → builder → trainer CV runs.
4. Keep the honest framing: pushing multi-year is a real upgrade but **not required** for a defensible submission — the calibrated estimate + working probabilistic forecast + proven ML pipeline already stand. If the deadline is tight, submit those and pursue CPA survey labels afterward as the true path to a trained model.
