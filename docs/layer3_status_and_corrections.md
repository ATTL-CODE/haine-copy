# HAINE — Layer 3 & 4: Status, Corrections, and Data-Gap Analysis
**Project:** HAINE (Hybrid AI–Numerical Engine) · **Pilot:** Cochin (Kochi) Port · **CRS:** EPSG:32643 (UTM 43N)
**Scope of this document:** what is built and validated, what was corrected by re-running, and — where correction is not possible with available data — exactly which data are needed and why.

---

## 1. What is done (end to end)

### Layer 1 — Hydrodynamic engine (TELEMAC-2D) — COMPLETE
- Selafin mesh, **35,364 nodes / 69,598 triangles**, reprojected to **UTM 43N** in metres (the single fix that resolved all prior CFL instability). Mesh is stored in a **local origin-shifted frame** (`x0=629505, y0=1079516`); true UTM is recovered by adding that offset — this was a critical discovery for coupling (see §3).
- 24-hour tidal run, `CORRECT END OF RUN`; tide propagates seaward→estuary with correct damping/lag. Output: `kochi_tidal_r.slf` (U, V, H, S, B; 49 frames, 5 s step).
- Boundary structure: **one liquid boundary** — the sea (491 KENT=5 nodes, prescribed elevation from `kochi_liq.txt`); all estuary/river edges are wall (637 KLOG=2). **No river inflow boundary exists yet.**
- Tidal constants used: M2=0.24, S2=0.10, K1=0.20, O1=0.10 m; phases 0° placeholder (FES2014 pending AVISO).

### Layer 2 — Satellite SPM fusion — COMPLETE
- 223 raw Sentinel-2 + Landsat-9 SPM tiles → **13 monthly, water-only, gap-free SPM fields** clipped to the port AOI, `port_spm_stack.nc` (13×112×133, EPSG:32643).
- Reproduces the documented monsoon-driven siltation cycle (Jan trough → spring peak). Physically validated independently of Layer 1.

### Layer 3 — Sub-models — BUILT (calibrated-physics, not ML-trained)
- **Coupling fix (done, verified):** per-pixel tidally-averaged Krone deposition probability `p_dep` extracted from the TELEMAC velocity field, gridded onto the SPM grid. Resolved the "no-deposition wall" (uniform placeholder gave `p_dep=0` everywhere) and the coordinate mis-registration (mesh local-frame vs SPM true-UTM). Output: `port_pdep_tidal.tif`, `port_taub_mean.tif`.
- **Sub-model A (siltation volume + depth reduction):** monthly SPM × settling flux × tidal gating × parametric river scour → per-pixel accretion, integrated to annual volume, calibrated to the public dredging record.
- **UKC calculator:** Barrass squat + nautical-depth (1170 kg/m³) per channel, with tidal-window requirement.
- **Dredger selection** and **post-dredge nautical-depth QC:** rule-based from volume/sediment/depth.

### Layer 4 — Decision dashboard — BUILT
- Six-panel dashboard (accretion map, zone volumes, monthly siltation vs river Q, UKC, sensitivity tornado, summary) + self-contained HTML report + results JSON.

---

## 2. Corrections applied by re-running (v2 pipeline)

The v1 run exposed real problems. `haine_layer3_4_pipeline_v2.py` fixes those that **can** be fixed with current data:

| # | Problem in v1 | Correction in v2 | Effect |
|---|---|---|---|
| **C1** | Volume integrated over the **whole AOI** incl. non-dredged backwater → raw 389.6 vs 24 Mm³ (×16 overshoot) | Integrate maintenance-dredging volume over the **dredged zones only**; report non-dredged estuary separately | Overshoot decomposed; footprint made physical |
| **C2** | Raw SPM×p_dep set zone shares → Mattancherry too high (5.14 vs Ferket ~2 m), basins too low (0.82 vs ~5 m) | **Ferket-anchored disaggregation**: zone volume ∝ (Ferket thickness × zone area); satellite used only for within-zone texture | Zone split now matches literature structure |
| **C3** | All error hidden in one ×0.06 pin (implied BED_ENHANCE 0.12, physically impossible) | **Decompose** the factor into dredged-footprint fraction + residual proxy/net-gross factor | Calibration is interpretable, not a black box |
| **C4** | No comparison to the year-by-year record | Print estimate vs 2016-17…2024-25 record with HARD/contracted flags | Makes the single-year limitation explicit |

**What the corrections change conceptually:** v2's **total** is still pinned to the ~24 Mm³/yr record (that is a calibration, not a prediction), and the **zone split is now an imposed Ferket prior** downscaled by satellite texture — i.e. it is a defensible *disaggregation*, not an independent prediction of how siltation divides between zones. This is stated plainly so the output is not over-claimed.

---

## 3. Key technical findings (resources for continuity)

- **Mesh is in a local origin-shifted frame.** `kochi_tidal_r.slf` stores small coordinates; true UTM = local + (629505, 1079516), recovered by reprojecting the degree-mesh `kochi_r.slf` (identical node order, verified `std≈0`). Any future step sampling the SLF against the SPM/UTM grid MUST apply this offset.
- **`.cli` format:** global node number is column `[-2]`; boundary rank is line order / IPOBO (verified consecutive 1..1128), **not** the trailing 0/1 (a color tag).
- **Periyar inlet** at UTM (640263, 1113426) is a broad ~9 m-deep basin at the mesh edge, not a narrow channel — candidate flowrate-boundary nodes `[31940, 32022, 32103, 32184, 32265, 32264, 31371]` preserved for the optional re-run.
- **Ferket et al. 2017 used TELEMAC-2D + TOMAWAC (waves) + SISYPHE (sediment) + DELWAQ (dredging)** — HAINE's Layer 1 is tidal-only; waves are omitted (see §4).

---

## 4. What CANNOT be corrected now — data needed and why

These are limitations that no amount of re-running the current data can fix. Each states the data required and the reason.

### 4.1 It is a calibrated physical estimate, not a trained ML model
- **Why:** Sub-model A's total is pinned to the aggregate dredging figure; there are no per-pixel or per-survey labels to *learn* from. The "prediction" of the total is therefore circular (we told it the answer).
- **Data needed:** **CPA paired pre/post-dredge hydrographic surveys** (bed depth before/after dredging). Under the Dredging Guidelines 2021, dredged volume is computed by differencing these — so they exist. They are the true per-location labels that convert Sub-model A from calibrated-physics into a trainable regression.
- **Route:** `philo@cochinport.gov.in` (Suptdg Engineer, Dredging), `secretary@cochinport.gov.in`, or via the **Sagar Samriddhi** system (NTCPWC/IIT Madras) on which Cochin is a first adopter — it already fuses daily reports + pre/post surveys.

### 4.2 No inter-annual training (only one SPM year)
- **Why:** the fused SPM cube covers **one year** (2024-06…2025-06). The dredging record has genuine inter-annual variation (22.2 / 21.0 / 25.21 Mm³) but with one predictor-year we cannot fit it — every year returns the same estimate (see C4 output).
- **Data needed:** **multiple years of Sentinel-2/Landsat-9 SPM cubes** (re-run Layer 2 for 2016-17…2019-20, the years with HARD published volumes). Then the year-by-year volumes become real training targets and the model can learn a supply→volume relationship.

### 4.3 No independent per-zone labels
- **Why:** the public record and DCI contracts give the **aggregate** volume and the *list* of zones dredged, never m³ per zone. So the zone split cannot be validated — only imposed from Ferket.
- **Data needed:** **per-zone dredged quantities** (from CPA dredging measurement books / Sagar Samriddhi per-area logs), or the Ferket full report's per-zone siltation volumes (the conference paper gives thicknesses, not zone m³).

### 4.4 Waves are not modelled
- **Why:** Cochin siltation is driven substantially by **monsoon wave-induced resuspension** of coastal/mud-bank sediment (Ferket, Grokipedia). Layer 1 is tidal-only, so the hydrodynamic lever is systematically incomplete — the outer channel's wave-driven component is missing.
- **Data needed / action:** couple **TOMAWAC** (waves) to Layer 1 as Ferket did, forced by **ERA5 significant wave height / wind**. This is a Layer 1 extension, not a data request, but it needs the wave model set up and run.

### 4.5 River forcing is parametric, not solver-coupled
- **Why:** the mesh has no river boundary; adding Periyar discharge properly means editing the `.cli` to open a flowrate boundary and re-running TELEMAC (a broad, deep inlet makes this a multi-cycle solver task). v2 folds Periyar in as a monthly-Q scour term instead.
- **Data needed / action:** the **TELEMAC river re-run** using the preserved `PERIYAR_BOUNDARY_NODES`, ideally with a **CWC Neeleeswaram gauge** discharge series (replacing the ERA5-apportionment proxy). After the run, re-run `11_extract_taub` on `kochi_monsoon_r.slf`.

### 4.6 Sediment parameters are literature proxies
- **Why:** D50 (~20 µm), settling velocity (~0.5 mm/s), critical shear (~0.2 Pa), dry density (~450 kg/m³) have **no Cochin-measured value** in open sources (only in-situ wet bulk density 1180–1450 kg/m³, Rao et al. 1987). These directly scale the deposition flux (see the sensitivity panel: ±20% on each moves the raw volume ±40%).
- **Data needed:** measured Cochin cohesive-mud properties — Ferket 2017 full report, Rasheed (1997) CUSAT PhD, or fresh grab-sample lab analysis (settling column + rheometer).

---

## 5. Resources and references (all consulted)

**Data / official**
- Cochin Port Authority Administration Reports & tender pages — `cochinport.gov.in/annual-report`; Ministry of Ports/Shipping — `shipmin.gov.in`.
- Dredging Guidelines for Major Ports 2021 (pre/post-survey volume computation).
- Sagar Samriddhi dredging-monitoring system (NTCPWC / IIT Madras).
- Sagarmala Master Plan for Cochin Port (`sagarmala.gov.in`) — channel/estuary morphology.
- DCI contract disclosures (2020-21…2024-25 maintenance dredging).

**Literature**
- Ferket, B. et al. (2017) *Assessment of Siltation Processes and Implementation of Nautical Depth in the Port of Cochin, India*, CEDA Dredging Days — "+20 Mm³/yr"; per-zone fluid-mud thickness; nautical depth 1170 kg/m³; TELEMAC-2D+TOMAWAC+SISYPHE+DELWAQ.
- Naik et al. (1983) — 70/30 outer/interior siltation split; monsoon dominance (via Van Rijn note).
- Rao et al. (1987) *Marine Geology* — Cochin sediment wet bulk density 1180–1450 kg/m³.
- Adithya et al. (2025) *Defence Science Journal* (DRDO-NPOL) — Cochin tidal constants, currents.
- Roshan et al. (2025) *Hydrology* (MDPI) — Periyar ≈33% of ~19,000 Mm³/yr river discharge.
- Barrass (squat), PIANC (UKC), Krone (cohesive deposition) — method references.

**Figures used as anchors (with status)**
- Maintenance dredging: 22.20 (2016-17), 21.00 (2017-18), 25.21 (2019-20) Mm³ **[HARD]**; 24 Mm³/yr 2020-21…2024-25 **[contracted]**.
- Ferket total **+20 Mm³/yr [HARD, grey lit]**; per-zone fluid mud: outer 6–7 m, basins ~5 m, inner channels ~2 m.
- Channel geometry: approach 16 km × 260 m × 14.5 m (→16 m); Ernakulam 260 m × 13.25 m; Mattancherry 183 m × 9.75 m **[HARD]**.
- Periyar discharge: proxy from basin apportionment; **CWC Neeleeswaram** = authoritative source (not yet obtained).

---

## 6. Bottom line
- **Built and trustworthy:** Layers 1–2; the coupling geometry/physics; the UKC calculator; the decision framework and dashboard.
- **Built but calibrated-not-trained:** Sub-model A total (pinned to record) and zone split (Ferket-imposed).
- **Blocking a true model:** CPA paired surveys (labels) + multiple SPM years (inter-annual) are the two decisive missing datasets; waves + solver-coupled river + measured sediment are the accuracy upgrades.
