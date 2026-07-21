# HAINE — Master Log
### Tools · Resources · Timeline · Every Problem & Correction (Layer 1 → now → future)
**Project:** HAINE (Hybrid AI–Numerical Engine) — AI-assisted dredging assessment for Indian Ocean Region ports
**Pilot:** Cochin (Kochi) Port · **Secondary (future):** Paradip, Kandla · **CRS:** EPSG:32643 (UTM 43N)
**Architecture:** Layer 1 hydrodynamics → Layer 2 satellite SPM → Layer 3 ML/physics sub-models → Layer 4 decision dashboard

---

## 1. Tools & software stack (everything used)

### Environment & shell
| Tool | Role |
|---|---|
| WSL2 / Ubuntu on Windows | host OS for the whole pipeline |
| fish shell (default) + bash | must type `bash` first each session |
| miniforge / conda, env `telemac` | Python environment manager |
| `sed -i 's/\r$//'` | strip CRLF from Windows-copied files (recurring necessity) |
| `grep -c` | gate on updated scripts before running (avoid stale copies) |

### Numerical / hydrodynamic (Layer 1)
| Tool | Role |
|---|---|
| openTELEMAC **v8p5r0** (conda-forge) | TELEMAC-2D shallow-water solver |
| `data_manip.formats.selafin.Selafin` | read/write Selafin (`.slf`) mesh & results |
| Selafin `.cli` (Conlim) + `kochi_liq.txt` | boundary conditions + liquid-boundary time series |
| IPOBO array | authoritative boundary-point ordering (1..1128) |
| `pyproj.Transformer` (EPSG:4326→32643) | mesh reprojection / UTM offset recovery |
| `OMPI_ALLOW_RUN_AS_ROOT` flags | required for MPI runs as root |

### Satellite / remote sensing (Layer 2)
| Tool | Role |
|---|---|
| Google Earth Engine (project `your-gee-project-id`) | satellite data access & SPM extraction |
| Sentinel-2 + Landsat-9 | source imagery (223 raw tiles) |
| Nechad algorithm | SPM / turbidity retrieval |
| `gdown` (URL form) | bulk download via Drive folder zip |
| `pykrige` | temporal kriging fusion |
| `rasterio`, `netCDF4` | raster I/O, cube assembly |
| JRC Global Surface Water (planned) | publication-grade water mask upgrade |

### ML / physics sub-models (Layer 3)
| Tool | Role |
|---|---|
| `numpy` | all array math |
| `scipy.interpolate.griddata` | mesh → SPM-grid interpolation (linear + nearest fill) |
| Krone deposition formula | `p_dep = max(0, 1 − τ_b/τ_cr)` |
| Quadratic drag / Manning friction | bed shear `τ_b = ρ·C_d·U²` (Manning n=0.025 in solver) |
| Barrass squat formula | UKC calculator |
| Nautical-depth criterion (1170 kg/m³) | UKC + post-dredge QC |
| `matplotlib` | accretion maps, dashboards, all figures |

### Presentation / documentation (Layer 4 + deliverables)
| Tool | Role |
|---|---|
| `pptxgenjs` (Node.js) | slide-deck generation |
| LibreOffice `soffice`, `pdftoppm`, `markitdown`, `pdf2image`, `Pillow` | deck render + QA |
| Markdown | handoffs, status, validity, this log |
| JSON + self-contained HTML | machine-readable results + report |

### Referenced but not yet integrated (future)
FES2014 / AVISO (tidal phases), ERA5 (precip→discharge, wind), TOMAWAC (waves), SISYPHE/DELWAQ (Ferket's sediment/dredging modules), CWC Neeleeswaram gauge (Periyar discharge), Sagar Samriddhi / NTCPWC (dredging survey data).

---

## 2. Data resources & references

### Peer-reviewed / technical literature
| Source | What it provides | Status |
|---|---|---|
| **Ferket et al. 2017** (CEDA Dredging Days) | +20 Mm³/yr total siltation; per-zone fluid-mud thickness (outer 6–7 m, basins ~5 m, inner 1–3 m); nautical depth 1170 kg/m³; monsoon-restricted; used TELEMAC-2D+TOMAWAC+SISYPHE+DELWAQ | HARD (grey lit) |
| **Naik et al. 1983** (via Van Rijn note) | 70/30 outer/interior siltation split; monsoon dominance | HARD |
| **Rao et al. 1987** (*Marine Geology*) | Cochin sediment wet bulk density 1180–1450 kg/m³ | HARD |
| **Adithya et al. 2025** (*Defence Science Journal*, DRDO-NPOL) | Cochin tidal constants (M2 0.24, S2 0.10, K1 0.20, O1 0.10 m), currents | HARD |
| **Roshan et al. 2025** (*Hydrology*, MDPI) | Periyar ≈33% of ~19,000 Mm³/yr river discharge | HARD |
| Barrass / PIANC / Krone / Van Rijn | squat, UKC, cohesive deposition, harbour siltation methods | method refs |

### Official / government data
| Source | What it provides | Status |
|---|---|---|
| Cochin Port Administration Reports (`cochinport.gov.in/annual-report`) | annual dredged volumes | HARD (survey) / contracted |
| Dredging Guidelines for Major Ports 2021 (`shipmin.gov.in`) | pre/post-survey volume method; 22.20/21.00/27.00 Mm³ table | HARD |
| DCI contract disclosures 2020-21…2024-25 | ~24 Mm³/yr, tender values (Rs 98.64–156.5 cr) | contracted |
| Sagarmala Master Plan for Cochin | channel/estuary morphology, geometry | HARD |
| Indian Infrastructure (industry) | 2019-20 = 25.21 Mm³ (highest major port); cost share | HARD |
| Sagar Samriddhi / NTCPWC (IIT Madras) | dredging-monitoring system (Cochin first adopter) — route to real survey labels | access-gated |

### Key anchor figures (used in the model)
- **Dredging record:** 22.20 (2016-17), 21.00 (2017-18), 25.21 (2019-20) Mm³ **[HARD]**; 24 Mm³/yr 2020-21…2024-25 **[contracted]**.
- **Channel geometry:** approach 16 km × 260 m × 14.5 m (→16 m target); Ernakulam 260 m × 13.25 m; Mattancherry 183 m × 9.75 m **[HARD]**.
- **Periyar discharge (PROXY):** annual mean ~200 m³/s; SW-monsoon mean ~390; flood 1,000–1,500; dry ~105. Authoritative source = CWC Neeleeswaram (not yet obtained).
- **Sediment (PROXY, no Cochin measurement):** D50 ~20 µm, w_s ~0.5 mm/s, τ_cr ~0.2 Pa, dry density ~450 kg/m³.

---

## 3. Full timeline (chronological)

### Phase 0 — Setup
WSL2 + conda `telemac` env; openTELEMAC v8p5r0 installed; GEE project provisioned.

### Phase 1 — Layer 1 (TELEMAC-2D) — COMPLETE
1. Built Selafin mesh (35,364 nodes / 69,598 triangles) for the Kochi estuary/coast.
2. Set liquid (sea) boundary + tidal forcing via `kochi_liq.txt` (real Cochin amplitudes; phases 0° placeholder).
3. **Reprojected mesh to UTM 43N** — the single fix that resolved all CFL instability; solver stored the mesh in a **local origin-shifted frame**.
4. 24-h tidal run → `CORRECT END OF RUN`; output `kochi_tidal_r.slf`.

### Phase 2 — Layer 2 (satellite SPM fusion) — COMPLETE
1. Downloaded 223 Sentinel-2 + Landsat-9 SPM tiles (Drive folder zip).
2. `08_temporal_kriging_fusion.py`: grouped by sensor+date, mosaicked (223→82), reprojected to mesh CRS, one global temporal variogram.
3. `09_clip_to_port_aoi.py` + `cleanup_port_spm.py`: clipped to port AOI, clamped negatives, masked land.
4. Output `port_spm_stack.nc` (13×112×133); monsoon cycle reproduced.

### Phase 3 — Layer 3 coupling & sub-models — THIS WORK (session by session)
1. **First-order estimator** `10_first_order_siltation_volume.py` — hit the "no-deposition wall" (uniform placeholder τ_b > τ_cr → p_dep=0 everywhere).
2. **τ_b extraction v1** `11_extract_taub_from_telemac.py` — per-pixel tidally-averaged Krone p_dep from the velocity field. First run: **flat p_dep≈1** (coordinate mismatch).
3. **Coordinate diagnosis** — mesh in local frame (x:1–23040) vs SPM in true UTM (x:629300–642600); every AOI pixel fell to one stagnant nearest node.
4. **Offset recovery** — SLF stored no origin (Option A failed); recovered `(x0,y0)=(629505,1079516)` by reprojecting degree-mesh `kochi_r.slf` (Option B); verified identical node order (`std≈0`). Re-run: **82% of pixels gated** — wall resolved.
5. **y-registration scare** — briefly suspected 15 km y-error; disproven (misread mesh southern edge; node order confirmed identical). Enclosure warning traced to a cosmetic ~200 m western sliver.
6. **Decision: strengthen the hydrodynamic lever (Option B)** — add river/monsoon forcing.
7. **Boundary structure discovery** — mesh has **one liquid boundary (the sea)**; no river inflow exists; Periyar inlet is a broad ~9 m basin.
8. **Discharge data** gathered (Periyar monthly Q proxy series).
9. **Route diagnosis** `12_locate_river_inflow.py` → Route A (walled river edge to open).
10. **Periyar segment** `13_periyar_segment.py` — exposed the **`.cli` K-column bug** (trailing 0/1 is a color tag, not rank); corrected to line-order / IPOBO (verified consecutive 1..1128).
11. **Consolidated pipeline v1** `haine_layer3_4_pipeline.py` — river folded in **parametrically**; ran end-to-end (raw 389.6 → pinned 24 Mm³; ×0.06 factor exposed as unphysical).
12. **Honesty checkpoint** — confirmed it is a *calibrated physical estimate*, not a trained ML model.
13. **Research** for feasible corrections (per-zone split, actuals, Ferket rate — confirmed no public per-zone m³; Ferket used waves).
14. **Corrected pipeline v2** `haine_layer3_4_pipeline_v2.py` — applied **C1–C4** (below); ran clean.
15. **Documents** — `HAINE_Layer3_4_Status_and_Corrections.md`, `HAINE_Prediction_Validity.md`.
16. **Deck** — `HAINE_Cochin_Dredging.pptx` (15 slides, validated).

---

## 4. Every problem & correction (catalog)

### Layer 1
| # | Problem | Root cause | Fix |
|---|---|---|---|
| L1-1 | Every timestep CFL-unstable | mesh in geographic degrees | reproject to **UTM 43N metres** — fixed all instability at once |
| L1-2 | Boundary numbering wrong | node-set heuristic for IPOBO | raw **edge-counting** (edges appearing once) |
| L1-3 | Liquid boundary read silently failing | strict Conlim/liq format | exact header `T SL(1)`, units line, KENT=5 for elevation |

### Layer 2
| # | Problem | Root cause | Fix |
|---|---|---|---|
| L2-1 | `gdown --id` broken | removed in gdown 6.x | URL form `gdown "…uc?id=…"`; folder-zip for bulk |
| L2-2 | Apparent duplicate tiles `name(1).tif` | Google zip renaming | they are **different tiles** — mosaic (nanmean), not dedupe |
| L2-3 | Whole-coast output (143×111 km) | script 07 extraction AOI was a huge rectangle | clip to the small port AOI |
| L2-4 | Negative SPM values | kriging undershoot | clamp to 0 (6 pixel-months) |
| L2-5 | Land pixels polluting SPM | Nechad unstable over land | temporal land-mask heuristic (median/cap rule) |

### Recurring workflow
| # | Problem | Fix |
|---|---|---|
| W-1 | Scripts fail on CRLF | `sed -i 's/\r$//'` before every run |
| W-2 | `conda activate` swallowed → runs in `(base)` | verify `(telemac)` prompt; activate on its own line |
| W-3 | Pasted Python block executed in bash | run copy/activate/run lines **one at a time** |

### Layer 3 — coupling
| # | Problem | Root cause | Fix |
|---|---|---|---|
| L3-1 | "No-deposition wall" (zero everywhere) | uniform placeholder `U_REP` → τ_b=0.314 > τ_cr=0.2 | per-pixel **tidally-averaged** Krone p_dep from TELEMAC (slack-water gives p>0) |
| L3-2 | Flat p_dep≈1 over AOI | mesh local-frame vs SPM true-UTM → griddata sampled one stagnant node | recover & add UTM offset |
| L3-3 | SLF stores no origin | offset dropped when result written | recover `(629505,1079516)` from degree-mesh reprojection (Option B) |
| L3-4 | Suspected 15 km y-error | misread mesh southern edge as the port | disproven — node order identical (`dx/dy std≈0`); offset correct |
| L3-5 | Enclosure warning | SPM left edge 206 m west of mesh | cosmetic sliver → nearest-node fill; negligible |
| L3-6 | No river forcing possible | mesh has only the sea liquid boundary | Route A identified; then folded river in **parametrically** (broad basin made solver run heavy) |
| L3-7 | `.cli` "K=0/1" collapse | trailing column is a **color tag**, not the boundary rank | use **line order / IPOBO** (verified consecutive 1..1128); node number = column `[-2]` |

### Layer 3 — Sub-model A corrections (v1 → v2)
| # | Problem | Correction |
|---|---|---|
| **C1** | Raw volume 389.6 vs 24 Mm³ (×16) — integrated over whole AOI incl. non-dredged backwater | integrate over the **dredged footprint** only; report non-dredged estuary (~11 Mm³) separately |
| **C2** | Zone ratios wrong (Mattancherry 5.14 vs Ferket ~2 m; basins 0.82 vs ~5 m) | **Ferket-anchored** disaggregation (thickness × area); satellite textures within zones |
| **C3** | Single ×0.06 pin (implied BED_ENHANCE 0.12, impossible) | **decompose** into dredged-footprint fraction + interpretable residual factor |
| **C4** | No year-by-year comparison; single-year illusion | print estimate vs 2016–2025 record; flag that 1 SPM year can't fit inter-annual variation |

### Deliverables / tooling
| # | Problem | Fix |
|---|---|---|
| D-1 | `f-string: unmatched '['` (nested quotes) | replace inline f-string ternary with plain string concatenation |
| D-2 | pptx build `MODULE_NOT_FOUND` after sandbox reset | reinstall pptxgenjs; set `NODE_PATH`; recreate build script |

---

## 5. Current state (honest)

- **Solid / independent:** Layer 1 tides, Layer 2 SPM, the coupling geometry & physics, the UKC calculator, and the Layer 4 dashboard/report framework.
- **Anchored (pinned/imposed, not predicted):** the ~24 Mm³/yr total (calibrated to record) and the zone split (Ferket-imposed).
- **Indicative:** per-pixel accretion pattern (relative hotspots trustworthy; absolute rate not).
- **Weak / incomplete:** absolute deposition magnitude, channel scour (tidal-only, no waves/solver-river), and any non-calibration year.

**Bottom line:** a sound, literature-consistent **decision-support baseline**, not yet a predictive siltation model.

---

## 6. Future work — data needed & why

### Decisive (turn baseline → trained model)
| Need | Why | Route |
|---|---|---|
| **CPA paired pre/post-dredge surveys** | true per-location labels → trainable regression (removes the calibration pin) | `philo@cochinport.gov.in`, `secretary@cochinport.gov.in`, or **Sagar Samriddhi** |
| **Multiple years of SPM cubes** (2016–2020) | inter-annual predictor → real year-to-year skill vs the 22.2/21.0/25.21 record | re-run Layer 2 |

### Accuracy upgrades
| Need | Why | Action |
|---|---|---|
| **TOMAWAC wave forcing** | monsoon wave resuspension is a primary Cochin driver (Ferket) — missing from tidal-only Layer 1 | couple TOMAWAC (ERA5 Hs/wind) |
| **Solver-coupled river run** | real channel scour instead of the parametric proxy | open Periyar flowrate boundary (nodes preserved: `31940,32022,32103,32184,32265,32264,31371`), re-run, re-run `11_extract_taub` on `kochi_monsoon_r.slf` |
| **Measured Cochin sediment** (D50, w_s, τ_cr) | each ±20% moves raw volume ±40% (sensitivity) | Ferket full report / Rasheed 1997 / fresh lab analysis |
| **FES2014 tidal phases** | replace 0° placeholder; validate vs tide gauge | AVISO registration |
| **CWC Neeleeswaram discharge** | replace ERA5-apportionment Periyar proxy | CWC data request |

### Model & scale-out
- Train Sub-model A as ML once labels exist; add **SHAP** explainability (currently a sensitivity tornado stands in).
- Remaining Layer 3 sub-models: tool selection (built rule-based), UKC (built), post-dredge quality (built) — upgrade with real data.
- **Multi-port extension:** Paradip, Kandla (different siltation regimes) → IOR / MAHASAGAR deployment.

---

## 7. Key file inventory
**Layer 1:** `kochi.slf`, `kochi_r.slf` (degree mesh), `kochi.cli`, `kochi_liq.txt`, `kochi_t2d_tidal.cas`, `kochi_tidal_r.slf`, `reproject_mesh_to_utm.py`
**Layer 2:** `08_temporal_kriging_fusion.py`, `09_clip_to_port_aoi.py`, `cleanup_port_spm.py`, `view_layer2.py`, `port_spm_stack.nc`, `port_spm_<date>.tif`
**Layer 3:** `10_first_order_siltation_volume.py`, `11_extract_taub_from_telemac.py`, `12_locate_river_inflow.py`, `13_periyar_segment.py`, `haine_layer3_4_pipeline.py`, `haine_layer3_4_pipeline_v2.py`, `port_pdep_tidal.tif`, `port_taub_mean.tif`, `siltation_accretion_v2.tif`
**Deliverables:** `HAINE_Layer3_4_Status_and_Corrections.md`, `HAINE_Prediction_Validity.md`, `haine_results_v2.json`, `haine_report.html`, `HAINE_Cochin_Dredging.pptx`, `HAINE_Master_Log.md` (this file)

### Critical continuity notes
- Mesh is in a **local frame**; add `(629505, 1079516)` to get true UTM before any SLF↔SPM sampling.
- `.cli`: node number = column `[-2]`; boundary rank = line order / IPOBO (not the trailing color tag).
- Never `find /` (crawls the Windows drive); always `sed` CRLF; always confirm `(telemac)` prompt.
