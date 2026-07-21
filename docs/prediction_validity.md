# HAINE — Prediction Validity Map
**What the model gets right, where it starts to deviate from reality, and why.**
Read this alongside the dashboard and `haine_results_v2.json`. The guiding principle: a HAINE output is only as trustworthy as its weakest upstream input, so each prediction below is rated by the quality of what feeds it.

Legend: **[SOLID]** independently correct · **[ANCHORED]** correct because pinned/imposed from a trusted source (not independently predicted) · **[INDICATIVE]** right shape, uncertain magnitude · **[WEAK]** structurally incomplete.

---

## 1. The validity ladder — correct → deviating

### [SOLID] Fully trustworthy (independent of the siltation calibration)

**Tidal hydrodynamics (Layer 1).** The tidal field is a genuine solve on a correct mesh; free surface oscillates −0.38…+0.63 m with proper damping/lag. Constants match DRDO-NPOL (M2≈0.24 m). *Correct.*

**Satellite SPM (Layer 2).** The 13-month cube reproduces the monsoon siltation cycle independently of any dredging figure. The *spatial and temporal pattern of sediment supply* is real, sensor-derived data. *Correct.*

**Coupling geometry.** After the offset fix, the mesh and SPM grid are co-located to the metre; `p_dep` and `tau_b` are sampled from the right ground. *Correct.*

**UKC calculator.** Independent of the siltation model entirely. Charted depths, tidal constants, and Barrass squat are all HARD inputs. Outputs — approach channel needs **+1.6 m tide** for a 14.5 m-draft vessel, Ernakulam **+0.95 m**, Mattancherry clears — are directly usable for tidal-window planning. *Correct within the squat-formula and spring-tide assumptions.*

### [ANCHORED] Correct, but because it is pinned/imposed — not predicted

**Annual siltation total (~24 Mm³/yr).** Matches the record **by construction** — the pipeline calibrates to it. This is a *consistency check that the physics can be scaled to reality*, not a forecast. It also sits sensibly against Ferket's independent +20 Mm³/yr. Trust the number as a calibrated baseline; do **not** present it as a model that discovered 24. *Anchored.*

**Per-zone split (approach > basins > Ernakulam ≈ Mattancherry).** In v2 the zone volumes are **imposed from Ferket** (thickness × area), so they agree with the literature *by definition*. The satellite adds within-zone texture. Trust the *ordering and rough proportions* (they come from a peer study); do **not** treat the split as an independent HAINE result. *Anchored.*

### [INDICATIVE] Right shape, uncertain magnitude — deviation begins here

**Per-pixel accretion map.** The *pattern* — where within each zone siltation concentrates — is driven by real SPM and real tidal gating, so the hot/cold structure is informative. But the *absolute metres per year* per pixel inherit the whole proxy chain and the residual calibration factor, so individual pixel values can be off by a large factor. Trust *relative* hotspots; distrust the absolute depth-reduction at a point. **This is where the model transitions from correct to indicative.**

**Monthly siltation seasonality.** Follows the SPM cycle (real) modulated by the river-Q proxy. The *timing* (monsoon-weighted) is credible; the *month-to-month magnitudes* depend on the parametric river term and the same proxies. Trust the seasonal shape, not the monthly tonnage.

### [WEAK] Structurally incomplete — expect real deviation

**Absolute deposition physics (pre-calibration).** Raw gross deposition overshoots the record ~16× (v1) / and needs a residual ×-factor even over the dredged footprint (v2). The physics captures *relative* deposition but its *absolute* rate is unconstrained without labels. **Deviates strongly from reality in absolute terms; only the calibrated total and relative pattern survive.**

**Hydrodynamic lever over the port.** Tidal-only currents are very low over the AOI (`tau_b` median ~0.002 Pa ≪ τ_cr 0.2), so `p_dep` barely gates (0.97–1.0). The model is currently **supply-dominated** (SPM), with hydrodynamics nearly inert. Reality has stronger channel currents from **fluvial throughflow and waves**, both missing. So the model under-represents channel scour and the outer-channel wave-siltation component. **Deviates wherever hydrodynamics or waves matter — most importantly the outer/approach channel.**

**Inter-annual prediction.** Flat across years (one SPM year). Any claim about a *specific* year other than the calibration year is unsupported. **Deviates for all non-calibration years.**

---

## 2. Where, physically, deviation concentrates
- **Outer/approach channel:** most exposed to the missing **wave** driver (monsoon swell resuspension) and the missing **fluvial ebb** scour — the model's weakest region despite being the highest-siltation zone. Its volume is Ferket-anchored, so the *total* is fine, but the *process attribution* is incomplete.
- **Deep quiescent basins:** the tidal-only gating is most defensible here (genuinely low energy), so the *relative* pattern is more reliable.
- **Inner channels (Ernakulam/Mattancherry):** intermediate; some fluvial influence missing.

---

## 3. What would move each item up the ladder

| Prediction | Currently | Becomes SOLID when… |
|---|---|---|
| Annual total | ANCHORED (pinned) | multiple SPM years + record → learned, not pinned |
| Zone split | ANCHORED (Ferket-imposed) | CPA per-zone survey volumes → validated |
| Per-pixel accretion | INDICATIVE | CPA paired pre/post surveys → per-location labels |
| Channel scour | WEAK | TELEMAC river re-run + TOMAWAC waves |
| Inter-annual | WEAK | Layer 2 re-run for 2016–2020 |
| Sediment magnitude | WEAK | measured Cochin D50 / w_s / τ_cr |

---

## 4. One-line honest summary
**Correct:** tides, SPM, coupling, UKC, and the calibrated ~24 Mm³/yr baseline with a Ferket-consistent zone structure. **Starts deviating at:** absolute per-pixel rates, month-to-month magnitudes, channel scour, and any specific non-calibration year. **Reason for deviation:** the model is supply-driven and label-free — it lacks CPA survey labels (absolute calibration), multi-year SPM (inter-annual skill), and wave + solver-coupled river forcing (hydrodynamic completeness). Until those arrive it is a sound, literature-consistent *decision-support baseline*, not a predictive siltation model.
