# HAINE — All Data Extraction Pipelines (Non-ERA5)

This covers everything left in your extraction stack besides ERA5: satellite
turbidity/SPM for Layer 2 fusion, the temporal kriging step that actually
makes Layer 2 "fusion" rather than just raw imagery, and FES2014 tidal
constituent extraction for Layer 1 boundary conditions. Satellite-derived
bathymetry (SDB) was already delivered separately — this set is everything
around it.

Since you're on Delft3D-FM built with Intel oneAPI and have QGIS: QGIS is
where you'll visually sanity-check every raster/NetCDF this pipeline
produces before trusting it in the model (Crayfish plugin handles the
NetCDF outputs directly). Delft3D-FM itself only consumes the final
outputs of Script 09 (tidal `.bc`/`.cmp` files) and the bathymetry `.xyz`
from the earlier pipeline — everything else here feeds Layer 2/3, not
directly into the solver.

---

## Pipeline map

| Script | Purpose | Feeds |
|---|---|---|
| `07_satellite_spm_turbidity_extraction.py` | Sentinel-2 + Landsat-9 suspended sediment / turbidity retrieval | Layer 2 |
| `08_temporal_kriging_fusion.py` | Fills cloud/gap gaps across time per pixel, turns snapshots into a continuous turbidity time series | Layer 2 → Layer 3 |
| `09_fes2014_tidal_extraction.py` | Extracts amplitude/phase for 34 tidal constituents at your model boundary nodes from a downloaded FES2014 grid | Layer 1 (Delft3D-FM boundary forcing) |

Run order: 07 → 08 independently of 09. Script 09 only runs once your AVISO
FES2014 download completes — it doesn't depend on 07/08 at all.

---

## Step 1 — Satellite turbidity/SPM (`07_satellite_spm_turbidity_extraction.py`)

Uses the **Nechad et al. (2010) single-band algorithm** — the standard
approach also used in your literature review's Sentinel-2 MSI / Landsat-9
OLI SPM retrieval. Unlike bathymetry, turbidity retrieval works fine in
deep water and doesn't have the optical-depth-limit problem SDB has — this
is genuinely useful across your whole domain, monsoon plumes included.

Requires the same `earthengine-api` setup as the SDB pipeline. Pulls both
Sentinel-2 (10m, higher revisit) and Landsat-9 (30m, longer historical
record) so you have both fine spatial detail and a longer baseline to work
with once you extend past a single season.

## Step 2 — Temporal kriging fusion (`08_temporal_kriging_fusion.py`)

This is the actual "fusion with temporal kriging" your Layer 2 architecture
calls for. Individual satellite passes have gaps — clouds, monsoon cover,
orbital revisit gaps. Rather than just interpolating spatially (like the
SDB gradient-fix did), this treats **each pixel's own time series** as the
signal to krige: for a given location, you have SPM values on whatever
dates had clear imagery, and you krige across the time axis to estimate
what the value would have been on the missing dates. Output is a regular,
gap-free time series per pixel — this is what Layer 3's siltation
sub-models actually want as an input feature, not raw scattered scenes.

## Step 3 — FES2014 tidal extraction (`09_fes2014_tidal_extraction.py`)

Once your AVISO download of `ocean_tide_extrapolated` lands (2.2 GB,
34 constituents), this script samples amplitude and phase at your
Delft3D-FM open-boundary node coordinates and writes them out in the
`.cmp` component format the model's tidal forcing module expects. You'll
need your boundary node lon/lat list — export this from RGFGRID/your grid
tool the same way you'll do for the bathymetry `.xyz` file.

**This script cannot run until the AVISO data exists locally** — there's
no substitute or placeholder version worth building for tides the way
GEBCO stood in for bathymetry; a wrong tidal phase is worse than no tidal
forcing at all for a channel this tide-dominated.

---

## What's still not covered by any script

- **Periyar river discharge** — you already have the ERA5-precipitation +
  runoff-coefficient approximation from earlier; no separate extraction
  script needed unless CWC data comes through, in which case it's a simple
  CSV load, not worth a dedicated pipeline.
- **Sediment properties** (D50, settling velocity, critical shear stress) —
  literature defaults already established; these aren't something to
  "extract" programmatically, they come from the NIO/CSIR literature or
  eventual lab analysis of real samples.
- **Dredging cost/tender data** — this lives on eprocure.gov.in and similar
  procurement portals as scattered PDF tenders, not a queryable API. That's
  a manual document-mining task, not an automatable extraction script —
  flag it if you want help parsing specific tender PDFs once you have them.
