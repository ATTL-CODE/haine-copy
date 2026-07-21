# Layer 2 — Satellite SPM & Bathymetry Fusion

Builds the gap-free monthly suspended-particulate-matter (SPM) cube over the port AOI.

- `04_sdb_extraction_gee.py`, `05_sdb_calibration_krige.py`, `06_bathy_merge_export_delft3d.py` — satellite-derived bathymetry (Stumpf log-ratio), kriged calibration to GEBCO, merged bathy export.
- `07_satellite_spm_turbidity_extraction.py` — Nechad single-band SPM from Sentinel-2 / Landsat via Google Earth Engine.
- `07b_multiyear_spm_extraction.py` — multi-year extraction (Landsat-8 substituted for pre-2021 years).
- `08_temporal_kriging_fusion.py` — temporal kriging into monthly fields. **Use ONE global variogram** — per-pixel refit causes SIGKILL.

**Output:** `port_spm_stack.nc` (13×112×133, EPSG:32643) + monthly `port_spm_YYYY-MM.tif`. Validation: the monthly median independently reproduces Cochin's monsoon siltation cycle.
