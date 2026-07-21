"""
HAINE — Layer 2 Satellite Fusion
Temporal kriging fusion: turns a stack of gap-riddled, irregularly-dated
SPM rasters (from 07_satellite_spm_turbidity_extraction.py) into a
regular, gap-free time series per pixel.

This is the "temporal kriging" step named explicitly in your Layer 2
architecture — spatial kriging (used in the SDB pipeline) fills gaps
within one image; this fills gaps ACROSS dates for a given location.

Requires: rasterio, numpy, pandas, scipy, pykrige
          pip install rasterio numpy pandas scipy pykrige --break-system-packages

Inputs:
  - INPUT_DIR   : folder of per-date GeoTIFFs from Script 07, named like
                  s2_spm_2025-01-15.tif / l9_spm_2025-01-20.tif
  - TARGET_DATES: the regular date grid you want output on, e.g. monthly

Output:
  - fused_spm_<date>.tif : one GeoTIFF per target date, gap-filled
  - fused_spm_stack.nc   : all target dates combined into one NetCDF cube
                           (time, y, x) — this is what Layer 3 should
                           actually ingest as a feature.
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    raise SystemExit("Install pykrige first: pip install pykrige --break-system-packages")

try:
    import netCDF4
except ImportError:
    netCDF4 = None


# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------
INPUT_DIR = "./spm_rasters"
OUTPUT_DIR = "./spm_fused"
TARGET_DATES = pd.date_range("2024-06-01", "2025-06-01", freq="MS")  # monthly targets
KRIGE_VARIOGRAM_MODEL = "spherical"
MIN_VALID_OBS_PER_PIXEL = 3   # pixels with fewer real observations across
                                # the whole record are left as NaN rather
                                # than krige-extrapolated from near-nothing


DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def load_stack(input_dir):
    """Load every dated GeoTIFF into a (date, array) list, plus geo metadata."""
    files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not files:
        raise SystemExit(f"No GeoTIFFs found in {input_dir}")

    dates = []
    arrays = []
    transform = crs = shape = None

    for f in files:
        match = DATE_PATTERN.search(os.path.basename(f))
        if not match:
            print(f"Skipping {f} — no date found in filename.")
            continue
        date = datetime.strptime(match.group(1), "%Y-%m-%d")
        with rasterio.open(f) as src:
            arr = src.read(1).astype(float)
            arr[arr == src.nodata] = np.nan
            if transform is None:
                transform, crs, shape = src.transform, src.crs, arr.shape
        dates.append(date)
        arrays.append(arr)

    stack = np.stack(arrays, axis=0)  # (n_dates, y, x)
    return np.array(dates), stack, transform, crs, shape


def krige_pixel_timeseries(obs_days, obs_values, target_days):
    """
    1D "kriging" across time for a single pixel. pykrige's OrdinaryKriging
    needs 2 spatial dims, so we fake a second dimension (constant y=0) —
    a standard trick for treating 1D kriging as a degenerate 2D case.
    """
    valid = ~np.isnan(obs_values)
    if valid.sum() < MIN_VALID_OBS_PER_PIXEL:
        return np.full(len(target_days), np.nan)

    x = obs_days[valid].astype(float)
    y = np.zeros(valid.sum())
    z = obs_values[valid]

    try:
        ok = OrdinaryKriging(
            x, y, z, variogram_model=KRIGE_VARIOGRAM_MODEL,
            verbose=False, enable_plotting=False,
        )
        pred, _ = ok.execute("points", target_days.astype(float), np.zeros(len(target_days)))
        return pred
    except Exception:
        # degenerate variogram (e.g. all obs on ~same day) — fall back to
        # simple forward/backward fill
        return np.interp(target_days, x, z)


def fuse(dates, stack, target_dates):
    ref_day0 = dates.min()
    obs_days = np.array([(d - ref_day0).days for d in dates])
    target_days = np.array([(pd.Timestamp(d) - ref_day0).days for d in target_dates])

    n_t, ny, nx = len(target_dates), stack.shape[1], stack.shape[2]
    fused = np.full((n_t, ny, nx), np.nan)

    total_px = ny * nx
    done = 0
    for i in range(ny):
        for j in range(nx):
            series = stack[:, i, j]
            if np.all(np.isnan(series)):
                continue
            fused[:, i, j] = krige_pixel_timeseries(obs_days, series, target_days)
            done += 1
        if i % max(1, ny // 10) == 0:
            print(f"  ...row {i}/{ny}")

    return fused


def write_outputs(fused, target_dates, transform, crs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    profile = dict(
        driver="GTiff", height=fused.shape[1], width=fused.shape[2],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=np.nan,
    )

    for t, date in enumerate(target_dates):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        path = os.path.join(output_dir, f"fused_spm_{date_str}.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(fused[t].astype("float32"), 1)

    print(f"Wrote {len(target_dates)} fused GeoTIFFs to {output_dir}")

    if netCDF4 is not None:
        nc_path = os.path.join(output_dir, "fused_spm_stack.nc")
        with netCDF4.Dataset(nc_path, "w") as ds:
            ds.createDimension("time", fused.shape[0])
            ds.createDimension("y", fused.shape[1])
            ds.createDimension("x", fused.shape[2])
            time_var = ds.createVariable("time", "f8", ("time",))
            spm_var = ds.createVariable("spm_mgL", "f4", ("time", "y", "x"), fill_value=np.nan)
            time_var.units = "days since 1970-01-01"
            time_var[:] = [(pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days for d in target_dates]
            spm_var[:] = fused
        print(f"Wrote {nc_path} — load this stack directly in Layer 3.")
    else:
        print("netCDF4 not installed — skipped combined stack export "
              "(pip install netCDF4 --break-system-packages)")


def main():
    print(f"Loading raster stack from {INPUT_DIR} ...")
    dates, stack, transform, crs, shape = load_stack(INPUT_DIR)
    print(f"Loaded {len(dates)} dated rasters, shape {shape}.")

    print(f"Fusing onto {len(TARGET_DATES)} target dates via temporal kriging ...")
    fused = fuse(dates, stack, TARGET_DATES)

    write_outputs(fused, TARGET_DATES, transform, crs, OUTPUT_DIR)


if __name__ == "__main__":
    main()
