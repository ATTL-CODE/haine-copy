"""
HAINE — Layer 2 Satellite Fusion
Step 2: Calibrate the Stumpf band-ratio raster into real depths, then
        detect and smooth non-gradual / physically implausible patches
        (sun glint, turbidity fronts, boat wakes) using ordinary kriging.

Requires: rasterio, numpy, pandas, scipy, pykrige
          pip install rasterio numpy pandas scipy pykrige --break-system-packages

Inputs:
  - RATIO_TIF          : GeoTIFF from 04_sdb_extraction_gee.py
  - CALIBRATION_CSV     : lon,lat,depth_m of known depth points
                           (use gebco_placeholder_calibration() below if you
                           don't have real points yet — see 00_SDB_PIPELINE_STEPS.md
                           for why this is a weak substitute)

Output:
  - kochi_sdb_depth_smoothed.tif : calibrated + smoothed depth raster
  - kochi_sdb_qa_mask.tif        : 0 = direct calibration, 1 = kriged-in
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy import ndimage
from scipy.stats import linregress

try:
    from pykrige.ok import OrdinaryKriging
except ImportError:
    raise SystemExit("Install pykrige first: pip install pykrige --break-system-packages")


# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------
RATIO_TIF = "kochi_sdb_ratio.tif"
CALIBRATION_CSV = "calibration_points.csv"          # columns: lon,lat,depth_m
OUTPUT_DEPTH_TIF = "kochi_sdb_depth_smoothed.tif"
OUTPUT_QA_TIF = "kochi_sdb_qa_mask.tif"

# Max plausible depth change per pixel step (metres). Tune to your channel
# slopes — this is what flags "non-gradual" cells for kriged re-interpolation.
MAX_GRADIENT_PER_PIXEL = 2.0

KRIGE_VARIOGRAM_MODEL = "spherical"   # standard choice for coastal bathymetry
KRIGE_NEIGHBOR_RADIUS_PX = 25          # how far to search for valid neighbors


def load_ratio_raster(path):
    with rasterio.open(path) as src:
        ratio = src.read(1).astype(float)
        ratio[ratio == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs
    return ratio, transform, crs


def sample_ratio_at_points(ratio, transform, lons, lats):
    rows, cols = rowcol(transform, lons, lats)
    values = []
    for r, c in zip(rows, cols):
        if 0 <= r < ratio.shape[0] and 0 <= c < ratio.shape[1]:
            values.append(ratio[r, c])
        else:
            values.append(np.nan)
    return np.array(values)


def fit_calibration(ratio, transform, calib_df):
    """Linear regression: depth = m * ratio + b"""
    sampled = sample_ratio_at_points(
        ratio, transform, calib_df["lon"].values, calib_df["lat"].values
    )
    valid = ~np.isnan(sampled) & ~np.isnan(calib_df["depth_m"].values)
    if valid.sum() < 5:
        raise ValueError(
            f"Only {valid.sum()} usable calibration points — need at least 5-10 "
            "spread across a depth range for a stable regression."
        )
    slope, intercept, r, p, stderr = linregress(
        sampled[valid], calib_df["depth_m"].values[valid]
    )
    print(f"Calibration fit: depth = {slope:.4f} * ratio + {intercept:.4f}")
    print(f"  R^2 = {r**2:.3f}, n = {valid.sum()} points")
    if r**2 < 0.5:
        print(
            "  WARNING: weak fit (R^2 < 0.5). Depth values downstream should be "
            "treated as indicative only. Recalibrate with better/more points."
        )
    return slope, intercept


def detect_nongradual_cells(depth):
    """
    Flag cells whose local gradient magnitude exceeds a physically
    implausible threshold — i.e. the "not gradual" patches referenced
    in the task (sun glint, turbidity fronts, wakes, artifacts).
    """
    gy, gx = np.gradient(depth)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    flagged = gradient_mag > MAX_GRADIENT_PER_PIXEL
    flagged |= np.isnan(depth)
    return flagged


def krige_fill(depth, flagged):
    """
    Re-interpolate flagged cells using ordinary kriging against their
    valid neighbors, producing a spatially-correlated smooth fill rather
    than a naive linear patch.
    """
    valid_rows, valid_cols = np.where(~flagged)
    valid_vals = depth[valid_rows, valid_cols]

    # Subsample valid points if huge, kriging does not scale to millions of points
    max_points = 3000
    if len(valid_vals) > max_points:
        idx = np.random.choice(len(valid_vals), max_points, replace=False)
        valid_rows, valid_cols, valid_vals = (
            valid_rows[idx],
            valid_cols[idx],
            valid_vals[idx],
        )

    ok = OrdinaryKriging(
        valid_cols,
        valid_rows,
        valid_vals,
        variogram_model=KRIGE_VARIOGRAM_MODEL,
        verbose=False,
        enable_plotting=False,
    )

    flagged_rows, flagged_cols = np.where(flagged)
    filled = depth.copy()

    # Krige in batches to keep memory sane
    batch = 2000
    for start in range(0, len(flagged_rows), batch):
        rr = flagged_rows[start:start + batch]
        cc = flagged_cols[start:start + batch]
        z, ss = ok.execute("points", cc.astype(float), rr.astype(float))
        filled[rr, cc] = z

    return filled


def gebco_placeholder_calibration(n_points=40):
    """
    Helper: if you have no real depth points yet, this stub shows the
    expected structure. Replace with an actual GEBCO extraction
    (e.g. via the GEBCO web service or a downloaded GEBCO grid sampled
    with rasterio) before running fit_calibration(). Treat any resulting
    depths as rough placeholders — see 00_SDB_PIPELINE_STEPS.md.
    """
    raise NotImplementedError(
        "Populate this with real GEBCO-sampled points or, better, real "
        "survey depths once Cochin Port Authority / Sagar Samriddhi data "
        "arrives. Expected output: DataFrame with columns lon, lat, depth_m."
    )


def main():
    ratio, transform, crs = load_ratio_raster(RATIO_TIF)
    calib_df = pd.read_csv(CALIBRATION_CSV)

    slope, intercept = fit_calibration(ratio, transform, calib_df)
    depth = slope * ratio + intercept

    flagged = detect_nongradual_cells(depth)
    pct_flagged = 100 * flagged.sum() / flagged.size
    print(f"Flagged {pct_flagged:.1f}% of cells as non-gradual / invalid — kriging these in.")

    smoothed = krige_fill(depth, flagged)

    profile_kwargs = dict(
        driver="GTiff",
        height=smoothed.shape[0],
        width=smoothed.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    )

    with rasterio.open(OUTPUT_DEPTH_TIF, "w", **profile_kwargs) as dst:
        dst.write(smoothed.astype("float32"), 1)

    with rasterio.open(OUTPUT_QA_TIF, "w", **profile_kwargs) as dst:
        dst.write(flagged.astype("float32"), 1)

    print(f"Wrote {OUTPUT_DEPTH_TIF} and {OUTPUT_QA_TIF}")


if __name__ == "__main__":
    main()
