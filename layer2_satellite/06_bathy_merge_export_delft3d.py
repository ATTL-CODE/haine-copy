"""
HAINE — Layer 1 Bathymetry Assembly
Step 3: Merge the smoothed SDB depth raster with any real survey points
        you have (Cochin Port Authority / Sagar Samriddhi / CWPRS / tender
        docs), letting real data override SDB, then export a Delft3D-FM
        compatible sample file plus a NetCDF for QGIS visualization.

Requires: rasterio, numpy, pandas, scipy, netCDF4 (or xarray)
          pip install rasterio numpy pandas scipy netCDF4 --break-system-packages

Inputs:
  - SDB_DEPTH_TIF     : output of 05_sdb_calibration_krige.py
  - SURVEY_CSV        : lon,lat,depth_m — real survey points (optional; if
                         missing/empty, output is SDB-only, clearly flagged)
  - MODEL_GRID_CSV    : lon,lat of your Delft3D-FM grid node locations
                         (export this from RGFGRID/your mesh generator first)

Output:
  - kochi_bathy_delft3d.xyz   : plain x y z sample file for Delft3D-FM
  - kochi_bathy_merged.nc     : georeferenced NetCDF for QGIS/inspection
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy.interpolate import griddata

try:
    import netCDF4
except ImportError:
    netCDF4 = None


# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------
SDB_DEPTH_TIF = "kochi_sdb_depth_smoothed.tif"
SURVEY_CSV = "survey_points.csv"          # set to None if you have nothing yet
MODEL_GRID_CSV = "model_grid_nodes.csv"   # columns: lon, lat
OVERRIDE_RADIUS_M = 150                   # real survey points override SDB within this radius
OUTPUT_XYZ = "kochi_bathy_delft3d.xyz"
OUTPUT_NC = "kochi_bathy_merged.nc"

# Sign convention: Delft3D-FM generally expects depth as a NEGATIVE
# elevation (bed level below datum) OR positive depth depending on your
# model's vertical reference setup. This script outputs POSITIVE depth
# below chart datum (m) — flip the sign in the final write step if your
# .mdu file expects bed level (negative) instead.


def load_sdb_raster(path):
    with rasterio.open(path) as src:
        depth = src.read(1).astype(float)
        depth[depth == src.nodata] = np.nan
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    lons = np.linspace(bounds.left, bounds.right, depth.shape[1])
    lats = np.linspace(bounds.top, bounds.bottom, depth.shape[0])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return depth, lon_grid, lat_grid, transform, crs


def apply_survey_override(depth, lon_grid, lat_grid, survey_df, transform):
    """Real survey points win within OVERRIDE_RADIUS_M of each point."""
    if survey_df is None or len(survey_df) == 0:
        print("No survey points provided — output is SDB-only. "
              "Treat channel/basin depths as unreliable (see pipeline notes).")
        return depth, np.zeros_like(depth, dtype=bool)

    override_mask = np.zeros_like(depth, dtype=bool)
    # crude degrees-to-metres conversion good enough at Kochi's latitude (~10N)
    deg_to_m_lat = 111_320
    deg_to_m_lon = 111_320 * np.cos(np.deg2rad(10.0))
    radius_deg_lat = OVERRIDE_RADIUS_M / deg_to_m_lat
    radius_deg_lon = OVERRIDE_RADIUS_M / deg_to_m_lon

    for _, row in survey_df.iterrows():
        dist2 = ((lon_grid - row["lon"]) / radius_deg_lon) ** 2 + (
            (lat_grid - row["lat"]) / radius_deg_lat
        ) ** 2
        nearby = dist2 <= 1.0
        if nearby.any():
            depth[nearby] = row["depth_m"]
            override_mask |= nearby

    print(f"Overrode {override_mask.sum()} cells with real survey data "
          f"({len(survey_df)} survey points, {OVERRIDE_RADIUS_M} m radius each).")
    return depth, override_mask


def interpolate_to_grid_nodes(depth, lon_grid, lat_grid, grid_nodes_df):
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    values = depth.ravel()
    valid = ~np.isnan(values)

    target = grid_nodes_df[["lon", "lat"]].values
    interpolated = griddata(points[valid], values[valid], target, method="linear")

    # fill any remaining NaN at domain edges with nearest-neighbor
    nan_mask = np.isnan(interpolated)
    if nan_mask.any():
        interpolated[nan_mask] = griddata(
            points[valid], values[valid], target[nan_mask], method="nearest"
        )

    grid_nodes_df = grid_nodes_df.copy()
    grid_nodes_df["depth_m"] = interpolated
    return grid_nodes_df


def write_delft3d_xyz(grid_nodes_df, path):
    with open(path, "w") as f:
        for _, row in grid_nodes_df.iterrows():
            f.write(f"{row['lon']:.6f} {row['lat']:.6f} {row['depth_m']:.3f}\n")
    print(f"Wrote {len(grid_nodes_df)} nodes to {path}")


def write_netcdf(depth, lon_grid, lat_grid, path):
    if netCDF4 is None:
        print("netCDF4 not installed — skipping NetCDF export "
              "(pip install netCDF4 --break-system-packages)")
        return
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("y", depth.shape[0])
        ds.createDimension("x", depth.shape[1])
        lon_var = ds.createVariable("lon", "f4", ("y", "x"))
        lat_var = ds.createVariable("lat", "f4", ("y", "x"))
        depth_var = ds.createVariable("depth_m", "f4", ("y", "x"), fill_value=np.nan)
        lon_var[:] = lon_grid
        lat_var[:] = lat_grid
        depth_var[:] = depth
        depth_var.long_name = "Water depth below chart datum"
        depth_var.units = "m"
    print(f"Wrote {path}")


def main():
    depth, lon_grid, lat_grid, transform, crs = load_sdb_raster(SDB_DEPTH_TIF)

    survey_df = None
    if SURVEY_CSV:
        try:
            survey_df = pd.read_csv(SURVEY_CSV)
        except FileNotFoundError:
            print(f"{SURVEY_CSV} not found — proceeding SDB-only.")

    depth, override_mask = apply_survey_override(depth, lon_grid, lat_grid, survey_df, transform)

    write_netcdf(depth, lon_grid, lat_grid, OUTPUT_NC)

    grid_nodes_df = pd.read_csv(MODEL_GRID_CSV)
    result = interpolate_to_grid_nodes(depth, lon_grid, lat_grid, grid_nodes_df)
    write_delft3d_xyz(result, OUTPUT_XYZ)

    print(
        "\nDone. Reminder: this bathymetry is calibrated placeholder data "
        "wherever it wasn't overridden by real survey points. Re-run this "
        "script the moment more Cochin Port Authority / Sagar Samriddhi "
        "data arrives — it's a five-minute update, not a rebuild."
    )


if __name__ == "__main__":
    main()
