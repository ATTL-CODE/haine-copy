"""
HAINE — Layer 1 Boundary Conditions
Extract FES2014 tidal constituent amplitude/phase at Delft3D-FM open
boundary node locations, and write out in Delft3D-FM's .cmp component
format for tidal forcing.

Cannot run until you have a local FES2014 download from AVISO
(ocean_tide_extrapolated, ~2.2 GB). No placeholder substitute is provided
for tides — see 00b_OTHER_EXTRACTION_OVERVIEW.md for why.

Requires: netCDF4 (or xarray), numpy, pandas
          pip install netCDF4 numpy pandas --break-system-packages

Inputs:
  - FES2014_DIR      : folder containing per-constituent NetCDF files,
                        e.g. m2.nc, s2.nc, k1.nc, o1.nc, ... (FES2014's
                        standard distribution layout — one file per
                        constituent, each with amplitude + phase grids)
  - BOUNDARY_NODES_CSV: lon,lat of your Delft3D-FM open boundary nodes,
                        same format/source as the bathymetry grid nodes

Output:
  - kochi_tides.cmp   : Delft3D-FM tidal component file, one block per
                        boundary node, ready to reference from your .bnd
                        / .ext file
"""

import glob
import os

import numpy as np
import pandas as pd

try:
    import netCDF4
except ImportError:
    raise SystemExit("Install netCDF4 first: pip install netCDF4 --break-system-packages")


# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------
FES2014_DIR = "./fes2014/ocean_tide_extrapolated"
BOUNDARY_NODES_CSV = "boundary_nodes.csv"   # columns: node_id, lon, lat
OUTPUT_CMP = "kochi_tides.cmp"

# The major constituents worth keeping for a coastal domain like Kochi.
# FES2014 provides 34 total; start with these, add more if validation
# against real tide gauge data (once INCOIS responds) shows you need them.
CONSTITUENTS = ["m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1"]


def find_constituent_file(constituent):
    matches = glob.glob(os.path.join(FES2014_DIR, f"{constituent}*.nc"))
    if not matches:
        raise FileNotFoundError(
            f"No file found for constituent '{constituent}' in {FES2014_DIR}. "
            "Check your AVISO download completed and the folder layout matches."
        )
    return matches[0]


def bilinear_sample(grid_lon, grid_lat, grid_values, lon, lat):
    """Simple bilinear interpolation onto an arbitrary point from a regular grid."""
    i = np.searchsorted(grid_lon, lon) - 1
    j = np.searchsorted(grid_lat, lat) - 1
    i = np.clip(i, 0, len(grid_lon) - 2)
    j = np.clip(j, 0, len(grid_lat) - 2)

    x0, x1 = grid_lon[i], grid_lon[i + 1]
    y0, y1 = grid_lat[j], grid_lat[j + 1]
    tx = (lon - x0) / (x1 - x0) if x1 != x0 else 0
    ty = (lat - y0) / (y1 - y0) if y1 != y0 else 0

    v00 = grid_values[j, i]
    v10 = grid_values[j, i + 1]
    v01 = grid_values[j + 1, i]
    v11 = grid_values[j + 1, i + 1]

    if np.isnan([v00, v10, v01, v11]).any():
        # nearest-neighbor fallback near coastal masked cells
        candidates = [v00, v10, v01, v11]
        candidates = [v for v in candidates if not np.isnan(v)]
        return np.nan if not candidates else np.mean(candidates)

    return (
        v00 * (1 - tx) * (1 - ty)
        + v10 * tx * (1 - ty)
        + v01 * (1 - tx) * ty
        + v11 * tx * ty
    )


def load_constituent_grid(constituent):
    path = find_constituent_file(constituent)
    with netCDF4.Dataset(path) as ds:
        lon = ds.variables["lon"][:]
        lat = ds.variables["lat"][:]
        # FES2014 standard variable names; amplitude in cm, phase in degrees
        amplitude = ds.variables["amplitude"][:]
        phase = ds.variables["phase"][:]
    return lon, lat, amplitude, phase


def extract_all_constituents(boundary_df):
    results = {node_id: {} for node_id in boundary_df["node_id"]}

    for constituent in CONSTITUENTS:
        print(f"Loading {constituent.upper()} ...")
        lon, lat, amp_grid, phase_grid = load_constituent_grid(constituent)

        for _, row in boundary_df.iterrows():
            amp = bilinear_sample(lon, lat, amp_grid, row["lon"], row["lat"])
            phase = bilinear_sample(lon, lat, phase_grid, row["lon"], row["lat"])
            results[row["node_id"]][constituent] = (amp, phase)

    return results


def write_cmp_file(results, boundary_df, path):
    """
    Delft3D-FM .cmp format: one file per boundary point in practice, but
    this writes a combined reference file with clear per-node sections —
    split into individual <node_id>.cmp files if your .bnd setup expects
    that instead (uncomment the per-node write loop below).
    """
    with open(path, "w") as f:
        for _, row in boundary_df.iterrows():
            node_id = row["node_id"]
            f.write(f"# Boundary node: {node_id} ({row['lon']:.5f}, {row['lat']:.5f})\n")
            f.write("* COMPONENT     AMPLITUDE   PHASE\n")
            for constituent in CONSTITUENTS:
                amp, phase = results[node_id][constituent]
                if np.isnan(amp) or np.isnan(phase):
                    print(f"  WARNING: {node_id} / {constituent.upper()} — "
                          f"no valid data (likely too close to coast mask). "
                          f"Consider nearest valid offshore point instead.")
                    continue
                amp_m = amp / 100.0  # FES2014 amplitude is in cm -> convert to m
                f.write(f"{constituent.upper():<12} {amp_m:>10.4f}  {phase:>8.2f}\n")
            f.write("\n")

    print(f"Wrote {path}")

    # Uncomment if your Delft3D-FM setup wants one .cmp file per node instead:
    # for _, row in boundary_df.iterrows():
    #     node_path = f"{row['node_id']}.cmp"
    #     with open(node_path, "w") as f:
    #         for constituent in CONSTITUENTS:
    #             amp, phase = results[row["node_id"]][constituent]
    #             if not (np.isnan(amp) or np.isnan(phase)):
    #                 f.write(f"{constituent.upper():<12} {amp/100:>10.4f}  {phase:>8.2f}\n")


def main():
    boundary_df = pd.read_csv(BOUNDARY_NODES_CSV)
    print(f"Extracting {len(CONSTITUENTS)} constituents at {len(boundary_df)} boundary nodes ...")

    results = extract_all_constituents(boundary_df)
    write_cmp_file(results, boundary_df, OUTPUT_CMP)

    print(
        "\nDone. Reference this .cmp output from your .bnd/.ext boundary "
        "definition in the .mdu file. Validate against INCOIS Kochi tide "
        "gauge data once available — FES2014 is generally strong for "
        "coastal India but local channel effects can still shift phase."
    )


if __name__ == "__main__":
    main()
