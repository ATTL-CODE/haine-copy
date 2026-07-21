"""
HAINE Layer 3 - Step 1: per-pixel tidal deposition gating from TELEMAC.

Replaces the uniform placeholder bed shear in 10_first_order_siltation_volume.py
with a genuine Layer 1 -> Layer 3 coupling.

For every mesh node and every timestep of the tidal run:
    speed  s(t)   = sqrt(U^2 + V^2)            depth-averaged, m/s
    shear  tau(t) = RHO_W * CD * s(t)^2         Pa
    Krone  p(t)   = max(0, 1 - tau(t)/TAU_CR)   deposition probability
Tidally-average p over the run -> p_dep_mean per node. This is WHY the "no
deposition" wall disappears: even where PEAK shear exceeds tau_cr, slack water
(tau -> 0) contributes p -> 1, so the cycle-averaged p_dep is > 0 everywhere,
with low-energy basins gating more deposition than the swept channel/gut.

Then interpolate p_dep_mean from the mesh (EPSG:32643) onto the Layer 2 SPM grid
(EPSG:32643, 100 m) and save port_pdep_tidal.tif for the estimator to consume.

Run: conda activate telemac && python 11_extract_taub_from_telemac.py
"""
import os
import glob
import numpy as np
import rasterio
from scipy.interpolate import griddata

# ---------------- config (kept consistent with 10_...py) ----------------
SLF      = os.path.expanduser("~/haine_telemac/kochi_tidal_r.slf")
CUBE_DIR = os.path.expanduser("~/haine_layer2/spm_fused_port")
OUT_TIF  = os.path.join(CUBE_DIR, "port_pdep_tidal.tif")
OUT_TAU  = os.path.join(CUBE_DIR, "port_taub_mean.tif")   # diagnostic

RHO_W  = 1025.0     # seawater density, kg/m3
CD     = 2.5e-3     # bed drag coefficient (referenced to depth-avg velocity)
TAU_CR = 0.2        # critical shear for deposition, Pa
# ------------------------------------------------------------------------


def read_selafin(fname):
    """Return (slf, iu, iv, times, meshx, meshy, names) using the openTELEMAC API."""
    from data_manip.formats.selafin import Selafin  # requires the telemac env
    slf = Selafin(fname)
    names = [(n.decode() if isinstance(n, bytes) else n).strip().upper()
             for n in slf.varnames]

    def find(prefixes):
        for i, n in enumerate(names):
            if any(n.startswith(p) for p in prefixes):
                return i
        raise SystemExit(f"Velocity variable {prefixes} not found in {names}")

    iu = find(("VELOCITY U", "VITESSE U"))
    iv = find(("VELOCITY V", "VITESSE V"))
    try:
        times = np.asarray(slf.tags['times'], dtype=float)
    except Exception:
        times = np.asarray(slf.times, dtype=float)
    return slf, iu, iv, times, np.asarray(slf.meshx), np.asarray(slf.meshy), names


def main():
    if not os.path.exists(SLF):
        raise SystemExit(f"Missing TELEMAC result: {SLF}")

    slf, iu, iv, times, mx, my, names = read_selafin(SLF)
    print(f"SLF: {os.path.basename(SLF)}")
    print(f"  variables : {names}")
    print(f"  using U=idx{iu} ('{names[iu]}'), V=idx{iv} ('{names[iv]}')")
    print(f"  nodes {len(mx):,}, timesteps {len(times)}, "
          f"span {times[0]:.0f}-{times[-1]:.0f} s "
          f"({(times[-1]-times[0])/3600:.1f} h)")

    npoin = len(mx)
    pdep_sum = np.zeros(npoin)
    tau_sum  = np.zeros(npoin)
    smax     = np.zeros(npoin)
    nrec = len(times)

    for it in range(nrec):
        uv = slf.get_variables_at(it, [iu, iv])       # (2, npoin)
        u, v = np.asarray(uv[0]), np.asarray(uv[1])
        s   = np.hypot(u, v)
        tau = RHO_W * CD * s * s
        pdep_sum += np.clip(1.0 - tau / TAU_CR, 0.0, None)
        tau_sum  += tau
        smax = np.maximum(smax, s)

    pdep_mean = pdep_sum / nrec
    tau_mean  = tau_sum / nrec

    print("\nNode-wise tidally-averaged Krone p_dep:")
    print(f"  p_dep  min {pdep_mean.min():.3f}  median {np.median(pdep_mean):.3f}  "
          f"max {pdep_mean.max():.3f}")
    print(f"  tau_b (time-mean field)  min {tau_mean.min():.3f}  "
          f"median {np.median(tau_mean):.3f}  max {tau_mean.max():.3f} Pa  "
          f"(tau_cr={TAU_CR})")
    print(f"  peak speed across run: median {np.median(smax):.2f}  "
          f"max {smax.max():.2f} m/s")
    if np.median(pdep_mean) <= 0:
        raise SystemExit("Tidally-averaged p_dep still ~0 -> check CD/TAU_CR or "
                         "velocity units in the SLF.")
    print("  -> p_dep > 0: the uniform 'no deposition' wall is resolved.")

    # ---- interpolate the node field onto the Layer 2 SPM raster grid ----
    tifs = sorted(glob.glob(os.path.join(CUBE_DIR, "port_spm_*.tif")))
    if not tifs:
        raise SystemExit(f"No port_spm_*.tif in {CUBE_DIR} - run Layer 2 first.")
    ref = rasterio.open(tifs[0])
    H, W = ref.height, ref.width
    rows, cols = np.mgrid[0:H, 0:W]
    xs, ys = rasterio.transform.xy(ref.transform, rows.ravel(), cols.ravel())
    gx = np.asarray(xs).reshape(H, W)
    gy = np.asarray(ys).reshape(H, W)

    def to_grid(vals):
        g = griddata((mx, my), vals, (gx, gy), method="linear")
        nan = ~np.isfinite(g)
        if nan.any():   # fill outside the mesh convex hull with nearest node
            g[nan] = griddata((mx, my), vals, (gx[nan], gy[nan]), method="nearest")
        return g

    pdep_grid = to_grid(pdep_mean)
    tau_grid  = to_grid(tau_mean)

    # water mask from the SPM stack (land = all-NaN across all months)
    stack = np.stack([rasterio.open(t).read(1).astype("float64") for t in tifs], 0)
    water = np.isfinite(stack).any(axis=0)
    pdep_grid[~water] = np.nan
    tau_grid[~water]  = np.nan

    prof = ref.profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(OUT_TIF, "w", **prof) as d:
        d.write(pdep_grid.astype("float32"), 1)
    with rasterio.open(OUT_TAU, "w", **prof) as d:
        d.write(tau_grid.astype("float32"), 1)

    print(f"\nWrote per-pixel deposition gating -> {OUT_TIF}")
    print(f"Wrote mean bed-shear diagnostic    -> {OUT_TAU}")
    print(f"  grid {H}x{W}, water pixels {int(water.sum()):,}")
    print(f"  p_dep on water: min {np.nanmin(pdep_grid):.3f}  "
          f"median {np.nanmedian(pdep_grid):.3f}  max {np.nanmax(pdep_grid):.3f}")

    # optional quick-look
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        im0 = ax[0].imshow(pdep_grid, vmin=0, vmax=1, cmap="viridis")
        ax[0].set_title("Tidally-averaged Krone p_dep")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)
        im1 = ax[1].imshow(tau_grid, vmin=0,
                           vmax=np.nanpercentile(tau_grid, 98), cmap="magma")
        ax[1].set_title("Mean bed shear tau_b (Pa)")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)
        plt.tight_layout()
        png = os.path.expanduser("~/Downloads/pdep_taub_fields.png")
        plt.savefig(png, dpi=120)
        print(f"Saved quick-look -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\nNext: patch 10_first_order_siltation_volume.py to load port_pdep_tidal.tif "
          "as a per-pixel p_dep field (see the 3 edits), then re-run it and compare "
          "the annual volume against the ~24 Mcum literature target.")


if __name__ == "__main__":
    main()
