"""
HAINE Layer 3 - MULTI-YEAR inter-annual calibration + leave-one-year-out CV.

This is the payoff of the multi-year Layer 2 work. For each labeled year it
computes ONE physical sediment-supply feature (the raw dredged-footprint
deposition from the SPM cube, same physics as pipeline v2), pairs it with that
year's dredged-volume label, and:

  1. fits a 1-parameter calibration  volume = k * feature   (through origin)
  2. fits a 2-parameter linear        volume = a * feature + b
  3. runs LEAVE-ONE-YEAR-OUT cross-validation (the honest skill estimate)
  4. reports per-year predicted vs actual, RMSE, and the calibration constant
     with its cross-year spread -- this REPLACES the single x0.06 pin.

HONESTY: with ~4 annual labels this is inter-annual CALIBRATION, not a large ML
train. It tells you whether satellite supply tracks dredged volume across years
and how well -- a real step up from a single pinned year.

Grid safety: every year's cube is resampled onto ONE reference grid at load, so
years are pixel-comparable even if the per-year fusion grids differ slightly.

Run: conda activate telemac && python haine_multiyear_calibrate.py
"""
import os, glob, json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ---- map each FY to its cube folder + dredged-volume label (Mm3) ----
BASE = os.path.expanduser("~/haine_layer2")
YEARS = [
    ("FY2016-17", f"{BASE}/spm_fused_port_FY2016-17", 22.20),
    ("FY2017-18", f"{BASE}/spm_fused_port_FY2017-18", 21.00),
    ("FY2019-20", f"{BASE}/spm_fused_port_FY2019-20", 25.21),
    ("FY2024-25", f"{BASE}/spm_fused_port", 24.00),   # existing cube
]
REF_YEAR = "FY2024-25"   # its grid is the reference all years resample onto

# ---- physics (same as v2) ----
W_S, TAU_CR, RHO_W, RHO_DRY, BED_ENH0 = 0.5e-3, 0.2, 1025.0, 450.0, 2.0
SEC_PER_MONTH = 365.25*24*3600/12.0
RIVER_Q = {1:90,2:70,3:60,4:70,5:120,6:350,7:550,8:500,9:380,10:300,11:200,12:130}
Q_MAX = max(RIVER_Q.values()); SCOUR_K = 0.6
# dredged-footprint zone boxes (EPSG:32643), same as v2
ZONE_BOXES = [(629300,638200,1099000,1104500),(638200,642600,1101200,1105700),
              (637000,641500,1094500,1100200),(637500,641800,1100200,1103800)]

CUBE_PDEP = f"{BASE}/spm_fused_port/port_pdep_tidal.tif"
CUBE_TAUB = f"{BASE}/spm_fused_port/port_taub_mean.tif"


def ref_grid():
    folder = dict((y[0], y[1]) for y in YEARS)[REF_YEAR]
    t = sorted(glob.glob(os.path.join(folder, "port_spm_*.tif")))[0]
    r = rasterio.open(t)
    return r.transform, r.crs, r.height, r.width


def load_year_on_grid(folder, transform, crs, H, W):
    """Load a year's monthly SPM tifs, resampled onto the reference grid."""
    tifs = sorted(glob.glob(os.path.join(folder, "port_spm_*.tif")))
    if not tifs:
        return None, None
    stack, months = [], []
    for t in tifs:
        src = rasterio.open(t)
        dst = np.full((H, W), np.nan, "float64")
        reproject(rasterio.band(src, 1), dst, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear)
        stack.append(dst)
        mm = os.path.basename(t).replace("port_spm_", "").replace(".tif", "")
        months.append(int(mm.split("-")[1]))
    return np.stack(stack, 0), months


def load_aux(transform, crs, H, W, path, default):
    if os.path.exists(path):
        src = rasterio.open(path)
        dst = np.full((H, W), np.nan, "float64")
        reproject(rasterio.band(src, 1), dst, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear)
        return dst
    return np.full((H, W), default)


def dredged_feature(stack, months, pdep, taub, transform):
    """Raw dredged-footprint deposition (Mm3) = the per-year supply feature."""
    H, W = stack.shape[1:]
    water = np.isfinite(stack).any(0)
    # pixel centre coords
    rows, cols = np.mgrid[0:H, 0:W]
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    gx = np.asarray(xs).reshape(H, W); gy = np.asarray(ys).reshape(H, W)
    dredged = np.zeros((H, W), bool)
    for a, b, c, d in ZONE_BOXES:
        dredged |= (gx >= a) & (gx < b) & (gy >= c) & (gy < d)
    dredged &= water
    tmax = np.nanmax(np.where(water, taub, np.nan))
    tn = np.clip(taub / tmax, 0, 1) if (np.isfinite(tmax) and tmax > 0) else taub*0
    area = abs(transform.a) * abs(transform.e)
    acc = np.zeros((H, W))
    for k, mo in enumerate(months):
        spm = stack[k]; qf = RIVER_Q.get(mo, 0.5*Q_MAX)/Q_MAX
        eff = np.clip(pdep*np.clip(1-SCOUR_K*qf*tn, 0, 1), 0, 1)
        acc += np.where(np.isfinite(spm), spm*BED_ENH0*W_S*eff, 0.0)*area*SEC_PER_MONTH
    acc = (acc/1e6)*1000.0/RHO_DRY/area   # m/yr per pixel
    vol_mcum = float(np.nansum(acc[dredged])*area/1e6)
    return vol_mcum


def main():
    transform, crs, H, W = ref_grid()
    pdep = load_aux(transform, crs, H, W, CUBE_PDEP, 1.0)
    taub = load_aux(transform, crs, H, W, CUBE_TAUB, 0.01)
    print(f"reference grid {H}x{W}  pdep/taub loaded")

    feats, labels, names = [], [], []
    print("\nPer-year sediment-supply feature vs dredged label:")
    for fy, folder, label in YEARS:
        if not os.path.isdir(folder):
            print(f"  {fy}: MISSING folder {folder} -> skip (build this year's cube first)")
            continue
        stack, months = load_year_on_grid(folder, transform, crs, H, W)
        if stack is None:
            print(f"  {fy}: no port_spm_*.tif -> skip"); continue
        f = dredged_feature(stack, months, pdep, taub, transform)
        feats.append(f); labels.append(label); names.append(fy)
        print(f"  {fy}: feature {f:8.1f} (raw Mm3)   label {label:5.2f} Mm3   "
              f"({len(months)} months)")

    n = len(feats)
    if n < 2:
        raise SystemExit("\nNeed >=2 years with cubes to calibrate. Build the "
                         "historical cubes (07b -> fuse -> clip) and re-run.")
    x = np.array(feats); y = np.array(labels)

    # 1-parameter (through origin): k = <x,y>/<x,x>
    k = float(np.dot(x, y)/np.dot(x, x))
    pred1 = k*x
    # 2-parameter linear
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    pred2 = a*x + b

    def rmse(p): return float(np.sqrt(np.mean((p-y)**2)))
    print(f"\n1-param  volume = k*feature   k={k:.4f}   RMSE={rmse(pred1):.2f} Mm3")
    print(f"2-param  volume = a*feature+b a={a:.4f} b={b:.2f}  RMSE={rmse(pred2):.2f} Mm3")

    # leave-one-year-out CV on the 1-param model (honest skill)
    print("\nLeave-one-year-out cross-validation (1-param):")
    loo_err = []
    for i in range(n):
        xi = np.delete(x, i); yi = np.delete(y, i)
        ki = float(np.dot(xi, yi)/np.dot(xi, xi))
        pi = ki*x[i]; e = pi - y[i]; loo_err.append(e)
        print(f"  hold out {names[i]}: predicted {pi:5.2f}  actual {y[i]:5.2f}  "
              f"err {e:+.2f} Mm3")
    loo_rmse = float(np.sqrt(np.mean(np.square(loo_err))))
    print(f"  LOO-CV RMSE = {loo_rmse:.2f} Mm3  "
          f"({100*loo_rmse/np.mean(y):.1f}% of mean volume)")
    print("\nThis LOO error is the real inter-annual skill -- it REPLACES the single "
          "x0.06 pin with a validated calibration constant k.")

    # save + plot
    out = dict(years=names, feature_mcum=[round(v,2) for v in feats],
               label_mcum=labels, k_1param=round(k,4),
               linear_a=round(float(a),4), linear_b=round(float(b),2),
               rmse_1param=round(rmse(pred1),3), rmse_2param=round(rmse(pred2),3),
               loo_cv_rmse=round(loo_rmse,3),
               note="inter-annual calibration; ~N=4 labels -> calibration+CV, not large ML train")
    jp = os.path.expanduser("~/haine_layer3/haine_multiyear_calibration.json")
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    json.dump(out, open(jp, "w"), indent=2)
    print(f"\nsaved -> {jp}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].scatter(x, y, c="#0E7C9B", s=60, zorder=3)
        for i, nm in enumerate(names): ax[0].annotate(nm, (x[i], y[i]), fontsize=8)
        xr = np.linspace(0, x.max()*1.1, 50)
        ax[0].plot(xr, k*xr, "--", color="#2EC4B6", label=f"k={k:.3f}")
        ax[0].set_xlabel("SPM supply feature (raw Mm³)"); ax[0].set_ylabel("dredged volume (Mm³)")
        ax[0].set_title("Inter-annual calibration"); ax[0].legend()
        ax[1].scatter(y, pred1, c="#E76F51", s=60, zorder=3)
        lims=[min(y.min(),pred1.min())-2, max(y.max(),pred1.max())+2]
        ax[1].plot(lims, lims, "k--", alpha=0.5)
        for i, nm in enumerate(names): ax[1].annotate(nm, (y[i], pred1[i]), fontsize=8)
        ax[1].set_xlabel("actual (Mm³)"); ax[1].set_ylabel("predicted (Mm³)")
        ax[1].set_title(f"Predicted vs actual (LOO RMSE {loo_rmse:.2f})")
        plt.tight_layout()
        png = os.path.expanduser("~/Downloads/haine_multiyear_calibration.png")
        plt.savefig(png, dpi=130); print(f"plot -> {png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
