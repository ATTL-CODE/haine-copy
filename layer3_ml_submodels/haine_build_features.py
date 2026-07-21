"""
HAINE Layer 3 - REAL feature-table assembly for Sub-model A training.

Reads your actual pipeline outputs and builds the (year, zone) feature table the
trainer (haine_submodelA_train.py) consumes. Run this FIRST, then the trainer
auto-loads the CSV it writes. Replaces the synthetic stand-in with real data.

INPUTS it expects (paths configurable at the top):
  * multi-year SPM cubes : ~/haine_layer2/spm_fused_port_FY####-##/port_spm_*.tif
                           (+ the existing ~/haine_layer2/spm_fused_port for FY2024-25)
  * coupling fields      : ~/haine_layer2/spm_fused_port/port_pdep_tidal.tif
                           ~/haine_layer2/spm_fused_port/port_taub_mean.tif
  * dredging labels      : annual dredged volume per FY (below), disaggregated to
                           zones by the Ferket structure -> per-(year,zone) label
OUTPUT: ~/haine_layer3/haine_submodelA_features.csv

LABEL MODES:
  * ANNUAL  (available now): label = FY dredged volume x Ferket zone fraction,
    converted to accretion rate (Mm3 / zone-area). Gives (n_years x 4) samples -
    real but few; good enough to prove real skill direction, not final accuracy.
  * PIXEL   (when CPA paired surveys arrive): swap in per-pixel bed-level change
    as the label -> thousands of samples -> genuine trained skill. Hook marked.

Run: conda activate telemac && python haine_build_features.py
"""
import os, glob, json
import numpy as np, pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

BASE   = os.path.expanduser("~/haine_layer2")
REFDIR = f"{BASE}/spm_fused_port"                       # FY2024-25 + coupling fields
PDEP   = f"{REFDIR}/port_pdep_tidal.tif"
TAUB   = f"{REFDIR}/port_taub_mean.tif"
OUTCSV = os.path.expanduser("~/haine_layer3/haine_submodelA_features.csv")

# FY -> (cube folder, dredged volume Mm3)   [HARD record + existing]
YEARS = {
    "FY2016-17": (f"{BASE}/spm_fused_port_FY2016-17", 22.20),
    "FY2017-18": (f"{BASE}/spm_fused_port_FY2017-18", 21.00),
    "FY2019-20": (f"{BASE}/spm_fused_port_FY2019-20", 25.21),
    "FY2024-25": (REFDIR,                              24.00),
}
# Ferket-anchored zone split (Mm3 weights) and channel depth (m)
ZONES = {"approach":(19.1,14.5), "ernakulam":(1.4,13.25),
         "mattancherry":(2.5,9.75), "basins":(0.9,11.0)}
ZFRAC = {z: v[0]/sum(w[0] for w in ZONES.values()) for z,v in ZONES.items()}
# zone bounding boxes in true UTM 43N (EPSG:32643)
ZBOX = {"approach":(629300,638200,1099000,1104500),
        "ernakulam":(638200,642600,1101200,1105700),
        "mattancherry":(637000,641500,1094500,1100200),
        "basins":(637500,641800,1100200,1103800)}
MONSOON = {6,7,8,9}
RIVER_Q = {1:90,2:70,3:60,4:70,5:120,6:350,7:550,8:500,9:380,10:300,11:200,12:130}
DSR = 0.6   # Periyar dam storage ratio (effective-discharge correction)


def _read_on_ref(path, ref):
    """Read a raster resampled onto the reference grid (transform,crs,H,W)."""
    tr, crs, H, W = ref
    with rasterio.open(path) as src:
        dst = np.full((H, W), np.nan, "float64")
        reproject(rasterio.band(src, 1), dst, dst_transform=tr, dst_crs=crs,
                  resampling=Resampling.bilinear)
    return dst


def _zone_masks(ref):
    tr, crs, H, W = ref
    rows, cols = np.mgrid[0:H, 0:W]
    xs, ys = rasterio.transform.xy(tr, rows.ravel(), cols.ravel())
    gx = np.asarray(xs).reshape(H, W); gy = np.asarray(ys).reshape(H, W)
    masks = {}
    for z,(a,b,c,d) in ZBOX.items():
        masks[z] = (gx>=a)&(gx<b)&(gy>=c)&(gy<d)
    px_area = abs(tr.a)*abs(tr.e)
    return masks, px_area


def _month_of(tif):
    b = os.path.basename(tif).replace("port_spm_","").replace(".tif","")
    return int(b.split("-")[1])


def build(label_mode="annual"):
    if not os.path.isdir(REFDIR):
        raise FileNotFoundError(f"reference dir {REFDIR} not found - run Layer 2 first")
    # reference grid from the existing cube's first tif
    ref_tif = sorted(glob.glob(f"{REFDIR}/port_spm_*.tif"))[0]
    with rasterio.open(ref_tif) as r:
        ref = (r.transform, r.crs, r.height, r.width)
    masks, px_area = _zone_masks(ref)
    pdep = _read_on_ref(PDEP, ref) if os.path.exists(PDEP) else np.full(ref[2:], 0.98)
    taub = _read_on_ref(TAUB, ref) if os.path.exists(TAUB) else np.full(ref[2:], 0.003)

    # annual effective discharge + flood index (parametric until CWC series wired)
    qvals = np.array(list(RIVER_Q.values()))
    Qeff = float(qvals.mean()*(1-DSR)); flood_index = float(qvals.max()/qvals.mean())

    rows = []
    for fy,(folder, vol_total) in YEARS.items():
        tifs = sorted(glob.glob(f"{folder}/port_spm_*.tif"))
        if not tifs:
            print(f"  {fy}: no cube in {folder} -> skip (build it first)"); continue
        # monthly SPM means per zone
        stack, months = [], []
        for t in tifs:
            stack.append(_read_on_ref(t, ref)); months.append(_month_of(t))
        stack = np.stack(stack); months = np.array(months)
        for z, m in masks.items():
            zpx = m & np.isfinite(stack).any(0)
            if zpx.sum() == 0:  # zone not covered
                continue
            zmean = np.array([np.nanmean(np.where(zpx, s, np.nan)) for s in stack])
            spm_annual = float(np.nanmean(zmean))
            spm_peak   = float(np.nanmax(zmean))
            spm_mons   = float(np.nansum(zmean[np.isin(months, list(MONSOON))]))
            monsoon_f  = float(spm_mons / (np.nansum(zmean)+1e-9))
            taub_z = float(np.nanmean(np.where(zpx, taub, np.nan)))
            pdep_z = float(np.nanmean(np.where(zpx, pdep, np.nan)))
            zone_area = float(zpx.sum()*px_area)
            # LABEL: FY volume x Ferket fraction -> accretion rate (m/yr)
            zone_vol = vol_total*ZFRAC[z]               # Mm3/yr
            accretion = zone_vol*1e6/zone_area          # m/yr
            rows.append(dict(year=int(fy[2:6]), zone=z,
                mean_taub=taub_z, pdep_mean=pdep_z,
                spm_monsoon_integral=spm_mons, spm_annual_mean=spm_annual,
                spm_peak=spm_peak, monsoon_phase_frac=monsoon_f,
                Qeff=Qeff, flood_index=flood_index, antecedent_interval=275.0,
                tidal_prism_proxy=zone_area/1e6, segment_area=zone_area,
                segment_depth=ZONES[z][1], accretion_m_yr=accretion,
                cell_area=zone_area))
        print(f"  {fy}: {len([r for r in rows if r['year']==int(fy[2:6])])} zone-rows")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
    df.to_csv(OUTCSV, index=False)
    print(f"\nwrote {len(df)} samples -> {OUTCSV}")
    print("next:  python haine_submodelA_train.py   (auto-loads this CSV)")
    return df




def build_pixel(label_rasters):
    """PER-PIXEL training table -> the path to REAL skill.
    label_rasters: dict {FY_folder_key: path_to_bedlevel_change_raster_m_per_yr}
    (from CPA paired pre/post-dredge surveys, differenced, resampled to the SPM
    grid). Produces one row PER WATER PIXEL PER YEAR (thousands of samples).
    Same feature columns as the annual builder, so the trainer is unchanged."""
    ref_tif=sorted(glob.glob(f"{REFDIR}/port_spm_*.tif"))[0]
    with rasterio.open(ref_tif) as r:
        ref=(r.transform,r.crs,r.height,r.width)
    masks,px_area=_zone_masks(ref)
    pdep=_read_on_ref(PDEP,ref); taub=_read_on_ref(TAUB,ref)
    qv=np.array(list(RIVER_Q.values()))
    Qeff=float(qv.mean()*(1-DSR)); flood=float(qv.max()/qv.mean())
    zone_of=np.full(ref[2:],"",object); depth_of=np.zeros(ref[2:])
    for z,m in masks.items():
        zone_of[m]=z; depth_of[m]=ZONES[z][1]
    rows=[]
    for fy,(folder,_) in YEARS.items():
        lab_path=label_rasters.get(fy)
        if not lab_path or not os.path.exists(lab_path):
            print(f"  {fy}: no label raster -> skip"); continue
        tifs=sorted(glob.glob(f"{folder}/port_spm_*.tif"))
        if not tifs: print(f"  {fy}: no cube -> skip"); continue
        stack=np.stack([_read_on_ref(t,ref) for t in tifs])
        months=np.array([_month_of(t) for t in tifs])
        mons=np.isin(months,list(MONSOON))
        lab=_read_on_ref(lab_path,ref)
        spm_annual=np.nanmean(stack,0); spm_peak=np.nanmax(stack,0)
        spm_mons=np.nansum(stack[mons],0); spm_tot=np.nansum(stack,0)+1e-9
        water=np.isfinite(spm_annual)&np.isfinite(lab)&(zone_of!="")
        idx=np.argwhere(water)
        for (i,j) in idx:
            rows.append(dict(year=int(fy[2:6]),zone=zone_of[i,j],
                mean_taub=float(taub[i,j]),pdep_mean=float(pdep[i,j]),
                spm_monsoon_integral=float(spm_mons[i,j]),
                spm_annual_mean=float(spm_annual[i,j]),spm_peak=float(spm_peak[i,j]),
                monsoon_phase_frac=float(spm_mons[i,j]/spm_tot[i,j]),
                Qeff=Qeff,flood_index=flood,antecedent_interval=275.0,
                tidal_prism_proxy=px_area/1e6,segment_area=px_area,
                segment_depth=float(depth_of[i,j]),
                accretion_m_yr=float(lab[i,j]),cell_area=px_area))
        print(f"  {fy}: {int(water.sum())} pixel-rows")
    df=pd.DataFrame(rows); os.makedirs(os.path.dirname(OUTCSV),exist_ok=True)
    df.to_csv(OUTCSV,index=False)
    print(f"\nwrote {len(df)} PIXEL samples -> {OUTCSV}")
    return df

if __name__ == "__main__":
    build()   # annual mode (runs now); for pixel mode call build_pixel({...})
