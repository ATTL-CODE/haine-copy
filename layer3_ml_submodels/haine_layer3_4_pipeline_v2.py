"""
=============================================================================
 HAINE  -  Layer 3 + Layer 4  |  CORRECTED PIPELINE v2  |  Cochin Port
=============================================================================
Corrections vs v1 (all feasible with current data, no new inputs):

 [C1] DREDGED-FOOTPRINT VOLUME. v1 integrated deposition over the WHOLE 13x11 km
      AOI (incl. non-dredged backwater), which is why raw overshot 16x. v2
      integrates the maintenance-dredging volume over the DREDGED ZONES only
      (approach/Ernakulam/Mattancherry/basins); the non-dredged estuary is
      reported separately, not counted as dredged volume.

 [C2] FERKET-ANCHORED ZONE DISAGGREGATION. v1 let raw SPM*p_dep set each zone's
      share -> Mattancherry too high, basins too low vs Ferket. v2 imposes the
      Ferket per-zone fluid-mud structure (thickness x zone area) as the zone
      weights, and uses the satellite SPM*p_dep only for WITHIN-zone spatial
      texture. Zone totals now match the literature structure by construction.

 [C3] CALIBRATION DECOMPOSITION. v1 hid all error in one x0.06 pin. v2 splits it
      into (a) dredged-footprint fraction and (b) residual proxy/net-gross factor,
      so the number is interpretable instead of a black-box fudge.

 [C4] PER-YEAR RECORD COMPARISON. v2 prints the estimate against the year-by-year
      dredging record (22.2/21.0/25.21 + contracted 24), flagging that only ONE
      SPM year (2024-06..2025-06) is available so inter-annual training is not
      yet possible.

Still parametric (NOT corrected here - needs the TELEMAC re-run / new data):
   river forcing, wave forcing (Ferket used TOMAWAC), absolute per-pixel rate.
   See the accompanying Status_and_Corrections.md for the data needed and why.

RUN:  conda activate telemac && python haine_layer3_4_pipeline_v2.py
=============================================================================
"""
import os, glob, json, datetime as dt
import numpy as np
import rasterio

CUBE_DIR = os.path.expanduser("~/haine_layer2/spm_fused_port")
OUT_DIR  = os.path.expanduser("~/haine_layer3")
DL_DIR   = os.environ.get("HAINE_DOWNLOAD_DIR", os.path.expanduser("~/Downloads"))  # output dir; override via HAINE_DOWNLOAD_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ---- sediment physics (PROXY, literature defaults) ----
W_S, TAU_CR, RHO_W, RHO_DRY, BED_ENH0 = 0.5e-3, 0.2, 1025.0, 450.0, 2.0
NAUT_RHO = 1170.0

# ---- dredging record (labels). HARD where noted; 24 = contracted ----
DREDGE_VOL_MCUM = {"2016-17":22.20,"2017-18":21.00,"2019-20":25.21,
                   "2020-21":24.00,"2021-22":24.00,"2022-23":24.00,
                   "2023-24":24.00,"2024-25":24.00}
LIT_TARGET_MCUM = 24.0
FERKET_TOTAL_MCUM = 20.0   # Ferket 2017 "+20 Mm3/yr"

# ---- Ferket per-zone monsoon fluid-mud thickness, m (spatial prior) ----
FERKET_THICK_M = {"approach":6.5,"basins":5.0,"ernakulam":2.0,
                  "mattancherry":2.0,"other":1.0}
DREDGED_ZONES = ["approach","ernakulam","mattancherry","basins"]

# ---- channel geometry (HARD) ----
CHANNELS = {
 "approach":     dict(depth=14.5, width=260, draft=14.5, target=16.0),
 "ernakulam":    dict(depth=13.25,width=260, draft=12.5, target=13.25),
 "mattancherry": dict(depth=9.75, width=183, draft=9.0,  target=9.75)}

# ---- tides (HARD, DRDO-NPOL 2025) ----
TIDE_AMP = {"M2":0.24,"S2":0.10,"K1":0.20,"O1":0.10}
TIDE_SPRING_HAT = sum(TIDE_AMP.values())

# ---- Periyar monthly discharge, m3/s (PROXY) ----
RIVER_Q_MONTHLY = {1:90,2:70,3:60,4:70,5:120,6:350,7:550,8:500,9:380,10:300,11:200,12:130}
Q_MAX = max(RIVER_Q_MONTHLY.values()); SCOUR_K = 0.6

PERIYAR_BOUNDARY_NODES = [31940,32022,32103,32184,32265,32264,31371]

GUT = (639000.0, 1101000.0)
ZONE_BOXES = [("approach",(629300,638200,1099000,1104500)),
              ("ernakulam",(638200,642600,1101200,1105700)),
              ("mattancherry",(637000,641500,1094500,1100200)),
              ("basins",(637500,641800,1100200,1103800))]
SEC_PER_MONTH = 365.25*24*3600/12.0

def load_stack():
    tifs = sorted(glob.glob(os.path.join(CUBE_DIR,"port_spm_*.tif")))
    if not tifs: raise SystemExit(f"No port_spm_*.tif in {CUBE_DIR}")
    ref = rasterio.open(tifs[0])
    stack = np.stack([rasterio.open(t).read(1).astype("float64") for t in tifs],0)
    dates=[os.path.basename(t).replace("port_spm_","").replace(".tif","") for t in tifs]
    months=[int(d.split("-")[1]) for d in dates]
    return stack,dates,months,ref,tifs

def load_or_default(name,ref,d):
    p=os.path.join(CUBE_DIR,name)
    if os.path.exists(p):
        a=rasterio.open(p).read(1).astype("float64")
        if a.shape==(ref.height,ref.width): return a,True
    return np.full((ref.height,ref.width),d),False

def grid_coords(ref):
    H,W=ref.height,ref.width
    r,c=np.mgrid[0:H,0:W]
    xs,ys=rasterio.transform.xy(ref.transform,r.ravel(),c.ravel())
    return np.asarray(xs).reshape(H,W),np.asarray(ys).reshape(H,W)

def zone_map(gx,gy,water):
    zm=np.full(gx.shape,"other",dtype=object); done=np.zeros(gx.shape,bool)
    for name,(a,b,cc,d) in ZONE_BOXES:
        m=(~done)&water&(gx>=a)&(gx<b)&(gy>=cc)&(gy<d); zm[m]=name; done|=m
    return zm

def write_raster(a,ref,p):
    pr=ref.profile.copy(); pr.update(dtype="float32",count=1,nodata=np.nan)
    with rasterio.open(p,"w",**pr) as d: d.write(a.astype("float32"),1)

def raw_gross_accretion(stack,months,pdep,taub,water,area):
    """Uncalibrated gross accretion (m/yr) per pixel + effective p_dep stack."""
    tmax=np.nanmax(np.where(water,taub,np.nan)); tn=np.clip(taub/tmax,0,1) if tmax>0 else taub*0
    acc=np.zeros_like(taub); effs=[]
    for k,mo in enumerate(months):
        spm=stack[k]; qf=RIVER_Q_MONTHLY[mo]/Q_MAX
        eff=np.clip(pdep*np.clip(1-SCOUR_K*qf*tn,0,1),0,1); effs.append(eff)
        contrib=np.where(np.isfinite(spm),spm*BED_ENH0*W_S*eff,0.0)
        acc+=contrib*area*SEC_PER_MONTH
    acc=(acc/1e6)*1000.0/RHO_DRY/area; acc[~water]=np.nan
    return acc,effs,tn

def main():
    print("="*70); print("HAINE Layer 3+4  v2 (corrected)  |  Cochin Port"); print("="*70)
    stack,dates,months,ref,tifs=load_stack()
    area=abs(ref.transform.a)*abs(ref.transform.e)
    water=np.isfinite(stack).any(0); gx,gy=grid_coords(ref)
    pdep,gp=load_or_default("port_pdep_tidal.tif",ref,1.0)
    taub,gt=load_or_default("port_taub_mean.tif",ref,0.01)
    print(f"SPM {stack.shape} pixel {area:.0f}m2 water {int(water.sum()):,} | pdep {gp} taub {gt}")
    print(f"months {dates[0]}..{dates[-1]}  (ONE SPM year available)")

    # raw physics
    acc_raw,effs,tn=raw_gross_accretion(stack,months,pdep,taub,water,area)
    zm=zone_map(gx,gy,water)

    # [C1] volumes over dredged footprint vs whole AOI
    def zvol(mask): return float(np.nansum(acc_raw[mask])*area/1e6)  # Mm3
    raw_all=zvol(water&np.isfinite(acc_raw))
    dredged_mask=np.isin(zm,DREDGED_ZONES)&water&np.isfinite(acc_raw)
    raw_dredged=zvol(dredged_mask)
    other_mask=(zm=="other")&water&np.isfinite(acc_raw)
    raw_other=zvol(other_mask)
    frac_dredged=raw_dredged/raw_all if raw_all else 0

    # [C3] calibration decomposition
    resid=LIT_TARGET_MCUM/raw_dredged if raw_dredged else 1.0
    print(f"\n[C1/C3] raw gross deposition (whole AOI): {raw_all:.1f} Mm3/yr")
    print(f"        dredged footprint: {raw_dredged:.1f} Mm3/yr "
          f"({100*frac_dredged:.0f}% of gross); non-dredged estuary: {raw_other:.1f}")
    print(f"        residual proxy/net-gross factor to hit {LIT_TARGET_MCUM:.0f}: x{resid:.3f}")

    # [C2] Ferket-anchored zone weights: thickness x zone water area
    zarea={z:float((zm==z).sum())*area for z in DREDGED_ZONES}
    w={z:FERKET_THICK_M[z]*zarea[z] for z in DREDGED_ZONES}
    wsum=sum(w.values()) or 1.0
    zone_vol={z:LIT_TARGET_MCUM*w[z]/wsum for z in DREDGED_ZONES}

    # per-pixel calibrated accretion: within-zone texture from raw physics
    acc_cal=np.full_like(acc_raw,np.nan)
    for z in DREDGED_ZONES:
        m=(zm==z)&water&np.isfinite(acc_raw)
        tex=acc_raw[m]; s=np.nansum(tex)
        share=tex/s if s>0 else np.ones(int(m.sum()))/max(1,int(m.sum()))
        acc_cal[m]=(zone_vol[z]*1e6*share)/area   # m/yr
    write_raster(acc_cal,ref,os.path.join(OUT_DIR,"siltation_accretion_v2.tif"))
    print("\n[C2] Ferket-anchored zone volumes (Mm3/yr) + implied mean accretion:")
    for z in DREDGED_ZONES:
        m=(zm==z)&water&np.isfinite(acc_cal)
        ma=float(np.nanmean(acc_cal[m])) if m.sum() else 0
        print(f"     {z:12s} {zone_vol[z]:6.2f}  mean acc {ma:.2f} m  "
              f"(Ferket ~{FERKET_THICK_M[z]}m, area {zarea[z]/1e6:.2f} km2)")
    print(f"     dredged total: {sum(zone_vol.values()):.1f} Mm3/yr (== target by construction)")
    print(f"     non-dredged estuary siltation (context): ~{raw_other*resid:.1f} Mm3/yr")

    # [C4] per-year record comparison
    print("\n[C4] vs dredging record (single SPM year -> same estimate each row):")
    for yr,v in DREDGE_VOL_MCUM.items():
        tag="HARD" if yr in ("2016-17","2017-18","2019-20") else "contracted"
        print(f"     {yr}: record {v:5.2f}  estimate {LIT_TARGET_MCUM:.2f}  ({tag})")
    print("     -> cannot fit inter-annual variation with 1 SPM year (see doc).")

    # UKC
    def squat(Cb,V): return Cb*V*V/100.0*1.5
    ukc=[]
    for n,g in CHANNELS.items():
        s=squat(0.68,8.0); nb=min(0.5,0.10*FERKET_THICK_M.get(n,1))
        u=g["depth"]+TIDE_SPRING_HAT+nb-g["draft"]-s; need=0.10*g["draft"]
        tn_=max(0.0,need-(g["depth"]+nb-g["draft"]-s))
        ukc.append(dict(channel=n,draft=g["draft"],squat=round(s,2),
                        ukc=round(u,2),ok=bool(u>=need),tide_needed=round(tn_,2)))
    print("\n[UKC] spring high:")
    for r in ukc:
        status = "OK" if r["ok"] else ("needs +" + str(r["tide_needed"]) + "m")
        ch = r["channel"]; dr = r["draft"]; sq = r["squat"]; uk = r["ukc"]
        print("     " + ch.ljust(12) + " draft " + str(dr) + "m squat " + str(sq)
              + "m UKC " + str(uk) + "m -> " + status)

    dredger=dict(primary="Large TSHD (>=5000 m3) + Water Injection Dredging (WID)",
                 rationale="high volume soft fluid mud; nautical-depth maintenance")
    post=[dict(channel=n,charted=g["depth"],
               nautical_depth=round(g["depth"]+min(0.5,0.10*FERKET_THICK_M.get(n,1)),2),
               criterion=NAUT_RHO) for n,g in CHANNELS.items()]

    # sensitivity (raw, +/-20%)
    sens={}
    for k in ["W_S","RHO_DRY","BED_ENH0","SCOUR_K"]:
        base=globals()[k]; d=[]
        for f in (0.8,1.2):
            globals()[k]=base*f
            a2,_,_=raw_gross_accretion(stack,months,pdep,taub,water,area)
            d.append(float(np.nansum(a2[dredged_mask])*area/1e6)); globals()[k]=base
        lo,hi=d; b=0.5*(lo+hi); sens[k]=100*(hi-lo)/b if b else 0
    print("\n[Sensitivity] % change in dredged raw volume for +/-20%:")
    for k,v in sens.items(): print(f"     {k:10s} {v:+.1f}%")

    # dashboard
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(16,11))
        fig.suptitle("HAINE v2 - Cochin Port Dredging Dashboard (corrected)",
                     fontsize=15,fontweight="bold")
        ax=fig.add_subplot(2,3,1); v=acc_cal[np.isfinite(acc_cal)]
        im=ax.imshow(acc_cal,cmap="inferno",vmin=0,vmax=np.percentile(v,98) if v.size else 1)
        ax.set_title("Calibrated accretion (dredged zones, m/yr)"); ax.axis("off")
        plt.colorbar(im,ax=ax,fraction=0.046)
        ax=fig.add_subplot(2,3,2); zs=DREDGED_ZONES
        ax.bar(zs,[zone_vol[z] for z in zs],color="steelblue")
        for i,z in enumerate(zs): ax.text(i,zone_vol[z],f"{zone_vol[z]:.1f}",ha="center",va="bottom")
        ax.set_title("Ferket-anchored volume by zone (Mm3/yr)"); ax.tick_params(axis="x",rotation=25)
        ax=fig.add_subplot(2,3,3)
        ax.bar(zs,[FERKET_THICK_M[z] for z in zs],color="darkorange",alpha=.6,label="Ferket thick")
        ax.set_title("Ferket fluid-mud thickness prior (m)"); ax.tick_params(axis="x",rotation=25)
        ax=fig.add_subplot(2,3,4); ch=[r["channel"] for r in ukc]
        ax.bar(ch,[r["ukc"] for r in ukc],color=["seagreen" if r["ok"] else "firebrick" for r in ukc])
        ax.axhline(0,color="k",lw=.8); ax.set_title("UKC at spring high (m)")
        for i,r in enumerate(ukc): ax.text(i,r["ukc"],f"{r['ukc']:.2f}",ha="center",va="bottom")
        ax.tick_params(axis="x",rotation=20)
        ax=fig.add_subplot(2,3,5); ks=sorted(sens,key=lambda k:abs(sens[k]))
        ax.barh(ks,[sens[k] for k in ks],color="slateblue")
        ax.set_title("Sensitivity (+/-20%, % dredged vol)")
        ax=fig.add_subplot(2,3,6); ax.axis("off")
        ax.text(0,1,f"""CORRECTED SUB-MODEL A
 dredged total : {sum(zone_vol.values()):.1f} Mm3/yr (pinned to record)
 raw gross(AOI): {raw_all:.0f}  dredged: {raw_dredged:.0f} ({100*frac_dredged:.0f}%)
 residual factor: x{resid:.3f}  (proxy/net-gross)
 non-dredged estuary ~{raw_other*resid:.0f} Mm3/yr

 ZONE SPLIT = Ferket-imposed prior (downscaled by SPM)
 -> NOT an independent prediction of the split

 river Q Periyar (PROXY), scour k={SCOUR_K}
 WAVES NOT MODELLED (Ferket used TOMAWAC)

 Dredger: Large TSHD + WID
 Post-dredge: nautical depth 1170 kg/m3""",
                va="top",fontsize=8.5,family="monospace")
        plt.tight_layout(rect=[0,0,1,.96])
        p=os.path.join(OUT_DIR,"haine_dashboard_v2.png"); plt.savefig(p,dpi=130)
        try: plt.savefig(os.path.join(DL_DIR,"haine_dashboard_v2.png"),dpi=130)
        except Exception: pass
        print(f"\ndashboard -> {p}")
    except Exception as e:
        print("dashboard skipped:",e)

    results=dict(generated=dt.datetime.now().isoformat(timespec="seconds"),
        version="v2-corrected",port="Cochin",crs="EPSG:32643",
        dredged_total_mcum=round(sum(zone_vol.values()),2),
        raw_gross_aoi_mcum=round(raw_all,2),raw_dredged_mcum=round(raw_dredged,2),
        dredged_footprint_fraction=round(frac_dredged,3),
        residual_factor=round(resid,3),
        non_dredged_estuary_mcum=round(raw_other*resid,2),
        zone_volume_mcum={z:round(zone_vol[z],3) for z in DREDGED_ZONES},
        zone_split_source="Ferket 2017 thickness x area (imposed prior, not predicted)",
        dredge_record=DREDGE_VOL_MCUM,ukc=ukc,dredger=dredger,post_dredge=post,
        sensitivity_pct=sens,river_Q_monthly=RIVER_Q_MONTHLY,
        limitations=["single SPM year -> no inter-annual training",
                     "waves not modelled (Ferket used TOMAWAC)",
                     "river parametric not solver-coupled",
                     "sediment params are literature proxies",
                     "no CPA paired survey labels -> physics-calibrated not ML-trained"])
    jp=os.path.join(OUT_DIR,"haine_results_v2.json")
    json.dump(results,open(jp,"w"),indent=2,default=str)
    print(f"results JSON -> {jp}")
    print("\nDONE (v2). Corrections C1-C4 applied. See Status_and_Corrections.md and")
    print("Prediction_Validity.md for what is trustworthy and what needs new data.")

if __name__=="__main__":
    main()
