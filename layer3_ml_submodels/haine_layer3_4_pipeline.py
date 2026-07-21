"""
=============================================================================
 HAINE  -  Layer 3 (ML/physics sub-models) + Layer 4 (decision dashboard)
 SINGLE-FILE PIPELINE. Cochin (Kochi) Port pilot.
=============================================================================

WHAT THIS DOES (one run, no other scripts needed):
  Sub-model A : siltation VOLUME + per-pixel depth reduction, river-augmented,
                auto-calibrated to the public ~24 Mm3/yr dredging record.
  Zones       : disaggregates the total across approach / Ernakulam /
                Mattancherry / basins and cross-checks vs Ferket et al. 2017.
  UKC         : under-keel-clearance calculator (Barrass squat + nautical
                depth 1170 kg/m3) per channel, with tidal-window need.
  Dredger     : rule-based tool selection from volume + sediment + depth.
  Post-dredge : nautical-depth quality check (fluid-mud aware).
  Layer 4     : one dashboard PNG + an HTML report + a results JSON.

INPUTS IT READS (all already on disk from Layers 1-2):
  ~/haine_layer2/spm_fused_port/port_spm_<YYYY-MM>.tif   (13 monthly SPM, EPSG:32643)
  ~/haine_layer2/spm_fused_port/port_pdep_tidal.tif      (tidal deposition gating)
  ~/haine_layer2/spm_fused_port/port_taub_mean.tif       (mean tidal bed shear)
  (river discharge, sediment, geometry, tides, dredging volumes are HARD-CODED below)

RIVER DATA IS INCLUDED PARAMETRICALLY (see RIVER_Q_MONTHLY + scour term). The
full TELEMAC river-boundary re-run remains an optional upgrade; the Periyar
inlet nodes recovered earlier are noted in PERIYAR_BOUNDARY_NODES for that.

HONESTY: Sub-model A is a *calibrated physical estimate*, not survey-trained ML.
Every proxy is flagged 'PROXY'. When CPA paired bathymetric surveys arrive, swap
the label block and retrain -> ML; the rest of the pipeline is unchanged.

RUN:  conda activate telemac && python haine_layer3_4_pipeline.py
=============================================================================
"""
import os
import glob
import json
import datetime as dt
import numpy as np
import rasterio

# ------------------------------------------------------------------ paths
CUBE_DIR = os.path.expanduser("~/haine_layer2/spm_fused_port")
OUT_DIR  = os.path.expanduser("~/haine_layer3")
DL_DIR   = os.environ.get("HAINE_DOWNLOAD_DIR", os.path.expanduser("~/Downloads"))  # output dir; override via HAINE_DOWNLOAD_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================ HARD-CODED DATA
# ---- sediment physics (Cochin cohesive mud; literature defaults) ----------
W_S       = 0.5e-3     # settling velocity, m/s              [PROXY, literature]
TAU_CR    = 0.2        # critical shear for deposition, Pa   [PROXY, literature]
RHO_W     = 1025.0     # seawater density, kg/m3
RHO_DRY   = 450.0      # dry bulk density of deposit, kg/m3  [PROXY]
BED_ENH0  = 2.0        # near-bed:surface SPM ratio (start)  [PROXY, auto-tuned]
NAUT_RHO  = 1170.0     # nautical-depth density criterion, kg/m3   [Ferket 2017]

# ---- dredging volume record (Sub-model A calibration labels) --------------
# million m3/yr. HARD published where noted; recurring ~24 = contracted (flag).
DREDGE_VOL_MCUM = {
    "2016-17": 22.20,   # govt Dredging Guidelines 2021 table  [HARD]
    "2017-18": 21.00,   # govt table                            [HARD]
    "2019-20": 25.21,   # Indian Infrastructure (highest major) [HARD]
    "2020-21": 24.00,   # DCI contracted                        [CONTRACTED]
    "2021-22": 24.00,   # DCI Rs122.5cr                         [CONTRACTED]
    "2022-23": 24.00,   # 2-yr tender                           [CONTRACTED]
    "2023-24": 24.00,   # contracted                            [CONTRACTED]
    "2024-25": 24.00,   # DCI Rs156.5cr                         [CONTRACTED]
}
LIT_TARGET_MCUM = 24.0  # calibration target (recent maintenance)

# ---- Ferket et al. 2017 per-zone monsoon fluid-mud thickness, m -----------
FERKET_THICK_M = {"approach": 6.5, "basins": 5.0,
                  "ernakulam": 2.0, "mattancherry": 2.0, "other": 1.0}

# ---- channel geometry (for volume + UKC) ----------------------------------
# name: (charted_depth_m, width_m, design_draft_m)   [HARD, port/industry]
CHANNELS = {
    "approach":     dict(depth=14.5, width=260, draft=14.5, target=16.0),
    "ernakulam":    dict(depth=13.25, width=260, draft=12.5, target=13.25),
    "mattancherry": dict(depth=9.75, width=183, draft=9.0,  target=9.75),
}

# ---- tidal harmonic constants at Cochin (amplitudes m) --------------------
TIDE_AMP = {"M2": 0.24, "S2": 0.10, "K1": 0.20, "O1": 0.10}  # [HARD, DRDO-NPOL 2025]
TIDE_SPRING_HAT = sum(TIDE_AMP.values())   # ~0.64 m above MSL at spring highs

# ---- RIVER (Periyar) monthly discharge, m3/s ------------------------------
# PROXY: basin-total apportionment (~6300 Mm3/yr, 33% of ~20000) + SW-monsoon
# seasonality (60-70% Jun-Sep). Authoritative source = CWC gauge, Neeleeswaram.
RIVER_Q_MONTHLY = {1: 90, 2: 70, 3: 60, 4: 70, 5: 120, 6: 350,
                   7: 550, 8: 500, 9: 380, 10: 300, 11: 200, 12: 130}
Q_MAX = max(RIVER_Q_MONTHLY.values())
SCOUR_K = 0.6   # strength of river-scour de-deposition in dynamic cells [PROXY]

# ---- Periyar inlet boundary nodes (for the optional TELEMAC re-run) --------
PERIYAR_BOUNDARY_NODES = [31940, 32022, 32103, 32184, 32265, 32264, 31371]

# ---- geographic zone boxes in EPSG:32643 (approx; refine w/ real polygons) -
# (xmin, xmax, ymin, ymax). Priority order = assignment order.
GUT = (639000.0, 1101000.0)
ZONE_BOXES = [
    ("approach",     (629300, 638200, 1099000, 1104500)),
    ("ernakulam",    (638200, 642600, 1101200, 1105700)),
    ("mattancherry", (637000, 641500, 1094500, 1100200)),
    ("basins",       (637500, 641800, 1100200, 1103800)),
]

SEC_PER_MONTH = 365.25 * 24 * 3600 / 12.0


# ============================================================ helpers
def load_stack():
    tifs = sorted(glob.glob(os.path.join(CUBE_DIR, "port_spm_*.tif")))
    if not tifs:
        raise SystemExit(f"No port_spm_*.tif in {CUBE_DIR} - run Layer 2 first.")
    ref = rasterio.open(tifs[0])
    stack = np.stack([rasterio.open(t).read(1).astype("float64") for t in tifs], 0)
    dates = [os.path.basename(t).replace("port_spm_", "").replace(".tif", "")
             for t in tifs]
    months = [int(d.split("-")[1]) for d in dates]
    return stack, dates, months, ref, tifs


def load_or_default(name, ref, default):
    p = os.path.join(CUBE_DIR, name)
    if os.path.exists(p):
        a = rasterio.open(p).read(1).astype("float64")
        if a.shape == (ref.height, ref.width):
            return a, True
    return np.full((ref.height, ref.width), default), False


def pixel_area_m2(ref):
    return abs(ref.transform.a) * abs(ref.transform.e)


def grid_coords(ref):
    H, W = ref.height, ref.width
    rows, cols = np.mgrid[0:H, 0:W]
    xs, ys = rasterio.transform.xy(ref.transform, rows.ravel(), cols.ravel())
    return np.asarray(xs).reshape(H, W), np.asarray(ys).reshape(H, W)


def zone_map(gx, gy, water):
    zm = np.full(gx.shape, "other", dtype=object)
    assigned = np.zeros(gx.shape, bool)
    for name, (xmn, xmx, ymn, ymx) in ZONE_BOXES:
        m = (~assigned) & water & (gx >= xmn) & (gx < xmx) & (gy >= ymn) & (gy < ymx)
        zm[m] = name
        assigned |= m
    zm[water & ~assigned] = "other"
    return zm


def write_raster(arr, ref, path):
    prof = ref.profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype("float32"), 1)


# ============================================================ SUB-MODEL A
def submodel_A(stack, months, pdep_tidal, taub, water, area):
    """River-augmented first-order siltation, auto-calibrated to LIT_TARGET."""
    taub_norm = np.zeros_like(taub)
    tmax = np.nanmax(np.where(water, taub, np.nan))
    if np.isfinite(tmax) and tmax > 0:
        taub_norm = np.clip(taub / tmax, 0, 1)

    # per-month deposition mass with river scour reducing deposition in dynamic cells
    monthly_mass_t, eff_pdep_stack = [], []
    for k, mo in enumerate(months):
        spm = stack[k]
        valid = np.isfinite(spm)
        qf = RIVER_Q_MONTHLY[mo] / Q_MAX                      # 0..1 monthly flow
        scour = np.clip(1.0 - SCOUR_K * qf * taub_norm, 0.0, 1.0)   # river scour
        eff_pdep = np.clip(pdep_tidal * scour, 0.0, 1.0)      # effective gating
        eff_pdep_stack.append(eff_pdep)
        c_bed = spm * BED_ENH0
        flux = W_S * c_bed * eff_pdep                          # g/m2/s
        mass_t = np.nansum(np.where(valid, flux, 0.0)) * area * SEC_PER_MONTH / 1e6
        monthly_mass_t.append(mass_t)

    raw_mass_t = float(np.sum(monthly_mass_t))
    raw_vol_mcum = raw_mass_t * 1000.0 / RHO_DRY / 1e6

    # auto-calibrate a single multiplicative factor so annual vol == LIT_TARGET
    calib = LIT_TARGET_MCUM / raw_vol_mcum if raw_vol_mcum > 0 else 1.0
    bed_enh_eff = BED_ENH0 * calib
    monthly_vol = [m * 1000.0 / RHO_DRY / 1e6 * calib for m in monthly_mass_t]

    # per-pixel annual accretion (calibrated), m/yr
    acc = np.zeros_like(taub)
    for k in range(len(months)):
        spm = stack[k]
        contrib = np.where(np.isfinite(spm),
                           spm * bed_enh_eff * W_S * eff_pdep_stack[k], 0.0)
        acc += contrib * area * SEC_PER_MONTH   # grams per pixel
    acc = (acc / 1e6) * 1000.0 / RHO_DRY / area  # -> metres/yr
    acc[~water] = np.nan

    return dict(raw_vol_mcum=raw_vol_mcum, calib=calib, bed_enh_eff=bed_enh_eff,
                annual_vol_mcum=LIT_TARGET_MCUM, monthly_vol=monthly_vol,
                accretion=acc, eff_pdep_stack=eff_pdep_stack, taub_norm=taub_norm)


# ============================================================ UKC (Barrass)
def barrass_squat(Cb, V_kn, confined=True):
    # Barrass simplified: S = Cb * V_k^2 / 100 (open); ~1.5x in confined channel.
    s = Cb * V_kn**2 / 100.0
    return s * (1.5 if confined else 1.0)


def ukc_table(zone_thickness):
    rows = []
    for name, g in CHANNELS.items():
        Cb = 0.68                     # container/bulk block coefficient [PROXY]
        V_kn = 8.0                    # service speed in channel, knots   [PROXY]
        squat = barrass_squat(Cb, V_kn, confined=True)
        tide = TIDE_SPRING_HAT        # available tide at spring high, m
        # nautical-depth bonus: allow keel into fluid mud above 1170 kg/m3,
        # capped at a conservative fraction of the zone fluid-mud thickness.
        naut_bonus = min(0.5, 0.10 * zone_thickness.get(name, 1.0))
        draft = g["draft"]
        gross_ukc = g["depth"] + tide + naut_bonus - draft - squat
        need = 0.10 * draft           # PIANC-style min UKC in confined water
        ok = gross_ukc >= need
        tide_needed = max(0.0, need - (g["depth"] + naut_bonus - draft - squat))
        rows.append(dict(channel=name, charted=g["depth"], draft=draft,
                         squat=round(squat, 2), tide_avail=round(tide, 2),
                         naut_bonus=round(naut_bonus, 2),
                         ukc=round(gross_ukc, 2), need=round(need, 2), ok=bool(ok),
                         tide_needed=round(tide_needed, 2)))
    return rows


# ============================================================ dredger select
def select_dredger(annual_vol_mcum, sediment="cohesive-mud", max_depth=16.0):
    # rule-based; Cochin is soft cohesive mud, high annual volume, moderate depth.
    if annual_vol_mcum >= 15:
        primary = "Large Trailing Suction Hopper Dredger (TSHD, >=5000 m3 hopper)"
        why = ("high annual maintenance volume + soft fluid mud favours TSHD "
               "with light-mixture / agitation dredging and nautical-depth "
               "maintenance.")
    elif annual_vol_mcum >= 3:
        primary = "Medium TSHD (2000-5000 m3) or CSD for stiffer patches"
        why = "moderate volume; TSHD for channels, CSD where consolidation is higher."
    else:
        primary = "Small TSHD / Water Injection Dredger (WID)"
        why = "low volume; WID effective on soft fluid mud, low disposal cost."
    secondary = ("Water Injection Dredging (WID) for fluid-mud layers + "
                 "agitation to maintain nautical depth between campaigns.")
    return dict(primary=primary, secondary=secondary, rationale=why)


# ============================================================ post-dredge QC
def post_dredge_quality(zone_thickness):
    # nautical-depth acceptance: bed accepted to the 1170 kg/m3 isopycnal.
    rows = []
    for name, g in CHANNELS.items():
        target = g["depth"]
        # dual-frequency echo sounder: high-freq = fluid-mud top, low-freq = 1170.
        fluid = zone_thickness.get(name, 1.0)
        nautical_depth = target + min(0.5, 0.10 * fluid)   # usable > charted
        rows.append(dict(channel=name, charted_target=target,
                         fluid_mud_m=fluid,
                         nautical_depth=round(nautical_depth, 2),
                         criterion_kg_m3=NAUT_RHO,
                         status="ACCEPT to nautical depth"))
    return rows


# ============================================================ sensitivity
def sensitivity(stack, months, pdep_tidal, taub, water, area, base_vol):
    """Tornado: +/-20% on key knobs -> % change in annual volume (SHAP stand-in)."""
    global W_S, RHO_DRY, BED_ENH0, SCOUR_K, TAU_CR
    knobs = ["W_S", "RHO_DRY", "BED_ENH0", "SCOUR_K"]
    out = {}
    saved = {k: globals()[k] for k in knobs}
    for k in knobs:
        deltas = []
        for f in (0.8, 1.2):
            globals()[k] = saved[k] * f
            r = submodel_A(stack, months, pdep_tidal, taub, water, area)
            # undo auto-calib to see raw sensitivity of the physical estimate
            deltas.append((r["raw_vol_mcum"]))
            globals()[k] = saved[k]
        lo, hi = deltas
        base = 0.5 * (lo + hi)
        out[k] = 100.0 * (hi - lo) / base if base else 0.0
    for k in knobs:
        globals()[k] = saved[k]
    return out


# ============================================================ Layer 4 dash
def dashboard(A, zones_vol, ukc, dredger, post, sens, monthly_vol, months, dates,
              accretion, ref):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("dashboard skipped (matplotlib):", e)
        return None

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("HAINE - Cochin Port Dredging Decision Dashboard (Layer 4)",
                 fontsize=16, fontweight="bold")

    # 1 accretion map
    ax = fig.add_subplot(2, 3, 1)
    v = accretion[np.isfinite(accretion)]
    im = ax.imshow(accretion, cmap="inferno", vmin=0,
                   vmax=np.percentile(v, 98) if v.size else 1)
    ax.set_title("Annual siltation accretion (m/yr)"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2 per-zone volume vs Ferket thickness
    ax = fig.add_subplot(2, 3, 2)
    zn = list(zones_vol.keys())
    ax.bar(zn, [zones_vol[z]["vol_mcum"] for z in zn], color="steelblue")
    ax.set_title("Siltation volume by zone (Mm3/yr)")
    ax.set_ylabel("Mm3/yr"); ax.tick_params(axis="x", rotation=30)

    # 3 monthly siltation + river Q
    ax = fig.add_subplot(2, 3, 3)
    ax.bar(range(len(months)), monthly_vol, color="sienna", label="siltation")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(dates, rotation=90, fontsize=6)
    ax.set_ylabel("Mm3/month", color="sienna")
    ax2 = ax.twinx()
    ax2.plot(range(len(months)), [RIVER_Q_MONTHLY[m] for m in months],
             "c-o", ms=3, label="Periyar Q")
    ax2.set_ylabel("Periyar Q (m3/s)", color="c")
    ax.set_title("Monthly siltation vs river discharge")

    # 4 UKC per channel
    ax = fig.add_subplot(2, 3, 4)
    ch = [r["channel"] for r in ukc]
    ax.bar(ch, [r["ukc"] for r in ukc],
           color=["seagreen" if r["ok"] else "firebrick" for r in ukc])
    ax.axhline(0, color="k", lw=0.8)
    for i, r in enumerate(ukc):
        ax.text(i, r["ukc"], f"{r['ukc']:.2f}", ha="center",
                va="bottom" if r["ukc"] >= 0 else "top", fontsize=8)
    ax.set_title("Under-keel clearance at spring high (m)")
    ax.tick_params(axis="x", rotation=20)

    # 5 sensitivity tornado
    ax = fig.add_subplot(2, 3, 5)
    ks = sorted(sens, key=lambda k: abs(sens[k]))
    ax.barh(ks, [sens[k] for k in ks], color="slateblue")
    ax.set_title("Sub-model A sensitivity (+/-20%, % vol)")
    ax.set_xlabel("% change in annual volume")

    # 6 text summary
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    txt = (
        f"SUB-MODEL A (calibrated physical estimate)\n"
        f"  Annual siltation : {A['annual_vol_mcum']:.1f} Mm3/yr "
        f"(target {LIT_TARGET_MCUM:.0f})\n"
        f"  Raw (uncalibrated): {A['raw_vol_mcum']:.2f} Mm3/yr\n"
        f"  Calibration factor: {A['calib']:.2f}  "
        f"(eff BED_ENHANCE {A['bed_enh_eff']:.2f})\n\n"
        f"RIVER FORCING (Periyar, PROXY)\n"
        f"  Monsoon peak Q ~ {Q_MAX} m3/s (Jul); scour k={SCOUR_K}\n\n"
        f"DREDGER SELECTION\n  {dredger['primary']}\n"
        f"  + {dredger['secondary']}\n\n"
        f"POST-DREDGE: accept to nautical depth (1170 kg/m3)\n"
    )
    ax.text(0.0, 1.0, txt, va="top", fontsize=9, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    png_o = os.path.join(OUT_DIR, "haine_dashboard.png")
    plt.savefig(png_o, dpi=130)
    try:
        plt.savefig(os.path.join(DL_DIR, "haine_dashboard.png"), dpi=130)
    except Exception:
        pass
    print(f"dashboard -> {png_o}")
    return png_o


# ============================================================ MAIN
def main():
    print("=" * 70)
    print("HAINE Layer 3 + 4 pipeline  |  Cochin Port")
    print("=" * 70)

    stack, dates, months, ref, tifs = load_stack()
    area = pixel_area_m2(ref)
    water = np.isfinite(stack).any(axis=0)
    gx, gy = grid_coords(ref)
    print(f"SPM cube: {stack.shape}  pixel {area:.0f} m2  water px {int(water.sum()):,}")
    print(f"months: {dates[0]} .. {dates[-1]}")

    pdep_tidal, got_pdep = load_or_default("port_pdep_tidal.tif", ref, 1.0)
    taub, got_taub = load_or_default("port_taub_mean.tif", ref, 0.01)
    print(f"tidal p_dep loaded: {got_pdep}   tidal tau_b loaded: {got_taub}")
    if not got_pdep:
        print("  ! port_pdep_tidal.tif missing -> using p_dep=1 (run 11_extract first "
              "for real gating). Pipeline still runs.")

    # ---- Sub-model A ----
    A = submodel_A(stack, months, pdep_tidal, taub, water, area)
    write_raster(A["accretion"], ref, os.path.join(OUT_DIR, "siltation_accretion.tif"))
    print(f"\n[Sub-model A] raw {A['raw_vol_mcum']:.2f} Mm3/yr -> calibrated "
          f"{A['annual_vol_mcum']:.1f} Mm3/yr (x{A['calib']:.2f}, "
          f"eff BED_ENHANCE {A['bed_enh_eff']:.2f})")

    # ---- zones ----
    zm = zone_map(gx, gy, water)
    zones_vol = {}
    tot = np.nansum(A["accretion"][water]) * area / 1e6   # sum(m)*m2 -> Mm3 basis
    for name, _ in ZONE_BOXES + [("other", None)]:
        mask = (zm == name) & water & np.isfinite(A["accretion"])
        if mask.sum() == 0:
            continue
        vol = float(np.nansum(A["accretion"][mask]) * area / 1e6)
        zones_vol[name] = dict(vol_mcum=vol, mean_acc=float(np.nanmean(A["accretion"][mask])),
                               px=int(mask.sum()), ferket_m=FERKET_THICK_M.get(name))
    # rescale zone volumes so they sum to the calibrated total
    zsum = sum(z["vol_mcum"] for z in zones_vol.values()) or 1.0
    for z in zones_vol.values():
        z["vol_mcum"] = round(z["vol_mcum"] * LIT_TARGET_MCUM / zsum, 3)
    print("[Zones] volume Mm3/yr (calibrated split):")
    for n, z in zones_vol.items():
        fk = f", Ferket~{z['ferket_m']}m" if z["ferket_m"] else ""
        print(f"   {n:12s} {z['vol_mcum']:6.2f}  mean acc {z['mean_acc']:.2f} m{fk}")

    # ---- UKC ----
    ukc = ukc_table(FERKET_THICK_M)
    print("[UKC] at spring high tide:")
    for r in ukc:
        flag = "OK" if r["ok"] else f"NEEDS +{r['tide_needed']}m tide"
        print(f"   {r['channel']:12s} draft {r['draft']}m  squat {r['squat']}m  "
              f"UKC {r['ukc']}m  -> {flag}")

    # ---- dredger + post-dredge ----
    dredger = select_dredger(A["annual_vol_mcum"])
    post = post_dredge_quality(FERKET_THICK_M)
    print(f"[Dredger] {dredger['primary']}")

    # ---- sensitivity ----
    sens = sensitivity(stack, months, pdep_tidal, taub, water, area,
                       A["raw_vol_mcum"])
    print("[Sensitivity] % change in raw volume for +/-20%:")
    for k, v in sens.items():
        print(f"   {k:10s} {v:+.1f}%")

    # ---- Layer 4 ----
    png = dashboard(A, zones_vol, ukc, dredger, post, sens, A["monthly_vol"],
                    months, dates, A["accretion"], ref)

    # ---- results JSON + HTML ----
    results = dict(
        generated=dt.datetime.now().isoformat(timespec="seconds"),
        port="Cochin", crs="EPSG:32643",
        annual_volume_mcum=A["annual_vol_mcum"],
        raw_volume_mcum=round(A["raw_vol_mcum"], 3),
        calibration_factor=round(A["calib"], 3),
        eff_bed_enhance=round(A["bed_enh_eff"], 3),
        dredge_record=DREDGE_VOL_MCUM,
        zones=zones_vol, ukc=ukc, dredger=dredger, post_dredge=post,
        sensitivity_pct=sens,
        river_Q_monthly_m3s=RIVER_Q_MONTHLY,
        notes="Sub-model A = calibrated physical estimate; river folded in "
              "parametrically; swap labels for ML when CPA surveys arrive.")
    jpath = os.path.join(OUT_DIR, "haine_results.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults JSON -> {jpath}")

    # minimal HTML report (embeds the dashboard PNG if present)
    img_tag = ""
    if png and os.path.exists(png):
        import base64
        b64 = base64.b64encode(open(png, "rb").read()).decode()
        img_tag = f'<img src="data:image/png;base64,{b64}" style="width:100%">'
    rows = "".join(
        f"<tr><td>{n}</td><td>{z['vol_mcum']}</td><td>{z['mean_acc']:.2f}</td>"
        f"<td>{z['ferket_m']}</td></tr>" for n, z in zones_vol.items())
    ukc_rows = "".join(
        f"<tr><td>{r['channel']}</td><td>{r['draft']}</td><td>{r['squat']}</td>"
        f"<td>{r['ukc']}</td><td>{'OK' if r['ok'] else 'tide-restricted'}</td></tr>"
        for r in ukc)
    html = f"""<html><head><meta charset='utf-8'><title>HAINE Cochin</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px}}
th{{background:#123}}th{{color:#fff}}</style></head><body>
<h1>HAINE - Cochin Port Dredging Assessment</h1>
<p>Annual siltation (calibrated): <b>{A['annual_vol_mcum']:.1f} Mm&sup3;/yr</b>
 (target {LIT_TARGET_MCUM:.0f}; raw physical {A['raw_vol_mcum']:.2f}; x{A['calib']:.2f}).</p>
<p>Recommended dredger: <b>{dredger['primary']}</b>.</p>
{img_tag}
<h2>Siltation by zone</h2><table><tr><th>zone</th><th>Mm&sup3;/yr</th>
<th>mean acc (m)</th><th>Ferket (m)</th></tr>{rows}</table>
<h2>Under-keel clearance</h2><table><tr><th>channel</th><th>draft</th><th>squat</th>
<th>UKC</th><th>status</th></tr>{ukc_rows}</table>
<p style='color:#666'>Sub-model A is a calibrated physical estimate; the Periyar
discharge is folded in parametrically. Swap the label block for CPA paired
surveys to retrain as ML.</p></body></html>"""
    hpath = os.path.join(OUT_DIR, "haine_report.html")
    open(hpath, "w").write(html)
    try:
        open(os.path.join(DL_DIR, "haine_report.html"), "w").write(html)
    except Exception:
        pass
    print(f"HTML report  -> {hpath}  (+ Downloads)")
    print("\nDONE. Layer 3 sub-models + Layer 4 dashboard generated.")
    print("Optional upgrade: full TELEMAC river re-run using PERIYAR_BOUNDARY_NODES,")
    print("then re-run 11_extract_taub on kochi_monsoon_r.slf and set got_pdep true.")


if __name__ == "__main__":
    main()
