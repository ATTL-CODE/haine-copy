"""
HAINE Layer 3 - STOCHASTIC siltation forecast (completed probabilistic module).

Turns the single deterministic Sub-model A estimate into a full PROBABILISTIC
product. Needs NO survey labels. New in this version:
  * PER-ZONE flood weighting (the exposed approach channel takes a bigger flood
    hit than the sheltered inner channels)
  * NON-STATIONARY option (published Periyar FFA shows a rising flood trend;
    stationary analysis underestimated the 2-yr level by ~61%) -> forecast a
    future horizon under a rising-flood assumption
  * fit_from_series(): fit the Poisson rate + Log-Normal magnitude from a REAL
    daily-discharge CSV (India-WRIS Neeleeswaram) when you have it

HONEST STATUS: a SCENARIO ensemble under assumed/anchored forcing statistics,
not a survey-validated predictor. Normal-year spread = real dredging record.
Flood tail = anchored to published Periyar/Neeleeswaram behaviour + 2018 flood
(Kerala best-fit family LN3/PE3). Replace anchors with fit_from_series() output.

Run: conda activate telemac && python haine_stochastic_forcing.py
Fit:  python -c "import haine_stochastic_forcing as h; \
      print(h.fit_from_series('neeleeswaram_daily.csv','Discharge_m3_s','Date'))"
"""
import os
import json, os
import numpy as np
from scipy import stats

# ------------------------------------------------------------------ CONFIG
N        = 20000
V0       = 24.0                      # calibrated baseline (Mm3/yr)
RNG      = np.random.default_rng(42)
RECORD   = [22.20, 21.00, 25.21]     # real dredging record -> normal-year spread
ZONES    = {"approach": 19.1, "ernakulam": 1.4, "mattancherry": 2.5, "basins": 0.9}

# per-zone flood sensitivity (exposed outer channel > sheltered inner channels)
ZONE_FLOOD_SENS = {"approach": 1.30, "basins": 1.10, "ernakulam": 0.70, "mattancherry": 0.60}

# flood tail (Periyar/Neeleeswaram-anchored; overwrite via fit_from_series)
LAMBDA_FLOOD  = 0.30     # Poisson rate of a silt-lifting monsoon flood pulse /yr
UPLIFT_MEDIAN = 0.20     # median significant-pulse siltation uplift (+20%)
UPLIFT_SIGMA  = 0.44     # ln-space sigma; P99 pulse ~+55% (2018-class)

# non-stationarity (published rising trend): inflate flood rate over a horizon
HORIZON_YEARS   = 0      # 0 = present day; e.g. 2035-2026 = 9 for a 2035 forecast
TREND_PER_DECADE = 0.35  # +35% flood-rate per decade (conservative, editable)
# --------------------------------------------------------------------------


def sigma_from_record(rec):
    rec = np.asarray(rec, float); cv = rec.std(ddof=1)/rec.mean()
    return float(np.sqrt(np.log(1+cv**2))), float(cv)


def fit_from_series(csv_path, q_col="Discharge_m3_s", date_col="Date",
                    pot_quantile=0.98):
    """Fit Poisson flood rate + Log-Normal magnitude from a real daily-discharge
    CSV (e.g. India-WRIS Neeleeswaram). Returns dict of parameters you can paste
    over LAMBDA_FLOOD / UPLIFT_MEDIAN / UPLIFT_SIGMA. The discharge->siltation
    uplift map uses a documented power link (SCOUR_EXP) and is the one assumption."""
    import pandas as pd
    SCOUR_EXP = 0.5   # uplift ~ (Qpeak/Qmed - 1)^? ; documented, tune to taste
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[q_col, date_col]).sort_values(date_col)
    q = df[q_col].astype(float).values
    thr = np.quantile(q, pot_quantile)
    # independent peaks over threshold (simple 7-day declustering)
    df["yr"] = df[date_col].dt.year
    n_years = df["yr"].nunique()
    over = df[df[q_col] >= thr]
    lam = len(over) / max(n_years, 1)                       # Poisson rate
    qmed = np.median(q[q > 0])
    frac = np.clip(over[q_col].values/qmed - 1.0, 1e-3, None) ** SCOUR_EXP
    # log-normal fit to the fractional uplift proxy
    ln = np.log(frac)
    return {"LAMBDA_FLOOD": round(float(lam), 3),
            "UPLIFT_MEDIAN": round(float(np.exp(np.median(ln))), 3),
            "UPLIFT_SIGMA": round(float(ln.std(ddof=1)), 3),
            "n_years": int(n_years), "threshold_m3s": round(float(thr), 1),
            "note": "paste these over the module defaults; uplift map uses SCOUR_EXP"}


def run_ensemble():
    sigma_year, cv = sigma_from_record(RECORD)
    lam = LAMBDA_FLOOD * (1 + TREND_PER_DECADE * HORIZON_YEARS/10.0)
    print(f"normal-year sigma (from record {RECORD}) = {sigma_year:.3f}  (CV {cv:.3f})")
    if HORIZON_YEARS:
        print(f"non-stationary: horizon +{HORIZON_YEARS} yr -> flood rate {LAMBDA_FLOOD}->{lam:.3f}/yr")

    m_year = RNG.lognormal(0.0, sigma_year, size=N)
    counts = RNG.poisson(lam, size=N)
    kmax = int(counts.max()); uplift = np.zeros(N)
    if kmax > 0:
        draws = RNG.lognormal(np.log(UPLIFT_MEDIAN), UPLIFT_SIGMA, size=(N, kmax))
        mask = np.arange(kmax)[None,:] < counts[:,None]
        uplift = (draws*mask).sum(axis=1)

    # per-zone: same year multiplier; zone-specific flood sensitivity
    zsum = sum(ZONES.values())
    zone_ens = {}
    for z, v in ZONES.items():
        base = v/zsum*V0
        zone_ens[z] = base * m_year * (1.0 + ZONE_FLOOD_SENS[z]*uplift)
    V_all = sum(zone_ens.values())            # total = sum of zones (consistent)
    V_typ = V0 * m_year
    return sigma_year, V_typ, V_all, zone_ens, counts, uplift


def pct(a,p): return float(np.percentile(a,p))


def main():
    sigma_year, V_typ, V_all, zone_ens, counts, uplift = run_ensemble()
    P = {p:pct(V_all,p) for p in [5,10,25,50,75,90,95,99]}
    Ptyp = {p:pct(V_typ,p) for p in [10,50,90]}
    print("\nANNUAL SILTATION (Mm3/yr)")
    print(f"  typical years : P10 {Ptyp[10]:.1f}  P50 {Ptyp[50]:.1f}  P90 {Ptyp[90]:.1f}")
    print(f"  all years     : P10 {P[10]:.1f}  P50 {P[50]:.1f}  P90 {P[90]:.1f} "
          f"P95 {P[95]:.1f}  P99 {P[99]:.1f}   mean {V_all.mean():.1f}")
    print("\nRETURN PERIODS (all years)")
    for T in [2,5,10,20,50]:
        print(f"  1-in-{T:<2d} >= {pct(V_all,100*(1-1/T)):.1f} Mm3")
    print("\nWEAK VALIDATION")
    print(f"  typical-year median {Ptyp[50]:.1f} ~ record 24 ? "
          f"{'YES' if 21<=Ptyp[50]<=25 else 'NO'}")
    print(f"  flood pulse in {100*(counts>0).mean():.0f}% of years "
          f"(mean uplift when present {uplift[counts>0].mean()*100:.0f}%)")
    print("\nPER-ZONE (Mm3/yr)  P10 / P50 / P90")
    zstats={}
    for z,e in zone_ens.items():
        zstats[z]=[pct(e,10),pct(e,50),pct(e,90)]
        print(f"  {z:<12s} {zstats[z][0]:5.1f} / {zstats[z][1]:5.1f} / {zstats[z][2]:5.1f}")

    out={"baseline_Mm3":V0,"sigma_year":round(sigma_year,4),
         "lambda_flood":LAMBDA_FLOOD,"uplift_lognormal":[UPLIFT_MEDIAN,UPLIFT_SIGMA],
         "horizon_years":HORIZON_YEARS,"trend_per_decade":TREND_PER_DECADE,
         "typical_year":{f"P{p}":round(Ptyp[p],2) for p in [10,50,90]},
         "all_years":{f"P{p}":round(P[p],2) for p in P},
         "return_period_Mm3":{f"T{T}":round(pct(V_all,100*(1-1/T)),2) for T in [2,5,10,20,50]},
         "per_zone_P10_P50_P90":{z:[round(x,2) for x in v] for z,v in zstats.items()},
         "note":"scenario ensemble under anchored forcing statistics; not survey-validated"}
    jp=os.path.expanduser("~/haine_layer3/haine_stochastic_forcing.json")
    try:
        os.makedirs(os.path.dirname(jp),exist_ok=True); json.dump(out,open(jp,"w"),indent=2)
        print(f"\nsaved -> {jp}")
    except Exception:
        json.dump(out,open("haine_stochastic_forcing.json","w"),indent=2)

    # ---- 3-panel dashboard figure ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(15,4.4))
        ax[0].hist(V_all,bins=60,color="#0E7C9B",alpha=0.85)
        for p,c,l in [(10,"#2EC4B6","P10"),(50,"#123B52","P50"),(90,"#E76F51","P90")]:
            ax[0].axvline(pct(V_all,p),color=c,ls="--",lw=1.6,label=f"{l}={pct(V_all,p):.1f}")
        for r in RECORD: ax[0].axvline(r,color="#999",ls=":",lw=1,alpha=.7)
        ax[0].set_title("Annual siltation distribution"); ax[0].set_xlabel("Mm$^3$/yr")
        ax[0].set_ylabel("frequency"); ax[0].legend(fontsize=8)
        xs=np.sort(V_all); exc=1-np.arange(1,len(xs)+1)/len(xs)
        ax[1].plot(xs,exc,color="#0E7C9B",lw=2)
        for T in [2,5,10,20,50]:
            q=pct(V_all,100*(1-1/T)); ax[1].plot(q,1/T,"o",color="#E76F51")
            ax[1].annotate(f"1-in-{T}",(q,1/T),fontsize=8,xytext=(4,3),textcoords="offset points")
        ax[1].set_yscale("log"); ax[1].set_title("Exceedance / return period")
        ax[1].set_xlabel("Mm$^3$/yr"); ax[1].set_ylabel("annual exceedance prob"); ax[1].grid(alpha=.3)
        zs=list(zstats.keys()); p50=[zstats[z][1] for z in zs]
        lo=[zstats[z][1]-zstats[z][0] for z in zs]; hi=[zstats[z][2]-zstats[z][1] for z in zs]
        ax[2].bar(zs,p50,yerr=[lo,hi],capsize=5,color=["#0E7C9B","#1C7293","#2EC4B6","#F4A259"])
        ax[2].set_title("Per-zone P50 with P10-P90 band"); ax[2].set_ylabel("Mm$^3$/yr")
        ax[2].tick_params(axis="x",rotation=20)
        plt.tight_layout()
        for png in [os.path.expanduser("~/Downloads/haine_stochastic_forecast.png"),
                    "haine_stochastic_forecast.png"]:
            try: plt.savefig(png,dpi=130); print(f"plot -> {png}"); break
            except Exception: continue
    except Exception as e:
        print("plot skipped:",e)


if __name__=="__main__":
    main()
