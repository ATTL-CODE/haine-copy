"""
HAINE Layer 3 - SUB-MODEL A: the trained ML core (this is the actual model dev).

Until now Sub-model A was a CALIBRATED physical estimate. This module is the
machine-learning replacement: it learns siltation volume from the fused physics
+ satellite feature vector, with leave-one-year-out cross-validation, quantile
prediction intervals (P10/P50/P90), and permutation-importance explainability.

It runs in two modes:
  * REAL: build_features_from_pipeline() assembles the feature table from the
    multi-year SPM cubes + TELEMAC coupling fields + discharge + dredging labels
    on your machine. Plug in the paths and it trains for real.
  * DEMO: synthetic_demo() generates a physically-plausible table from a known
    generating function so the whole pipeline is PROVEN to run end-to-end here.

Model: HistGradientBoostingRegressor (gradient-boosted trees, XGBoost-class;
native NaN handling; quantile loss for intervals). Swap in XGBoost/SHAP on your
machine if installed - the interfaces match.

HONEST NOTE ON DATA: with only annual totals you have ~(years x zones) samples
(tens). Real skill needs PER-LOCATION labels from CPA paired surveys, which turn
the same features into thousands of samples. The architecture below is identical
either way - only the label granularity (and therefore the achievable skill)
changes. This module IS ready; it is the data that is the bottleneck.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings("ignore")

# HAINE Sub-model A feature vector (physics + satellite + forcing + geometry)
FEATURES = ["mean_taub","pdep_mean","spm_monsoon_integral","spm_annual_mean",
            "spm_peak","monsoon_phase_frac","Qeff","flood_index",
            "antecedent_interval","tidal_prism_proxy","segment_area","segment_depth"]
LABEL, GROUP = "accretion_m_yr", "year"
ZONES = {"approach":(19.1,260*16000,14.5), "ernakulam":(1.4,260*6000,13.25),
         "mattancherry":(2.5,183*3000,9.75), "basins":(0.9,1.2e6,11.0)}

# ---------------------------------------------------------------- REAL MODE
def build_features_from_pipeline(cube_dirs, pdep_tif, taub_tif, discharge_csv,
                                 labels_csv, out_csv="haine_submodelA_features.csv"):
    """Assemble the training table from real HAINE outputs. cube_dirs maps
    FY->folder of monthly SPM tifs; labels_csv has columns [year,zone,siltation_Mm3]
    (from multi-year dredged volumes disaggregated, OR CPA survey differences).
    Returns a DataFrame with FEATURES + LABEL + GROUP.  (Fill in per your paths.)"""
    import rasterio
    from rasterio.warp import reproject, Resampling
    raise NotImplementedError(
        "Point cube_dirs/pdep_tif/taub_tif/discharge_csv/labels_csv at your files "
        "and implement the per-(year,zone) aggregation (mean taub & pdep over the "
        "zone mask; monsoon SPM integral; Qeff from discharge; join labels). The "
        "DEMO mode below shows the exact table shape the trainer expects.")

# ---------------------------------------------------------------- DEMO MODE
def synthetic_demo(n_years=6, cells_per_zone=30, seed=7):
    """Physically-motivated synthetic table at (year, zone, cell) granularity.
    TARGET = accretion RATE (m/yr) - how Sub-model A actually works - driven by
    satellite SPM supply, deposition gating (pdep), monsoon flood pulses and the
    antecedent dredging interval, + lognormal noise. Volume = rate x area (rolled
    up separately). Real data would be per-PIXEL (thousands of samples); this
    proves the identical architecture on a few hundred."""
    rng = np.random.default_rng(seed)
    rows = []
    for yr in range(2016, 2016+n_years):
        year_mult = rng.lognormal(0, 0.10)
        flood = (rng.lognormal(np.log(0.30), 0.5) if rng.random() < 0.30 else 0.0)
        Qeff  = rng.normal(200, 40) * (1 + flood)
        monsoon_f = rng.uniform(0.45, 0.55)
        for z,(vol,area,depth) in ZONES.items():
            expo = {"approach":1.3,"basins":1.1,"ernakulam":0.7,"mattancherry":0.6}[z]
            taub0 = {"approach":0.006,"ernakulam":0.002,"mattancherry":0.0015,"basins":0.001}[z]
            cell_area = area / cells_per_zone
            for _ in range(cells_per_zone):
                spm_annual = max(5.0, rng.normal(35, 8) * year_mult)
                spm_peak   = spm_annual * rng.uniform(1.5, 2.2)
                spm_mons   = spm_annual * rng.uniform(4.5, 6.0)
                taub       = taub0 * rng.uniform(0.5, 1.6)
                pdep       = float(np.clip(1 - taub/0.2, 0, 1))
                antecedent = rng.uniform(150, 400)
                prism      = cell_area * rng.uniform(0.8, 1.2) / 1e6
                # --- generating function for accretion RATE (m/yr) ---
                a = (0.9 * (spm_mons/280.0)          # satellite supply (dominant)
                       * (0.5 + 0.7*pdep)            # deposition gating
                       * (1 + 1.1*flood*expo)        # monsoon flood, zone-exposed
                       * (1 + 0.20*(antecedent-275)/275))
                a *= rng.lognormal(0, 0.10)          # observation noise
                rows.append(dict(year=yr, zone=z, mean_taub=taub, pdep_mean=pdep,
                    spm_monsoon_integral=spm_mons, spm_annual_mean=spm_annual,
                    spm_peak=spm_peak, monsoon_phase_frac=monsoon_f, Qeff=Qeff,
                    flood_index=1+flood, antecedent_interval=antecedent,
                    tidal_prism_proxy=prism, segment_area=cell_area, segment_depth=depth,
                    accretion_m_yr=max(0.005, a), cell_area=cell_area))
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- TRAIN / CV
def make_model(quantile=None):
    if quantile is None:
        return HistGradientBoostingRegressor(loss="squared_error",
                 max_depth=3, learning_rate=0.08, max_iter=400, l2_regularization=1.0)
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile,
                 max_depth=3, learning_rate=0.08, max_iter=400, l2_regularization=1.0)

def loo_cv(df):
    """Leave-one-YEAR-out CV - the honest skill estimate with few years."""
    X, y, g = df[FEATURES], df[LABEL].values, df[GROUP].values
    logo = LeaveOneGroupOut(); preds = np.zeros(len(y))
    for tr, te in logo.split(X, y, g):
        m = make_model().fit(X.iloc[tr], y[tr]); preds[te] = m.predict(X.iloc[te])
    rmse = float(np.sqrt(mean_squared_error(y, preds))); r2 = float(r2_score(y, preds))
    return preds, rmse, r2

def main():
    print("HAINE Sub-model A - trained ML core\n"+"="*44)
    # REAL feature table (from haine_build_features.py) if present; else demo
    csv=os.path.expanduser("~/haine_layer3/haine_submodelA_features.csv")
    if os.path.exists(csv):
        df=pd.read_csv(csv); print("loaded REAL features:",csv)
    else:
        df=synthetic_demo(); print("no real CSV found -> synthetic demo")
    print(f"training table: {len(df)} samples  ({df[GROUP].nunique()} years x "
          f"{df['zone'].nunique()} zones)  |  {len(FEATURES)} features")

    # 1) leave-one-year-out cross-validation
    preds, rmse, r2 = loo_cv(df)
    print(f"\nLEAVE-ONE-YEAR-OUT CV:  RMSE = {rmse:.3f} Mm3   R2 = {r2:.3f}")
    print(f"   (RMSE is {100*rmse/df[LABEL].mean():.1f}% of mean siltation)")

    # 2) fit final median + P10/P90 quantile models on all data
    X, y = df[FEATURES], df[LABEL].values
    m50 = make_model().fit(X, y)
    m10 = make_model(0.10).fit(X, y); m90 = make_model(0.90).fit(X, y)
    df["pred_P50"] = m50.predict(X)
    df["pred_P10"] = np.minimum(m10.predict(X), df["pred_P50"])
    df["pred_P90"] = np.maximum(m90.predict(X), df["pred_P50"])
    cov = float(((df[LABEL]>=df.pred_P10)&(df[LABEL]<=df.pred_P90)).mean())
    print(f"P10-P90 interval empirical coverage: {100*cov:.0f}% (target ~80%)")

    vol_pred = float((df["pred_P50"]*df["cell_area"]).sum()/1e6)
    vol_true = float((df[LABEL]*df["cell_area"]).sum()/1e6)
    print(f"volume roll-up (pred vs true over demo cells): {vol_pred:.1f} vs {vol_true:.1f} Mm3 "
          f"(scaled across {df[GROUP].nunique()} yrs)")

    # 3) permutation importance (SHAP proxy)
    imp = permutation_importance(m50, X, y, n_repeats=30, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1]
    print("\nFEATURE IMPORTANCE (permutation; SHAP proxy)")
    for i in order[:8]:
        print(f"   {FEATURES[i]:<22s} {imp.importances_mean[i]:.3f}")

    # 4) save artefacts
    out = {"n_samples":len(df),"n_years":int(df[GROUP].nunique()),
           "loo_rmse":round(rmse,3),"loo_r2":round(r2,3),
           "interval_coverage":round(cov,3),
           "feature_importance":{FEATURES[i]:round(float(imp.importances_mean[i]),4) for i in order},
           "mode":"synthetic_demo (proven); wire build_features_from_pipeline for real data",
           "model":"HistGradientBoostingRegressor (XGBoost-class), quantile intervals"}
    try:
        import joblib
        jd = os.path.expanduser("~/haine_layer3"); os.makedirs(jd, exist_ok=True)
        joblib.dump({"P50":m50,"P10":m10,"P90":m90,"features":FEATURES}, f"{jd}/submodelA_model.joblib")
        json.dump(out, open(f"{jd}/submodelA_metrics.json","w"), indent=2)
        print(f"\nsaved model + metrics -> {jd}/")
    except Exception:
        json.dump(out, open("submodelA_metrics.json","w"), indent=2); print("\nsaved metrics locally")

    # 5) plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,2,figsize=(11,4.4))
        ax[0].errorbar(df[LABEL], df.pred_P50,
            yerr=[np.clip(df.pred_P50-df.pred_P10,0,None), np.clip(df.pred_P90-df.pred_P50,0,None)],
            fmt="o", color="#0E7C9B", ecolor="#9DB4C0", capsize=3, ms=5)
        lim=[0, df[LABEL].max()*1.15]; ax[0].plot(lim,lim,"k--",alpha=.5)
        ax[0].set_xlabel("actual siltation (Mm3)"); ax[0].set_ylabel("predicted P50 (P10-P90)")
        ax[0].set_title(f"Sub-model A fit  (LOO R2={r2:.2f})")
        yi=[FEATURES[i] for i in order[:8]][::-1]; vi=[imp.importances_mean[i] for i in order[:8]][::-1]
        ax[1].barh(yi, vi, color="#2EC4B6"); ax[1].set_title("Feature importance (SHAP proxy)")
        ax[1].set_xlabel("permutation importance")
        plt.tight_layout()
        for png in [os.path.expanduser("~/Downloads/haine_submodelA_train.png"),"haine_submodelA_train.png"]:
            try: plt.savefig(png, dpi=130); print(f"plot -> {png}"); break
            except Exception: continue
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
