"""
HAINE Layer 2 - MULTI-YEAR SPM extraction (adapted from 07_...).

Extracts Nechad SPM per satellite pass for SEVERAL year-windows, so each year's
sediment-supply signal can be paired with that year's dredged-volume label for
inter-annual calibration (Layer 3).

KEY DIFFERENCE vs the single-year script:
  * loops over YEAR_WINDOWS (Jun->May, monsoon-centered, mapped to the FY it feeds)
  * adds LANDSAT-8 (LC08) as a source -- Landsat-9 (LC09) did NOT exist before
    Sept 2021, so L9 is gated to windows starting 2021+. L8 shares band structure
    and scaling with L9, so the same Nechad code reuses.
  * exports each year to its OWN Drive folder: HAINE_SPM_<FY>
  * SAME AOI + scale for every year -> cubes stay comparable after fusion.

Sparsity note: 2016 has only Sentinel-2A (S2B came online 2017), so that window
is thinner -> the temporal kriging in 08_ does more gap-filling there.

Auth: earthengine authenticate (once).  Run: python 07b_multiyear_spm_extraction.py
"""
import os
import ee

# ---------------------------------------------------------------- CONFIG
GEE_PROJECT     = os.environ.get("GEE_PROJECT", "your-gee-project-id")     # your GEE project
AOI_BOUNDS      = [75.5, 9.5, 76.8, 10.5]         # SAME domain as the 2024-25 run
CLOUD_COVER_MAX = 50                               # a touch higher for thin historical years
EXPORT_SCALE_S2 = 10
EXPORT_SCALE_LS = 30
NECHAD_A, NECHAD_C = 355.85, 0.1728

# window start (Jun Y) -> end (Jun Y+1); label is the FY its monsoon feeds
YEAR_WINDOWS = [
    ("2016-06-01", "2017-06-01", "FY2016-17"),
    ("2017-06-01", "2018-06-01", "FY2017-18"),
    ("2019-06-01", "2020-06-01", "FY2019-20"),
    # ("2024-06-01","2025-06-01","FY2024-25"),   # already have this cube; uncomment to redo
]
# --------------------------------------------------------------------------


def mask_s2(image):
    scl = image.select("SCL")
    good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(good)


def mask_ls(image):
    qa = image.select("QA_PIXEL")
    m = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(m)


def nechad(rho):
    denom = ee.Image(1).subtract(rho.divide(NECHAD_C))
    return rho.multiply(NECHAD_A).divide(denom).rename("spm_mgL")


def s2_coll(aoi, start, end):
    c = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
         .filterBounds(aoi).filterDate(start, end)
         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_MAX)).map(mask_s2))
    def add(img):
        spm = nechad(img.select("B4").multiply(0.0001))
        return img.addBands(spm).set("date", img.date().format("YYYY-MM-dd"))
    return c.map(add)


def landsat_coll(aoi, start, end, coll_id):
    c = (ee.ImageCollection(coll_id).filterBounds(aoi).filterDate(start, end)
         .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_COVER_MAX)).map(mask_ls))
    def add(img):
        red = img.select("SR_B4").multiply(0.0000275).add(-0.2)
        return img.addBands(nechad(red)).set("date", img.date().format("YYYY-MM-dd"))
    return c.map(add)


def export_coll(coll, aoi, prefix, folder, scale):
    n = coll.size().getInfo()
    if n == 0:
        print(f"  {prefix}: 0 scenes (skipped)"); return
    print(f"  {prefix}: {n} scenes -> exporting to Drive/{folder}")
    lst = coll.toList(n)
    for i in range(n):
        img = ee.Image(lst.get(i))
        d = img.get("date").getInfo()
        ee.batch.Export.image.toDrive(
            image=img.select("spm_mgL").clip(aoi),
            description=f"{prefix}_spm_{d}", folder=folder, region=aoi,
            scale=scale, maxPixels=1e9).start()


def main():
    ee.Initialize(project=GEE_PROJECT)
    aoi = ee.Geometry.Rectangle(AOI_BOUNDS)
    for start, end, fy in YEAR_WINDOWS:
        folder = f"HAINE_SPM_{fy}"
        yr = int(start[:4])
        print(f"\n=== {fy}  ({start} .. {end}) -> Drive/{folder} ===")
        export_coll(s2_coll(aoi, start, end), aoi, "s2", folder, EXPORT_SCALE_S2)
        # Landsat-8 for all years; Landsat-9 only from 2021 onward
        export_coll(landsat_coll(aoi, start, end, "LANDSAT/LC08/C02/T1_L2"),
                    aoi, "l8", folder, EXPORT_SCALE_LS)
        if yr >= 2021:
            export_coll(landsat_coll(aoi, start, end, "LANDSAT/LC09/C02/T1_L2"),
                        aoi, "l9", folder, EXPORT_SCALE_LS)
    print("\nAll export tasks started. Monitor: https://code.earthengine.google.com/tasks")
    print("Then download each Drive/HAINE_SPM_<FY> folder and fuse+clip per year (see runbook).")


if __name__ == "__main__":
    main()
