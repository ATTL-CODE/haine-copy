"""
HAINE — Layer 2 Satellite Fusion
Suspended Particulate Matter (SPM) / turbidity extraction from Sentinel-2
MSI and Landsat-9 OLI, using the Nechad et al. (2010) single-band algorithm.

Unlike bathymetry (SDB), this works fine in deep and turbid water — it's
directly usable across the whole Kochi domain, including the monsoon
sediment plume near the Periyar outflow.

Requires: earthengine-api (pip install earthengine-api --break-system-packages)
Auth:     `earthengine authenticate` once, or ee.Authenticate() interactively.

Output: per-date SPM GeoTIFFs exported to Drive, one per satellite pass in
        range — these feed directly into 08_temporal_kriging_fusion.py.
"""

import ee

# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------
GEE_PROJECT = "your-gee-project-id"          # TODO: same project as SDB pipeline
AOI_BOUNDS = [75.5, 9.5, 76.8, 10.5]          # same domain as ERA5 / SDB
START_DATE = "2024-06-01"                      # cover a full monsoon cycle if possible
END_DATE   = "2025-06-01"
CLOUD_COVER_MAX = 40                           # higher tolerance than SDB — turbidity
                                                # retrieval is more robust to partial cloud
EXPORT_FOLDER = "HAINE_SPM"
EXPORT_SCALE_M = 10                            # Sentinel-2 native; Landsat exports at 30m

# Nechad et al. (2010) calibration constants for the red band (B4 / Landsat B4).
# These are generic literature defaults — recalibrate against in-situ SSC
# measurements if/when NIO or field data becomes available.
NECHAD_A = 355.85
NECHAD_C = 0.1728


def mask_s2_clouds(image):
    scl = image.select("SCL")
    good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(good)


def mask_landsat_clouds(image):
    qa = image.select("QA_PIXEL")
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(shadow_bit).eq(0))
    return image.updateMask(mask)


def nechad_spm(reflectance_band):
    """
    SPM (mg/L) = A * rho / (1 - rho/C) + offset-style Nechad formulation,
    simplified single-band form using top-of-atmosphere-corrected reflectance.
    """
    rho = reflectance_band
    denom = ee.Image(1).subtract(rho.divide(NECHAD_C))
    spm = rho.multiply(NECHAD_A).divide(denom).rename("spm_mgL")
    return spm


def process_sentinel2(aoi):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_MAX))
        .map(mask_s2_clouds)
    )

    def add_spm(image):
        red = image.select("B4").multiply(0.0001)  # scale to reflectance
        spm = nechad_spm(red)
        return image.addBands(spm).set(
            "date", image.date().format("YYYY-MM-dd")
        )

    return s2.map(add_spm)


def process_landsat9(aoi):
    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_COVER_MAX))
        .map(mask_landsat_clouds)
    )

    def add_spm(image):
        # Landsat Collection 2 L2 surface reflectance scaling
        red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
        spm = nechad_spm(red)
        return image.addBands(spm).set(
            "date", image.date().format("YYYY-MM-dd")
        )

    return l9.map(add_spm)


def export_collection(collection, aoi, prefix, scale):
    n = collection.size().getInfo()
    print(f"{prefix}: {n} scenes found, starting exports...")
    image_list = collection.toList(n)
    for i in range(n):
        img = ee.Image(image_list.get(i))
        date_str = img.get("date").getInfo()
        task = ee.batch.Export.image.toDrive(
            image=img.select("spm_mgL").clip(aoi),
            description=f"{prefix}_spm_{date_str}",
            folder=EXPORT_FOLDER,
            region=aoi,
            scale=scale,
            maxPixels=1e9,
        )
        task.start()
    print(f"{prefix}: {n} export tasks started. "
          f"Monitor at https://code.earthengine.google.com/tasks")


def main():
    ee.Initialize(project=GEE_PROJECT)
    aoi = ee.Geometry.Rectangle(AOI_BOUNDS)

    s2_spm = process_sentinel2(aoi)
    l9_spm = process_landsat9(aoi)

    export_collection(s2_spm, aoi, "s2", EXPORT_SCALE_M)
    export_collection(l9_spm, aoi, "l9", 30)


if __name__ == "__main__":
    main()
