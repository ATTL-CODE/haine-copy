"""
HAINE — Layer 2 Satellite Fusion
Step 1: Sentinel-2 extraction + Stumpf log-ratio band computation for
        satellite-derived bathymetry (SDB) over the Kochi domain.

Requires: earthengine-api (pip install earthengine-api --break-system-packages)
Auth:     run `earthengine authenticate` once before using this script,
          or use ee.Authenticate() interactively the first time.

Output:   a water-masked band-ratio GeoTIFF exported to Google Drive
          (or locally if you set LOCAL_EXPORT = True and have geemap installed).

See 00_SDB_PIPELINE_STEPS.md for the full pipeline context and limitations.
"""

import ee

# ----------------------------------------------------------------------
# CONFIG — edit these
# ----------------------------------------------------------------------
GEE_PROJECT = "your-gee-project-id"          # TODO: set your GEE cloud project
AOI_BOUNDS = [75.5, 9.5, 76.8, 10.5]          # [minLon, minLat, maxLon, maxLat] — same as ERA5 pipeline
START_DATE = "2024-11-01"                      # post-monsoon window recommended: clearer water
END_DATE   = "2025-02-28"
CLOUD_COVER_MAX = 15                           # percent
NDWI_THRESHOLD = 0.0                           # standard water/land NDWI cutoff
EXPORT_DESCRIPTION = "kochi_sdb_ratio_2025"
EXPORT_FOLDER = "HAINE_SDB"
EXPORT_SCALE_M = 10                            # Sentinel-2 native resolution for B2/B3/B4/B8
LOCAL_EXPORT = False                           # True = try direct local download via geemap (small AOIs only)


def mask_s2_clouds(image):
    """Cloud/shadow mask using the Scene Classification Layer (SCL)."""
    scl = image.select("SCL")
    # Keep: 6 = water, 4 = vegetation, 5 = bare soil, 11 = snow (irrelevant here)
    # Reject: 3 = cloud shadow, 8/9/10 = clouds/cirrus
    good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(good)


def add_ndwi(image):
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")  # Green, NIR
    return image.addBands(ndwi)


def add_stumpf_ratio(image, n=1000):
    """
    Stumpf et al. (2003) log-ratio transform.
    Blue = B2, Green = B3 for Sentinel-2.
    Result correlates (roughly linearly, after calibration) with depth
    in optically shallow water.
    """
    blue = image.select("B2").multiply(n).log()
    green = image.select("B3").multiply(n).log()
    ratio = blue.divide(green).rename("sdb_ratio")
    return image.addBands(ratio)


def main():
    ee.Initialize(project=GEE_PROJECT)

    aoi = ee.Geometry.Rectangle(AOI_BOUNDS)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_COVER_MAX))
        .map(mask_s2_clouds)
        .map(add_ndwi)
    )

    n_images = s2.size().getInfo()
    print(f"Found {n_images} Sentinel-2 scenes matching filters.")
    if n_images == 0:
        raise SystemExit(
            "No scenes matched. Widen START_DATE/END_DATE or raise CLOUD_COVER_MAX."
        )

    # Median composite reduces sun glint and transient turbidity noise —
    # important given monsoon-driven sediment plumes near the Periyar outflow.
    composite = s2.median().clip(aoi)
    composite = add_stumpf_ratio(composite)

    # Water mask: only keep pixels the composite agrees are water
    water_mask = composite.select("NDWI").gt(NDWI_THRESHOLD)
    sdb_ratio_masked = composite.select("sdb_ratio").updateMask(water_mask)

    if LOCAL_EXPORT:
        try:
            import geemap
            geemap.ee_export_image(
                sdb_ratio_masked,
                filename="kochi_sdb_ratio.tif",
                scale=EXPORT_SCALE_M,
                region=aoi,
                file_per_band=False,
            )
            print("Exported locally as kochi_sdb_ratio.tif")
            return
        except ImportError:
            print("geemap not installed — falling back to Drive export.")

    task = ee.batch.Export.image.toDrive(
        image=sdb_ratio_masked,
        description=EXPORT_DESCRIPTION,
        folder=EXPORT_FOLDER,
        region=aoi,
        scale=EXPORT_SCALE_M,
        maxPixels=1e9,
    )
    task.start()
    print(
        f"Export task '{EXPORT_DESCRIPTION}' started. "
        f"Check progress at https://code.earthengine.google.com/tasks "
        f"— file will land in Drive folder '{EXPORT_FOLDER}'."
    )


if __name__ == "__main__":
    main()
