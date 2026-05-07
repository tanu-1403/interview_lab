"""
00_data_download/gee_download.py
=================================
Google Earth Engine script to download:
  - Landsat 8/9 TIRS Land Surface Temperature (LST)
  - Sentinel-2 NDVI and NDBI
  - MODIS Terra LST monthly time series

Study Area: Surat, Gujarat, India (adaptable)
Time Range: 2015-2024
"""

import ee
import geemap
import os
import json
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

STUDY_AREA_NAME = "Surat"
# Surat bounding box [west, south, east, north]
BBOX = [72.7, 21.0, 73.1, 21.35]
START_YEAR = 2015
END_YEAR = 2024
OUTPUT_DIR = "../data/raw"

# ── Initialize GEE ─────────────────────────────────────────────────────────────

def initialize_gee():
    """Authenticate and initialize Google Earth Engine."""
    try:
        ee.Initialize(project='urban-heat-atlas')
        print("✅ GEE initialized successfully.")
    except Exception:
        print("🔑 Authenticating GEE...")
        ee.Authenticate()
        ee.Initialize(project='urban-heat-atlas')
        print("✅ GEE initialized.")

# ── Study Area Geometry ─────────────────────────────────────────────────────────

def get_study_area():
    """Define the study area geometry."""
    return ee.Geometry.BBox(*BBOX)

# ── Landsat LST Retrieval ───────────────────────────────────────────────────────

def compute_lst_landsat(image):
    """
    Compute Land Surface Temperature from Landsat 8/9 TIRS Band 10.
    Uses the split-window algorithm approximation.
    """
    # Brightness Temperature (BT) from Band 10
    bt = image.select('ST_B10').multiply(0.00341802).add(149.0)
    
    # NDVI for emissivity correction
    red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
    nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    # Proportion of vegetation (Pv)
    ndvi_min = ee.Number(0.2)
    ndvi_max = ee.Number(0.8)
    pv = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).pow(2).rename('Pv')
    
    # Land Surface Emissivity (LSE)
    lse = pv.multiply(0.004).add(0.986).rename('LSE')
    
    # LST in Celsius
    lst = bt.divide(
        bt.divide(14388).multiply(
            lse.log().add(1)
        ).exp()
    ).subtract(273.15).rename('LST_C')
    
    return image.addBands([ndvi, lst]) \
                .set('system:time_start', image.get('system:time_start'))

def get_landsat_collection(year, season='summer'):
    """Get cloud-masked Landsat collection for a given year and season."""
    area = get_study_area()
    
    # Summer months (peak heat) — April to June for India
    date_start = f"{year}-04-01"
    date_end = f"{year}-06-30"
    
    # Landsat 8 (2015-2021) and Landsat 9 (2022+)
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
           .filterBounds(area) \
           .filterDate(date_start, date_end) \
           .filter(ee.Filter.lt('CLOUD_COVER', 20))
    
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
           .filterBounds(area) \
           .filterDate(date_start, date_end) \
           .filter(ee.Filter.lt('CLOUD_COVER', 20))
    
    collection = l8.merge(l9).map(compute_lst_landsat)
    return collection.median().clip(area)

# ── Sentinel-2 NDVI / NDBI ─────────────────────────────────────────────────────

def compute_indices_s2(image):
    """Compute NDVI and NDBI from Sentinel-2."""
    # NDVI — vegetation
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    # NDBI — built-up index
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    # MNDWI — water index (to mask water bodies)
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    return image.addBands([ndvi, ndbi, mndwi])

def get_sentinel2_collection(year):
    """Get cloud-masked Sentinel-2 collection."""
    area = get_study_area()
    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
             .filterBounds(area) \
             .filterDate(f"{year}-04-01", f"{year}-06-30") \
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
             .map(compute_indices_s2) \
             .median() \
             .clip(area)

# ── MODIS Monthly LST Time Series ──────────────────────────────────────────────

def get_modis_lst_timeseries():
    """
    Get MODIS Terra monthly LST time series for trend analysis.
    Returns a multi-band image with one band per month-year.
    """
    area = get_study_area()
    
    def process_month(date_str):
        start = ee.Date(date_str)
        end = start.advance(1, 'month')
        img = ee.ImageCollection('MODIS/061/MOD11A2') \
                .filterBounds(area) \
                .filterDate(start, end) \
                .select('LST_Day_1km') \
                .mean() \
                .multiply(0.02) \
                .subtract(273.15) \
                .clip(area) \
                .rename(date_str[:7].replace('-', '_'))
        return img
    
    # Generate monthly dates
    months = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            months.append(f"{year}-{month:02d}-01")
    
    return months  # Return list for sequential processing

# ── Export Functions ────────────────────────────────────────────────────────────

def export_to_drive(image, description, folder='HeatAtlas', scale=30):
    """Export a GEE image to Google Drive."""
    area = get_study_area()
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=area,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e13
    )
    task.start()
    print(f"📤 Export started: {description} — check GEE Tasks tab")
    return task

def export_collection_local(image, filename, scale=30):
    """Export GEE image directly to local files using geemap."""
    area = get_study_area()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    geemap.ee_export_image(
        image,
        filename=out_path,
        scale=scale,
        region=area,
        file_per_band=False
    )
    print(f"✅ Saved: {out_path}")

# ── Main Pipeline ───────────────────────────────────────────────────────────────

def main():
    initialize_gee()
    area = get_study_area()
    
    print(f"\n🌍 Study Area: {STUDY_AREA_NAME}")
    print(f"📅 Period: {START_YEAR}–{END_YEAR}\n")
    
    # 1. Download annual Landsat LST (summer composite)
    print("── Step 1: Landsat LST annual composites ──")
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"  Processing {year}...")
        landsat_img = get_landsat_collection(year)
        lst_band = landsat_img.select('LST_C')
        ndvi_band = landsat_img.select('NDVI')
        composite = lst_band.addBands(ndvi_band)
        export_to_drive(composite, f"landsat_lst_ndvi_{year}", scale=30)
    
    # 2. Download Sentinel-2 NDVI/NDBI for latest year
    print("\n── Step 2: Sentinel-2 NDVI + NDBI ──")
    for year in [2020, 2022, 2024]:
        s2_img = get_sentinel2_collection(year)
        export_to_drive(
            s2_img.select(['NDVI', 'NDBI', 'MNDWI']),
            f"sentinel2_indices_{year}",
            scale=10
        )
    
    # 3. Download MODIS monthly time series (lower res, full period)
    print("\n── Step 3: MODIS LST time series ──")
    modis_collection = ee.ImageCollection('MODIS/061/MOD11A2') \
        .filterBounds(area) \
        .filterDate(f'{START_YEAR}-01-01', f'{END_YEAR}-12-31') \
        .select('LST_Day_1km') \
        .map(lambda img: img.multiply(0.02).subtract(273.15).clip(area))
    
    # Export as CSV of zonal statistics
    modis_ts = modis_collection.toBands()
    export_to_drive(modis_ts, 'modis_lst_monthly_timeseries', scale=1000)
    
    print("\n✅ All export tasks submitted to GEE.")
    print("📋 Check https://code.earthengine.google.com/tasks for progress.")
    print("📁 Files will appear in Google Drive > HeatAtlas/ folder.")

if __name__ == "__main__":
    main()
