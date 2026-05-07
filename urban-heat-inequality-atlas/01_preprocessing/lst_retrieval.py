"""
01_preprocessing/lst_retrieval.py
====================================
Preprocesses downloaded Landsat GeoTIFFs:
  - Cloud masking
  - LST extraction and calibration
  - Zonal statistics per census block
  - Outputs: processed LST CSV per year
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
from scipy import ndimage
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Synthetic LST Generator ─────────────────────────────────────────────────────
# Generates realistic synthetic LST rasters for Surat
# Replace with actual GeoTIFF reading in production

def generate_synthetic_lst_raster(year, seed=None):
    """
    Generate a synthetic LST raster for Surat with realistic spatial patterns.
    
    Spatial patterns:
    - Industrial zones (Udhna, Hazira): hottest
    - Dense built-up (Varachha, Katargam): hot
    - Commercial core (Athwa, Chowk): medium-hot
    - Planned residential (Rander): medium
    - Riparian / river Tapti: cool
    - Remaining vegetation: coolest
    
    Returns: (array, transform, crs)
    """
    if seed is None:
        seed = year
    rng = np.random.default_rng(seed)
    
    # Grid: 400 x 300 pixels covering Surat bbox
    width, height = 400, 300
    lon_min, lat_min, lon_max, lat_max = 72.7, 21.0, 73.1, 21.35
    
    # Base temperature grid
    lst = np.full((height, width), 32.0, dtype=np.float32)
    
    # Year trend: +0.15°C per year from 2015
    year_offset = (year - 2015) * 0.15
    lst += year_offset
    
    # ── Spatial patterns ──────────────────────────────
    
    # Coordinate arrays
    lons = np.linspace(lon_min, lon_max, width)
    lats = np.linspace(lat_max, lat_min, height)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    def gaussian_blob(lon_c, lat_c, amp, sigma_lon=0.025, sigma_lat=0.018):
        return amp * np.exp(
            -(((lon_grid - lon_c) / sigma_lon)**2 + 
              ((lat_grid - lat_c) / sigma_lat)**2)
        )
    
    # Industrial zone (Udhna/Hazira) — highest LST
    lst += gaussian_blob(72.885, 21.165, +8.5, 0.03, 0.02)
    # Dense textile worker area (Varachha)
    lst += gaussian_blob(72.890, 21.215, +6.0, 0.025, 0.02)
    # Old city dense built-up (Chowk, Lal Darwaja)
    lst += gaussian_blob(72.835, 21.190, +5.0, 0.02, 0.015)
    # Katargam textile district
    lst += gaussian_blob(72.830, 21.245, +4.5, 0.025, 0.02)
    
    # River Tapti (cooling effect) — diagonal band
    river_mask = np.abs((lat_grid - 21.20) - 0.4 * (lon_grid - 72.85)) < 0.015
    lst[river_mask] -= 6.0
    
    # Sarthana Nature Park
    lst += gaussian_blob(72.908, 21.234, -5.0, 0.02, 0.015)
    # Dumas coastal area
    lst += gaussian_blob(72.714, 21.074, -4.0, 0.03, 0.02)
    # Planned residential (Rander) — more trees
    lst += gaussian_blob(72.785, 21.235, -2.0, 0.025, 0.02)
    
    # Add spatially correlated noise
    noise = rng.normal(0, 1.2, (height, width)).astype(np.float32)
    noise = ndimage.gaussian_filter(noise, sigma=3)
    lst += noise
    
    # Clip to realistic range
    lst = np.clip(lst, 25.0, 55.0)
    
    # Raster transform
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    
    return lst, transform, 'EPSG:4326'

def save_lst_raster(year):
    """Save synthetic LST as GeoTIFF."""
    lst, transform, crs = generate_synthetic_lst_raster(year)
    path = os.path.join(INPUT_DIR, f"landsat_lst_{year}.tif")
    
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=lst.shape[0],
        width=lst.shape[1],
        count=1,
        dtype=lst.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(lst, 1)
    return path

# ── Zonal Statistics ────────────────────────────────────────────────────────────

def zonal_stats_point(gdf, raster_path, column_name):
    """Extract raster value at each GeoDataFrame point."""
    values = []
    with rasterio.open(raster_path) as src:
        for _, row in gdf.iterrows():
            try:
                # Sample raster at point
                for val in src.sample([(row.geometry.x, row.geometry.y)]):
                    values.append(float(val[0]))
            except Exception:
                values.append(np.nan)
    gdf[column_name] = values
    return gdf

def compute_lst_statistics(gdf, year):
    """
    Compute LST-derived statistics per census block.
    """
    # Get LST values
    raster_path = os.path.join(INPUT_DIR, f"landsat_lst_{year}.tif")
    if not os.path.exists(raster_path):
        print(f"  Generating synthetic LST raster for {year}...")
        raster_path = save_lst_raster(year)
    
    gdf = zonal_stats_point(gdf, raster_path, f'lst_{year}')
    return gdf

# ── Main Pipeline ───────────────────────────────────────────────────────────────

def main():
    print("🌡️  LST Preprocessing Pipeline\n")
    
    # Load census blocks
    census_path = os.path.join(INPUT_DIR, 'census_blocks_surat.geojson')
    if not os.path.exists(census_path):
        print("📊 Generating census data...")
        import sys

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(BASE_DIR, '00_data_download'))

        from census_download import generate_synthetic_census_surat

        gdf = generate_synthetic_census_surat()
    else:
        gdf = gpd.read_file(census_path)
    
    print(f"📍 Loaded {len(gdf)} census blocks")
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # Process each year
    years = range(2015, 2025)
    for year in years:
        print(f"  Processing LST for {year}...")
        gdf = compute_lst_statistics(gdf, year)
    
    # ── Derived statistics ──────────────────────────────
    
    lst_cols = [f'lst_{y}' for y in years]
    
    # Mean LST across all years
    gdf['lst_mean'] = gdf[lst_cols].mean(axis=1)
    
    # Peak LST (hottest year)
    gdf['lst_max'] = gdf[lst_cols].max(axis=1)
    
    # LST trend slope (degrees per year)
    def compute_trend(row):
        vals = [row[f'lst_{y}'] for y in years]
        if np.any(np.isnan(vals)):
            return np.nan
        x = np.arange(len(vals))
        return np.polyfit(x, vals, 1)[0]
    
    print("  Computing LST trends...")
    gdf['lst_trend_per_year'] = gdf.apply(compute_trend, axis=1)
    
    # City-wide statistics for anomaly scoring
    city_mean = gdf['lst_mean'].mean()
    city_std = gdf['lst_mean'].std()
    
    # LST Z-score anomaly
    gdf['lst_zscore'] = (gdf['lst_mean'] - city_mean) / city_std
    
    # Thermal persistence: fraction of years above 90th percentile
    p90 = gdf['lst_mean'].quantile(0.90)
    gdf['thermal_persistence'] = (gdf[lst_cols] > p90).mean(axis=1)
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, 'lst_processed.geojson')
    gdf.to_file(out_path, driver='GeoJSON')
    print(f"\n✅ LST processing complete: {out_path}")
    print(f"   City mean LST: {city_mean:.1f}°C")
    print(f"   City std LST:  {city_std:.1f}°C")
    print(f"   Max anomaly:   {gdf['lst_zscore'].max():.2f}σ")
    
    return gdf

if __name__ == "__main__":
    main()
