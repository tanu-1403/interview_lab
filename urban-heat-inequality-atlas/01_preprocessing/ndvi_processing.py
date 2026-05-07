"""
01_preprocessing/ndvi_processing.py
======================================
Processes Sentinel-2 derived NDVI:
  - NDVI extraction per census block
  - NDVI deficit calculation vs. city baseline
  - NDBI (built-up index) extraction
  - Green space distance calculation
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.transform import from_bounds
import rasterio
from scipy.spatial import cKDTree
import os

INPUT_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Synthetic NDVI Generator ────────────────────────────────────────────────────

def generate_synthetic_ndvi(year, seed=None):
    """
    Generate synthetic NDVI raster. Spatial pattern is inverse of LST:
    - Parks and river: high NDVI (~0.5–0.7)
    - Planned residential: medium NDVI (~0.2–0.4)
    - Dense built-up: low NDVI (~0.05–0.2)
    - Industrial: very low NDVI (~0.0–0.1)
    
    With slow urban expansion reducing NDVI over years.
    """
    if seed is None:
        seed = year + 100
    rng = np.random.default_rng(seed)
    
    width, height = 400, 300
    lon_min, lat_min, lon_max, lat_max = 72.7, 21.0, 73.1, 21.35
    
    lons = np.linspace(lon_min, lon_max, width)
    lats = np.linspace(lat_max, lat_min, height)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Base NDVI
    ndvi = np.full((height, width), 0.15, dtype=np.float32)
    
    # Annual urban expansion: -0.003 NDVI/year
    year_offset = -(year - 2015) * 0.003
    ndvi += year_offset
    
    def gaussian_blob(lon_c, lat_c, amp, sw=0.025, sh=0.018):
        return amp * np.exp(
            -(((lon_grid - lon_c) / sw)**2 + 
              ((lat_grid - lat_c) / sh)**2)
        )
    
    # High NDVI zones
    ndvi += gaussian_blob(72.908, 21.234, +0.5, 0.02, 0.015)  # Sarthana Nature Park
    ndvi += gaussian_blob(72.714, 21.074, +0.4, 0.03, 0.025)  # Dumas coast
    
    # River Tapti corridor
    river = np.abs((lat_grid - 21.20) - 0.4 * (lon_grid - 72.85)) < 0.018
    ndvi[river] += 0.3
    
    # Medium NDVI — planned residential
    ndvi += gaussian_blob(72.785, 21.235, +0.15, 0.025, 0.02)
    
    # Low NDVI — dense built-up
    ndvi += gaussian_blob(72.885, 21.165, -0.12, 0.03, 0.02)  # Industrial
    ndvi += gaussian_blob(72.890, 21.215, -0.10, 0.025, 0.02) # Varachha
    ndvi += gaussian_blob(72.835, 21.190, -0.08, 0.02, 0.015) # Old city
    
    # Add noise
    noise = rng.normal(0, 0.02, (height, width)).astype(np.float32)
    ndvi += noise
    ndvi = np.clip(ndvi, -0.1, 0.8)
    
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    return ndvi, transform, 'EPSG:4326'

def save_ndvi_raster(year):
    """Save NDVI raster as GeoTIFF."""
    ndvi, transform, crs = generate_synthetic_ndvi(year)
    path = os.path.join(INPUT_DIR, f"sentinel2_ndvi_{year}.tif")
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=ndvi.shape[0], width=ndvi.shape[1],
        count=1, dtype=ndvi.dtype, crs=crs, transform=transform
    ) as dst:
        dst.write(ndvi, 1)
    return path

def sample_raster_at_points(gdf, raster_path, col_name):
    """Sample raster values at GDF point locations."""
    values = []
    with rasterio.open(raster_path) as src:
        for _, row in gdf.iterrows():
            try:
                for val in src.sample([(row.geometry.x, row.geometry.y)]):
                    values.append(float(val[0]))
            except Exception:
                values.append(np.nan)
    gdf = gdf.copy()
    gdf[col_name] = values
    return gdf

# ── Green Space Distance ────────────────────────────────────────────────────────

def compute_cooling_access_gap(gdf):
    """
    Compute distance from each census block to nearest cool refuge:
    parks, water bodies, community cooling centres.
    
    Returns updated GDF with 'cooling_gap_km' column.
    """
    parks_path = os.path.join(INPUT_DIR, 'osm_green_spaces.csv')
    
    if os.path.exists(parks_path):
        parks = pd.read_csv(parks_path)
    else:
        # Synthetic fallback parks
        parks = pd.DataFrame({
            'lon': [72.908, 72.714, 72.837, 72.885, 72.880, 72.760, 72.870, 72.810],
            'lat': [21.234, 21.074, 21.198, 21.210, 21.168, 21.200, 21.240, 21.220],
            'name': ['Sarthana NP', 'Dumas Beach', 'VR Gardens', 'Varachha Park',
                     'Udhna Park', 'Rander Park', 'North Park', 'Majura Garden']
        })
    
    # Build KD-tree of park coordinates
    park_coords = parks[['lon', 'lat']].values
    block_coords = np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    
    tree = cKDTree(park_coords)
    dist_deg, _ = tree.query(block_coords)
    
    # Convert degrees to km (approximate: 1 deg lat ≈ 111 km)
    dist_km = dist_deg * 111.0
    
    gdf = gdf.copy()
    gdf['cooling_gap_km'] = np.round(dist_km, 3)
    
    return gdf

# ── NDVI Deficit ────────────────────────────────────────────────────────────────

def compute_ndvi_deficit(gdf, target_ndvi=0.35):
    """
    Compute NDVI deficit: how far below the 'healthy city' NDVI threshold
    each block sits. Target NDVI of 0.35 represents adequate cooling canopy.
    """
    gdf = gdf.copy()
    gdf['ndvi_deficit'] = (target_ndvi - gdf['ndvi_mean']).clip(lower=0)
    # Normalize 0–1
    max_deficit = gdf['ndvi_deficit'].max()
    if max_deficit > 0:
        gdf['ndvi_deficit_norm'] = gdf['ndvi_deficit'] / max_deficit
    else:
        gdf['ndvi_deficit_norm'] = 0.0
    return gdf

# ── Main Pipeline ───────────────────────────────────────────────────────────────

def main():
    print("🌿 NDVI Preprocessing Pipeline\n")
    
    # Load LST-processed data
    lst_path = os.path.join(OUTPUT_DIR, 'lst_processed.geojson')
    if not os.path.exists(lst_path):
        print("⚠️  Running LST preprocessing first...")
        import sys; sys.path.append('.')
        from lst_retrieval import main as lst_main
        gdf = lst_main()
    else:
        gdf = gpd.read_file(lst_path)
    
    print(f"📍 Loaded {len(gdf)} census blocks")
    
    # Process NDVI for key years
    ndvi_years = [2016, 2018, 2020, 2022, 2024]
    
    for year in ndvi_years:
        print(f"  Processing NDVI for {year}...")
        path = os.path.join(INPUT_DIR, f"sentinel2_ndvi_{year}.tif")
        if not os.path.exists(path):
            path = save_ndvi_raster(year)
        gdf = sample_raster_at_points(gdf, path, f'ndvi_{year}')
    
    # Derived NDVI statistics
    ndvi_cols = [f'ndvi_{y}' for y in ndvi_years]
    gdf['ndvi_mean'] = gdf[ndvi_cols].mean(axis=1)
    gdf['ndvi_latest'] = gdf[f'ndvi_{ndvi_years[-1]}']
    
    # NDVI trend
    def ndvi_trend(row):
        vals = [row[f'ndvi_{y}'] for y in ndvi_years]
        if np.any(np.isnan(vals)):
            return np.nan
        x = np.arange(len(vals))
        return np.polyfit(x, vals, 1)[0]
    
    gdf['ndvi_trend_per_step'] = gdf.apply(ndvi_trend, axis=1)
    
    # NDVI deficit
    gdf = compute_ndvi_deficit(gdf)
    
    # Cooling access gap
    print("  Computing cooling access gap...")
    gdf = compute_cooling_access_gap(gdf)
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, 'ndvi_processed.geojson')
    gdf.to_file(out_path, driver='GeoJSON')
    
    print(f"\n✅ NDVI processing complete: {out_path}")
    print(f"   Mean city NDVI:   {gdf['ndvi_mean'].mean():.3f}")
    print(f"   Mean NDVI trend:  {gdf['ndvi_trend_per_step'].mean():.4f}/step")
    print(f"   Mean cooling gap: {gdf['cooling_gap_km'].mean():.2f} km")
    
    return gdf

if __name__ == "__main__":
    main()
