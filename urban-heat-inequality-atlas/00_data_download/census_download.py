"""
00_data_download/census_download.py
=====================================
Downloads socioeconomic data for Surat from:
  - Census of India 2011 (Primary Census Abstract)
  - Open government data portals
  - OSM for green spaces and cooling centres
"""

import requests
import pandas as pd
import geopandas as gpd
import json
import os
from shapely.geometry import Point

OUTPUT_DIR = "../data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Synthetic Census Data Generator ────────────────────────────────────────────
# For demonstration — replace with actual Census API calls
# India Census API: https://censusindia.gov.in/

def generate_synthetic_census_surat(n_blocks=150, seed=42):
    """
    Generate synthetic census block data for Surat.
    In production: replace with actual Census of India API calls.
    
    Variables modeled after real Surat ward demographics.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    
    # Surat ward centres (approximate)
    ward_centres = {
        'Rander': (72.785, 21.235),
        'Katargam': (72.830, 21.245),
        'Varachha': (72.890, 21.215),
        'Udhna': (72.885, 21.175),
        'Limbayat': (72.895, 21.155),
        'Majura': (72.840, 21.205),
        'Athwa': (72.820, 21.175),
        'Chowk': (72.830, 21.195),
        'Lal_Darwaja': (72.835, 21.190),
        'Salabatpura': (72.848, 21.185),
    }
    
    blocks = []
    block_id = 0
    
    for ward, (lon_c, lat_c) in ward_centres.items():
        n_ward = n_blocks // len(ward_centres)
        
        # Economic profile per ward (based on known Surat demographics)
        income_profiles = {
            'Athwa': (65000, 15000),       # High income — business district
            'Rander': (55000, 12000),      # Upper-middle — planned area
            'Majura': (45000, 10000),      # Middle income
            'Katargam': (30000, 8000),     # Lower-middle — textile workers
            'Varachha': (28000, 7000),     # Lower-middle — dense residential
            'Udhna': (22000, 6000),        # Low income — industrial zone
            'Limbayat': (20000, 5000),     # Low income — periurban
            'Chowk': (35000, 9000),
            'Lal_Darwaja': (25000, 7000),
            'Salabatpura': (40000, 11000),
        }
        
        inc_mean, inc_std = income_profiles.get(ward, (35000, 10000))
        
        for i in range(n_ward):
            lon = lon_c + rng.uniform(-0.02, 0.02)
            lat = lat_c + rng.uniform(-0.015, 0.015)
            
            median_income = max(8000, rng.normal(inc_mean, inc_std))
            population = int(rng.integers(800, 5000))
            elderly_share = rng.beta(2, 10) * 0.25  # 0–25%
            
            # Distance to nearest park (km) — inversely correlated with income
            income_factor = min(1, median_income / 70000)
            park_distance = max(0.1, rng.exponential(2.5 - income_factor * 1.5))
            
            blocks.append({
                'block_id': f"SRT_{block_id:04d}",
                'ward': ward,
                'lon': lon,
                'lat': lat,
                'median_income': round(median_income, 0),
                'population': population,
                'elderly_share': round(elderly_share, 4),
                'literacy_rate': round(min(0.99, rng.beta(7, 3) * (0.6 + income_factor * 0.4)), 3),
                'park_distance_km': round(park_distance, 3),
                'cooling_centre_distance_km': round(park_distance * 1.5, 3),
            })
            block_id += 1
    
    df = pd.DataFrame(blocks)
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    out_path = os.path.join(OUTPUT_DIR, 'census_blocks_surat.geojson')
    gdf.to_file(out_path, driver='GeoJSON')
    print(f"✅ Census data saved: {out_path} ({len(gdf)} blocks)")
    return gdf

# ── OSM Green Space Download ────────────────────────────────────────────────────

def download_osm_greenspaces():
    """
    Download parks and green spaces from OpenStreetMap via Overpass API.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Surat bounding box
    bbox = "21.0,72.7,21.35,73.1"
    
    query = f"""
    [out:json][timeout:60];
    (
      way["leisure"="park"]({bbox});
      way["leisure"="garden"]({bbox});
      way["landuse"="forest"]({bbox});
      way["natural"="wood"]({bbox});
      node["leisure"="park"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """
    
    print("📥 Downloading OSM green spaces...")
    try:
        response = requests.post(overpass_url, data=query, timeout=60)
        data = response.json()
        
        # Parse to simple point list
        parks = []
        for element in data.get('elements', []):
            if element['type'] == 'node':
                parks.append({
                    'id': element['id'],
                    'name': element.get('tags', {}).get('name', 'Park'),
                    'lon': element['lon'],
                    'lat': element['lat'],
                    'type': element.get('tags', {}).get('leisure', 'park')
                })
        
        df_parks = pd.DataFrame(parks)
        if len(df_parks) > 0:
            out_path = os.path.join(OUTPUT_DIR, 'osm_green_spaces.csv')
            df_parks.to_csv(out_path, index=False)
            print(f"✅ Green spaces saved: {out_path} ({len(df_parks)} features)")
        else:
            print("⚠️  No OSM features returned — using synthetic data")
            _generate_synthetic_parks()
    
    except Exception as e:
        print(f"⚠️  OSM download failed ({e}) — using synthetic fallback")
        _generate_synthetic_parks()

def _generate_synthetic_parks():
    """Fallback synthetic parks for Surat."""
    import numpy as np
    rng = np.random.default_rng(99)
    parks = [
        {'name': 'Sarthana Nature Park', 'lon': 72.908, 'lat': 21.234, 'type': 'nature_park'},
        {'name': 'Dumas Beach Park', 'lon': 72.714, 'lat': 21.074, 'type': 'beach_park'},
        {'name': 'VR Surat Gardens', 'lon': 72.837, 'lat': 21.198, 'type': 'garden'},
        {'name': 'Varachha Park', 'lon': 72.885, 'lat': 21.210, 'type': 'park'},
        {'name': 'Udhna Park', 'lon': 72.880, 'lat': 21.168, 'type': 'park'},
    ]
    for i in range(20):
        parks.append({
            'name': f'Community Park {i+1}',
            'lon': 72.75 + rng.uniform(0, 0.35),
            'lat': 21.05 + rng.uniform(0, 0.28),
            'type': 'small_park'
        })
    df = pd.DataFrame(parks)
    out_path = os.path.join(OUTPUT_DIR, 'osm_green_spaces.csv')
    df.to_csv(out_path, index=False)
    print(f"✅ Synthetic parks saved: {out_path}")

# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📊 Downloading socioeconomic & spatial data for Surat...\n")
    
    # 1. Census blocks
    gdf_census = generate_synthetic_census_surat(n_blocks=150)
    
    # 2. OSM green spaces
    download_osm_greenspaces()
    
    print("\n✅ Data download complete.")
    print(f"📁 Files saved in: {OUTPUT_DIR}/")
