"""
02_feature_engineering/tii_calculator.py
==========================================
Builds the Thermal Injustice Index (TII) — the signature feature of this project.

TII Formula:
    TII = w1*(LST_anomaly_z) + w2*(NDVI_deficit_norm) + w3*(income_vulnerability)
          + w4*(cooling_gap_norm) + w5*(elderly_share_norm)

Weights are derived from XGBoost feature importance (see 04_ml_models/).
Default equal weighting used initially, then refined post-modelling.

Output: 0–100 ranked score per census block, saved to processed/features_final.geojson
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
import os
import json

INPUT_DIR = "../data/processed"
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TII Weights ─────────────────────────────────────────────────────────────────
# Initial equal weights — updated after XGBoost feature importance
TII_WEIGHTS = {
    'lst_zscore_norm':       0.30,  # LST anomaly (dominant driver)
    'ndvi_deficit_norm':     0.25,  # Vegetation deficit
    'income_vuln_norm':      0.20,  # Income vulnerability (inverse)
    'cooling_gap_norm':      0.15,  # Distance to cooling refuge
    'elderly_share_norm':    0.10,  # Elderly population share
}

# ── Feature Normalization ───────────────────────────────────────────────────────

def normalize_column(series, invert=False):
    """Min-max normalize a series to [0, 1]. Optionally invert."""
    scaler = MinMaxScaler()
    vals = series.values.reshape(-1, 1)
    normalized = scaler.fit_transform(vals).flatten()
    if invert:
        normalized = 1.0 - normalized
    return normalized

def build_features(gdf):
    """
    Engineer all features needed for TII computation.
    Expects GDF with LST and NDVI columns already present.
    """
    gdf = gdf.copy()
    
    # ── 1. LST anomaly score ──────────────────────────────────────────────────
    # City-relative Z-score of mean LST
    city_mean = gdf['lst_mean'].mean()
    city_std  = gdf['lst_mean'].std()
    gdf['lst_zscore'] = (gdf['lst_mean'] - city_mean) / city_std
    gdf['lst_zscore_norm'] = normalize_column(gdf['lst_zscore'])
    
    # ── 2. NDVI deficit ───────────────────────────────────────────────────────
    if 'ndvi_deficit_norm' not in gdf.columns:
        target_ndvi = 0.35
        gdf['ndvi_deficit'] = (target_ndvi - gdf.get('ndvi_mean', 0.2)).clip(lower=0)
        gdf['ndvi_deficit_norm'] = normalize_column(gdf['ndvi_deficit'])
    
    # ── 3. Income vulnerability (inverse of income percentile) ────────────────
    gdf['income_percentile'] = gdf['median_income'].rank(pct=True)
    gdf['income_vuln'] = 1.0 - gdf['income_percentile']
    gdf['income_vuln_norm'] = normalize_column(gdf['income_vuln'])
    
    # ── 4. Cooling access gap ─────────────────────────────────────────────────
    if 'cooling_gap_km' in gdf.columns:
        gdf['cooling_gap_norm'] = normalize_column(gdf['cooling_gap_km'])
    else:
        gdf['cooling_gap_norm'] = 0.5
    
    # ── 5. Elderly share ──────────────────────────────────────────────────────
    if 'elderly_share' in gdf.columns:
        gdf['elderly_share_norm'] = normalize_column(gdf['elderly_share'])
    else:
        gdf['elderly_share_norm'] = 0.0
    
    # ── 6. Additional derived features ───────────────────────────────────────
    
    # Thermal persistence (fraction of years in top-10% LST)
    lst_cols = [c for c in gdf.columns if c.startswith('lst_20')]
    if lst_cols:
        p90 = np.percentile(gdf[lst_cols].values, 90)
        gdf['thermal_persistence'] = (gdf[lst_cols] > p90).mean(axis=1)
    
    # Heat-income interaction term (key for injustice narrative)
    gdf['heat_poverty_interaction'] = gdf['lst_zscore_norm'] * gdf['income_vuln_norm']
    
    return gdf

# ── TII Calculation ─────────────────────────────────────────────────────────────

def compute_tii(gdf, weights=None):
    """
    Compute the Thermal Injustice Index (0–100) for each census block.
    
    Parameters:
        gdf: GeoDataFrame with engineered features
        weights: dict of {feature_col: weight} — must sum to 1.0
    
    Returns: GDF with 'TII' column added and ranked.
    """
    if weights is None:
        weights = TII_WEIGHTS
    
    # Validate weights
    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0 (got {total:.3f})")
    
    # Check all features present
    missing = [f for f in weights if f not in gdf.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    # Weighted sum
    tii_raw = sum(
        weight * gdf[feature]
        for feature, weight in weights.items()
    )
    
    # Scale to 0–100
    tii_scaled = (tii_raw - tii_raw.min()) / (tii_raw.max() - tii_raw.min()) * 100
    
    gdf = gdf.copy()
    gdf['TII'] = tii_scaled.round(2)
    gdf['TII_rank'] = gdf['TII'].rank(ascending=False).astype(int)
    
    # Severity tier
    gdf['heat_tier'] = pd.cut(
        gdf['TII'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Minimal', 'Low', 'Moderate', 'High', 'Critical'],
        include_lowest=True
    )
    
    return gdf

# ── Anomaly Detection Feature ───────────────────────────────────────────────────

def compute_change_anomaly(gdf):
    """
    Detect blocks where LST rose faster than NDVI loss explains.
    Uses a residual approach: actual LST change vs. predicted from NDVI change.
    """
    lst_cols = [c for c in gdf.columns if c.startswith('lst_20')]
    ndvi_cols = [c for c in gdf.columns if c.startswith('ndvi_20')]
    
    if len(lst_cols) < 2 or len(ndvi_cols) < 2:
        gdf['anomaly_score'] = 0.0
        return gdf
    
    # LST change: last year - first year
    gdf['lst_change'] = gdf[lst_cols[-1]] - gdf[lst_cols[0]]
    
    # NDVI change
    if ndvi_cols:
        gdf['ndvi_change'] = gdf[ndvi_cols[-1]] - gdf[ndvi_cols[0]]
    else:
        gdf['ndvi_change'] = 0.0
    
    # Expected LST change from NDVI (empirical: 1 unit NDVI ≈ -5°C LST)
    gdf['lst_change_expected'] = gdf['ndvi_change'] * -5.0
    gdf['lst_change_residual'] = gdf['lst_change'] - gdf['lst_change_expected']
    
    # Anomaly score: standardized positive residuals
    res_mean = gdf['lst_change_residual'].mean()
    res_std  = gdf['lst_change_residual'].std()
    gdf['anomaly_score'] = ((gdf['lst_change_residual'] - res_mean) / res_std).clip(lower=0)
    gdf['is_anomaly'] = gdf['anomaly_score'] > 2.0  # 2-sigma threshold
    
    return gdf

# ── Summary Report ──────────────────────────────────────────────────────────────

def print_tii_summary(gdf):
    """Print a summary of the TII results."""
    print("\n" + "="*55)
    print("       THERMAL INJUSTICE INDEX — SUMMARY REPORT")
    print("="*55)
    print(f"  City: Surat, Gujarat, India")
    print(f"  Census blocks analysed: {len(gdf)}")
    print(f"\n  TII Distribution:")
    print(f"  ├── Mean:   {gdf['TII'].mean():.1f}")
    print(f"  ├── Median: {gdf['TII'].median():.1f}")
    print(f"  ├── Min:    {gdf['TII'].min():.1f}")
    print(f"  └── Max:    {gdf['TII'].max():.1f}")
    
    print(f"\n  Tier Distribution:")
    tier_counts = gdf['heat_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(gdf) * 100
        bar = '█' * int(pct / 4)
        print(f"  {tier:<10} {bar:<20} {count:3d} blocks ({pct:.0f}%)")
    
    print(f"\n  🔴 TOP 5 CRITICAL ZONES (highest TII):")
    top5 = gdf.nlargest(5, 'TII')[['block_id', 'ward', 'TII', 'lst_mean', 
                                     'ndvi_mean', 'median_income', 'heat_tier']]
    for _, row in top5.iterrows():
        print(f"     {row['block_id']} | {row['ward']:<12} | "
              f"TII={row['TII']:.1f} | LST={row.get('lst_mean', 0):.1f}°C | "
              f"Income=₹{row['median_income']:.0f}")
    
    print(f"\n  🟢 TOP 5 SAFEST ZONES (lowest TII):")
    bot5 = gdf.nsmallest(5, 'TII')[['block_id', 'ward', 'TII', 'lst_mean', 
                                     'ndvi_mean', 'median_income']]
    for _, row in bot5.iterrows():
        print(f"     {row['block_id']} | {row['ward']:<12} | "
              f"TII={row['TII']:.1f} | LST={row.get('lst_mean', 0):.1f}°C | "
              f"Income=₹{row['median_income']:.0f}")
    
    n_anomalies = gdf.get('is_anomaly', pd.Series([False]*len(gdf))).sum()
    print(f"\n  ⚠️  Unexplained heat anomalies detected: {n_anomalies}")
    print("="*55 + "\n")

# ── Main ────────────────────────────────────────────────────────────────────────

def main(weights=None):
    print("🧮 Thermal Injustice Index (TII) Builder\n")
    
    # Load processed NDVI+LST data
    ndvi_path = os.path.join(INPUT_DIR, 'ndvi_processed.geojson')
    lst_path  = os.path.join(INPUT_DIR, 'lst_processed.geojson')
    
    if os.path.exists(ndvi_path):
        gdf = gpd.read_file(ndvi_path)
        print(f"📍 Loaded {len(gdf)} blocks from NDVI processed data")
    elif os.path.exists(lst_path):
        gdf = gpd.read_file(lst_path)
        print(f"📍 Loaded {len(gdf)} blocks from LST processed data")
    else:
        print("🔄 Running full preprocessing pipeline...")
        import subprocess
        subprocess.run(['python', '../01_preprocessing/lst_retrieval.py'])
        subprocess.run(['python', '../01_preprocessing/ndvi_processing.py'])
        gdf = gpd.read_file(ndvi_path)
    
    # Build features
    print("  Engineering features...")
    gdf = build_features(gdf)
    
    # Compute TII
    print("  Computing Thermal Injustice Index...")
    gdf = compute_tii(gdf, weights=weights)
    
    # Anomaly detection
    print("  Computing change anomaly scores...")
    gdf = compute_change_anomaly(gdf)
    
    # Save weights used
    weights_used = weights or TII_WEIGHTS
    weights_path = os.path.join(OUTPUT_DIR, 'tii_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(weights_used, f, indent=2)
    
    # Save features
    out_path = os.path.join(OUTPUT_DIR, 'features_final.geojson')
    gdf.to_file(out_path, driver='GeoJSON')
    
    # Print summary
    print_tii_summary(gdf)
    
    print(f"✅ Features saved: {out_path}")
    return gdf

if __name__ == "__main__":
    main()
