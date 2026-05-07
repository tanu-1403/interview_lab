"""
03_clustering/thermal_zones.py
================================
Thermal zone segmentation using K-Means + DBSCAN clustering.

Identifies distinct "thermal personality" zones across the city:
  🔴 Zone A: Hot-dry pavement deserts (industrial/dense built-up)
  🟠 Zone B: Hot-humid residential (low-income dense housing)
  🟡 Zone C: Moderate mixed commercial
  🟢 Zone D: Cool planned residential (green corridors)
  🔵 Zone E: Cool riparian / park zones

Outputs:
  - clusters_map.html (folium choropleth)
  - thermal_zone_profiles.csv
  - clusters_geojson.geojson
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
import json
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR  = "../data/processed"
OUTPUT_DIR = "../data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Zone labels and colours
ZONE_CONFIG = {
    0: {'label': 'Hot pavement desert',    'color': '#D62728', 'emoji': '🔴'},
    1: {'label': 'Hot low-income housing', 'color': '#FF7F0E', 'emoji': '🟠'},
    2: {'label': 'Moderate mixed-use',     'color': '#FFBB78', 'emoji': '🟡'},
    3: {'label': 'Cool planned residential','color': '#98DF8A', 'emoji': '🟢'},
    4: {'label': 'Cool riparian/park zone', 'color': '#1F77B4', 'emoji': '🔵'},
}

# ── Feature Selection for Clustering ───────────────────────────────────────────

CLUSTER_FEATURES = [
    'lst_zscore_norm',
    'ndvi_deficit_norm',
    'income_vuln_norm',
    'cooling_gap_norm',
    'thermal_persistence',
]

def prepare_cluster_features(gdf):
    """Select and scale features for clustering."""
    available = [f for f in CLUSTER_FEATURES if f in gdf.columns]
    X = gdf[available].fillna(gdf[available].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, available, scaler

# ── Optimal K Selection ─────────────────────────────────────────────────────────

def find_optimal_k(X, k_range=range(2, 9)):
    """
    Use elbow method + silhouette score to find optimal K.
    Returns plot + recommended K.
    """
    inertias = []
    silhouettes = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia (WCSS)', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(list(k_range), silhouettes, 'rs-', linewidth=2, markersize=8)
    best_k_idx = np.argmax(silhouettes)
    best_k = list(k_range)[best_k_idx]
    axes[1].axvline(x=best_k, color='green', linestyle='--', label=f'Best K={best_k}')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'optimal_k_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 K-selection plot saved: {path}")
    print(f"  ✅ Recommended K = {best_k} (silhouette = {silhouettes[best_k_idx]:.3f})")
    
    return best_k

# ── K-Means Clustering ──────────────────────────────────────────────────────────

def run_kmeans(gdf, X_scaled, n_clusters=5):
    """Run K-Means and assign zone labels."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    raw_labels = km.fit_predict(X_scaled)
    
    # Reorder labels by mean LST (0 = hottest, n-1 = coolest)
    cluster_lst_means = {}
    for k in range(n_clusters):
        mask = raw_labels == k
        cluster_lst_means[k] = gdf.loc[mask, 'lst_zscore_norm'].mean() \
                                if 'lst_zscore_norm' in gdf.columns else k
    
    sorted_clusters = sorted(cluster_lst_means, key=cluster_lst_means.get, reverse=True)
    label_map = {old: new for new, old in enumerate(sorted_clusters)}
    labels = np.array([label_map[l] for l in raw_labels])
    
    gdf = gdf.copy()
    gdf['zone_id'] = labels
    gdf['zone_label'] = gdf['zone_id'].map(
        {k: v['label'] for k, v in ZONE_CONFIG.items()}
    ).fillna('Unknown')
    gdf['zone_color'] = gdf['zone_id'].map(
        {k: v['color'] for k, v in ZONE_CONFIG.items()}
    ).fillna('#888888')
    
    return gdf, km

# ── Zone Profile Analysis ───────────────────────────────────────────────────────

def build_zone_profiles(gdf):
    """Compute mean statistics per thermal zone."""
    cols = ['lst_mean', 'ndvi_mean', 'median_income', 'elderly_share',
            'cooling_gap_km', 'TII', 'zone_label']
    cols_avail = [c for c in cols if c in gdf.columns] + ['zone_id']
    
    profile = gdf[cols_avail].groupby('zone_id').agg({
        'lst_mean': 'mean',
        'ndvi_mean': 'mean',
        'median_income': 'mean',
        'elderly_share': 'mean',
        'cooling_gap_km': 'mean',
        'TII': 'mean',
        'zone_label': 'first',
    }).round(2).reset_index()
    
    profile.columns = profile.columns.str.replace('_', ' ').str.title()
    
    path = os.path.join(OUTPUT_DIR, 'thermal_zone_profiles.csv')
    profile.to_csv(path, index=False)
    print(f"\n  📊 Zone profiles saved: {path}")
    
    # Print table
    print("\n  THERMAL ZONE PROFILES:")
    print(f"  {'Zone':<30} {'LST(°C)':<10} {'NDVI':<8} {'Income(₹)':<12} {'TII':<8}")
    print("  " + "-"*65)
    for _, row in profile.iterrows():
        zone_id = int(row['Zone Id'])
        emoji = ZONE_CONFIG.get(zone_id, {}).get('emoji', '⬜')
        print(f"  {emoji} {str(row.get('Zone Label', '')):<28} "
              f"{row.get('Lst Mean', 0):>7.1f}  "
              f"{row.get('Ndvi Mean', 0):>6.3f}  "
              f"{row.get('Median Income', 0):>10.0f}  "
              f"{row.get('Tii', 0):>6.1f}")
    
    return profile

# ── Folium Cluster Map ──────────────────────────────────────────────────────────

def create_cluster_map(gdf):
    """Create an interactive folium map of thermal zones."""
    m = folium.Map(
        location=[21.18, 72.83],
        zoom_start=12,
        tiles='CartoDB positron'
    )
    
    for _, row in gdf.iterrows():
        zone_id = int(row.get('zone_id', 0))
        zone_info = ZONE_CONFIG.get(zone_id, {'label': 'Unknown', 'color': '#888'})
        color = zone_info['color']
        
        tii_val = row.get('TII', 0)
        lst_val = row.get('lst_mean', 0)
        inc_val = row.get('median_income', 0)
        
        popup_html = f"""
        <div style="font-family: sans-serif; width: 200px;">
          <b>{zone_info['label']}</b><br>
          <hr style="margin: 4px 0"/>
          Block: {row.get('block_id', 'N/A')}<br>
          Ward: {row.get('ward', 'N/A')}<br>
          TII Score: <b style="color:{color}">{tii_val:.1f}</b><br>
          LST: {lst_val:.1f}°C<br>
          Income: ₹{inc_val:,.0f}<br>
          Tier: {row.get('heat_tier', 'N/A')}
        </div>
        """
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{zone_info['label']} | TII={tii_val:.0f}"
        ).add_to(m)
    
    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 999;
                background: white; border: 1px solid #ccc; border-radius: 8px;
                padding: 12px; font-family: sans-serif; font-size: 12px;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.15)">
      <b>Thermal Zones</b><br><br>
    """
    for zone_id, cfg in ZONE_CONFIG.items():
        legend_html += f"""<span style="color:{cfg['color']}">●</span> 
                          {cfg['label']}<br>"""
    legend_html += "</div>"
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    path = os.path.join(OUTPUT_DIR, 'thermal_zones_map.html')
    m.save(path)
    print(f"  🗺️  Cluster map saved: {path}")
    return m

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("🔬 Thermal Zone Clustering\n")
    
    # Load feature data
    feat_path = os.path.join(INPUT_DIR, 'features_final.geojson')
    if not os.path.exists(feat_path):
        print("🔄 Running feature engineering first...")
        import sys; sys.path.append('../02_feature_engineering')
        from tii_calculator import main as tii_main
        gdf = tii_main()
    else:
        gdf = gpd.read_file(feat_path)
    
    print(f"📍 Loaded {len(gdf)} blocks")
    
    # Prepare features
    X_scaled, features_used, scaler = prepare_cluster_features(gdf)
    print(f"  Features used: {features_used}")
    
    # Find optimal K
    print("\n  Finding optimal number of clusters...")
    best_k = find_optimal_k(X_scaled)
    best_k = min(best_k, 5)  # Cap at 5 for interpretability
    
    # Run clustering
    print(f"\n  Running K-Means with K={best_k}...")
    gdf, km_model = run_kmeans(gdf, X_scaled, n_clusters=best_k)
    
    # Zone profiles
    build_zone_profiles(gdf)
    
    # Save clustered data
    out_path = os.path.join(OUTPUT_DIR, 'clusters_geojson.geojson')
    gdf.to_file(out_path, driver='GeoJSON')
    
    # Create folium map
    print("\n  Creating cluster map...")
    create_cluster_map(gdf)
    
    # Update features file with cluster labels
    gdf.to_file(feat_path, driver='GeoJSON')
    
    print(f"\n✅ Clustering complete.")
    print(f"   Clustered GeoJSON: {out_path}")
    
    return gdf

if __name__ == "__main__":
    main()
