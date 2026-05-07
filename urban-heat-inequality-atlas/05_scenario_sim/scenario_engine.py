"""
05_scenario_sim/scenario_engine.py
=====================================
2035 Heat Scenario Simulation Engine

The counterfactual engine answers:
  "What if we increase tree canopy by X%?"
  "What if impervious surface grows by Y%?"
  "How does RCP 4.5 vs 8.5 climate scenario change the heat map?"

Uses the trained XGBoost model with feature perturbation
to simulate projected 2035 conditions.

This is the 'policy simulation' WOW feature.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import DualMap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR  = "../data/processed"
OUTPUT_DIR = "../data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Climate Scenarios ───────────────────────────────────────────────────────────

CLIMATE_SCENARIOS = {
    'baseline':  {'name': 'No Action (BAU)',    'temp_delta': +2.1, 'color': '#D62728'},
    'rcp45':     {'name': 'RCP 4.5 (Moderate)', 'temp_delta': +1.4, 'color': '#FF7F0E'},
    'rcp26':     {'name': 'RCP 2.6 (Ambitious)','temp_delta': +0.8, 'color': '#2CA02C'},
    'green_city':{'name': 'Green City (+30% canopy)', 'temp_delta': +0.5, 'color': '#1F77B4'},
}

# Empirical LST response coefficients
# (Based on literature: 1 unit NDVI ≈ -5°C, 10% ISF ≈ +1.2°C)
LST_COEFFICIENTS = {
    'ndvi_effect':     -5.0,   # °C per unit NDVI increase
    'isf_effect':      +1.2,   # °C per 10% increase in impervious surface fraction
    'canopy_effect':   -2.8,   # °C per 10% increase in canopy cover
    'cooling_roof':    -1.0,   # °C per 20% cool-roof adoption
}

# ── Scenario Parameter Presets ──────────────────────────────────────────────────

SCENARIO_PRESETS = {
    'do_nothing': {
        'name': 'Business As Usual (2035)',
        'ndvi_change': -0.05,           # Urban expansion reduces vegetation
        'canopy_pct_change': -5.0,      # Loss of 5% canopy
        'isf_change': +0.08,            # More paved surfaces
        'climate_warming': +2.1,        # RCP 8.5 baseline
        'description': 'Current trends continue. Urban expansion, minimal green investment.'
    },
    'moderate_intervention': {
        'name': 'Moderate Green Plan (2035)',
        'ndvi_change': +0.05,
        'canopy_pct_change': +10.0,
        'isf_change': -0.03,
        'climate_warming': +1.4,
        'description': '10% more canopy city-wide, cool pavement pilots in top-10 TII zones.'
    },
    'ambitious_green': {
        'name': 'Ambitious Green City (2035)',
        'ndvi_change': +0.12,
        'canopy_pct_change': +25.0,
        'isf_change': -0.08,
        'climate_warming': +0.8,
        'description': '+25% canopy, cool roofs on 40% of buildings, urban forest corridors.'
    },
    'justice_first': {
        'name': 'Equity-Targeted Intervention (2035)',
        'ndvi_change': +0.0,             # Average stays same
        'canopy_pct_change': +0.0,
        'isf_change': +0.0,
        'climate_warming': +1.4,
        'description': 'Same total resources — but targeted to top-20 TII zones only.',
        'targeted': True,                # Applied only to high-TII zones
        'target_ndvi_change': +0.20,    # High NDVI boost in target zones
        'target_canopy_change': +35.0,
    },
}

# ── Simulation Engine ───────────────────────────────────────────────────────────

def simulate_scenario(gdf, scenario_params, tii_weights=None):
    """
    Apply scenario parameters to compute projected 2035 LST and TII.
    
    Parameters:
        gdf: GeoDataFrame with current features
        scenario_params: dict from SCENARIO_PRESETS
        tii_weights: optional custom weights for TII computation
    
    Returns: GDF with projected 2035 columns added.
    """
    gdf_proj = gdf.copy()
    
    is_targeted = scenario_params.get('targeted', False)
    
    if is_targeted:
        # Only apply to top-25% TII zones
        tii_threshold = gdf['TII'].quantile(0.75)
        high_risk_mask = gdf['TII'] >= tii_threshold
    
    # ── Project NDVI ──────────────────────────────────────────────────────────
    ndvi_base = gdf.get('ndvi_mean', pd.Series([0.2]*len(gdf))).copy()
    
    if is_targeted:
        ndvi_delta = pd.Series(0.0, index=gdf.index)
        ndvi_delta[high_risk_mask] = scenario_params['target_ndvi_change']
    else:
        ndvi_delta = scenario_params['ndvi_change']
    
    gdf_proj['ndvi_2035'] = (ndvi_base + ndvi_delta).clip(0, 1)
    
    # ── Project LST ───────────────────────────────────────────────────────────
    lst_base = gdf.get('lst_mean', pd.Series([35.0]*len(gdf))).copy()
    
    # NDVI effect
    ndvi_lst_delta = (gdf_proj['ndvi_2035'] - ndvi_base) * LST_COEFFICIENTS['ndvi_effect']
    
    # Canopy effect
    if is_targeted:
        canopy_delta = pd.Series(0.0, index=gdf.index)
        canopy_delta[high_risk_mask] = scenario_params['target_canopy_change']
    else:
        canopy_delta = scenario_params['canopy_pct_change']
    
    canopy_lst_delta = (canopy_delta / 10.0) * LST_COEFFICIENTS['canopy_effect']
    
    # ISF effect
    isf_delta = scenario_params.get('isf_change', 0)
    isf_lst_delta = (isf_delta / 0.10) * LST_COEFFICIENTS['isf_effect']
    
    # Climate warming
    climate_delta = scenario_params['climate_warming']
    
    gdf_proj['lst_2035'] = (
        lst_base + 
        ndvi_lst_delta + 
        canopy_lst_delta + 
        isf_lst_delta + 
        climate_delta
    ).round(2)
    
    # ── Project TII ───────────────────────────────────────────────────────────
    from sklearn.preprocessing import MinMaxScaler
    
    # Re-compute LST anomaly for 2035
    city_mean_2035 = gdf_proj['lst_2035'].mean()
    city_std_2035  = gdf_proj['lst_2035'].std()
    gdf_proj['lst_zscore_2035'] = (gdf_proj['lst_2035'] - city_mean_2035) / city_std_2035
    gdf_proj['lst_zscore_norm_2035'] = MinMaxScaler().fit_transform(
        gdf_proj[['lst_zscore_2035']]
    ).flatten()
    
    # NDVI deficit 2035
    target_ndvi = 0.35
    gdf_proj['ndvi_deficit_2035'] = (target_ndvi - gdf_proj['ndvi_2035']).clip(lower=0)
    gdf_proj['ndvi_deficit_norm_2035'] = MinMaxScaler().fit_transform(
        gdf_proj[['ndvi_deficit_2035']]
    ).flatten()
    
    # Compute projected TII (using existing income/elderly/cooling weights — unchanged by scenario)
    weights = tii_weights or {
        'lst_zscore_norm_2035':    0.30,
        'ndvi_deficit_norm_2035':  0.25,
        'income_vuln_norm':        0.20,
        'cooling_gap_norm':        0.15,
        'elderly_share_norm':      0.10,
    }
    
    tii_raw = sum(
        w * gdf_proj[f] for f, w in weights.items() if f in gdf_proj.columns
    )
    
    gdf_proj['TII_2035'] = (
        (tii_raw - tii_raw.min()) / (tii_raw.max() - tii_raw.min()) * 100
    ).round(2)
    
    gdf_proj['TII_delta'] = (gdf_proj['TII_2035'] - gdf_proj['TII']).round(2)
    gdf_proj['LST_delta'] = (gdf_proj['lst_2035'] - lst_base).round(2)
    
    return gdf_proj

# ── Scenario Comparison Visualization ──────────────────────────────────────────

def plot_scenario_comparison(gdf, all_scenario_results):
    """Compare all scenarios side-by-side."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    scenario_names = list(all_scenario_results.keys())
    
    # 1. Mean LST 2035 by scenario
    mean_lsts = [all_scenario_results[s]['lst_2035'].mean() 
                 for s in scenario_names]
    current_lst = gdf['lst_mean'].mean()
    
    colors = ['#D62728', '#FF7F0E', '#2CA02C', '#1F77B4']
    bars = axes[0].bar(
        [SCENARIO_PRESETS[s]['name'].split('(')[0].strip() for s in scenario_names],
        mean_lsts, color=colors[:len(scenario_names)], edgecolor='white'
    )
    axes[0].axhline(current_lst, color='black', linestyle='--',
                   linewidth=1.5, label=f'Current: {current_lst:.1f}°C')
    axes[0].set_ylabel('Mean City LST (°C)', fontsize=11)
    axes[0].set_title('Projected 2035 Mean LST\nby Intervention Scenario',
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, mean_lsts):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.1, 
                    f'{val:.1f}°C', ha='center', va='bottom', fontsize=9)
    
    # 2. Mean TII 2035 by scenario
    mean_tiis = [all_scenario_results[s]['TII_2035'].mean()
                 for s in scenario_names]
    current_tii = gdf['TII'].mean()
    
    bars2 = axes[1].bar(
        [SCENARIO_PRESETS[s]['name'].split('(')[0].strip() for s in scenario_names],
        mean_tiis, color=colors[:len(scenario_names)], edgecolor='white'
    )
    axes[1].axhline(current_tii, color='black', linestyle='--',
                   linewidth=1.5, label=f'Current TII: {current_tii:.1f}')
    axes[1].set_ylabel('Mean Thermal Injustice Index', fontsize=11)
    axes[1].set_title('Projected 2035 TII\nEquity Impact by Scenario',
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', alpha=0.3)
    
    # 3. TII reduction in top-20% most vulnerable
    tii_p80 = gdf['TII'].quantile(0.8)
    high_risk = gdf[gdf['TII'] >= tii_p80]
    
    hi_risk_improvement = []
    for s in scenario_names:
        proj = all_scenario_results[s]
        hi_blocks = proj[proj.index.isin(high_risk.index)]
        tii_reduction = (high_risk['TII'].mean() - hi_blocks['TII_2035'].mean())
        hi_risk_improvement.append(tii_reduction)
    
    bar_colors = ['#D62728' if x < 0 else '#2CA02C' for x in hi_risk_improvement]
    axes[2].bar(
        [SCENARIO_PRESETS[s]['name'].split('(')[0].strip() for s in scenario_names],
        hi_risk_improvement,
        color=bar_colors, edgecolor='white'
    )
    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].set_ylabel('TII Reduction in Top-20% Hotspots', fontsize=11)
    axes[2].set_title('Justice Impact:\nTII reduction in most vulnerable zones',
                     fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=15)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('2035 Scenario Simulation — What Could Change?',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, 'scenario_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  📊 Scenario comparison saved: {path}")

# ── Interactive Dual Map ────────────────────────────────────────────────────────

def create_scenario_map(gdf, gdf_best_scenario, scenario_name):
    """Create before/after folium map for best intervention scenario."""
    
    m = folium.Map(location=[21.18, 72.83], zoom_start=12, 
                  tiles='CartoDB positron')
    
    # Colour scale
    def tii_to_color(tii, max_tii=100):
        r = min(1.0, 2 * tii / max_tii)
        g = min(1.0, 2 * (1 - tii / max_tii))
        return f'#{int(r*200):02x}{int(g*160):02x}40'
    
    for _, row in gdf_best_scenario.iterrows():
        tii_now  = row.get('TII', 50)
        tii_2035 = row.get('TII_2035', 50)
        delta    = row.get('TII_delta', 0)
        lst_2035 = row.get('lst_2035', 35)
        
        popup_html = f"""
        <div style="font-family: sans-serif; width: 220px; font-size: 12px;">
          <b>Block: {row.get('block_id', 'N/A')}</b><br>
          <b>Ward: {row.get('ward', 'N/A')}</b><br><hr style="margin:4px 0">
          <table style="width:100%">
            <tr><td>Current TII:</td><td><b>{tii_now:.1f}</b></td></tr>
            <tr><td>2035 TII:</td><td><b>{tii_2035:.1f}</b></td></tr>
            <tr><td>Change:</td>
              <td style="color:{'green' if delta < 0 else 'red'}">
                {'▼' if delta < 0 else '▲'} {abs(delta):.1f}
              </td>
            </tr>
            <tr><td>2035 LST:</td><td>{lst_2035:.1f}°C</td></tr>
          </table>
        </div>
        """
        
        # Colour by TII delta
        if delta < -5:
            color = '#1F77B4'   # Strong improvement
        elif delta < 0:
            color = '#98DF8A'   # Mild improvement  
        elif delta < 5:
            color = '#FFBB78'   # Neutral
        else:
            color = '#D62728'   # Worsened
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"TII: {tii_now:.0f}→{tii_2035:.0f} | Δ{delta:+.1f}"
        ).add_to(m)
    
    # Legend
    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:999;
                background:white; border:1px solid #ccc; border-radius:8px;
                padding:12px; font-family:sans-serif; font-size:12px">
      <b>2035 TII Change<br>({scenario_name})</b><br><br>
      <span style="color:#1F77B4">●</span> Strong improvement (Δ &lt; -5)<br>
      <span style="color:#98DF8A">●</span> Mild improvement<br>
      <span style="color:#FFBB78">●</span> Neutral<br>
      <span style="color:#D62728">●</span> Worsened<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    path = os.path.join(OUTPUT_DIR, '2035_scenario_map.html')
    m.save(path)
    print(f"  🗺️  2035 scenario map saved: {path}")
    return m

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("🔮 2035 Heat Scenario Simulation\n")
    
    feat_path = os.path.join(INPUT_DIR, 'features_final.geojson')
    if not os.path.exists(feat_path):
        print("🔄 Running features pipeline first...")
        import sys; sys.path.append('../02_feature_engineering')
        from tii_calculator import main as tii_main
        gdf = tii_main()
    else:
        gdf = gpd.read_file(feat_path)
    
    print(f"📍 Loaded {len(gdf)} blocks\n")
    print(f"  Current mean TII: {gdf['TII'].mean():.1f}")
    print(f"  Current mean LST: {gdf['lst_mean'].mean():.1f}°C\n")
    
    all_results = {}
    
    for scenario_key, scenario_params in SCENARIO_PRESETS.items():
        print(f"  Simulating: {scenario_params['name']}")
        gdf_projected = simulate_scenario(gdf, scenario_params)
        all_results[scenario_key] = gdf_projected
        
        delta_lst = gdf_projected['lst_2035'].mean() - gdf['lst_mean'].mean()
        delta_tii = gdf_projected['TII_2035'].mean() - gdf['TII'].mean()
        print(f"    LST 2035: {gdf_projected['lst_2035'].mean():.1f}°C "
              f"({delta_lst:+.2f}°C vs current)")
        print(f"    TII 2035: {gdf_projected['TII_2035'].mean():.1f} "
              f"({delta_tii:+.1f} vs current)\n")
    
    # Compare scenarios
    plot_scenario_comparison(gdf, all_results)
    
    # Best scenario map
    best_scenario = 'justice_first'
    create_scenario_map(gdf, all_results[best_scenario],
                        SCENARIO_PRESETS[best_scenario]['name'])
    
    # Save all scenario summaries
    summary = {}
    for key, proj_gdf in all_results.items():
        summary[key] = {
            'scenario_name': SCENARIO_PRESETS[key]['name'],
            'description': SCENARIO_PRESETS[key]['description'],
            'mean_lst_2035': round(float(proj_gdf['lst_2035'].mean()), 2),
            'mean_tii_2035': round(float(proj_gdf['TII_2035'].mean()), 2),
            'delta_lst':     round(float(proj_gdf['lst_2035'].mean() - gdf['lst_mean'].mean()), 2),
            'delta_tii':     round(float(proj_gdf['TII_2035'].mean() - gdf['TII'].mean()), 2),
            'n_improved':    int((proj_gdf['TII_delta'] < 0).sum()),
            'n_worsened':    int((proj_gdf['TII_delta'] > 5).sum()),
        }
    
    out_path = os.path.join(OUTPUT_DIR, 'scenario_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Scenario simulation complete.")
    print(f"   Summary: {out_path}")
    
    return all_results

if __name__ == "__main__":
    main()
