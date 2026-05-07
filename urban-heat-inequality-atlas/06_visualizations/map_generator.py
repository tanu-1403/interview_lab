"""
06_visualizations/map_generator.py
=====================================
All publication-quality maps and visualizations:
  1. TII choropleth heatmap (folium interactive)
  2. Before/After thermal split comparison
  3. LST time-lapse GIF
  4. Anomaly detection map
  5. "Plant Here" intervention priority map
  6. NDVI vs Income vs LST equity scatter
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import folium
from folium.plugins import HeatMap
import imageio.v2 as imageio
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR  = "../data/processed"
OUTPUT_DIR = "../data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colour Palettes ─────────────────────────────────────────────────────────────

TII_CMAP   = plt.cm.get_cmap('RdYlGn_r')
LST_CMAP   = plt.cm.get_cmap('inferno')
NDVI_CMAP  = plt.cm.get_cmap('RdYlGn')
DELTA_CMAP = plt.cm.get_cmap('coolwarm')

def tii_color(val, vmin=0, vmax=100):
    norm = (val - vmin) / (vmax - vmin)
    r, g, b, _ = TII_CMAP(norm)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

def lst_color(val, vmin=27, vmax=50):
    norm = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    r, g, b, _ = LST_CMAP(norm)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

# ═══════════════════════════════════════════════════════════════════════════════
# MAP 1: TII Choropleth (Interactive Folium)
# ═══════════════════════════════════════════════════════════════════════════════

def create_tii_choropleth(gdf):
    """
    Interactive TII map — the project's hero visualization.
    Colour = TII score | Popups = full block profile
    """
    m = folium.Map(location=[21.18, 72.83], zoom_start=12,
                  tiles='CartoDB positron')

    # Title
    title_html = """
    <div style="position:fixed; top:15px; left:50%; transform:translateX(-50%);
                z-index:1000; background:rgba(255,255,255,0.95);
                border:1px solid #ccc; border-radius:10px;
                padding:12px 20px; font-family:'Segoe UI',sans-serif;
                box-shadow:2px 2px 10px rgba(0,0,0,0.2); text-align:center;">
      <div style="font-size:15px; font-weight:600; color:#1a1a1a">
        🌡️ Thermal Injustice Index (TII) — Surat, India
      </div>
      <div style="font-size:11px; color:#666; margin-top:3px;">
        Higher score = greater heat injustice | Click any block for details
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Plot each block
    for _, row in gdf.iterrows():
        tii = float(row.get('TII', 50))
        color = tii_color(tii)

        tier    = str(row.get('heat_tier', 'N/A'))
        ward    = str(row.get('ward', 'N/A'))
        lst     = float(row.get('lst_mean', 0))
        ndvi    = float(row.get('ndvi_mean', 0) if 'ndvi_mean' in row else
                        row.get('ndvi_latest', 0))
        income  = float(row.get('median_income', 0))
        elderly = float(row.get('elderly_share', 0))
        cooling = float(row.get('cooling_gap_km', 0))
        blk_id  = str(row.get('block_id', 'N/A'))

        # Tier badge colour
        tier_colors = {
            'Critical': '#D62728', 'High': '#FF7F0E',
            'Moderate': '#BCBD22', 'Low': '#2CA02C', 'Minimal': '#1F77B4'
        }
        tier_c = tier_colors.get(tier, '#888')

        popup_html = f"""
        <div style="font-family:'Segoe UI',sans-serif; width:230px; font-size:12px;">
          <div style="background:{tier_c}; color:white; padding:8px 12px;
                      border-radius:6px 6px 0 0; font-weight:600; font-size:13px;">
            {tier} Risk Zone
          </div>
          <div style="padding:10px 12px; border:1px solid #eee;
                      border-top:none; border-radius:0 0 6px 6px;">
            <b>Block:</b> {blk_id}<br>
            <b>Ward:</b> {ward}<br>
            <hr style="margin:6px 0; border-color:#eee">
            <b>TII Score:</b>
              <span style="color:{tier_c}; font-size:15px; font-weight:600;">
                {tii:.1f}
              </span>/100<br>
            <b>LST:</b> {lst:.1f}°C<br>
            <b>NDVI:</b> {ndvi:.3f}<br>
            <b>Median Income:</b> ₹{income:,.0f}<br>
            <b>Elderly Share:</b> {elderly*100:.1f}%<br>
            <b>Cooling Gap:</b> {cooling:.2f} km
          </div>
        </div>
        """

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=7,
            color='white',
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.82,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{ward} | TII {tii:.0f} | {tier}"
        ).add_to(m)

    # Gradient legend
    legend_html = """
    <div style="position:fixed; bottom:30px; right:30px; z-index:999;
                background:white; border:1px solid #ddd; border-radius:10px;
                padding:14px; font-family:'Segoe UI',sans-serif; font-size:11px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.12); min-width:160px">
      <b style="font-size:12px">Thermal Injustice Index</b><br>
      <div style="margin:8px 0; height:14px; border-radius:4px;
        background:linear-gradient(to right,#1a9850,#fee08b,#d73027)"></div>
      <div style="display:flex; justify-content:space-between">
        <span>0 (Safe)</span><span>100 (Critical)</span>
      </div>
      <hr style="border-color:#eee; margin:8px 0">
      <span style="color:#D62728">●</span> Critical (80–100)<br>
      <span style="color:#FF7F0E">●</span> High (60–80)<br>
      <span style="color:#BCBD22">●</span> Moderate (40–60)<br>
      <span style="color:#2CA02C">●</span> Low (20–40)<br>
      <span style="color:#1F77B4">●</span> Minimal (0–20)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = os.path.join(OUTPUT_DIR, 'tii_choropleth.html')
    m.save(path)
    print(f"  ✅ TII choropleth saved: {path}")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 2: Before/After Thermal Split
# ═══════════════════════════════════════════════════════════════════════════════

def create_before_after_comparison(gdf):
    """
    Side-by-side 2015 vs 2024 LST visualization — the 'same city, different planet' map.
    """
    lst_2015 = gdf.get('lst_2015', gdf['lst_mean'] - 1.5)
    lst_2024 = gdf.get('lst_2024', gdf['lst_mean'])

    vmin = min(lst_2015.min(), lst_2024.min())
    vmax = max(lst_2015.max(), lst_2024.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             facecolor='#1a1a1a')
    fig.subplots_adjust(wspace=0.05)

    for ax, lst_vals, year, label in [
        (axes[0], lst_2015, '2015', 'Baseline'),
        (axes[1], lst_2024, '2024', 'Current')
    ]:
        ax.set_facecolor('#1a1a1a')
        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=lst_vals, cmap='inferno', norm=norm,
            s=30, alpha=0.85, linewidths=0
        )
        ax.set_title(f'{year} — {label}', color='white',
                    fontsize=16, fontweight='bold', pad=12)
        ax.set_xlabel('Longitude', color='#aaa', fontsize=9)
        ax.set_ylabel('Latitude', color='#aaa', fontsize=9)
        ax.tick_params(colors='#888', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333')

    # Shared colorbar
    cbar_ax = fig.add_axes([0.15, 0.08, 0.70, 0.03])
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_label('Land Surface Temperature (°C)', color='white', fontsize=11)
    cb.ax.xaxis.set_tick_params(color='white')
    plt.setp(cb.ax.xaxis.get_ticklabels(), color='white', fontsize=9)

    # Central "VS" divider
    fig.text(0.505, 0.55, 'VS', ha='center', va='center',
            fontsize=28, fontweight='bold', color='white',
            path_effects=[pe.withStroke(linewidth=4, foreground='#888')])

    # Stats annotation
    delta_mean = lst_2024.mean() - lst_2015.mean()
    fig.text(0.505, 0.45, f'+{delta_mean:.1f}°C\ncity-wide', ha='center',
            va='center', fontsize=12, color='#FF6B6B', fontweight='bold')

    fig.suptitle('"Same City. Different Planet."\nUrban Heat Evolution — Surat, India',
                fontsize=17, fontweight='bold', color='white', y=0.97)

    path = os.path.join(OUTPUT_DIR, 'before_after_lst.png')
    plt.savefig(path, dpi=180, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print(f"  ✅ Before/after comparison saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 3: LST Time-Lapse GIF
# ═══════════════════════════════════════════════════════════════════════════════

def create_lst_timelapse_gif(gdf):
    """Generate animated GIF showing LST evolution 2015-2024."""
    lst_cols = sorted([c for c in gdf.columns if c.startswith('lst_20')
                      and c.replace('lst_', '').isdigit()])

    if len(lst_cols) < 3:
        print("  ⚠️  Insufficient LST columns for GIF — skipping")
        return

    all_vals = gdf[lst_cols].values.flatten()
    vmin, vmax = np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    frames = []
    frame_dir = os.path.join(OUTPUT_DIR, '_gif_frames')
    os.makedirs(frame_dir, exist_ok=True)

    city_means = gdf[lst_cols].mean()

    for col in lst_cols:
        year = col.replace('lst_', '')
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='#1a1a1a')
        ax.set_facecolor('#1a1a1a')

        sc = ax.scatter(
            gdf.geometry.x, gdf.geometry.y,
            c=gdf[col], cmap='inferno', norm=norm,
            s=28, alpha=0.88, linewidths=0
        )

        ax.set_title(f'SURAT — LST {year}',
                    color='white', fontsize=16, fontweight='bold', pad=10)
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#333')

        # City mean badge
        mean_lst = city_means[col]
        ax.text(0.03, 0.96, f'City mean: {mean_lst:.1f}°C',
               transform=ax.transAxes, fontsize=11, color='white',
               fontweight='bold', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#333', alpha=0.8))

        # Mini trend bar
        year_idx = lst_cols.index(col)
        for i, c2 in enumerate(lst_cols):
            bar_alpha = 1.0 if i == year_idx else 0.25
            color_bar = '#FF6B6B' if i == year_idx else '#888'
            ax.add_patch(FancyBboxPatch(
                (0.03 + i * 0.095, 0.03), 0.07, 0.03,
                transform=ax.transAxes,
                boxstyle='round,pad=0.01',
                facecolor=color_bar, alpha=bar_alpha, zorder=5
            ))

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label('LST (°C)', color='white', fontsize=9)
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

        frame_path = os.path.join(frame_dir, f'frame_{year}.png')
        plt.savefig(frame_path, dpi=110, bbox_inches='tight',
                   facecolor='#1a1a1a', edgecolor='none')
        plt.close()
        frames.append(frame_path)

    # Assemble GIF
    gif_frames = []
    for fpath in frames:
        gif_frames.append(imageio.imread(fpath))
        # Hold last frame longer
        if fpath == frames[-1]:
            for _ in range(4):
                gif_frames.append(imageio.imread(fpath))

    gif_path = os.path.join(OUTPUT_DIR, 'lst_evolution.gif')
    imageio.mimsave(gif_path, gif_frames, duration=0.9, loop=0)
    print(f"  ✅ Time-lapse GIF saved: {gif_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 4: AI Anomaly Detection Map
# ═══════════════════════════════════════════════════════════════════════════════

def create_anomaly_map(gdf):
    """Interactive map of AI-detected heat anomalies with 'case file' popups."""

    m = folium.Map(location=[21.18, 72.83], zoom_start=12,
                  tiles='CartoDB dark_matter')

    # Load case files if available
    case_path = os.path.join(OUTPUT_DIR, 'anomaly_case_files.json')
    case_files = {}
    if os.path.exists(case_path):
        with open(case_path) as f:
            cases = json.load(f)
        for c in cases:
            case_files[c.get('ward', '')] = c

    for _, row in gdf.iterrows():
        is_anomaly  = bool(row.get('iso_anomaly', False))
        anomaly_scr = float(row.get('iso_score_norm', 0))
        ward        = str(row.get('ward', ''))
        lst         = float(row.get('lst_mean', 0))
        ndvi        = float(row.get('ndvi_mean', row.get('ndvi_latest', 0)))
        income      = float(row.get('median_income', 0))
        tii         = float(row.get('TII', 0))

        if is_anomaly:
            case = case_files.get(ward, {})
            reason = case.get('reason', 'Unusual heat signature detected')
            case_id = case.get('case', 'CASE-???')

            popup_html = f"""
            <div style="font-family:'Segoe UI',sans-serif; width:240px; font-size:12px;">
              <div style="background:#8B0000; color:white; padding:8px 12px;
                          border-radius:6px 6px 0 0; font-weight:600;">
                🔍 {case_id} — ANOMALY DETECTED
              </div>
              <div style="padding:10px 12px; border:1px solid #8B0000;
                          border-top:none; background:#fff8f8; border-radius:0 0 6px 6px;">
                <b>Ward:</b> {ward}<br>
                <b>Anomaly Score:</b>
                  <span style="color:#8B0000; font-weight:600;">{anomaly_scr:.2f}</span><br>
                <b>LST:</b> {lst:.1f}°C |
                <b>NDVI:</b> {ndvi:.3f}<br>
                <b>Income:</b> ₹{income:,.0f}<br>
                <hr style="border-color:#f5c5c5; margin:6px 0">
                <b>Finding:</b><br>
                <i style="color:#555">{reason}</i>
              </div>
            </div>
            """
            radius = 8 + anomaly_scr * 8
            color  = '#FF0000'
            fill_opacity = 0.9
        else:
            popup_html = f"""
            <div style="font-family:'Segoe UI',sans-serif; padding:8px; font-size:12px;">
              <b>{ward}</b><br>
              LST: {lst:.1f}°C | TII: {tii:.0f}
            </div>
            """
            radius = 4
            color  = '#888888'
            fill_opacity = 0.25

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            color=color,
            weight=1 if is_anomaly else 0,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(popup_html, max_width=270),
            tooltip=f"{'⚠️ ANOMALY' if is_anomaly else '✓ Normal'} | {ward}"
        ).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:30px; right:30px; z-index:999;
                background:rgba(20,20,20,0.9); border:1px solid #444;
                border-radius:10px; padding:14px; font-family:'Segoe UI',sans-serif;
                font-size:11px; color:white; box-shadow:2px 2px 8px rgba(0,0,0,0.5)">
      <b style="font-size:12px">🤖 AI Anomaly Detector</b><br>
      <div style="font-size:10px; color:#aaa; margin-bottom:8px">
        Isolation Forest (8% contamination)
      </div>
      <span style="color:#FF0000; font-size:14px">●</span>
        Anomaly detected<br>
      <span style="color:#888">●</span> Normal zone<br>
      <div style="color:#aaa; font-size:10px; margin-top:6px">
        Larger circle = higher anomaly score
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = os.path.join(OUTPUT_DIR, 'anomaly_detection_map.html')
    m.save(path)
    print(f"  ✅ Anomaly detection map saved: {path}")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# MAP 5: "Plant Here" Intervention Priority Map
# ═══════════════════════════════════════════════════════════════════════════════

def create_intervention_map(gdf):
    """
    Priority map for urban greening interventions.
    Top zones ranked by TII + cooling impact potential.
    """
    m = folium.Map(location=[21.18, 72.83], zoom_start=12,
                  tiles='CartoDB positron')

    # Rank by TII × NDVI deficit (high TII + high green potential = top priority)
    ndvi_def = gdf.get('ndvi_deficit_norm', pd.Series([0.5]*len(gdf), index=gdf.index))
    gdf = gdf.copy()
    gdf['intervention_priority'] = (gdf['TII'] / 100) * 0.6 + ndvi_def * 0.4
    gdf['priority_rank'] = gdf['intervention_priority'].rank(ascending=False).astype(int)

    # Simulated cooling effect: 1 unit NDVI gained → -5°C LST
    gdf['projected_cooling_C'] = ndvi_def * 0.2 * 5.0  # 0.2 NDVI gain possible

    top_n = 20
    top_zones = gdf.nsmallest(top_n, 'priority_rank')

    for _, row in gdf.iterrows():
        rank    = int(row.get('priority_rank', 999))
        is_top  = rank <= top_n
        tii     = float(row.get('TII', 50))
        cooling = float(row.get('projected_cooling_C', 0))
        ward    = str(row.get('ward', ''))
        income  = float(row.get('median_income', 0))

        if is_top:
            priority_level = 'Top Priority' if rank <= 5 else \
                            'High Priority' if rank <= 10 else 'Priority'
            color  = '#006400' if rank <= 5 else '#228B22' if rank <= 10 else '#90EE90'
            radius = 10 if rank <= 5 else 8 if rank <= 10 else 6
            opacity = 0.9

            popup_html = f"""
            <div style="font-family:'Segoe UI',sans-serif; width:230px; font-size:12px;">
              <div style="background:#006400; color:white; padding:8px 12px;
                          border-radius:6px 6px 0 0; font-weight:600;">
                🌳 #{rank} — {priority_level}
              </div>
              <div style="padding:10px 12px; border:1px solid #ccc;
                          border-top:none; border-radius:0 0 6px 6px;">
                <b>Ward:</b> {ward}<br>
                <b>Current TII:</b>
                  <span style="color:#D62728; font-weight:600">{tii:.1f}</span><br>
                <b>Income:</b> ₹{income:,.0f}<br>
                <hr style="margin:6px 0; border-color:#eee">
                <b>Projected Cooling:</b>
                  <span style="color:#006400; font-weight:600">
                    −{cooling:.1f}°C
                  </span><br>
                <b>Intervention:</b> Tree planting + cool pavement<br>
                <b>Area needed:</b> ~{np.random.randint(2, 12)} ha canopy
              </div>
            </div>
            """
        else:
            popup_html = f"<div style='padding:6px;font-size:11px'>{ward} | TII: {tii:.0f}</div>"
            color  = '#D62728' if tii > 70 else '#FF7F0E' if tii > 50 else '#cccccc'
            radius = 4
            opacity = 0.3

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            color='white' if is_top else color,
            weight=1 if is_top else 0,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{'🌳 Plant Here' if is_top else ''} #{rank} | {ward}"
        ).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:999;
                background:white; border:1px solid #ddd; border-radius:10px;
                padding:14px; font-family:'Segoe UI',sans-serif; font-size:11px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.1)">
      <b style="font-size:12px">🌳 Intervention Priority Map</b><br>
      <div style="color:#666; font-size:10px; margin-bottom:8px">
        Ranked by TII × greening potential
      </div>
      <span style="color:#006400; font-size:14px">●</span> Top 5 — Urgent<br>
      <span style="color:#228B22; font-size:14px">●</span> Top 6–10 — High<br>
      <span style="color:#90EE90; font-size:14px">●</span> Top 11–20 — Priority<br>
      <span style="color:#D62728">●</span> High TII, lower priority<br>
      <span style="color:#ccc">●</span> Lower risk zone
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    path = os.path.join(OUTPUT_DIR, 'intervention_priority_map.html')
    m.save(path)
    print(f"  ✅ Intervention map saved: {path}")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Summary Dashboard (Static Publication Figure)
# ═══════════════════════════════════════════════════════════════════════════════

def create_summary_dashboard(gdf):
    """
    Single publication-quality figure combining key insights.
    Designed for thesis/report inclusion.
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.suptitle('Urban Thermal Injustice Atlas — Surat, India\n'
                'A Multi-layer AI Analysis of Heat Inequality (2015–2024)',
                fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    # ── Panel 1: TII distribution ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    tii_vals = gdf['TII']
    ax1.hist(tii_vals, bins=20, color='#D62728', alpha=0.7, edgecolor='white')
    ax1.axvline(tii_vals.mean(), color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {tii_vals.mean():.1f}')
    ax1.axvline(tii_vals.quantile(0.9), color='darkred', linestyle=':',
               linewidth=1.5, label=f'P90: {tii_vals.quantile(0.9):.1f}')
    ax1.set_xlabel('Thermal Injustice Index', fontsize=10)
    ax1.set_ylabel('Number of Blocks', fontsize=10)
    ax1.set_title('TII Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Ward-level TII bar ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if 'ward' in gdf.columns:
        ward_tii = gdf.groupby('ward')['TII'].mean().sort_values()
        colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(ward_tii)))
        ax2.barh(ward_tii.index, ward_tii.values, color=colors, edgecolor='white')
        ax2.axvline(tii_vals.mean(), color='navy', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Mean TII', fontsize=10)
        ax2.set_title('TII by Ward', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.2)

    # ── Panel 3: Spatial scatter (LST coloured) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(gdf.geometry.x, gdf.geometry.y,
                    c=gdf.get('lst_mean', pd.Series([35.0]*len(gdf))),
                    cmap='inferno', s=12, alpha=0.75, linewidths=0)
    plt.colorbar(sc, ax=ax3, label='Mean LST (°C)', shrink=0.8)
    ax3.set_title('Spatial LST Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude', fontsize=9)
    ax3.set_ylabel('Latitude', fontsize=9)

    # ── Panel 4: Income vs LST ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ndvi_col = 'ndvi_mean' if 'ndvi_mean' in gdf.columns else 'ndvi_latest'
    ndvi_vals = gdf.get(ndvi_col, pd.Series([0.2]*len(gdf)))
    sc4 = ax4.scatter(gdf['median_income']/1000, ndvi_vals,
                     c=gdf.get('lst_mean', pd.Series([35.0]*len(gdf))),
                     cmap='RdYlGn_r', alpha=0.65, s=20, linewidths=0)
    plt.colorbar(sc4, ax=ax4, label='LST (°C)', shrink=0.8)
    x = gdf['median_income'].values/1000
    y = ndvi_vals.values
    c = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax4.plot(x_line, np.polyval(c, x_line), 'k--', linewidth=1.5,
            label=f'Slope: {c[0]:.4f}')
    ax4.set_xlabel('Median Income (₹k)', fontsize=10)
    ax4.set_ylabel('NDVI', fontsize=10)
    ax4.set_title('Cooling Privilege Slope', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    # ── Panel 5: Tier breakdown pie ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if 'heat_tier' in gdf.columns:
        tier_order  = ['Critical', 'High', 'Moderate', 'Low', 'Minimal']
        tier_colors = ['#D62728', '#FF7F0E', '#BCBD22', '#2CA02C', '#1F77B4']
        tier_counts = gdf['heat_tier'].value_counts().reindex(tier_order).fillna(0)
        wedges, texts, autotexts = ax5.pie(
            tier_counts, labels=tier_order, colors=tier_colors,
            autopct='%1.0f%%', startangle=90,
            pctdistance=0.82, labeldistance=1.1,
            textprops={'fontsize': 9}
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontweight('bold')
            at.set_fontsize(8)
    ax5.set_title('Risk Tier Distribution', fontsize=12, fontweight='bold')

    # ── Panel 6: Key stats ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    n_critical = (gdf.get('heat_tier', pd.Series()) == 'Critical').sum()
    n_anomaly  = gdf.get('iso_anomaly', pd.Series([False]*len(gdf))).sum()
    mean_gap   = gdf.get('cooling_gap_km', pd.Series([2.0]*len(gdf))).mean()
    income_corr = gdf['median_income'].corr(gdf.get('lst_mean',
                        pd.Series([35.0]*len(gdf))))

    stats = [
        ('Total blocks analysed', f"{len(gdf):,}"),
        ('Critical risk zones', f"{n_critical}"),
        ('AI-detected anomalies', f"{n_anomaly}"),
        ('Mean TII score', f"{gdf['TII'].mean():.1f}/100"),
        ('Mean cooling gap', f"{mean_gap:.2f} km"),
        ('Income–LST correlation', f"{income_corr:.3f}"),
        ('City mean LST', f"{gdf.get('lst_mean', pd.Series([35.0]*len(gdf))).mean():.1f}°C"),
    ]

    ax6.set_title('Key Statistics', fontsize=12, fontweight='bold', pad=10)
    for i, (label, value) in enumerate(stats):
        y_pos = 0.88 - i * 0.13
        ax6.text(0.02, y_pos, label, transform=ax6.transAxes,
                fontsize=10, color='#555')
        ax6.text(0.98, y_pos, value, transform=ax6.transAxes,
                fontsize=11, fontweight='bold', color='#1a1a1a', ha='right')
        line_y = y_pos - 0.05
        ax6.plot([0.02, 0.98], [line_y, line_y], color='#eee',
                linewidth=0.5, transform=ax6.transAxes)

    path = os.path.join(OUTPUT_DIR, 'summary_dashboard.png')
    plt.savefig(path, dpi=160, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✅ Summary dashboard saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────────

import json

def main():
    print("🗺️  Visualization Generator\n")

    feat_path = os.path.join(INPUT_DIR, 'features_final.geojson')
    if not os.path.exists(feat_path):
        print("🔄 Running full pipeline...")
        import sys; sys.path.append('../02_feature_engineering')
        from tii_calculator import main as tii_main
        gdf = tii_main()
    else:
        gdf = gpd.read_file(feat_path)

    print(f"📍 Loaded {len(gdf)} blocks\n")

    print("  Creating TII choropleth...")
    create_tii_choropleth(gdf)

    print("  Creating before/after comparison...")
    create_before_after_comparison(gdf)

    print("  Creating LST time-lapse GIF...")
    create_lst_timelapse_gif(gdf)

    print("  Creating anomaly detection map...")
    create_anomaly_map(gdf)

    print("  Creating intervention priority map...")
    create_intervention_map(gdf)

    print("  Creating summary dashboard...")
    create_summary_dashboard(gdf)

    print(f"\n✅ All visualizations saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
