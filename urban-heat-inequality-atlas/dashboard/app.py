"""
dashboard/app.py
==================
Streamlit Interactive Dashboard — Urban Heat Inequality Atlas

Features:
  1. City-wide TII overview with live metrics
  2. Interactive 2035 Scenario Simulator (slider-based)
  3. Anomaly Detective case files viewer
  4. Ward comparison tool
  5. "Plant Here" intervention calculator
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
FEAT_PATH     = os.path.join(os.path.dirname(__file__), '../data/processed/features_final.geojson')
OUTPUTS_PATH  = os.path.join(os.path.dirname(__file__), '../data/outputs')
SCENARIO_PATH = os.path.join(OUTPUTS_PATH, 'scenario_summary.json')
ANOMALY_PATH  = os.path.join(OUTPUTS_PATH, 'anomaly_case_files.json')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Heat Inequality Atlas — Surat",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    color: white;
  }
  .metric-value { font-size: 28px; font-weight: 700; margin-bottom: 4px; }
  .metric-label { font-size: 12px; color: #aaa; letter-spacing: 0.05em; }
  .tier-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
  }
  .section-header {
    font-size: 20px; font-weight: 700;
    border-left: 4px solid #e74c3c;
    padding-left: 12px; margin: 24px 0 16px 0;
  }
  .case-file {
    background: #fff8f8; border: 1px solid #f5c5c5;
    border-left: 4px solid #8B0000;
    border-radius: 8px; padding: 14px; margin-bottom: 12px;
  }
  .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Data Loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load or generate all required data."""
    if os.path.exists(FEAT_PATH):
        gdf = gpd.read_file(FEAT_PATH)
    else:
        st.info("🔄 First run — generating data pipeline...")
        from run_pipeline import run_full_pipeline
        gdf = run_full_pipeline()
    return gdf

@st.cache_data
def load_anomaly_cases():
    if os.path.exists(ANOMALY_PATH):
        with open(ANOMALY_PATH) as f:
            return json.load(f)
    return []

@st.cache_data
def load_scenario_summary():
    if os.path.exists(SCENARIO_PATH):
        with open(SCENARIO_PATH) as f:
            return json.load(f)
    return {}

# ── Scenario Simulation (live) ─────────────────────────────────────────────────
def simulate_live(gdf, canopy_change, isf_change, climate_warming):
    """
    Real-time scenario simulation for the dashboard sliders.
    Returns projected mean LST, mean TII, and per-block deltas.
    """
    from sklearn.preprocessing import MinMaxScaler

    LST_COEFFICIENTS = {
        'canopy': -2.8,   # °C per 10% canopy increase
        'isf':    +1.2,   # °C per 10% ISF increase
    }

    ndvi_col = 'ndvi_mean' if 'ndvi_mean' in gdf.columns else 'ndvi_latest'
    ndvi_base = gdf.get(ndvi_col, pd.Series([0.2]*len(gdf), index=gdf.index))
    lst_base  = gdf.get('lst_mean', pd.Series([35.0]*len(gdf), index=gdf.index))

    canopy_lst_delta = (canopy_change / 10.0) * LST_COEFFICIENTS['canopy']
    isf_lst_delta    = (isf_change   / 10.0) * LST_COEFFICIENTS['isf']

    lst_2035 = lst_base + canopy_lst_delta + isf_lst_delta + climate_warming
    delta_lst = lst_2035 - lst_base

    # NDVI change from canopy
    ndvi_change = canopy_change * 0.004
    ndvi_2035   = (ndvi_base + ndvi_change).clip(0, 1)

    # TII re-score
    target_ndvi = 0.35
    ndvi_def_2035 = (target_ndvi - ndvi_2035).clip(lower=0)
    lst_z_2035    = (lst_2035 - lst_2035.mean()) / (lst_2035.std() + 1e-6)

    ndvi_def_n = MinMaxScaler().fit_transform(ndvi_def_2035.values.reshape(-1,1)).flatten()
    lst_z_n    = MinMaxScaler().fit_transform(lst_z_2035.values.reshape(-1,1)).flatten()
    inc_vuln   = gdf.get('income_vuln_norm', pd.Series([0.5]*len(gdf))).values
    cool_gap   = gdf.get('cooling_gap_norm', pd.Series([0.5]*len(gdf))).values
    elderly    = gdf.get('elderly_share_norm', pd.Series([0.3]*len(gdf))).values

    tii_raw = (0.30 * lst_z_n + 0.25 * ndvi_def_n +
               0.20 * inc_vuln + 0.15 * cool_gap + 0.10 * elderly)
    tii_2035 = pd.Series(
        (tii_raw - tii_raw.min()) / (tii_raw.max() - tii_raw.min() + 1e-6) * 100,
        index=gdf.index
    )

    return {
        'mean_lst_2035': float(lst_2035.mean()),
        'mean_tii_2035': float(tii_2035.mean()),
        'delta_lst':     float(delta_lst.mean()),
        'delta_tii':     float(tii_2035.mean() - gdf['TII'].mean()),
        'tii_2035':      tii_2035,
        'lst_2035':      lst_2035,
        'n_improved':    int((tii_2035 < gdf['TII']).sum()),
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/placeholder/logo.png",
             use_column_width=True) if False else None
    st.markdown("## 🌡️ Heat Atlas")
    st.markdown("**Urban Thermal Injustice Analysis**")
    st.markdown("📍 Surat, Gujarat, India")
    st.markdown("📅 2015–2024")
    st.divider()

    st.markdown("### Filters")
    ward_filter = st.multiselect("Filter by Ward", options=[],
                                  placeholder="All wards")
    tier_filter = st.multiselect(
        "Filter by Risk Tier",
        options=['Critical', 'High', 'Moderate', 'Low', 'Minimal'],
        default=['Critical', 'High']
    )
    st.divider()
    st.markdown("### About")
    st.caption(
        "This dashboard is part of an MSc Geospatial AI case study. "
        "The Thermal Injustice Index (TII) combines satellite-derived "
        "LST, NDVI, and socioeconomic vulnerability into a 0–100 risk score."
    )

# ── Main Content ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:28px; font-weight:700; margin-bottom:4px'>
  🌡️ Urban Heat Inequality Atlas
</h1>
<p style='color:#666; font-size:14px; margin-bottom:20px'>
  AI-powered thermal injustice analysis · Surat, India · 2015–2024
</p>
""", unsafe_allow_html=True)

# Load data
with st.spinner("Loading spatial data..."):
    gdf = load_data()
    cases = load_anomaly_cases()

# Update sidebar ward filter options
wards = sorted(gdf['ward'].unique().tolist()) if 'ward' in gdf.columns else []
with st.sidebar:
    ward_filter = st.multiselect("Filter by Ward", options=wards,
                                  placeholder="All wards", key='ward_filter_real')

# Apply filters
gdf_filtered = gdf.copy()
if ward_filter:
    gdf_filtered = gdf_filtered[gdf_filtered['ward'].isin(ward_filter)]
if tier_filter and 'heat_tier' in gdf_filtered.columns:
    gdf_filtered = gdf_filtered[gdf_filtered['heat_tier'].astype(str).isin(tier_filter)]

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🗺️ TII Map",
    "🔮 2035 Simulator",
    "🔍 Anomaly Detective",
    "🌳 Intervention Planner"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    mean_tii  = gdf_filtered['TII'].mean()
    n_critical = (gdf_filtered.get('heat_tier', pd.Series()) == 'Critical').sum()
    mean_lst  = gdf_filtered.get('lst_mean', pd.Series([35.0]*len(gdf_filtered))).mean()
    n_anomaly = gdf_filtered.get('iso_anomaly', pd.Series([False]*len(gdf_filtered))).sum()
    mean_gap  = gdf_filtered.get('cooling_gap_km', pd.Series([2.0]*len(gdf_filtered))).mean()

    def metric_card(col, value, label, color="#e74c3c"):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color}">{value}</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    metric_card(col1, f"{mean_tii:.1f}", "Mean TII Score", "#e74c3c")
    metric_card(col2, f"{mean_lst:.1f}°C", "Mean City LST", "#FF7F0E")
    metric_card(col3, str(int(n_critical)), "Critical Zones", "#8B0000")
    metric_card(col4, str(int(n_anomaly)), "AI Anomalies", "#9B59B6")
    metric_card(col5, f"{mean_gap:.1f}km", "Mean Cooling Gap", "#3498DB")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">TII Distribution</div>',
                   unsafe_allow_html=True)
        fig_hist = px.histogram(
            gdf_filtered, x='TII', nbins=25,
            color_discrete_sequence=['#e74c3c'],
            labels={'TII': 'Thermal Injustice Index'},
            template='plotly_white'
        )
        fig_hist.add_vline(x=mean_tii, line_dash='dash', line_color='navy',
                           annotation_text=f"Mean: {mean_tii:.1f}",
                           annotation_position="top right")
        fig_hist.update_layout(height=300, margin=dict(t=20, b=40))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">TII by Ward</div>',
                   unsafe_allow_html=True)
        if 'ward' in gdf_filtered.columns:
            ward_avg = gdf_filtered.groupby('ward')['TII'].mean().sort_values()
            fig_ward = px.bar(
                ward_avg.reset_index(),
                x='TII', y='ward', orientation='h',
                color='TII', color_continuous_scale='RdYlGn_r',
                range_color=[0, 100],
                template='plotly_white',
                labels={'TII': 'Mean TII', 'ward': ''}
            )
            fig_ward.update_layout(height=300, margin=dict(t=20, b=40),
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_ward, use_container_width=True)

    # Equity scatter
    st.markdown('<div class="section-header">Cooling Privilege Slope</div>',
               unsafe_allow_html=True)
    ndvi_col = 'ndvi_mean' if 'ndvi_mean' in gdf_filtered.columns else 'ndvi_latest'
    if ndvi_col in gdf_filtered.columns:
        fig_scatter = px.scatter(
            gdf_filtered,
            x='median_income', y=ndvi_col,
            color='lst_mean' if 'lst_mean' in gdf_filtered.columns else 'TII',
            color_continuous_scale='RdYlGn_r',
            size='TII', size_max=15,
            hover_data=['ward', 'TII', 'heat_tier'] if 'ward' in gdf_filtered.columns else ['TII'],
            trendline='ols',
            labels={
                'median_income': 'Median Income (₹)',
                ndvi_col: 'NDVI (Vegetation)',
                'lst_mean': 'LST (°C)',
                'color': 'LST (°C)'
            },
            template='plotly_white',
            title='Income → Vegetation → Cooling: the privilege gradient'
        )
        fig_scatter.update_layout(height=380, margin=dict(t=50, b=40))
        st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: TII MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ Interactive Thermal Injustice Index Map")
    st.caption("Colour scale: red = high TII (injustice), green = low TII (safe)")

    map_col, info_col = st.columns([3, 1])

    with map_col:
        m = folium.Map(location=[21.18, 72.83], zoom_start=12,
                      tiles='CartoDB positron')

        tier_colors = {
            'Critical': '#D62728', 'High': '#FF7F0E',
            'Moderate': '#BCBD22', 'Low': '#2CA02C', 'Minimal': '#1F77B4',
            'nan': '#888888'
        }

        for _, row in gdf_filtered.iterrows():
            tii   = float(row.get('TII', 50))
            tier  = str(row.get('heat_tier', 'Moderate'))
            color = tier_colors.get(tier, '#888888')
            ward  = str(row.get('ward', ''))
            lst   = float(row.get('lst_mean', 0))
            inc   = float(row.get('median_income', 0))

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=6, color='white', weight=0.4,
                fill=True, fill_color=color, fill_opacity=0.82,
                tooltip=f"{ward} | TII: {tii:.0f} | {tier}",
                popup=folium.Popup(
                    f"<b>{ward}</b><br>TII: {tii:.1f}<br>LST: {lst:.1f}°C<br>"
                    f"Income: ₹{inc:,.0f}<br>Tier: {tier}",
                    max_width=200
                )
            ).add_to(m)

        st_folium(m, width=700, height=500)

    with info_col:
        st.markdown("#### Risk Summary")
        if 'heat_tier' in gdf_filtered.columns:
            for tier in ['Critical', 'High', 'Moderate', 'Low', 'Minimal']:
                n = (gdf_filtered['heat_tier'].astype(str) == tier).sum()
                pct = n / len(gdf_filtered) * 100
                color = tier_colors.get(tier, '#888')
                st.markdown(
                    f"<span style='color:{color}'>●</span> **{tier}**: "
                    f"{n} blocks ({pct:.0f}%)",
                    unsafe_allow_html=True
                )

        st.divider()
        st.markdown("#### Top 5 Critical Blocks")
        top5 = gdf_filtered.nlargest(5, 'TII')[
            ['block_id', 'ward', 'TII']
        ] if 'block_id' in gdf_filtered.columns else gdf_filtered.nlargest(5, 'TII')[['ward', 'TII']]
        st.dataframe(top5.round(1), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: 2035 SCENARIO SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔮 2035 Heat Scenario Simulator")
    st.markdown(
        "Adjust the sliders to simulate different urban planning interventions "
        "and see how Surat's thermal injustice profile could change by 2035."
    )

    sim_col, result_col = st.columns([1, 2])

    with sim_col:
        st.markdown("#### 🎛️ Intervention Sliders")

        canopy_change = st.slider(
            "🌳 Tree Canopy Increase (%)",
            min_value=-10, max_value=40, value=0, step=2,
            help="Increase in city-wide canopy cover percentage"
        )
        isf_change = st.slider(
            "🏗️ Impervious Surface Change (%)",
            min_value=-15, max_value=20, value=5, step=1,
            help="Change in impervious surface fraction (positive = more paving)"
        )
        climate_warming = st.slider(
            "🌐 Climate Warming Scenario (°C by 2035)",
            min_value=0.5, max_value=3.0, value=1.4, step=0.1,
            help="RCP 2.6 ≈ 0.8°C | RCP 4.5 ≈ 1.4°C | RCP 8.5 ≈ 2.1°C"
        )

        st.markdown("#### 🔖 Quick Presets")
        col_p1, col_p2 = st.columns(2)
        # (Preset buttons are illustrative — in real app use st.session_state)
        col_p1.button("😟 Do Nothing", type="secondary")
        col_p2.button("🌿 Green City", type="primary")

    # Run simulation
    sim_result = simulate_live(gdf, canopy_change, isf_change, climate_warming)

    with result_col:
        st.markdown("#### 📈 Projected 2035 Outcomes")

        r1, r2, r3, r4 = st.columns(4)
        delta_lst = sim_result['delta_lst']
        delta_tii = sim_result['delta_tii']

        r1.metric("Projected LST", f"{sim_result['mean_lst_2035']:.1f}°C",
                  f"{delta_lst:+.2f}°C",
                  delta_color="inverse")
        r2.metric("Projected TII", f"{sim_result['mean_tii_2035']:.1f}",
                  f"{delta_tii:+.1f}",
                  delta_color="inverse")
        r3.metric("Blocks Improved", str(sim_result['n_improved']),
                  f"{sim_result['n_improved']/len(gdf)*100:.0f}% of city")
        r4.metric("Canopy Effect",
                  f"{canopy_change * -0.28:+.1f}°C",
                  "from trees alone")

        # Comparison bar chart
        scenarios_data = {
            'Scenario': ['Current', 'Your Simulation', 'Do Nothing', 'Green City (+30%)'],
            'Mean LST (°C)': [
                gdf['lst_mean'].mean() if 'lst_mean' in gdf.columns else 35.5,
                sim_result['mean_lst_2035'],
                (gdf['lst_mean'].mean() if 'lst_mean' in gdf.columns else 35.5) + 2.1,
                (gdf['lst_mean'].mean() if 'lst_mean' in gdf.columns else 35.5) - 0.5,
            ],
            'Mean TII': [
                gdf['TII'].mean(),
                sim_result['mean_tii_2035'],
                gdf['TII'].mean() + 8,
                gdf['TII'].mean() - 12,
            ]
        }

        df_comp = pd.DataFrame(scenarios_data)
        fig_comp = make_subplots(rows=1, cols=2,
                                  subplot_titles=('Mean LST (°C)', 'Mean TII'))

        colors_bar = ['#888888', '#3498DB', '#D62728', '#2CA02C']

        for i, (metric, row_n) in enumerate([('Mean LST (°C)', 1), ('Mean TII', 2)]):
            for j, (scen, val, col) in enumerate(zip(
                df_comp['Scenario'], df_comp[metric], colors_bar
            )):
                fig_comp.add_trace(
                    go.Bar(name=scen, x=[scen], y=[val],
                           marker_color=col, showlegend=(i == 0)),
                    row=1, col=row_n
                )

        fig_comp.update_layout(height=300, template='plotly_white',
                               margin=dict(t=40, b=20), barmode='group')
        st.plotly_chart(fig_comp, use_container_width=True)

        # TII distribution shift
        tii_now  = gdf['TII']
        tii_2035 = sim_result['tii_2035']
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=tii_now, name='Current',
                                        opacity=0.6, marker_color='#e74c3c',
                                        nbinsx=20))
        fig_dist.add_trace(go.Histogram(x=tii_2035, name='2035 Projected',
                                        opacity=0.6, marker_color='#3498DB',
                                        nbinsx=20))
        fig_dist.update_layout(
            barmode='overlay',
            title='TII Distribution Shift: Current vs 2035',
            template='plotly_white',
            height=260,
            margin=dict(t=40, b=20),
            legend=dict(x=0.75, y=0.95)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ANOMALY DETECTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔍 AI Heat Anomaly Detective")
    st.markdown(
        "The Isolation Forest algorithm detected zones where the heat signature "
        "is **statistically unexpected** — hotter than their land cover, income, "
        "and history would predict. Each is a 'case file.'"
    )

    n_anomalies = int(gdf.get('iso_anomaly', pd.Series([False]*len(gdf))).sum())
    st.info(
        f"🚨 **{n_anomalies} anomalous zones detected** across {len(gdf)} census blocks "
        f"({n_anomalies/len(gdf)*100:.1f}% of the city)"
    )

    if cases:
        st.markdown("#### 📁 Case Files")
        for i, case in enumerate(cases[:5], 1):
            with st.expander(
                f"📁 {case.get('case','?')} — Ward: {case.get('ward','?')} "
                f"| Anomaly Score: {case.get('anomaly_score',0):.2f}",
                expanded=(i == 1)
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("LST",    f"{case.get('lst',0):.1f}°C")
                c2.metric("NDVI",   f"{case.get('ndvi',0):.3f}")
                c3.metric("TII",    f"{case.get('tii',0):.1f}")
                st.markdown(f"**Finding:** _{case.get('reason','N/A')}_")
                st.markdown(f"**Income:** ₹{case.get('income',0):,.0f}")
    else:
        st.warning("Run `python 04_ml_models/xgboost_vulnerability.py` to generate case files.")

    st.markdown("#### Anomaly Score Distribution")
    if 'iso_score_norm' in gdf.columns:
        fig_iso = px.histogram(
            gdf, x='iso_score_norm', nbins=30,
            color='iso_anomaly' if 'iso_anomaly' in gdf.columns else None,
            color_discrete_map={True: '#D62728', False: '#3498DB'},
            labels={'iso_score_norm': 'Anomaly Score (normalised)'},
            template='plotly_white',
            title='Distribution of Isolation Forest Anomaly Scores'
        )
        fig_iso.update_layout(height=300, margin=dict(t=50, b=40))
        st.plotly_chart(fig_iso, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: INTERVENTION PLANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🌳 Intervention Priority Planner")
    st.markdown(
        "Ranked list of census blocks where urban greening will have the "
        "**maximum cooling and equity impact**. Prioritised by TII × NDVI deficit."
    )

    top_n = st.slider("Show top N priority zones", 5, 30, 10)

    ndvi_def = gdf.get('ndvi_deficit_norm',
                        pd.Series([0.5]*len(gdf), index=gdf.index))
    gdf_plan = gdf.copy()
    gdf_plan['priority_score'] = (gdf_plan['TII'] / 100) * 0.6 + ndvi_def * 0.4
    gdf_plan['projected_cooling'] = ndvi_def * 0.2 * 5.0
    gdf_plan['priority_rank'] = gdf_plan['priority_score'].rank(ascending=False).astype(int)

    top_zones = gdf_plan.nsmallest(top_n, 'priority_rank').copy()

    display_cols = ['priority_rank', 'block_id', 'ward', 'TII',
                    'lst_mean', 'median_income', 'projected_cooling']
    display_cols = [c for c in display_cols if c in top_zones.columns]

    top_zones_display = top_zones[display_cols].copy()
    if 'projected_cooling' in top_zones_display.columns:
        top_zones_display['projected_cooling'] = top_zones_display['projected_cooling'].round(2)
        top_zones_display = top_zones_display.rename(
            columns={'projected_cooling': 'Projected Cooling (°C)'}
        )
    if 'lst_mean' in top_zones_display.columns:
        top_zones_display = top_zones_display.rename(columns={'lst_mean': 'LST (°C)'})

    st.dataframe(
        top_zones_display.round(2),
        hide_index=True,
        use_container_width=True,
        column_config={
            'TII': st.column_config.ProgressColumn('TII', min_value=0, max_value=100),
            'priority_rank': st.column_config.NumberColumn('Rank', format="%d"),
        }
    )

    fig_priority = px.scatter(
        top_zones,
        x='TII', y='projected_cooling',
        size='TII', color='median_income',
        hover_name='ward' if 'ward' in top_zones.columns else None,
        color_continuous_scale='RdYlGn',
        labels={
            'TII': 'Thermal Injustice Index',
            'projected_cooling': 'Projected Cooling (°C)',
            'median_income': 'Median Income (₹)'
        },
        template='plotly_white',
        title=f'Top {top_n} Zones: TII vs Projected Cooling Benefit'
    )
    fig_priority.update_layout(height=380, margin=dict(t=50, b=40))
    st.plotly_chart(fig_priority, use_container_width=True)

    st.divider()
    st.markdown("#### 💡 Policy Recommendation Generator")
    if st.button("Generate Policy Brief for Top 5 Zones", type="primary"):
        st.markdown("---")
        for _, row in top_zones.head(5).iterrows():
            rank = int(row.get('priority_rank', 0))
            ward = str(row.get('ward', 'Unknown'))
            tii  = float(row.get('TII', 0))
            lst  = float(row.get('lst_mean' if 'lst_mean' in row else 'LST (°C)', 0))
            inc  = float(row.get('median_income', 0))
            cool = float(row.get('projected_cooling', row.get('Projected Cooling (°C)', 0)))

            st.markdown(f"""
**Priority #{rank} — {ward} Ward**
- Current TII: `{tii:.1f}/100` | LST: `{lst:.1f}°C` | Median Income: `₹{inc:,.0f}`
- **Recommended action:** {'Emergency tree planting + cool pavement' if tii > 70 else 'Targeted canopy expansion + shading structures'}
- **Projected benefit:** `−{cool:.1f}°C` surface cooling with 0.2 NDVI gain
- **Investment priority:** {'🔴 Urgent' if tii > 75 else '🟠 High' if tii > 60 else '🟡 Medium'}
---
""")
