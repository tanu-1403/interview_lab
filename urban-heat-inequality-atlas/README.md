# 🌡️ Urban Heat Inequality Atlas
### AI-Driven Thermal Justice Analysis using Remote Sensing

> *"This project doesn't just map heat. It maps who is being failed by their city — and gives planners a ranked action list to start fixing it."*

---

## 📌 Project Summary

A complete geospatial AI pipeline that transforms Landsat/Sentinel satellite data into a **Thermal Injustice Index (TII)** — exposing how urban heat disproportionately affects low-income and vulnerable communities. Combines remote sensing, ML anomaly detection, temporal forecasting, and policy-grade scenario simulation.

**Study Area:** Surat, Gujarat, India (adaptable to any city)  
**Time Span:** 2015–2024  
**Key Output:** Ranked "heat injustice" scorecard per census block + 2035 scenario simulator

---

## 🗂️ Project Structure

```
urban-heat-inequality-atlas/
├── 00_data_download/         ← GEE + Census API scripts
├── 01_preprocessing/         ← LST retrieval, NDVI, cloud masking
├── 02_feature_engineering/   ← TII construction, anomaly scores
├── 03_clustering/            ← Thermal zone segmentation
├── 04_ml_models/             ← XGBoost + LSTM + SHAP explainability
├── 05_scenario_sim/          ← 2035 counterfactual engine
├── 06_visualizations/        ← Folium, Plotly, static maps
├── dashboard/                ← Streamlit interactive app
├── report/                   ← MSc-level LaTeX report
├── data/
│   ├── raw/                  ← Downloaded satellite + census data
│   ├── processed/            ← Cleaned, merged feature tables
│   └── outputs/              ← Model results, maps, exports
└── notebooks/                ← End-to-end narrative notebooks
```

---

## 🚀 Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/yourusername/urban-heat-inequality-atlas
cd urban-heat-inequality-atlas
pip install -r requirements.txt

# 2. Download data (requires GEE authentication)
python 00_data_download/gee_download.py

# 3. Run preprocessing pipeline
python 01_preprocessing/lst_retrieval.py
python 01_preprocessing/ndvi_processing.py

# 4. Build features and TII
python 02_feature_engineering/tii_calculator.py

# 5. Run clustering
python 03_clustering/thermal_zones.py

# 6. Train models
python 04_ml_models/xgboost_vulnerability.py
python 04_ml_models/lstm_temporal.py

# 7. Launch dashboard
streamlit run dashboard/app.py
```

---

## 🧠 The Thermal Injustice Index (TII)

The TII is the signature output of this project — a 0–100 composite score per spatial unit:

```
TII = w1*(LST_anomaly_z) + w2*(NDVI_deficit) + w3*(1 - income_percentile) 
      + w4*(cooling_gap_km) + w5*(elderly_share)
```

Where weights are learned via XGBoost feature importance, normalized to sum to 1.

**Higher TII = greater thermal injustice = higher intervention priority**

---

## 📊 Key Outputs

| Output | Description |
|--------|-------------|
| `tii_choropleth.html` | Interactive folium map of TII scores |
| `thermal_zones_map.html` | K-Means cluster zones |
| `anomaly_detection_map.html` | AI-flagged heat anomalies |
| `2035_scenario_simulator` | Streamlit slider-based future projection |
| `shap_analysis.png` | XGBoost feature importance |
| `lst_timeseries.gif` | Animated heat evolution 2015–2024 |

---

## 📦 Dependencies

See `requirements.txt` for full list. Key packages:
- `earthengine-api` — Google Earth Engine
- `geemap` — GEE + folium integration
- `geopandas`, `rasterio`, `shapely` — spatial processing
- `xgboost`, `shap` — ML + explainability
- `tensorflow` — LSTM temporal model
- `streamlit`, `plotly`, `folium` — visualization & dashboard
- `prophet` — time series forecasting

---

## 🎓 Academic Context

This project was developed as an MSc Geospatial AI case study. The methodology draws on:
- Urban Heat Island literature (Oke, 1982; Heaviside et al., 2017)
- Environmental justice frameworks (Mohai et al., 2009)
- Remote sensing best practices (USGS Landsat Collection 2)
- Explainable AI for spatial analysis (Reichstein et al., 2019)

---

## 📄 License

MIT License — free to use, adapt, and build upon with attribution.
