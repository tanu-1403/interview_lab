# 🔮 SalePulse AI
## Retail Intelligence Engine — 8-Week Sales Forecasting Platform

> *"Don't just predict the future. Understand it."*

---

## What Is SalePulse AI?

SalePulse AI is a **production-grade, multi-model time series forecasting system** designed to predict weekly beverage sales for 43 US states up to 8 weeks into the future.

It's not a single model. It's an **intelligent pipeline** that:
1. Engineers 24+ features from raw sales history
2. Trains 4 different models per state (SARIMA, Prophet, XGBoost, Holt-Winters)
3. Evaluates them on a held-out validation window (zero data leakage)
4. **Automatically selects the best model** per state using a composite RMSE+MAPE score
5. Serves live forecasts through a REST API

---

## 🏗️ System Architecture

```
Raw Sales Data (.xlsx)
        │
        ▼
┌─────────────────────┐
│   Data Pipeline     │  ← core/data_pipeline.py
│  • Weekly resample  │
│  • Lag features     │
│  • Rolling stats    │
│  • Cyclical time    │
│  • Holiday flags    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│           Model Zoo                      │  ← models/model_zoo.py
│  ┌──────────┐  ┌───────────────────┐   │
│  │HoltWinters│  │     SARIMA        │   │
│  │(baseline) │  │(1,1,1)(1,1,1,52) │   │
│  └──────────┘  └───────────────────┘   │
│  ┌──────────┐  ┌───────────────────┐   │
│  │  Prophet  │  │     XGBoost       │   │
│  │(Bayesian) │  │(24-feature + lag) │   │
│  └──────────┘  └───────────────────┘   │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Auto Model Select  │  ← Composite score: 60% RMSE + 40% MAPE
│  (per-state winner) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    FastAPI Server   │  ← api/main.py
│  /forecast          │
│  /compare-models    │
│  /insights          │
│  /feature-importance│
│  /health            │
└─────────────────────┘
```

---

## 📁 Project Structure

```
salepulse_ai/
├── core/
│   └── data_pipeline.py      # Feature engineering & analytics
├── models/
│   └── model_zoo.py          # All 4 models + evaluation + selection
├── api/
│   └── main.py               # FastAPI REST service
├── data/
│   └── sales.xlsx            # Input data (place here)
├── outputs/
│   └── SalePulse_AI_Report.xlsx
├── run_pipeline.py           # CLI training runner
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost statsmodels prophet fastapi uvicorn openpyxl
```

### 2. Run the Training Pipeline
```bash
cd salepulse_ai
python run_pipeline.py --data data/sales.xlsx --output outputs/
```

### 3. Start the API Server
```bash
SALEPULSE_DATA=data/sales.xlsx uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 API Reference

### GET /health
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "data_loaded": true,
  "states_available": 43
}
```

### GET /forecast?state=California&weeks=8
```json
{
  "state": "California",
  "model_used": "HoltWinters",
  "horizon_weeks": 8,
  "generated_at": "2024-01-15T10:30:00Z",
  "forecast": [
    { "date": "2023-12-10", "forecast_value": 834009000.0, "forecast_value_M": 834.009 },
    { "date": "2023-12-17", "forecast_value": 844454000.0, "forecast_value_M": 844.454 }
  ],
  "model_notes": "Exponential smoothing with additive trend + seasonality"
}
```

### GET /compare-models?state=Texas
```json
{
  "state": "Texas",
  "winner": "HoltWinters",
  "winner_rationale": "HoltWinters achieved the lowest composite score...",
  "comparison": [
    { "model": "HoltWinters", "rmse": 269361000, "mape_pct": 17.69, "is_best": true },
    { "model": "XGBoost",     "rmse": 310000000, "mape_pct": 19.2,  "is_best": false },
    { "model": "SARIMA",      "rmse": 345000000, "mape_pct": 21.1,  "is_best": false },
    { "model": "Prophet",     "rmse": 290000000, "mape_pct": 18.5,  "is_best": false }
  ]
}
```

### GET /insights?top_n=10
Returns top-N states by volatility with CoV, trend slope, seasonality index, and anomaly count.

### GET /feature-importance?state=New York
Returns XGBoost feature importances — explains which lag/time features drive each state's pattern.

---

## 🤖 Model Selection Logic

| Model | Best For | Wins When |
|-------|----------|-----------|
| **Holt-Winters** | Stable, recurring patterns | Low CoV, clear seasonal rhythm |
| **SARIMA** | Autocorrelated series | Strong AR structure, medium volatility |
| **Prophet** | Holiday-sensitive states | Known event effects, changepoints |
| **XGBoost** | Complex, volatile states | High CoV, non-linear feature interactions |

Auto-selection uses a **composite score**:
```
score = 0.6 × (RMSE / max_RMSE) + 0.4 × (MAPE / max_MAPE)
```
Lower score = winner.

---

## 📊 Feature Engineering (24 Features)

| Category | Features |
|----------|----------|
| **Lag** | t-1, t-2, t-4, t-8, t-13, t-26, t-52 weeks |
| **Rolling Mean** | 4, 8, 13, 26-week windows |
| **Rolling Std** | 4, 8-week windows |
| **Rolling Max** | 4, 8, 26-week windows |
| **Momentum** | Current / 8-wk mean, Current / 26-wk mean |
| **Time** | Week-of-year, month, quarter, year index |
| **Cyclical** | sin/cos of week, sin/cos of month |
| **Flags** | is_holiday, is_q4 |

---

## 🎯 Key Business Insights

1. **Nebraska, Mississippi, West Virginia** → Highest volatility (CoV > 0.40). Deploy XGBoost + safety stock buffers.
2. **California, Texas** → Highest revenue states with positive trend. Priority markets for Q4 inventory positioning.
3. **Q4 Surge** → Holiday uplift of ~40% observed across large states. Pre-position by Week 40 (early October).
4. **COVID Anomalies** → March–May 2020 weeks are outliers. Consider training on 2021+ data only.
5. **New England** → Summer tourism spike. Prophet with seasonal regressors outperforms.

---

## 🚀 Production Deployment

For production use, the following upgrades are recommended:

- **Cache layer**: Replace in-memory dict with Redis for multi-instance API scaling
- **Model store**: Save trained models to S3/GCS using `joblib` or `cloudpickle`
- **Monitoring**: Add Prometheus metrics + Grafana dashboards
- **Retraining**: Schedule weekly `run_pipeline.py` via Airflow or AWS EventBridge
- **Auth**: Add JWT auth to API endpoints for multi-tenant access control

---

*Built by SalePulse AI — v1.0.0*
