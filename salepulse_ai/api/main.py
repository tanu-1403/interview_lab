"""
SalePulse AI — REST API
========================
FastAPI service exposing forecast endpoints.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health
    GET  /states
    GET  /forecast?state=California&weeks=8
    GET  /compare-models?state=California
    GET  /insights
    GET  /feature-importance?state=California
"""

from __future__ import annotations
import os, json, time
from datetime import datetime
from typing import Optional

# ── FastAPI ─────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# ── SalePulse internals ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.data_pipeline import build_features, compute_state_analytics
from models.model_zoo   import train_all_models, select_best_model

# ────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("SALEPULSE_DATA", "data/sales.xlsx")

# In-memory cache (production: Redis / Memcached)
_cache: dict = {}


def _get_features():
    if "_features" not in _cache:
        _cache["_features"] = build_features(DATA_PATH)
    return _cache["_features"]


def _get_analytics():
    if "_analytics" not in _cache:
        _cache["_analytics"] = compute_state_analytics(_get_features())
    return _cache["_analytics"]


def _get_forecast(state: str, weeks: int = 8):
    key = f"fc_{state}_{weeks}"
    if key not in _cache:
        df = _get_features()
        if state not in df["State"].unique():
            raise ValueError(f"State '{state}' not found in dataset.")
        results = train_all_models(state, df, n_forecast=weeks)
        best    = select_best_model(results)
        _cache[key] = {"results": results, "best": best}
    return _cache[key]


# ────────────────────────────────────────────────────────────────────────────
# Pydantic response schemas
# ────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    data_loaded: bool
    states_available: int


class ForecastPoint(BaseModel):
    date: str
    forecast_value: float
    forecast_value_M: float     # in millions


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    horizon_weeks: int
    generated_at: str
    forecast: list[ForecastPoint]
    model_notes: str


class ModelMetrics(BaseModel):
    model: str
    rmse: float
    mae: float
    mape_pct: float
    is_best: bool
    notes: str


class CompareResponse(BaseModel):
    state: str
    generated_at: str
    comparison: list[ModelMetrics]
    winner: str
    winner_rationale: str


class InsightRow(BaseModel):
    state: str
    volatility: str
    cov: float
    trend_slope_M_per_week: float
    seasonality_index: float
    anomaly_count: int
    mean_weekly_M: float


class FeatureImportanceResponse(BaseModel):
    state: str
    model: str
    top_features: list[dict]


# ────────────────────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────────────────────

def create_app() -> "FastAPI":
    app = FastAPI(
        title="SalePulse AI",
        description=(
            "🔮 Production-grade retail sales forecasting engine.\n\n"
            "Forecast next N weeks of beverage sales per US state using an ensemble "
            "of SARIMA, Facebook Prophet, XGBoost, and Holt-Winters models with "
            "automatic best-model selection."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # ── /health ──────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health():
        try:
            df = _get_features()
            loaded = True
            n_states = df["State"].nunique()
        except Exception:
            loaded, n_states = False, 0
        return HealthResponse(
            status="ok", version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            data_loaded=loaded, states_available=n_states,
        )

    # ── /states ───────────────────────────────────────────────────────────────
    @app.get("/states", tags=["Data"])
    def list_states():
        df = _get_features()
        return {"states": sorted(df["State"].unique().tolist())}

    # ── /forecast ─────────────────────────────────────────────────────────────
    @app.get("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
    def forecast(
        state: str = Query(..., example="California", description="US state name"),
        weeks: int = Query(8, ge=1, le=26, description="Forecast horizon in weeks"),
    ):
        """
        Returns weekly sales forecast for the given state.
        Automatically selects the best-performing model (lowest composite RMSE+MAPE score).

        **Example:**
        ```
        GET /forecast?state=California&weeks=8
        ```
        """
        t0 = time.time()
        try:
            fc_data = _get_forecast(state, weeks)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        best_name   = fc_data["best"]
        best_result = fc_data["results"][best_name]

        points = [
            ForecastPoint(
                date=str(d.date()),
                forecast_value=round(v, 2),
                forecast_value_M=round(v / 1e6, 3),
            )
            for d, v in zip(best_result.forecast_dates, best_result.forecast)
        ]
        return ForecastResponse(
            state=state, model_used=best_name, horizon_weeks=weeks,
            generated_at=datetime.utcnow().isoformat() + "Z",
            forecast=points, model_notes=best_result.notes,
        )

    # ── /compare-models ────────────────────────────────────────────────────────
    @app.get("/compare-models", response_model=CompareResponse, tags=["Forecasting"])
    def compare_models(
        state: str = Query(..., example="Texas", description="US state name"),
    ):
        """
        Returns a comparison table of all trained models for a given state,
        including RMSE, MAE, MAPE, and the automatic winner.

        **Example:**
        ```
        GET /compare-models?state=Texas
        ```
        """
        try:
            fc_data = _get_forecast(state)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        best = fc_data["best"]
        rows = []
        for name, res in fc_data["results"].items():
            rows.append(ModelMetrics(
                model=name,
                rmse=round(res.rmse, 2),
                mae=round(res.mae, 2),
                mape_pct=round(res.mape, 3),
                is_best=(name == best),
                notes=res.notes,
            ))
        rows.sort(key=lambda r: r.mape_pct)

        rationale = (
            f"{best} achieved the lowest composite score across RMSE and MAPE "
            f"on the held-out {8}-week validation window."
        )
        return CompareResponse(
            state=state,
            generated_at=datetime.utcnow().isoformat() + "Z",
            comparison=rows, winner=best, winner_rationale=rationale,
        )

    # ── /insights ─────────────────────────────────────────────────────────────
    @app.get("/insights", tags=["Analytics"])
    def insights(top_n: int = Query(10, ge=3, le=43)):
        """
        Returns analytical insights per state:
        volatility classification, trend slope, seasonality index, anomaly count.
        """
        analytics = _get_analytics()
        rows = analytics.head(top_n).to_dict(orient="records")
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "description": (
                "States ranked by Coefficient of Variation (CoV). "
                "High CoV → unpredictable demand requiring safety-stock buffers."
            ),
            "insights": rows,
        }

    # ── /feature-importance ────────────────────────────────────────────────────
    @app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Analytics"])
    def feature_importance(state: str = Query(..., example="Florida")):
        """
        Returns XGBoost feature importances for a given state.
        Explains WHICH signals drive that state's sales pattern.
        """
        try:
            fc_data = _get_forecast(state)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        fi = fc_data["results"]["XGBoost"].feature_importance
        top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12]
        return FeatureImportanceResponse(
            state=state, model="XGBoost",
            top_features=[{"feature": k, "importance": round(v, 5)} for k, v in top],
        )

    return app


if HAS_FASTAPI:
    app = create_app()
