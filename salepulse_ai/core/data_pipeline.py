"""
SalePulse AI — Data Pipeline & Feature Engineering
===================================================
Transforms raw sales records into a rich feature matrix
ready for multi-model forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample each state to a consistent weekly (Sunday) grid.
    Missing weeks → forward-fill to avoid gaps that confuse lag features.
    """
    out = []
    for state, grp in df.groupby("State"):
        grp = grp.set_index("Date")["Total"].resample("W").sum()
        # Drop zero-sum weeks that are just padding artefacts
        grp = grp[grp > 0]
        out.append(pd.DataFrame({"State": state, "ds": grp.index, "y": grp.values}))
    weekly = pd.concat(out, ignore_index=True)
    weekly = weekly.sort_values(["State", "ds"]).reset_index(drop=True)
    return weekly


# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────

US_HOLIDAYS = {
    # (month, week) — approximate weekly buckets for major retail holidays
    (1, 1): "new_year", (11, 4): "thanksgiving",
    (12, 4): "christmas_week", (12, 3): "pre_christmas",
    (7, 1): "july4", (2, 2): "valentines",
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)
    df["month"]        = df["ds"].dt.month
    df["quarter"]      = df["ds"].dt.quarter
    df["year"]         = df["ds"].dt.year
    df["year_idx"]     = df["year"] - df["year"].min()          # linear trend proxy
    # Cyclical encoding — avoids ordinality cliff at Dec→Jan
    df["sin_week"]     = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"]     = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["sin_month"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]    = np.cos(2 * np.pi * df["month"] / 12)
    # Holiday flag
    df["is_holiday"]   = df.apply(
        lambda r: 1 if (r["month"], (r["ds"].day - 1) // 7 + 1) in US_HOLIDAYS else 0,
        axis=1
    )
    # Q4 retail surge flag
    df["is_q4"]        = (df["quarter"] == 4).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 2, 4, 8, 13, 26, 52)) -> pd.DataFrame:
    df = df.copy().sort_values(["State", "ds"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("State")["y"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["State", "ds"])
    grp = df.groupby("State")["y"]
    for window in [4, 8, 13, 26]:
        df[f"roll_mean_{window}"] = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
        df[f"roll_std_{window}"]  = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
        df[f"roll_max_{window}"]  = grp.transform(lambda s: s.shift(1).rolling(window, min_periods=2).max())
    # Momentum: current-vs-rolling-mean ratio
    df["momentum_8"]  = df["y"].shift(1) / (df["roll_mean_8"]  + 1e-9)
    df["momentum_26"] = df["y"].shift(1) / (df["roll_mean_26"] + 1e-9)
    return df


def build_features(raw_path: str) -> pd.DataFrame:
    raw    = load_raw(raw_path)
    weekly = aggregate_weekly(raw)
    weekly = add_time_features(weekly)
    weekly = add_lag_features(weekly)
    weekly = add_rolling_features(weekly)
    return weekly


# ─────────────────────────────────────────────
# 3.  ANALYTICS  (volatility / seasonality)
# ─────────────────────────────────────────────

def compute_state_analytics(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, g in weekly.groupby("State"):
        g = g.sort_values("ds")
        cv          = g["y"].std() / g["y"].mean()          # coefficient of variation
        trend_slope = np.polyfit(np.arange(len(g)), g["y"], 1)[0]
        # Seasonality: variance of monthly means vs overall variance
        monthly_var = g.groupby("month")["y"].mean().var()
        overall_var = g["y"].var()
        seasonality_idx = monthly_var / (overall_var + 1e-9)
        # Anomaly count: points > 2.5 σ from rolling mean
        roll = g["y"].rolling(8, min_periods=4).mean()
        std  = g["y"].rolling(8, min_periods=4).std()
        anomalies = ((g["y"] - roll).abs() > 2.5 * std).sum()
        rows.append({
            "State": state,
            "CoV":                round(cv, 4),
            "Volatility_Label":   "High" if cv > 0.2 else ("Medium" if cv > 0.12 else "Low"),
            "Trend_Slope_$M_wk":  round(trend_slope / 1e6, 3),
            "Seasonality_Index":  round(float(seasonality_idx), 4),
            "Anomaly_Count":      int(anomalies),
            "Mean_Weekly_$M":     round(g["y"].mean() / 1e6, 2),
        })
    return pd.DataFrame(rows).sort_values("CoV", ascending=False).reset_index(drop=True)
