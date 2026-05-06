"""
SalePulse AI — Training Pipeline Runner
========================================
Trains models for all states and saves results + Excel report.

Usage:
    python run_pipeline.py --data data/sales.xlsx --output outputs/
"""

import sys, os, warnings, argparse
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from core.data_pipeline import build_features, compute_state_analytics
from models.model_zoo   import train_all_models, select_best_model


def run(data_path: str, output_dir: str, sample_states=None):
    print("\n" + "="*60)
    print("  🔮 SalePulse AI  |  Retail Intelligence Engine v1.0")
    print("="*60 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    print("▶ Building feature matrix…")
    df = build_features(data_path)
    print(f"  ✓ {len(df)} rows | {df['State'].nunique()} states | {df['ds'].min().date()} → {df['ds'].max().date()}")

    print("\n▶ Computing state analytics…")
    analytics = compute_state_analytics(df)
    print(analytics[["State","CoV","Volatility_Label","Trend_Slope_$M_wk","Anomaly_Count"]].head(10).to_string(index=False))

    states = sample_states or sorted(df["State"].unique().tolist())
    print(f"\n▶ Training models for {len(states)} states…\n")

    all_metrics = []
    all_forecasts = []

    for i, state in enumerate(states, 1):
        print(f"  [{i:02d}/{len(states)}] {state}…", end=" ", flush=True)
        try:
            results = train_all_models(state, df, n_forecast=8)
            best    = select_best_model(results)
            br      = results[best]
            print(f"best={best} | MAPE={br.mape:.2f}% | RMSE=${br.rmse/1e6:.1f}M")

            # Collect metrics
            for name, res in results.items():
                all_metrics.append({
                    "State": state, "Model": name,
                    "RMSE_$M": round(res.rmse/1e6, 3),
                    "MAE_$M":  round(res.mae/1e6, 3),
                    "MAPE_%":  round(res.mape, 3),
                    "Is_Best": (name == best),
                })

            # Collect 8-week forecast
            for d, v in zip(br.forecast_dates, br.forecast):
                all_forecasts.append({
                    "State": state, "Model": best,
                    "Forecast_Date": d.date(),
                    "Forecast_Sales_$": round(v, 2),
                    "Forecast_Sales_$M": round(v/1e6, 3),
                })
        except Exception as e:
            print(f"ERROR: {e}")

    metrics_df   = pd.DataFrame(all_metrics)
    forecasts_df = pd.DataFrame(all_forecasts)
    analytics_df = analytics.copy()

    # ── Save Excel report ───────────────────────────────────────────────────
    out_xlsx = os.path.join(output_dir, "SalePulse_AI_Report.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        forecasts_df.to_excel(writer, sheet_name="8-Week Forecasts", index=False)
        metrics_df.to_excel(writer,   sheet_name="Model Comparison",  index=False)
        analytics_df.to_excel(writer, sheet_name="State Analytics",   index=False)

    print(f"\n✅  Report saved → {out_xlsx}")
    return metrics_df, forecasts_df, analytics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/sales.xlsx")
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--states", nargs="*", default=None, help="Subset of states")
    args = parser.parse_args()
    run(args.data, args.output, args.states)
