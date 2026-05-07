"""
run_pipeline.py
================
Master pipeline runner — executes all steps in sequence.
Run this once to generate all data, models, and outputs.

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --steps 1,2,3      # Specific steps
    python run_pipeline.py --skip-models      # Skip ML (faster demo)
"""

import os
import sys
import time
import argparse
import traceback
import geopandas as gpd

# ── Pipeline Steps ──────────────────────────────────────────────────────────────

STEPS = {
    1: ("Census & spatial data generation",    "00_data_download.census_download"),
    2: ("LST preprocessing",                   "01_preprocessing.lst_retrieval"),
    3: ("NDVI preprocessing",                  "01_preprocessing.ndvi_processing"),
    4: ("Thermal Injustice Index (TII)",        "02_feature_engineering.tii_calculator"),
    5: ("Thermal zone clustering",             "03_clustering.thermal_zones"),
    6: ("XGBoost + SHAP analysis",             "04_ml_models.xgboost_vulnerability"),
    7: ("LSTM temporal forecasting",           "04_ml_models.lstm_temporal"),
    8: ("2035 scenario simulation",            "05_scenario_sim.scenario_engine"),
    9: ("Visualization generation",            "06_visualizations.map_generator"),
}

STEP_COLORS = {
    'SUCCESS': '\033[92m✅',
    'FAILED':  '\033[91m❌',
    'SKIP':    '\033[93m⏭️',
    'RUNNING': '\033[94m🔄',
    'RESET':   '\033[0m'
}

def print_header():
    print("\n" + "═"*62)
    print("   🌡️  URBAN HEAT INEQUALITY ATLAS — PIPELINE RUNNER")
    print("   Surat, India · Geospatial AI Case Study")
    print("═"*62 + "\n")

def print_step(n, name, status, elapsed=None):
    icon = STEP_COLORS.get(status, '')
    reset = STEP_COLORS['RESET']
    time_str = f"  [{elapsed:.1f}s]" if elapsed else ""
    print(f"  {icon} Step {n}: {name}{time_str}{reset}")

def run_step(module_path, step_num, step_name):
    """Dynamically import and run a pipeline step."""
    start = time.time()
    print_step(step_num, step_name, 'RUNNING')

    try:
        # Dynamic import
        parts = module_path.split('.')
        if len(parts) == 2:
            sys.path.insert(0, os.path.dirname(__file__))
            folder, module = parts
            mod_path = os.path.join(os.path.dirname(__file__), folder, f"{module}.py")
            
            import importlib.util
            spec = importlib.util.spec_from_file_location(module, mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            if hasattr(mod, 'main'):
                mod.main()
        
        elapsed = time.time() - start
        print_step(step_num, step_name, 'SUCCESS', elapsed)
        return True

    except Exception as e:
        elapsed = time.time() - start
        print_step(step_num, step_name, 'FAILED', elapsed)
        print(f"    Error: {e}")
        if '--verbose' in sys.argv:
            traceback.print_exc()
        return False

def run_full_pipeline(steps=None, skip_models=False):
    """
    Run the complete pipeline.
    Returns final GeoDataFrame with all features.
    """
    print_header()

    steps_to_run = steps or list(STEPS.keys())

    if skip_models:
        steps_to_run = [s for s in steps_to_run if s not in [6, 7]]
        print("  ℹ️  Skipping ML models (--skip-models flag)\n")

    results = {}
    total_start = time.time()

    print(f"  Running {len(steps_to_run)} pipeline steps\n")
    print("  " + "─"*55)

    for step_num in steps_to_run:
        if step_num not in STEPS:
            continue
        step_name, module_path = STEPS[step_num]
        success = run_step(module_path, step_num, step_name)
        results[step_num] = success

        if not success and step_num in [1, 2, 3, 4]:
            print(f"\n  ⚠️  Critical step {step_num} failed — aborting pipeline")
            break

    total_elapsed = time.time() - total_start

    # Summary
    n_success = sum(results.values())
    n_failed  = len(results) - n_success

    print("\n  " + "─"*55)
    print(f"\n  Pipeline complete in {total_elapsed:.1f}s")
    print(f"  ✅ {n_success} steps succeeded")
    if n_failed:
        print(f"  ❌ {n_failed} steps failed")

    # Load and return final GDF
    feat_path = os.path.join(os.path.dirname(__file__),
                             'data/processed/features_final.geojson')
    if os.path.exists(feat_path):
        gdf = gpd.read_file(feat_path)
        print(f"\n  📍 Final dataset: {len(gdf)} census blocks loaded")
        print_outputs_summary()
        return gdf
    else:
        print("\n  ⚠️  features_final.geojson not found — check step 4")
        return None

def print_outputs_summary():
    """Print list of generated output files."""
    out_dir = os.path.join(os.path.dirname(__file__), 'data/outputs')
    if not os.path.exists(out_dir):
        return

    files = os.listdir(out_dir)
    if not files:
        return

    print("\n  📁 Generated outputs:")
    file_icons = {
        '.html': '🗺️ ',
        '.png':  '🖼️ ',
        '.gif':  '🎞️ ',
        '.json': '📋',
        '.csv':  '📊',
    }

    for fname in sorted(files):
        if fname.startswith('_'):
            continue
        ext = os.path.splitext(fname)[1]
        icon = file_icons.get(ext, '📄')
        fpath = os.path.join(out_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"     {icon} {fname:<45} {size_kb:6.1f} KB")

# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Urban Heat Inequality Atlas — Pipeline Runner'
    )
    parser.add_argument(
        '--steps', type=str, default=None,
        help='Comma-separated step numbers to run (e.g., "1,2,3")'
    )
    parser.add_argument(
        '--skip-models', action='store_true',
        help='Skip ML model steps (faster for demo/testing)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show full tracebacks on errors'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all pipeline steps and exit'
    )

    args = parser.parse_args()

    if args.list:
        print_header()
        print("  Pipeline steps:")
        for n, (name, _) in STEPS.items():
            print(f"    {n}. {name}")
        sys.exit(0)

    steps = None
    if args.steps:
        try:
            steps = [int(s.strip()) for s in args.steps.split(',')]
        except ValueError:
            print("❌ Invalid --steps format. Use e.g. --steps 1,2,3")
            sys.exit(1)

    run_full_pipeline(steps=steps, skip_models=args.skip_models)
