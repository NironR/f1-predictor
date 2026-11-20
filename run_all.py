"""Runs the whole pipeline in order: fetch new data, re-fit every model, rebuild the results page.

Usage:  python run_all.py          # fast steps, ~10 min (the slow backtests keep their last results)
        python run_all.py --full   # also re-run the strategy backtests and replay checks, ~1.5 h
"""
import subprocess
import sys
import time

FAST = [
    ["fetch.py"],        # -> races.csv, sc_periods.csv, tyre_laps.csv
    ["model.py"],        # -> predictions.csv
    ["risk.py"],         # -> dnf_predictions.csv, sc_predictions.csv
    ["sc_laps.py"],      # -> sc_lap_predictions.csv
    ["tyres.py"],        # -> tyre_deg.csv, tyre_shape.csv, team_wear.csv
    ["sc_calls.py"],     # -> sc_decisions.csv
]
SLOW = [
    ["sim.py", "backtest"],       # -> strategy_backtest.csv
    ["race_sim.py", "backtest"],  # -> race_sim_backtest.csv (~40 min)
    ["assist.py", "check"],       # -> assist_window_check.csv, assist_sc_check.csv
]
SITE = [["build_site.py"]]        # -> docs/data.js

if __name__ == "__main__":
    steps = FAST + (SLOW if "--full" in sys.argv else []) + SITE
    for step in steps:
        start = time.time()
        print(f"== {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, *step], check=True)
        print(f"   done in {time.time() - start:.0f}s", flush=True)
