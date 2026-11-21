"""Builds docs/data.js for the results page (docs/index.html) from the model outputs in data/.

Usage:  python build_site.py
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from tyres import DRY, history

SEASONS = range(2021, 2027)
TEAM_NAMES = {"racing_bulls": "Racing Bulls", "sauber": "Sauber / Audi", "aston_martin": "Aston Martin",
              "alpine": "Alpine", "haas": "Haas", "red_bull": "Red Bull", "mercedes": "Mercedes",
              "ferrari": "Ferrari", "mclaren": "McLaren", "williams": "Williams", "cadillac": "Cadillac"}


def drivers():
    """driverId -> (3-letter code, surname), from the cached Jolpica results."""
    out = {}
    for f in Path("data/raw").glob("*_results_*.json"):
        for race in json.loads(f.read_text())["MRData"]["RaceTable"]["Races"]:
            for r in race["Results"]:
                d = r["Driver"]
                out[d["driverId"]] = (d.get("code") or d["familyName"][:3].upper(), d["familyName"])
    return out


def rnd(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def main():
    names = drivers()
    races = pd.read_csv("data/races.csv")
    pred = pd.read_csv("data/predictions.csv")
    sc = pd.read_csv("data/sc_lap_predictions.csv")
    periods = pd.read_csv("data/sc_periods.csv")
    deg = pd.read_csv("data/tyre_deg.csv")
    team = pd.read_csv("data/team_wear.csv")
    strat = pd.read_csv("data/race_sim_backtest.csv")
    window = pd.read_csv("data/assist_window_check.csv")
    pre = {c: history(deg, f"deg_{c}").set_index(["season", "round"]).circuit_prior for c in DRY}

    out = []
    for (s, r), g in races[races.season.isin(SEASONS)].groupby(["season", "round"]):
        race = {"season": int(s), "round": int(r), "name": g.race.iat[0], "circuit": g.circuit.iat[0],
                "laps": int(g.laps.max()), "date": str(g.date.iat[0])}
        p = pred[(pred.season == s) & (pred["round"] == r)]
        race["finish"] = [{"code": names.get(x.driver, (x.driver[:3].upper(), x.driver))[0],
                           "name": names.get(x.driver, ("", x.driver.replace("_", " ").title()))[1],
                           "team": x.team, "grid": int(x.grid), "pred": int(x.pred_rank),
                           "actual": int(x.position),
                           "finished": str(x.status) in ("Finished", "Lapped") or str(x.status).startswith("+")}
                          for x in p.sort_values("position").itertuples()]
        h = sc[(sc.season == s) & (sc["round"] == r)].sort_values("lap")
        src = "openf1" if s >= 2023 else "laps"
        per = periods[(periods.season == s) & (periods["round"] == r) & (periods.source == src)]
        race["sc"] = {"lap": h.lap.astype(int).tolist(), "p": [rnd(v, 4) for v in h.sc_start_prob],
                      "starts": per.onset.astype(int).tolist(), "ends": per.end.astype(int).tolist(),
                      "inferred": bool(s < 2023)}
        d = deg[(deg.season == s) & (deg["round"] == r)]
        if len(d):
            d = d.iloc[0]
            race["tyres"] = {c: {"measured": rnd(d[f"deg_{c}"], 4), "se": rnd(d[f"deg_{c}_se"], 4),
                                 "pre": rnd(pre[c].get((s, r)), 4), "laps": int(d[f"n_{c}"])}
                             for c in DRY if pd.notna(d[f"deg_{c}"])}
            tw = team[(team.season == s) & (team["round"] == r)].sort_values("tw")
            race["teams"] = [{"team": TEAM_NAMES.get(x.team, x.team), "tw": rnd(x.tw, 4), "pred": rnd(x.pred, 4)}
                             for x in tw.itertuples()]
        st = strat[(strat.season == s) & (strat["round"] == r)]
        if len(st):
            x = st.iloc[0]
            race["strategy"] = {"actual": int(min(x.actual, 2)), "sim": int(x.choice), "history": int(min(x.hist_stops, 2)),
                                "time_only": int(x.timeonly)}
        w = window[(window.season == s) & (window["round"] == r)]
        race["pit_window"] = [{"code": x.code, "actual": int(x.actual), "assistant": int(x.best),
                               "pre_race": int(x.pre_race)} for x in w.itertuples()]
        out.append(race)

    # Headline results, written by model.py / sc_laps.py / tyres.py / assist.py / sc_calls.py
    # (all walk-forward or held-out; see README).
    summary = json.loads(Path("data/metrics.json").read_text())
    summary["data"] = {"laps": len(pd.read_csv("data/tyre_laps.csv", usecols=["lap"]))}
    Path("docs").mkdir(exist_ok=True)
    Path("docs/data.js").write_text("window.F1 = " + json.dumps({"races": out, "summary": summary},
                                                                  separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote docs/data.js: {len(out)} races, {Path('docs/data.js').stat().st_size // 1024} KB")
    if shutil.which("node"):  # bake the latest race into index.html, so it reads without scripts too
        subprocess.run(["node", "docs/prerender.mjs"], check=True)
    else:
        print("node not found: docs/index.html keeps its last prerendered race")


if __name__ == "__main__":
    main()
