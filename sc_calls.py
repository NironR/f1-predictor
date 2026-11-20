"""Pit or stay out under a safety car, learned from what teams did.

Within one safety car ~80% of cars make the same call, so the decision is driven by race-level context
more than each car's tyres. One row per (safety car, top-10 car), with race-level features shared by
every car at that SC plus the car's own state. Walk-forward by season against simple baselines.

Usage:  python sc_calls.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import TEST_SEASONS

CAR = ["stops_done", "tyre_age", "position"]
RACE = ["laps_left", "progress", "field_unstopped", "field_tyre_age", "pit_loss", "season"]


def sc_decisions(t, stops):
    """Every safety car (track status 4) in dry races, and what each top-10 car did."""
    rows = []
    green_loss = stops[~stops.under_sc].groupby("circuit").loss.median()
    for (s, r), g in t.groupby(["season", "round"]):
        if g.compound.isin(["INTERMEDIATE", "WET"]).any():
            continue
        n = int(g.lap.max())
        sc = g.groupby("lap").track_status.agg(lambda x: x.astype("Int64").astype(str).str.contains("4").mean() > 0.5)
        onsets = [l for l in sc.index if sc[l] and not sc.get(l - 1, False) and 3 <= l <= n - 3]
        for L in onsets:
            before = g[g.lap == L - 1].dropna(subset=["position"]).nsmallest(10, "position")
            if len(before) < 6:
                continue
            stints = g[g.lap <= L - 1].groupby("code").stint.max()
            state = pd.DataFrame({"code": before.code, "position": before.position,
                                  "tyre_age": before.tyre_life.fillna(before.tyre_life.median()),
                                  "stops_done": before.code.map(stints).fillna(1) - 1})
            pits = g[g.pit_in.notna() & g.lap.between(L, L + 2)].code
            state["pitted"] = state.code.isin(pits)
            state = state.assign(season=s, round=r, onset=L, circuit=g.circuit.iat[0], laps_left=n - L,
                                 progress=L / n, field_unstopped=(state.stops_done == 0).mean(),
                                 field_tyre_age=state.tyre_age.mean(),
                                 pit_loss=green_loss.get(g.circuit.iat[0], green_loss.median()))
            rows.append(state)
    return pd.concat(rows, ignore_index=True)


def walk_forward(d):
    out = []
    for s in TEST_SEASONS:
        train, test = d[d.season < s], d[d.season == s].copy()
        for name, cols in (("car_only", CAR), ("sc_level", CAR + RACE)):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            test[name] = m.fit(train[cols], train.pitted).predict_proba(test[cols])[:, 1] > 0.5
        test["always_pit"] = True
        test["rule_of_thumb"] = (test.stops_done == 0) | (test.tyre_age >= 15)
        out.append(test)
    return pd.concat(out)


if __name__ == "__main__":
    from sim import load
    data = load()
    d = sc_decisions(data["tyre_laps"], data["stops"])
    d.to_csv("data/sc_decisions.csv", index=False)
    n_sc = d.groupby(["season", "round", "onset"]).ngroups
    agree = d.groupby(["season", "round", "onset"]).pitted.agg(lambda x: max(x.mean(), 1 - x.mean())).mean()
    print(f"{n_sc} safety cars, {len(d)} top-10 car decisions, {d.pitted.mean():.0%} pitted; "
          f"{agree:.0%} of cars make the majority call within one SC")
    p = walk_forward(d)
    print(f"Walk-forward 2021-2026 ({len(p)} decisions), share matching what the team did:")
    for k in ("always_pit", "rule_of_thumb", "car_only", "sc_level"):
        print(f"  {k:14s} {np.mean(p[k] == p.pitted):.0%}")
    print("By season:", p.groupby("season").apply(
        lambda g: f"{np.mean(g.sc_level == g.pitted):.0%} vs always-pit {g.pitted.mean():.0%}",
        include_groups=False).to_dict())
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(d[CAR + RACE], d.pitted)
    print("What drives the call (standardised coefficients, + = more likely to pit):")
    print(pd.Series(m[-1].coef_[0], CAR + RACE).sort_values(key=abs, ascending=False).round(2).to_string())
