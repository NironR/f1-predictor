"""Predict F1 finishing order from pre-race info; walk-forward test vs the grid baseline.

Each test season is predicted by a model trained only on earlier seasons.
Writes data/predictions.csv (predicted vs actual for every tested race).
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

FEATURES = ["grid", "quali_gap", "teammate_grid_diff", "driver_form", "team_form",
            "out_of_position", "team_dnf_rate", "track_order_corr", "rain", "air_temp"]
METRICS = Path("data/metrics.json")
TEST_SEASONS = range(2021, datetime.now().year + 1)  # every season since 2021, incl. the current one
FINISHED = r"^(?:Finished|Lapped|\+\d+ Laps?)$"


def save_metrics(key, values):
    """Record a script's headline results in data/metrics.json (read by build_site.py)."""
    m = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    m[key] = {k: round(float(v), 4) for k, v in values.items()}
    METRICS.write_text(json.dumps(m, indent=2) + "\n")


def roll(df, by, col, n):
    """Mean of the previous n races (never includes the current one)."""
    return df.groupby(by)[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())


def features(df):
    df = df.sort_values(["season", "round", "position"]).reset_index(drop=True)
    race = ["season", "round"]
    df["race_id"] = df.season * 100 + df["round"]
    df["grid"] = df.grid.replace(0, 21)  # pit-lane start
    df["dnf"] = (~df.status.str.match(FINISHED)).astype(int)

    # Quali gap: % off the fastest time in the last session the driver reached (same conditions).
    sess = np.select([df.q3.notna(), df.q2.notna()], [3, 2], 1)
    t = df.q3.fillna(df.q2).fillna(df.q1)
    best = t.groupby([df.race_id, sess]).transform("min")
    df["quali_gap"] = (t / best - 1).clip(0, 0.07)

    df["teammate_grid_diff"] = df.grid - df.groupby(race + ["team"]).grid.transform("mean")
    df["driver_form"] = roll(df, "driver", "position", 5)

    # Team-level form, pace and reliability, computed per team-race then lagged.
    team = df.groupby(race + ["team"], as_index=False).agg(
        t_pos=("position", "mean"), t_gap=("quali_gap", "mean"), t_dnf=("dnf", "mean"))
    team["team_form"] = roll(team, "team", "t_pos", 5)
    team["team_pace"] = roll(team, "team", "t_gap", 5)
    team["team_dnf_rate"] = roll(team, "team", "t_dnf", 10)
    df = df.merge(team[race + ["team", "team_form", "team_pace", "team_dnf_rate"]], on=race + ["team"])

    # Out of position: grid slot vs where recent car pace says the car belongs.
    df["out_of_position"] = df.grid - df.groupby(race).team_pace.rank()

    # Track overtaking difficulty: grid-vs-finish correlation at this circuit in earlier races.
    rc = (df.groupby(race + ["circuit"]).apply(lambda g: spearmanr(g.grid, g.position)[0],
                                              include_groups=False)
          .rename("corr").reset_index())
    rc["track_order_corr"] = rc.groupby("circuit")["corr"].transform(lambda s: s.shift(1).expanding().mean())
    df = df.merge(rc[race + ["track_order_corr"]], on=race)

    df["rain"] = (df.rain_mm > 0.5).astype(int)
    return df.sort_values(["season", "round", "position"]).reset_index(drop=True)


def check_no_leak(raw):
    """Features for a race must not change when later races are added."""
    cut = raw[(raw.season < 2024) | ((raw.season == 2024) & (raw["round"] <= 10))]
    a = features(cut).query("season == 2024 and round == 10")[FEATURES].reset_index(drop=True)
    b = features(raw).query("season == 2024 and round == 10")[FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def score(df, col):
    """Per-race metrics for a predicted-rank column, averaged."""
    out = []
    for _, g in df.groupby("race_id"):
        top = set(g.nsmallest(3, "position").driver)
        out.append({"spearman": spearmanr(g[col], g.position)[0],
                    "mae": (g[col] - g.position).abs().mean(),
                    "winner": g.loc[g[col].idxmin(), "position"] == 1,
                    "podium": len(top & set(g.nsmallest(3, col).driver)) / 3})
    return pd.DataFrame(out).mean()


if __name__ == "__main__":
    raw = pd.read_csv("data/races.csv")
    check_no_leak(raw)
    df = features(raw)
    df["grid_rank"] = df.groupby("race_id").grid.rank(method="first")

    preds = []
    for s in TEST_SEASONS:
        # Train on finishers only: DNF positions are mostly noise; DNF risk gets its own model later.
        train, test = df[(df.season < s) & (df.dnf == 0)], df[df.season == s].copy()
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3,
                                              min_samples_leaf=40, l2_regularization=1.0,
                                              random_state=0)
        model.fit(train[FEATURES], train.position)
        test["pred_score"] = model.predict(test[FEATURES])
        test["pred_rank"] = test.groupby("race_id").pred_score.rank(method="first")
        preds.append(test)

    p = pd.concat(preds)
    rows = []
    for s, g in [*p.groupby("season"), ("ALL", p)]:
        b, m = score(g, "grid_rank"), score(g, "pred_rank")
        rows.append({"season": s, "races": g.race_id.nunique(),
                     **{f"grid_{k}": v for k, v in b.items()}, **{f"model_{k}": v for k, v in m.items()}})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    allr = rows[-1]
    save_metrics("finish", {"model_spearman": allr["model_spearman"], "grid_spearman": allr["grid_spearman"],
                            "model_mae": allr["model_mae"], "grid_mae": allr["grid_mae"], "races": allr["races"]})

    last = p[p.season == max(TEST_SEASONS)]
    imp = permutation_importance(model, last[FEATURES], last.position, n_repeats=10, random_state=0)
    print("\nFeature importance (latest season, MAE-style drop):")
    print(pd.Series(imp.importances_mean, FEATURES).sort_values(ascending=False).round(3).to_string())

    p["beat_prediction_by"] = p.pred_rank - p.position
    p[["season", "round", "race", "driver", "team", "grid", "pred_rank", "position", "status",
       "beat_prediction_by"]].to_csv("data/predictions.csv", index=False)
    print("\nBiggest over-performers vs prediction (finished, latest season):")
    print(last.assign(beat=last.pred_rank - last.position)[last.dnf == 0]
          .nlargest(5, "beat")[["race", "driver", "grid", "pred_rank", "position"]].to_string(index=False))
