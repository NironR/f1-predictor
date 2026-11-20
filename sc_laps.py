"""Lap-level safety car / red flag hazard: P(an SC or red flag starts on this lap | race so far).

One row per green-flag lap, with only what's known at the start of that lap. Walk-forward
tested like the other models, against a constant per-lap rate and a "lap 1 vs every other lap"
two-rate baseline. Labels: OpenF1 race control (2023+), lap-time inference before that.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_auc_score

from model import features, save_metrics
from risk import NOT_RACED, coefficients, logit, track_history, walk_forward

FEATURES = ["lap1", "lap_frac", "restart", "recent_retirements", "cars_running", "rain_mm",
            "track_lap_rate", "lap1_out_of_position"]


def lap_table(df, periods):
    rows = []
    for (season, rnd), g in df.groupby(["season", "round"]):
        src = "openf1" if g.sc.notna().any() else "laps" if g.slow_laps.notna().any() else None
        if src is None:
            continue
        p = periods[(periods.season == season) & (periods["round"] == rnd) & (periods.source == src)]
        onsets = set(p.onset)
        covered = {lap for o, e in zip(p.onset, p.end) for lap in range(o + 1, e + 1)}
        raced = g[~g.status.isin(NOT_RACED)]
        retired = raced[raced.dnf == 1].laps  # laps completed before retiring
        n_oop = int((g.out_of_position >= 4).sum())
        last_end = 0
        for lap in range(1, g.laps.max() + 1):
            if lap in covered:
                last_end = lap
                continue
            rows.append({"season": season, "round": rnd, "race_id": g.race_id.iat[0],
                         "race": g.race.iat[0], "circuit": g.circuit.iat[0], "lap": lap,
                         "lap1": int(lap == 1), "lap_frac": lap / g.laps.max(),
                         "restart": int(last_end > 0 and lap - last_end <= 2),
                         # retired during the previous two laps (completed lap-2 or lap-3 laps)
                         "recent_retirements": int(retired.between(lap - 3, lap - 2).sum()),
                         "cars_running": len(raced) - int((retired <= lap - 2).sum()),
                         "rain_mm": g.rain_mm.iat[0], "lap1_out_of_position": n_oop * (lap == 1),
                         "sc_start": int(lap in onsets)})
    laps = pd.DataFrame(rows)
    races = laps.groupby(["race_id", "circuit"], as_index=False).sc_start.mean().sort_values("race_id")
    races["track_lap_rate"] = track_history(races, "sc_start")
    return laps.merge(races[["race_id", "track_lap_rate"]], on="race_id")


def validate(periods, season=2023):
    """How well lap-time inference finds the real (OpenF1) SC / red flag start laps."""
    s = periods[periods.season == season]
    real, inferred = s[s.source == "openf1"], s[s.source == "laps"]
    hit = lambda a, b: a.apply(lambda r: ((b["round"] == r["round"]) & ((b.onset - r.onset).abs() <= 1)).any(),
                               axis=1).mean()
    print(f"{season} check: lap-time inference finds {hit(real, inferred):.0%} of real SC/red starts "
          f"(within 1 lap); {hit(inferred, real):.0%} of inferred starts are real")


def lap1_baseline():
    return make_pipeline(ColumnTransformer([("lap1", "passthrough", ["lap1"])]), LogisticRegression())


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    periods = pd.read_csv("data/sc_periods.csv")
    validate(periods)
    laps = lap_table(features(pd.read_csv("data/races.csv")), periods)
    p = walk_forward(laps, FEATURES, "sc_start", baselines=[], models={
        "lap1_vs_rest": lap1_baseline,
        "logistic": logit(1.0),
        "boosted": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_depth=3,
                                                          min_samples_leaf=100, random_state=0)})

    save_metrics("sc_lap", {"auc": roc_auc_score(p.sc_start, p.logistic),
                            "lap1_baseline_auc": roc_auc_score(p.sc_start, p.lap1_vs_rest)})

    phase = pd.cut(p.lap_frac, [0, 0.33, 0.67, 1.0], labels=["early", "middle", "late"]).astype(str)
    p["phase"] = phase.where(p.lap1 == 0, "lap 1").where(p.restart == 0, "restart")
    print("\nPer-lap SC/red start rate by race phase (logistic):")
    print(p.groupby("phase").agg(laps=("sc_start", "size"), actual=("sc_start", "mean"),
                                 predicted=("logistic", "mean")).round(4).to_string())
    print("\nAfter a retirement in the last 2 laps:")
    print(p.groupby(p.recent_retirements.clip(upper=1)).agg(laps=("sc_start", "size"), actual=("sc_start", "mean"),
                                                            predicted=("logistic", "mean")).round(4).to_string())
    print("\nWhat drives per-lap SC / red flag risk:")
    coefficients(laps, FEATURES, "sc_start", 1.0)
    p[["season", "round", "race", "lap", "logistic", "sc_start"]] \
        .rename(columns={"logistic": "sc_start_prob"}).to_csv("data/sc_lap_predictions.csv", index=False)
