"""DNF (per driver) and safety-car (per race) probability models.

Walk-forward tested like model.py, against simple baselines: the historical average rate,
and the obvious heuristic (team's recent DNF rate / this track's SC history).
"sc_or_red" = a full safety car or red flag during the race (VSCs excluded: 2026 has one in
74% of races, so including them makes the target near-certain).
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import TEST_SEASONS, features, roll

NOT_RACED = ["Did not start", "Disqualified", "Withdrew", "Did not qualify", "Excluded"]
DNF_FEATURES = ["grid", "quali_gap", "out_of_position", "team_form", "team_dnf_rate",
                "driver_dnf_rate", "track_dnf_rate", "rain"]
SC_FEATURES = ["track_sc_rate", "rain_mm", "n_out_of_position", "max_out_of_position",
               "field_spread", "track_order_corr", "field_dnf_rate", "air_temp"]


def track_history(races, col, m=3):
    """Rate of `col` at this circuit in earlier races, shrunk toward the overall earlier rate
    by m pseudo-races, so a track seen once doesn't get 0% or 100%."""
    prior = races[col].shift(1).expanding().mean()
    g = races.groupby("circuit")[col]
    return (g.transform(lambda x: x.shift(1).cumsum()).fillna(0) + m * prior) / (g.cumcount() + m)


def logit(C):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=C, max_iter=1000))


def walk_forward(df, feats, target, models, baselines):
    preds = []
    for s in TEST_SEASONS:
        train, test = df[df.season < s], df[df.season == s].copy()
        test["avg_rate"] = train[target].mean()
        for b in baselines:
            test[b] = test[b].fillna(test.avg_rate)
        for name, make in models.items():
            test[name] = make().fit(train[feats], train[target]).predict_proba(test[feats])[:, 1]
        preds.append(test)
    p = pd.concat(preds)
    cols = ["avg_rate", *baselines, *models]
    t = pd.DataFrame({c: {"brier": brier_score_loss(p[target], p[c]),
                          "log_loss": log_loss(p[target], p[c].clip(0.01, 0.99)),
                          "auc": roc_auc_score(p[target], p[c])} for c in cols}).T
    t["brier_skill_vs_avg"] = 1 - t.brier / t.brier["avg_rate"]
    print(f"\n=== {target}: {len(p)} test rows, actual rate {p[target].mean():.1%} ===")
    print(t.round(3).to_string())
    return p


def coefficients(df, feats, target, C):
    """Standardised logistic coefficients on all data: + raises risk, - lowers it."""
    m = logit(C)().fit(df[feats], df[target])
    print(pd.Series(m[-1].coef_[0], feats).sort_values(key=abs, ascending=False).round(3).to_string())


def dnf_report(df):
    d = df[~df.status.isin(NOT_RACED)].copy()
    d["driver_dnf_rate"] = roll(d, "driver", "dnf", 10)
    tr = d.groupby(["race_id", "circuit"], as_index=False).dnf.mean().sort_values("race_id")
    tr["track_dnf_rate"] = track_history(tr, "dnf")
    d = d.merge(tr[["race_id", "track_dnf_rate"]], on="race_id")
    p = walk_forward(d, DNF_FEATURES, "dnf", baselines=["team_dnf_rate"], models={
        "logistic": logit(0.1),
        "boosted": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_depth=3,
                                                          min_samples_leaf=50, random_state=0)})
    print("\nCalibration (logistic, by predicted-risk quintile):")
    print(p.groupby(pd.qcut(p.logistic, 5), observed=True).agg(predicted=("logistic", "mean"),
                                                               actual=("dnf", "mean")).round(3).to_string())
    print("\nWhat drives DNF risk:")
    coefficients(d, DNF_FEATURES, "dnf", 0.1)
    return p


def sc_report(df):
    of1 = df[["sc", "red"]].max(axis=1)
    inferred = (df.slow_laps > 0).astype(float).where(df.slow_laps.notna())
    df["sc_or_red"] = of1.where(df.sc.notna(), inferred)

    both = df[df.sc.notna() & df.slow_laps.notna()].drop_duplicates("race_id")
    print(f"Lap-time inference vs OpenF1 ({len(both)} overlapping races): "
          f"{((both.slow_laps > 0) == of1[both.index].astype(bool)).mean():.0%} agree")

    races = (df.dropna(subset=["sc_or_red"])
             .groupby(["season", "round", "race_id", "race", "circuit"], as_index=False)
             .agg(sc_or_red=("sc_or_red", "first"), rain_mm=("rain_mm", "first"),
                  air_temp=("air_temp", "first"), track_order_corr=("track_order_corr", "first"),
                  n_out_of_position=("out_of_position", lambda x: (x >= 4).sum()),
                  max_out_of_position=("out_of_position", "max"),
                  field_spread=("quali_gap", "std"), field_dnf_rate=("team_dnf_rate", "mean"))
             .sort_values("race_id"))
    races["sc_or_red"] = races.sc_or_red.astype(int)
    races["track_sc_rate"] = track_history(races, "sc_or_red")
    p = walk_forward(races, SC_FEATURES, "sc_or_red", baselines=["track_sc_rate"],
                     models={"logistic": logit(0.1)})
    print("\nWhat drives SC / red flag risk:")
    coefficients(races, SC_FEATURES, "sc_or_red", 0.1)
    return p


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = features(pd.read_csv("data/races.csv"))
    dnf = dnf_report(df)
    sc = sc_report(df)
    dnf[["season", "round", "race", "driver", "team", "grid", "logistic", "dnf", "status"]] \
        .rename(columns={"logistic": "dnf_prob"}).to_csv("data/dnf_predictions.csv", index=False, float_format="%.6g")
    sc[["season", "round", "race", "logistic", "track_sc_rate", "sc_or_red"]] \
        .rename(columns={"logistic": "sc_or_red_prob"}).to_csv("data/sc_predictions.csv", index=False, float_format="%.6g")
