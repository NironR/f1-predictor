"""Tyre degradation: measure it per race, then predict it pre-race from history.

Per race, on clean laps:  time = driver pace + fuel * lap + compound offset + deg[compound] * tyre age
Pit stops reset tyre age but not race lap, which is what separates tyre wear from fuel burn-off.
The lap term is fuel + track evolution combined (both make the car faster as the race goes on).
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from model import TEST_SEASONS, save_metrics
from risk import track_history

DRY = ["SOFT", "MEDIUM", "HARD"]
SOFTEST_FIRST = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"]
MIN_LAPS = 40  # per compound per race, to estimate a slope
# Weak priors (mean, sd). Only bite when a race can't separate fuel from wear, e.g. a wet start
# where the whole field switches to slicks on the same lap (Spa 2025, Monaco 2022).
PRIORS = {"lap": (-0.05, 0.02), "deg_": (0.04, 0.05), "off_": (0.0, 0.5), "stint_": (0.0, 0.5),
          "tw_": (0.0, 0.03)}
# Same team across renames, so its tyre-wear history carries over.
TEAM_LINEAGE = {"Toro Rosso": "racing_bulls", "AlphaTauri": "racing_bulls", "RB": "racing_bulls",
                "Racing Bulls": "racing_bulls", "Sauber": "sauber", "Alfa Romeo Racing": "sauber",
                "Alfa Romeo": "sauber", "Kick Sauber": "sauber", "Audi": "sauber", "Force India": "aston_martin",
                "Racing Point": "aston_martin", "Aston Martin": "aston_martin", "Renault": "alpine",
                "Alpine": "alpine", "Haas F1 Team": "haas", "Red Bull Racing": "red_bull"}
FIT_LAP_SD = 0.65  # typical residual sd (s) of a clean lap, scales prior strength vs data


def relative_compounds(t):
    """2018 used absolute names (hypersoft..hard). Map each race's 3 dry compounds to SOFT/MEDIUM/HARD;
    drop 2018 races where fewer than 3 dry compounds were used (mapping would be ambiguous)."""
    old = t.season == 2018
    rank = {c: i for i, c in enumerate(SOFTEST_FIRST)}
    out = [t[~old]]
    for _, g in t[old].groupby("round"):
        used = sorted(set(g.compound) & set(rank), key=rank.get)
        if len(used) == 3:
            out.append(g.assign(compound=g.compound.map(dict(zip(used, DRY))).fillna(g.compound)))
    return pd.concat(out)


def clean(t):
    t = relative_compounds(t)
    ok = ((t.track_status == 1) & t.pit_in.isna() & t.pit_out.isna() & (t.lap > 1)
          & (t.deleted != True) & (t.accurate == True) & (t.raining != True)
          & t.compound.isin(DRY) & t.tyre_life.notna() & t.time.notna())
    t = t[ok].copy()
    t["team"] = t.team_name.map(lambda n: TEAM_LINEAGE.get(n, str(n).lower().replace(" ", "_")))
    t = t[t.time < 1.07 * t.groupby(["season", "round", "code"]).time.transform("median")]
    n = t.groupby(["season", "round", "compound"]).time.transform("size")
    return t[n >= MIN_LAPS]


def race_fit(g, priors=PRIORS, noise=FIT_LAP_SD, resid_only=False, with_pace=False):
    """One race's OLS. Returns fuel effect, compound offsets vs MEDIUM, and deg slopes (s per lap of age),
    or with resid_only, each lap's residual (actual - fitted)."""
    used = [c for c in DRY if (g.compound == c).any()]
    X = pd.get_dummies(g.code, dtype=float)  # driver pace
    X["lap"] = g.lap.values
    ref = "MEDIUM" if "MEDIUM" in used else used[0]
    for c in used:
        if c != ref:
            X[f"off_{c}"] = (g.compound == c).astype(float).values
        X[f"deg_{c}"] = ((g.compound == c) * g.tyre_life).values
    # Drivers manage tyres early and push in the last stint, which is usually on hards; without
    # this, that push gets credited to the compound. Identified because strategies differ.
    for n in (2, 3):
        X[f"stint_{n}"] = (g.stint.clip(upper=3) == n).astype(float).values
    teams = sorted(g.team.unique())
    for tm in teams:  # team's extra wear vs the field, s per lap of tyre age (all compounds)
        X[f"tw_{tm}"] = ((g.team == tm) * g.tyre_life).values
    # MAP estimate: append one pseudo-observation per prior, weighted by noise / prior sd.
    prior_rows, prior_y = [], []
    for j, col in enumerate(X.columns):
        key = next((k for k in priors if col.startswith(k)), None)  # most specific key first
        if key:
            mean, sd = priors[key]
            prior_rows.append(np.eye(len(X.columns))[j] * noise / sd)
            prior_y.append(mean * noise / sd)
    A, y = np.vstack([X.values, prior_rows]), np.concatenate([g.time.values, prior_y])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = g.time.values - X.values @ beta
    if resid_only:
        return resid
    se =np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * resid.var())
    b, b_se = dict(zip(X.columns, beta)), dict(zip(X.columns, se))
    out = {"fuel_per_lap": b["lap"], "laps": len(g), "track_temp": g.track_temp.mean(),
           "resid_sd": resid.std()}
    # Centre team wear on the (lap-weighted) field average and fold that average into the compounds.
    share = g.team.value_counts(normalize=True)
    field = sum(b[f"tw_{tm}"] * share[tm] for tm in teams)
    for tm in teams:
        out[f"tw_{tm}"] = b[f"tw_{tm}"] - field
        out[f"tw_{tm}_se"] = b_se[f"tw_{tm}"]
    for c in DRY:
        out[f"deg_{c}"] = b.get(f"deg_{c}", np.nan) + field
        out[f"deg_{c}_se"] = b_se.get(f"deg_{c}", np.nan)
        out[f"n_{c}"] = int((g.compound == c).sum())
        vs_medium = ref == "MEDIUM"  # offsets only comparable across races when MEDIUM is the reference
        out[f"off_{c}"] = (0.0 if c == ref else b.get(f"off_{c}", np.nan)) if vs_medium else np.nan
        out[f"off_{c}_se"] = b_se.get(f"off_{c}", np.nan) if vs_medium else np.nan
    if with_pace:  # each driver's pace vs the fastest, s/lap (tyre, fuel and stint effects removed)
        drivers = g.code.unique()
        fastest = min(b[d] for d in drivers)
        out.update({f"pace_{d}": b[d] - fastest for d in drivers})
    return pd.Series(out)


def measure(t):
    return pd.DataFrame([{"season": s, "round": r, "circuit": c, **race_fit(g)}
                         for (s, r, c), g in t.groupby(["season", "round", "circuit"])])


AGE_BINS = [0, 3, 6, 10, 15, 20, 25, 30, 35, 40, 50, 80]


def wear_shape(laps):
    """Pooled non-linear part of wear per compound: mean residual of the linear fit by tyre-age bin.
    Captures warm-up (first laps slower) and the late-stint cliff. Add it to the linear slope.
    One curve per season, fitted only on earlier seasons, so walk-forward tests never see the future
    (the last one, for next season, uses everything).
    ponytail: one curve per compound; long stints are mostly where tyres last (selection bias),
    so the cliff is if anything understated. Per-circuit curves if there's enough data."""
    r = pd.concat(g.assign(resid=race_fit(g, resid_only=True))
                  for _, g in laps.groupby(["season", "round"]))
    r["age_bin"] = pd.cut(r.tyre_life, AGE_BINS)
    out = []
    for season in range(int(r.season.min()) + 1, int(r.season.max()) + 2):
        shape = (r[r.season < season].groupby(["compound", "age_bin"], observed=True).resid
                 .agg(["mean", "size"]).reset_index())
        shape["mean"] = shape["mean"].where(shape["size"] >= 100)  # too few laps: no correction
        shape["age_from"] = shape.age_bin.map(lambda b: b.left + 1).astype(int)
        shape["age_to"] = shape.age_bin.map(lambda b: b.right).astype(int)
        out.append(shape.rename(columns={"mean": "extra_s"}).assign(season=season))
    return pd.concat(out)[["season", "compound", "age_from", "age_to", "extra_s", "size"]]


TARGETS = ["fuel_per_lap", *[f"deg_{c}" for c in DRY], "off_SOFT", "off_HARD"]
PRED_FEATURES = ["global_prior", "circuit_prior", "season_prior", "track_temp"]


def history(deg, q):
    """Pre-race features for target q, from earlier races only."""
    d = deg[deg[q].notna()].sort_values(["season", "round"]).copy()
    d["global_prior"] = d[q].shift(1).expanding().mean()
    d["circuit_prior"] = track_history(d, q, m=2)  # this track's earlier races, shrunk to global
    d["season_prior"] = d.groupby("season")[q].transform(lambda x: x.shift(1).expanding().mean())
    return d  # season_prior is NaN for a season's first race; the imputer fills it


def predict_walk_forward(deg, q):
    d = history(deg, q)
    out = []
    for s in TEST_SEASONS:
        train, test = d[(d.season < s) & d.global_prior.notna()], d[d.season == s].copy()
        m = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
        test["pred"] = m.fit(train[PRED_FEATURES], train[q]).predict(test[PRED_FEATURES])
        out.append(test)
    return pd.concat(out)


def prediction_report(deg):
    rows = []
    for q in TARGETS:
        p = predict_walk_forward(deg, q)
        rows.append({"target": q, "races": len(p),
                     "mae_global_avg": (p.global_prior - p[q]).abs().mean(),
                     "mae_circuit_prior": (p.circuit_prior - p[q]).abs().mean(),
                     "mae_model": (p.pred - p[q]).abs().mean(),
                     "noise_floor": p[f"{q}_se"].median() * 0.8 if f"{q}_se" in p else np.nan})
    print("\nPre-race prediction, walk-forward 2021-2026 (MAE in seconds; noise_floor ~ MAE of a perfect"
          " prediction given measurement error):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def in_race_errors(laps, deg, seasons, trust=1.0, cutoffs=(10, 20, 30, 40)):
    """How fast this race's own laps pin down its degradation, starting from the pre-race
    (track history) estimate as the prior. 'Truth' is the full-race fit.
    trust > 1 treats laps as noisier than they look: early-race laps are correlated (whole field
    managing tyres on the same heavy-fuel first stint), so they carry less information."""
    pre = {q: history(deg, q).set_index(["season", "round"]).circuit_prior for q in TARGETS}
    rows = []
    for (s, r), g in laps[laps.season.isin(seasons)].groupby(["season", "round"]):
        truth = deg[(deg.season == s) & (deg["round"] == r)].iloc[0]
        specific = {"lap": (pre["fuel_per_lap"].get((s, r), PRIORS["lap"][0]), 0.015),
                    **{f"deg_{c}": (pre[f"deg_{c}"].get((s, r), PRIORS["deg_"][0]), 0.045) for c in DRY}}
        priors = {**specific, **{k: v for k, v in PRIORS.items() if k not in specific}}
        for cut in (0, *cutoffs):
            sub = g[g.lap <= cut]
            est = race_fit(sub, priors, noise=FIT_LAP_SD * trust) if len(sub) else {}
            for c in DRY:
                if pd.notna(truth[f"deg_{c}"]):
                    seen = int((sub.compound == c).sum())
                    guess = est[f"deg_{c}"] if seen else specific[f"deg_{c}"][0]
                    rows.append({"after_lap": cut, "compound": c, "seen": seen > 0,
                                 "err": abs(guess - truth[f"deg_{c}"])})
    return pd.DataFrame(rows)


def in_race_report(laps, deg):
    tune = {t: in_race_errors(laps, deg, [2019, 2020], t).err.mean() for t in (1, 2, 4, 8)}
    trust = min(tune, key=tune.get)
    print(f"\nPrior-trust tuned on 2019-2020: {trust}x  (MAE by setting: "
          f"{ {k: round(float(v), 4) for k, v in tune.items()} })")
    e = in_race_errors(laps, deg, TEST_SEASONS, trust)
    print("\nIn-race updating: MAE of deg estimate (s/lap) vs laps of this race seen so far")
    print(e.groupby("after_lap").agg(mae=("err", "mean"), share_compounds_seen=("seen", "mean"))
          .round(4).to_string())
    mae = e.groupby("after_lap").err.mean()
    save_metrics("tyre_in_race", {"mae_pre_race": mae[0], "mae_after_20": mae[20], "mae_after_40": mae[40]})
    print("\n...only compounds already run in this race:")
    print(e[e.seen].groupby(["after_lap", "compound"]).err.mean().unstack().round(4).to_string())


def team_wear(deg, half_life=6, past_season_weight=0.3, m=2):
    """Each team's wear deviation per race (measured) next to a forecast from its earlier races only:
    recency-weighted (half-life in races), earlier seasons down-weighted (cars change), shrunk to 0."""
    cols = [c for c in deg if c.startswith("tw_") and not c.endswith("_se")]
    long = (deg.melt(id_vars=["season", "round", "circuit"], value_vars=cols, var_name="team", value_name="tw")
            .dropna().assign(team=lambda d: d.team.str[3:]).sort_values(["season", "round"]))
    long["idx"] = long.groupby(["season", "round"]).ngroup()
    out = []
    for tm, g in long.groupby("team"):
        for i, row in enumerate(g.itertuples()):
            past = g.iloc[:i]
            w = 0.5 ** ((row.idx - past.idx) / half_life) * np.where(past.season == row.season, 1, past_season_weight)
            out.append({"season": row.season, "round": row.round, "team": tm, "tw": row.tw,
                        "pred": float((w * past.tw).sum() / (w.sum() + m)) if len(past) else 0.0})
    return pd.DataFrame(out)


def team_wear_report(deg):
    tw = team_wear(deg)
    tw.to_csv("data/team_wear.csv", index=False)
    test = tw[tw.season.isin(TEST_SEASONS)]
    print("\nTeam tyre wear vs field (s per lap of tyre age), walk-forward 2021-2026:")
    print(f"  spread of measured team wear: sd {test.tw.std():.4f}; forecast vs measured correlation "
          f"{test[['tw', 'pred']].corr().iat[0, 1]:.2f}")
    print(f"  MAE predicting 'same as field' {test.tw.abs().mean():.4f} -> with team history "
          f"{(test.tw - test.pred).abs().mean():.4f}")
    last = tw[tw.season == tw.season.max()].groupby("team").tw.mean().sort_values()
    print("  this season, kindest to tyres first:", ", ".join(f"{t} {v:+.3f}" for t, v in last.items()))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    laps = clean(pd.read_csv("data/tyre_laps.csv", low_memory=False))
    deg = measure(laps)
    deg.to_csv("data/tyre_deg.csv", index=False)
    print(f"{len(laps)} clean laps, {len(deg)} races measured")
    print("\nPer-race estimates (median, [10th-90th pct]):")
    for c in ["fuel_per_lap", *[f"deg_{c}" for c in DRY], "off_SOFT", "off_HARD", "resid_sd",
              *[f"deg_{c}_se" for c in DRY], "off_SOFT_se", "off_HARD_se"]:
        q = deg[c].quantile([0.1, 0.5, 0.9])
        print(f"  {c:13s} {q[0.5]:+.3f}  [{q[0.1]:+.3f}, {q[0.9]:+.3f}]   ({deg[c].notna().sum()} races)")
    prediction_report(deg)
    in_race_report(laps, deg)
    team_wear_report(deg)
    shapes = wear_shape(laps)
    shapes.to_csv("data/tyre_shape.csv", index=False)
    shape = shapes[shapes.season == shapes.season.max()]
    print("\nNon-linear wear on top of the linear slope, all seasons (s per lap; + = slower; blank = <100 laps):")
    print(shape.assign(age=shape.age_from.astype(str) + "-" + shape.age_to.astype(str))
          .pivot(index=["age_from", "age"], columns="compound", values="extra_s")
          .droplevel(0)[DRY].round(3).to_string())
