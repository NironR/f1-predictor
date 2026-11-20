"""Race strategy simulator: best pit plan for one car, and Monte Carlo under uncertain tyre wear and
random safety cars.

Time-only, single car: no traffic, undercut or track position (those need a multi-car sim).
Fuel burn-off and safety car lap times are the same for every plan, so they cancel and are left out.

Usage:  python sim.py 2024 21      # strategy report for one race (pre-race information only)
        python sim.py backtest     # recommended stops / first stop lap vs what top-10 cars did
        python sim.py pitcheck     # check the SC pit loss against places really lost under SC
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from model import TEST_SEASONS
from tyres import DRY

GREEN_LOSS_DEFAULT = 22.0      # s, used for a circuit with no stop history
SC_PIT_FACTOR = 13.5 / 21.9    # median gap lost pitting under SC vs green (measured in pit_stops)
PARAM_SD = {"deg": 0.04, "off": 0.4}  # pre-race uncertainty, ~ walk-forward MAE x 1.25 (tyres.py)
MIN_STINT = 5


# ---------- inputs, all from races strictly before the target race ----------

def pit_stops(t):
    """Gap lost per stop vs cars that stayed out: from the lap before the stop to 2 laps after
    (or to the lap after a neutralisation ends, so SC queue compression is counted)."""
    t = t.assign(neutral=t.track_status.astype("Int64").astype(str).str.contains("4|6").fillna(False))
    out = []
    for (s, r, circ), g in t.groupby(["season", "round", "circuit"]):
        T = g.pivot_table(index="lap", columns="code", values="session_time", aggfunc="first")
        neutral = g.groupby("lap").neutral.mean() > 0.5
        pitted = g[g.pit_in.notna()].groupby("code").lap.apply(set)
        for p in g[g.pit_in.notna() & (g.lap > 1)].itertuples():
            L, end = p.lap, p.lap + 2
            under_sc = bool(neutral.get(L, False))
            while under_sc and neutral.get(end - 1, False):
                end += 1
            if L - 1 not in T.index or end not in T.index:
                continue
            stay = [c for c in T.columns
                    if c != p.code and not pitted.get(c, set()) & set(range(L - 1, end + 1))]
            base = (T.loc[end, stay] - T.loc[L - 1, stay]).dropna()
            mine = T.loc[end, p.code] - T.loc[L - 1, p.code]
            if len(base) >= 3 and pd.notna(mine) and -5 < mine - base.median() < 60:
                out.append({"season": s, "round": r, "circuit": circ, "under_sc": under_sc,
                            "loss": mine - base.median()})
    return pd.DataFrame(out)


def overtakes(g):
    """On-track position swaps in one real race: green laps, neither car pitting."""
    P = g.pivot_table(index="lap", columns="code", values="position", aggfunc="first")
    flag = lambda col: g.pivot_table(index="lap", columns="code", values=col,
                                     aggfunc=lambda x: x.notna().any()).reindex_like(P).fillna(False).astype(bool)
    pit = flag("pit_in") | flag("pit_out")
    green = g.groupby("lap").track_status.agg(lambda s: (s == 1).mean() > 0.9)
    n = 0
    for L in P.index[2:]:
        ok = P.loc[L].notna() & P.loc[L - 1].notna() & ~pit.loc[L] & ~pit.loc[L - 1]
        if green.get(L, False):
            a, b = P.loc[L - 1, ok].to_numpy(), P.loc[L, ok].to_numpy()
            n += int(((a[:, None] < a[None, :]) & (b[:, None] > b[None, :])).sum())
    return n


def queue_loss(t):
    """Checks SC_PIT_FACTOR in the terms the multi-car sim uses: for each stop, cars that stayed out and
    went from behind to ahead, vs cars within X s behind just before. Best-fit X per kind of stop."""
    t = t.assign(st=t.track_status.astype("Int64").astype(str))
    rows = []
    for _, g in t.groupby(["season", "round"]):
        T = g.pivot_table(index="lap", columns="code", values="session_time", aggfunc="first")
        sc = g.groupby("lap").st.agg(lambda x: x.str.contains("4").mean() > 0.5)
        pitted = g[g.pit_in.notna()].groupby("code").lap.apply(set)
        for p in g[g.pit_in.notna() & (g.lap > 2)].itertuples():
            L, end = p.lap, p.lap + 2
            while sc.get(L, False) and sc.get(end - 1, False):
                end += 1
            end += sc.get(L, False)  # a lap after the restart
            if L - 1 not in T.index or end not in T.index or pd.isna(T.loc[L - 1, p.code]) or pd.isna(T.loc[end, p.code]):
                continue
            stay = [c for c in T.columns if c != p.code and not pitted.get(c, set()) & set(range(L - 1, end + 1))]
            gap0, gap1 = T.loc[L - 1, stay] - T.loc[L - 1, p.code], T.loc[end, stay] - T.loc[end, p.code]
            ok = gap0.notna() & gap1.notna() & (gap0 > 0)
            rows.append({"sc": bool(sc.get(L, False)), "gaps": gap0[ok].to_numpy(), "lost": int((gap1[ok] < 0).sum())})
    d = pd.DataFrame(rows)
    for under_sc, g in d.groupby("sc"):
        err = {x: np.mean([abs((gaps < x).sum() - lost) for gaps, lost in zip(g.gaps, g.lost)])
               for x in np.arange(2, 40, 0.5)}
        x = min(err, key=err.get)
        print(f"{'SC' if under_sc else 'green':5s} {len(g)} stops: places lost best explained by a {x:.1f}s loss "
              f"(mean error {err[x]:.2f} places)")


def shrunk(values, prior, m):
    """Mean of values pulled toward prior by m pseudo-observations."""
    return (values.sum() + m * prior) / (len(values) + m)


def before(df, season, rnd):
    return df[(df.season < season) | ((df.season == season) & (df["round"] < rnd))]


def race_inputs(season, rnd, data, wear_scale=1.0):
    """Pre-race estimates for one race: tyre params, pit loss, SC hazard, race length.
    wear_scale multiplies measured wear: race laps understate it, because on long stints drivers
    nurse tyres and that lost pace lands in 'driver pace', not 'wear'. Calibrated by tune_wear()."""
    deg, stops, laps_sc, periods, races, shape = (data[k] for k in
                                                  ("deg", "stops", "laps_sc", "periods", "races", "shape"))
    circuit = races[(races.season == season) & (races["round"] == rnd)].circuit.iat[0]
    prior_deg = before(deg, season, rnd)
    params = {}
    for c in DRY:
        for kind in ("deg", "off"):
            q = prior_deg[f"{kind}_{c}"].dropna()
            here = prior_deg.loc[prior_deg.circuit == circuit, f"{kind}_{c}"].dropna()
            # Wear: this track's history helps. Compound pace: per-race estimates are noisy and track
            # history predicts them worse than the overall median does (tyres.py report), so use that.
            params[(kind, c)] = (wear_scale * shrunk(here, q.mean(), 2) if kind == "deg"
                                 else 0.0 if c == "MEDIUM" else q.median())
    green = before(stops, season, rnd).query("not under_sc")
    pit = shrunk(green[green.circuit == circuit].loss, green.loss.median() if len(green) else GREEN_LOSS_DEFAULT, 5)
    hist = before(laps_sc, season, rnd)
    lap1 = hist[hist.lap1 == 1].sc_start.mean()
    later = hist[hist.lap1 == 0]
    per_lap = shrunk(later[later.circuit == circuit].sc_start, later.sc_start.mean(), 300)
    durations = (before(periods, season, rnd).eval("end - onset + 1").clip(1, 10)).to_numpy()
    n_laps = int(races[(races.season == season) & (races["round"] == rnd)].laps.max())
    curve = {c: np.zeros(100) for c in DRY}  # extra s/lap by tyre age (index = age)
    for row in shape[shape.season == season].fillna({"extra_s": 0}).itertuples():  # earlier seasons only
        curve[row.compound][row.age_from:row.age_to + 1] = wear_scale * row.extra_s
    for c in DRY:
        curve[c][81:] = curve[c][80]
    return {"circuit": circuit, "params": params, "pit": pit, "sc_lap1": lap1, "sc_per_lap": per_lap,
            "sc_durations": durations, "n_laps": n_laps, "curve": curve}


# ---------- planning: exact best plan by dynamic programming ----------

def stint_tables(params, curve):
    """cum[c][a] = total time cost (vs a zero-wear MEDIUM) of tyre ages 1..a on compound c."""
    ages = np.arange(100)
    return {c: np.concatenate([[0], np.cumsum(params[("off", c)] + params[("deg", c)] * ages[1:]
                                              + curve[c][1:])]) for c in DRY}


def planner(n_laps, params, curve, pit_loss):
    """best(lap, k, mask, c=None, age=0, exact=False) -> (cost, plan), plan = [(compound, first, last)].
    Plans the race from `lap` with k stops (exactly k if exact, else up to k), compounds used so far
    in bitmask `mask`. c=None: start a fresh stint (compound chosen); else continue c at tyre `age`.
    Rules: 2+ dry compounds over the race, stints of at least MIN_STINT laps."""
    cum = stint_tables(params, curve)
    bit = {c: 1 << i for i, c in enumerate(DRY)}
    cost = lambda c, age, n: cum[c][age + n] - cum[c][age]

    @lru_cache(maxsize=None)
    def best(lap, k, mask, c=None, age=0, exact=False):
        if c is None:
            return min((best(lap, k, mask | bit[c2], c2, 0, exact) for c2 in DRY), key=lambda o: o[0])
        top = (np.inf, [])
        if bin(mask).count("1") >= 2 and (k == 0 or not exact):
            top = (cost(c, age, n_laps - lap + 1), [(c, lap, n_laps)])
        if k > 0:
            for m in range(lap + MIN_STINT, n_laps - MIN_STINT + 2):  # m = first lap of next stint
                rest = best(m, k - 1, mask, None, 0, exact)
                total = cost(c, age, m - lap) + pit_loss + rest[0]
                if total < top[0]:
                    top = (total, [(c, lap, m - 1)] + rest[1])
        return top

    return best, bit


# ---------- Monte Carlo ----------

def sc_timeline(inp, rng):
    """Laps under safety car / red flag in one simulated race."""
    laps, lap = set(), 1
    while lap <= inp["n_laps"]:
        if rng.random() < (inp["sc_lap1"] if lap == 1 else inp["sc_per_lap"]):
            dur = int(rng.choice(inp["sc_durations"]))
            laps |= set(range(lap, lap + dur))
            lap += dur + 1
        else:
            lap += 1
    return laps


def race_time(plan, params, curve, pit_loss, sc_laps):
    cum = stint_tables(params, curve)
    total = sum(cum[c][last - first + 1] for c, first, last in plan)
    for c, first, last in plan[:-1]:  # pit at the end of `last`
        total += pit_loss * (SC_PIT_FACTOR if last in sc_laps else 1.0)
    return total


def adaptive(plan, best, bit, inp, sc_laps, max_stops=3):
    """Follow plan; at each SC onset decide with pre-race params: pit now (cheap stop) or stay out,
    re-planning the rest either way."""
    for onset in sorted(l for l in sc_laps if l - 1 not in sc_laps and l < inp["n_laps"] - 2):
        i = next(j for j, s in enumerate(plan) if s[1] <= onset <= s[2])
        c, first, _ = plan[i]
        if onset - first < 1:  # just pitted
            continue
        mask = sum({bit[s[0]] for s in plan[:i + 1]})
        k = max_stops - i  # stops still allowed
        pit_now = best(onset + 1, k - 1, mask) if k > 0 else (np.inf, [])
        stay = best(onset + 1, k, mask, c, onset - first + 1)
        if inp["pit"] * SC_PIT_FACTOR + pit_now[0] < stay[0]:
            plan = plan[:i] + [(c, first, onset)] + pit_now[1]
        else:
            plan = plan[:i] + [(c, first, stay[1][0][2])] + stay[1][1:]
    return plan


def monte_carlo(inp, n_sims=1000, seed=0):
    rng = np.random.default_rng(seed)
    best, bit = planner(inp["n_laps"], inp["params"], inp["curve"], inp["pit"])
    fixed = {f"{k}-stop": best(1, k, 0, exact=True)[1] for k in (1, 2, 3)}
    overall = best(1, 3, 0)[1]
    times = []
    for _ in range(n_sims):
        true = {key: v + rng.normal(0, PARAM_SD[key[0]]) * (key != ("off", "MEDIUM"))
                for key, v in inp["params"].items()}
        sc = sc_timeline(inp, rng)
        row = {name: race_time(p, true, inp["curve"], inp["pit"], sc) for name, p in fixed.items()}
        row["adaptive"] = race_time(adaptive(overall, best, bit, inp, sc), true, inp["curve"], inp["pit"], sc)
        row["had_sc"] = bool(sc)
        times.append(row)
    return fixed, overall, pd.DataFrame(times)


# ---------- data + reports ----------

def cached(path, build, *sources):
    """Read `path`, rebuilding it first if it's missing or older than any of its source files."""
    p = Path(path)
    if not p.exists() or any(Path(s).stat().st_mtime > p.stat().st_mtime for s in sources):
        build().to_csv(p, index=False)
    return pd.read_csv(p)


def load():
    from sc_laps import lap_table
    from model import features
    t = pd.read_csv("data/tyre_laps.csv", low_memory=False)
    races = pd.read_csv("data/races.csv")
    periods = pd.read_csv("data/sc_periods.csv")
    count = lambda: (t.groupby(["season", "round", "circuit"]).apply(overtakes, include_groups=False)
                     .rename("overtakes").reset_index())
    return {"tyre_laps": t, "races": races, "periods": periods,
            "stops": cached("data/pit_stops.csv", lambda: pit_stops(t), "data/tyre_laps.csv"),
            "overtakes": cached("data/overtakes.csv", count, "data/tyre_laps.csv"),
            "laps_sc": cached("data/sc_lap_table.csv", lambda: lap_table(features(races), periods),
                              "data/races.csv", "data/sc_periods.csv"),
            "deg": pd.read_csv("data/tyre_deg.csv"), "shape": pd.read_csv("data/tyre_shape.csv")}


def fmt(plan):
    stints = " > ".join(f"{c[0]}{last - first + 1}" for c, first, last in plan)
    return f"{stints}  (pit laps {', '.join(str(last) for _, _, last in plan[:-1])})"


def report(season, rnd, data):
    inp = race_inputs(season, rnd, data)
    race = data["races"].query("season == @season and round == @rnd").race.iat[0]
    print(f"{season} {race} ({inp['n_laps']} laps). Pit loss {inp['pit']:.1f}s green, "
          f"{inp['pit'] * SC_PIT_FACTOR:.1f}s under SC. SC risk: {inp['sc_lap1']:.0%} lap 1, "
          f"{inp['sc_per_lap']:.1%}/lap after")
    print("Pre-race tyre estimates (wear s/lap of age | pace vs medium):  " + "   ".join(
        f"{c[0]} {inp['params'][('deg', c)]:+.3f} | {inp['params'][('off', c)]:+.2f}" for c in DRY))
    fixed, overall, mc = monte_carlo(inp)
    names = [*fixed, "adaptive"]
    base = mc[names].mean().min()
    fastest = mc[names].idxmin(axis=1).value_counts(normalize=True)
    print()
    print(f"{'strategy':10s} {'plan':40s} {'mean vs best':>12s} {'P(fastest)':>11s}")
    for name in names:
        plan = fmt(fixed[name]) if name in fixed else "best plan, re-planned at each SC"
        print(f"{name:10s} {plan:40s} {mc[name].mean() - base:+11.1f}s {fastest.get(name, 0):11.0%}")
    best_fixed = min(fixed, key=lambda n: mc[n].mean())
    sc_gain = (mc[mc.had_sc][best_fixed] - mc[mc.had_sc].adaptive).mean()
    print(f"Re-planning at each SC saves {sc_gain:.1f}s on average vs sticking to {best_fixed}, "
          f"in races with an SC ({mc.had_sc.mean():.0%} of simulations)")


def actual_strategies(t, n_laps):
    """Stops and first stop lap of the top-10 finishers on the lead lap."""
    fin = t.groupby("code").filter(lambda g: g.lap.max() == n_laps)
    top = fin[fin.lap == n_laps].nsmallest(10, "position").code
    f = fin[fin.code.isin(top)]
    return f.groupby("code").stint.nunique() - 1, f[f.pit_in.notna()].groupby("code").lap.min()


def strategy_rows(data, seasons, wear_scale):
    t, rows = data["tyre_laps"], []
    for (s, r), g in t[t.season.isin(seasons)].groupby(["season", "round"]):
        wet = g.compound.isin(["INTERMEDIATE", "WET"]).mean() > 0.02
        if wet or data["deg"].query("season == @s and round == @r").empty:
            continue  # wet races: strategy is driven by the weather
        inp = race_inputs(s, r, data, wear_scale)
        best, _ = planner(inp["n_laps"], inp["params"], inp["curve"], inp["pit"])
        plan = best(1, 3, 0)[1]
        stops, first = actual_strategies(g, inp["n_laps"])
        if stops.empty:
            continue
        mode = int(stops.mode().iat[0])
        hist = before(t[t.circuit == inp["circuit"]], s, r)
        hs = [actual_strategies(hg, hg.lap.max())[0] for _, hg in hist.groupby(["season", "round"])]
        hs = pd.concat(hs) if hs else pd.Series(dtype=int)
        on_mode = stops[stops == mode].index.intersection(first.index)
        rows.append({"season": s, "round": r, "circuit": inp["circuit"], "pred_stops": len(plan) - 1,
                     "actual_stops": mode, "hist_stops": int(hs.mode().iat[0]) if len(hs) else 1,
                     "pred_first": plan[0][2], "actual_first": first[on_mode].median()})
    return pd.DataFrame(rows)


def backtest(data, scales=(1, 1.5, 2, 2.5, 3, 4)):
    """Walk-forward: each test season uses the wear scale whose recommended stop counts best matched
    the top-10 cars' choices in all earlier seasons (teams know their real wear; their strategies
    reveal it). Eras shift (2022 cars, 2026 rules), so the scale is re-tuned every season."""
    rows = {(s, w): strategy_rows(data, [s], w) for s in range(2019, max(TEST_SEASONS) + 1) for w in scales}
    hit = lambda b: (b.pred_stops == b.actual_stops).sum()
    picked = []
    for s in TEST_SEASONS:
        past = {w: sum(hit(rows[(p, w)]) for p in range(2019, s)) for w in scales}
        w = max(past, key=past.get)
        picked.append(rows[(s, w)].assign(wear_scale=w))
        print(f"{s}: wear scale {w}x (fit on 2019-{s - 1})")
    b = pd.concat(picked)
    b.to_csv("data/strategy_backtest.csv", index=False)
    print(f"{len(b)} dry races 2021-2026, vs the most common strategy of the top-10 finishers:")
    print(f"  stop count right: model {np.mean(b.pred_stops == b.actual_stops):.0%}, "
          f"this track's past most common {np.mean(b.hist_stops == b.actual_stops):.0%}")
    same = b[b.pred_stops == b.actual_stops]
    print(f"  first stop lap, races where the stop count matched: median error "
          f"{(same.pred_first - same.actual_first).abs().median():.1f} laps ({len(same)} races)")
    print(pd.crosstab(b.pred_stops, b.actual_stops, rownames=["predicted stops"],
                      colnames=["actual stops"]).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    data = load()
    if sys.argv[1:] == ["backtest"]:
        backtest(data)
    elif sys.argv[1:] == ["pitcheck"]:
        queue_loss(data["tyre_laps"])
    else:
        report(int(sys.argv[1]), int(sys.argv[2]), data)


