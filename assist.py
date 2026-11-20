"""Live race strategy assistant: from the state of a race at a given lap, the pit window for one car.

Replays a real race as if live, using only laps up to `lap`:
  1. race state: order, gaps, compound, tyre age and stints so far for every running car
  2. tyre wear re-estimated from this race's laps so far (field per compound, each team, each driver's
     pace), starting from the pre-race estimates as the prior
  3. pit window: expected finishing position for each possible lap of the next stop, from the
     multi-car simulation (rivals re-plan too, random safety cars)
  4. "if a safety car came out now": pit or stay out

Usage:  python assist.py 2024 6 NOR 20       # Miami 2024, Norris, after lap 20
        python assist.py 2024 6 NOR replay   # its call every 5 laps vs what Norris did
        python assist.py check               # vs what teams did, all dry races 2023-2026 (~12 min)
        python assist.py tune                # pick the wear scale on 2023, test on 2024-2026 (~45 min)
"""
import sys

import numpy as np
import pandas as pd

from model import TEST_SEASONS, save_metrics
from race_sim import THRESHOLDS, calibrate, field, jittered, simulate
from sim import MIN_STINT, before, load, planner, race_inputs, shrunk
from tyres import DRY, FIT_LAP_SD, PRIORS, TEAM_LINEAGE, clean, race_fit

WEAR_SCALE = 3  # best for pit timing on 2023 decisions (python assist.py tune); held-out 2024-26 below
TRUST = 4       # how much less than face value to trust this race's laps (tuned in tyres.in_race_report)
MAX_STOPS = 3
N_SIMS = 120


def race_state(t, lap):
    """Every car running at the end of `lap`, in order: gap to leader, stints so far, tyre age."""
    now = t[(t.lap == lap) & t.session_time.notna()].sort_values("session_time")
    cars = []
    for r in now.itertuples():
        d = t[(t.code == r.code) & (t.lap <= lap)].sort_values("lap")
        stint_no = d.stint.ffill().bfill().fillna(1)  # a few laps have no stint number in the data
        stints = [(s.compound.iat[0], int(s.lap.min()), int(s.lap.max())) for _, s in d.groupby(stint_no)]
        c, first, _ = stints[-1]
        age = int(r.tyre_life) if pd.notna(r.tyre_life) else lap - first + 1
        stints[-1] = (c, lap - age + 1, lap)  # current set: count age from when it was new
        cars.append({"code": r.code, "team": r.team_name, "gap": r.session_time - now.session_time.iat[0],
                     "stints": stints, "compound": c, "age": age})
    return cars


def live_params(t, lap, inp, wear):
    """Wear re-estimated from this race's clean laps so far (pre-race estimates as priors)."""
    laps = clean(t[t.lap <= lap])
    pre = {c: inp["params"][("deg", c)] / wear for c in DRY}
    priors = {**{f"deg_{c}": (pre[c], 0.045) for c in DRY}, **PRIORS}
    est = race_fit(laps, priors, noise=FIT_LAP_SD * TRUST, with_pace=True) if len(laps) else pd.Series(dtype=float)
    deg = {c: wear * (est.get(f"deg_{c}", np.nan) if (laps.compound == c).any() else pre[c])
           for c in DRY}
    team = {k[3:]: wear * v for k, v in est.items() if k.startswith("tw_") and not k.endswith("_se")}
    pace = {k[5:]: v for k, v in est.items() if k.startswith("pace_")}
    return deg, team, pace, laps


_pass_difficulty = {}  # per race: calibration only uses pre-race information, so reuse it


def assist(season, rnd, code, lap, data, verbose=True, window=True, wear=WEAR_SCALE):
    """window=False skips the pit window (only the safety car call), for the replay check."""
    t = data["tyre_laps"].query("season == @season and round == @rnd")
    inp = race_inputs(season, rnd, data, wear)
    N = inp["n_laps"]
    cars = race_state(t, lap)
    if any(s[0] not in DRY for c in cars for s in c["stints"]):
        raise SystemExit("Wet tyres used in this race so far: the tyre model only covers dry compounds.")
    me = next((i for i, c in enumerate(cars) if c["code"] == code), None)
    if me is None:
        raise SystemExit(f"{code} isn't running at lap {lap}.")
    deg, team_wear, pace_now, laps = live_params(t, lap, inp, wear)
    lineage = lambda name: TEAM_LINEAGE.get(name, str(name).lower().replace(" ", "_"))
    inp = {**inp, "params": {**inp["params"], **{("deg", c): v for c, v in deg.items()}}}

    # one planner per team: its own wear on top of the field's
    planners = {}
    def plan_for(car):
        tm = lineage(car["team"])
        if tm not in planners:
            extra = team_wear.get(tm, 0.0)
            params = {**inp["params"], **{("deg", c): deg[c] + extra for c in DRY}}
            planners[tm] = planner(N, params, inp["curve"], inp["pit"])
        return planners[tm]

    bit = {c: 1 << i for i, c in enumerate(DRY)}
    def rest_of_race(car):  # best plan from here, stints so far included
        best, _ = plan_for(car)
        mask = sum({bit[s[0]] for s in car["stints"]})
        done = len(car["stints"]) - 1
        cont = best(lap + 1, MAX_STOPS - done, mask, car["compound"], car["age"])[1] if lap < N else []
        c, first, _ = car["stints"][-1]
        cont = cont or [(c, first, N)]
        return car["stints"][:-1] + [(c, first, cont[0][2])] + cont[1:]

    plans = [rest_of_race(c) for c in cars]
    reacts = [plan_for(c) for c in cars]
    car_deg = [team_wear.get(lineage(c["team"]), 0.0) for c in cars]
    median_pace = np.median(list(pace_now.values())) if pace_now else 0.0
    pace = np.array([pace_now.get(c["code"], median_pace) for c in cars])
    base = laps.time.median() if len(laps) else field(season, rnd, data)[1]
    T0 = np.array([c["gap"] for c in cars])

    # pass difficulty: calibrated pre-race exactly as race_sim.decide does
    if (season, rnd, wear) not in _pass_difficulty:
        rng = np.random.default_rng(0)
        pre = race_inputs(season, rnd, data, wear)
        q1_pace, q1_base = field(season, rnd, data)
        best0, _ = planner(N, pre["params"], pre["curve"], pre["pit"])
        ot = before(data["overtakes"], season, rnd)
        target = shrunk(ot[ot.circuit == pre["circuit"]].overtakes, ot.overtakes.median(), 2)
        _pass_difficulty[(season, rnd, wear)] = calibrate(pre, q1_pace, q1_base, [jittered(best0(1, 3, 0)[1], N, rng)
                                                                           for _ in q1_pace], target, rng, None)[0]
    th = _pass_difficulty[(season, rnd, wear)]

    def expected(plan, sc_fixed=None, react_me=True):
        """react_me=False: my car sticks to `plan` even under SC (when that choice is what's tested)."""
        ps, rs = list(plans), list(reacts)
        ps[me] = plan
        rs[me] = reacts[me] if react_me else None
        pos, _ = simulate(inp, pace, base, ps, th, N_SIMS, np.random.default_rng(1), rs, sc_fixed,
                          start_lap=lap, T0=T0, car_deg=car_deg)  # same seed: candidates face the same races
        return pos[:, me].mean() + 1, (pos[:, me] < me).mean()

    # candidate plans for my car: next stop on lap m (then the best rest), or no more stops
    mine = cars[me]
    best_me, _ = plan_for(mine)
    mask = sum({bit[s[0]] for s in mine["stints"]})
    done = len(mine["stints"]) - 1
    c, first, _ = mine["stints"][-1]
    cands = {}
    if not window:
        pass
    elif bin(mask).count("1") >= 2:
        cands["no more stops"] = mine["stints"][:-1] + [(c, first, N)]
    if window and done < MAX_STOPS:
        for m in range(lap + 1, N - MIN_STINT + 1, 2):
            rest = best_me(m + 1, MAX_STOPS - done - 1, mask)[1]
            if rest:
                cands[f"pit lap {m}"] = mine["stints"][:-1] + [(c, first, m)] + rest
    results = {name: expected(p) for name, p in cands.items()}
    best_name = min(results, key=lambda k: results[k][0]) if results else None

    sc_now = set(range(lap + 1, lap + 5))
    pit_now = mine["stints"][:-1] + [(c, first, lap + 1)] + best_me(lap + 2, MAX_STOPS - done - 1, mask)[1] \
        if done < MAX_STOPS else None
    stay = plans[me]
    sc = {"pit under SC": expected(pit_now, sc_now, False) if pit_now else None,
          "stay out": expected(stay, sc_now, False)}

    actual = t[(t.code == code) & (t.lap > lap) & t.pit_in.notna()].lap.tolist()
    if verbose:
        race = data["races"].query("season == @season and round == @rnd").race.iat[0]
        ahead = cars[me - 1]["gap"] if me else None
        print(f"{season} {race}, after lap {lap}/{N}: {code} P{me + 1}, {mine['compound']} age {mine['age']}, "
              f"{done} stop(s), {mine['gap'] - ahead if me else 0:+.1f}s to the car ahead")
        print(f"Wear now (s per lap of tyre age, x{wear} strategy scale): "
              + ", ".join(f"{c[0]} {deg[c]:+.3f}" for c in DRY)
              + f"; {mine['team']} {team_wear.get(lineage(mine['team']), 0):+.3f} vs field"
              f"   pass difficulty {th}s, pit loss {inp['pit']:.1f}s")
        print("Pit window: expected finish (P(finishing ahead of current position)):")
        for name, (p, gain) in results.items():
            bar = "#" * max(1, int(round(25 - p)))
            print(f"  {name:14s} P{p:4.1f}  ({gain:4.0%})  {bar}{'  <- best' if name == best_name else ''}")
        verdict = min((k for k in sc if sc[k]), key=lambda k: sc[k][0])
        if sc["pit under SC"] and abs(sc["pit under SC"][0] - sc["stay out"][0]) < 0.15:
            verdict = "no real difference"
        print("If a safety car came out now: " + ", ".join(f"{k} -> P{v[0]:.1f}" for k, v in sc.items() if v)
              + f"  => {verdict}")
        print(f"What actually happened: pitted on lap(s) {actual or 'none (no more stops)'}")
    return results, best_name, sc, actual


def replay_check(data, seasons=tuple(s for s in TEST_SEASONS if s >= 2023), wear=WEAR_SCALE, lead=8, near=3, tol=0.25,
                 out="data/assist", verbose=True):
    """The assistant against what teams actually did, in dry races with exact OpenF1 safety car laps.
    Pit window: `lead` laps before each top-6 finisher's first green-flag stop, is the best lap within
    `near` laps of the real stop, and is the real stop in the window (laps within `tol` positions of best)?
    Safety cars: the lap before each real SC, every top-10 car's pit/stay call vs what its team did."""
    from sim import actual_strategies
    t_all, periods = data["tyre_laps"], data["periods"].query("source == 'openf1'")
    win, scs = [], []
    for (s, r), t in t_all[t_all.season.isin(seasons)].groupby(["season", "round"]):
        if t.compound.isin(["INTERMEDIATE", "WET"]).any() or data["deg"].query(
                "season == @s and round == @r").empty:
            continue
        N = int(t.lap.max())
        stops, _ = actual_strategies(t, N)
        top = list(stops.index)  # top-10 finishers on the lead lap, in finishing order
        p = periods.query("season == @s and round == @r")
        sc_laps = {l for o, e in zip(p.onset, p.end) for l in range(o, e + 1)}
        pre = race_inputs(s, r, data, wear)
        pre_plan = planner(N, pre["params"], pre["curve"], pre["pit"])[0](1, 3, 0)[1]
        fin = t[t.lap == N].set_index("code").position
        for code in sorted(top, key=lambda c: fin.get(c, 99))[:6]:
            first = t[(t.code == code) & t.pit_in.notna()].lap.min()
            if pd.isna(first) or first in sc_laps or first - lead < 3:
                continue  # SC stops are tested below
            try:
                res, best, _, _ = assist(s, r, code, int(first - lead), data, verbose=False, wear=wear)
            except SystemExit:
                continue
            pit = {int(k.split()[-1]): v[0] for k, v in res.items() if k.startswith("pit lap")}
            if not pit or best is None or not best.startswith("pit lap"):
                continue
            lo = min(pit.values())
            window_laps = [m for m, v in pit.items() if v <= lo + tol]
            best_lap = int(best.split()[-1])
            win.append({"season": s, "round": r, "code": code, "actual": first, "best": best_lap,
                        "in_window": min(window_laps) - 1 <= first <= max(window_laps) + 1,
                        "width": max(window_laps) - min(window_laps) + 2, "pre_race": pre_plan[0][2]})
        for onset, end in zip(p.onset, p.end):
            if onset < 3 or onset > N - 3:
                continue
            for code in top:
                try:
                    _, _, sc, _ = assist(s, r, code, int(onset - 1), data, verbose=False, window=False, wear=wear)
                except SystemExit:
                    continue
                if not sc["pit under SC"]:
                    continue
                d = t[(t.code == code) & (t.lap == onset - 1)]
                pitted = t[(t.code == code) & t.pit_in.notna()].lap.between(onset, min(end, onset + 2)).any()
                stops_done = int(d.stint.iat[0]) - 1 if len(d) else 0
                scs.append({"season": s, "round": r, "onset": onset, "code": code, "pitted": pitted,
                            "call_pit": sc["pit under SC"][0] < sc["stay out"][0],
                            "margin": sc["stay out"][0] - sc["pit under SC"][0],
                            "tyre_age": int(d.tyre_life.iat[0]) if len(d) and pd.notna(d.tyre_life.iat[0]) else 0,
                            "stops_done": stops_done})
        if verbose:
            print(f"{s} R{r}: {len(win)} pit-window checks, {len(scs)} SC calls so far", flush=True)
    w, c = pd.DataFrame(win), pd.DataFrame(scs)
    if out:
        w.to_csv(f"{out}_window_check.csv", index=False)
        c.to_csv(f"{out}_sc_check.csv", index=False)
    if verbose:
        summarise(w, c, lead, near)
    if out:  # the canonical run: record headline results, pit timing held out from the 2023 tuning
        held = w[w.season >= 2024]
        save_metrics("assistant", {"pit_within_3": ((held.best - held.actual).abs() <= near).mean(),
                                   "pre_race_within_3": ((held.pre_race - held.actual).abs() <= near).mean(),
                                   "sc_calls": (c.call_pit == c.pitted).mean(), "always_pit": c.pitted.mean()})
    return w, c


def summarise(w, c, lead=8, near=3):
    print(f"Pit window, {len(w)} first stops (asked {lead} laps before):")
    print(f"  best lap within {near} laps of the real stop: assistant {np.mean((w.best - w.actual).abs() <= near):.0%}, "
          f"pre-race plan {np.mean((w.pre_race - w.actual).abs() <= near):.0%}")
    print(f"  median error: assistant {(w.best - w.actual).abs().median():.0f} laps, "
          f"pre-race plan {(w.pre_race - w.actual).abs().median():.0f} laps")
    print(f"  real stop inside the window: {w.in_window.mean():.0%} (median window {w.width.median():.0f} laps wide)")
    rule = (c.stops_done == 0) | (c.tyre_age >= 15)  # rule of thumb: pit if no stop yet or tyres 15+ laps old
    print(f"Safety car calls, {len(c)} car-SCs ({c.pitted.mean():.0%} actually pitted):")
    print(f"  assistant {np.mean(c.call_pit == c.pitted):.0%}   always pit {c.pitted.mean():.0%}   "
          f"never pit {1 - c.pitted.mean():.0%}   rule of thumb {np.mean(rule == c.pitted):.0%}")
    print(pd.crosstab(c.call_pit, c.pitted, rownames=["assistant: pit"], colnames=["team pitted"]).to_string())


def score(w, c, near=3):
    """Mean of: share of first stops called within `near` laps, share of SC calls matching the team."""
    return ((w.best - w.actual).abs() <= near).mean() / 2 + (c.call_pit == c.pitted).mean() / 2


def tune(data, scales=(1, 1.5, 2, 3)):
    """Pick the assistant's wear scale on 2023 decisions, then test it on 2024-2026 against 3x."""
    fit = {x: score(*replay_check(data, [2023], x, out=None, verbose=False)) for x in scales}
    print("2023 score by wear scale (mean of pit-window and SC-call accuracy):",
          {x: f"{v:.0%}" for x, v in fit.items()})
    pick = max(fit, key=fit.get)
    test = (2024, 2025, 2026)
    for x in dict.fromkeys([pick, 3]):
        print(f"\n2024-2026 with wear x{x}:")
        summarise(*replay_check(data, test, x, out=None, verbose=False))
    return pick


if __name__ == "__main__":
    data = load()
    if sys.argv[1:] == ["check"]:
        replay_check(data)
        raise SystemExit
    if sys.argv[1:] == ["tune"]:
        tune(data)
        raise SystemExit
    s, r, code = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].upper()
    if sys.argv[4] != "replay":
        assist(s, r, code, int(sys.argv[4]), data)
    else:  # the assistant's call every 5 laps through the race, next to what the driver did
        t = data["tyre_laps"].query("season == @s and round == @r")
        pits = t[(t.code == code) & t.pit_in.notna()].lap.tolist()
        print(f"{code}: actually pitted on lap(s) {pits}")
        for lap in range(5, int(t.lap.max()) - 4, 5):
            res, best, sc, _ = assist(s, r, code, lap, data, verbose=False)
            sc_call = min((k for k in sc if sc[k]), key=lambda k: sc[k][0])
            print(f"  after lap {lap:2d}: best = {best:14s} (P{res[best][0]:.1f})   SC now -> {sc_call}")
