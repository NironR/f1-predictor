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
"""
import sys

import numpy as np
import pandas as pd

from race_sim import THRESHOLDS, calibrate, field, jittered, simulate
from sim import MIN_STINT, before, load, planner, race_inputs, shrunk
from tyres import DRY, LAP_NOISE, PRIORS, clean, race_fit

WEAR_SCALE = 3  # wear multiplier the walk-forward backtests chose for 2024-2026
TRUST = 4       # how much less than face value to trust this race's laps (tuned in tyres.in_race_report)
MAX_STOPS = 3
N_SIMS = 120


def race_state(t, lap):
    """Every car running at the end of `lap`, in order: gap to leader, stints so far, tyre age."""
    now = t[(t.lap == lap) & t.session_time.notna()].sort_values("session_time")
    cars = []
    for r in now.itertuples():
        d = t[(t.code == r.code) & (t.lap <= lap)].sort_values("lap")
        stints = [(s.compound.iat[0], int(s.lap.min()), int(s.lap.max())) for _, s in d.groupby("stint")]
        c, first, _ = stints[-1]
        age = int(r.tyre_life) if pd.notna(r.tyre_life) else lap - first + 1
        stints[-1] = (c, lap - age + 1, lap)  # current set: count age from when it was new
        cars.append({"code": r.code, "team": r.team_name, "gap": r.session_time - now.session_time.iat[0],
                     "stints": stints, "compound": c, "age": age})
    return cars


def live_params(t, lap, inp):
    """Wear re-estimated from this race's clean laps so far (pre-race estimates as priors)."""
    laps = clean(t[t.lap <= lap])
    pre = {c: inp["params"][("deg", c)] / WEAR_SCALE for c in DRY}
    priors = {**{f"deg_{c}": (pre[c], 0.045) for c in DRY}, **PRIORS}
    est = race_fit(laps, priors, noise=LAP_NOISE * TRUST, with_pace=True) if len(laps) else pd.Series(dtype=float)
    deg = {c: WEAR_SCALE * (est.get(f"deg_{c}", np.nan) if (laps.compound == c).any() else pre[c])
           for c in DRY}
    team = {k[3:]: WEAR_SCALE * v for k, v in est.items() if k.startswith("tw_") and not k.endswith("_se")}
    pace = {k[5:]: v for k, v in est.items() if k.startswith("pace_")}
    return deg, team, pace, laps


def assist(season, rnd, code, lap, data, verbose=True):
    t = data["tyre_laps"].query("season == @season and round == @rnd")
    inp = race_inputs(season, rnd, data, WEAR_SCALE)
    N = inp["n_laps"]
    cars = race_state(t, lap)
    if any(c["compound"] not in DRY for c in cars):
        raise SystemExit("Wet tyres in use: the tyre model only covers dry compounds.")
    me = next((i for i, c in enumerate(cars) if c["code"] == code), None)
    if me is None:
        raise SystemExit(f"{code} isn't running at lap {lap}.")
    deg, team_wear, pace_now, laps = live_params(t, lap, inp)
    from tyres import TEAM_LINEAGE
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
    rng = np.random.default_rng(0)
    q1_pace, q1_base = field(season, rnd, data)
    best0, _ = planner(N, inp["params"], inp["curve"], inp["pit"])
    ot = before(data["overtakes"], season, rnd)
    target = shrunk(ot[ot.circuit == inp["circuit"]].overtakes, ot.overtakes.median(), 2)
    th, _ = calibrate(inp, q1_pace, q1_base, [jittered(best0(1, 3, 0)[1], N, rng) for _ in q1_pace],
                      target, rng, None)

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
    if bin(mask).count("1") >= 2:
        cands["no more stops"] = mine["stints"][:-1] + [(c, first, N)]
    if done < MAX_STOPS:
        for m in range(lap + 1, N - MIN_STINT + 1, 2):
            rest = best_me(m + 1, MAX_STOPS - done - 1, mask)[1]
            if rest:
                cands[f"pit lap {m}"] = mine["stints"][:-1] + [(c, first, m)] + rest
    results = {name: expected(p) for name, p in cands.items()}
    best_name = min(results, key=lambda k: results[k][0])

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
        print(f"Wear now (s per lap of tyre age, x{WEAR_SCALE} strategy scale): "
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


if __name__ == "__main__":
    data = load()
    data["overtakes"] = pd.read_csv("data/overtakes.csv")
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
