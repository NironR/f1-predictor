"""Multi-car race simulation: strategy chosen for finishing position, not just race time.

All 20 cars run lap by lap (many races at once, vectorised over simulations). A faster car must pass
the car ahead: pass chance rises with its pace advantage, scaled by a per-track difficulty calibrated
so simulated on-track overtakes match the real count there. A car that can't pass is held in dirty
air. That's what makes track position, undercuts and traffic matter.

Usage:  python race_sim.py 2024 11     # 1-stop vs 2-stop by expected finishing position
        python race_sim.py backtest    # chosen stop count vs what the top-10 cars did, 2021-2026
"""
import sys

import numpy as np
import pandas as pd

from model import TEST_SEASONS, features
from sim import (PARAM_SD, SC_PIT_FACTOR, actual_strategies, before, load, planner, race_inputs,
                 sc_timeline, shrunk)
from tyres import DRY

MIN_GAP = 0.4            # s; closest a car can follow without passing
SC_GAP = 0.8             # s between cars when bunched behind the safety car
LAP_NOISE = 0.3          # s, lap-to-lap variation per car
THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # pace advantage (s/lap) for a 50% pass chance
PASS_SOFTNESS = 0.15  # s; how sharply pass chance rises around the threshold
REP_SLOTS = (3, 8)       # grid slots of the car whose strategy we choose


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


def field(season, rnd, data):
    """The real grid for this race: grid order and pace offset (s/lap) from Q1, the one session
    everyone runs in the same conditions (features' quali_gap is per-session, so a Q1 exit 0.1s off
    the Q1 best would look as fast as pole).
    ponytail: front-runners don't always push in Q1, so their edge is a little understated."""
    f = features(data["races"]).query("season == @season and round == @rnd").sort_values("grid")
    base = data["tyre_laps"].query("season == @season and round == @rnd and track_status == 1").time.median()
    gap = (f.q1 / f.q1.min() - 1).clip(0, 0.05)
    return (gap.fillna(gap.max()).fillna(0.03) * base).to_numpy(), base


def plan_arrays(plan, n_laps):
    comp, age, pit = np.zeros(n_laps + 1, int), np.zeros(n_laps + 1, int), np.zeros(n_laps + 1, bool)
    for c, first, last in plan:
        comp[first:last + 1] = DRY.index(c)
        age[first:last + 1] = np.arange(1, last - first + 2)
        pit[last] = last < n_laps
    return comp, age, pit


def simulate(inp, pace, base, plans, threshold, n_sims, rng):
    """Returns final positions (sims x cars, 0 = winner) and on-track passes per race."""
    n_cars, N = len(pace), inp["n_laps"]
    comp, age, pit = map(np.array, zip(*(plan_arrays(p, N) for p in plans)))   # cars x laps
    off = np.array([inp["params"][("off", c)] for c in DRY])
    deg = np.array([inp["params"][("deg", c)] for c in DRY])
    true_off = off + rng.normal(0, PARAM_SD["off"], (n_sims, 3)) * (np.array(DRY) != "MEDIUM")
    true_deg = deg + rng.normal(0, PARAM_SD["deg"], (n_sims, 3))
    curve = np.array([inp["curve"][c] for c in DRY])                             # 3 x ages
    car_pace = pace + rng.normal(0, 0.1, (n_sims, n_cars))
    sc = np.zeros((n_sims, N + 1), bool)
    for s in range(n_sims):
        sc[s, [l for l in sc_timeline(inp, rng) if l <= N]] = True
    T = np.tile(np.arange(n_cars) * 0.3, (n_sims, 1))                           # grid gaps
    rows = np.arange(n_sims)
    passes = np.zeros(n_sims)
    for lap in range(1, N + 1):
        c, a = comp[:, lap], age[:, lap]
        lt = (base + car_pace + true_off[:, c] + true_deg[:, c] * a + curve[c, a]
              + rng.normal(0, LAP_NOISE, (n_sims, n_cars)))
        pitting = np.broadcast_to(pit[:, lap], (n_sims, n_cars))
        loss = np.where(sc[:, lap], SC_PIT_FACTOR, 1.0)[:, None] * inp["pit"]
        prov = T + lt + pitting * loss
        if lap > 1 and sc[:, lap].any():  # bunch the field behind the SC, order kept
            rank = prov.argsort(1).argsort(1)
            bunched = prov.min(1, keepdims=True) + SC_GAP * rank
            prov = np.where(sc[:, lap][:, None], bunched, prov)
        order = T.argsort(1)
        new = prov.copy()
        ahead = np.full(n_sims, -np.inf)  # time of the car just ahead (running max, non-pitting)
        for p in range(n_cars):
            car = order[:, p]
            t_car = prov[rows, car]
            racing = ~pitting[rows, car] & ~sc[:, lap] & (lap > 1)
            blocked = racing & (t_car < ahead + MIN_GAP)
            adv = np.maximum(ahead + MIN_GAP - t_car, 0)  # how much faster than the car ahead
            p_pass = 1 / (1 + np.exp(-(adv - threshold) / PASS_SOFTNESS))
            passed = blocked & (rng.random(n_sims) < p_pass)
            stuck = blocked & ~passed
            new[rows, car] = np.where(stuck, ahead + MIN_GAP, t_car)
            passes += passed & (t_car < ahead)
            ahead = np.where(pitting[rows, car], ahead, np.maximum(ahead, new[rows, car]))
        T = new
    return T.argsort(1).argsort(1), passes


def calibrate(inp, pace, base, plans, target, rng, n_sims=30):
    """Pass difficulty whose simulated overtakes per race best match the real count."""
    got = {th: simulate(inp, pace, base, plans, th, n_sims, rng)[1].mean() for th in THRESHOLDS}
    return min(got, key=lambda th: abs(got[th] - target)), got


def jittered(plan, n_laps, rng, spread=3):
    """Same compounds, pit laps moved a few laps, so rivals don't all stop on the same lap."""
    out, shift = [], 0
    for i, (c, first, last) in enumerate(plan):
        first = first + shift
        if i < len(plan) - 1:
            shift = int(rng.integers(-spread, spread + 1))
            last = int(np.clip(last + shift, first + 4, n_laps - 4))
            shift = last - plan[i][2]
        out.append((c, first, n_laps if i == len(plan) - 1 else last))
    return out


def decide(season, rnd, data, wear_scale, n_sims=150, seed=0, verbose=False):
    """Expected finishing position of a front-runner and a midfield car on the best 1-stop vs best
    2-stop plan, with the rest of the field on the time-optimal plan."""
    rng = np.random.default_rng(seed)
    inp = race_inputs(season, rnd, data, wear_scale)
    pace, base = field(season, rnd, data)
    best, _ = planner(inp["n_laps"], inp["params"], inp["curve"], inp["pit"])
    default = best(1, 3, 0)[1]
    cands = {k: best(1, k, 0, exact=True)[1] for k in (1, 2)}
    rivals = [jittered(default, inp["n_laps"], rng) for _ in pace]
    ot = before(data["overtakes"], season, rnd)
    target = shrunk(ot[ot.circuit == inp["circuit"]].overtakes, ot.overtakes.median(), 2)
    th, _ = calibrate(inp, pace, base, rivals, target, rng)
    pos = {}
    for k, plan in cands.items():
        for slot in REP_SLOTS:
            plans = list(rivals)
            plans[slot - 1] = plan
            pos[(k, slot)] = simulate(inp, pace, base, plans, th, n_sims, rng)[0][:, slot - 1].mean() + 1
    score = {k: np.mean([pos[(k, s)] for s in REP_SLOTS]) for k in cands}
    choice = min(score, key=score.get)
    if verbose:
        print(f"{inp['circuit']}: {inp['n_laps']} laps, pass difficulty {th}s "
              f"(target {target:.0f} overtakes/race), pit loss {inp['pit']:.1f}s, wear x{wear_scale}")
        for k, plan in cands.items():
            stints = " > ".join(f"{c[0]}{l - f + 1}" for c, f, l in plan)
            print(f"  {k}-stop {stints:22s} expected finish: "
                  + ", ".join(f"from P{s} -> {pos[(k, s)]:.1f}" for s in REP_SLOTS))
        print(f"  choose {choice}-stop")
    return choice, default


def backtest(data, scales=(1, 2, 3)):
    """Walk-forward like sim.backtest: the wear scale for each test season is the one whose
    choices best matched the top-10 cars in earlier seasons."""
    t, rows = data["tyre_laps"], []
    for (s, r), g in t[t.season >= 2019].groupby(["season", "round"]):
        if g.compound.isin(["INTERMEDIATE", "WET"]).mean() > 0.02 or data["deg"].query(
                "season == @s and round == @r").empty:
            continue
        stops, _ = actual_strategies(g, int(g.lap.max()))
        if stops.empty:
            continue
        ot = before(data["overtakes"], s, r)
        hist = before(t[t.circuit == g.circuit.iat[0]], s, r)
        hs = [actual_strategies(hg, hg.lap.max())[0] for _, hg in hist.groupby(["season", "round"])]
        hs = pd.concat(hs) if hs else pd.Series(dtype=int)
        row = {"season": s, "round": r, "circuit": g.circuit.iat[0], "actual": int(stops.mode().iat[0]),
               "hist_stops": int(hs.mode().iat[0]) if len(hs) else 1}
        for w in scales:
            row[f"choice_{w}"], default = decide(s, r, data, w)
            row[f"timeonly_{w}"] = min(len(default) - 1, 2)
        rows.append(row)
        print(f"{s} R{r} {row['circuit']}: actual {row['actual']}, "
              + " ".join(f"x{w}->{row[f'choice_{w}']}" for w in scales), flush=True)
    b = pd.DataFrame(rows)
    picked = []
    for s in TEST_SEASONS:
        past = b[b.season < s]
        w = max(scales, key=lambda w: (past[f"choice_{w}"] == past.actual.clip(upper=2)).sum())
        picked.append(b[b.season == s].assign(wear_scale=w, choice=b[f"choice_{w}"],
                                              timeonly=b[f"timeonly_{w}"]))
    p = pd.concat(picked)
    p.to_csv("data/race_sim_backtest.csv", index=False)
    actual = p.actual.clip(upper=2)  # we choose between 1 and 2 stops; 3+ counts as "2+"
    print(f"\n{len(p)} dry races 2021-2026, stop count (1 vs 2+) matching the top-10 cars' most common:")
    print(f"  multi-car sim {np.mean(p.choice == actual):.0%}   time-only sim {np.mean(p.timeonly == actual):.0%}   "
          f"track history {np.mean(p.hist_stops.clip(upper=2) == actual):.0%}   always 1-stop {np.mean(actual == 1):.0%}")
    print("  wear scale used per season:", p.groupby("season").wear_scale.first().to_dict())


if __name__ == "__main__":
    data = load()
    try:
        data["overtakes"] = pd.read_csv("data/overtakes.csv")
    except FileNotFoundError:
        data["overtakes"] = (data["tyre_laps"].groupby(["season", "round", "circuit"])
                             .apply(overtakes, include_groups=False).rename("overtakes").reset_index())
        data["overtakes"].to_csv("data/overtakes.csv", index=False)
    if sys.argv[1:] == ["backtest"]:
        backtest(data)
    else:
        decide(int(sys.argv[1]), int(sys.argv[2]), data, float(sys.argv[3]) if len(sys.argv) > 3 else 3,
               verbose=True)
