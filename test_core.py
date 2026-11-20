"""Checks for the core logic that has broken before. Run: python test_core.py (or pytest)."""
import itertools
import os
import tempfile
import time

import numpy as np
import pandas as pd

from tyres import DRY

PARAMS = {("off", "SOFT"): -0.5, ("off", "MEDIUM"): 0.0, ("off", "HARD"): 0.3,
          ("deg", "SOFT"): 0.15, ("deg", "MEDIUM"): 0.08, ("deg", "HARD"): 0.04}
FLAT = {c: np.zeros(100) for c in DRY}


def test_planner_finds_the_true_optimum():
    """The DP planner's best plan costs the same as brute force over every legal plan."""
    from sim import MIN_STINT, planner, race_time
    n, pit = 24, 12.0
    best, _ = planner(n, PARAMS, FLAT, pit)
    brute = np.inf
    for stops in range(1, 4):
        for cuts in itertools.combinations(range(1, n), stops):  # last lap of each stint but the final one
            bounds = [0, *cuts, n]
            if min(b - a for a, b in zip(bounds, bounds[1:])) < MIN_STINT:
                continue
            for comps in itertools.product(DRY, repeat=stops + 1):
                if len(set(comps)) >= 2:
                    plan = [(c, a + 1, b) for c, a, b in zip(comps, bounds, bounds[1:])]
                    brute = min(brute, race_time(plan, PARAMS, FLAT, pit, set()))
    cost, plan = best(1, 3, 0)
    assert abs(cost - brute) < 1e-9, (cost, brute)
    assert abs(race_time(plan, PARAMS, FLAT, pit, set()) - brute) < 1e-9


def test_used_tyres_count_age_from_new():
    """A stint on a set with laps already on it starts at that age (first lap can be < 1)."""
    from race_sim import plan_arrays
    comp, age, pit = plan_arrays([("MEDIUM", -2, 5), ("HARD", 6, 10)], 10)
    assert age[1] == 4 and age[5] == 8 and age[6] == 1
    assert comp[5] == DRY.index("MEDIUM") and comp[6] == DRY.index("HARD")
    assert pit[5] and not pit[10]


def test_safety_car_brings_the_stop_forward():
    """A cheap stop under a safety car near the planned stop gets taken; no SC, no change."""
    from sim import adaptive, planner
    n, pit = 30, 20.0
    best, bit = planner(n, PARAMS, FLAT, pit)
    plan = best(1, 3, 0)[1]
    first_stop = plan[0][2]
    inp = {"n_laps": n, "pit": pit}
    assert adaptive(plan, best, bit, inp, set()) == plan
    sc = set(range(first_stop - 3, first_stop))
    reacted = adaptive(plan, best, bit, inp, sc)
    assert reacted[0][2] == min(sc), (plan, reacted)


def test_race_state_survives_missing_stint_numbers():
    """Some laps have no stint number in the source data (crashed the replay check once)."""
    from assist import race_state
    t = pd.DataFrame({"code": ["AAA"] * 3 + ["BBB"] * 3, "lap": [1, 2, 3] * 2,
                      "session_time": [100, 190, 280, 101, 191, 281.0],
                      "stint": [np.nan] * 3 + [1] * 3, "compound": ["MEDIUM"] * 6,
                      "tyre_life": [1, 2, 3, 1, 2, 3], "team_name": ["X"] * 3 + ["Y"] * 3})
    cars = race_state(t, 3)
    assert [c["code"] for c in cars] == ["AAA", "BBB"]
    assert len(cars[0]["stints"]) == 1 and cars[0]["age"] == 3


def test_safety_car_periods_merge_and_cap():
    """A red flag during an SC is one period; an SC with no end message is capped at 10 laps."""
    from fetch import periods
    msgs = [{"date": "t1", "lap_number": 10, "message": "SAFETY CAR DEPLOYED"},
            {"date": "t2", "lap_number": 12, "message": "RED FLAG - RACE SUSPENDED"},
            {"date": "t3", "lap_number": 14, "message": "SAFETY CAR IN THIS LAP"},
            {"date": "t4", "lap_number": 30, "message": "SAFETY CAR DEPLOYED"}]
    assert periods(msgs) == [(10, 14), (30, 40)]


def test_cached_file_rebuilds_when_its_source_changes():
    from sim import cached
    with tempfile.TemporaryDirectory() as d:
        src, out, calls = os.path.join(d, "src.csv"), os.path.join(d, "out.csv"), []
        build = lambda: calls.append(1) or pd.DataFrame({"x": [len(calls)]})
        open(src, "w").write("x\n1\n")
        cached(out, build, src)
        cached(out, build, src)
        assert len(calls) == 1  # fresh cache: not rebuilt
        later = time.time() + 5
        os.utime(src, (later, later))
        assert cached(out, build, src).x.iat[0] == 2 and len(calls) == 2


def test_features_never_use_later_races():
    from model import check_no_leak
    check_no_leak(pd.read_csv("data/races.csv"))


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
