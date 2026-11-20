# F1 race predictor

Predicts F1 finishing order, DNF risk and safety-car / red-flag likelihood from pre-race
information (grid, qualifying gaps, form, reliability, track history, weather), and compares
predictions with what actually happened.

Every model is tested walk-forward: each season is predicted by a model trained only on
earlier seasons, and scored against a simple baseline.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

## Run

```bash
python fetch.py   # pull data into data/races.csv (first full run ~1h due to API rate limits; cached after)
python model.py   # finishing-order model vs grid order -> data/predictions.csv
python risk.py    # DNF + safety car models -> data/dnf_predictions.csv, data/sc_predictions.csv
python sc_laps.py # lap-level safety car hazard -> data/sc_lap_predictions.csv
python tyres.py   # tyre degradation per race + pre-race / in-race prediction -> data/tyre_deg.csv, tyre_shape.csv
python sim.py 2024 11      # strategy report for one race: best 1/2/3-stop plans, Monte Carlo, SC re-planning
python sim.py backtest     # recommended strategy vs what the top-10 cars did, 2021-2026
python race_sim.py 2024 11 # multi-car: 1-stop vs 2-stop by expected finishing position
python race_sim.py backtest  # ~40 min
python assist.py 2024 6 NOR 20      # live strategy assistant: Miami 2024, Norris, after lap 20
python assist.py 2024 6 NOR replay  # its call every 5 laps vs what Norris actually did
python sc_calls.py                  # pit-or-stay under SC learned from teams, vs baselines
```

## Results page

`docs/index.html` shows the models race by race (2021-2026): predicted vs actual finishing order,
safety car risk by lap, tyre wear (pre-race estimate vs measured, and each team vs the field), and the
strategy calls vs what teams did. No dependencies; rebuild its data after re-running the models with:

```bash
python build_site.py   # -> docs/data.js
```

To view it locally: `python -m http.server 8765 --directory docs`, then open http://localhost:8765.

## Data sources

- [Jolpica](https://github.com/jolpica/jolpica-f1) (Ergast-compatible): results, qualifying, lap times
- [OpenF1](https://openf1.org): race control messages (safety car / VSC / red flag), 2023+
- [Open-Meteo](https://open-meteo.com) historical archive: rain and temperature at the circuit during the race
- [TracingInsights](https://github.com/TracingInsights) (Apache-2.0): every race lap 2018+ with tyre compound,
  tyre age, pit and track status (FastF1-style data, no F1 TV login needed)

Before 2023, safety cars / red flags are inferred from lap times (2+ of the top 3 running
30%+ off their median pace). On 2023 races, where both sources exist, this agrees with OpenF1 95% of the time.

## Results so far (2021–2026, 133 races)

| Model | Baseline | Result |
|---|---|---|
| Finishing order | Grid order: Spearman 0.634, MAE 3.43 | Spearman 0.666, MAE 3.31 |
| DNF probability | Average rate: Brier 0.120 | Brier 0.118, AUC 0.59 |
| SC / red flag (per race, pre-race) | Average rate: Brier 0.248 | Brier 0.245, AUC 0.55 |
| SC / red flag start (per lap, in-race) | Lap 1 vs rest: AUC 0.59 | AUC 0.79, Brier skill +6.9% |

DNFs and safety cars turn out to be mostly unpredictable from pre-race information alone.
In-race, per lap, they're much more predictable: lap 1 carries ~21% risk vs ~1% for a typical
lap, and a retirement in the last two laps raises the next lap's risk about 10x.

## Tyre degradation

Per race, on ~160k clean laps: `lap time = driver pace + fuel × lap + compound offset + wear × tyre age`.
Pit stops reset tyre age but not race lap, which separates tyre wear from fuel burn-off. Weak priors keep
races where the whole field pitted together (wet starts) from producing nonsense.

- Fuel + track evolution: cars get ~0.057 s/lap faster through a race; wear ~0.04 s per lap of tyre age.
- Wear isn't linear: tyres are slow for the first ~3 laps (warm-up), and softs/mediums fall off a cliff
  past ~30 laps (+0.1 to +0.5 s/lap); hards hold to ~40.
- Pre-race, the track's own history predicts wear best but only ~5% better than the overall average;
  race-to-race variation is ~8x the measurement noise.
- In-race, the race's own laps pin wear down: error 0.032 s/lap pre-race → 0.026 after 20 laps → 0.019 after 40.
- Team-specific wear is real: after removing measurement noise, teams differ by ~±0.016 s per lap of tyre
  age (~7 s over a 30-lap stint), and it holds within a season (first vs second half correlation 0.42).
  But it doesn't carry from one race to the next (forecast correlation 0.19): how kind a car is to its
  tyres depends heavily on whether the track suits it. Pre-race it adds little; in-race it can be
  measured from the laps run so far.

## Strategy simulator (work in progress)

Single-car, time-only. A dynamic programme finds the exact fastest plan (compounds + pit laps, 2+ dry
compounds); a Monte Carlo tests plans against uncertain wear and random safety cars, including
re-planning at each SC ("pit now under SC or stay out?").

- Pit loss measured as gap lost vs cars that stayed out: ~22 s green, ~13.5 s under SC (~40% cheaper).
- Race laps understate wear (drivers nurse tyres on long stints; that lost pace looks like driver pace),
  so wear is scaled up by a factor calibrated on teams' past choices: ~3x since 2023.
- Honest result: picks the top-10 cars' stop count 47% of the time vs 54% for "what usually works at this
  track". Missing piece: track position (a stop can cost places you can't win back), which needs a
  multi-car simulation.

### Multi-car simulation

All 20 cars lap by lap, pace from Q1 times. A faster car has to pass the one ahead; pass chance depends
on its pace advantage vs a per-track difficulty, calibrated so simulated on-track overtakes match the
real count there (~4 a race at Monaco, ~57 at Las Vegas). Cars that can't pass sit in dirty air, so
undercuts, traffic and track position emerge. A front-runner (P3) and a midfield car (P8) pick the
plan with the better expected finishing position.

Stop count (1 vs 2+) matching the top-10 cars, 115 dry races 2021-2026:

Every car also re-plans at each safety car (pit now cheaply, or stay out), and each chosen plan is
replayed through the real race's safety cars to count the stops it would really have made.

| Multi-car sim | ...replayed through real SCs | Time-only sim | Track history | Always 1-stop |
|---|---|---|---|---|
| 62% | 62% | 61% | 65% | 45% |

It roughly ties track history overall: better in 2023 (70% vs 50%) when history from the pre-2022 cars was
stale, worse in stable eras. SC reactions mostly move a planned stop to the cheap SC lap rather than add
one, so they barely change stop counts. Not yet modelled: team-specific tyre wear and pace, and rivals
choosing strategies in response to each other.

## Live strategy assistant

Replays a real race as if live, from a chosen lap, for a chosen driver:
1. Race state from the timing data: order, gaps, compound, tyre age and stints for every car.
2. Tyre wear re-estimated from this race's laps so far (field per compound, each team, each driver's
   pace), with the pre-race estimates as the prior.
3. Pit window: expected finishing position for each lap of the next stop, from the multi-car simulation
   started from the real gaps (rivals re-plan with their own team's wear; random safety cars).
4. "If a safety car came out now": pit or stay out.

Miami 2024, Norris: after lap 27, leading on 27-lap-old mediums with no stop yet, it says a safety car
now means pit (P2.0 vs P7.0 staying out). The safety car came on lap 28, he pitted and won. Verstappen,
already stopped, gets "stay out", which is what he did.

Checked against what teams actually did (`python assist.py check`, dry races 2023-2026 with exact OpenF1
safety car laps):

| | Assistant | Baseline |
|---|---|---|
| Pit window: best lap within 3 laps of the real first stop (350 stops, asked 8 laps before) | 35% | pre-race plan 30% |
| Pit window: median error | 5 laps | pre-race plan 6 laps |
| Safety car calls: pit or stay out matches the team (334 car-SCs) | 57% | always pit 62%, "pit if no stop yet or tyres 15+ laps old" 64% |

Calibrating the assistant's own wear scale (`python assist.py tune`: chosen on 2023, tested on 2024-2026):

| Wear scale | Pit window within 3 laps | Median timing bias | SC calls match team | Says pit / teams pitted |
|---|---|---|---|---|
| 1.5x | 26% | 7 laps late | 58% | 49% / 68% |
| **3x** | **47%** | 1 lap early | 59% | 51% / 68% |
| 4x | 46% | 3 laps early | 62% | 64% / 68% |
| 5x | 39% | 3 laps early | 46% | 51% / 68% |

(2023 decisions.) 3x is right for pit timing, and held out on 2024-2026 it still beats the pre-race plan
(31% vs 26% within 3 laps). Safety car calls barely move with the wear scale and never beat "always pit",
so that weakness isn't tyre wear.

Nor is it the safety car mechanics. `python sim.py pitcheck` fits the stop cost that best explains the
places really lost, in the same queue terms the simulation uses: 13.5 s under SC (mean error 0.19 places
over 1,114 stops) and 20.5 s under green, which is what the simulation already uses. Keeping the 3x wear
for decisions but measured wear for simulated lap times didn't help either (58-60% SC calls).

The ceiling is the issue: a model trained on teams' own SC decisions from per-car information (tyre age,
stops made, laps left, race progress) only reaches 62%, the same as "always pit". Within one safety car,
80% of cars make the same choice, so what drives it is race-level context (when the SC falls relative to
everyone's pit windows, track, expected SC length), not each car's tyres.

`python sc_calls.py` tries that: every safety car in dry races 2018-2026 (79 SCs, 790 top-10 decisions),
race-level features (laps left, progress, share of the field still to stop, field tyre age, track pit
loss, season) plus each car's state, walk-forward by season:

| Always pit | Rule of thumb | Per-car model | SC-level model |
|---|---|---|---|
| 60% | 62% | 61% | 61% |

Adding a "free stop" feature (still needs its mandatory stop and a new set would last to the end) or a
boosted model made it worse (56-58%): 79 safety cars is too few. The "free stop" effect is real (teams pit
74% of the time vs 50% otherwise), but most of what's learnable is each season's base rate, which drifts
from 47% (2021) to 71% (2023). The honest conclusion: pit-or-stay under a safety car isn't predictable
from public timing data beyond "usually pit"; it depends on things the data doesn't show (team radio,
expected SC length, double-stacking teammates).
