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
```

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
