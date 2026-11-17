"""Pull F1 results, qualifying and race-time weather into data/races.csv.

Raw API responses are cached in data/raw/, so re-runs only fetch what's missing.
Usage: python fetch.py [first_season] [last_season]
"""
import json, sys, time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
JOLPICA = "https://api.jolpi.ca/ergast/f1"
METEO = "https://archive-api.open-meteo.com/v1/archive"
OPENF1 = "https://api.openf1.org/v1"
OPENF1_FROM = 2023  # OpenF1 starts 2023; 2023 overlaps to validate lap inference


def get(url, params, cache, pause=0.3):
    path = RAW / cache
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(8):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:  # Jolpica: 4 req/s, 500 req/hour
            time.sleep(60 * (attempt + 1))
            continue
        if r.status_code == 404:  # OpenF1: session has no messages
            path.write_text("[]")
            return []
        r.raise_for_status()
        time.sleep(pause)
        path.write_text(r.text)
        return r.json()
    raise RuntimeError(f"rate limited: {url}")


def paged(season, kind):
    """All Races for a season/endpoint, following Jolpica's 100-row pages."""
    races, offset = [], 0
    while True:
        d = get(f"{JOLPICA}/{season}/{kind}.json", {"limit": 100, "offset": offset},
                f"{season}_{kind}_{offset}.json")["MRData"]
        races += d["RaceTable"]["Races"]
        offset += 100
        if offset >= int(d["total"]):
            return races


def secs(t):
    if not t:
        return None
    return sum(float(x) * 60 ** i for i, x in enumerate(reversed(t.split(":"))))


def season_rows(season):
    rows = {}
    for race in paged(season, "results"):
        for r in race["Results"]:
            rows[(race["round"], r["Driver"]["driverId"])] = {
                "season": season, "round": int(race["round"]), "race": race["raceName"],
                "circuit": race["Circuit"]["circuitId"], "date": race["date"],
                "time": race.get("time", "12:00:00Z"),
                "lat": float(race["Circuit"]["Location"]["lat"]),
                "lon": float(race["Circuit"]["Location"]["long"]),
                "driver": r["Driver"]["driverId"], "team": r["Constructor"]["constructorId"],
                "grid": int(r["grid"]), "position": int(r["position"]), "status": r["status"],
                "points": float(r["points"]),
            }
    for race in paged(season, "qualifying"):
        for q in race["QualifyingResults"]:
            row = rows.get((race["round"], q["Driver"]["driverId"]))
            if row:
                row.update(q_pos=int(q["position"]),
                           q1=secs(q.get("Q1")), q2=secs(q.get("Q2")), q3=secs(q.get("Q3")))
    return list(rows.values())


def weather(race):
    """Rain (mm) and air temp over the race window at the circuit."""
    start = datetime.fromisoformat(f"{race.date}T{race.time.rstrip('Z')}")
    d = get(METEO, {"latitude": race.lat, "longitude": race.lon, "start_date": race.date,
                    "end_date": race.date, "hourly": "precipitation,temperature_2m",
                    "timezone": "UTC"}, f"wx_{race.season}_{race.round}.json")["hourly"]
    hrs = [i for i, t in enumerate(d["time"])
           if start - timedelta(hours=1) <= datetime.fromisoformat(t) <= start + timedelta(hours=2)]
    rain = sum(d["precipitation"][i] or 0 for i in hrs)
    temps = [d["temperature_2m"][i] for i in hrs if d["temperature_2m"][i] is not None]
    return rain, (sum(temps) / len(temps) if temps else None)


def lap_times(race, driver):
    races = get(f"{JOLPICA}/{race.season}/{race.round}/drivers/{driver}/laps.json", {"limit": 100},
                f"laps_{race.season}_{race.round}_{driver}.json", pause=7.5)["MRData"]["RaceTable"]["Races"]
    return pd.Series({int(l["number"]): secs(l["Timings"][0]["time"]) for l in races[0]["Laps"]} if races else {},
                     dtype=float)


def slow_laps(race, drivers, slow=1.3):
    """Laps where 2+ of the top 3 ran `slow`x off their own median pace (SC / VSC / red flag).
    Lap 1 (standing start) only counts if it's 2x off."""
    flags = []
    for d in drivers:
        t = lap_times(race, d)
        if len(t) > 5:
            f = t > slow * t.iloc[1:].median()
            f.iloc[0] = t.iloc[0] > 2 * t.iloc[1:].median()
            flags.append(f)
    return int((pd.concat(flags, axis=1).sum(axis=1) >= 2).sum()) if len(flags) >= 2 else None


def race_control(season):
    """{race date: {sc, vsc, red}} from OpenF1 race control messages."""
    out = {}
    for s in get(f"{OPENF1}/sessions", {"year": season, "session_name": "Race"},
                 f"of1_sessions_{season}.json", pause=2):
        msgs = get(f"{OPENF1}/race_control", {"session_key": s["session_key"]},
                   f"of1_rc_{s['session_key']}.json", pause=2)
        text = {(m.get("message") or "").upper().strip() for m in msgs}
        # Wording changed in 2026 ("VSC DEPLOYED", "RED FLAG - RACE SUSPENDED" with no flag field).
        out[s["date_start"][:10]] = {
            "sc": int("SAFETY CAR DEPLOYED" in text),
            "vsc": int(bool(text & {"VIRTUAL SAFETY CAR DEPLOYED", "VSC DEPLOYED"})),
            "red": int(any(m.get("flag") == "RED" for m in msgs) or any(t.startswith("RED FLAG") for t in text))}
    return out


def lookup(rc, date):
    """Match on date, allowing +/-1 day (night races cross midnight UTC)."""
    d = datetime.fromisoformat(date)
    for k in (d, d + timedelta(days=1), d - timedelta(days=1)):
        if k.strftime("%Y-%m-%d") in rc:
            return rc[k.strftime("%Y-%m-%d")]
    return {}


if __name__ == "__main__":
    first, last = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (2018, datetime.now().year)
    rows = []
    for s in range(first, last + 1):
        rows += season_rows(s)
        print(f"{s}: {len(rows)} rows so far")
    df = pd.DataFrame(rows)
    races = df.drop_duplicates(["season", "round"])
    wx = {(r.season, r.round): weather(r) for r in races.itertuples()}
    df["rain_mm"] = [wx[k][0] for k in zip(df.season, df["round"])]
    df["air_temp"] = [wx[k][1] for k in zip(df.season, df["round"])]

    rc = {}
    for s in sorted(set(races.season)):
        if s >= OPENF1_FROM:
            rc.update(race_control(s))
    neut = []
    for r in races.itertuples():
        top3 = df[(df.season == r.season) & (df["round"] == r.round)].nsmallest(3, "position").driver
        row = {"season": r.season, "round": r.round, **lookup(rc, r.date)}
        if r.season <= OPENF1_FROM:
            row["slow_laps"] = slow_laps(r, top3)
        neut.append(row)
        print(f"{r.season} R{r.round}: {row}", flush=True)
    df = df.merge(pd.DataFrame(neut), on=["season", "round"], how="left")
    df.sort_values(["season", "round", "position"]).to_csv("data/races.csv", index=False)
    print(f"wrote data/races.csv: {len(df)} rows, {len(races)} races")
