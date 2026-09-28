"""Upcoming Premier League fixtures.

Sources (merged, de-duplicated on home/away within the season, earlier wins for kickoff):
1. Fantasy Premier League fixtures (official kickoff times and gameweek numbers).
2. Understat's season schedule (every unplayed match with a UTC kickoff).
3. football-data.co.uk ``fixtures.csv`` (the coming week's matches, UK local time, with
   Bet365 prices used by the EV engine).

Player availability (tentative lineups) also comes from FPL: injury/suspension status and
chance of playing next round, see :mod:`src.availability`.

Anything already present in the completed-match table is dropped, so a match that
finished an hour ago never shows up as "upcoming" once either source has its result.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config
from . import data_loader as dl
from .align_teams import normalize_team

log = logging.getLogger(__name__)

UK_TZ = ZoneInfo(config.UK_TZ_NAME)
ROUND_GAP = timedelta(days=2, hours=12)  # a longer gap between kickoffs starts a new round
ROUND_MAX_SPAN = timedelta(days=7)


FPL_POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def fpl_team_names(bootstrap: dict) -> dict[int, str]:
    return {t["id"]: normalize_team(t["name"]) for t in bootstrap.get("teams", [])}


def fpl_next_gameweek(bootstrap: dict) -> int | None:
    for event in bootstrap.get("events", []):
        if event.get("is_next"):
            return int(event["id"])
    return None


def _from_fpl(season: int) -> pd.DataFrame:
    bootstrap, fixtures = dl.load_fpl()
    names = fpl_team_names(bootstrap)
    rows = []
    for f in fixtures:
        if f.get("finished") or not f.get("kickoff_time") or f.get("team_h") not in names:
            continue
        rows.append({"kickoff_utc": pd.Timestamp(f["kickoff_time"]), "home": names[f["team_h"]],
                     "away": names[f["team_a"]], "gameweek": f.get("event"), "source": "fpl"})
    out = pd.DataFrame(rows, columns=["kickoff_utc", "home", "away", "gameweek", "source"])
    if out.empty:
        return out
    return out[[config.season_start_year(k) == season for k in out["kickoff_utc"]]]


def fpl_players(bootstrap: dict | None = None) -> pd.DataFrame:
    """Current squads with availability: status a/d/i/s/u/n and chance of playing next round."""
    if bootstrap is None:
        bootstrap, _ = dl.load_fpl()
    names = fpl_team_names(bootstrap)
    rows = []
    for e in bootstrap.get("elements", []):
        if e.get("team") not in names:
            continue
        chance = e.get("chance_of_playing_next_round")
        rows.append({
            "fpl_id": e["id"], "team": names[e["team"]], "web_name": e.get("web_name", ""),
            "name": f"{e.get('first_name', '')} {e.get('second_name', '')}".strip(),
            "position": FPL_POSITIONS.get(e.get("element_type"), "?"), "status": e.get("status", "a"),
            "chance": float(chance) if chance is not None else np.nan, "news": e.get("news", ""),
            "minutes": e.get("minutes", 0),
        })
    return pd.DataFrame(rows)


def _from_understat(season: int) -> pd.DataFrame:
    us = dl.load_understat(season)
    if us.empty:
        return pd.DataFrame(columns=["kickoff_utc", "home", "away", "source"])
    todo = us[~us["is_result"]]
    return todo[["kickoff_utc", "home", "away"]].assign(source="understat")


def _from_fixtures_csv(season: int) -> pd.DataFrame:
    path = dl.fixtures_csv_path()
    empty = pd.DataFrame(columns=["kickoff_utc", "home", "away", "source"])
    if not path.exists():
        return empty
    try:
        raw = dl._read_csv_bytes(path)
    except Exception as exc:  # malformed file should not stop predictions
        log.warning("Could not parse %s: %s", path.name, exc)
        return empty
    raw = raw[raw.get("Div", pd.Series(dtype=str)).astype(str).str.strip() == config.PREMIER_LEAGUE]
    if raw.empty:
        return empty
    raw = raw.reset_index(drop=True)
    dates = dl._parse_dates(raw["Date"].astype(str).str.strip())
    times = raw["Time"].astype(str).str.strip() if "Time" in raw else pd.Series("15:00", index=raw.index)
    local = pd.to_datetime(dates.dt.strftime("%Y-%m-%d") + " " + times.where(times.str.match(r"^\d{1,2}:\d{2}$"), "15:00"),
                           errors="coerce")
    kickoff = local.dt.tz_localize(UK_TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    out = pd.DataFrame({
        "kickoff_utc": kickoff,
        "home": [normalize_team(t) for t in raw["HomeTeam"]],
        "away": [normalize_team(t) for t in raw["AwayTeam"]],
        "source": "football-data",
    })
    for src, dst in (("B365H", "b365_h"), ("B365D", "b365_d"), ("B365A", "b365_a"),
                     ("B365>2.5", "b365_over25"), ("B365<2.5", "b365_under25")):
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else np.nan
    out = out.dropna(subset=["kickoff_utc"])
    return out[[config.season_start_year(k) == season for k in out["kickoff_utc"]]]


def all_unplayed(matches: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """Every scheduled but not-yet-completed match of the season."""
    season = season or config.season_start_year()
    fpl = _from_fpl(season)
    csv = _from_fixtures_csv(season)
    frames = [f for f in (fpl, _from_understat(season), csv) if not f.empty]
    if not frames:
        log.warning("No fixture source available for %s; run --sync", config.season_label(season))
        return pd.DataFrame(columns=["kickoff_utc", "date", "season", "home", "away", "source", "gameweek"])
    fixtures = pd.concat([f.drop(columns=[c for c in f.columns if c.startswith("b365") or c == "gameweek"])
                          for f in frames], ignore_index=True)
    fixtures["kickoff_utc"] = pd.to_datetime(fixtures["kickoff_utc"], utc=True)
    # Prefer FPL's official kickoff, then Understat, then football-data.
    fixtures["_rank"] = fixtures["source"].map({"fpl": 0, "understat": 1, "football-data": 2})
    fixtures = fixtures.sort_values("_rank").drop_duplicates(["home", "away"], keep="first").drop(columns="_rank")
    gw = fpl[["home", "away", "gameweek"]] if not fpl.empty else pd.DataFrame(columns=["home", "away", "gameweek"])
    fixtures = fixtures.merge(gw, on=["home", "away"], how="left")
    odds_cols = [c for c in csv.columns if c.startswith("b365")]
    if odds_cols:
        fixtures = fixtures.merge(csv[["home", "away"] + odds_cols], on=["home", "away"], how="left")

    done = matches[matches["season"] == season]
    played = set(zip(done["home"], done["away"]))
    fixtures = fixtures[[(h, a) not in played for h, a in zip(fixtures["home"], fixtures["away"])]]
    fixtures["date"] = fixtures["kickoff_utc"].dt.tz_convert(UK_TZ).dt.tz_localize(None).dt.normalize()
    fixtures["season"] = season
    return fixtures.sort_values(["kickoff_utc", "home"]).reset_index(drop=True)


def upcoming(matches: pd.DataFrame, now: datetime | None = None, season: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(future fixtures, past-kickoff fixtures still awaiting a result or postponed)."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    fixtures = all_unplayed(matches, season)
    future = fixtures[fixtures["kickoff_utc"] > now].reset_index(drop=True)
    pending = fixtures[fixtures["kickoff_utc"] <= now].reset_index(drop=True)
    if not pending.empty:
        log.info("%d fixture(s) kicked off already but have no result yet (in play, awaiting sync or postponed): %s",
                 len(pending), ", ".join(f"{h} v {a}" for h, a in zip(pending["home"], pending["away"])))
    return future, pending


def next_round(future: pd.DataFrame) -> pd.DataFrame:
    """The next gameweek: FPL's gameweek number when known, otherwise a date cluster.

    Without gameweek numbers a round ends at the first long gap between kickoffs, when a
    club would appear twice, or after seven days.
    """
    if future.empty:
        return future
    if "gameweek" in future and pd.notna(future["gameweek"].iloc[0]):
        gw = future["gameweek"].iloc[0]
        return future[future["gameweek"] == gw].reset_index(drop=True)
    rows, seen = [], set()
    first = prev = future["kickoff_utc"].iloc[0]
    for idx, row in future.iterrows():
        kickoff = row["kickoff_utc"]
        if rows and (kickoff - prev > ROUND_GAP or kickoff - first > ROUND_MAX_SPAN):
            break
        if row["home"] in seen or row["away"] in seen:
            break
        rows.append(idx)
        seen.update((row["home"], row["away"]))
        prev = kickoff
    return future.loc[rows].reset_index(drop=True)


def within_days(future: pd.DataFrame, days: int, now: datetime | None = None) -> pd.DataFrame:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    return future[future["kickoff_utc"] <= now + pd.Timedelta(days=days)].reset_index(drop=True)


def round_number(matches: pd.DataFrame, fixtures: pd.DataFrame, season: int) -> int | None:
    """Matchweek number: FPL's gameweek, else games already played by the clubs involved + 1."""
    if fixtures.empty:
        return None
    if "gameweek" in fixtures and fixtures["gameweek"].notna().any():
        return int(fixtures["gameweek"].dropna().mode().iloc[0])
    done = matches[matches["season"] == season]
    played = pd.concat([done["home"], done["away"]]).value_counts()
    counts = [played.get(t, 0) for t in pd.concat([fixtures["home"], fixtures["away"]])]
    return int(np.median(counts)) + 1
