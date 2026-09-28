"""Download, cache and merge match data.

Sources
-------
* football-data.co.uk ``E0.csv`` per season: results, shots, shots on target, odds.
  The current season's file is re-downloaded on every sync (it updates after each
  round); completed seasons are downloaded once and cached.
* football-data.co.uk ``E1.csv`` (Championship): used only to build priors for
  newly promoted clubs.
* Understat league JSON: per-match xG, per-team npxG / PPDA / deep completions, and
  per-player xG/xA (used to find each club's key attackers). Understat usually publishes
  results before football-data refreshes its CSV, so completed matches that exist only
  on Understat are merged in to keep rolling features fresh.
* Fantasy Premier League API: gameweek numbers, kickoff times and player availability
  (injuries, suspensions, doubts) for the availability modifier.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config
from .align_teams import normalize_team
from .utils import DataSourceError, atomic_write_bytes, fetch

log = logging.getLogger(__name__)

UK_TZ = ZoneInfo(config.UK_TZ_NAME)
MAX_ARCHIVES_PER_FILE = 30

MATCH_COLUMNS = [
    "match_key", "season", "date", "kickoff_utc", "home", "away",
    "fthg", "ftag", "ftr", "hs", "as", "hst", "ast",
    "home_xg", "away_xg", "home_npxg", "away_npxg",
    "home_ppda_att", "home_ppda_def", "away_ppda_att", "away_ppda_def", "home_deep", "away_deep",
    "odds_h", "odds_d", "odds_a",                       # market average (closing if available): benchmark
    "b365_h", "b365_d", "b365_a", "b365_over25", "b365_under25",  # Bet365 pre-match: EV engine
    "source",
]


# --------------------------------------------------------------------------- paths
def football_data_path(start_year: int, division: str = config.PREMIER_LEAGUE) -> Path:
    return config.FOOTBALL_DATA_DIR / f"{division}_{config.season_code(start_year)}.csv"


def understat_path(start_year: int) -> Path:
    return config.UNDERSTAT_DIR / f"EPL_{start_year}.json"


def fixtures_csv_path() -> Path:
    return config.FOOTBALL_DATA_DIR / "fixtures.csv"


# ------------------------------------------------------------------ sync metadata
def _load_meta() -> dict:
    try:
        return json.loads(config.SYNC_META_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_meta(meta: dict) -> None:
    atomic_write_bytes(config.SYNC_META_PATH, json.dumps(meta, indent=2, sort_keys=True).encode())


def last_sync_time() -> datetime | None:
    stamps = [v.get("synced_at") for v in _load_meta().values() if v.get("synced_at")]
    return max(datetime.fromisoformat(s) for s in stamps) if stamps else None


@dataclass
class SourceResult:
    name: str
    status: str  # downloaded | updated | unchanged | cached | failed
    detail: str = ""


@dataclass
class SyncReport:
    results: list[SourceResult] = field(default_factory=list)

    def add(self, result: SourceResult) -> None:
        self.results.append(result)
        level = logging.WARNING if result.status == "failed" else logging.DEBUG
        log.log(level, "%-22s %-10s %s", result.name, result.status, result.detail)

    @property
    def failures(self) -> list[SourceResult]:
        return [r for r in self.results if r.status == "failed"]


def _store(name: str, path: Path, payload: bytes, url: str, *, archive: bool) -> str:
    """Write payload if it changed; archive the previous version of live files."""
    digest = hashlib.sha1(payload).hexdigest()
    meta = _load_meta()
    previous = meta.get(name, {})
    status = "downloaded"
    if path.exists():
        old = path.read_bytes()
        if hashlib.sha1(old).hexdigest() == digest:
            status = "unchanged"
        else:
            status = "updated"
            if archive:
                _archive(path)
    if status != "unchanged":
        atomic_write_bytes(path, payload)
    meta[name] = {
        "url": url,
        "path": str(path.relative_to(config.ROOT)),
        "sha1": digest,
        "bytes": len(payload),
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "changed_at": previous.get("changed_at") if status == "unchanged" else
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_meta(meta)
    return status


def _archive(path: Path) -> None:
    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = config.ARCHIVE_DIR / f"{path.stem}.{stamp}{path.suffix}"
    shutil.copy2(path, target)
    old = sorted(config.ARCHIVE_DIR.glob(f"{path.stem}.*{path.suffix}"))
    for stale in old[:-MAX_ARCHIVES_PER_FILE]:
        stale.unlink(missing_ok=True)
    log.debug("archived %s -> %s", path.name, target.name)


# ------------------------------------------------------------------- downloaders
def _validate_csv(payload: bytes, url: str) -> None:
    head = payload[:400].decode("utf-8-sig", errors="ignore")
    if "HomeTeam" not in head or "<html" in head.lower():
        raise DataSourceError(f"{url} did not return a football-data CSV")


def download_football_data(start_year: int, division: str = config.PREMIER_LEAGUE,
                           *, live: bool = False, force: bool = False) -> SourceResult:
    name = f"{division} {config.season_label(start_year)}"
    path = football_data_path(start_year, division)
    if path.exists() and not (live or force):
        return SourceResult(name, "cached", path.name)
    url = config.FOOTBALL_DATA_URL.format(code=config.season_code(start_year), division=division)
    try:
        payload = fetch(url).content
        _validate_csv(payload, url)
    except DataSourceError as exc:
        return SourceResult(name, "failed", f"{exc} (cached copy {'kept' if path.exists() else 'missing'})")
    status = _store(f"football_data:{division}:{start_year}", path, payload, url, archive=live)
    rows = max(payload.count(b"\n") - 1, 0)
    return SourceResult(name, status, f"{rows} matches")


def download_fixtures_csv() -> SourceResult:
    path = fixtures_csv_path()
    url = config.FOOTBALL_DATA_FIXTURES_URL
    try:
        payload = fetch(url).content
        _validate_csv(payload, url)
    except DataSourceError as exc:
        return SourceResult("fixtures.csv", "failed", str(exc))
    status = _store("football_data:fixtures", path, payload, url, archive=False)
    return SourceResult("fixtures.csv", status, "football-data fixture list")


def _understat_legacy(year: int) -> dict:
    """Fallback for Understat's older page format: data embedded as JSON.parse('...') in <script> tags."""
    from bs4 import BeautifulSoup

    url = config.UNDERSTAT_LEAGUE_PAGE.format(year=year)
    soup = BeautifulSoup(fetch(url).text, "html.parser")
    scripts = [tag.string or tag.get_text() for tag in soup.find_all("script")]
    out = {}
    for key, var in (("dates", "datesData"), ("teams", "teamsData"), ("players", "playersData")):
        pattern = re.compile(rf"var\s+{var}\s*=\s*JSON\.parse\('(.*?)'\)", re.S)
        match = next((m for m in (pattern.search(s or "") for s in scripts) if m), None)
        if not match:
            if key == "players":
                continue  # optional
            raise DataSourceError(f"{url}: could not find {var} in page")
        out[key] = json.loads(match.group(1).encode("utf-8").decode("unicode_escape"))
    return out


def fetch_understat_league(year: int) -> dict:
    url = config.UNDERSTAT_LEAGUE_API.format(year=year)
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": config.UNDERSTAT_LEAGUE_PAGE.format(year=year),
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    try:
        payload = fetch(url, headers=headers).json()
        if not isinstance(payload, dict) or "dates" not in payload or "teams" not in payload:
            raise DataSourceError(f"{url}: unexpected JSON structure")
        return payload
    except (DataSourceError, ValueError) as exc:
        log.warning("Understat JSON API failed for %s (%s); trying legacy page parser", year, exc)
        return _understat_legacy(year)


def _understat_has_players(start_year: int) -> bool:
    try:
        return bool(json.loads(understat_path(start_year).read_text()).get("players"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def download_understat(start_year: int, *, live: bool = False, force: bool = False) -> SourceResult:
    name = f"Understat {config.season_label(start_year)}"
    path = understat_path(start_year)
    if path.exists() and not (live or force):
        return SourceResult(name, "cached", path.name)
    try:
        data = fetch_understat_league(start_year)
    except DataSourceError as exc:
        return SourceResult(name, "failed", f"{exc} (cached copy {'kept' if path.exists() else 'missing'})")
    keep = {"dates": data["dates"], "teams": data["teams"], "players": data.get("players", [])}
    payload = json.dumps(keep, separators=(",", ":"), sort_keys=True).encode()
    status = _store(f"understat:{start_year}", path, payload, config.UNDERSTAT_LEAGUE_API.format(year=start_year),
                    archive=live)
    played = sum(1 for d in data["dates"] if d.get("isResult"))
    return SourceResult(name, status, f"{played} played / {len(data['dates'])} scheduled")


def download_fpl() -> list[SourceResult]:
    """FPL bootstrap (players, availability, gameweeks) and the fixture list."""
    results = []
    for name, url, filename in (("FPL players", config.FPL_BOOTSTRAP_URL, "bootstrap.json"),
                                ("FPL fixtures", config.FPL_FIXTURES_URL, "fixtures.json")):
        path = config.FPL_DIR / filename
        try:
            data = fetch(url).json()
        except (DataSourceError, ValueError) as exc:
            results.append(SourceResult(name, "failed", f"{exc} (cached copy {'kept' if path.exists() else 'missing'})"))
            continue
        if name == "FPL players":
            data = {k: data[k] for k in ("events", "teams", "elements", "element_types") if k in data}
            detail = f"{len(data.get('elements', []))} players"
        else:
            detail = f"{len(data)} fixtures"
        payload = json.dumps(data, separators=(",", ":"), sort_keys=True).encode()
        status = _store(f"fpl:{filename}", path, payload, url, archive=False)
        results.append(SourceResult(name, status, detail))
    return results


def sync_all(n_history: int = config.DEFAULT_HISTORY_SEASONS, *, refresh_history: bool = False) -> SyncReport:
    """Refresh the live season and fill any gaps in the historical cache."""
    plan = config.season_plan(n_history)
    current = plan["current"]
    report = SyncReport()
    log.info("Syncing data: current season %s, history %s, warm-up %s",
             config.season_label(current), ", ".join(config.season_label(s) for s in plan["history"]),
             ", ".join(config.season_label(s) for s in plan["warmup"]))

    report.add(download_football_data(current, live=True))
    report.add(download_understat(current, live=True))
    report.add(download_fixtures_csv())
    for result in download_fpl():
        report.add(result)
    for season in plan["warmup"] + plan["history"]:
        report.add(download_football_data(season, force=refresh_history))
        # Last season's player table ranks key attackers; older caches may predate it.
        needs_players = season == current - 1 and not _understat_has_players(season)
        report.add(download_understat(season, force=refresh_history or needs_players))
    for season in plan["championship"]:
        # The Championship season that just finished can still be corrected, so refresh it too.
        report.add(download_football_data(season, config.CHAMPIONSHIP,
                                          force=refresh_history or season == current - 1))
    return report


def has_minimum_data(n_history: int = config.DEFAULT_HISTORY_SEASONS) -> bool:
    plan = config.season_plan(n_history)
    return all(football_data_path(s).exists() for s in plan["history"])


# ------------------------------------------------------------------------ parsers
def _read_csv_bytes(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise DataSourceError(f"could not decode {path}")


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(values, format="%d/%m/%y", errors="coerce")
    return parsed.fillna(short)


def _first_available(df: pd.DataFrame, candidates: list[tuple[str, str, str]]) -> pd.DataFrame:
    for cols in candidates:
        if all(c in df.columns for c in cols):
            odds = df[list(cols)].apply(pd.to_numeric, errors="coerce")
            if odds.notna().all(axis=1).mean() > 0.5:
                return odds.set_axis(["odds_h", "odds_d", "odds_a"], axis=1)
    return pd.DataFrame(np.nan, index=df.index, columns=["odds_h", "odds_d", "odds_a"])


def load_football_data(start_year: int, division: str = config.PREMIER_LEAGUE) -> pd.DataFrame:
    path = football_data_path(start_year, division)
    if not path.exists():
        return pd.DataFrame()
    df = _read_csv_bytes(path)
    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    df = df[df["HomeTeam"].astype(str).str.strip() != ""]
    quiet = division != config.PREMIER_LEAGUE
    out = pd.DataFrame({
        "season": start_year,
        "date": _parse_dates(df["Date"].astype(str).str.strip()),
        "time": df["Time"].astype(str).str.strip() if "Time" in df.columns else "",
        "home": [normalize_team(t, warn=not quiet) for t in df["HomeTeam"]],
        "away": [normalize_team(t, warn=not quiet) for t in df["AwayTeam"]],
    })
    for src, dst in (("FTHG", "fthg"), ("FTAG", "ftag"), ("HS", "hs"), ("AS", "as"),
                     ("HST", "hst"), ("AST", "ast")):
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    out["ftr"] = df["FTR"].astype(str).str.strip() if "FTR" in df.columns else np.nan
    # Closing market average is the sharpest benchmark; fall back to pre-match prices.
    odds = _first_available(df, [("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"),
                                 ("BbAvH", "BbAvD", "BbAvA"), ("PSCH", "PSCD", "PSCA"),
                                 ("B365H", "B365D", "B365A")])
    out = pd.concat([out, odds], axis=1)
    for src, dst in (("B365H", "b365_h"), ("B365D", "b365_d"), ("B365A", "b365_a"),
                     ("B365>2.5", "b365_over25"), ("B365<2.5", "b365_under25")):
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    bad_dates = out["date"].isna().sum()
    if bad_dates:
        log.warning("%s: dropped %d rows with unparseable dates", path.name, bad_dates)
    out = out.dropna(subset=["date", "fthg", "ftag"]).reset_index(drop=True)
    return out


def _team_history_index(teams: dict) -> dict[tuple[str, str], dict]:
    index = {}
    for team in teams.values():
        title = team.get("title")
        for row in team.get("history", []):
            index[(title, row.get("date"))] = row
    return index


def _ppda(entry: dict | None, key: str, part: str) -> float:
    try:
        return float(entry[key][part])
    except (TypeError, KeyError, ValueError):
        return np.nan


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_understat(start_year: int) -> pd.DataFrame:
    """All scheduled matches of an Understat season (played and unplayed)."""
    path = understat_path(start_year)
    if not path.exists():
        return pd.DataFrame()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        log.error("Corrupt Understat cache %s: %s", path, exc)
        return pd.DataFrame()
    history = _team_history_index(data.get("teams", {}))
    rows = []
    for m in data.get("dates", []):
        home_title, away_title = m["h"]["title"], m["a"]["title"]
        hist_h = history.get((home_title, m["datetime"]))
        hist_a = history.get((away_title, m["datetime"]))
        kickoff = pd.Timestamp(m["datetime"], tz="UTC")
        rows.append({
            "understat_id": m.get("id"),
            "season": start_year,
            "kickoff_utc": kickoff,
            "date": kickoff.tz_convert(UK_TZ).tz_localize(None).normalize(),
            "home": normalize_team(home_title),
            "away": normalize_team(away_title),
            "is_result": bool(m.get("isResult")),
            "us_home_goals": _num(m.get("goals", {}).get("h")),
            "us_away_goals": _num(m.get("goals", {}).get("a")),
            "home_xg": _num(m.get("xG", {}).get("h")),
            "away_xg": _num(m.get("xG", {}).get("a")),
            "home_npxg": _num(hist_h.get("npxG")) if hist_h else np.nan,
            "away_npxg": _num(hist_a.get("npxG")) if hist_a else np.nan,
            "home_ppda_att": _ppda(hist_h, "ppda", "att"),
            "home_ppda_def": _ppda(hist_h, "ppda", "def"),
            "away_ppda_att": _ppda(hist_a, "ppda", "att"),
            "away_ppda_def": _ppda(hist_a, "ppda", "def"),
            "home_deep": _num(hist_h.get("deep")) if hist_h else np.nan,
            "away_deep": _num(hist_a.get("deep")) if hist_a else np.nan,
        })
    return pd.DataFrame(rows)


def _result_code(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    return pd.Series(np.select([home_goals > away_goals, home_goals < away_goals], ["H", "A"], "D"),
                     index=home_goals.index)


def build_match_table(n_history: int = config.DEFAULT_HISTORY_SEASONS, *, save: bool = True) -> pd.DataFrame:
    """One row per completed EPL match, football-data and Understat merged."""
    plan = config.season_plan(n_history)
    frames = []
    for season in plan["all_pl"]:
        fd = load_football_data(season)
        us = load_understat(season)
        if fd.empty and us.empty:
            log.warning("No data at all for %s", config.season_label(season))
            continue
        if not us.empty:
            us = us[us["is_result"]].drop(columns=["is_result"])
        if fd.empty:
            merged = us.assign(source="understat")
        elif us.empty:
            merged = fd.assign(source="football-data")
        else:
            merged = fd.merge(us.drop(columns=["season"]), on=["home", "away"], how="outer",
                              suffixes=("", "_us"), indicator=True)
            merged["source"] = merged["_merge"].map(
                {"both": "both", "left_only": "football-data", "right_only": "understat"}).astype(str)
            merged["date"] = merged["date"].fillna(merged["date_us"])
            merged["season"] = season
            merged = merged.drop(columns=["_merge", "date_us"])
            _check_goal_consistency(merged, season)
            if (merged["source"] == "football-data").any():
                log.debug("%s: %d matches missing on Understat (no xG)", config.season_label(season),
                          (merged["source"] == "football-data").sum())
        # Understat-only matches (football-data CSV not refreshed yet) get Understat goals.
        if "us_home_goals" in merged.columns:
            merged["fthg"] = merged.get("fthg", pd.Series(np.nan, index=merged.index)).fillna(merged["us_home_goals"])
            merged["ftag"] = merged.get("ftag", pd.Series(np.nan, index=merged.index)).fillna(merged["us_away_goals"])
        frames.append(merged)
        fresh = (merged["source"] == "understat").sum()
        if fresh and season == plan["current"]:
            log.info("%d completed %s matches taken from Understat ahead of football-data's CSV",
                     fresh, config.season_label(season))

    if not frames:
        raise DataSourceError("no match data available; run with --sync first")
    matches = pd.concat(frames, ignore_index=True)
    for col in MATCH_COLUMNS:
        if col not in matches.columns:
            matches[col] = np.nan
    matches = matches.dropna(subset=["fthg", "ftag"])
    matches["ftr"] = _result_code(matches["fthg"], matches["ftag"])
    matches["date"] = pd.to_datetime(matches["date"]).dt.normalize()
    matches["season"] = matches["season"].astype(int)
    matches = matches.sort_values(["date", "home"]).reset_index(drop=True)
    matches["match_key"] = (matches["season"].astype(str) + ":" + matches["home"] + ":" + matches["away"])
    dupes = matches["match_key"].duplicated()
    if dupes.any():
        log.warning("Dropping %d duplicated fixtures", dupes.sum())
        matches = matches[~dupes]
    matches = matches[MATCH_COLUMNS].reset_index(drop=True)

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        matches.to_csv(config.PROCESSED_DIR / "matches.csv", index=False)
    return matches


def _check_goal_consistency(merged: pd.DataFrame, season: int) -> None:
    both = merged[merged["source"] == "both"]
    if both.empty or "us_home_goals" not in both:
        return
    mismatch = both[(both["fthg"] != both["us_home_goals"]) | (both["ftag"] != both["us_away_goals"])]
    for _, row in mismatch.iterrows():
        log.warning("%s %s v %s: football-data %s-%s vs Understat %s-%s (keeping football-data)",
                    config.season_label(season), row["home"], row["away"], int(row["fthg"]),
                    int(row["ftag"]), row["us_home_goals"], row["us_away_goals"])


def load_championship_profiles(seasons: list[int]) -> pd.DataFrame:
    """Per-team, per-game Championship stats (season = the Championship season)."""
    frames = []
    for season in seasons:
        df = load_football_data(season, config.CHAMPIONSHIP)
        if df.empty:
            continue
        home = pd.DataFrame({"team": df["home"], "gf": df["fthg"], "ga": df["ftag"], "sh_f": df["hs"],
                             "sh_a": df["as"], "sot_f": df["hst"], "sot_a": df["ast"],
                             "pts": np.select([df["fthg"] > df["ftag"], df["fthg"] == df["ftag"]], [3, 1], 0)})
        away = pd.DataFrame({"team": df["away"], "gf": df["ftag"], "ga": df["fthg"], "sh_f": df["as"],
                             "sh_a": df["hs"], "sot_f": df["ast"], "sot_a": df["hst"],
                             "pts": np.select([df["ftag"] > df["fthg"], df["fthg"] == df["ftag"]], [3, 1], 0)})
        per_game = pd.concat([home, away]).groupby("team").mean(numeric_only=True)
        per_game["games"] = pd.concat([home, away]).groupby("team").size()
        per_game["season"] = season
        frames.append(per_game.reset_index())
    if not frames:
        return pd.DataFrame(columns=["team", "season", "games", "gf", "ga", "sh_f", "sh_a", "sot_f", "sot_a", "pts"])
    return pd.concat(frames, ignore_index=True)


def load_understat_players(start_year: int) -> pd.DataFrame:
    """Season player table: xG, npxG, xA, minutes, position, canonical team."""
    path = understat_path(start_year)
    try:
        players = json.loads(path.read_text()).get("players", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()
    if not players:
        return pd.DataFrame()
    df = pd.DataFrame(players)
    for col in ("games", "time", "goals", "xG", "npxG", "xA", "assists", "shots", "key_passes"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Mid-season transfers are listed as "Team A,Team B": credit the latest club.
    df["team"] = [normalize_team(t.split(",")[-1].strip()) for t in df["team_title"].astype(str)]
    df["season"] = start_year
    return df[["id", "player_name", "team", "season", "position", "games", "time", "goals", "npxG", "xG", "xA",
               "assists", "shots", "key_passes"]]


def load_fpl() -> tuple[dict, list]:
    """Cached FPL bootstrap and fixtures ({} / [] when never synced)."""
    out = []
    for filename, empty in (("bootstrap.json", {}), ("fixtures.json", [])):
        try:
            out.append(json.loads((config.FPL_DIR / filename).read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            out.append(empty)
    return out[0], out[1]
