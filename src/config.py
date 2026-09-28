"""Project-wide paths, data-source URLs and season helpers."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARCHIVE_DIR = RAW_DIR / "archive"
FOOTBALL_DATA_DIR = RAW_DIR / "football_data"
UNDERSTAT_DIR = RAW_DIR / "understat"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
SYNC_META_PATH = RAW_DIR / "sync_meta.json"
FPL_DIR = RAW_DIR / "fpl"
AVAILABILITY_OVERRIDES_PATH = DATA_DIR / "availability_overrides.json"
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"

FOOTBALL_DATA_URL = "https://football-data.co.uk/mmz4281/{code}/{division}.csv"
FOOTBALL_DATA_FIXTURES_URL = "https://football-data.co.uk/fixtures.csv"
UNDERSTAT_BASE_URL = "https://understat.com"
UNDERSTAT_LEAGUE_API = UNDERSTAT_BASE_URL + "/getLeagueData/EPL/{year}"
UNDERSTAT_LEAGUE_PAGE = UNDERSTAT_BASE_URL + "/league/EPL/{year}"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

PREMIER_LEAGUE = "E0"
CHAMPIONSHIP = "E1"

# Completed seasons used for training (the spec allows 3-5).
DEFAULT_HISTORY_SEASONS = 5
MIN_HISTORY_SEASONS = 3
MAX_HISTORY_SEASONS = 5
# Extra season loaded before the first training season purely to warm up rolling
# windows and Elo ratings. It is never used as a training row.
WARMUP_SEASONS = 1

UK_TZ_NAME = "Europe/London"


def season_start_year(day: date | datetime | None = None) -> int:
    """EPL seasons run Aug-May, so anything from July onwards belongs to the new season."""
    day = day or date.today()
    return day.year if day.month >= 7 else day.year - 1


def season_code(start_year: int) -> str:
    """football-data.co.uk code, e.g. 2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """Human label, e.g. 2025 -> '2025-26'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_plan(n_history: int = DEFAULT_HISTORY_SEASONS, today: date | None = None) -> dict:
    """Which seasons to load for a given number of completed training seasons."""
    if not MIN_HISTORY_SEASONS <= n_history <= MAX_HISTORY_SEASONS:
        raise ValueError(f"history seasons must be between {MIN_HISTORY_SEASONS} and {MAX_HISTORY_SEASONS}")
    current = season_start_year(today)
    history = list(range(current - n_history, current))
    warmup = list(range(history[0] - WARMUP_SEASONS, history[0]))
    return {
        "current": current,
        "history": history,
        "warmup": warmup,
        "all_pl": warmup + history + [current],
        # Championship season before every loaded PL season (promoted-team priors).
        "championship": [s - 1 for s in warmup + history + [current]],
    }
