"""Canonical team-name normalisation across football-data.co.uk, Understat and user input.

Every source spells clubs differently ("Man United", "Manchester United", "Man Utd").
All modules call :func:`normalize_team` so that joins, rolling windows and the CLI
all speak the same names. Championship clubs are included so promoted/relegated
teams keep one identity as they move between divisions.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from functools import lru_cache
from typing import Iterable

log = logging.getLogger(__name__)

# canonical name -> aliases (matched after normalisation, so case/punctuation don't matter)
CANONICAL_TEAMS: dict[str, tuple[str, ...]] = {
    "Arsenal": ("Arsenal FC", "The Gunners", "Gunners", "ARS"),
    "Aston Villa": ("Villa", "AVL"),
    "Bournemouth": ("AFC Bournemouth", "Cherries", "BOU"),
    "Brentford": ("Bees", "BRE"),
    "Brighton": ("Brighton & Hove Albion", "Brighton and Hove Albion", "Brighton Hove", "BHA"),
    "Burnley": ("Clarets", "BUR"),
    "Chelsea": ("CHE",),
    "Coventry City": ("Coventry", "COV"),
    "Crystal Palace": ("Palace", "C Palace", "CRY"),
    "Everton": ("Toffees", "EVE"),
    "Fulham": ("FUL",),
    "Hull City": ("Hull", "HUL"),
    "Ipswich Town": ("Ipswich", "IPS"),
    "Leeds United": ("Leeds", "Leeds Utd", "LEE"),
    "Leicester City": ("Leicester", "LEI"),
    "Liverpool": ("LFC", "LIV"),
    "Luton Town": ("Luton", "LUT"),
    "Manchester City": ("Man City", "Man. City", "Manchester C", "MCFC", "MCI", "Mancity"),
    "Manchester United": (
        "Man United", "Man Utd", "Man. United", "Man. Utd", "Manchester Utd", "Man U", "MUFC", "MUN",
    ),
    "Middlesbrough": ("Boro", "Middlesboro"),
    "Newcastle United": ("Newcastle", "Newcastle Utd", "NEW"),
    "Norwich City": ("Norwich", "NOR"),
    "Nottingham Forest": ("Nott'm Forest", "Nottm Forest", "Notts Forest", "Nottingham", "Forest", "NFO"),
    "Sheffield United": ("Sheffield Utd", "Sheff Utd", "Sheffield U", "SHU"),
    "Southampton": ("Saints", "SOU"),
    "Sunderland": ("SUN",),
    "Tottenham": ("Tottenham Hotspur", "Spurs", "TOT"),
    "Watford": ("WAT",),
    "West Bromwich Albion": ("West Brom", "WBA", "West Bromwich"),
    "West Ham United": ("West Ham", "Hammers", "WHU"),
    "Wolverhampton Wanderers": ("Wolves", "Wolverhampton", "WOL"),
}

# Recent Championship clubs: only needed so their names normalise quietly.
CHAMPIONSHIP_ONLY: dict[str, tuple[str, ...]] = {
    "Barnsley": (),
    "Birmingham City": ("Birmingham",),
    "Blackburn Rovers": ("Blackburn",),
    "Blackpool": (),
    "Bristol City": (),
    "Cardiff City": ("Cardiff",),
    "Charlton Athletic": ("Charlton",),
    "Derby County": ("Derby",),
    "Huddersfield Town": ("Huddersfield",),
    "Millwall": (),
    "Oxford United": ("Oxford",),
    "Peterborough United": ("Peterboro", "Peterborough"),
    "Plymouth Argyle": ("Plymouth",),
    "Portsmouth": (),
    "Preston North End": ("Preston",),
    "Queens Park Rangers": ("QPR",),
    "Reading": (),
    "Rotherham United": ("Rotherham",),
    "Sheffield Wednesday": ("Sheffield Weds", "Sheff Wed", "Sheffield W"),
    "Stoke City": ("Stoke",),
    "Swansea City": ("Swansea",),
    "Wigan Athletic": ("Wigan",),
    "Wrexham": (),
}
PREMIER_LEAGUE_CLUBS = tuple(CANONICAL_TEAMS)
CANONICAL_TEAMS.update(CHAMPIONSHIP_ONLY)

_DROP_TOKENS = {"fc", "afc"}


class UnknownTeamError(ValueError):
    def __init__(self, name: str, suggestions: list[str]):
        self.name = name
        self.suggestions = suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"Unknown team '{name}'.{hint}")


def _key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    tokens = [t for t in text.split() if t not in _DROP_TOKENS]
    return " ".join(tokens)


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, aliases in CANONICAL_TEAMS.items():
        for alias in (canonical, *aliases):
            key = _key(alias)
            if key in index and index[key] != canonical:
                raise ValueError(f"alias '{alias}' maps to both {index[key]} and {canonical}")
            index[key] = canonical
    return index


_warned: set[str] = set()


@lru_cache(maxsize=4096)
def _lookup(name: str) -> str | None:
    index = _alias_index()
    key = _key(name)
    if key in index:
        return index[key]
    # Tolerate small spelling slips ("Totenham") but not wild guesses.
    match = difflib.get_close_matches(key, list(index), n=1, cutoff=0.88)
    return index[match[0]] if match else None


def normalize_team(name: str, *, strict: bool = False, warn: bool = True) -> str:
    """Map any source spelling to the canonical name.

    Unknown names are returned stripped (non-strict) so a new club never crashes the
    pipeline, or raise UnknownTeamError (strict, used for CLI input).
    """
    if name is None or (isinstance(name, float) and name != name):
        raise ValueError("team name is missing")
    raw = str(name).strip()
    found = _lookup(raw)
    if found:
        return found
    if strict:
        raise UnknownTeamError(raw, suggest_teams(raw))
    if warn and raw not in _warned:
        _warned.add(raw)
        log.warning("No canonical mapping for team '%s'; using it as-is", raw)
    return raw


def suggest_teams(name: str, candidates: Iterable[str] | None = None, n: int = 3) -> list[str]:
    """Closest canonical names, searching aliases too ("Utd" -> Manchester United, Leeds United...)."""
    pool = set(candidates) if candidates is not None else set(PREMIER_LEAGUE_CLUBS)
    alias_keys = {k: c for k, c in _alias_index().items() if c in pool}
    key = _key(name)
    tokens = sorted({c for k, c in alias_keys.items() if key and key in k.split()})
    close = [alias_keys[k] for k in difflib.get_close_matches(key, list(alias_keys), n=n * 3, cutoff=0.6)]
    contains = sorted({c for k, c in alias_keys.items() if key and key in k})
    return list(dict.fromkeys(tokens + close + contains))[:n]


def resolve_user_team(name: str, known_teams: Iterable[str]) -> str:
    """Strict resolution for CLI input, restricted to teams present in the data."""
    known = sorted(set(known_teams))
    canonical = normalize_team(name, strict=True)
    if canonical not in known:
        raise UnknownTeamError(name, suggest_teams(name, known))
    return canonical


def normalize_series(values) -> list[str]:
    return [normalize_team(v) for v in values]


# Compact names for terminal tables.
SHORT_NAMES = {
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def short_name(team: str) -> str:
    return SHORT_NAMES.get(team, team)
