"""Leakage-free feature engineering shared by training and live inference.

How leakage is prevented
------------------------
1. Every completed match produces a *post-match state* per team: rolling sums and
   counts over that team's last N matches **including** the match.
2. A fixture (historical training row or upcoming match) reads each team's most
   recent state with ``date < kickoff date`` (``merge_asof`` with
   ``allow_exact_matches=False``). The fixture's own result, and anything played
   on or after its date, can never reach its features.
3. Elo ratings are read before the match updates them.
4. Priors for promoted teams only use seasons that finished before the fixture's
   season (plus the promoted club's own completed Championship season).

The exact same :meth:`FeatureBuilder.build` call produces training rows and live rows,
so the model never sees features computed differently from the ones it was trained on.

Newly promoted teams
--------------------
Rolling windows reset when a club returns to the Premier League (a new "spell").
While a window is not yet full, each metric is blended with a prior:

    feature = (n / N) * rolling_mean + (1 - n / N) * prior

The prior for a promoted club is the average of promoted sides' first 10 PL matches
in earlier seasons, scaled by the club's Championship profile relative to other
promoted clubs (a shrunk Championship->PL conversion factor). Clubs with no known
history fall back to league averages, so missing data never crashes the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Per-team, per-match quantities that get rolled.
# PPDA bookkeeping (Understat counts passes/defensive actions in the pressing zone):
#   ppda_att = opponent passes allowed, ppda_def = own defensive actions  -> PPDA (pressing)
#   pass_f   = own passes,              opp_def  = opponent's actions      -> press resistance
#   pass_f / (pass_f + ppda_att) is a possession-share proxy.
STATS = ["gf", "ga", "sot_f", "sot_a", "sh_f", "sh_a", "xg_f", "xg_a", "npxg_f", "npxg_a",
         "xgd", "npxgd", "fin_var", "pts", "ppda_att", "ppda_def", "pass_f", "opp_def", "deep_f", "deep_a"]
WINDOWS = (3, 5, 10, 19)
VENUE_STATS = ["gf", "ga", "xgd", "pts"]
VENUE_WINDOWS = (5, 10)
REST_CAP_DAYS = 14
PROMOTED_PRIOR_GAMES = 10
CHAMPIONSHIP_SHRINK = 0.5  # exponent on the Championship strength ratio (0 = ignore, 1 = full)

ELO_INIT = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 55.0
ELO_SEASON_REGRESSION = 0.20

CLASSES = ("H", "D", "A")
TARGET_MAP = {c: i for i, c in enumerate(CLASSES)}


# ------------------------------------------------------------------ long format
def team_match_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per completed match, one from each team's perspective."""
    base = matches[["match_key", "season", "date"]]
    home = base.assign(
        team=matches["home"], opp=matches["away"], is_home=1,
        gf=matches["fthg"], ga=matches["ftag"], sot_f=matches["hst"], sot_a=matches["ast"],
        sh_f=matches["hs"], sh_a=matches["as"], xg_f=matches["home_xg"], xg_a=matches["away_xg"],
        npxg_f=matches["home_npxg"], npxg_a=matches["away_npxg"],
        ppda_att=matches["home_ppda_att"], ppda_def=matches["home_ppda_def"],
        pass_f=matches["away_ppda_att"], opp_def=matches["away_ppda_def"],
        deep_f=matches["home_deep"], deep_a=matches["away_deep"],
    )
    away = base.assign(
        team=matches["away"], opp=matches["home"], is_home=0,
        gf=matches["ftag"], ga=matches["fthg"], sot_f=matches["ast"], sot_a=matches["hst"],
        sh_f=matches["as"], sh_a=matches["hs"], xg_f=matches["away_xg"], xg_a=matches["home_xg"],
        npxg_f=matches["away_npxg"], npxg_a=matches["home_npxg"],
        ppda_att=matches["away_ppda_att"], ppda_def=matches["away_ppda_def"],
        pass_f=matches["home_ppda_att"], opp_def=matches["home_ppda_def"],
        deep_f=matches["away_deep"], deep_a=matches["home_deep"],
    )
    tm = pd.concat([home, away], ignore_index=True)
    for col in STATS:
        if col in tm:
            tm[col] = pd.to_numeric(tm[col], errors="coerce")
    tm["pts"] = np.select([tm["gf"] > tm["ga"], tm["gf"] == tm["ga"]], [3.0, 1.0], 0.0)
    tm["xgd"] = tm["xg_f"] - tm["xg_a"]
    tm["npxgd"] = tm["npxg_f"] - tm["npxg_a"]
    tm["fin_var"] = tm["gf"] - tm["xg_f"]
    tm["date"] = to_day(tm["date"])
    return tm.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)


def to_day(values) -> pd.Series:
    """Midnight timestamps in one fixed unit (pandas 3 infers s/us/ns from the input)."""
    return pd.to_datetime(values).dt.normalize().astype("datetime64[ns]")


def season_of(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return pd.Series(np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1), index=dates.index)


# ------------------------------------------------------------------ membership
class Membership:
    """Which clubs were in the Premier League in which season, and their current spell."""

    def __init__(self, team_seasons: set[tuple[str, int]], championship: pd.DataFrame | None = None):
        self.members = set(team_seasons)
        self.first_season = min(s for _, s in self.members) if self.members else None
        self.championship_members: dict[int, set[str]] = {}
        if championship is not None and not championship.empty:
            for season, grp in championship.groupby("season"):
                self.championship_members[int(season)] = set(grp["team"])

    def add(self, pairs) -> None:
        self.members.update(pairs)

    def spell_start(self, team: str, season: int) -> int:
        s = season
        while (team, s - 1) in self.members:
            s -= 1
        return s

    def is_promoted(self, team: str, season: int) -> bool:
        """True if the club came up from the Championship for `season`."""
        if (team, season - 1) in self.members:
            return False
        if self.first_season is not None and season - 1 >= self.first_season:
            return True
        return team in self.championship_members.get(season - 1, set())


# ----------------------------------------------------------------------- priors
class Priors:
    def __init__(self, tm: pd.DataFrame, membership: Membership, championship: pd.DataFrame | None):
        self.tm = tm
        self.membership = membership
        self.championship = championship if championship is not None else pd.DataFrame()
        self._cache: dict = {}
        tm = tm.copy()
        tm["spell"] = [membership.spell_start(t, s) for t, s in zip(tm["team"], tm["season"])]
        tm["spell_game"] = tm.groupby(["team", "spell"]).cumcount()
        tm["promoted_start"] = [
            sp == s and membership.is_promoted(t, s) for t, s, sp in zip(tm["team"], tm["season"], tm["spell"])
        ]
        self._promoted_rows = tm[tm["promoted_start"] & (tm["spell_game"] < PROMOTED_PRIOR_GAMES)]

    @staticmethod
    def _summarise(rows: pd.DataFrame) -> pd.Series:
        out = rows[STATS].mean(numeric_only=True)
        for name, (num, den, scale) in RATIOS.items():
            top, bottom = rows[num].sum(), sum(rows[d].sum() for d in den)
            out[name] = scale * top / bottom if bottom > 0 else np.nan
        return out

    def league(self, season: int, venue: int | None = None) -> pd.Series:
        key = ("league", season, venue)
        if key not in self._cache:
            rows = self.tm[self.tm["season"] < season]
            if rows.empty:  # first loaded (warm-up) season only
                rows = self.tm[self.tm["season"] == season]
            if rows.empty:
                rows = self.tm
            if venue is not None:
                rows = rows[rows["is_home"] == venue]
            self._cache[key] = self._summarise(rows)
        return self._cache[key]

    def promoted_base(self, season: int) -> pd.Series:
        key = ("promoted", season)
        if key not in self._cache:
            rows = self._promoted_rows[self._promoted_rows["season"] < season]
            if rows.empty:
                rows = self._promoted_rows  # warm-up season: no earlier promoted clubs to learn from
            if rows.empty:
                league = self.league(season).copy()
                for col, mult in (("gf", 0.8), ("xg_f", 0.8), ("npxg_f", 0.8), ("sot_f", 0.8), ("sh_f", 0.85),
                                  ("ga", 1.25), ("xg_a", 1.25), ("npxg_a", 1.25), ("sot_a", 1.2),
                                  ("sh_a", 1.15), ("pts", 0.75)):
                    league[col] *= mult
                league["xgd"] = league["xg_f"] - league["xg_a"]
                league["npxgd"] = league["npxg_f"] - league["npxg_a"]
                self._cache[key] = league
            else:
                self._cache[key] = self._summarise(rows)
        return self._cache[key]

    def _championship_ratio(self, team: str, season: int) -> dict[str, float] | None:
        champ = self.championship
        if champ.empty:
            return None
        row = champ[(champ["team"] == team) & (champ["season"] == season - 1)]
        if row.empty:
            return None
        # Reference: Championship profiles of clubs promoted up to and including this season.
        promoted = sorted((t, s) for (t, s) in self.membership.members
                          if s <= season and self.membership.is_promoted(t, s))
        ref_rows = pd.concat([champ[(champ["team"] == t) & (champ["season"] == s - 1)] for t, s in promoted]) \
            if promoted else pd.DataFrame()
        if ref_rows.empty:
            return None
        ref = ref_rows[["gf", "ga", "sot_f", "sot_a", "sh_f", "sh_a", "pts"]].mean()
        team_row = row.iloc[0]
        ratios = {}
        for col in ref.index:
            if ref[col] > 0 and pd.notna(team_row[col]) and team_row[col] > 0:
                ratios[col] = float(np.clip((team_row[col] / ref[col]) ** CHAMPIONSHIP_SHRINK, 0.7, 1.4))
            else:
                ratios[col] = 1.0
        return ratios

    def for_team(self, team: str, season: int, spell: int) -> pd.Series:
        """Prior for a team whose rolling window is not yet full."""
        key = ("team", team, season, spell)
        if key in self._cache:
            return self._cache[key]
        if spell == season and self.membership.is_promoted(team, season):
            prior = self.promoted_base(season).copy()
            ratios = self._championship_ratio(team, season)
            if ratios:
                attack = np.sqrt(ratios["gf"] * ratios["sot_f"])
                defence = np.sqrt(ratios["ga"] * ratios["sot_a"])
                for col in ("gf", "xg_f", "npxg_f"):
                    prior[col] *= attack
                for col in ("ga", "xg_a", "npxg_a"):
                    prior[col] *= defence
                prior["sot_f"] *= ratios["sot_f"]
                prior["sot_a"] *= ratios["sot_a"]
                prior["sh_f"] *= ratios["sh_f"]
                prior["sh_a"] *= ratios["sh_a"]
                prior["pts"] *= ratios["pts"]
                prior["xgd"] = prior["xg_f"] - prior["xg_a"]
                prior["npxgd"] = prior["npxg_f"] - prior["npxg_a"]
        elif spell == season and self.membership.first_season is not None and season > self.membership.first_season:
            # A club we have never seen in the PL and cannot place (should not happen) -> promoted prior.
            prior = self.promoted_base(season).copy()
        else:
            prior = self.league(season).copy()
        self._cache[key] = prior
        return prior

    def venue_offset(self, season: int, venue: int) -> pd.Series:
        return self.league(season, venue) - self.league(season)


# ---------------------------------------------------------------- rolling state
def _rolling_state(tm: pd.DataFrame, group_cols: list[str], stats: list[str], windows) -> pd.DataFrame:
    """Post-match rolling sums/counts per group. Rows must be date-sorted within group."""
    keys = [tm[c] for c in group_cols]
    values = tm[stats].astype(float)
    filled = values.fillna(0.0)
    present = values.notna().astype(float)
    out = tm[group_cols + ["date"]].copy()
    games = tm.groupby(group_cols, sort=False).cumcount() + 1
    out["games"] = games
    for n in windows:
        sums = filled.groupby(keys, sort=False).rolling(n, min_periods=1).sum()
        cnts = present.groupby(keys, sort=False).rolling(n, min_periods=1).sum()
        sums = sums.reset_index(level=list(range(len(group_cols))), drop=True).reindex(tm.index)
        cnts = cnts.reset_index(level=list(range(len(group_cols))), drop=True).reindex(tm.index)
        out = pd.concat([out, sums.add_prefix("sum_").add_suffix(f"_{n}"),
                         cnts.add_prefix("cnt_").add_suffix(f"_{n}")], axis=1)
        out[f"n_{n}"] = games.clip(upper=n)
    return out


def _lookup_state(side: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Latest state strictly before each fixture date, same team, same spell."""
    left = side[["row_id", "team", "date", "spell"]].sort_values("date", kind="mergesort")
    right = states.rename(columns={"date": "state_date", "spell": "state_spell"}).sort_values(
        "state_date", kind="mergesort")
    merged = pd.merge_asof(left, right, left_on="date", right_on="state_date", by="team",
                           allow_exact_matches=False, direction="backward")
    merged = merged.set_index("row_id").reindex(side["row_id"])
    stale = merged["state_spell"].isna() | (merged["state_spell"] != merged["spell"])
    n_cols = [c for c in merged.columns if c.startswith("n_")] + ["games"]
    merged.loc[stale, n_cols] = 0
    merged.loc[stale, "state_date"] = pd.NaT
    return merged


def _blend(state: pd.DataFrame, stat: str, n: int, prior: np.ndarray) -> np.ndarray:
    sums = state[f"sum_{stat}_{n}"].to_numpy(dtype=float)
    cnts = state[f"cnt_{stat}_{n}"].to_numpy(dtype=float)
    k = state[f"n_{n}"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnts > 0, sums / np.where(cnts > 0, cnts, 1), prior)
    weight = np.nan_to_num(k / n)
    return weight * np.nan_to_num(mean, nan=0.0) + (1 - weight) * prior


# Ratio metrics are ratios of window sums (not means of per-match ratios).
RATIOS = {
    "ppda": ("ppda_att", ("ppda_def",), 1.0),                 # lower = more intense pressing
    "press_res": ("pass_f", ("opp_def",), 1.0),               # higher = opponents struggle to press you
    "pass_share": ("pass_f", ("pass_f", "ppda_att"), 1.0),    # possession proxy, 0..1
    "directness": ("deep_f", ("pass_f",), 100.0),             # deep completions per 100 passes
}


def _blend_ratio(state: pd.DataFrame, name: str, n: int, prior: np.ndarray) -> np.ndarray:
    num, den, scale = RATIOS[name]
    top = state[f"sum_{num}_{n}"].to_numpy(dtype=float)
    bottom = sum(state[f"sum_{d}_{n}"].to_numpy(dtype=float) for d in den)
    cnt = state[f"cnt_{num}_{n}"].to_numpy(dtype=float)
    k = state[f"n_{n}"].to_numpy(dtype=float)
    ok = (cnt > 0) & (bottom > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(ok, scale * top / np.where(bottom > 0, bottom, 1), prior)
    weight = np.nan_to_num(k / n)
    return weight * np.nan_to_num(ratio, nan=0.0) + (1 - weight) * prior


# ------------------------------------------------------------------------- Elo
def _gd_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def elo_ratings(matches: pd.DataFrame, queries: pd.DataFrame, membership: Membership) -> pd.DataFrame:
    """Pre-match Elo for each query row (row_id, date, season, home, away)."""
    ratings: dict[str, float] = {}
    events = [
        (d, 1, s, h, a, hg, ag, None)
        for d, s, h, a, hg, ag in zip(matches["date"], matches["season"], matches["home"], matches["away"],
                                      matches["fthg"], matches["ftag"])
    ] + [
        (d, 0, s, h, a, None, None, rid)
        for d, s, h, a, rid in zip(queries["date"], queries["season"], queries["home"], queries["away"],
                                   queries["row_id"])
    ]
    events.sort(key=lambda e: (e[0], e[1]))  # queries before same-day results
    current_season = None
    out = {}
    for date, kind, season, home, away, hg, ag, rid in events:
        if current_season is None:
            current_season = season
        elif season > current_season:
            _new_season(ratings, current_season, season, membership)
            current_season = season
        rh = ratings.setdefault(home, _entry_rating(ratings, home, season, membership))
        ra = ratings.setdefault(away, _entry_rating(ratings, away, season, membership))
        if kind == 0:
            out[rid] = (rh, ra)
            continue
        expected = 1.0 / (1.0 + 10 ** ((ra - rh - ELO_HOME_ADV) / 400.0))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = ELO_K * _gd_multiplier(int(hg - ag)) * (score - expected)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    res = pd.DataFrame.from_dict(out, orient="index", columns=["h_elo", "a_elo"])
    return res.reindex(queries["row_id"])


def _entry_rating(ratings: dict, team: str, season: int, membership: Membership) -> float:
    if ratings and membership.is_promoted(team, season):
        # Promoted clubs start at the level of the clubs they replace.
        relegated = [ratings[t] for t in ratings if (t, season - 1) in membership.members
                     and (t, season) not in membership.members]
        return float(np.mean(relegated)) if relegated else ELO_INIT - 100
    return ELO_INIT


def _new_season(ratings: dict, old: int, new: int, membership: Membership) -> None:
    promoted_start = None
    relegated = [ratings[t] for t in ratings if (t, old) in membership.members and (t, new) not in membership.members]
    if relegated:
        promoted_start = float(np.mean(relegated))
    for team in list(ratings):
        ratings[team] = ELO_INIT + (1 - ELO_SEASON_REGRESSION) * (ratings[team] - ELO_INIT)
    if promoted_start is not None:
        start = ELO_INIT + (1 - ELO_SEASON_REGRESSION) * (promoted_start - ELO_INIT)
        for team, season in membership.members:
            if season == new and (team, old) not in membership.members:
                ratings[team] = start


# -------------------------------------------------------------------- builder
@dataclass
class FeatureSpec:
    columns: list[str]
    explain: list[tuple[str, str, str]]  # (suffix, label, short label) for one-off match reports


def _side_feature_names() -> list[str]:
    names = ["form_pts5", "form_pts3"]
    for n in (3, 5):
        names += [f"gf{n}", f"ga{n}", f"sot_f{n}", f"sot_a{n}", f"sh_f{n}", f"sh_a{n}",
                  f"xg_f{n}", f"xg_a{n}", f"npxgd{n}", f"fin_var{n}", f"ppda{n}"]
    names += ["press_res5", "pass_share5", "deep_f5", "deep_a5"]
    names += ["xgd19", "ppg19", "rest", "elo", "promoted", "spell_games",
              "venue_pts5", "venue_gf5", "venue_ga5", "venue_xgd5", "home_edge"]
    names += STYLE_FEATURES
    return names


# Slow-moving style profile (last 10 matches) consumed by src/tactics.py.
STYLE_FEATURES = ["ppda10", "pass_share10", "directness10", "press_res10"]


SIDE_FEATURES = _side_feature_names()
# Home-minus-away differences for every rolling metric (plus venue form: home side at home
# vs away side on the road).
DIFF_FEATURES = [n for n in SIDE_FEATURES
                 if n not in ("promoted", "spell_games", "home_edge") and not n.startswith("venue")
                 and n not in STYLE_FEATURES] + ["venue_xgd5"]
ALL_FEATURE_COLUMNS = ([f"{s}_{n}" for s in ("h", "a") for n in SIDE_FEATURES]
                       + [f"d_{n}" for n in DIFF_FEATURES] + ["elo_exp_home"])

# What the model actually consumes. Out-of-time backtests (2023-24 .. 2025-26) showed that
# feeding both raw sides of every metric (86 columns) overfits ~2k training matches;
# differences carry the same information at half the dimensionality.
FEATURES = FeatureSpec(
    columns=[f"d_{n}" for n in DIFF_FEATURES] + [
        "elo_exp_home", "h_promoted", "a_promoted", "h_home_edge", "a_home_edge",
        "h_venue_pts5", "a_venue_pts5", "h_venue_gf5", "a_venue_ga5", "h_rest", "a_rest",
    ],
    explain=[
        ("elo", "Elo rating", "Elo"),
        ("form_pts5", "Form pts (last 5)", "Form pts L5"),
        ("ppg19", "Points/game (last 19)", "Pts/g L19"),
        ("gf5", "Goals for /g (last 5)", "Goals/g L5"),
        ("ga5", "Goals against /g (last 5)", "Conceded/g L5"),
        ("xg_f5", "xG for /g (last 5)", "xG/g L5"),
        ("xg_a5", "xG against /g (last 5)", "xGA/g L5"),
        ("npxgd5", "npxG diff /g (last 5)", "npxGD/g L5"),
        ("fin_var5", "Goals - xG /g (last 5)", "G-xG/g L5"),
        ("sot_f5", "Shots on target /g (last 5)", "SoT/g L5"),
        ("sh_a5", "Shots conceded /g (last 5)", "Shots agst L5"),
        ("ppda5", "PPDA (last 5)", "PPDA L5"),
        ("pass_share5", "Possession share (last 5)", "Poss. L5"),
        ("press_res5", "Press resistance (last 5)", "Press res L5"),
        ("deep_f5", "Deep completions /g (last 5)", "Deep/g L5"),
        ("venue_pts5", "Venue pts/g (last 5 H or A)", "Venue pts L5"),
        ("rest", "Days of rest", "Rest days"),
        ("spell_games", "PL games this spell", "PL games"),
    ],
)


class FeatureBuilder:
    """Holds completed-match history and turns any fixture list into model features."""

    def __init__(self, matches: pd.DataFrame, championship: pd.DataFrame | None = None):
        matches = matches.copy()
        matches["date"] = to_day(matches["date"])
        self.matches = matches.sort_values("date", kind="mergesort").reset_index(drop=True)
        self.championship = championship
        self.tm = team_match_table(self.matches)
        self.membership = Membership(set(zip(self.tm["team"], self.tm["season"])), championship)
        self.priors = Priors(self.tm, self.membership, championship)
        self.tm["spell"] = [self.membership.spell_start(t, s) for t, s in zip(self.tm["team"], self.tm["season"])]
        self.states = _rolling_state(self.tm, ["team", "spell"], STATS, WINDOWS)
        venue = self.tm.sort_values(["team", "is_home", "date"], kind="mergesort")
        self.venue_states = {
            flag: _rolling_state(venue[venue["is_home"] == flag], ["team", "spell"], VENUE_STATS, VENUE_WINDOWS)
            for flag in (0, 1)
        }

    @property
    def teams(self) -> list[str]:
        return sorted(self.tm["team"].unique())

    def build(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Features for fixtures with columns date, home, away (any extra columns are kept)."""
        fx = fixtures.copy().reset_index(drop=True)
        fx["date"] = to_day(fx["date"])
        if "season" not in fx:
            fx["season"] = season_of(fx["date"])
        fx["season"] = fx["season"].astype(int)
        fx["row_id"] = np.arange(len(fx))
        # Clubs in an upcoming fixture belong to that season even before they have played.
        self.membership.add(set(zip(fx["home"], fx["season"])) | set(zip(fx["away"], fx["season"])))

        feats = pd.concat([self._side_features(fx, col, side, venue_flag)
                           for side, col, venue_flag in (("h", "home", 1), ("a", "away", 0))], axis=1)

        elo = elo_ratings(self.matches, fx[["row_id", "date", "season", "home", "away"]], self.membership)
        feats["h_elo"] = elo["h_elo"].to_numpy()
        feats["a_elo"] = elo["a_elo"].to_numpy()
        derived = {f"d_{name}": feats[f"h_{name}"] - feats[f"a_{name}"] for name in DIFF_FEATURES}
        derived["elo_exp_home"] = 1.0 / (1.0 + 10 ** ((feats["a_elo"] - feats["h_elo"] - ELO_HOME_ADV) / 400.0))
        feats = pd.concat([feats, pd.DataFrame(derived, index=feats.index)], axis=1)

        missing = feats[ALL_FEATURE_COLUMNS].isna().sum()
        if missing.any():
            log.warning("NaN features after prior blending: %s", missing[missing > 0].to_dict())
        return pd.concat([fx.drop(columns=["row_id"]), feats[ALL_FEATURE_COLUMNS]], axis=1).copy()

    def _side_features(self, fx: pd.DataFrame, col: str, side: str, venue_flag: int) -> pd.DataFrame:
        s = fx[["row_id", "date", "season"]].assign(team=fx[col])
        s["spell"] = [self.membership.spell_start(t, se) for t, se in zip(s["team"], s["season"])]
        prior_rows = [self.priors.for_team(t, se, sp) for t, se, sp in zip(s["team"], s["season"], s["spell"])]
        prior = pd.DataFrame(prior_rows).reset_index(drop=True)

        state = _lookup_state(s, self.states)
        out: dict[str, np.ndarray] = {}

        def p(stat):
            return prior[stat].to_numpy(dtype=float)

        out[f"{side}_form_pts5"] = 5 * _blend(state, "pts", 5, p("pts"))
        out[f"{side}_form_pts3"] = 3 * _blend(state, "pts", 3, p("pts"))
        for n in (3, 5):
            for stat in ("gf", "ga", "sot_f", "sot_a", "sh_f", "sh_a", "xg_f", "xg_a", "npxgd", "fin_var"):
                out[f"{side}_{stat}{n}"] = _blend(state, stat, n, p(stat))
            out[f"{side}_ppda{n}"] = _blend_ratio(state, "ppda", n, p("ppda"))
        out[f"{side}_press_res5"] = _blend_ratio(state, "press_res", 5, p("press_res"))
        out[f"{side}_pass_share5"] = _blend_ratio(state, "pass_share", 5, p("pass_share"))
        out[f"{side}_deep_f5"] = _blend(state, "deep_f", 5, p("deep_f"))
        out[f"{side}_deep_a5"] = _blend(state, "deep_a", 5, p("deep_a"))
        for name in ("ppda", "pass_share", "directness", "press_res"):
            out[f"{side}_{name}10"] = _blend_ratio(state, name, 10, p(name))
        out[f"{side}_xgd19"] = _blend(state, "xgd", 19, p("xgd"))
        out[f"{side}_ppg19"] = _blend(state, "pts", 19, p("pts"))

        rest = (s["date"].to_numpy() - state["state_date"].to_numpy()) / np.timedelta64(1, "D")
        out[f"{side}_rest"] = np.clip(np.nan_to_num(rest, nan=REST_CAP_DAYS), 1, REST_CAP_DAYS)
        promoted = [sp == se and self.membership.is_promoted(t, se)
                    for t, se, sp in zip(s["team"], s["season"], s["spell"])]
        out[f"{side}_promoted"] = np.array(promoted, dtype=float)
        out[f"{side}_spell_games"] = np.minimum(state["games"].to_numpy(dtype=float), 38)

        # Venue-specific form: home side at home, away side away, plus the club's home/away gap.
        venue_state = {flag: _lookup_state(s, self.venue_states[flag]) for flag in (0, 1)}
        venue_prior = {}
        for flag in (0, 1):
            offsets = pd.DataFrame([self.priors.venue_offset(se, flag) for se in s["season"]]).reset_index(drop=True)
            venue_prior[flag] = prior[VENUE_STATS].add(offsets[VENUE_STATS])
        vs, vp = venue_state[venue_flag], venue_prior[venue_flag]
        out[f"{side}_venue_pts5"] = _blend(vs, "pts", 5, vp["pts"].to_numpy(dtype=float))
        out[f"{side}_venue_gf5"] = _blend(vs, "gf", 5, vp["gf"].to_numpy(dtype=float))
        out[f"{side}_venue_ga5"] = _blend(vs, "ga", 5, vp["ga"].to_numpy(dtype=float))
        out[f"{side}_venue_xgd5"] = _blend(vs, "xgd", 5, vp["xgd"].to_numpy(dtype=float))
        home_xgd = _blend(venue_state[1], "xgd", 10, venue_prior[1]["xgd"].to_numpy(dtype=float))
        away_xgd = _blend(venue_state[0], "xgd", 10, venue_prior[0]["xgd"].to_numpy(dtype=float))
        out[f"{side}_home_edge"] = home_xgd - away_xgd
        return pd.DataFrame(out, index=fx.index)


def training_frame(builder: FeatureBuilder) -> pd.DataFrame:
    """Features + target for every completed match (features only use prior matches)."""
    meta_cols = ["match_key", "season", "date", "home", "away", "fthg", "ftag", "ftr",
                 "odds_h", "odds_d", "odds_a", "b365_h", "b365_d", "b365_a", "b365_over25", "b365_under25",
                 "source"]
    data = builder.build(builder.matches[meta_cols])
    data["y"] = data["ftr"].map(TARGET_MAP).astype(int)
    return data
