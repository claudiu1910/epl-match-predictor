"""Offline tests: leakage guarantees, promoted-team handling, name alignment, scoring."""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate as ev  # noqa: E402
from src import fixture_fetcher as ff  # noqa: E402
from src.align_teams import UnknownTeamError, normalize_team, resolve_user_team  # noqa: E402
from src.data_loader import MATCH_COLUMNS  # noqa: E402
from src.features import ALL_FEATURE_COLUMNS, FEATURES, FeatureBuilder, training_frame  # noqa: E402

SEASON_TEAMS = {
    2020: ["Arsenal", "Chelsea", "Everton", "Fulham", "Liverpool", "Tottenham"],
    2021: ["Arsenal", "Chelsea", "Everton", "Liverpool", "Tottenham", "Brentford"],   # Fulham down, Brentford up
    2022: ["Arsenal", "Chelsea", "Liverpool", "Tottenham", "Brentford", "Fulham"],     # Everton down, Fulham back
}


def synthetic_matches(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season, teams in SEASON_TEAMS.items():
        start = pd.Timestamp(f"{season}-08-14")
        fixtures = list(permutations(teams, 2))
        rng.shuffle(fixtures)
        # 3 matches per weekend, one per team, greedily scheduled.
        day = 0
        pending = fixtures
        while pending:
            used, left = set(), []
            for h, a in pending:
                if h in used or a in used:
                    left.append((h, a))
                    continue
                used.update((h, a))
                hxg, axg = rng.gamma(2.0, 0.75), rng.gamma(2.0, 0.6)
                hg, ag = rng.poisson(hxg), rng.poisson(axg)
                rows.append({
                    "season": season, "date": start + pd.Timedelta(days=7 * day), "kickoff_utc": pd.NaT,
                    "home": h, "away": a, "fthg": hg, "ftag": ag,
                    "ftr": "H" if hg > ag else "A" if hg < ag else "D",
                    "hs": rng.integers(5, 20), "as": rng.integers(4, 18),
                    "hst": rng.integers(1, 8), "ast": rng.integers(1, 7),
                    "home_xg": hxg, "away_xg": axg, "home_npxg": hxg * 0.9, "away_npxg": axg * 0.9,
                    "home_ppda_att": rng.integers(150, 400), "home_ppda_def": rng.integers(10, 35),
                    "away_ppda_att": rng.integers(150, 400), "away_ppda_def": rng.integers(10, 35),
                    "home_deep": rng.integers(2, 15), "away_deep": rng.integers(1, 12),
                    "odds_h": 2.2, "odds_d": 3.3, "odds_a": 3.4, "b365_h": 2.2, "b365_d": 3.3, "b365_a": 3.4,
                    "b365_over25": 1.9, "b365_under25": 1.9, "source": "both",
                })
            pending = left
            day += 1
    df = pd.DataFrame(rows)
    df["match_key"] = df["season"].astype(str) + ":" + df["home"] + ":" + df["away"]
    return df[MATCH_COLUMNS].sort_values("date").reset_index(drop=True)


@pytest.fixture(scope="module")
def matches():
    return synthetic_matches()


@pytest.fixture(scope="module")
def train(matches):
    return training_frame(FeatureBuilder(matches))


def _pick_mid_season(matches: pd.DataFrame, season: int = 2022) -> pd.Series:
    s = matches[matches["season"] == season].reset_index(drop=True)
    return s.iloc[len(s) // 2]


def test_features_ignore_future_matches(matches, train):
    """Features for a match are identical whether or not later matches exist."""
    target = _pick_mid_season(matches)
    past_only = matches[matches["date"] < target["date"]]
    live = FeatureBuilder(past_only).build(pd.DataFrame([target[["date", "home", "away"]]]))
    full_row = train[train["match_key"] == target["match_key"]].iloc[0]
    np.testing.assert_allclose(live.iloc[0][ALL_FEATURE_COLUMNS].to_numpy(dtype=float),
                               full_row[ALL_FEATURE_COLUMNS].to_numpy(dtype=float), rtol=1e-9, atol=1e-9)


def test_own_result_does_not_leak(matches, train):
    """Changing a match's own score/xG must not change that match's features."""
    target = _pick_mid_season(matches)
    tampered = matches.copy()
    idx = tampered.index[tampered["match_key"] == target["match_key"]][0]
    tampered.loc[idx, ["fthg", "ftag", "home_xg", "away_xg", "hst", "ast"]] = [9, 0, 6.0, 0.1, 15, 0]
    tampered.loc[idx, "ftr"] = "H"
    new = training_frame(FeatureBuilder(tampered))
    before = train.set_index("match_key")
    after = new.set_index("match_key")
    key = target["match_key"]
    np.testing.assert_allclose(after.loc[key, ALL_FEATURE_COLUMNS].to_numpy(dtype=float),
                               before.loc[key, ALL_FEATURE_COLUMNS].to_numpy(dtype=float))
    # ... while the home side's *next* match does see it.
    later = matches[(matches["date"] > target["date"]) &
                    ((matches["home"] == target["home"]) | (matches["away"] == target["home"]))].iloc[0]
    assert not np.allclose(after.loc[later["match_key"], ALL_FEATURE_COLUMNS].to_numpy(dtype=float),
                           before.loc[later["match_key"], ALL_FEATURE_COLUMNS].to_numpy(dtype=float))


def test_same_day_matches_are_excluded(matches):
    """A fixture never sees results from its own kickoff date."""
    target = _pick_mid_season(matches)
    same_day = matches[matches["date"] <= target["date"]]
    upto = FeatureBuilder(same_day).build(pd.DataFrame([target[["date", "home", "away"]]]))
    before = FeatureBuilder(matches[matches["date"] < target["date"]]).build(
        pd.DataFrame([target[["date", "home", "away"]]]))
    np.testing.assert_allclose(upto[ALL_FEATURE_COLUMNS].to_numpy(dtype=float),
                               before[ALL_FEATURE_COLUMNS].to_numpy(dtype=float))


def test_promoted_team_gets_finite_prior_features(matches, train):
    brentford_debut = train[(train["season"] == 2021) &
                            ((train["home"] == "Brentford") | (train["away"] == "Brentford"))].iloc[0]
    side = "h" if brentford_debut["home"] == "Brentford" else "a"
    assert brentford_debut[f"{side}_promoted"] == 1.0
    assert brentford_debut[f"{side}_spell_games"] == 0
    assert np.isfinite(train[ALL_FEATURE_COLUMNS].to_numpy(dtype=float)).all()


def test_returning_team_starts_a_new_spell(train):
    """Fulham's 2020 form must not bleed into its 2022 return."""
    back = train[(train["season"] == 2022) & ((train["home"] == "Fulham") | (train["away"] == "Fulham"))].iloc[0]
    side = "h" if back["home"] == "Fulham" else "a"
    assert back[f"{side}_promoted"] == 1.0
    assert back[f"{side}_spell_games"] == 0


def test_unseen_team_prediction_does_not_crash(matches):
    """A club with no PL history at all (e.g. a first-time promotion) still gets features."""
    fixture = pd.DataFrame([{"date": pd.Timestamp("2023-08-12"), "home": "Arsenal", "away": "Luton Town"}])
    feats = FeatureBuilder(matches).build(fixture)
    assert np.isfinite(feats[FEATURES.columns].to_numpy(dtype=float)).all()
    assert feats.loc[0, "a_promoted"] == 1.0


@pytest.mark.parametrize("raw,expected", [
    ("Spurs", "Tottenham"), ("Tottenham Hotspur", "Tottenham"), ("Man Utd", "Manchester United"),
    ("Man United", "Manchester United"), ("Manchester City", "Manchester City"), ("Nott'm Forest", "Nottingham Forest"),
    ("Wolves", "Wolverhampton Wanderers"), ("West Brom", "West Bromwich Albion"), ("AFC Bournemouth", "Bournemouth"),
    ("Brighton & Hove Albion", "Brighton"), ("Sheffield Weds", "Sheffield Wednesday"), ("Hull", "Hull City"),
    ("Newcastle United FC", "Newcastle United"), ("  liverpool ", "Liverpool"),
])
def test_team_aliases(raw, expected):
    assert normalize_team(raw) == expected


def test_user_team_resolution_is_strict():
    with pytest.raises(UnknownTeamError):
        resolve_user_team("Barcelona", ["Arsenal", "Liverpool"])
    with pytest.raises(UnknownTeamError):
        resolve_user_team("Burnley", ["Arsenal", "Liverpool"])  # real club, not in the data
    assert resolve_user_team("Spurs", ["Tottenham", "Arsenal"]) == "Tottenham"


def test_next_round_grouping():
    t0 = pd.Timestamp("2026-10-10 11:30", tz="UTC")
    fixtures = pd.DataFrame({
        "kickoff_utc": [t0, t0 + pd.Timedelta(hours=3), t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=7)],
        "home": ["Arsenal", "Chelsea", "Everton", "Arsenal"],
        "away": ["Leeds United", "Fulham", "Liverpool", "Chelsea"],
    })
    rnd = ff.next_round(fixtures)
    assert list(rnd["home"]) == ["Arsenal", "Chelsea", "Everton"]


def test_scoring_rules():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert ev.multiclass_brier(y, perfect) == pytest.approx(0.0)
    assert ev.ranked_probability_score(y, perfect) == pytest.approx(0.0)
    uniform = np.full((3, 3), 1 / 3)
    assert ev.multiclass_brier(y, uniform) == pytest.approx(2 / 3)
    assert ev.score(y, uniform)["log_loss"] == pytest.approx(np.log(3))
    # RPS punishes predicting an away win for a home win more than predicting a draw.
    assert ev.ranked_probability_score(np.array([0]), np.array([[0, 0, 1.0]])) > \
        ev.ranked_probability_score(np.array([0]), np.array([[0, 1.0, 0]]))


def test_bookmaker_probs_remove_overround():
    frame = pd.DataFrame({"odds_h": [2.0, np.nan], "odds_d": [3.4, 3.0], "odds_a": [4.0, 2.5]})
    probs, valid = ev.bookmaker_probs(frame)
    assert valid.tolist() == [True, False]
    assert probs[0].sum() == pytest.approx(1.0)
