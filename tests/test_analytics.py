"""Offline tests for the analytics layer: simulation, value betting, tactics, SHAP, availability."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import availability as av  # noqa: E402
from src import bet_evaluator as bets  # noqa: E402
from src import fixture_fetcher as ff  # noqa: E402
from src.explainer import Explainer  # noqa: E402
from src.model import ModelConfig, fit_calibrated, make_xgb  # noqa: E402
from src.simulation import analytic_markets, dc_tau, fit_rho, score_matrix, simulate  # noqa: E402
from src.tactics import STYLE_NAMES, TacticalStyles  # noqa: E402


# ------------------------------------------------------------------ simulation
def test_score_matrix_is_a_distribution():
    m = score_matrix(1.6, 1.1, rho=-0.05)
    assert m.shape == (11, 11)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_dixon_coles_tau_only_touches_low_scores():
    h, a = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    tau = dc_tau(h, a, 1.4, 1.1, 0.1)
    assert tau[0, 0] != 1 and tau[1, 1] != 1
    assert np.allclose(tau[2:, :], 1) and np.allclose(tau[:, 2:], 1)
    assert np.allclose(dc_tau(h, a, 1.4, 1.1, 0.0), 1)


def test_monte_carlo_matches_analytic_probabilities():
    sim = simulate(1.7, 1.0, rho=-0.03, n_sims=10_000, seed=1)
    exact = analytic_markets([1.7], [1.0], -0.03).iloc[0]
    assert sim.p_home + sim.p_draw + sim.p_away == pytest.approx(1.0)
    for got, want in ((sim.p_home, exact["p_home"]), (sim.p_draw, exact["p_draw"]), (sim.p_away, exact["p_away"])):
        assert abs(got - want) < 0.02          # ~4 standard errors at n = 10,000
    d = sim.to_dict()
    assert sum(map(sum, d["matrix"])) == pytest.approx(1.0, abs=1e-3)
    assert d["over_under"]["2.5"]["over"] + d["over_under"]["2.5"]["under"] == pytest.approx(1.0)
    assert d["n_sims"] == 10_000


def test_simulation_is_reproducible_with_seed():
    a, b = simulate(1.3, 1.2, seed=7), simulate(1.3, 1.2, seed=7)
    assert (a.home_goals == b.home_goals).all() and (a.away_goals == b.away_goals).all()


def test_fit_rho_recovers_sign():
    rng = np.random.default_rng(0)
    lh, la = np.full(4000, 1.4), np.full(4000, 1.1)
    # Sample from a DC distribution with negative rho (more 0-0/1-1 than independent Poisson).
    joint = score_matrix(1.4, 1.1, -0.12)
    idx = rng.choice(joint.size, size=4000, p=joint.ravel())
    hg, ag = np.divmod(idx, joint.shape[1])
    assert fit_rho(hg, ag, lh, la) < -0.03


# ------------------------------------------------------------------ value betting
@pytest.mark.parametrize("method", ["proportional", "shin", "power"])
def test_margin_removal_gives_a_distribution(method):
    odds = [1.80, 3.80, 4.50]
    fair = bets.remove_margin(odds, method)
    assert fair.sum() == pytest.approx(1.0)
    assert fair[0] > fair[1] > fair[2]
    assert (fair < 1 / np.array(odds)).all()   # margin removed from every outcome


def test_shin_shifts_margin_towards_longshots():
    odds = [1.30, 5.50, 11.0]
    prop, shin = bets.remove_margin(odds, "proportional"), bets.remove_margin(odds, "shin")
    assert shin[0] > prop[0] and shin[2] < prop[2]


def test_expected_value_and_flags():
    assert bets.expected_value(0.5, 2.2) == pytest.approx(0.1)
    res = bets.evaluate_fixture({"home": 0.50, "draw": 0.25, "away": 0.25, "over25": 0.5, "under25": 0.5},
                                {"home": 2.20, "draw": 3.40, "away": 3.60})
    flagged = {s["selection"] for s in res["value"]}
    assert flagged == {"home"}                      # EV +10% > 5%; draw -15%, away -10%
    assert "Total goals 2.5" not in res["markets"]  # unpriced market skipped
    assert res["markets"]["1X2"]["margin"] > 0


def test_backtest_flat_stakes():
    probs = np.array([[0.6, 0.2, 0.2], [0.3, 0.3, 0.4], [0.5, 0.3, 0.2]])
    odds = np.array([[2.0, 3.5, 4.0], [2.5, 3.2, 3.0], [np.nan, 3.0, 3.0]])
    outcomes = np.array([0, 1, 0])
    r = bets.backtest(probs, odds, outcomes, threshold=0.05)
    # Row 0: home EV +0.20 (won, +1.0). Row 1: away EV +0.20 (lost, -1). Row 2 unpriced home -> skipped.
    assert r["bets"] == 2
    assert r["profit"] == pytest.approx(0.0)
    assert r["roi"] == pytest.approx(0.0)


# ------------------------------------------------------------------ tactics
def _style_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    centres = {"possession": (9.0, 0.60, 3.8), "low_block": (15.0, 0.40, 2.8), "transition": (11.5, 0.50, 2.4)}
    rows = []
    for _ in range(n):
        h, a = rng.choice(list(centres), 2)
        ch, ca = centres[h], centres[a]
        rows.append({"h_ppda10": ch[0] + rng.normal(0, .5), "h_pass_share10": ch[1] + rng.normal(0, .015),
                     "h_directness10": ch[2] + rng.normal(0, .1), "a_ppda10": ca[0] + rng.normal(0, .5),
                     "a_pass_share10": ca[1] + rng.normal(0, .015), "a_directness10": ca[2] + rng.normal(0, .1),
                     "true_h": h})
    return pd.DataFrame(rows)


def test_tactical_clusters_are_named_from_centroids():
    frame = _style_frame()
    tactics = TacticalStyles().fit(frame)
    out = tactics.transform(frame)
    expected = {"possession": STYLE_NAMES[0], "low_block": STYLE_NAMES[1], "transition": STYLE_NAMES[2]}
    accuracy = (out["h_style"] == frame["true_h"].map(expected)).mean()
    assert accuracy > 0.95
    assert out[["h_style_possession", "h_style_low_block", "h_style_transition"]].sum(axis=1).eq(1).all()


# ------------------------------------------------------------------ calibration + SHAP
@pytest.fixture(scope="module")
def toy_model():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(1500, 6))
    logits = np.column_stack([X[:, 0] + 0.3, 0.2 * X[:, 1], -X[:, 0]])
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    cfg = ModelConfig()
    calibrated, booster = fit_calibrated(X, y, cfg)
    return X, y, calibrated, booster


def test_chronological_calibration_outputs_probabilities(toy_model):
    X, _, calibrated, booster = toy_model
    probs = calibrated.predict_proba(X[:50])
    assert np.allclose(probs.sum(axis=1), 1.0)
    # The refit booster (full window) is the one the calibrator wraps.
    assert calibrated.calibrated_classifiers_[0].estimator.estimator is booster


def test_calibration_needs_enough_recent_data():
    with pytest.raises(ValueError):
        fit_calibrated(np.zeros((150, 3)), np.tile([0, 1, 2], 50), ModelConfig())


def test_shap_drivers_add_up_to_final_probability(toy_model):
    X, _, calibrated, booster = toy_model
    names = [f"f{i}" for i in range(X.shape[1])]
    explainer = Explainer(booster, names)
    final = calibrated.predict_proba(X[:1])[0]
    e = explainer.explain(X[0], final, top=None)
    for i, key in enumerate(("home", "draw", "away")):
        total = e["baseline"][key] * 100 + sum(d[key] for d in e["drivers"]) + e["calibration_adjustment"][key]
        assert total == pytest.approx(final[i] * 100, abs=0.15)


def test_shap_backend_matches_xgboost_native(toy_model):
    X, _, _, booster = toy_model
    names = [f"f{i}" for i in range(X.shape[1])]
    explainer = Explainer(booster, names)
    phi, base = explainer.contributions(X[:5])
    from xgboost import DMatrix
    native = booster.get_booster().predict(DMatrix(X[:5]), pred_contribs=True)
    assert np.allclose(phi, native[:, :, :-1].transpose(0, 2, 1), atol=1e-4)
    assert np.allclose(base, native[0, :, -1], atol=1e-4)


# ------------------------------------------------------------------ availability
def _availability_model(overrides=None):
    squads = pd.DataFrame([
        {"fpl_id": 1, "team": "Arsenal", "web_name": "Striker", "name": "Sam Striker", "position": "FWD",
         "status": "i", "chance": 0.0, "news": "Knee injury", "minutes": 900},
        {"fpl_id": 2, "team": "Arsenal", "web_name": "Winger", "name": "Will Winger", "position": "MID",
         "status": "a", "chance": np.nan, "news": "", "minutes": 900},
        {"fpl_id": 3, "team": "Arsenal", "web_name": "Keeper", "name": "Ken Keeper", "position": "GK",
         "status": "d", "chance": 50.0, "news": "Ill", "minutes": 900},
        {"fpl_id": 4, "team": "Arsenal", "web_name": "Gone", "name": "Gus Gone", "position": "FWD",
         "status": "u", "chance": 0.0, "news": "Has joined Other FC permanently", "minutes": 0},
    ])
    cur = pd.DataFrame([
        {"player_name": "Sam Striker", "team": "Arsenal", "npxG": 2.0, "xA": 0.0, "time": 900, "position": "F", "games": 10},
        {"player_name": "Will Winger", "team": "Arsenal", "npxG": 2.0, "xA": 2.0, "time": 900, "position": "M", "games": 10},
        {"player_name": "Ken Keeper", "team": "Arsenal", "npxG": 0.0, "xA": 0.0, "time": 900, "position": "GK", "games": 10},
    ])
    return av.AvailabilityModel(squads, cur, pd.DataFrame(), overrides)


def test_availability_penalises_missing_key_attacker():
    report = _availability_model().team_report("Arsenal")
    # Striker holds 2 / (2 + 4) = 1/3 of attack; out for sure; replacement recovers 40%.
    assert report["attack_penalty"] == pytest.approx((1 / 3) * (1 - av.REPLACEMENT_RECOVERY), abs=1e-3)
    # First-choice keeper 50% doubtful -> half the GK penalty.
    assert report["defence_penalty"] == pytest.approx(av.GK_PENALTY * 0.5, abs=1e-3)
    assert all(m["name"] != "Gus Gone" for m in report["missing"])   # departed players are not "missing"


def test_availability_penalty_is_capped():
    model = _availability_model({"Arsenal": {"out": ["Will Winger"]}})
    report = model.team_report("Arsenal")   # striker + winger out = 100% of attack
    assert report["attack_penalty"] == pytest.approx(av.ATTACK_CAP)


def test_availability_overrides_force_players_in():
    report = _availability_model({"Arsenal": {"in": ["Sam Striker"]}}).team_report("Arsenal")
    assert report["attack_penalty"] == pytest.approx(0.0)


def test_availability_multipliers_cross_apply():
    model = _availability_model()
    mult_h, mult_a, _, _ = model.multipliers("Arsenal", "Chelsea")
    assert mult_h < 1.0            # Arsenal attack weakened
    assert mult_a > 1.0            # Chelsea benefit from Arsenal's doubtful keeper


def test_next_round_prefers_fpl_gameweek():
    t0 = pd.Timestamp("2026-10-10 11:30", tz="UTC")
    fixtures = pd.DataFrame({
        "kickoff_utc": [t0, t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=4)],
        "home": ["Arsenal", "Chelsea", "Everton"], "away": ["Leeds United", "Fulham", "Liverpool"],
        "gameweek": [6, 6, 6],
    })
    assert len(ff.next_round(fixtures)) == 3
