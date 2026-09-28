"""API smoke tests against the locally cached data (skipped when no data has been synced).

No network calls: the sync endpoint is not exercised here.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ENABLE_SCHEDULER"] = "0"

from src import data_loader as dl  # noqa: E402

pytestmark = pytest.mark.skipif(not dl.has_minimum_data(), reason="no local data; run `python run_live.py --sync`")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as c:
        deadline = time.time() + 180
        while time.time() < deadline and not c.get("/api/health").json()["ready"]:
            if app_module.service.error:
                pytest.fail(f"service failed to load: {app_module.service.error}")
            time.sleep(0.5)
        assert c.get("/api/health").json()["ready"], "service did not become ready"
        yield c


def test_dashboard_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "EPL Predictor" in r.text and "/static/js/app.js" in r.text


def test_status(client):
    s = client.get("/api/status").json()
    assert s["ready"] and s["model"]["n_train"] > 1000
    assert s["scheduler"]["enabled"] is False  # disabled for tests


def test_gameweek_and_deep_dive(client):
    gw = client.get("/api/fixtures").json()
    assert gw["fixtures"], "expected upcoming fixtures"
    f = gw["fixtures"][0]
    assert sum(f["probs"].values()) == pytest.approx(1.0, abs=1e-6)
    detail = client.get(f"/api/fixtures/{f['id']}").json()
    sim = detail["simulation"]
    assert sim["n_sims"] == 10_000
    assert sum(map(sum, sim["matrix"])) == pytest.approx(1.0, abs=1e-3)
    assert len(detail["radar"]["home"]) == len(detail["radar"]["axes"]) == 6
    assert detail["explain"]["drivers"] and detail["explain"]["summary"]
    assert detail["styles"]["home"] in ("High-Press Possession", "Counter / Low Block", "Direct Transition")


def test_unknown_fixture_404(client):
    assert client.get("/api/fixtures/1999-foo-bar").status_code == 404


def test_simulate_with_odds(client):
    r = client.post("/api/simulate", json={"home": "Spurs", "away": "Man Utd",
                                           "odds": {"home": 3.1, "draw": 3.6, "away": 2.3}})
    assert r.status_code == 200
    d = r.json()
    assert d["home"] == "Tottenham" and d["away"] == "Manchester United"
    sel = {s["selection"]: s for s in d["value"]["selections"]}
    assert sel["home"]["ev"] == pytest.approx(d["probs"]["home"] * 3.1 - 1, abs=1e-3)
    assert d["value"]["markets"]["1X2"]["margin"] > 0


def test_simulate_validation_errors(client):
    r = client.post("/api/simulate", json={"home": "Barcelona", "away": "Arsenal"})
    assert r.status_code == 404 and "suggestions" in r.json()
    assert client.post("/api/simulate", json={"home": "Arsenal", "away": "Arsenal"}).status_code == 422
    assert client.post("/api/simulate", json={"home": "Arsenal", "away": "Chelsea",
                                              "odds": {"home": 0.5}}).status_code == 422


def test_teams_and_metrics(client):
    teams = client.get("/api/teams").json()
    assert len(teams) == 20 and all(t["style"] for t in teams)
    m = client.get("/api/metrics").json()
    assert {"holdout", "backtest", "goals", "tactics"} <= set(m)
