"""Market margin removal, Expected Value and value-bet backtesting.

EV of a 1-unit stake at decimal odds ``o`` when the model says the outcome has
probability ``p``:

    EV = p * o - 1

A selection is flagged as value when ``EV > VALUE_THRESHOLD`` (5%).

Bookmaker prices include a margin (overround): the implied probabilities 1/o sum to
more than 1. "Fair" probabilities remove it using one of three methods:

* ``proportional``: divide by the booksum (simple, over-penalises favourites)
* ``shin``: Shin (1993) insider-trading model; accounts for favourite-longshot bias
* ``power``: find k with sum (1/o)^k = 1

Fair probabilities are shown next to the model's so "edge" is visible, but EV is always
computed against the actual price you would be paid.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize

VALUE_THRESHOLD = 0.05
KELLY_FRACTION = 0.25  # quarter Kelly, shown for information only
LONGSHOT_ODDS = 5.0    # above this, flags carry a caution: the favourite-longshot bias bites hardest


def overround(odds) -> float:
    odds = np.asarray(odds, dtype=float)
    return float(np.sum(1.0 / odds) - 1.0)


def remove_margin(odds, method: str = "shin") -> np.ndarray:
    """Fair probabilities from a complete market's decimal odds."""
    odds = np.asarray(odds, dtype=float)
    if odds.ndim != 1 or len(odds) < 2 or np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        raise ValueError("need a complete market of decimal odds > 1")
    q = 1.0 / odds
    booksum = q.sum()
    if method == "proportional" or booksum <= 1.0:
        return q / booksum
    if method == "power":
        k = optimize.brentq(lambda k: np.sum(q ** k) - 1.0, 1.0, 10.0)
        p = q ** k
        return p / p.sum()
    if method == "shin":
        def total(z):
            return np.sum((np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / booksum) - z) / (2 * (1 - z))) - 1.0
        try:
            z = optimize.brentq(total, 0.0, 0.4)
        except ValueError:
            return q / booksum
        p = (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / booksum) - z) / (2 * (1 - z))
        return p / p.sum()
    raise ValueError(f"unknown margin method '{method}'")


def expected_value(prob: float, odds: float) -> float:
    return prob * odds - 1.0


def kelly(prob: float, odds: float, fraction: float = KELLY_FRACTION) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, fraction * (prob * odds - 1.0) / b)


def _valid(o) -> bool:
    return o is not None and isinstance(o, (int, float)) and math.isfinite(o) and o > 1.0


def evaluate_fixture(probs: dict, markets: dict, threshold: float = VALUE_THRESHOLD,
                     method: str = "shin") -> dict:
    """EV for every priced selection.

    probs:   {"home","draw","away","over25","under25"} -> model probability
    markets: {"home","draw","away","over25","under25"} -> decimal odds (missing = unpriced)
    """
    selections = []
    summary = {}
    groups = {"1X2": ("home", "draw", "away"), "Total goals 2.5": ("over25", "under25")}
    labels = {"home": "Home win", "draw": "Draw", "away": "Away win", "over25": "Over 2.5", "under25": "Under 2.5"}
    for market, keys in groups.items():
        prices = [markets.get(k) for k in keys]
        if not all(_valid(o) for o in prices):
            continue
        fair = remove_margin(prices, method)
        summary[market] = {"margin": round(overround(prices), 4), "method": method,
                           "fair": {k: round(float(f), 4) for k, f in zip(keys, fair)}}
        for key, price, fair_p in zip(keys, prices, fair):
            p = probs.get(key)
            if p is None:
                continue
            ev = expected_value(p, price)
            selections.append({
                "market": market, "selection": key, "label": labels[key], "odds": round(float(price), 3),
                "model_prob": round(float(p), 4), "fair_prob": round(float(fair_p), 4),
                "edge": round(float(p - fair_p), 4), "ev": round(float(ev), 4),
                "kelly": round(kelly(p, price), 4), "is_value": bool(ev > threshold),
                "longshot": bool(price > LONGSHOT_ODDS),
            })
    return {"threshold": threshold, "markets": summary, "selections": selections,
            "value": [s for s in selections if s["is_value"]]}



def backtest(probs: np.ndarray, odds: np.ndarray, outcomes: np.ndarray,
             threshold: float = VALUE_THRESHOLD) -> dict:
    """Flat 1-unit stake on every selection with EV > threshold (exactly what the UI flags).

    probs/odds: (n, k) arrays, outcomes: (n,) index of the winning selection.
    Selections with missing odds are skipped. Returns ROI with its standard error and a
    breakdown by odds band.
    """
    probs = np.asarray(probs, float)
    odds = np.asarray(odds, float)
    outcomes = np.asarray(outcomes, int)
    priced = np.isfinite(odds) & (odds > 1.0)
    ev = np.where(priced, probs * np.where(priced, odds, 1.0) - 1.0, -np.inf)
    rows, cols = np.where(ev > threshold)
    n_priced = int(priced.all(axis=1).sum())
    if len(rows) == 0:
        return {"bets": 0, "matches": n_priced, "roi": None, "roi_se": None, "profit": 0.0, "hit_rate": None,
                "avg_odds": None, "avg_ev": None, "bands": []}
    price = odds[rows, cols]
    won = outcomes[rows] == cols
    pnl = np.where(won, price - 1.0, -1.0)
    bands = []
    for lo, hi in ((1.0, 2.0), (2.0, 3.0), (3.0, LONGSHOT_ODDS), (LONGSHOT_ODDS, 1000.0)):
        m = (price > lo) & (price <= hi)
        if m.any():
            bands.append({"odds": f"{lo:g}-{hi:g}" if hi < 1000 else f">{lo:g}", "bets": int(m.sum()),
                          "roi": round(float(pnl[m].mean()), 4)})
    return {
        "bets": int(len(pnl)), "matches": n_priced, "hit_rate": round(float(won.mean()), 4),
        "profit": round(float(pnl.sum()), 2), "roi": round(float(pnl.mean()), 4),
        "roi_se": round(float(pnl.std(ddof=1) / np.sqrt(len(pnl))), 4) if len(pnl) > 1 else None,
        "avg_odds": round(float(price.mean()), 3), "avg_ev": round(float(ev[rows, cols].mean()), 4),
        "bands": bands,
    }


def market_odds_from_row(row: pd.Series | dict) -> dict:
    """Bet365 prices from a football-data row (keys may be missing)."""
    get = row.get if isinstance(row, dict) else row.get
    out = {}
    for key, col in (("home", "b365_h"), ("draw", "b365_d"), ("away", "b365_a"),
                     ("over25", "b365_over25"), ("under25", "b365_under25")):
        value = get(col)
        if value is not None and pd.notna(value):
            out[key] = float(value)
    return out
