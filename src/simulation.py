"""Scoreline engine: expected-goal regressors + Dixon-Coles bivariate Poisson + Monte Carlo.

1. Two ``XGBRegressor(objective="count:poisson")`` models predict lambda_home and
   lambda_away from the same leak-free pre-match features as the classifier, plus each
   side's absolute attacking/defensive rates (differences alone can't set total goals).
2. Independent Poisson under-predicts 0-0 and 1-1 and over-predicts 1-0 and 0-1. The
   Dixon-Coles correction ``tau`` (dependence parameter ``rho`` fitted by maximum
   likelihood with ``scipy.optimize``) turns the two marginals into a dependent bivariate
   distribution.
3. 10,000 scorelines are sampled from that joint distribution. Exact scores, 1X2,
   over/under lines and BTTS are read off the simulated sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from xgboost import XGBRegressor

log = logging.getLogger(__name__)

MAX_GOALS = 10           # joint pmf support per side (P(>10) is negligible)
DISPLAY_GOALS = 6        # heatmap size (0..5 plus "6+")
N_SIMULATIONS = 10_000
OU_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
RHO_BOUNDS = (-0.2, 0.2)

GOAL_EXTRA_FEATURES = [
    "h_xg_f5", "h_xg_a5", "a_xg_f5", "a_xg_a5", "h_gf5", "h_ga5", "a_gf5", "a_ga5",
    "h_sot_f5", "h_sot_a5", "a_sot_f5", "a_sot_a5", "h_xgd19", "a_xgd19", "h_elo", "a_elo",
    "h_venue_gf5", "a_venue_ga5",
]


# --------------------------------------------------------------- Dixon-Coles
def dc_tau(h: np.ndarray, a: np.ndarray, lh, la, rho: float) -> np.ndarray:
    """Dixon-Coles low-score adjustment factor (1 everywhere except 0-0, 0-1, 1-0, 1-1)."""
    tau = np.ones(np.broadcast(h, a, lh, la).shape)
    tau = np.where((h == 0) & (a == 0), 1 - lh * la * rho, tau)
    tau = np.where((h == 0) & (a == 1), 1 + lh * rho, tau)
    tau = np.where((h == 1) & (a == 0), 1 + la * rho, tau)
    tau = np.where((h == 1) & (a == 1), 1 - rho, tau)
    return np.clip(tau, 1e-9, None)


def fit_rho(home_goals, away_goals, lam_h, lam_a) -> float:
    """MLE of rho given fitted marginal means (the Poisson terms don't depend on rho)."""
    hg, ag = np.asarray(home_goals, int), np.asarray(away_goals, int)
    lh, la = np.asarray(lam_h, float), np.asarray(lam_a, float)
    low = (hg <= 1) & (ag <= 1)

    def nll(rho):
        return -np.sum(np.log(dc_tau(hg[low], ag[low], lh[low], la[low], rho)))

    res = optimize.minimize_scalar(nll, bounds=RHO_BOUNDS, method="bounded")
    return float(res.x)


def score_matrix(lam_h: float, lam_a: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Joint P(home=i, away=j) for i, j in 0..max_goals, normalised."""
    goals = np.arange(max_goals + 1)
    ph = stats.poisson.pmf(goals, lam_h)
    pa = stats.poisson.pmf(goals, lam_a)
    joint = np.outer(ph, pa)
    hh, aa = np.meshgrid(goals, goals, indexing="ij")
    joint *= dc_tau(hh, aa, lam_h, lam_a, rho)
    return joint / joint.sum()


# ------------------------------------------------------------- simulation
@dataclass
class SimulationResult:
    lambda_home: float
    lambda_away: float
    rho: float
    n_sims: int
    home_goals: np.ndarray = field(repr=False)
    away_goals: np.ndarray = field(repr=False)

    @property
    def p_home(self) -> float:
        return float(np.mean(self.home_goals > self.away_goals))

    @property
    def p_draw(self) -> float:
        return float(np.mean(self.home_goals == self.away_goals))

    @property
    def p_away(self) -> float:
        return float(np.mean(self.home_goals < self.away_goals))

    def matrix(self, size: int = DISPLAY_GOALS) -> list[list[float]]:
        """Empirical scoreline grid; the last row/column collects `size`+ goals."""
        h = np.minimum(self.home_goals, size)
        a = np.minimum(self.away_goals, size)
        grid = np.zeros((size + 1, size + 1))
        np.add.at(grid, (h, a), 1)
        return (grid / self.n_sims).round(4).tolist()

    def top_scores(self, k: int = 8) -> list[dict]:
        pairs, counts = np.unique(np.stack([self.home_goals, self.away_goals], axis=1), axis=0, return_counts=True)
        order = np.argsort(-counts)[:k]
        return [{"score": f"{pairs[i][0]}-{pairs[i][1]}", "home": int(pairs[i][0]), "away": int(pairs[i][1]),
                 "prob": round(counts[i] / self.n_sims, 4)} for i in order]

    def best_by_outcome(self) -> dict:
        """Most likely scoreline within each result (the overall mode is usually 1-1)."""
        out = {}
        for key, mask in (("home", self.home_goals > self.away_goals), ("draw", self.home_goals == self.away_goals),
                          ("away", self.home_goals < self.away_goals)):
            if not mask.any():
                out[key] = None
                continue
            pairs, counts = np.unique(np.stack([self.home_goals[mask], self.away_goals[mask]], axis=1), axis=0,
                                      return_counts=True)
            i = int(np.argmax(counts))
            out[key] = {"score": f"{pairs[i][0]}-{pairs[i][1]}", "prob": round(counts[i] / self.n_sims, 4)}
        return out

    def over_under(self) -> dict:
        total = self.home_goals + self.away_goals
        return {str(line): {"over": round(float(np.mean(total > line)), 4),
                            "under": round(float(np.mean(total < line)), 4)} for line in OU_LINES}

    def btts(self) -> dict:
        yes = float(np.mean((self.home_goals > 0) & (self.away_goals > 0)))
        return {"yes": round(yes, 4), "no": round(1 - yes, 4)}

    def to_dict(self) -> dict:
        top = self.top_scores()
        return {
            "n_sims": self.n_sims,
            "lambda_home": round(self.lambda_home, 3),
            "lambda_away": round(self.lambda_away, 3),
            "rho": round(self.rho, 4),
            "probs": {"home": round(self.p_home, 4), "draw": round(self.p_draw, 4), "away": round(self.p_away, 4)},
            "most_likely": top[0],
            "top_scores": top,
            "best_by_outcome": self.best_by_outcome(),
            "matrix": self.matrix(),
            "matrix_labels": [str(i) for i in range(DISPLAY_GOALS)] + [f"{DISPLAY_GOALS}+"],
            "over_under": self.over_under(),
            "btts": self.btts(),
            "clean_sheet": {"home": round(float(np.mean(self.away_goals == 0)), 4),
                            "away": round(float(np.mean(self.home_goals == 0)), 4)},
        }


def simulate(lam_h: float, lam_a: float, rho: float = 0.0, n_sims: int = N_SIMULATIONS,
             seed: int | None = None) -> SimulationResult:
    """Monte Carlo draw of `n_sims` scorelines from the Dixon-Coles bivariate Poisson."""
    lam_h = float(np.clip(lam_h, 0.05, 6.0))
    lam_a = float(np.clip(lam_a, 0.05, 6.0))
    joint = score_matrix(lam_h, lam_a, rho)
    rng = np.random.default_rng(seed)
    idx = rng.choice(joint.size, size=n_sims, p=joint.ravel())
    home, away = np.divmod(idx, joint.shape[1])
    return SimulationResult(lam_h, lam_a, rho, n_sims, home.astype(int), away.astype(int))


def analytic_markets(lam_h: np.ndarray, lam_a: np.ndarray, rho: float) -> pd.DataFrame:
    """Exact (non-simulated) 1X2 / O2.5 / BTTS probabilities; used for fast validation."""
    rows = []
    for lh, la in zip(np.asarray(lam_h, float), np.asarray(lam_a, float)):
        m = score_matrix(lh, la, rho)
        hh, aa = np.indices(m.shape)
        rows.append({
            "p_home": m[hh > aa].sum(), "p_draw": m[hh == aa].sum(), "p_away": m[hh < aa].sum(),
            "p_over25": m[hh + aa > 2.5].sum(), "p_btts": m[(hh > 0) & (aa > 0)].sum(),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- goal models
class GoalModel:
    """lambda_home / lambda_away regressors plus the fitted Dixon-Coles rho."""

    def __init__(self, features: list[str], params: dict, seed: int = 42):
        self.features = list(features)
        self.params = dict(params)
        self.seed = seed
        self.home = self._make()
        self.away = self._make()
        self.rho = 0.0

    def _make(self) -> XGBRegressor:
        return XGBRegressor(objective="count:poisson", tree_method="hist", random_state=self.seed,
                            n_jobs=-1, **self.params)

    def fit(self, frame: pd.DataFrame) -> "GoalModel":
        X = frame[self.features].to_numpy(dtype=float)
        self.home.fit(X, frame["fthg"].to_numpy(dtype=float))
        self.away.fit(X, frame["ftag"].to_numpy(dtype=float))
        lh, la = self.predict(frame)
        self.rho = fit_rho(frame["fthg"], frame["ftag"], lh, la)
        log.debug("Goal model fitted: mean lambda %.2f / %.2f, rho %.3f", lh.mean(), la.mean(), self.rho)
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = frame[self.features].to_numpy(dtype=float)
        return self.home.predict(X), self.away.predict(X)
