"""SHAP explanations for individual fixtures.

``shap.TreeExplainer`` gives exact per-feature contributions to the XGBoost margins
(log-odds, one per outcome class). Margins aren't intuitive, so each contribution is
converted into percentage points of probability:

    dp_k ~ p_k * (phi_jk - sum_c p_c * phi_jc)        (softmax derivative at the prediction)

These are rescaled so they add up to (model probability - baseline probability) per
class; any remainder from the linearisation is reported as "Interactions (non-linear)".
The baseline is the model's average fixture, which already contains the league-wide
home advantage. The calibration step is reported as its own line, so
baseline + drivers + calibration = the probability shown in the UI.

Related features are grouped (e.g. 3- and 5-match xG created), giving statements like
"+7.1 pts to Home win from Team strength (Elo)".
"""

from __future__ import annotations

import logging

import numpy as np

from .tactics import TACTIC_COLUMNS

log = logging.getLogger(__name__)

OUTCOMES = ("home", "draw", "away")

FEATURE_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Team strength (Elo)", ("d_elo", "elo_exp_home")),
    ("Long-run form (last 19)", ("d_xgd19", "d_ppg19")),
    ("Recent points form", ("d_form_pts3", "d_form_pts5")),
    ("xG created (rolling)", ("d_xg_f3", "d_xg_f5")),
    ("xG conceded (rolling)", ("d_xg_a3", "d_xg_a5")),
    ("npxG difference (rolling)", ("d_npxgd3", "d_npxgd5")),
    ("Finishing vs xG", ("d_fin_var3", "d_fin_var5")),
    ("Goals scored / conceded", ("d_gf3", "d_gf5", "d_ga3", "d_ga5")),
    ("Shots & shots on target", ("d_sot_f3", "d_sot_f5", "d_sot_a3", "d_sot_a5",
                                 "d_sh_f3", "d_sh_f5", "d_sh_a3", "d_sh_a5")),
    ("Pressing intensity (PPDA)", ("d_ppda3", "d_ppda5")),
    ("Possession & press resistance", ("d_pass_share5", "d_press_res5")),
    ("Deep completions", ("d_deep_f5", "d_deep_a5")),
    ("Venue form (home at home, away on road)", ("h_venue_pts5", "a_venue_pts5", "h_venue_gf5",
                                                 "a_venue_ga5", "d_venue_xgd5")),
    ("Club-specific home edge", ("h_home_edge", "a_home_edge")),
    ("Rest days", ("d_rest", "h_rest", "a_rest")),
    ("Newly promoted", ("h_promoted", "a_promoted")),
    ("Tactical matchup", tuple(TACTIC_COLUMNS)),
]


def _softmax(m: np.ndarray) -> np.ndarray:
    e = np.exp(m - m.max())
    return e / e.sum()


class Explainer:
    def __init__(self, model, features: list[str]):
        self.model = model
        self.features = list(features)
        self.backend = "xgboost-native"
        self._shap = None
        try:
            import shap
            self._shap = shap.TreeExplainer(model)
            self.backend = f"shap {shap.__version__} TreeExplainer"
        except Exception as exc:  # shap missing/incompatible: XGBoost's built-in TreeSHAP gives the same values
            log.warning("shap unavailable (%s); using XGBoost pred_contribs", exc)
        self._groups = self._group_index()

    def _group_index(self) -> list[tuple[str, list[int]]]:
        used, groups = set(), []
        for label, members in FEATURE_GROUPS:
            idx = [self.features.index(f) for f in members if f in self.features]
            if idx:
                groups.append((label, idx))
                used.update(idx)
        for i, name in enumerate(self.features):
            if i not in used:
                groups.append((name, [i]))
        return groups

    def contributions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(phi with shape (n, n_features, 3), base margins (3,))."""
        X = np.asarray(X, dtype=float)
        if self._shap is not None:
            values = self._shap.shap_values(X)
            if isinstance(values, list):  # older shap: list of (n, f) per class
                values = np.stack(values, axis=-1)
            base = np.asarray(self._shap.expected_value, dtype=float).reshape(-1)
            return np.asarray(values, dtype=float), base
        from xgboost import DMatrix
        contribs = self.model.get_booster().predict(DMatrix(X), pred_contribs=True)  # (n, 3, f+1)
        return contribs[:, :, :-1].transpose(0, 2, 1), contribs[0, :, -1]

    def explain(self, x_row: np.ndarray, calibrated: np.ndarray, top: int | None = 8) -> dict:
        phi, base = self.contributions(np.asarray(x_row, dtype=float).reshape(1, -1))
        phi = phi[0]                                   # (f, 3)
        p0 = _softmax(base)
        p1 = _softmax(base + phi.sum(axis=0))
        w = phi - (phi @ p1)[:, None]
        contrib = w * p1[None, :]                      # linearised dp per feature/class
        totals = contrib.sum(axis=0)
        target = p1 - p0
        scale = np.where((np.abs(totals) > 1e-6) & (np.sign(totals) == np.sign(target)), target / totals, 1.0)
        contrib = contrib * scale[None, :]

        drivers = []
        for label, idx in self._groups:
            vals = contrib[idx].sum(axis=0)
            drivers.append({"label": label, **{o: round(float(v) * 100, 2) for o, v in zip(OUTCOMES, vals)},
                            "features": [self.features[i] for i in idx]})
        residual = target - contrib.sum(axis=0)
        if np.abs(residual).max() > 5e-4:
            drivers.append({"label": "Interactions (non-linear)", "features": [],
                            **{o: round(float(v) * 100, 2) for o, v in zip(OUTCOMES, residual)}})
        calibrated = np.asarray(calibrated, dtype=float)
        favoured = int(np.argmax(calibrated))
        key = OUTCOMES[favoured]
        drivers.sort(key=lambda d: abs(d[key]), reverse=True)
        summary = [f"{d[key]:+.1f} pts {d['label']}" for d in drivers[:4] if abs(d[key]) >= 0.1]
        return {
            "backend": self.backend,
            "favoured": key,
            "baseline": {o: round(float(v), 4) for o, v in zip(OUTCOMES, p0)},
            "model_raw": {o: round(float(v), 4) for o, v in zip(OUTCOMES, p1)},
            "calibration_adjustment": {o: round(float(v) * 100, 2) for o, v in zip(OUTCOMES, calibrated - p1)},
            "drivers": drivers[:top],
            "summary": summary,
        }
