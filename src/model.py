"""Model training, chronological validation and caching.

Components (bundled in :class:`MatchModel`):

* Tactical style clusterer (``src/tactics.py``): adds matchup features.
* 1X2 classifier: ``XGBClassifier`` inside ``CalibratedClassifierCV``. Calibration is
  chronological, never shuffled: a booster trained on the oldest 80% of the window is
  frozen, and the calibrator (temperature scaling, one parameter for all three classes)
  is fitted on the newest 20%. That out-of-time calibration map is then applied to a
  booster refit on the whole window.
* RandomForest baseline (validation only).
* Goal model (``src/simulation.py``): Poisson XGB regressors for lambda_home and lambda_away,
  plus a Dixon-Coles rho.

Validation is out-of-time: the most recent completed season is predicted by models
trained only on earlier seasons, and the current season's completed matches by models
trained on all previous seasons. The live model is then refit on every completed match.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from . import bet_evaluator as bets
from . import config
from . import evaluate as ev
from .explainer import Explainer
from .features import FEATURES
from .simulation import GOAL_EXTRA_FEATURES, GoalModel, analytic_markets
from .tactics import TACTIC_COLUMNS, TacticalStyles

log = logging.getLogger(__name__)

MODEL_VERSION = "2.0"
BUNDLE_PATH = config.MODELS_DIR / "predictor.joblib"
CLASS_FEATURES = list(FEATURES.columns) + TACTIC_COLUMNS
GOAL_FEATURES = CLASS_FEATURES + [c for c in GOAL_EXTRA_FEATURES if c not in CLASS_FEATURES]


@dataclass
class ModelConfig:
    # Shallow, heavily regularised trees: ~2k matches of a noisy 3-way outcome.
    # Chosen by out-of-time backtests on 2023-24, 2024-25 and 2025-26.
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 250, "learning_rate": 0.02, "max_depth": 2, "min_child_weight": 40,
        "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 5.0, "reg_alpha": 0.5,
    })
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 500, "max_depth": 6, "min_samples_leaf": 20, "max_features": 0.3,
    })
    goal_params: dict = field(default_factory=lambda: {
        "n_estimators": 300, "learning_rate": 0.02, "max_depth": 2, "min_child_weight": 40,
        "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 5.0, "max_delta_step": 0.7,
    })
    # Backtests: chronological temperature scaling + refit 0.984 log loss, raw booster 0.986,
    # chronological sigmoid 0.993 (three one-vs-rest sigmoids overfit ~350 calibration rows).
    calibration: str = "temperature"
    calibration_fraction: float = 0.2   # newest share of the training window used to calibrate
    refit_after_calibration: bool = True
    seed: int = 42


def make_xgb(cfg: ModelConfig) -> XGBClassifier:
    return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
                         random_state=cfg.seed, n_jobs=-1, **cfg.xgb_params)


def make_rf(cfg: ModelConfig):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(random_state=cfg.seed, n_jobs=-1, **cfg.rf_params))


def fit_calibrated(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> tuple[CalibratedClassifierCV, XGBClassifier]:
    """Chronological calibration: rows must be in date order (no shuffling anywhere).

    1. booster_old = XGB fitted on the oldest (1 - f) of the window
    2. CalibratedClassifierCV(FrozenEstimator(booster_old)) fitted on the newest f, so the
       calibration map measures how the model extrapolates forward in time
    3. (refit) the same map is applied to a booster refit on the full window
    """
    cut = int(len(y) * (1 - cfg.calibration_fraction))
    if cut < 200 or len(y) - cut < 100 or len(np.unique(y[cut:])) < 3:
        raise ValueError("not enough chronological data to calibrate")
    booster = make_xgb(cfg).fit(X[:cut], y[:cut])
    calibrated = CalibratedClassifierCV(FrozenEstimator(booster), method=cfg.calibration)
    calibrated.fit(X[cut:], y[cut:])
    if cfg.refit_after_calibration:
        booster = make_xgb(cfg).fit(X, y)
        calibrated.calibrated_classifiers_[0].estimator = FrozenEstimator(booster)
    return calibrated, booster


class MatchModel:
    """Tactics + calibrated classifier + goal model, trained on one chronological window."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.class_features = list(CLASS_FEATURES)
        self.goal_features = list(GOAL_FEATURES)
        self.tactics: TacticalStyles | None = None
        self.classifier: CalibratedClassifierCV | None = None
        self.booster: XGBClassifier | None = None
        self.goals: GoalModel | None = None
        self.base_rates: np.ndarray | None = None
        self._explainer: Explainer | None = None

    def fit(self, train: pd.DataFrame, with_goals: bool = True) -> "MatchModel":
        train = train.sort_values("date", kind="mergesort")
        if len(np.unique(train["y"])) < 3:
            raise ValueError("training data must contain home wins, draws and away wins")
        self.tactics = TacticalStyles(seed=self.cfg.seed).fit(train)
        frame = self.tactics.transform(train)
        X, y = frame[self.class_features].to_numpy(float), frame["y"].to_numpy(int)
        self.classifier, self.booster = fit_calibrated(X, y, self.cfg)
        self.base_rates = np.bincount(y, minlength=3) / len(y)
        if with_goals:
            self.goals = GoalModel(self.goal_features, self.cfg.goal_params, self.cfg.seed).fit(frame)
        return self

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.tactics.transform(frame)

    def predict_proba(self, prepared: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(prepared[self.class_features].to_numpy(float))

    def predict_goals(self, prepared: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.goals.predict(prepared)

    @property
    def explainer(self) -> Explainer:
        if self._explainer is None:
            self._explainer = Explainer(self.booster, self.class_features)
        return self._explainer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_explainer"] = None  # rebuilt lazily; SHAP explainers don't need to be pickled
        return state


MODEL_LABELS = {
    "random_forest": "RandomForest (baseline)",
    "xgboost": "XGBoost (uncalibrated)",
    "xgboost_calibrated": "XGBoost + calibration (live model)",
    "poisson": "Poisson simulation (1X2)",
    "base_rate": "Base rate",
    "bookmakers": "Bookmakers (avg closing odds)",
    "bet365": "Bet365 (pre-match, margin removed)",
}


def _market_probs(frame: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    odds = frame[cols].to_numpy(float)
    valid = np.all(np.isfinite(odds) & (odds > 1.0), axis=1)
    probs = np.full(odds.shape, np.nan)
    for i in np.where(valid)[0]:
        probs[i] = bets.remove_margin(odds[i], "shin")
    return probs, valid


def _score_block(test: pd.DataFrame, probs: dict[str, np.ndarray], y_train) -> dict:
    y = test["y"].to_numpy(int)
    out = {name: ev.score(y, p) for name, p in probs.items()}
    out["base_rate"] = ev.score(y, ev.base_rate_probs(y_train, len(y)))
    for name, cols in (("bookmakers", ["odds_h", "odds_d", "odds_a"]), ("bet365", ["b365_h", "b365_d", "b365_a"])):
        book, valid = _market_probs(test, cols)
        if valid.sum() >= 20:
            out[name] = ev.score(y[valid], book[valid])
    book, valid = _market_probs(test, ["b365_h", "b365_d", "b365_a"])
    if valid.sum() >= 20 and "xgboost_calibrated" in probs:
        out["xgboost_calibrated_on_odds_subset"] = ev.score(y[valid], probs["xgboost_calibrated"][valid])
    return out


def _goal_block(test: pd.DataFrame, model: MatchModel, train: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    prepared = model.prepare(test)
    lh, la = model.predict_goals(prepared)
    markets = analytic_markets(lh, la, model.goals.rho)
    hg, ag = test["fthg"].to_numpy(float), test["ftag"].to_numpy(float)
    over = ((hg + ag) > 2.5).astype(int)
    btts = ((hg > 0) & (ag > 0)).astype(int)
    base_over = float(((train["fthg"] + train["ftag"]) > 2.5).mean())
    base_btts = float(((train["fthg"] > 0) & (train["ftag"] > 0)).mean())
    out = {
        "rho": round(model.goals.rho, 4),
        "mean_lambda": {"home": round(float(lh.mean()), 3), "away": round(float(la.mean()), 3)},
        "mean_goals": {"home": round(float(hg.mean()), 3), "away": round(float(ag.mean()), 3)},
        "poisson_deviance": {"home": round(ev.poisson_deviance(hg, lh), 4),
                             "away": round(ev.poisson_deviance(ag, la), 4),
                             "home_baseline": round(ev.poisson_deviance(hg, np.full_like(hg, train["fthg"].mean())), 4),
                             "away_baseline": round(ev.poisson_deviance(ag, np.full_like(ag, train["ftag"].mean())), 4)},
        "over25": {"model": ev.binary_score(over, markets["p_over25"].to_numpy()),
                   "base_rate": ev.binary_score(over, np.full(len(over), base_over))},
        "btts": {"model": ev.binary_score(btts, markets["p_btts"].to_numpy()),
                 "base_rate": ev.binary_score(btts, np.full(len(btts), base_btts))},
    }
    ou_book, ou_valid = _market_probs(test, ["b365_over25", "b365_under25"])
    if ou_valid.sum() >= 20:
        out["over25"]["bet365"] = ev.binary_score(over[ou_valid], ou_book[ou_valid, 0])
        out["over25"]["model_on_odds_subset"] = ev.binary_score(over[ou_valid],
                                                                markets["p_over25"].to_numpy()[ou_valid])
    return out, markets


def _season_predictions(train: pd.DataFrame, test: pd.DataFrame, cfg: ModelConfig) -> dict:
    model = MatchModel(cfg).fit(train)
    prepared = model.prepare(test)
    lh, la = model.predict_goals(prepared)
    return {"model": model, "probs": model.predict_proba(prepared),
            "markets": analytic_markets(lh, la, model.goals.rho)}


def _rolling_backtest(trainable: pd.DataFrame, seasons: list[int], cfg: ModelConfig, cache: dict) -> dict:
    """Walk-forward over several completed seasons, each predicted by a model trained only on
    the seasons before it. One season is too noisy to judge betting ROI."""
    frames, probs, over_p = [], [], []
    for season in seasons:
        train = trainable[trainable["season"] < season]
        test = trainable[trainable["season"] == season]
        if season not in cache:
            cache[season] = _season_predictions(train, test, cfg)
        frames.append(test)
        probs.append(cache[season]["probs"])
        over_p.append(cache[season]["markets"]["p_over25"].to_numpy())
    test = pd.concat(frames)
    probs = np.vstack(probs)
    p_over = np.concatenate(over_p)
    y = test["y"].to_numpy(int)
    over = ((test["fthg"] + test["ftag"]) > 2.5).to_numpy()
    b365, valid = _market_probs(test, ["b365_h", "b365_d", "b365_a"])
    ou_b365, ou_valid = _market_probs(test, ["b365_over25", "b365_under25"])
    earlier = trainable[trainable["season"] < min(seasons)]
    base_over = np.full(len(over), float(((earlier["fthg"] + earlier["ftag"]) > 2.5).mean()))
    return {
        "seasons": [config.season_label(s) for s in seasons],
        "matches": int(len(test)),
        "classifier": ev.score(y, probs),
        "base_rate": ev.score(y, ev.base_rate_probs(earlier["y"], len(y))),
        "bet365": ev.score(y[valid], b365[valid]) if valid.any() else None,
        "over25": {"model": ev.binary_score(over, p_over), "base_rate": ev.binary_score(over, base_over),
                   "bet365": ev.binary_score(over[ou_valid], ou_b365[ou_valid, 0]) if ou_valid.any() else None},
        "value_bets": {
            "threshold": bets.VALUE_THRESHOLD,
            "1x2": bets.backtest(probs, test[["b365_h", "b365_d", "b365_a"]].to_numpy(float), y),
            "over_under_2_5": bets.backtest(np.column_stack([p_over, 1 - p_over]),
                                            test[["b365_over25", "b365_under25"]].to_numpy(float),
                                            np.where(over, 0, 1)),
        },
    }


def validate(dataset: pd.DataFrame, plan: dict, cfg: ModelConfig, make_plots: bool = True) -> dict:
    trainable = dataset[~dataset["season"].isin(plan["warmup"])].sort_values("date", kind="mergesort")
    holdout = plan["history"][-1]
    report: dict = {"holdout_season": config.season_label(holdout)}

    train = trainable[trainable["season"] < holdout]
    test = trainable[trainable["season"] == holdout]
    report["holdout_train_seasons"] = [config.season_label(s) for s in sorted(train["season"].unique())]
    log.info("Validating: train %s (%d matches) -> test %s (%d matches)",
             " + ".join(report["holdout_train_seasons"]), len(train), report["holdout_season"], len(test))

    cache = {holdout: _season_predictions(train, test, cfg)}
    model = cache[holdout]["model"]
    prep_train, prep_test = model.prepare(train), model.prepare(test)
    X_tr, X_te = prep_train[CLASS_FEATURES].to_numpy(float), prep_test[CLASS_FEATURES].to_numpy(float)
    y_tr, y_te = prep_train["y"].to_numpy(int), prep_test["y"].to_numpy(int)
    probs = {
        "random_forest": make_rf(cfg).fit(X_tr, y_tr).predict_proba(X_te),
        "xgboost": make_xgb(cfg).fit(X_tr, y_tr).predict_proba(X_te),
        "xgboost_calibrated": cache[holdout]["probs"],
    }
    goals, markets = _goal_block(test, model, train)
    probs["poisson"] = markets[["p_home", "p_draw", "p_away"]].to_numpy()
    report["holdout"] = _score_block(test, probs, train["y"])
    report["holdout_confusion"] = ev.confusion(y_te, probs["xgboost_calibrated"]).tolist()
    report["goals"] = goals
    report["tactics"] = model.tactics.describe()

    # Walk-forward backtest over up to three seasons, each with >= 2 earlier training seasons.
    seasons = [s for s in plan["history"] if s - plan["history"][0] >= 2][-3:] or [holdout]
    log.info("Walk-forward backtest over %s", ", ".join(config.season_label(s) for s in seasons))
    report["backtest"] = _rolling_backtest(trainable, seasons, cfg, cache)

    if make_plots:
        book, valid = _market_probs(test, ["odds_h", "odds_d", "odds_a"])
        curves = {MODEL_LABELS[k]: v for k, v in probs.items() if k != "poisson"}
        curves[MODEL_LABELS["bookmakers"]] = np.where(valid[:, None], book, np.nan)
        cal_path = ev.plot_calibration(y_te, curves, config.REPORTS_DIR / "calibration_holdout.png",
                                       f"Calibration: hold-out season {report['holdout_season']}")
        cm_path = ev.plot_confusion(y_te, probs["xgboost_calibrated"], config.REPORTS_DIR / "confusion_holdout.png",
                                    f"XGBoost+cal, {report['holdout_season']}")
        report["plots"] = [str(p.relative_to(config.ROOT)) for p in (cal_path, cm_path) if p]

    current = trainable[trainable["season"] == plan["current"]]
    if len(current) >= 10:
        past = trainable[trainable["season"] < plan["current"]]
        live = MatchModel(cfg).fit(past, with_goals=False)
        report["current_season"] = _score_block(current, {"xgboost_calibrated": live.predict_proba(live.prepare(current))},
                                                past["y"])
        report["current_season_label"] = config.season_label(plan["current"])
    return report


def data_fingerprint(dataset: pd.DataFrame, cfg: ModelConfig, plan: dict) -> str:
    digest = hashlib.sha1()
    digest.update(MODEL_VERSION.encode())
    digest.update(json.dumps(asdict(cfg), sort_keys=True).encode())
    digest.update(json.dumps(plan, sort_keys=True).encode())
    digest.update(json.dumps(GOAL_FEATURES).encode())
    cols = sorted(set(GOAL_FEATURES) - set(TACTIC_COLUMNS)) + ["y", "fthg", "ftag", "season", "odds_h", "b365_h",
                                                               "b365_over25"]
    # Rounded so float noise in the last bits never forces a pointless retrain.
    digest.update(pd.util.hash_pandas_object(dataset[cols].round(8), index=False).to_numpy().tobytes())
    return digest.hexdigest()[:16]


@dataclass
class PredictorBundle:
    model: MatchModel
    fingerprint: str
    trained_at: str
    trained_through: str
    n_train: int
    train_seasons: list[str]
    validation: dict
    config: dict
    version: str = MODEL_VERSION

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        probs = self.model.predict_proba(self.model.prepare(features))
        return pd.DataFrame(probs, columns=["p_home", "p_draw", "p_away"], index=features.index)


def train_live_model(dataset: pd.DataFrame, plan: dict, cfg: ModelConfig, fingerprint: str,
                     make_plots: bool = True) -> PredictorBundle:
    validation = validate(dataset, plan, cfg, make_plots=make_plots)
    train = dataset[~dataset["season"].isin(plan["warmup"])]
    log.info("Fitting live model on %d completed matches (%s -> %s)", len(train),
             config.season_label(int(train["season"].min())), pd.Timestamp(train["date"].max()).date())
    model = MatchModel(cfg).fit(train)
    return PredictorBundle(
        model=model,
        fingerprint=fingerprint,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        trained_through=str(pd.Timestamp(train["date"].max()).date()),
        n_train=int(len(train)),
        train_seasons=[config.season_label(s) for s in sorted(train["season"].unique())],
        validation=validation,
        config=asdict(cfg),
    )


def load_or_train(dataset: pd.DataFrame, plan: dict, cfg: ModelConfig | None = None, *, force: bool = False,
                  make_plots: bool = True) -> tuple[PredictorBundle, bool]:
    """Reuse the cached model if the training data is unchanged; otherwise retrain.

    Returns (bundle, retrained).
    """
    cfg = cfg or ModelConfig()
    fingerprint = data_fingerprint(dataset, cfg, plan)
    if BUNDLE_PATH.exists() and not force:
        try:
            bundle: PredictorBundle = joblib.load(BUNDLE_PATH)
            if getattr(bundle, "version", None) == MODEL_VERSION and bundle.fingerprint == fingerprint:
                log.info("Training data unchanged since %s; reusing cached model", bundle.trained_at)
                return bundle, False
            log.info("New completed matches or settings detected; retraining")
        except Exception as exc:  # corrupt or incompatible pickle -> retrain
            log.warning("Could not load cached model (%s); retraining", exc)
    bundle = train_live_model(dataset, plan, cfg, fingerprint, make_plots=make_plots)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(bundle.validation, indent=2))
    return bundle, True
