"""Prediction service shared by the web app (app.py) and the CLI (run_live.py).

A :class:`PredictorService` owns an immutable :class:`State` snapshot (data, features,
model, fixtures, availability). ``refresh()`` builds a new snapshot and swaps it in
atomically, so API requests keep being served from the old snapshot while a sync runs.

Per-fixture inference:
    FeatureBuilder (leak-free, pre-kickoff) -> tactics -> calibrated 1X2 classifier
    -> goal regressors (lambda_home, lambda_away) -> availability modifier on the lambdas
    -> 10,000-run Dixon-Coles Monte Carlo -> availability-adjusted 1X2
    -> EV vs Bet365 (or user-supplied) odds -> SHAP drivers -> radar profile.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from . import availability as av
from . import bet_evaluator as bets
from . import config
from . import data_loader as dl
from . import fixture_fetcher as ff
from .align_teams import resolve_user_team, short_name
from .features import FEATURES, FeatureBuilder, training_frame
from .model import PredictorBundle, load_or_train
from .simulation import N_SIMULATIONS, analytic_markets, simulate
from .utils import DataSourceError

log = logging.getLogger(__name__)

RADAR_AXES = [
    # (label, feature suffix, higher_is_better)
    ("xG created", "xg_f5", True),
    ("xG conceded (inv.)", "xg_a5", False),
    ("Press resistance", "press_res5", True),
    ("Pressing intensity", "ppda5", False),
    ("Possession share", "pass_share5", True),
    ("Finishing vs xG", "fin_var5", True),
]


class NotReady(RuntimeError):
    """The service has no model/data snapshot yet (first load or sync in progress)."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def fixture_id(season: int, home: str, away: str) -> str:
    return f"{season}-{_slug(home)}-{_slug(away)}"


def _clean(obj):
    """JSON-safe: numpy scalars -> Python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    return obj


def _confidence(p: float) -> str:
    return "high" if p >= 0.60 else "medium" if p >= 0.45 else "low"


@dataclass
class State:
    plan: dict
    matches: pd.DataFrame
    builder: FeatureBuilder
    dataset: pd.DataFrame
    bundle: PredictorBundle
    retrained: bool
    future: pd.DataFrame
    pending: pd.DataFrame
    availability: av.AvailabilityModel
    next_gameweek: int | None
    radar_reference: dict[str, np.ndarray]
    current_teams: list[str]
    built_at: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    cache: dict = field(default_factory=dict)


class PredictorService:
    def __init__(self, n_history: int = config.DEFAULT_HISTORY_SEASONS, make_plots: bool = True):
        self.n_history = n_history
        self.make_plots = make_plots
        self._state: State | None = None
        self._refresh_lock = threading.Lock()
        self.loading = False
        self.error: str | None = None

    # ------------------------------------------------------------ lifecycle
    @property
    def state(self) -> State:
        if self._state is None:
            raise NotReady(self.error or "model is loading")
        return self._state

    @property
    def ready(self) -> bool:
        return self._state is not None

    def ensure_data(self) -> None:
        if not dl.has_minimum_data(self.n_history):
            log.info("No local data found; running an initial sync")
            dl.sync_all(self.n_history)
            if not dl.has_minimum_data(self.n_history):
                raise DataSourceError("historical data is missing and could not be downloaded")

    def refresh(self, retrain: bool = False) -> State:
        """Rebuild data/features/model/fixtures and swap the snapshot in atomically."""
        with self._refresh_lock:
            self.loading = True
            try:
                self.ensure_data()
                state = self._build_state(retrain)
                self._state = state
                self.error = None
                return state
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.loading = False

    def sync(self, retrain: bool = False, refresh_history: bool = False) -> dict:
        before = len(self._state.matches) if self._state is not None else None
        report = dl.sync_all(self.n_history, refresh_history=refresh_history)
        state = self.refresh(retrain=retrain)
        return {
            "sources": [{"name": r.name, "status": r.status, "detail": r.detail} for r in report.results
                        if r.status != "cached"],
            "failures": len(report.failures),
            "completed_matches": int(len(state.matches)),
            "new_results": None if before is None else int(len(state.matches) - before),
            "retrained": state.retrained,
        }

    def _build_state(self, retrain: bool) -> State:
        plan = config.season_plan(self.n_history)
        matches = dl.build_match_table(self.n_history)
        championship = dl.load_championship_profiles(plan["championship"])
        builder = FeatureBuilder(matches, championship)
        dataset = training_frame(builder)
        dataset.to_csv(config.PROCESSED_DIR / "training_features.csv", index=False)
        bundle, retrained = load_or_train(dataset, plan, force=retrain, make_plots=self.make_plots)

        future, pending = ff.upcoming(matches, season=plan["current"])
        bootstrap, _ = dl.load_fpl()
        squads = ff.fpl_players(bootstrap)
        availability = av.AvailabilityModel(squads, dl.load_understat_players(plan["current"]),
                                            dl.load_understat_players(plan["current"] - 1), av.load_overrides())
        recent = dataset[dataset["season"] >= plan["current"] - 2]
        reference = {suffix: np.sort(np.concatenate([recent[f"h_{suffix}"].to_numpy(float),
                                                     recent[f"a_{suffix}"].to_numpy(float)]))
                     for _, suffix, _ in RADAR_AXES}
        current = matches[matches["season"] == plan["current"]]
        teams = sorted(set(current["home"]) | set(current["away"]) | set(future["home"]) | set(future["away"]))
        state = State(plan=plan, matches=matches, builder=builder, dataset=dataset, bundle=bundle,
                      retrained=retrained, future=future, pending=pending, availability=availability,
                      next_gameweek=ff.fpl_next_gameweek(bootstrap), radar_reference=reference,
                      current_teams=teams, built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        log.info("Snapshot ready: %d completed matches, %d upcoming fixtures, model trained through %s",
                 len(matches), len(future), bundle.trained_through)
        return state

    # ------------------------------------------------------------- queries
    def status(self) -> dict:
        info = {"ready": self.ready, "loading": self.loading, "error": self.error}
        last = dl.last_sync_time()
        info["last_sync"] = last.isoformat() if last else None
        if self._state is not None:
            s, b = self._state, self._state.bundle
            current = s.matches[s.matches["season"] == s.plan["current"]]
            info.update({
                "season": config.season_label(s.plan["current"]),
                "train_seasons": b.train_seasons,
                "completed_matches": int(len(s.matches)),
                "current_season_matches": int(len(current)),
                "latest_result": str(current["date"].max().date()) if len(current) else None,
                "upcoming_fixtures": int(len(s.future)),
                "next_gameweek": s.next_gameweek,
                "model": {"trained_at": b.trained_at, "trained_through": b.trained_through, "n_train": b.n_train,
                          "version": b.version, "calibration": b.config.get("calibration")},
                "snapshot_built_at": s.built_at,
                "availability_enabled": s.availability.enabled,
            })
        return _clean(info)

    def teams(self, include_all: bool = False) -> list[dict]:
        s = self.state
        names = s.builder.teams if include_all else s.current_teams
        key = ("teams", include_all)
        if key not in s.cache:
            today = pd.Timestamp(date.today())
            # Only the home side's pre-match profile is used; the opponent is arbitrary.
            fx = pd.DataFrame({"date": today, "home": names, "away": names[1:] + names[:1]})
            with s.lock:
                feats = s.builder.build(fx)
            prepared = s.bundle.model.prepare(feats)
            s.cache[key] = [{"name": n, "short": short_name(n), "style": prepared.loc[i, "h_style"],
                             "in_current_season": n in s.current_teams,
                             "profile": {"ppda": round(float(feats.loc[i, "h_ppda10"]), 2),
                                         "pass_share": round(float(feats.loc[i, "h_pass_share10"]), 3),
                                         "directness": round(float(feats.loc[i, "h_directness10"]), 2)}}
                            for i, n in enumerate(names)]
        return s.cache[key]

    def resolve(self, name: str) -> str:
        return resolve_user_team(name, self.state.builder.teams)

    def gameweek(self, days: int | None = None) -> dict:
        s = self.state
        fixtures = ff.within_days(s.future, days) if days else ff.next_round(s.future)
        preds = self._predict_fixtures(fixtures)
        gw = ff.round_number(s.matches, fixtures, s.plan["current"]) if not days else None
        return _clean({
            "season": config.season_label(s.plan["current"]),
            "gameweek": gw,
            "label": f"Next {days} days" if days else (f"Gameweek {gw}" if gw else "Next round"),
            "fixtures": [self._summary(p) for p in preds],
            "pending": [{"home": h, "away": a, "kickoff_utc": k} for h, a, k in
                        zip(s.pending["home"], s.pending["away"], s.pending["kickoff_utc"])],
            "value_bets": sum(len(p["value"]["value"]) for p in preds),
        })

    def fixture(self, fid: str) -> dict:
        s = self.state
        if fid in s.cache:
            return s.cache[fid]
        ids = [fixture_id(se, h, a) for se, h, a in zip(s.future["season"], s.future["home"], s.future["away"])]
        if fid not in ids:
            raise KeyError(fid)
        return self._predict_fixtures(s.future.iloc[[ids.index(fid)]])[0]

    def predict(self, home: str, away: str, on: date | None = None, odds: dict | None = None,
                apply_availability: bool = True, n_sims: int = N_SIMULATIONS) -> dict:
        """On-demand prediction for any two clubs (the simulator tool and --match)."""
        s = self.state
        home, away = self.resolve(home), self.resolve(away)
        if home == away:
            raise ValueError("home and away team must differ")
        scheduled = s.future[(s.future["home"] == home) & (s.future["away"] == away)]
        if on is None and not scheduled.empty:
            fx = scheduled.iloc[[0]].copy()
        else:
            day = pd.Timestamp(on or date.today())
            fx = pd.DataFrame([{"date": day, "home": home, "away": away, "kickoff_utc": pd.NaT,
                                "season": config.season_start_year(day)}])
        if odds:
            for key, col in (("home", "b365_h"), ("draw", "b365_d"), ("away", "b365_a"),
                             ("over25", "b365_over25"), ("under25", "b365_under25")):
                if odds.get(key):
                    fx[col] = float(odds[key])
        pred = self._predict_fixtures(fx, apply_availability=apply_availability, n_sims=n_sims,
                                      use_cache=False, odds_source="user" if odds else None)[0]
        pred["scheduled"] = not scheduled.empty
        return pred

    # ------------------------------------------------------------ inference
    def _predict_fixtures(self, fixtures: pd.DataFrame, apply_availability: bool = True,
                          n_sims: int = N_SIMULATIONS, use_cache: bool = True,
                          odds_source: str | None = None) -> list[dict]:
        s = self.state
        if fixtures.empty:
            return []
        fixtures = fixtures.reset_index(drop=True).copy()
        if "season" not in fixtures:
            fixtures["season"] = [config.season_start_year(d) for d in fixtures["date"]]
        ids = [fixture_id(se, h, a) for se, h, a in zip(fixtures["season"], fixtures["home"], fixtures["away"])]
        todo = [i for i, fid in enumerate(ids) if not (use_cache and fid in s.cache)]
        fresh: dict[str, dict] = {}
        if todo:
            batch = fixtures.iloc[todo].reset_index(drop=True)
            with s.lock:  # FeatureBuilder keeps small caches; serialise builds across API threads
                feats = s.builder.build(batch)
            model = s.bundle.model
            prepared = model.prepare(feats)
            probs = model.predict_proba(prepared)
            lam_h, lam_a = model.predict_goals(prepared)
            X = prepared[model.class_features].to_numpy(float)
            for j, i in enumerate(todo):
                fresh[ids[i]] = self._one(s, ids[i], batch.iloc[j], prepared.iloc[j], X[j], probs[j], lam_h[j],
                                          lam_a[j], apply_availability, n_sims, odds_source)
            if use_cache:
                s.cache.update(fresh)
        return [fresh[fid] if fid in fresh else s.cache[fid] for fid in ids]

    def _one(self, s: State, fid: str, fx: pd.Series, row: pd.Series, x: np.ndarray, p_model: np.ndarray,
             lam_h: float, lam_a: float, apply_availability: bool, n_sims: int, odds_source: str | None) -> dict:
        home, away = fx["home"], fx["away"]
        model = s.bundle.model
        rho = model.goals.rho

        # FPL news describes the *next* gameweek, so it is applied to that round and to
        # unscheduled on-demand simulations, not to fixtures weeks away.
        gw = fx.get("gameweek", np.nan)
        in_scope = apply_availability and (pd.isna(gw) or s.next_gameweek is None or int(gw) == s.next_gameweek)
        mult_h = mult_a = 1.0
        rep_h = rep_a = None
        if in_scope and s.availability.enabled:
            mult_h, mult_a, rep_h, rep_a = s.availability.multipliers(home, away)
        adj_h, adj_a = lam_h * mult_h, lam_a * mult_a

        seed = int(hashlib.sha1(fid.encode()).hexdigest()[:8], 16)
        sim = simulate(adj_h, adj_a, rho, n_sims=n_sims, seed=seed)

        # Availability shifts the classifier's 1X2 by the same relative amount it shifts the
        # Poisson 1X2 (identity when nobody important is missing).
        p_final = np.asarray(p_model, float)
        if mult_h != 1.0 or mult_a != 1.0:
            base = analytic_markets([lam_h], [lam_a], rho).iloc[0][["p_home", "p_draw", "p_away"]].to_numpy(float)
            adj = analytic_markets([adj_h], [adj_a], rho).iloc[0][["p_home", "p_draw", "p_away"]].to_numpy(float)
            p_final = p_final * adj / base
            p_final = p_final / p_final.sum()

        probs = {"home": float(p_final[0]), "draw": float(p_final[1]), "away": float(p_final[2])}
        best = max(probs, key=probs.get)
        pick_label = {"home": f"{home} win", "draw": "Draw", "away": f"{away} win"}[best]

        explanation = model.explainer.explain(x, p_model)
        delta = (p_final - np.asarray(p_model, float)) * 100
        if np.abs(delta).max() >= 0.05:
            explanation["drivers"].insert(0, {"label": "Squad availability (FPL news)", "home": round(float(delta[0]), 2),
                                              "draw": round(float(delta[1]), 2), "away": round(float(delta[2]), 2),
                                              "features": []})
        explanation["final"] = {k: round(v, 4) for k, v in probs.items()}

        sim_d = sim.to_dict()
        markets = bets.market_odds_from_row(fx)
        value = bets.evaluate_fixture(
            {**probs, "over25": sim_d["over_under"]["2.5"]["over"], "under25": sim_d["over_under"]["2.5"]["under"]},
            markets)
        value["odds_source"] = (odds_source or "Bet365 via football-data.co.uk") if markets else None
        value["odds"] = markets

        kickoff = fx.get("kickoff_utc")
        return _clean({
            "id": fid,
            "home": home, "away": away, "home_short": short_name(home), "away_short": short_name(away),
            "kickoff_utc": None if kickoff is None or pd.isna(kickoff) else pd.Timestamp(kickoff).isoformat(),
            "date": str(pd.Timestamp(fx["date"]).date()),
            "gameweek": None if pd.isna(gw) else int(gw),
            "probs": probs,
            "probs_model": {"home": float(p_model[0]), "draw": float(p_model[1]), "away": float(p_model[2])},
            "pick": {"outcome": best, "label": pick_label, "prob": probs[best], "confidence": _confidence(probs[best])},
            "simulation": sim_d,
            "lambdas_before_availability": {"home": float(lam_h), "away": float(lam_a)},
            "value": value,
            "explain": explanation,
            "styles": {"home": row["h_style"], "away": row["a_style"]},
            "radar": self._radar(s, row),
            "availability": {"applied": bool(in_scope and s.availability.enabled),
                             "home": rep_h, "away": rep_a,
                             "multipliers": {"home": round(mult_h, 4), "away": round(mult_a, 4)}},
            "features": self._key_features(row),
        })

    def _radar(self, s: State, row: pd.Series) -> dict:
        out = {"axes": [label for label, _, _ in RADAR_AXES], "home": [], "away": [], "raw": {"home": [], "away": []}}
        for label, suffix, higher in RADAR_AXES:
            ref = s.radar_reference[suffix]
            for side, key in (("h", "home"), ("a", "away")):
                value = float(row[f"{side}_{suffix}"])
                pct = np.searchsorted(ref, value) / max(len(ref), 1) * 100
                out[key].append(round(pct if higher else 100 - pct, 1))
                out["raw"][key].append(round(value, 3))
        return out

    @staticmethod
    def _key_features(row: pd.Series) -> list[dict]:
        items = [{"key": suffix, "label": label, "home": float(row[f"h_{suffix}"]), "away": float(row[f"a_{suffix}"])}
                 for suffix, label, _ in FEATURES.explain]
        items.append({"key": "home_edge", "label": "Home/away xGD gap (last 10+10)",
                      "home": float(row["h_home_edge"]), "away": float(row["a_home_edge"])})
        return items

    @staticmethod
    def _summary(pred: dict) -> dict:
        sim = pred["simulation"]
        return {
            "id": pred["id"], "home": pred["home"], "away": pred["away"],
            "home_short": pred["home_short"], "away_short": pred["away_short"],
            "kickoff_utc": pred["kickoff_utc"], "date": pred["date"], "gameweek": pred["gameweek"],
            "probs": pred["probs"], "pick": pred["pick"], "styles": pred["styles"],
            "most_likely": sim["most_likely"], "lambda_home": sim["lambda_home"], "lambda_away": sim["lambda_away"],
            "over25": sim["over_under"]["2.5"]["over"], "btts": sim["btts"]["yes"],
            "value": pred["value"]["value"], "has_odds": bool(pred["value"]["odds"]),
            "availability": {side: (pred["availability"][side] or {}).get("attack_penalty", 0.0)
                             for side in ("home", "away")},
            "explain_summary": pred["explain"]["summary"],
        }

    # --------------------------------------------------------- availability
    def set_availability_override(self, team: str, out: list[str], back: list[str]) -> dict:
        team = self.resolve(team)
        overrides = av.load_overrides()
        overrides[team] = {"out": out, "in": back}
        if not out and not back:
            overrides.pop(team, None)
        av.save_overrides(overrides)
        s = self.state
        s.availability.overrides = overrides
        s.cache.clear()
        return s.availability.team_report(team)

    def availability_reports(self) -> list[dict]:
        s = self.state
        return [s.availability.team_report(t) for t in s.current_teams]

