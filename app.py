"""FastAPI web app: dashboard + JSON API for the EPL match predictor.

Run:    uvicorn app:app --reload
Docs:   http://127.0.0.1:8000/docs

Environment (all optional):
    HISTORY_SEASONS=5            completed seasons to train on (3-5)
    ENABLE_SCHEDULER=1           automated sync (0 disables)
    SYNC_CRON="0 7 * * tue"      crontab for the automated sync, Europe/London time
    ADMIN_TOKEN=...              if set, POST /api/sync and /api/availability need X-Admin-Token
    DISCORD_WEBHOOK_URL / TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID   post predictions after syncs
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config, notify
from src.align_teams import UnknownTeamError
from src.pipeline import NotReady, PredictorService
from src.utils import DataSourceError, setup_logging

setup_logging(verbose=os.getenv("LOG_LEVEL", "").upper() == "DEBUG")
log = logging.getLogger("app")

TIMEZONE = config.UK_TZ_NAME
SYNC_CRON = os.getenv("SYNC_CRON", "0 7 * * tue")  # Tuesday 07:00 UK: after the Monday-night game
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

service = PredictorService(n_history=int(os.getenv("HISTORY_SEASONS", config.DEFAULT_HISTORY_SEASONS)))
scheduler = BackgroundScheduler(timezone=TIMEZONE)


# ------------------------------------------------------------------ sync job
class SyncJob:
    """One sync at a time, run in a background thread; status is polled by the UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self.status: dict = {"state": "idle"}

    def start(self, trigger: str, retrain: bool = False, send_notification: bool = False) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        self.status = {"state": "running", "trigger": trigger, "retrain": retrain,
                       "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        threading.Thread(target=self._run, args=(trigger, retrain, send_notification), daemon=True).start()
        return True

    def _run(self, trigger: str, retrain: bool, send_notification: bool) -> None:
        try:
            result = service.sync(retrain=retrain)
            if send_notification and notify.configured():
                result["notifications"] = notify.send(notify.format_gameweek(service.gameweek(), result))
            self.status.update(state="succeeded", result=result)
            log.info("%s sync finished: %s new result(s), retrained=%s", trigger, result["new_results"],
                     result["retrained"])
        except Exception as exc:  # report, never crash the server
            log.exception("%s sync failed", trigger)
            self.status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            self.status["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._lock.release()


sync_job = SyncJob()


def _initial_load() -> None:
    try:
        service.refresh()
    except Exception:
        log.exception("Initial load failed; use the Sync button or POST /api/sync to retry")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_initial_load, name="initial-load", daemon=True).start()
    if os.getenv("ENABLE_SCHEDULER", "1") != "0":
        scheduler.add_job(lambda: sync_job.start("scheduled", send_notification=True),
                          CronTrigger.from_crontab(SYNC_CRON, timezone=TIMEZONE),
                          id="weekly-sync", replace_existing=True, misfire_grace_time=3600, coalesce=True)
        scheduler.start()
        log.info("Automated sync scheduled: '%s' (%s)", SYNC_CRON, TIMEZONE)
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="EPL Match Predictor",
    version="2.0",
    description="Calibrated 1X2 probabilities, Monte Carlo scorelines, value bets and SHAP explanations "
                "for upcoming Premier League fixtures. Model output, not betting advice.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(config.REPORTS_DIR)), name="reports")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


# ------------------------------------------------------------------ errors
@app.exception_handler(NotReady)
async def _not_ready(_: Request, exc: NotReady):
    return JSONResponse(status_code=503, content={"detail": f"Model not ready: {exc}", "loading": service.loading})


@app.exception_handler(UnknownTeamError)
async def _unknown_team(_: Request, exc: UnknownTeamError):
    return JSONResponse(status_code=404, content={"detail": str(exc), "suggestions": exc.suggestions})


def _require_admin(token: str | None) -> None:
    if ADMIN_TOKEN and not (token and secrets.compare_digest(token, ADMIN_TOKEN)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-Admin-Token")


# ------------------------------------------------------------------ schemas
class Odds(BaseModel):
    home: float | None = Field(None, gt=1.0, description="Decimal odds, home win")
    draw: float | None = Field(None, gt=1.0)
    away: float | None = Field(None, gt=1.0)
    over25: float | None = Field(None, gt=1.0, description="Decimal odds, over 2.5 goals")
    under25: float | None = Field(None, gt=1.0)


class SimulateRequest(BaseModel):
    home: str = Field(..., examples=["Liverpool"])
    away: str = Field(..., examples=["Manchester City"])
    date: dt.date | None = Field(None, description="Evaluate as of this date (default: scheduled kickoff or today)")
    odds: Odds | None = Field(None, description="Optional prices for the EV engine")
    apply_availability: bool = True
    n_sims: int = Field(10_000, ge=1_000, le=100_000)

    @field_validator("home", "away")
    @classmethod
    def _strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("team name required")
        return v.strip()


class SyncRequest(BaseModel):
    retrain: bool = False
    notify: bool = False


class AvailabilityOverride(BaseModel):
    team: str
    out: list[str] = Field(default_factory=list, description="Players confirmed out")
    back_in: list[str] = Field(default_factory=list, description="Players confirmed available")


class Probabilities(BaseModel):
    home: float
    draw: float
    away: float


class FixtureSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    home: str
    away: str
    kickoff_utc: str | None
    probs: Probabilities
    pick: dict
    most_likely: dict
    value: list[dict]


class GameweekResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    season: str
    gameweek: int | None
    label: str
    fixtures: list[FixtureSummary]
    value_bets: int


class MatchPrediction(BaseModel):
    """Full deep-dive payload: probabilities, simulation, EV, SHAP, radar, availability."""
    model_config = ConfigDict(extra="allow")
    id: str
    home: str
    away: str
    probs: Probabilities
    probs_model: Probabilities
    pick: dict
    simulation: dict
    value: dict
    explain: dict
    radar: dict
    styles: dict
    availability: dict
    features: list[dict]


# ------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"version": app.version})


# ------------------------------------------------------------------ API
@app.get("/api/health", tags=["system"])
def health():
    return {"ok": True, "ready": service.ready}


@app.get("/api/status", tags=["system"])
def status():
    job = scheduler.get_job("weekly-sync") if scheduler.running else None
    return {
        **service.status(),
        "scheduler": {"enabled": job is not None, "cron": SYNC_CRON, "timezone": TIMEZONE,
                      "next_run": job.next_run_time.isoformat() if job and job.next_run_time else None},
        "sync_job": sync_job.status,
        "notifications": notify.configured(),
        "admin_token_required": bool(ADMIN_TOKEN),
    }


@app.get("/api/teams", tags=["predictions"])
def teams(all: bool = Query(False, description="Include every club in the training data")):
    return service.teams(include_all=all)


@app.get("/api/fixtures", response_model=GameweekResponse, tags=["predictions"])
def fixtures(days: int | None = Query(None, ge=1, le=60, description="All fixtures in the next N days "
                                                                      "instead of the next gameweek")):
    return service.gameweek(days)


@app.get("/api/fixtures/{fixture_id}", response_model=MatchPrediction, tags=["predictions"])
def fixture(fixture_id: str):
    try:
        return service.fixture(fixture_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No upcoming fixture '{fixture_id}'")


@app.post("/api/simulate", response_model=MatchPrediction, tags=["predictions"])
def simulate(req: SimulateRequest):
    try:
        return service.predict(req.home, req.away, on=req.date,
                               odds=req.odds.model_dump(exclude_none=True) if req.odds else None,
                               apply_availability=req.apply_availability, n_sims=req.n_sims)
    except ValueError as exc:
        if isinstance(exc, UnknownTeamError):
            raise
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/metrics", tags=["model"])
def metrics():
    return service.state.bundle.validation


@app.get("/api/availability", tags=["model"])
def availability():
    return service.availability_reports()


@app.post("/api/availability", tags=["model"])
def set_availability(req: AvailabilityOverride, x_admin_token: str | None = Header(None)):
    _require_admin(x_admin_token)
    return service.set_availability_override(req.team, req.out, req.back_in)


@app.post("/api/sync", status_code=202, tags=["system"])
def sync(req: SyncRequest | None = None, x_admin_token: str | None = Header(None)):
    _require_admin(x_admin_token)
    req = req or SyncRequest()
    if not sync_job.start("manual", retrain=req.retrain, send_notification=req.notify):
        raise HTTPException(status_code=409, detail="A sync is already running")
    return sync_job.status


@app.get("/api/sync/status", tags=["system"])
def sync_status() -> dict:
    return sync_job.status


@app.exception_handler(DataSourceError)
async def _data_error(_: Request, exc: DataSourceError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})
