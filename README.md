# EPL Match Predictor

This project predicts upcoming Premier League matches and serves the results through a
FastAPI + Tailwind dashboard and a terminal CLI. It trains on the last 3–5 completed
seasons plus every completed match of the current season. It re-syncs results weekly (or
when you press **Sync data**), so the rolling form, xG and PPDA features stay current.

- **1X2 probabilities:** XGBoost with chronological calibration inside `CalibratedClassifierCV`.
- **Scorelines:** Poisson goal regressors plus a Dixon–Coles bivariate Poisson model, sampled 10,000 times. This gives exact scores, Over/Under and Both Teams to Score (BTTS).
- **Tactics:** clubs are clustered into three styles (High-Press Possession, Counter / Low Block, Direct Transition), and the matchup between styles is a model feature.
- **Squad availability:** FPL injury and suspension news reduces a side's expected goals when key contributors are missing.
- **Value bets:** EV against Bet365 prices, with the bookmaker margin removed by the Shin, proportional or power method.
- **Explanations:** SHAP drivers for every fixture, in percentage points.

> Model output, not betting advice. The value flags are backtested below. So far they have
> **not** beaten Bet365.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install libomp            # macOS only: XGBoost needs the OpenMP runtime
```

```bash
uvicorn app:app --reload       # dashboard at http://127.0.0.1:8000, API docs at /docs
```

On first start the server downloads about 25 files (roughly 7 MB) into `data/raw/` and
trains the models, which takes 10–20 s. The page shows "Training model…" until it's ready.
After that, restarts reuse the cached model unless new results have arrived.

macOS has no bare `python` outside a virtualenv. Run `source .venv/bin/activate` in each
new terminal, or call `.venv/bin/python` / `.venv/bin/uvicorn` directly.

### CLI

```bash
python run_live.py                                      # sync -> retrain if needed -> predict next gameweek
python run_live.py --sync                               # refresh results, xG, fixtures, odds, availability
python run_live.py --predict-next [--days 14]           # table of the next gameweek (or next N days)
python run_live.py --match "Liverpool" "Manchester City"
python run_live.py --match Spurs "Man Utd" --odds 2.40 3.50 2.90   # + EV check
python run_live.py --predict-next --notify              # also post to Discord / Telegram
python run_live.py --watch 60                           # loop every 60 min
```

Every prediction run prints the model's log loss, Brier score, RPS and accuracy first. The
output adapts to narrow terminals.

## Dashboard

| Area | What it shows |
|---|---|
| **Status strip** | Latest result ingested, training size, last sync, next automated sync |
| **Gameweek hub** | A card per fixture: 1X2 bar, most likely score, expected goals, O2.5 / BTTS, tactical styles, missing-player warnings, +EV badge. Toggle: next gameweek / 14 days / 5 weeks |
| **Deep dive** (click a card) | Probability comparison (final vs model vs Poisson vs market), a radar of xG created, xG conceded, press resistance, pressing, possession and finishing (percentiles vs all PL sides), scoreline heatmap, goals markets, SHAP drivers, value table, squad availability, pre-match numbers |
| **Match simulator** | Any two clubs, an optional as-of date, optional odds for the EV engine, and a toggle for availability |
| **Model validation** | Hold-out season, 3-season walk-forward backtest, value-flag ROI by odds band, goal-model checks, tactical centroids, calibration plot |
| **Sync data** button | `POST /api/sync`: pulls the latest results, rebuilds rolling stats, and retrains only if the data changed |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/status` | Readiness, data freshness, model info, scheduler, sync job |
| GET | `/api/fixtures?days=N` | Next gameweek (or next N days): summary cards |
| GET | `/api/fixtures/{id}` | Full deep dive for one upcoming fixture |
| POST | `/api/simulate` | `{"home","away","date?","odds?":{home,draw,away,over25,under25},"apply_availability","n_sims"}` |
| GET | `/api/teams?all=false` | Clubs with current tactical style and profile |
| GET | `/api/metrics` | Validation report |
| GET/POST | `/api/availability` | Availability reports / override confirmed lineups `{"team","out":[...],"back_in":[...]}` |
| POST | `/api/sync` | Start a background sync (`{"retrain":false,"notify":false}`); 409 if one is already running |
| GET | `/api/sync/status` | idle / running / succeeded / failed, per-source results |

Interactive docs are at `/docs`. Unknown clubs return 404 with suggestions, the same club on
both sides returns 422, and the first load returns 503 until the model is ready.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `HISTORY_SEASONS` | `5` | Completed seasons to train on (3–5) |
| `ENABLE_SCHEDULER` | `1` | Automated sync via APScheduler (`0` disables) |
| `SYNC_CRON` | `0 7 * * tue` | Crontab, Europe/London time. Tuesday 07:00 is after the Monday-night game |
| `ADMIN_TOKEN` | *(unset)* | If set, `POST /api/sync` and `POST /api/availability` require header `X-Admin-Token` |
| `DISCORD_WEBHOOK_URL` | *(unset)* | Post gameweek predictions after scheduled syncs / `--notify` |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | *(unset)* | Same, to Telegram |
| `LOG_LEVEL` | `INFO` | `DEBUG` logs every HTTP request |

No notifications are sent unless you set the webhook variables.

## Layout

```
├── data/
│   ├── raw/                 # football_data/ (E0, E1, fixtures), understat/, fpl/, archive/ (older live snapshots)
│   └── processed/           # matches.csv, training_features.csv, upcoming_features.csv, predictions/
├── src/
│   ├── data_loader.py       # football-data (stats + Bet365 1X2/O/U odds), Understat (xG, npxG, deep, PPDA, players), FPL
│   ├── fixture_fetcher.py   # upcoming fixtures + gameweek numbers (FPL > Understat > fixtures.csv), squad availability feed
│   ├── align_teams.py       # canonical team names across all sources + CLI/API input
│   ├── features.py          # leak-free rolling features (3/5/10/19), PPDA, possession, press resistance, rest, Elo, priors
│   ├── tactics.py           # k-means style clusters + matchup features
│   ├── availability.py      # key-player modifier from FPL news x Understat xG/xA shares
│   ├── simulation.py        # Poisson goal regressors, Dixon-Coles, 10,000-run Monte Carlo
│   ├── bet_evaluator.py     # margin removal (Shin / proportional / power), EV, value flags, backtest
│   ├── explainer.py         # SHAP TreeExplainer -> grouped percentage-point drivers
│   ├── model.py             # chronological calibration, validation, walk-forward backtest, caching
│   ├── pipeline.py          # prediction service shared by app.py and run_live.py
│   ├── evaluate.py          # log loss, Brier, RPS, Poisson deviance, calibration plots
│   ├── notify.py            # Discord / Telegram webhooks
│   ├── config.py, utils.py  # paths, URLs, seasons; logging, HTTP retries, atomic writes
├── static/css/app.css, static/js/app.js
├── templates/index.html
├── tests/                   # 52 tests: leakage, priors, aliases, simulation, EV, tactics, SHAP, availability, API
├── app.py                   # FastAPI app + APScheduler
├── run_live.py              # CLI
└── requirements.txt         # pinned
```

## Data pipeline

| Source | Data | Refresh |
|---|---|---|
| football-data.co.uk `E0.csv` | Results, shots, shots on target, Bet365 1X2 and O/U 2.5 odds, market average closing odds | Current season every sync; history once |
| football-data.co.uk `E1.csv` | Championship profiles for promoted-team priors | Once (last season re-checked) |
| football-data.co.uk `fixtures.csv` | Bet365 prices for the coming week's fixtures | Every sync |
| Understat `getLeagueData/EPL/{year}` | xG, npxG, PPDA pass/action counts, deep completions, player xG/xA, schedule | Current season every sync (+ last season's player table) |
| FPL `bootstrap-static` / `fixtures` | Gameweek numbers, official kickoffs, injury/suspension status | Every sync |

- **Understat parsing:** Understat is read through its JSON endpoint. If that ever fails, a BeautifulSoup parser falls back to the older page layout.
- **Freshness:** Understat usually posts results within hours, while football-data updates about twice a week. Matches found only on Understat are merged in with goals and xG, and their shot counts fill in later.
- **Failures:** every request retries 429/5xx responses with backoff and a timeout. A failed source is reported and cached data is used. Downloads are written atomically, and changed live files are archived.
- **Team names:** `align_teams` maps every spelling ("Man Utd", "Spurs", "Nott'm Forest", FPL's "Man City") to one canonical name, Championship clubs included. All 2,330 matches line up across sources.

### Leak-free features

Each completed match writes a *post-match* rolling state per team. A fixture reads the
latest state strictly before its kickoff date (`merge_asof(..., allow_exact_matches=False)`).
Historical training rows and live fixtures go through the same `FeatureBuilder.build`.
Tests show that a match's features don't change when later matches are removed or when its
own score is altered. A deliberately injected leak fails 5 tests.

Rolling metrics over the last 3 and 5 matches (plus 10 and 19 for style and long-run strength):

- goals and shots on target for and against, shots for and conceded
- `xG_scored`, `xG_conceded`, `npxG_diff`, finishing (`goals − xG`)
- PPDA (pressing intensity), possession share, press resistance, deep completions
- form points, days of rest (capped at 14)
- venue splits (the home side at home, the away side on the road) and a club-specific home edge
- Elo

**Possession proxy:** Understat's PPDA counts are symmetric, so a team's passes are the
opponent's "passes allowed". Possession share = own passes / (own + opponent) in the
pressing zone. Press resistance = the PPDA a team faces.

**Promoted clubs:** rolling windows reset on promotion. Until a window fills, each metric is
blended with a prior:

```
feature = (n/N) · rolling + (1 − n/N) · prior
```

The prior is the average of promoted sides' first 10 PL matches in earlier seasons, scaled
by the club's own Championship season (shrunk Championship-to-PL conversion factors). A
club with no data falls back to league-average priors with a promoted handicap.

## Models

| Component | Details |
|---|---|
| 1X2 classifier | `XGBClassifier` (depth 2, 250 trees, strong regularisation) on 44 home-minus-away and venue features + 9 tactical-matchup features |
| Calibration | `CalibratedClassifierCV(FrozenEstimator(booster_80%), method="temperature")` fitted on the **newest 20%** of the window, then applied to the booster refit on 100%. Chronological, no shuffling |
| Baseline | `RandomForestClassifier` (validation only) |
| Goal model | Two `XGBRegressor(objective="count:poisson")` for λ_home and λ_away + Dixon–Coles ρ (MLE, `scipy.optimize`) |
| Simulation | 10,000 draws from the DC bivariate Poisson: exact scores, 1X2, O/U 0.5–4.5, BTTS, clean sheets |
| Availability | λ multipliers from missing key players, and the classifier's 1X2 shifted by the same ratio |
| Explanations | `shap.TreeExplainer` on the booster; drivers are grouped and converted to percentage points; baseline + drivers + calibration = final probability |

**Why temperature calibration:** backtests compared a chronological temperature fit with
refit (0.984 log loss), the raw booster (0.986), chronological sigmoid (0.993), shuffled
5-fold sigmoid (0.988) and `TimeSeriesSplit` sigmoid (0.999). Three one-vs-rest sigmoids
overfit the roughly 350 most recent matches. A single temperature parameter does not.

## Validation (out-of-time; printed on every run and served at `/api/metrics`)

Three-season walk-forward: 2023-24, 2024-25 and 2025-26, each predicted by models trained
only on the seasons before it (1,140 matches).

| | Log loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| Base rate | 1.074 | 0.650 | 0.232 | 43.2% |
| **XGBoost + calibration (live)** | **0.984** | **0.586** | **0.201** | **53.3%** |
| Bet365, margin removed | 0.966 | 0.574 | 0.196 | 54.2% |

The latest hold-out season (2025-26) was the hardest for every model: the live model
scored 1.037, RandomForest 1.036, Poisson 1X2 1.043, Bet365 1.019 and market closing odds
1.012.

**Goal model:** the Poisson deviance is better than a constant (1.061 vs 1.123 for home
goals), and the 1X2 derived from the simulation is competitive with the classifier.
Scoring levels drift between seasons (3.28, then 2.93, then 2.75 goals per match), so the
Over 2.5 Brier score (0.245) only matches the base rate (0.245). Bet365 scores 0.239.

**Value flags vs Bet365** (flat 1-unit stakes on every EV > 5% selection):

| Market | Bets | ROI ± SE |
|---|---|---|
| 1X2 (classifier) | 1,037 | −5.5% ± 5.1% |
| Over/Under 2.5 (Poisson) | 594 | −1.3% ± 4.7% |

By odds band, 1X2 flags on prices up to 5.0 roughly broke even (+2% to +10%, all within
noise), while long shots above 5.0 lost 38%. The UI marks those "+EV longshot". The honest
reading: the model has no demonstrated edge over the closing market. A badge means the
model and the market disagree, nothing more.

## Tactical archetypes

K-means (k=3) on each club's last-10-match profile, fitted on training rows only:

| Style | PPDA | Possession share | Deep completions / 100 passes |
|---|---|---|---|
| High-Press Possession | 9.3 | 57% | 3.8 |
| Counter / Low Block | 15.0 | 39% | 2.8 |
| Direct Transition | 11.7 | 51% | 2.5 |

The matchup features are one-hot styles for each side, a mirror flag, and pressing ×
opponent build-up interactions. In backtests they were neutral to slightly positive.

## Squad availability

FPL statuses (injured, suspended, doubtful with a % chance) are matched to Understat
players; 99.5% of players with minutes are matched.

- **Attack penalty:** a player's (npxG + xA) per 90 × his share of this season's minutes gives his share of the team's attack. A missing share × (1 − 40% replacement recovery) reduces λ, capped at −25%.
- **Defence penalty:** a missing first-choice goalkeeper raises the opponent's λ by up to 6%; a regular defender by up to 2.5% each (capped at 15%).

Weighting by this season's minutes avoids double counting players who have been out for
weeks, since their absence is already in the rolling form. FPL news covers the next
gameweek only, so the modifier applies to that round and to the simulator. You can confirm
lineups through `POST /api/availability` or `data/availability_overrides.json`.

All constants are documented heuristics, not fitted, because no historical availability
data exists to fit them on.

## Tests

```bash
pytest -q        # 52 tests; API tests use local data and skip when none has been synced
```

## Known limitations

- There's no possession data in the free sources; possession share is a PPDA-derived pass-share proxy.
- Rest days count league matches only (no cup or European fixtures).
- The availability modifier is heuristic, and real lineups are only known about an hour before kickoff.
- Bet365 prices for upcoming fixtures appear in football-data's `fixtures.csv` only a few days before kickoff. Until then the value check says so; you can enter prices in the simulator.
- Respect the data providers' terms. The automated sync runs weekly by default and the CLI's `--watch` enforces at least 15 minutes between syncs.

## Licence

MIT
