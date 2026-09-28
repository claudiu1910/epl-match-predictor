#!/usr/bin/env python3
"""EPL match predictor CLI: sync data -> retrain if new results -> predict.

The web dashboard (``uvicorn app:app --reload``) and this CLI share the same
prediction service (src/pipeline.py).

Examples
--------
  python run_live.py                         # sync + predict the next gameweek
  python run_live.py --sync                  # refresh results, xG, fixtures, odds, availability
  python run_live.py --predict-next          # predict the next gameweek from local data
  python run_live.py --predict-next --days 14
  python run_live.py --match "Liverpool" "Manchester City"
  python run_live.py --match Spurs "Man Utd" --odds 2.40 3.50 2.90
  python run_live.py --predict-next --notify # also post to Discord/Telegram (if configured)
  python run_live.py --watch 60              # re-sync and re-predict every 60 minutes
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timezone

import pandas as pd
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import config, notify
from src import data_loader as dl
from src.align_teams import UnknownTeamError, normalize_team
from src.features import FEATURES
from src.pipeline import PredictorService
from src.utils import DataSourceError, setup_logging

log = logging.getLogger("run_live")
console = Console()

STALE_AFTER_DAYS = 3
WIDE_WIDTH = 110    # full column names
EXTRA_WIDTH = 150   # also show likely score and value columns
NARROW_WIDTH = 78   # below this, stacked fixture cells and abbreviated labels
MIN_WATCH_MINUTES = 15
SHORT_LABELS = {suffix: short for suffix, _, short in FEATURES.explain}


# ------------------------------------------------------------------------ CLI
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="English Premier League match outcome predictor (calibrated XGBoost + Poisson simulation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[1].replace("--------\n", ""),
    )
    parser.add_argument("--sync", action="store_true",
                        help="download the latest results, xG, fixtures, odds and player availability")
    parser.add_argument("--predict-next", action="store_true", help="predict the next gameweek")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"), help="one-off prediction for HOME v AWAY")
    parser.add_argument("--date", type=date.fromisoformat,
                        help="as-of date for --match (YYYY-MM-DD); default: scheduled kickoff, else today")
    parser.add_argument("--odds", nargs=3, type=float, metavar=("HOME", "DRAW", "AWAY"),
                        help="decimal odds for --match, to run the EV engine")
    parser.add_argument("--no-availability", action="store_true", help="ignore FPL injury/suspension news")
    parser.add_argument("--days", type=int, metavar="N",
                        help="with --predict-next: every fixture in the next N days instead of one round")
    parser.add_argument("--seasons", type=int, default=config.DEFAULT_HISTORY_SEASONS,
                        choices=range(config.MIN_HISTORY_SEASONS, config.MAX_HISTORY_SEASONS + 1),
                        help="completed seasons to train on (default: %(default)s)")
    parser.add_argument("--retrain", action="store_true", help="retrain even if no new results arrived")
    parser.add_argument("--refresh-history", action="store_true", help="re-download completed seasons too")
    parser.add_argument("--no-plots", action="store_true", help="skip calibration / confusion-matrix PNGs")
    parser.add_argument("--notify", action="store_true", help="post gameweek predictions to Discord/Telegram")
    parser.add_argument("--watch", type=int, metavar="MINUTES",
                        help=f"loop: sync + predict every MINUTES (min {MIN_WATCH_MINUTES})")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging (HTTP requests etc.)")
    args = parser.parse_args(argv)
    if args.days is not None and args.days < 1:
        parser.error("--days must be >= 1")
    if args.odds and not args.match:
        parser.error("--odds needs --match")
    if args.odds and any(o <= 1.0 for o in args.odds):
        parser.error("--odds must be decimal odds greater than 1.0")
    if args.watch is not None and args.watch < MIN_WATCH_MINUTES:
        parser.error(f"--watch must be at least {MIN_WATCH_MINUTES} minutes (be kind to the data sources)")
    if not (args.sync or args.predict_next or args.match):
        args.sync = args.predict_next = True
    if args.watch:
        args.sync = True
        args.predict_next = args.predict_next or not args.match
    return args


# ----------------------------------------------------------------------- sync
def run_sync(args) -> None:
    with console.status("Syncing football-data.co.uk, Understat and FPL..."):
        report = dl.sync_all(args.seasons, refresh_history=args.refresh_history)
    table = Table(title="Data sync", box=box.SIMPLE_HEAVY, title_justify="left")
    table.add_column("Source")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    colours = {"downloaded": "green", "updated": "green", "unchanged": "dim", "cached": "dim", "failed": "red"}
    for r in report.results:
        if r.status == "cached" and not args.verbose:
            continue
        table.add_row(r.name, f"[{colours.get(r.status, 'white')}]{r.status}[/]", r.detail)
    console.print(table)
    if report.failures:
        console.print(f"[yellow]{len(report.failures)} source(s) failed; continuing with cached data where "
                      f"available. See logs/pipeline.log.[/]")
    if not dl.has_minimum_data(args.seasons):
        raise DataSourceError("historical data is missing and could not be downloaded")


def ensure_data(args) -> None:
    if not dl.has_minimum_data(args.seasons):
        log.info("No local data found; running an initial sync")
        run_sync(args)
        return
    last = dl.last_sync_time()
    if last and datetime.now(timezone.utc) - last > pd.Timedelta(days=STALE_AFTER_DAYS):
        console.print(f"[yellow]Local data was last synced {last:%Y-%m-%d %H:%M} UTC. "
                      f"Run with --sync for the latest results.[/]")


# ----------------------------------------------------------------- display
def _fmt(value, kind: str, compact: bool = False) -> str:
    if value is None:
        return "—"
    if kind == "accuracy":
        return f"{value:.0%}" if compact else f"{value:.1%}"
    return f"{value:.3f}" if compact else f"{value:.4f}"


def show_validation(service: PredictorService, retrained: bool) -> None:
    bundle = service.state.bundle
    v = bundle.validation
    narrow = console.width < NARROW_WIDTH
    labels = {
        "base_rate": "Base rate (H/D/A mix)", "bet365": "Bet365 (margin removed)", "bookmakers": "Market avg closing",
        "random_forest": "RandomForest baseline", "xgboost": "XGBoost, uncalibrated", "poisson": "Poisson simulation",
        "xgboost_calibrated": "XGBoost+calib. (live)", "classifier": "XGBoost+calib. (live)",
        "xgboost_calibrated_on_odds_subset": "  live, odds rows only",
    }
    if narrow:
        labels.update({"base_rate": "Base rate", "bet365": "Bet365", "bookmakers": "Market avg",
                       "random_forest": "RandomForest", "xgboost": "XGBoost raw", "poisson": "Poisson",
                       "xgboost_calibrated": "XGB+calib (live)", "classifier": "XGB+calib (live)",
                       "xgboost_calibrated_on_odds_subset": " live, odds rows"})
    metrics = (("log_loss", "LogLoss"), ("brier", "Brier"), ("accuracy", "Acc")) if narrow else (
        ("log_loss", "Log loss ↓"), ("brier", "Brier ↓"), ("rps", "RPS ↓"), ("accuracy", "Accuracy ↑"))
    live_keys = ("xgboost_calibrated", "classifier")

    def metrics_table(block: dict, order: list[str]) -> Table:
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, padding=(0, 1))
        table.add_column("Model", no_wrap=True)
        if not narrow:
            table.add_column("n", justify="right")
        for _, label in metrics:
            table.add_column(label, justify="right", no_wrap=True)
        for key in order:
            m = block.get(key)
            if not m:
                continue
            style = "bold green" if key in live_keys else "dim" if key in (
                "base_rate", "xgboost_calibrated_on_odds_subset") else ""
            cells = [_fmt(m.get(k), k, compact=narrow) for k, _ in metrics]
            table.add_row(labels[key], *([] if narrow else [str(m["n"])]), *cells, style=style)
        return table

    parts = [
        Text.from_markup(f"[bold]Hold-out season {v['holdout_season']}[/] [dim](trained on "
                         f"{v['holdout_train_seasons'][0]} → {v['holdout_train_seasons'][-1]} only)[/]"),
        metrics_table(v["holdout"], ["base_rate", "bet365", "random_forest", "xgboost", "poisson",
                                     "xgboost_calibrated"]),
    ]
    bt = v.get("backtest")
    if bt:
        parts += [Text.from_markup(f"[bold]Walk-forward {bt['seasons'][0]} → {bt['seasons'][-1]}[/] "
                                   f"[dim]({bt['matches']} matches)[/]"),
                  metrics_table(bt, ["base_rate", "bet365", "classifier"])]
        vb = bt["value_bets"]["1x2"]
        if vb.get("bets"):
            parts.append(Text.from_markup(
                f"[dim]Value flags (EV > {bt['value_bets']['threshold']:.0%}) vs Bet365: {vb['bets']} bets, ROI "
                f"{vb['roi']:+.1%} ± {vb['roi_se']:.1%}. No demonstrated edge.[/]"))
    if "current_season" in v:
        parts += [Text.from_markup(f"[bold]Current season {v['current_season_label']} so far[/] "
                                   f"[dim](trained on prior seasons only)[/]"),
                  metrics_table(v["current_season"], ["base_rate", "bet365", "xgboost_calibrated"])]
    status = "retrained with new results" if retrained else "no new results since last run, cached model reused"
    parts.append(Text.from_markup(
        f"[dim]Live model: XGBoost + {bundle.config['calibration']} calibration (chronological) on {bundle.n_train} "
        f"matches ({bundle.train_seasons[0]} → {bundle.trained_through}); {status}."
        + (f" Plots: {', '.join(v['plots'])}" if v.get("plots") else "") + "[/]"))
    console.print(Panel(Group(*parts), title="Model validation (out-of-time)", title_align="left",
                        border_style="blue", expand=False))


def _local_kickoff(iso, compact: bool = False) -> str:
    if not iso:
        return "TBC"
    local = pd.Timestamp(iso).tz_convert(datetime.now().astimezone().tzinfo)
    return local.strftime("%a %d %H:%M" if compact else "%a %d %b %H:%M")


def _pct_cells(probs: dict, digits: int = 1) -> list[str]:
    values = [probs["home"], probs["draw"], probs["away"]]
    top = max(values)
    return [f"[bold]{v:.{digits}%}[/]" if v == top else f"{v:.{digits}%}" for v in values]


def _pick_label(f: dict, short: bool) -> str:
    outcome = f["pick"]["outcome"]
    if outcome == "draw":
        return "Draw"
    name = f["home_short"] if outcome == "home" else f["away_short"]
    return name if short else f"{name} win"


CONF_COLOUR = {"high": "green", "medium": "yellow", "low": "dim"}


def predict_next(service: PredictorService, args) -> dict:
    gw = service.gameweek(args.days)
    fixtures = gw["fixtures"]
    if not fixtures:
        console.print("[yellow]No upcoming fixtures found.[/]")
        return gw
    save_predictions(fixtures)
    tz_name = datetime.now().astimezone().tzname()
    first, last = fixtures[0]["date"], fixtures[-1]["date"]
    span = f"{pd.Timestamp(first):%a %d %b} – {pd.Timestamp(last):%a %d %b}"
    title = f"Premier League {gw['season']} · {gw['label']} · {span}"
    wide = console.width >= WIDE_WIDTH
    extra = console.width >= EXTRA_WIDTH
    narrow = console.width < NARROW_WIDTH
    if narrow:
        table = Table(title=title, box=box.SIMPLE_HEAD, header_style="bold", title_style="bold",
                      title_justify="left", padding=(0, 0, 0, 1), pad_edge=False, show_lines=True)
        table.add_column(f"Match ({tz_name})", no_wrap=True)
        for label in ("H%", "D%", "A%"):
            table.add_column(label, justify="right", no_wrap=True)
        table.add_column("Pick", no_wrap=True)
        for f in fixtures:
            conf = f["pick"]["confidence"]
            ev = " [green]+EV[/]" if f["value"] else ""
            match = (f"{f['home_short']}\nv {f['away_short']}\n"
                     f"[dim]{_local_kickoff(f['kickoff_utc'], compact=True)} · {f['most_likely']['score']}[/]")
            table.add_row(match, *_pct_cells(f["probs"], digits=0),
                          f"[bold]{_pick_label(f, True)}[/]\n[{CONF_COLOUR[conf]}]{conf[:3]}[/]{ev}")
    else:
        table = Table(title=title, box=box.SIMPLE_HEAVY, header_style="bold", title_style="bold",
                      title_justify="left", padding=(0, 1) if wide else (0, 0), pad_edge=False)
        table.add_column(f"Kickoff ({tz_name})" if wide else f"Kickoff\n({tz_name})", no_wrap=True)
        table.add_column("Home", no_wrap=True)
        table.add_column("Away", no_wrap=True)
        table.add_column("Home Win %" if wide else "Home\nWin %", justify="right", no_wrap=True)
        table.add_column("Draw %" if wide else "Draw\n%", justify="right", no_wrap=True)
        table.add_column("Away Win %" if wide else "Away\nWin %", justify="right", no_wrap=True)
        table.add_column("Recommended" if wide else "Pick", no_wrap=True)
        table.add_column("Confidence" if wide else "Conf", no_wrap=True)
        if extra:
            table.add_column("Likely score", no_wrap=True)
            table.add_column("Value", no_wrap=True)
        for f in fixtures:
            conf = f["pick"]["confidence"]
            row = [_local_kickoff(f["kickoff_utc"], compact=not wide), f["home_short"], f["away_short"],
                   *_pct_cells(f["probs"]), f"[bold]{_pick_label(f, not wide)}[/]",
                   f"[{CONF_COLOUR[conf]}]{conf if wide else conf[:3]}[/]"]
            if extra:
                best = max(f["value"], key=lambda x: x["ev"]) if f["value"] else None
                row += [f"{f['most_likely']['score']} ({f['most_likely']['prob']:.0%})",
                        f"[green]+EV {best['label']} {best['ev']:+.0%}[/]" if best else
                        ("[dim]—[/]" if f["has_odds"] else "[dim]no odds[/]")]
            table.add_row(*row)
    console.print(table)

    notes = []
    warn = [f"{f[side + '_short']} attack −{f['availability'][side]:.0%}" for f in fixtures
            for side in ("home", "away") if (f["availability"] or {}).get(side, 0) >= 0.05]
    if warn:
        notes.append("Missing key players (FPL news): " + ", ".join(warn) + ".")
    if gw["pending"]:
        notes.append(f"{len(gw['pending'])} earlier fixture(s) have no result yet (in play, awaiting sync or postponed).")
    if narrow:
        notes.append("H% / D% / A% = Home Win %, Draw %, Away Win %.")
    notes.append("Confidence: high ≥ 60%, medium ≥ 45%, low otherwise. Features use only matches completed "
                 "before each kickoff. Model output, not betting advice.")
    console.print(Text("\n".join(notes), style="dim"))
    return gw


def save_predictions(fixtures: list[dict]) -> None:
    rows = [{"kickoff_utc": f["kickoff_utc"], "home": f["home"], "away": f["away"],
             "p_home": f["probs"]["home"], "p_draw": f["probs"]["draw"], "p_away": f["probs"]["away"],
             "pick": f["pick"]["label"], "most_likely": f["most_likely"]["score"],
             "lambda_home": f["lambda_home"], "lambda_away": f["lambda_away"], "p_over25": f["over25"],
             "p_btts": f["btts"], "value_bets": "; ".join(f"{v['label']} @{v['odds']} EV {v['ev']:+.2f}"
                                                         for v in f["value"])} for f in fixtures]
    out = pd.DataFrame(rows)
    folder = config.PROCESSED_DIR / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.to_csv(folder / f"predictions_{stamp}.csv", index=False)
    out.to_csv(config.PROCESSED_DIR / "predictions_latest.csv", index=False)


def predict_match(service: PredictorService, args) -> None:
    odds = dict(zip(("home", "draw", "away"), args.odds)) if args.odds else None
    d = service.predict(args.match[0], args.match[1], on=args.date, odds=odds,
                        apply_availability=not args.no_availability)
    home, away = d["home"], d["away"]
    if home not in service.state.current_teams or away not in service.state.current_teams:
        missing = [t for t in (home, away) if t not in service.state.current_teams]
        console.print(f"[yellow]{', '.join(missing)} not in the {service.status()['season']} Premier League; "
                      f"using the most recent PL form.[/]")
    narrow = console.width < NARROW_WIDTH
    home_label, away_label = (d["home_short"], d["away_short"]) if narrow else (home, away)
    probs, sim = d["probs"], d["simulation"]
    when = _local_kickoff(d["kickoff_utc"]) if d["kickoff_utc"] else f"{pd.Timestamp(d['date']):%a %d %b %Y}"

    bar_len = max(10, min(40, console.width - 30))
    bars = Table.grid(padding=(0, 2))
    bars.add_column(no_wrap=True)
    bars.add_column(justify="right")
    bars.add_column()
    for label, p, colour in ((f"{home_label} win", probs["home"], "cyan"), ("Draw", probs["draw"], "white"),
                             (f"{away_label} win", probs["away"], "yellow")):
        bars.add_row(label, f"{p:.1%}", f"[{colour}]{'█' * round(p * bar_len)}[/]")

    ml, bbo = sim["most_likely"], sim["best_by_outcome"]
    lines = [
        f"Recommended: [bold]{_pick_label(d, narrow)}[/] ({d['pick']['confidence']} confidence)",
        f"Styles: {d['styles']['home']} vs {d['styles']['away']}",
        f"Expected goals {sim['lambda_home']:.2f} – {sim['lambda_away']:.2f} · most likely {ml['score']} "
        f"({ml['prob']:.1%}) · home-win score {bbo['home']['score'] if bbo['home'] else '—'}",
        f"Over 2.5 {sim['over_under']['2.5']['over']:.1%} · BTTS {sim['btts']['yes']:.1%} · "
        f"{sim['n_sims']:,} simulations",
    ]
    e = d["explain"]
    if e["summary"]:
        lines.append(f"Why ({e['favoured']}): " + "; ".join(e["summary"][:3])
                     + f"; calibration {e['calibration_adjustment'][e['favoured']]:+.1f} pts")
    avail = d["availability"]
    if avail["applied"]:
        for side in ("home", "away"):
            rep = avail[side] or {}
            out = [m["web_name"] for m in rep.get("missing", []) if m["loss"] >= 0.5][:4]
            if rep.get("attack_penalty", 0) >= 0.01 or out:
                lines.append(f"{d[side + '_short']}: attack −{rep.get('attack_penalty', 0):.0%}"
                             + (f" (out: {', '.join(out)})" if out else ""))
    summary = Text.from_markup("\n".join(lines))

    parts = [bars, Text(""), summary]
    if d["value"]["selections"]:
        vt = Table(box=box.SIMPLE, header_style="bold", title="Value check (EV = p × odds − 1)", title_style="dim",
                   title_justify="left")
        for col in ("Selection", "Odds", "Model", "Fair", "EV"):
            vt.add_column(col, justify="left" if col == "Selection" else "right")
        for s in d["value"]["selections"]:
            vt.add_row(s["label"], f"{s['odds']:.2f}", f"{s['model_prob']:.1%}", f"{s['fair_prob']:.1%}",
                       f"[{'green' if s['is_value'] else 'dim'}]{s['ev']:+.1%}[/]")
        parts += [Text(""), vt]

    detail = Table(box=box.SIMPLE, header_style="bold", title_style="dim", title_justify="left",
                   title="Pre-match features" if narrow else "Pre-match features (only matches before kickoff)",
                   pad_edge=not narrow)
    detail.add_column("Metric", no_wrap=True)
    detail.add_column(home_label, justify="right", no_wrap=True)
    detail.add_column(away_label, justify="right", no_wrap=True)
    for f in d["features"]:
        fmt = "{:.0f}" if f["key"] in ("elo", "rest", "spell_games") else "{:.2f}"
        label = SHORT_LABELS.get(f["key"], "H/A xGD gap") if narrow else f["label"]
        detail.add_row(label, fmt.format(f["home"]), fmt.format(f["away"]))
    parts += [Text(""), detail]
    console.print(Panel(Group(*parts), title=f"{home_label} v {away_label} · {when}", title_align="left",
                        border_style="green"))


# ----------------------------------------------------------------------- main
def run_once(args) -> None:
    if args.match:
        for name in args.match:  # fail fast on typos, before any download or training
            normalize_team(name, strict=True)
    if args.sync:
        run_sync(args)
    else:
        ensure_data(args)
    if args.predict_next or args.match:
        service = PredictorService(n_history=args.seasons, make_plots=not args.no_plots)
        with console.status("Building features and loading the model..."):
            state = service.refresh(retrain=args.retrain)
        show_validation(service, state.retrained)
        if args.predict_next:
            gw = predict_next(service, args)
            if args.notify:
                if notify.configured():
                    results = notify.send(notify.format_gameweek(gw))
                    console.print(f"[dim]Notifications: {results}[/]")
                else:
                    console.print("[yellow]--notify: set DISCORD_WEBHOOK_URL or TELEGRAM_BOT_TOKEN + "
                                  "TELEGRAM_CHAT_ID first.[/]")
        if args.match:
            predict_match(service, args)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    try:
        if not args.watch:
            run_once(args)
            return 0
        while True:
            console.rule(f"[bold]{datetime.now():%Y-%m-%d %H:%M}")
            try:
                run_once(args)
            except DataSourceError as exc:
                log.error("Cycle failed: %s (retrying next cycle)", exc)
            console.print(f"[dim]Next update in {args.watch} min (Ctrl-C to stop)[/]")
            time.sleep(args.watch * 60)
    except UnknownTeamError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    except (DataSourceError, ValueError) as exc:
        log.debug("fatal error", exc_info=True)
        console.print(f"[red]Error:[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
