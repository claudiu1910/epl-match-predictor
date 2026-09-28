"""Squad availability modifier (tentative lineups).

No historical lineup data exists for training, so availability is not a model feature.
It is applied at inference time to the expected-goal rates, and through them to the 1X2
probabilities:

* **Attack penalty:** each player's attacking output rate ((npxG + xA) per 90, this
  season plus half-weighted last season, from Understat) times his share of this
  season's minutes gives his share of the team's current attack. Missing share x loss x
  (1 - replacement recovery) reduces the team's lambda.
* **Defence penalty:** a missing first-choice goalkeeper or regular defender raises the
  opponent's lambda (scaled by minutes share).

Weighting by *this season's* minutes matters: a player who has been out for weeks is
already absent from the rolling form features, so he barely moves the modifier.

Availability comes from the FPL API: status a (available), d (doubtful, with chance of
playing), i (injured), s (suspended), u (unavailable), n (not eligible). Players who have
left the club (status u with "joined"/"loan" news) are excluded from the squad entirely.
Confirmed lineups can be forced via ``data/availability_overrides.json`` or
``POST /api/availability``:

    {"Arsenal": {"out": ["Bukayo Saka"], "in": ["William Saliba"]}}

All constants below are heuristics, documented rather than fitted: no historical
availability data exists to fit them on.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

REPLACEMENT_RECOVERY = 0.4   # the stand-in recovers 40% of a missing attacker's output
ATTACK_CAP = 0.25            # at most -25% on a team's expected goals
GK_PENALTY = 0.06            # first-choice keeper missing: opponent lambda +6%
DEFENDER_PENALTY = 0.025     # per regular defender missing (scaled by minutes share)
DEFENCE_CAP = 0.15
PREV_SEASON_WEIGHT = 0.5
KEY_PLAYERS = 5
DEPARTED = re.compile(r"joined|loan|left the club|departed|transferred", re.I)


def _key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z ]+", " ", text).split())


def _loss(status: str, chance: float) -> float:
    """Probability-weighted absence: 1 = certainly out, 0 = certainly available."""
    if status in ("i", "s", "u", "n"):
        return 1.0 if not np.isfinite(chance) else 1.0 - chance / 100.0
    if status == "d":
        return 1.0 - (chance / 100.0 if np.isfinite(chance) else 0.5)
    return 0.0


def load_overrides() -> dict:
    try:
        data = json.loads(config.AVAILABILITY_OVERRIDES_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        log.warning("Ignoring malformed %s: %s", config.AVAILABILITY_OVERRIDES_PATH.name, exc)
        return {}


def save_overrides(overrides: dict) -> None:
    config.AVAILABILITY_OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2, sort_keys=True))


class AvailabilityModel:
    def __init__(self, squads: pd.DataFrame, understat_current: pd.DataFrame,
                 understat_previous: pd.DataFrame, overrides: dict | None = None):
        self.overrides = overrides or {}
        self.enabled = not squads.empty and not (understat_current.empty and understat_previous.empty)
        self.table = self._build(squads, understat_current, understat_previous) if self.enabled else pd.DataFrame()
        if not self.enabled:
            log.info("Availability modifier disabled (no FPL squads or Understat player data)")

    # ------------------------------------------------------------- building
    @staticmethod
    def _season_stats(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["key", "team", "out", "time", "position"])
        f = frame.copy()
        f["key"] = f["player_name"].map(_key)
        f["out"] = f["npxG"].fillna(0) + f["xA"].fillna(0)
        return f[["key", "team", "out", "time", "position", "games"]]

    def _match(self, squad_row, candidates: pd.DataFrame) -> str | None:
        """Understat key for an FPL player (same club preferred)."""
        if candidates.empty:
            return None
        full = _key(squad_row["name"])
        web = _key(squad_row["web_name"])
        keys = candidates["key"].tolist()
        if full in keys:
            return full
        tokens = full.split()
        if len(tokens) >= 2 and f"{tokens[0]} {tokens[-1]}" in keys:
            return f"{tokens[0]} {tokens[-1]}"
        same_team = candidates[candidates["team"] == squad_row["team"]]["key"].tolist()
        for pool in (same_team, keys):
            close = difflib.get_close_matches(full, pool, n=1, cutoff=0.78)
            if close:
                return close[0]
        # Long official names ("Bruno Guimaraes Rodriguez Moura") or mononyms ("Alisson"):
        # accept a unique candidate whose name tokens all appear in the FPL name.
        full_tokens = set(tokens) | set(web.split())
        for pool in (same_team, keys):
            subset = [k for k in pool if set(k.split()) <= full_tokens]
            if len(subset) == 1:
                return subset[0]
        surname = web.split()[-1] if web else ""
        hits = [k for k in same_team if surname and k.split()[-1] == surname]
        return hits[0] if len(hits) == 1 else None

    def _build(self, squads, cur, prev) -> pd.DataFrame:
        squads = squads[~((squads["status"] == "u") & squads["news"].str.contains(DEPARTED, na=False))].copy()
        cur_s, prev_s = self._season_stats(cur), self._season_stats(prev)
        candidates = pd.concat([cur_s[["key", "team"]], prev_s[["key", "team"]]]).drop_duplicates("key")
        squads["us_key"] = [self._match(r, candidates) for _, r in squads.iterrows()]

        cur_by = cur_s.groupby("key")[["out", "time"]].sum()
        prev_by = prev_s.groupby("key")[["out", "time"]].sum()
        team_games = cur.groupby("team")["games"].max() if not cur.empty else pd.Series(dtype=float)

        rows = []
        for _, r in squads.iterrows():
            k = r["us_key"]
            c_out, c_time = (cur_by.loc[k, "out"], cur_by.loc[k, "time"]) if k in cur_by.index else (0.0, 0.0)
            p_out, p_time = (prev_by.loc[k, "out"], prev_by.loc[k, "time"]) if k in prev_by.index else (0.0, 0.0)
            minutes = c_time + PREV_SEASON_WEIGHT * p_time
            rate = 90.0 * (c_out + PREV_SEASON_WEIGHT * p_out) / minutes if minutes >= 180 else 0.0
            games = team_games.get(r["team"], 0)
            if games and games > 0:
                involvement = min(1.0, c_time / (games * 90.0))
            else:  # pre-season: fall back to last season's share of 38 matches
                involvement = min(1.0, p_time / (38 * 90.0))
            rows.append({**r.to_dict(), "rate90": rate, "involvement": involvement,
                         "attack": rate * involvement})
        table = pd.DataFrame(rows)
        totals = table.groupby("team")["attack"].transform("sum")
        table["attack_share"] = np.where(totals > 0, table["attack"] / totals, 0.0)
        unmatched = table["us_key"].isna() & (table["minutes"] > 0)
        if unmatched.any():
            log.debug("%d FPL players with minutes not matched to Understat", int(unmatched.sum()))
        return table

    # ------------------------------------------------------------- queries
    def _override_loss(self, team: str, player: pd.Series) -> float | None:
        spec = self.overrides.get(team, {})
        names = {_key(player["name"]), _key(player["web_name"])}
        for loss, bucket in ((1.0, "out"), (0.0, "in")):
            for entry in spec.get(bucket, []):
                k = _key(entry)
                if k in names or any(difflib.SequenceMatcher(None, k, n).ratio() > 0.85 for n in names):
                    return loss
        return None

    def team_report(self, team: str) -> dict:
        empty = {"team": team, "attack_penalty": 0.0, "defence_penalty": 0.0, "missing": [],
                 "key_players": [], "enabled": self.enabled}
        if not self.enabled:
            return empty
        squad = self.table[self.table["team"] == team]
        if squad.empty:
            return empty
        attack_loss = 0.0
        defence = 0.0
        missing = []
        keepers = squad[squad["position"] == "GK"].sort_values("involvement", ascending=False)
        first_gk = keepers.index[0] if not keepers.empty else None
        for idx, p in squad.iterrows():
            override = self._override_loss(team, p)
            loss = override if override is not None else _loss(p["status"], p["chance"])
            if loss <= 0:
                continue
            a = p["attack_share"] * loss * (1 - REPLACEMENT_RECOVERY)
            d = 0.0
            if idx == first_gk:
                d = GK_PENALTY * loss * p["involvement"]
            elif p["position"] == "DEF":
                d = DEFENDER_PENALTY * loss * p["involvement"]
            attack_loss += a
            defence += d
            if a >= 0.002 or d >= 0.002 or p["attack_share"] >= 0.05 or p["involvement"] >= 0.5:
                missing.append({
                    "name": p["name"], "web_name": p["web_name"], "position": p["position"],
                    "status": p["status"], "chance": None if not np.isfinite(p["chance"]) else int(p["chance"]),
                    "news": p["news"], "attack_share": round(float(p["attack_share"]), 3),
                    "involvement": round(float(p["involvement"]), 2), "loss": round(float(loss), 2),
                    "override": override is not None,
                })
        key = squad.sort_values("attack_share", ascending=False).head(KEY_PLAYERS)
        return {
            "team": team,
            "enabled": True,
            "attack_penalty": round(float(min(ATTACK_CAP, attack_loss)), 4),
            "defence_penalty": round(float(min(DEFENCE_CAP, defence)), 4),
            "missing": sorted(missing, key=lambda m: -m["attack_share"] - m["involvement"] / 10),
            "key_players": [{"name": r["web_name"], "position": r["position"],
                             "attack_share": round(float(r["attack_share"]), 3),
                             "rate90": round(float(r["rate90"]), 2), "status": r["status"]}
                            for _, r in key.iterrows()],
        }

    def multipliers(self, home: str, away: str) -> tuple[float, float, dict, dict]:
        """(lambda_home multiplier, lambda_away multiplier, home report, away report)."""
        h, a = self.team_report(home), self.team_report(away)
        mult_h = (1 - h["attack_penalty"]) * (1 + a["defence_penalty"])
        mult_a = (1 - a["attack_penalty"]) * (1 + h["defence_penalty"])
        return mult_h, mult_a, h, a
