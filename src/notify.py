"""Optional Discord / Telegram alerts after a sync.

Configure with environment variables (nothing is sent when they are unset):

    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
    TELEGRAM_BOT_TOKEN=123456:ABC...   TELEGRAM_CHAT_ID=987654321
"""

from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

DISCORD_LIMIT = 1900   # Discord caps messages at 2000 characters
TELEGRAM_LIMIT = 4000  # Telegram caps at 4096


def configured() -> list[str]:
    channels = []
    if os.getenv("DISCORD_WEBHOOK_URL"):
        channels.append("discord")
    if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
        channels.append("telegram")
    return channels


def format_gameweek(gw: dict, sync: dict | None = None) -> str:
    lines = [f"⚽ EPL {gw.get('season', '')} · {gw.get('label', 'Next round')}"]
    if sync and sync.get("new_results"):
        lines.append(f"{sync['new_results']} new result(s) ingested; model "
                     f"{'retrained' if sync.get('retrained') else 'unchanged'}.")
    for f in gw.get("fixtures", []):
        p = f["probs"]
        line = (f"{f['home_short']} v {f['away_short']}: H {p['home']:.0%} · D {p['draw']:.0%} · A {p['away']:.0%}"
                f" | {f['most_likely']['score']}")
        if f.get("value"):
            best = max(f["value"], key=lambda v: v["ev"])
            line += f" | +EV {best['label']} @{best['odds']:.2f} ({best['ev']:+.0%})"
        lines.append(line)
    if not gw.get("fixtures"):
        lines.append("No upcoming fixtures found.")
    lines.append("Model output, not betting advice.")
    return "\n".join(lines)


def _chunks(text: str, limit: int) -> list[str]:
    out, current = [], ""
    for line in text.splitlines():
        if len(current) + len(line) + 1 > limit:
            out.append(current)
            current = ""
        current += line + "\n"
    if current:
        out.append(current)
    return out


def send(text: str) -> dict[str, str]:
    """Post to every configured channel. Never raises: returns channel -> 'ok' / error."""
    results = {}
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if url:
        try:
            for chunk in _chunks(text, DISCORD_LIMIT):
                requests.post(url, json={"content": chunk}, timeout=15).raise_for_status()
            results["discord"] = "ok"
        except requests.RequestException as exc:
            log.error("Discord webhook failed: %s", exc)
            results["discord"] = str(exc)
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            for chunk in _chunks(text, TELEGRAM_LIMIT):
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": chunk}, timeout=15).raise_for_status()
            results["telegram"] = "ok"
        except requests.RequestException as exc:
            # Never log the URL: it contains the bot token.
            log.error("Telegram notification failed: %s", type(exc).__name__)
            results["telegram"] = type(exc).__name__
    return results
