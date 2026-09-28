"""Logging setup and a hardened HTTP client shared by the data modules."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import LOG_DIR

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36 epl-match-predictor/1.0"
)


class DataSourceError(RuntimeError):
    """A remote data source could not be reached or returned unusable content."""


def setup_logging(verbose: bool = False) -> None:
    from rich.logging import RichHandler

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root.addHandler(console)

    file_handler = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    for noisy in ("urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_session: requests.Session | None = None


def http_session() -> requests.Session:
    global _session
    if _session is None:
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.8,  # ~0.8s, 1.6s, 3.2s between attempts
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
        _session = session
    return _session


def fetch(url: str, *, headers: dict | None = None, timeout: float = 25.0) -> requests.Response:
    """GET with retries and logging. Raises DataSourceError on any failure."""
    started = time.monotonic()
    try:
        response = http_session().get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        log.error("GET %s failed: %s", url, exc)
        raise DataSourceError(f"request to {url} failed: {exc}") from exc
    elapsed = time.monotonic() - started
    log.debug("GET %s -> %s (%d bytes, %.2fs)", url, response.status_code, len(response.content), elapsed)
    if response.status_code != 200:
        log.error("GET %s returned HTTP %s", url, response.status_code)
        raise DataSourceError(f"{url} returned HTTP {response.status_code}")
    if not response.content:
        raise DataSourceError(f"{url} returned an empty body")
    return response


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write via a temp file + rename so a failed download never corrupts the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
