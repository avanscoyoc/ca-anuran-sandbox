"""Rate-limited HTTP shared by the scrapers."""

from __future__ import annotations

import time
from pathlib import Path

import requests

USER_AGENT = "ca-anuran-research/0.1 (UC Berkeley; avanscoyoc@berkeley.edu)"


class Fetcher:
    """Waits `delay` seconds between network requests; skips files already on disk."""

    def __init__(self, delay: float):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self._last = 0.0

    def get(self, url: str, params: dict | None = None) -> requests.Response | None:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        for attempt in range(4):
            try:
                r = self.session.get(url, params=params, timeout=60)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 3:
                    raise
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r

    def json(self, url: str, params: dict | None = None) -> dict:
        r = self.get(url, params)
        return r.json() if r is not None else {}

    def file(self, url: str, dest: Path) -> bool:
        if dest.exists():
            return True
        r = self.get(url)
        if r is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.rename(dest)  # no truncated files if interrupted
        return True
