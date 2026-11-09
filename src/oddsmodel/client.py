"""Thin client for The Odds API (v4): https://the-odds-api.com/liveapi/guides/v4/"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)
BASE_URL = "https://api.the-odds-api.com/v4"


class OddsAPIClient:
    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set ODDS_API_KEY in the environment or .env")
        self.timeout = timeout
        self.session = requests.Session()
        self.requests_remaining: str | None = None

    def _get(self, path: str, **params: Any) -> Any:
        params["apiKey"] = self.api_key
        r = self.session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        self.requests_remaining = r.headers.get("x-requests-remaining")
        log.info("Odds API quota remaining: %s", self.requests_remaining)
        return r.json()

    def sports(self) -> list[dict]:
        return self._get("/sports")

    def odds(
        self,
        sport: str,
        regions: str = "uk,eu",
        markets: str = "h2h",
        odds_format: str = "decimal",
    ) -> list[dict]:
        return self._get(
            f"/sports/{sport}/odds",
            regions=regions,
            markets=markets,
            oddsFormat=odds_format,
        )


def load_snapshot(path: str | Path) -> list[dict]:
    """Load a saved API response (lets the pipeline run offline / reproducibly)."""
    return json.loads(Path(path).read_text())


def save_snapshot(events: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(events, indent=2))
