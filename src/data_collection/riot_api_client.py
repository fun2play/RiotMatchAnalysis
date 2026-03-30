from __future__ import annotations

import logging
import time
from typing import Any

import requests


class RiotAPIClient:
    def __init__(self, api_key: str, region: str = "americas", platform_region: str = "na1") -> None:
        self.api_key = api_key
        self.region = region
        self.platform_region = platform_region
        self.match_base_url = f"https://{region}.api.riotgames.com"
        self.platform_base_url = f"https://{platform_region}.api.riotgames.com"
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": api_key})
        self.logger = logging.getLogger(__name__)

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | list[Any]:
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                if attempt < 2:
                    time.sleep(2 + attempt)
                    continue
                self.logger.error("Riot API request error for %s: %s", url, exc)
                return {}
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else 5.0
                self.logger.warning("Riot API rate limited %s, retrying in %.1fs", url, sleep_seconds)
                time.sleep(sleep_seconds)
                continue
            if response.status_code >= 500 and attempt < 2:
                time.sleep(2 + attempt)
                continue
            self.logger.error(
                "Riot API request failed for %s with status %s and body %s",
                url,
                response.status_code,
                response.text[:300],
            )
            return {}
        self.logger.error("Riot API request failed for %s after retries", url)
        return {}

    def get_match_ids(
        self,
        puuid: str,
        count: int = 20,
        queue: int | None = None,
        match_type: str | None = "ranked",
    ) -> list[str]:
        url = f"{self.match_base_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params: dict[str, Any] = {"count": count}
        if queue is not None:
            params["queue"] = queue
        if match_type is not None:
            params["type"] = match_type
        payload = self._get(url, params=params)
        return payload if isinstance(payload, list) else []

    def get_match(self, match_id: str) -> dict[str, Any]:
        url = f"{self.match_base_url}/lol/match/v5/matches/{match_id}"
        payload = self._get(url)
        return payload if isinstance(payload, dict) else {}

    def get_timeline(self, match_id: str) -> dict[str, Any]:
        url = f"{self.match_base_url}/lol/match/v5/matches/{match_id}/timeline"
        payload = self._get(url)
        return payload if isinstance(payload, dict) else {}

    def get_master_league(self, queue_type: str = "RANKED_SOLO_5x5") -> dict[str, Any]:
        url = f"{self.platform_base_url}/lol/league/v4/masterleagues/by-queue/{queue_type}"
        payload = self._get(url)
        return payload if isinstance(payload, dict) else {}

    def get_summoner_by_id(self, summoner_id: str) -> dict[str, Any]:
        url = f"{self.platform_base_url}/lol/summoner/v4/summoners/{summoner_id}"
        payload = self._get(url)
        return payload if isinstance(payload, dict) else {}
