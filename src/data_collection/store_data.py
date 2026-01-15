import json
import logging
import os
import time
from typing import Dict, Iterable, List, Optional

from .match_collector import MatchCollector
from .riot_api_client import RiotAPIClient


class StoreData:
    def __init__(self, api_key: str, region: str = "na1", base_dir: str = "data/raw") -> None:
        self.api_key = api_key
        self.region = region
        self.base_url = f"https://{region}.api.riotgames.com"
        self.base_dir = base_dir
        self.matches_dir = os.path.join(base_dir, "matches")
        self.timelines_dir = os.path.join(base_dir, "timelines")
        self.frames_dir = os.path.join(base_dir, "frames")
        self.client = RiotAPIClient(api_key, region=region)
        self.collector = MatchCollector(api_key, region=region)
        self._ensure_dirs()
        self.logger = logging.getLogger(__name__)

    def store_from_master(self, tag: str, match_count: int = 20, delay: float = 1.0) -> Dict[str, int]:
        puuids = self.collector.get_master_puuids(self.api_key, tag)
        return self.store_for_puuids(puuids, match_count=match_count, delay=delay)

    def store_from_summoner_names(self, summoner_names: List[str], tag: str,match_count: int = 20, delay: float = 1.0) -> Dict[str, int]:
        summoners = self.collector.get_summoners(self.api_key, list(summoner_names), tag)
        puuids = [summoner.get("puuid") for summoner in summoners if summoner.get("puuid")]
        return self.store_for_puuids(puuids, match_count=match_count, delay=delay)

    def store_for_puuids(self, puuids: List[str], match_count: int = 20, delay: float = 1.0) -> Dict[str, int]:
        totals = {"matches": 0, "timelines": 0, "frames": 0}
        for puuid in puuids:
            match_ids = self.client.get_match_ids(puuid)
            if not isinstance(match_ids, list):
                continue
            for match_id in match_ids[:match_count]:
                if self.store_match(match_id):
                    totals["matches"] += 1
                timeline_saved, frames_saved = self.store_timeline(match_id)
                totals["timelines"] += int(timeline_saved)
                totals["frames"] += int(frames_saved)
                time.sleep(delay)
        return totals

    def store_match(self, match_id: str) -> bool:
        match_data = self.client.get_match(match_id)
        if not match_data:
            return False
        path = self._json_path(self.matches_dir, match_id)
        self._write_json(path, match_data)
        return True

    def store_timeline(self, match_id: str) -> tuple[bool, bool]:
        timeline_data = self.client.get_timeline(match_id)
        if not timeline_data:
            return False, False
        timeline_path = self._json_path(self.timelines_dir, match_id)
        self._write_json(timeline_path, timeline_data)
        frames_saved = self._store_frames(match_id, timeline_data)
        return True, frames_saved

    def _store_frames(self, match_id: str, timeline_data: Dict) -> bool:
        frames = timeline_data.get("info", {}).get("frames") or timeline_data.get("frames")
        if not frames:
            return False
        frames_path = self._json_path(self.frames_dir, match_id)
        self._write_json(frames_path, frames)
        return True

    def _ensure_dirs(self) -> None:
        for directory in (self.base_dir, self.matches_dir, self.timelines_dir, self.frames_dir):
            os.makedirs(directory, exist_ok=True)

    def _json_path(self, directory: str, name: str) -> str:
        safe_name = name.replace(os.sep, "_")
        return os.path.join(directory, f"{safe_name}.json")

    def _write_json(self, path: str, payload: Dict) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
    
