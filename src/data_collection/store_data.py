from __future__ import annotations

import logging
import time
from pathlib import Path

from src.utils.io import write_json
from src.utils.paths import ensure_directories

from .match_collector import MatchCollector
from .riot_api_client import RiotAPIClient


class StoreData:
    RANKED_SOLO_QUEUE_ID = 420

    def __init__(
        self,
        api_key: str,
        region: str = "americas",
        platform_region: str = "na1",
        base_dir: str = "data/raw",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.matches_dir = self.base_dir / "matches"
        self.timelines_dir = self.base_dir / "timelines"
        self.frames_dir = self.base_dir / "frames"
        self.logger = logging.getLogger(__name__)
        self.client = RiotAPIClient(api_key=api_key, region=region, platform_region=platform_region)
        self.collector = MatchCollector(api_key=api_key, region=region, platform_region=platform_region)
        ensure_directories()
        for directory in (self.base_dir, self.matches_dir, self.timelines_dir, self.frames_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def store_from_master(
        self,
        match_count: int = 20,
        delay: float = 1.0,
        queue: int = RANKED_SOLO_QUEUE_ID,
        match_type: str | None = "ranked",
        max_players: int | None = None,
    ) -> dict[str, int]:
        puuids = self.collector.get_master_puuids(max_players=max_players)
        return self.store_for_puuids(
            puuids=puuids,
            match_count=match_count,
            delay=delay,
            queue=queue,
            match_type=match_type,
        )

    def store_for_puuids(
        self,
        puuids: list[str],
        match_count: int = 20,
        delay: float = 1.0,
        queue: int | None = RANKED_SOLO_QUEUE_ID,
        match_type: str | None = "ranked",
    ) -> dict[str, int]:
        if not puuids:
            raise RuntimeError("No PUUIDs were provided to the storage pipeline.")

        totals = {
            "matches": 0,
            "timelines": 0,
            "frames": 0,
            "skipped_cached": 0,
            "skipped_wrong_queue": 0,
            "failed_matches": 0,
            "failed_timelines": 0,
        }
        match_ids = self.collector.get_match_ids_for_puuids(
            puuids=puuids,
            count=match_count,
            queue=queue,
            match_type=match_type,
            delay=delay,
        )
        total_match_ids = len(match_ids)
        self.logger.info("Download started: %s unique match ids queued", total_match_ids)
        for index, match_id in enumerate(match_ids, start=1):
            if self.is_cached(match_id):
                totals["skipped_cached"] += 1
                self.logger.info(
                    "Download progress: %s/%s matches processed | saved=%s cached=%s wrong_queue=%s failed_match=%s failed_timeline=%s",
                    index,
                    total_match_ids,
                    totals["matches"],
                    totals["skipped_cached"],
                    totals["skipped_wrong_queue"],
                    totals["failed_matches"],
                    totals["failed_timelines"],
                )
                continue

            match_saved, match_data = self.store_match(match_id, expected_queue=queue)
            if not match_saved:
                if match_data is None:
                    totals["failed_matches"] += 1
                else:
                    totals["skipped_wrong_queue"] += 1
                self.logger.info(
                    "Download progress: %s/%s matches processed | saved=%s cached=%s wrong_queue=%s failed_match=%s failed_timeline=%s",
                    index,
                    total_match_ids,
                    totals["matches"],
                    totals["skipped_cached"],
                    totals["skipped_wrong_queue"],
                    totals["failed_matches"],
                    totals["failed_timelines"],
                )
                time.sleep(delay)
                continue

            timeline_saved, frames_saved = self.store_timeline(match_id)
            if not timeline_saved:
                totals["failed_timelines"] += 1
                self.remove_partial_match(match_id)
                self.logger.info(
                    "Download progress: %s/%s matches processed | saved=%s timelines=%s frames=%s cached=%s wrong_queue=%s failed_match=%s failed_timeline=%s",
                    index,
                    total_match_ids,
                    totals["matches"],
                    totals["timelines"],
                    totals["frames"],
                    totals["skipped_cached"],
                    totals["skipped_wrong_queue"],
                    totals["failed_matches"],
                    totals["failed_timelines"],
                )
                time.sleep(delay)
                continue
            totals["matches"] += 1
            totals["timelines"] += int(timeline_saved)
            totals["frames"] += int(frames_saved)
            self.logger.info(
                "Download progress: %s/%s matches processed | saved=%s timelines=%s frames=%s cached=%s wrong_queue=%s failed_match=%s failed_timeline=%s",
                index,
                total_match_ids,
                totals["matches"],
                totals["timelines"],
                totals["frames"],
                totals["skipped_cached"],
                totals["skipped_wrong_queue"],
                totals["failed_matches"],
                totals["failed_timelines"],
            )
            time.sleep(delay)
        return totals

    def is_cached(self, match_id: str) -> bool:
        match_path = self.matches_dir / f"{match_id}.json"
        timeline_path = self.timelines_dir / f"{match_id}.json"
        return match_path.exists() and timeline_path.exists()

    def remove_partial_match(self, match_id: str) -> None:
        for path in (
            self.matches_dir / f"{match_id}.json",
            self.timelines_dir / f"{match_id}.json",
            self.frames_dir / f"{match_id}.json",
        ):
            if path.exists():
                path.unlink()

    def store_match(self, match_id: str, expected_queue: int | None = None) -> tuple[bool, dict | None]:
        match_data = self.client.get_match(match_id)
        if not match_data:
            self.logger.warning("Match download failed for %s", match_id)
            return False, None
        queue_id = match_data.get("info", {}).get("queueId")
        if expected_queue is not None and queue_id != expected_queue:
            self.logger.warning(
                "Skipping match %s because queueId=%s does not match expected queue %s",
                match_id,
                queue_id,
                expected_queue,
            )
            return False, match_data
        write_json(self.matches_dir / f"{match_id}.json", match_data)
        return True, match_data

    def store_timeline(self, match_id: str) -> tuple[bool, bool]:
        timeline_data = self.client.get_timeline(match_id)
        if not timeline_data:
            self.logger.warning("Timeline download failed for %s", match_id)
            return False, False
        write_json(self.timelines_dir / f"{match_id}.json", timeline_data)
        frames = timeline_data.get("info", {}).get("frames") or timeline_data.get("frames")
        if not isinstance(frames, list):
            return True, False
        write_json(self.frames_dir / f"{match_id}.json", frames)
        return True, True
