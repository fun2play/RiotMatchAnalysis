from __future__ import annotations

import logging
import time

from .riot_api_client import RiotAPIClient


class MatchCollector:
    def __init__(self, api_key: str, region: str = "americas", platform_region: str = "na1") -> None:
        self.client = RiotAPIClient(api_key=api_key, region=region, platform_region=platform_region)
        self.logger = logging.getLogger(__name__)

    def get_master_puuids(self, queue_type: str = "RANKED_SOLO_5x5", max_players: int | None = None) -> list[str]:
        payload = self.client.get_master_league(queue_type=queue_type)
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            raise RuntimeError(f"Master league payload is malformed: expected list entries, got {type(entries).__name__}.")
        if not entries:
            raise RuntimeError("Master league returned zero entries. Check your Riot API key, region, or current endpoint response.")

        puuids: list[str] = []
        seen: set[str] = set()
        resolved = 0
        total_entries = len(entries)
        target_players = max_players if max_players is not None else total_entries
        self.logger.info("Master seeding started: %s league entries, target=%s players", total_entries, target_players)
        for index, entry in enumerate(entries, start=1):
            if max_players is not None and len(puuids) >= max_players:
                self.logger.info("Master seeding stopped at configured limit: %s players", max_players)
                break
            if not isinstance(entry, dict):
                self.logger.warning("Skipping malformed master entry at position %s", index)
                continue
            puuid = entry.get("puuid")
            if not puuid:
                summoner_id = entry.get("summonerId")
                if not summoner_id:
                    self.logger.warning(
                        "Skipping master entry %s/%s with neither puuid nor summonerId",
                        index,
                        total_entries,
                    )
                    continue
                summoner = self.client.get_summoner_by_id(summoner_id)
                puuid = summoner.get("puuid")
            if puuid and puuid not in seen:
                puuids.append(puuid)
                seen.add(puuid)
                resolved += 1
            self.logger.info(
                "Master seed progress: %s/%s entries processed, %s/%s puuids resolved",
                index,
                total_entries,
                resolved,
                target_players,
            )
            time.sleep(0.5)
        if not puuids:
            raise RuntimeError("Master entries were fetched, but no PUUIDs could be resolved from summoner lookups.")
        return puuids

    def get_match_ids_for_puuids(
        self,
        puuids: list[str],
        count: int = 20,
        queue: int | None = None,
        match_type: str | None = "ranked",
        delay: float = 1.0,
    ) -> list[str]:
        match_ids: list[str] = []
        seen: set[str] = set()
        total_puuids = len(puuids)
        self.logger.info("Match ID collection started: %s seed players, up to %s matches each", total_puuids, count)
        for index, puuid in enumerate(puuids, start=1):
            player_match_ids = self.client.get_match_ids(
                puuid=puuid,
                count=count,
                queue=queue,
                match_type=match_type,
            )
            for match_id in player_match_ids:
                if match_id not in seen:
                    match_ids.append(match_id)
                    seen.add(match_id)
            self.logger.info(
                "Match ID progress: %s/%s players processed, %s unique match ids collected",
                index,
                total_puuids,
                len(match_ids),
            )
            time.sleep(delay)
        if not match_ids:
            raise RuntimeError("No ranked match IDs were returned for the resolved Master players.")
        return match_ids
