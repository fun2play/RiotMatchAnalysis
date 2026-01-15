import json
import logging
import os
import time
import requests # type: ignore
from typing import Dict, Iterable, List, Optional

from .riot_api_client import RiotAPIClient

class MatchCollector:
    def __init__(self, api_key: str, region: str = "na1"):
        self.api_key = api_key
        self.base_url = f"https://{region}.api.riotgames.com"
        self.region = region

    def get_master_puuids(self, api_key: str, tag: str, queue_type: str = "RANKED_SOLO_5x5", region: str = "na1") -> List[str]:
        url = f"{self.base_url}/lol/league/v4/masterleagues/by-queue/{queue_type}?api_key={api_key}"
        response = requests.get(url, params={'api_key': api_key, 'tag': tag})
        if response.status_code == 200:
            summoners = response.json()
            puuids = [summoner['puuid'] for summoner in summoners]
            return puuids
        else:
            logging.error(f"Failed to retrieve master summoners: {response.status_code}")
            return []
        
    def get_matches(self, puuid: List[str]) -> List[str]:
        match_ids = []
        for player_puuid in puuid:
            RiotAPIClient.get_match_ids(self, player_puuid)
            ids = RiotAPIClient.get_match_ids(self, player_puuid)
            match_ids.extend(ids)
            time.sleep(1)  # To respect rate limits
        return match_ids
    
    def fetch_and_cache_matches(self, match_ids: List[str], delay: float = 1.0) -> List[Dict]:
        return RiotAPIClient.fetch_and_cache(self, match_ids, delay)
    
    def fetch_timelines(self, match_ids: List[str], delay: float = 1.0) -> List[Dict]:
        return RiotAPIClient.fetch_timelines(self, match_ids, delay)
    
    
    

    
    