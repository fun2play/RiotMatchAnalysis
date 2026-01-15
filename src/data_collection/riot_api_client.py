import json
import logging
import os
import time
import requests # type: ignore
import pandas as pd # type: ignore 
from typing import Dict, Iterable, List, Optional

from warnings import simplefilter

class RiotAPIClient:
    def __init__(self, api_key: str, region: str = 'na1'):
        self.api_key = api_key
        self.base_url = f"https://{region}.api.riotgames.com"
        self.session = requests.Session()
        self.session.params.update({'api_key': self.api_key})
        simplefilter('ignore', pd.errors.PerformanceWarning)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_match_ids(self, puuid: str) -> Dict:
        url = f"{self.base_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        response = self.session.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Failed to retrieve match IDs: {response.status_code}")
            return {}

    def get_match(self, match_id: str) -> Dict:
        url = f"{self.base_url}/lol/match/v5/matches/{match_id}"
        response = self.session.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Failed to retrieve match data: {response.status_code}")
            return {}
    
    def get_timeline(self, match_id: str) -> Dict:
        url = f"{self.base_url}/lol/match/v5/matches/{match_id}/timeline"
        response = self.session.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Failed to retrieve timeline data: {response.status_code}")
            return {}
        
    def fetch_and_cache(self, match_ids: List[str], delay: float = 1.0) -> List[Dict]:
        matches = []
        for match_id in match_ids:
            match_data = self.get_match(match_id)
            if match_data:
                matches.append(match_data)
            time.sleep(delay)  # To respect rate limits
        return matches
    
    def fetch_timelines(self, match_ids: List[str], delay: float = 1.0) -> List[Dict]:
        timelines = []
        for match_id in match_ids:
            timeline_data = self.get_timeline(match_id)
            if timeline_data:
                timelines.append(timeline_data)
            time.sleep(delay)  # To respect rate limits
        return timelines
