import json
import logging
import os
import time
import requests # type: ignore
import pandas as pd # type: ignore 

from typing import Dict, Iterable, List, Optional

class TimelineParser:
    def __init__(self, timelines: List[Dict]):
        self.timelines = timelines
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def parse_timelines(self) -> pd.DataFrame:
        all_events = []
        for timeline in self.timelines:
            match_id = timeline.get("metadata", {}).get("matchId")
            for frame in timeline.get('info', {}).get('frames', []):
                timestamp = frame.get('timestamp', 0)
                for event in frame.get('events', []):
                    event_data = {
                        'match_id': match_id,
                        'timestamp': timestamp,
                        'event_type': event.get('type'),
                        'participant_id': event.get('participantId'),
                        'position': event.get('position'),
                        'killer_id': event.get('killerId'),
                        'victim_id': event.get('victimId'),
                        'assisting_participant_ids': event.get('assistingParticipantIds'),
                        'item_id': event.get('itemId'),
                        'skill_slot': event.get('skillSlot'),
                        'level_up_type': event.get('levelUpType'),
                        'ward_type': event.get('wardType'),
                    }
                    all_events.append(event_data)
        df = pd.DataFrame(all_events)
        return df

    def parse_frames(self) -> pd.DataFrame:
        """
        Parse per-participant frame stats for windowed delta features.
        These stats enable relative and volatility features without using outcome targets.
        """
        all_frames = []
        for timeline in self.timelines:
            match_id = timeline.get("metadata", {}).get("matchId")
            for frame in timeline.get('info', {}).get('frames', []):
                timestamp = frame.get('timestamp', 0)
                participant_frames = frame.get('participantFrames', {})
                for participant_id, pdata in participant_frames.items():
                    damage_stats = pdata.get("damageStats", {}) or {}
                    position = pdata.get("position", {}) or {}
                    all_frames.append(
                        {
                            "match_id": match_id,
                            "timestamp": timestamp,
                            "participant_id": int(participant_id),
                            "total_gold": pdata.get("totalGold"),
                            "current_gold": pdata.get("currentGold"),
                            "level": pdata.get("level"),
                            "xp": pdata.get("xp"),
                            "minions_killed": pdata.get("minionsKilled"),
                            "jungle_minions_killed": pdata.get("jungleMinionsKilled"),
                            "damage_done_to_champions": damage_stats.get("totalDamageDoneToChampions"),
                            "damage_taken": damage_stats.get("totalDamageTaken"),
                            "position_x": position.get("x"),
                            "position_y": position.get("y"),
                        }
                    )
        return pd.DataFrame(all_frames)
    
    
