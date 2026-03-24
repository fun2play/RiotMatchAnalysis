from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.paths import STAGE_LABELS, STAGE_RATIOS


FEATURE_COLUMNS = [
    "blue_total_gold",
    "red_total_gold",
    "gold_diff",
    "blue_total_kills",
    "red_total_kills",
    "kill_diff",
    "blue_total_deaths",
    "red_total_deaths",
    "death_diff",
    "blue_tower_kills",
    "red_tower_kills",
    "tower_diff",
    "blue_dragon_kills",
    "red_dragon_kills",
    "dragon_diff",
    "blue_herald_kills",
    "red_herald_kills",
    "herald_diff",
    "blue_baron_kills",
    "red_baron_kills",
    "baron_diff",
    "blue_inhibitor_kills",
    "red_inhibitor_kills",
    "inhibitor_diff",
    "blue_avg_level",
    "red_avg_level",
    "avg_level_diff",
    "blue_minions_killed",
    "red_minions_killed",
    "minion_diff",
    "blue_jungle_minions_killed",
    "red_jungle_minions_killed",
    "jungle_diff",
    "elapsed_time_seconds",
    "elapsed_time_ratio",
    "first_blood_blue",
    "first_tower_blue",
]


def get_stage_points(duration_seconds: int) -> list[dict[str, float | int | str]]:
    stage_points: list[dict[str, float | int | str]] = []
    for stage_label in STAGE_LABELS:
        ratio = STAGE_RATIOS[stage_label]
        elapsed_time_seconds = max(1, int(round(duration_seconds * ratio)))
        stage_points.append(
            {
                "stage_label": stage_label,
                "elapsed_time_ratio": ratio,
                "elapsed_time_seconds": elapsed_time_seconds,
                "target_timestamp_ms": elapsed_time_seconds * 1000,
            }
        )
    return stage_points


def get_game_duration_seconds(match_data: dict[str, Any]) -> int:
    info = match_data.get("info", {})
    game_duration = info.get("gameDuration")
    if isinstance(game_duration, (int, float)) and game_duration > 1000:
        return int(round(game_duration / 1000))
    if isinstance(game_duration, (int, float)):
        return int(round(game_duration))
    start = info.get("gameStartTimestamp")
    end = info.get("gameEndTimestamp")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return int(round((end - start) / 1000))
    return 0


def get_blue_win(match_data: dict[str, Any]) -> int | None:
    teams = match_data.get("info", {}).get("teams", [])
    for team in teams:
        if team.get("teamId") == 100:
            win_value = team.get("win")
            return int(bool(win_value == "Win" or win_value is True))
    participants = match_data.get("info", {}).get("participants", [])
    blue_players = [player for player in participants if player.get("teamId") == 100]
    if blue_players:
        return int(bool(blue_players[0].get("win")))
    return None


def get_participant_team_map(match_data: dict[str, Any]) -> dict[int, int]:
    participants = match_data.get("info", {}).get("participants", [])
    mapping: dict[int, int] = {}
    for participant in participants:
        participant_id = participant.get("participantId")
        team_id = participant.get("teamId")
        if isinstance(participant_id, int) and isinstance(team_id, int):
            mapping[participant_id] = team_id
    if mapping:
        return mapping
    return {participant_id: 100 if participant_id <= 5 else 200 for participant_id in range(1, 11)}


def get_timeline_frames(
    timeline_data: dict[str, Any],
    frames_data: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    frames = timeline_data.get("info", {}).get("frames") or timeline_data.get("frames")
    if isinstance(frames, list):
        return [frame for frame in frames if isinstance(frame, dict)]
    return frames_data or []


def build_feature_values(
    match_data: dict[str, Any],
    timeline_data: dict[str, Any],
    frames_data: list[dict[str, Any]] | None,
    target_timestamp_ms: int,
    elapsed_time_seconds: int,
    elapsed_time_ratio: float,
) -> dict[str, float | int]:
    participant_team_map = get_participant_team_map(match_data)
    frames = get_timeline_frames(timeline_data, frames_data)
    snapshot = get_snapshot_frame(frames, target_timestamp_ms)
    feature_values = empty_feature_values()
    feature_values.update(snapshot_features(snapshot, participant_team_map))
    feature_values.update(event_features(frames, participant_team_map, target_timestamp_ms))
    feature_values["elapsed_time_seconds"] = elapsed_time_seconds
    feature_values["elapsed_time_ratio"] = elapsed_time_ratio
    return feature_values


def empty_feature_values() -> dict[str, float | int]:
    return {feature: 0 for feature in FEATURE_COLUMNS}


def get_snapshot_frame(frames: list[dict[str, Any]], target_timestamp_ms: int) -> dict[str, Any] | None:
    if not frames:
        return None
    eligible = [frame for frame in frames if int(frame.get("timestamp", 0)) <= target_timestamp_ms]
    if eligible:
        return max(eligible, key=lambda frame: int(frame.get("timestamp", 0)))
    return min(frames, key=lambda frame: int(frame.get("timestamp", 0)))


def snapshot_features(
    snapshot: dict[str, Any] | None,
    participant_team_map: dict[int, int],
) -> dict[str, float | int]:
    values = {
        "blue_total_gold": 0,
        "red_total_gold": 0,
        "blue_avg_level": 0.0,
        "red_avg_level": 0.0,
        "blue_minions_killed": 0,
        "red_minions_killed": 0,
        "blue_jungle_minions_killed": 0,
        "red_jungle_minions_killed": 0,
    }
    if snapshot is None:
        values["gold_diff"] = 0
        values["minion_diff"] = 0
        values["jungle_diff"] = 0
        values["avg_level_diff"] = 0.0
        return values

    blue_levels: list[int] = []
    red_levels: list[int] = []
    participant_frames = snapshot.get("participantFrames", {})

    for participant_id_raw, participant_frame in participant_frames.items():
        participant_id = int(participant_id_raw)
        team_id = participant_team_map.get(participant_id, 100 if participant_id <= 5 else 200)
        total_gold = int(participant_frame.get("totalGold", 0) or 0)
        level = int(participant_frame.get("level", 0) or 0)
        minions = int(participant_frame.get("minionsKilled", 0) or 0)
        jungle_minions = int(participant_frame.get("jungleMinionsKilled", 0) or 0)

        if team_id == 100:
            values["blue_total_gold"] += total_gold
            values["blue_minions_killed"] += minions
            values["blue_jungle_minions_killed"] += jungle_minions
            blue_levels.append(level)
        else:
            values["red_total_gold"] += total_gold
            values["red_minions_killed"] += minions
            values["red_jungle_minions_killed"] += jungle_minions
            red_levels.append(level)

    values["blue_avg_level"] = round(sum(blue_levels) / len(blue_levels), 4) if blue_levels else 0.0
    values["red_avg_level"] = round(sum(red_levels) / len(red_levels), 4) if red_levels else 0.0
    values["gold_diff"] = int(values["blue_total_gold"]) - int(values["red_total_gold"])
    values["minion_diff"] = int(values["blue_minions_killed"]) - int(values["red_minions_killed"])
    values["jungle_diff"] = int(values["blue_jungle_minions_killed"]) - int(values["red_jungle_minions_killed"])
    values["avg_level_diff"] = round(float(values["blue_avg_level"]) - float(values["red_avg_level"]), 4)
    return values


def event_features(
    frames: list[dict[str, Any]],
    participant_team_map: dict[int, int],
    target_timestamp_ms: int,
) -> dict[str, int]:
    values = {
        "blue_total_kills": 0,
        "red_total_kills": 0,
        "blue_total_deaths": 0,
        "red_total_deaths": 0,
        "blue_tower_kills": 0,
        "red_tower_kills": 0,
        "blue_dragon_kills": 0,
        "red_dragon_kills": 0,
        "blue_herald_kills": 0,
        "red_herald_kills": 0,
        "blue_baron_kills": 0,
        "red_baron_kills": 0,
        "blue_inhibitor_kills": 0,
        "red_inhibitor_kills": 0,
        "first_blood_blue": 0,
        "first_tower_blue": 0,
    }

    first_blood_seen = False
    first_tower_seen = False

    for frame in frames:
        for event in frame.get("events", []):
            timestamp = int(event.get("timestamp", frame.get("timestamp", 0)) or 0)
            if timestamp > target_timestamp_ms:
                continue

            event_type = event.get("type")
            if event_type == "CHAMPION_KILL":
                killer_side = get_event_side(event, participant_team_map)
                victim_side = get_victim_side(event, participant_team_map)
                if killer_side == "blue":
                    values["blue_total_kills"] += 1
                elif killer_side == "red":
                    values["red_total_kills"] += 1
                if victim_side == "blue":
                    values["blue_total_deaths"] += 1
                elif victim_side == "red":
                    values["red_total_deaths"] += 1
                if not first_blood_seen and killer_side is not None:
                    values["first_blood_blue"] = int(killer_side == "blue")
                    first_blood_seen = True

            elif event_type == "BUILDING_KILL":
                killer_side = get_event_side(event, participant_team_map)
                if killer_side is None:
                    killer_side = get_opposite_side(event.get("teamId"))
                building_type = event.get("buildingType")
                if building_type == "TOWER_BUILDING" and killer_side is not None:
                    values[f"{killer_side}_tower_kills"] += 1
                    if not first_tower_seen:
                        values["first_tower_blue"] = int(killer_side == "blue")
                        first_tower_seen = True
                elif building_type == "INHIBITOR_BUILDING" and killer_side is not None:
                    values[f"{killer_side}_inhibitor_kills"] += 1

            elif event_type == "ELITE_MONSTER_KILL":
                killer_side = get_event_side(event, participant_team_map)
                monster_type = str(event.get("monsterType", "")).upper()
                if killer_side == "blue" and monster_type == "DRAGON":
                    values["blue_dragon_kills"] += 1
                elif killer_side == "red" and monster_type == "DRAGON":
                    values["red_dragon_kills"] += 1
                elif killer_side == "blue" and monster_type == "RIFTHERALD":
                    values["blue_herald_kills"] += 1
                elif killer_side == "red" and monster_type == "RIFTHERALD":
                    values["red_herald_kills"] += 1
                elif killer_side == "blue" and monster_type == "BARON_NASHOR":
                    values["blue_baron_kills"] += 1
                elif killer_side == "red" and monster_type == "BARON_NASHOR":
                    values["red_baron_kills"] += 1

    values["kill_diff"] = values["blue_total_kills"] - values["red_total_kills"]
    values["death_diff"] = values["blue_total_deaths"] - values["red_total_deaths"]
    values["tower_diff"] = values["blue_tower_kills"] - values["red_tower_kills"]
    values["dragon_diff"] = values["blue_dragon_kills"] - values["red_dragon_kills"]
    values["herald_diff"] = values["blue_herald_kills"] - values["red_herald_kills"]
    values["baron_diff"] = values["blue_baron_kills"] - values["red_baron_kills"]
    values["inhibitor_diff"] = values["blue_inhibitor_kills"] - values["red_inhibitor_kills"]
    return values


def get_event_side(event: dict[str, Any], participant_team_map: dict[int, int]) -> str | None:
    for key in ("killerId", "participantId", "creatorId"):
        participant_id = event.get(key)
        if isinstance(participant_id, int) and participant_id > 0:
            team_id = participant_team_map.get(participant_id)
            if team_id == 100:
                return "blue"
            if team_id == 200:
                return "red"
    killer_team_id = event.get("killerTeamId")
    if killer_team_id == 100:
        return "blue"
    if killer_team_id == 200:
        return "red"
    return None


def get_victim_side(event: dict[str, Any], participant_team_map: dict[int, int]) -> str | None:
    victim_id = event.get("victimId")
    if not isinstance(victim_id, int):
        return None
    team_id = participant_team_map.get(victim_id)
    if team_id == 100:
        return "blue"
    if team_id == 200:
        return "red"
    return None


def get_opposite_side(team_id: Any) -> str | None:
    if team_id == 100:
        return "red"
    if team_id == 200:
        return "blue"
    return None


def normalize_stage_frame(stage_frame: pd.DataFrame) -> pd.DataFrame:
    if stage_frame.empty:
        return stage_frame
    numeric_columns = [column for column in FEATURE_COLUMNS if column in stage_frame.columns]
    train_mask = stage_frame["split"] == "train"
    if not train_mask.any():
        return stage_frame
    scaler = StandardScaler()
    stage_frame = stage_frame.copy()
    stage_frame.loc[train_mask, numeric_columns] = scaler.fit_transform(
        stage_frame.loc[train_mask, numeric_columns]
    ).astype(float)
    if (~train_mask).any():
        stage_frame.loc[~train_mask, numeric_columns] = scaler.transform(stage_frame.loc[~train_mask, numeric_columns])
    return stage_frame
