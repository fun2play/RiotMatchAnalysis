from __future__ import annotations

from typing import Any

from .engineer_features import build_feature_values, get_blue_win, get_game_duration_seconds, get_stage_points
from .load_raw import RawMatch


def build_stage_rows(raw_match: RawMatch) -> list[dict[str, Any]]:
    duration_seconds = get_game_duration_seconds(raw_match.match_data)
    blue_win = get_blue_win(raw_match.match_data)
    if duration_seconds <= 0 or blue_win is None:
        return []

    rows: list[dict[str, Any]] = []
    for stage_point in get_stage_points(duration_seconds):
        feature_values = build_feature_values(
            match_data=raw_match.match_data,
            timeline_data=raw_match.timeline_data,
            frames_data=raw_match.frames_data,
            target_timestamp_ms=int(stage_point["target_timestamp_ms"]),
            elapsed_time_seconds=int(stage_point["elapsed_time_seconds"]),
            elapsed_time_ratio=float(stage_point["elapsed_time_ratio"]),
        )
        rows.append(
            {
                "match_id": raw_match.match_id,
                "stage_label": stage_point["stage_label"],
                "blue_win": blue_win,
                **feature_values,
            }
        )
    return rows
