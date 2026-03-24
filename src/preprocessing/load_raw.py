from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.io import read_json
from src.utils.paths import RAW_FRAMES_DIR, RAW_MATCHES_DIR, RAW_TIMELINES_DIR


@dataclass
class RawMatch:
    match_id: str
    match_data: dict[str, Any]
    timeline_data: dict[str, Any]
    frames_data: list[dict[str, Any]] | None


def load_raw_matches() -> tuple[list[RawMatch], pd.DataFrame]:
    match_files = {path.stem: path for path in RAW_MATCHES_DIR.glob("*.json")}
    timeline_files = {path.stem: path for path in RAW_TIMELINES_DIR.glob("*.json")}
    frames_files = {path.stem: path for path in RAW_FRAMES_DIR.glob("*.json")}

    all_match_ids = sorted(set(match_files) | set(timeline_files))
    raw_matches: list[RawMatch] = []
    report_rows: list[dict[str, str]] = []

    for match_id in all_match_ids:
        match_path = match_files.get(match_id)
        timeline_path = timeline_files.get(match_id)

        if match_path is None or timeline_path is None:
            report_rows.append(
                {"match_id": match_id, "status": "skipped", "reason": "missing_match_or_timeline"}
            )
            continue

        try:
            match_data = read_json(match_path)
            timeline_data = read_json(timeline_path)
            frames_payload = read_json(frames_files[match_id]) if match_id in frames_files else None
        except Exception:
            report_rows.append({"match_id": match_id, "status": "skipped", "reason": "json_read_error"})
            continue

        if not isinstance(match_data, dict) or not isinstance(timeline_data, dict):
            report_rows.append({"match_id": match_id, "status": "skipped", "reason": "malformed_json"})
            continue

        frames_data = frames_payload if isinstance(frames_payload, list) else None
        raw_matches.append(
            RawMatch(
                match_id=match_id,
                match_data=match_data,
                timeline_data=timeline_data,
                frames_data=frames_data,
            )
        )
        report_rows.append({"match_id": match_id, "status": "loaded", "reason": "ok"})

    return raw_matches, pd.DataFrame(report_rows)
