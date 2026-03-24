from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import write_csv
from src.utils.paths import (
    FILTER_REPORT_PATH,
    SPLIT_PATH,
    STAGE_LABELS,
    ensure_directories,
    stage_dataset_path,
    PROCESSED_DIR,
)

from .build_stage_rows import build_stage_rows
from .engineer_features import FEATURE_COLUMNS, get_blue_win, get_game_duration_seconds, normalize_stage_frame
from .load_raw import load_raw_matches


def build_datasets(min_duration_seconds: int = 600) -> dict[str, int]:
    ensure_directories()
    raw_matches, load_report = load_raw_matches()

    if not raw_matches:
        write_csv(FILTER_REPORT_PATH, load_report)
        raise ValueError("No raw match and timeline files were found in data/raw.")

    valid_matches = []
    filter_rows: list[dict[str, str]] = []
    for raw_match in raw_matches:
        duration_seconds = get_game_duration_seconds(raw_match.match_data)
        blue_win = get_blue_win(raw_match.match_data)
        if duration_seconds < min_duration_seconds:
            filter_rows.append({"match_id": raw_match.match_id, "status": "skipped", "reason": "short_game"})
            continue
        if blue_win is None:
            filter_rows.append({"match_id": raw_match.match_id, "status": "skipped", "reason": "missing_outcome"})
            continue
        valid_matches.append(raw_match)
        filter_rows.append({"match_id": raw_match.match_id, "status": "kept", "reason": "ok"})

    write_csv(FILTER_REPORT_PATH, pd.concat([load_report, pd.DataFrame(filter_rows)], ignore_index=True))

    if not valid_matches:
        raise ValueError(f"No matches passed preprocessing. Check {FILTER_REPORT_PATH}.")

    all_rows = []
    match_labels = []
    for raw_match in valid_matches:
        stage_rows = build_stage_rows(raw_match)
        if len(stage_rows) != 4:
            continue
        all_rows.extend(stage_rows)
        match_labels.append({"match_id": raw_match.match_id, "blue_win": stage_rows[0]["blue_win"]})

    if not all_rows:
        raise ValueError("No complete stage rows could be built from the raw matches.")

    split_frame = build_split(pd.DataFrame(match_labels).drop_duplicates("match_id"))
    write_csv(SPLIT_PATH, split_frame)

    all_stages = pd.DataFrame(all_rows)
    for column in FEATURE_COLUMNS:
        if column not in all_stages.columns:
            all_stages[column] = 0
    all_stages = all_stages[["match_id", "stage_label", *FEATURE_COLUMNS, "blue_win"]]
    all_stages = all_stages.merge(split_frame, on="match_id", how="left")
    all_stages = all_stages.dropna().reset_index(drop=True)

    stage_frames = []
    for stage_label in STAGE_LABELS:
        stage_frame = all_stages.loc[all_stages["stage_label"] == stage_label].copy()
        stage_frame = normalize_stage_frame(stage_frame)
        stage_frame = stage_frame.reset_index(drop=True)
        write_csv(stage_dataset_path(stage_label), stage_frame)
        stage_frames.append(stage_frame)

    combined = pd.concat(stage_frames, ignore_index=True)
    write_csv(PROCESSED_DIR / "all_stages.csv", combined)
    return {"matches": len(valid_matches), "rows": len(combined)}


def build_split(match_labels: pd.DataFrame) -> pd.DataFrame:
    stratify = match_labels["blue_win"] if match_labels["blue_win"].nunique() > 1 else None
    try:
        train_ids, test_ids = train_test_split(
            match_labels["match_id"],
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        train_ids, test_ids = train_test_split(
            match_labels["match_id"],
            test_size=0.2,
            random_state=42,
        )

    rows = [{"match_id": match_id, "split": "train"} for match_id in sorted(train_ids)]
    rows.extend({"match_id": match_id, "split": "test"} for match_id in sorted(test_ids))
    return pd.DataFrame(rows)
