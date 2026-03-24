from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_MATCHES_DIR = RAW_DIR / "matches"
RAW_TIMELINES_DIR = RAW_DIR / "timelines"
RAW_FRAMES_DIR = RAW_DIR / "frames"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
SPLIT_PATH = PROCESSED_DIR / "match_splits.csv"
FILTER_REPORT_PATH = PROCESSED_DIR / "load_report.csv"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.csv"
BEST_MODELS_PATH = MODELS_DIR / "best_models.csv"

STAGE_LABELS = ["stage_20", "stage_40", "stage_60", "stage_80"]
STAGE_RATIOS = {
    "stage_20": 0.2,
    "stage_40": 0.4,
    "stage_60": 0.6,
    "stage_80": 0.8,
}


def ensure_directories() -> None:
    for directory in (
        RAW_MATCHES_DIR,
        RAW_TIMELINES_DIR,
        RAW_FRAMES_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def stage_dataset_path(stage_label: str) -> Path:
    return PROCESSED_DIR / f"{stage_label}.csv"


def stage_model_dir(stage_label: str) -> Path:
    return MODELS_DIR / stage_label
