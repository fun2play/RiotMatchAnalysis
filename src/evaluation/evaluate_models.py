from __future__ import annotations

import pandas as pd

from src.utils.io import read_csv
from src.utils.paths import BEST_MODELS_PATH, MODEL_METRICS_PATH

from .reporting import save_reports
from .stage_contribution import make_stage_contribution_table


def evaluate_saved_models() -> dict[str, pd.DataFrame]:
    metrics_frame = read_csv(MODEL_METRICS_PATH)
    best_frame = read_csv(BEST_MODELS_PATH)
    contribution_frame = make_stage_contribution_table(best_frame)
    save_reports(metrics_frame=metrics_frame, best_frame=best_frame, contribution_frame=contribution_frame)
    return {
        "metrics": metrics_frame,
        "best_models": best_frame,
        "contribution": contribution_frame,
    }
