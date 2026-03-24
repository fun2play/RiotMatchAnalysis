from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import write_csv
from src.utils.paths import REPORTS_DIR


def save_reports(
    metrics_frame: pd.DataFrame,
    best_frame: pd.DataFrame,
    contribution_frame: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORTS_DIR / "metrics_summary.csv", metrics_frame)
    write_csv(REPORTS_DIR / "best_model_by_stage.csv", best_frame)
    write_csv(REPORTS_DIR / "stage_contribution.csv", contribution_frame)
    plot_stage_accuracy(REPORTS_DIR / "stage_accuracy.png", best_frame)
    plot_stage_contribution(REPORTS_DIR / "incremental_contribution.png", contribution_frame)


def plot_stage_accuracy(path: Path, best_frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ordered = best_frame.copy()
    ordered["stage_order"] = ordered["stage"].str.replace("stage_", "", regex=False).astype(int)
    ordered = ordered.sort_values("stage_order")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(ordered["stage"], ordered["accuracy"], marker="o")
    axis.set_xlabel("Stage")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Stage Accuracy")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_stage_contribution(path: Path, contribution_frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ordered = contribution_frame.copy()
    ordered["stage_order"] = ordered["stage"].str.replace("stage_", "", regex=False).astype(int)
    ordered = ordered.sort_values("stage_order")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(ordered["stage"], ordered["incremental_contribution"])
    axis.set_xlabel("Stage")
    axis.set_ylabel("Incremental Contribution")
    axis.set_title("Incremental Accuracy Contribution")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
