from __future__ import annotations

import logging

import pandas as pd

from src.utils.io import read_csv, write_csv, write_pickle
from src.utils.paths import BEST_MODELS_PATH, MODEL_METRICS_PATH, STAGE_LABELS, ensure_directories, stage_dataset_path, stage_model_dir

from .model_utils import build_models, score_predictions, split_features_and_target


def train_stage_models() -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    ensure_directories()
    metrics_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for stage_label in STAGE_LABELS:
        logger.info("Training started for %s", stage_label)
        stage_frame = read_csv(stage_dataset_path(stage_label))
        x_train, y_train, x_test, y_test = split_features_and_target(stage_frame)
        best_accuracy = -1.0
        best_model_name = ""
        best_model = None
        best_predictions = None

        for model_name, model in build_models().items():
            logger.info(
                "Fitting %s on %s | train_rows=%s test_rows=%s",
                model_name,
                stage_label,
                len(x_train),
                len(x_test),
            )
            fitted_model = model.fit(x_train, y_train)
            predictions = fitted_model.predict(x_test)
            scores = score_predictions(y_test, predictions)
            logger.info(
                "Finished %s on %s | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                model_name,
                stage_label,
                scores["accuracy"],
                scores["precision"],
                scores["recall"],
                scores["f1"],
            )
            metrics_rows.append(
                {
                    "stage": stage_label,
                    "model_name": model_name,
                    "accuracy": scores["accuracy"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                }
            )

            if scores["accuracy"] > best_accuracy:
                best_accuracy = scores["accuracy"]
                best_model_name = model_name
                best_model = fitted_model
                best_predictions = pd.DataFrame(
                    {
                        "match_id": stage_frame.loc[stage_frame["split"] == "test", "match_id"].values,
                        "stage": stage_label,
                        "blue_win": y_test.values,
                        "predicted_blue_win": predictions,
                    }
                )

        stage_dir = stage_model_dir(stage_label)
        stage_dir.mkdir(parents=True, exist_ok=True)
        write_pickle(stage_dir / "best_model.pkl", best_model)
        if best_predictions is not None:
            write_csv(stage_dir / "test_predictions.csv", best_predictions)

        best_metrics = next(
            row for row in metrics_rows if row["stage"] == stage_label and row["model_name"] == best_model_name
        )
        best_rows.append(
            {
                "stage": stage_label,
                "best_model": best_model_name,
                "accuracy": best_metrics["accuracy"],
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "f1": best_metrics["f1"],
            }
        )
        logger.info("Training finished for %s | best_model=%s accuracy=%.4f", stage_label, best_model_name, best_accuracy)

    metrics_frame = pd.DataFrame(metrics_rows)
    best_frame = pd.DataFrame(best_rows)
    write_csv(MODEL_METRICS_PATH, metrics_frame)
    write_csv(BEST_MODELS_PATH, best_frame)
    return metrics_frame
