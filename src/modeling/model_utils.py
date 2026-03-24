from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing.engineer_features import FEATURE_COLUMNS


def build_models() -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def split_features_and_target(stage_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_frame = stage_frame.loc[stage_frame["split"] == "train"].copy()
    test_frame = stage_frame.loc[stage_frame["split"] == "test"].copy()
    x_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["blue_win"].astype(int)
    x_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["blue_win"].astype(int)
    return x_train, y_train, x_test, y_test


def score_predictions(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
