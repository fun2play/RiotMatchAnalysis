from __future__ import annotations

import pandas as pd


def make_stage_contribution_table(best_models: pd.DataFrame) -> pd.DataFrame:
    ordered = best_models.copy()
    ordered["stage_order"] = ordered["stage"].str.replace("stage_", "", regex=False).astype(int)
    ordered = ordered.sort_values("stage_order").reset_index(drop=True)
    ordered["incremental_contribution"] = ordered["accuracy"].diff().fillna(ordered["accuracy"])
    total = float(ordered["incremental_contribution"].sum())
    if total > 0:
        ordered["normalized_share"] = ordered["incremental_contribution"] / total
    else:
        ordered["normalized_share"] = 0.0
    return ordered[["stage", "best_model", "accuracy", "incremental_contribution", "normalized_share"]]
