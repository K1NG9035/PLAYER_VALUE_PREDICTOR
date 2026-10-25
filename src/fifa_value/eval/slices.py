import numpy as np
import pandas as pd

from fifa_value.eval.metrics import compute_interval_metrics, compute_regression_metrics

TOP_LEAGUES = {
    "English Premier League",
    "Spain Primera Division",
    "Italian Serie A",
    "German 1. Bundesliga",
    "French Ligue 1",
}


def add_eval_slices(df: pd.DataFrame) -> pd.DataFrame:
    """Add segmentation columns for metric slicing."""
    df = df.copy()

    # Value tier
    val = df["value_eur"]
    conditions = [
        val <= 500000,
        (val > 500000) & (val <= 2000000),
        (val > 2000000) & (val <= 10000000),
        val > 10000000,
    ]
    choices = ["<=500k", "500k-2M", "2M-10M", ">10M"]
    df["value_tier"] = np.select(conditions, choices, default="<=500k")

    # Age band
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 21, 25, 29, 100],
        labels=["<=21", "22-25", "26-29", "30+"],
    )

    # History length
    if "n_seasons_observed" in df.columns:
        df["history_tier"] = df["n_seasons_observed"].apply(
            lambda x: "1 season" if x == 1 else ("2 seasons" if x == 2 else "3+ seasons")
        )
    else:
        df["history_tier"] = "Unknown"

    # League tier
    if "league_name" in df.columns:
        df["league_tier"] = df["league_name"].apply(
            lambda l: "Big 5 Leagues" if l in TOP_LEAGUES else "Other Leagues"
        )
    else:
        df["league_tier"] = "Other Leagues"

    return df


def evaluate_sliced_metrics(
    eval_df: pd.DataFrame,
    pred_col: str = "pred_value_eur",
    target_col: str = "value_next",
    lower_col: str | None = "lower_90_eur",
    upper_col: str | None = "upper_90_eur",
) -> dict[str, pd.DataFrame]:
    """
    Computes regression and interval metrics grouped by each slice.
    Returns dictionary mapping slice dimension to summary DataFrame.
    """
    df = add_eval_slices(eval_df)
    slice_dimensions = ["value_tier", "age_band", "position_group", "history_tier", "league_tier"]

    results = {}
    for dim in slice_dimensions:
        if dim not in df.columns:
            continue
        rows = []
        for segment, group in df.groupby(dim, observed=True):
            y_t = group[target_col].values
            y_p = group[pred_col].values
            reg_m = compute_regression_metrics(y_t, y_p)
            row = {
                "segment": str(segment),
                "count": len(group),
                "rmsle": reg_m["rmsle"],
                "mae_eur": reg_m["mae"],
                "median_ae_eur": reg_m["median_ae"],
                "within_25_pct": reg_m["within_25_pct"],
                "spearman_rho": reg_m["spearman_rho"],
            }
            if lower_col and upper_col and lower_col in group.columns and upper_col in group.columns:
                int_m = compute_interval_metrics(
                    y_t, group[lower_col].values, group[upper_col].values
                )
                row["coverage_90"] = int_m["coverage"] * 100.0
                row["mean_width_log"] = int_m["mean_width_log"]
                row["winkler_score_log"] = int_m["winkler_score_log"]
            rows.append(row)
        results[dim] = pd.DataFrame(rows)

    return results
