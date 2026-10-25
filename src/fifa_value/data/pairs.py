
import numpy as np
import pandas as pd


def build_pairs(
    panel_df: pd.DataFrame,
    filter_zero_values: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct labeled (t -> t+1) pairs from panel data where features for season t
    contain strictly information from seasons <= t.

    Returns:
    - labeled_pairs: DataFrame with valid transitions (2015->2016 through 2020->2021)
    - inference_2022: DataFrame for season 2021 predicting 2022
    """
    df = panel_df.copy()

    # Extract target values from season t+1
    target_source = df[["sofifa_id", "season", "value_eur", "wage_eur"]].copy()
    target_source["season"] = target_source["season"] - 1  # merge on season t aligns with t+1
    target_source.rename(
        columns={
            "value_eur": "value_next",
            "wage_eur": "wage_next",
        },
        inplace=True,
    )

    pairs = df.merge(target_source, on=["sofifa_id", "season"], how="left")

    # Define transition string
    pairs["transition"] = pairs["season"].apply(lambda s: f"{s}->{s+1}")

    # Separate labeled transitions from final inference transition (2021 -> 2022)
    inference_mask = pairs["season"] == 2021
    inference_df = pairs[inference_mask].copy().reset_index(drop=True)

    labeled_df = pairs[~inference_mask & pairs["value_next"].notna()].copy()

    # Compute targets
    labeled_df["target_a"] = np.log1p(np.maximum(labeled_df["value_next"], 0.0))
    labeled_df["target_b"] = labeled_df["target_a"] - labeled_df["log_value"]

    # Flag zero values
    labeled_df["is_zero_pair"] = (
        (labeled_df["value_eur"] == 0) | (labeled_df["value_next"] == 0)
    ).astype(int)

    if filter_zero_values:
        labeled_df = labeled_df[labeled_df["is_zero_pair"] == 0].reset_index(drop=True)
    else:
        labeled_df = labeled_df.reset_index(drop=True)

    return labeled_df, inference_df
