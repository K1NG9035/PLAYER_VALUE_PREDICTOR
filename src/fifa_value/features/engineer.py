import numpy as np
import pandas as pd

LAG_VARS = [
    "overall",
    "potential",
    "log_value",
    "log_wage",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag-1, lag-2, lag-3, and deltas by strictly joining historical observations
    at season - 1, season - 2, season - 3 for the same sofifa_id.
    Missing previous seasons result in NaN.
    """
    df = df.copy()

    # Pre-select columns for lag joins
    cols_to_lag = [c for c in LAG_VARS if c in df.columns]

    for lag in [1, 2, 3]:
        lag_df = df[["sofifa_id", "season"] + cols_to_lag].copy()
        lag_df["season"] = lag_df["season"] + lag  # so merging on 'season' brings t-lag to t
        rename_dict = {c: f"{c}_lag{lag}" for c in cols_to_lag}
        lag_df.rename(columns=rename_dict, inplace=True)

        df = df.merge(lag_df, on=["sofifa_id", "season"], how="left")

    # Compute deltas
    for c in cols_to_lag:
        if f"{c}_lag1" in df.columns:
            df[f"{c}_delta1"] = df[c] - df[f"{c}_lag1"]
        if f"{c}_lag2" in df.columns:
            df[f"{c}_delta2"] = df[c] - df[f"{c}_lag2"]

    return df


def add_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative career metrics per player strictly up to season t:
    - n_seasons_observed
    - is_first_observation
    - max_overall_so_far
    - seasons_since_peak
    - rolling mean (window 3)
    - OLS slope over observed history
    """
    df = df.sort_values(["sofifa_id", "season"]).reset_index(drop=True)

    grp = df.groupby("sofifa_id")

    # Cumulative observations
    cum_count = grp.cumcount() + 1
    df["n_seasons_observed"] = cum_count
    df["is_first_observation"] = (cum_count == 1).astype(int)

    # Peak analysis
    cummax_overall = grp["overall"].cummax()
    df["max_overall_so_far"] = cummax_overall
    df["peak_season_flag"] = np.where(df["overall"] == cummax_overall, df["season"], np.nan)
    df["last_peak_season"] = grp["peak_season_flag"].ffill()
    df["seasons_since_peak"] = (df["season"] - df["last_peak_season"]).fillna(0.0)
    df.drop(columns=["peak_season_flag", "last_peak_season"], inplace=True)

    # Rolling means (past 3 observed seasons)
    df["overall_rollmean_3"] = grp["overall"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["log_value_rollmean_3"] = grp["log_value"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    # Vectorized cumulative OLS slope for overall and log_value
    x = df["season"] - 2015.0
    x2 = x ** 2
    df["_x"] = x
    df["_x2"] = x2
    df["_xy_overall"] = x * df["overall"]
    df["_xy_value"] = x * df["log_value"]

    g = df.groupby("sofifa_id")
    sum_x = g["_x"].cumsum()
    sum_x2 = g["_x2"].cumsum()
    sum_y_ovr = g["overall"].cumsum()
    sum_xy_ovr = g["_xy_overall"].cumsum()
    sum_y_val = g["log_value"].cumsum()
    sum_xy_val = g["_xy_value"].cumsum()

    denom = cum_count * sum_x2 - sum_x ** 2

    # Slope is defined when at least 2 distinct seasons observed
    numer_ovr = cum_count * sum_xy_ovr - sum_x * sum_y_ovr
    numer_val = cum_count * sum_xy_val - sum_x * sum_y_val

    df["overall_slope"] = np.where(denom > 0, numer_ovr / denom, np.nan)
    df["log_value_slope"] = np.where(denom > 0, numer_val / denom, np.nan)

    df.drop(columns=["_x", "_x2", "_xy_overall", "_xy_value"], inplace=True)
    return df


def add_drift_normalization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-season percentiles and context aggregates to insulate models from
    EA valuation scale drift across editions.
    """
    df = df.copy()

    # Per-season percentile ranks
    rank_cols = ["overall", "potential", "log_value", "log_wage", "international_reputation"]
    for c in rank_cols:
        if c in df.columns:
            df[f"{c}_season_pct"] = df.groupby("season")[c].rank(pct=True)

    # Per-season x position-group aggregates
    if "position_group" in df.columns:
        pos_val_median = df.groupby(["season", "position_group"])["log_value"].transform("median")
        pos_ovr_median = df.groupby(["season", "position_group"])["overall"].transform("median")
        df["season_pos_median_value"] = pos_val_median
        df["rel_value_to_season_pos"] = df["log_value"] - pos_val_median
        df["rel_overall_to_season_pos"] = df["overall"] - pos_ovr_median

    # Per-season x league aggregates
    if "league_name" in df.columns:
        league_val_median = df.groupby(["season", "league_name"])["log_value"].transform("median")
        df["season_league_median_value"] = league_val_median
        df["rel_value_to_season_league"] = df["log_value"] - league_val_median

    # Deflated log value (relative to overall season median log_value)
    season_val_median = df.groupby("season")["log_value"].transform("median")
    df["season_median_log_value"] = season_val_median
    df["deflated_log_value"] = df["log_value"] - season_val_median

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Execute complete feature engineering pipeline on cleaned panel data."""
    df = add_lag_features(df)
    df = add_trajectory_features(df)
    df = add_drift_normalization_features(df)
    return df
