from collections.abc import Generator

import pandas as pd

TEMPORAL_FOLDS = [
    {
        "fold_idx": 1,
        "name": "fold_1",
        "train_transitions": ["2015->2016", "2016->2017"],
        "val_transition": "2017->2018",
    },
    {
        "fold_idx": 2,
        "name": "fold_2",
        "train_transitions": ["2015->2016", "2016->2017", "2017->2018"],
        "val_transition": "2018->2019",
    },
    {
        "fold_idx": 3,
        "name": "fold_3",
        "train_transitions": ["2015->2016", "2016->2017", "2017->2018", "2018->2019"],
        "val_transition": "2019->2020",
    },
]

FINAL_TEST_SPLIT = {
    "name": "final_test",
    "train_transitions": ["2015->2016", "2016->2017", "2017->2018", "2018->2019", "2019->2020"],
    "calib_transition": "2019->2020",
    "test_transition": "2020->2021",
}


def get_temporal_cv_splits(
    pairs_df: pd.DataFrame,
) -> Generator[tuple[dict[str, any], pd.DataFrame, pd.DataFrame], None, None]:
    """
    Yields (fold_meta, train_df, val_df) for expanding-window temporal CV.
    """
    for fold in TEMPORAL_FOLDS:
        train_mask = pairs_df["transition"].isin(fold["train_transitions"])
        val_mask = pairs_df["transition"] == fold["val_transition"]

        train_df = pairs_df[train_mask].copy().reset_index(drop=True)
        val_df = pairs_df[val_mask].copy().reset_index(drop=True)

        yield fold, train_df, val_df


def get_final_test_split(
    pairs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, calib_df, test_df) for final held-out test evaluation.
    - train_df: 2015->2016 through 2019->2020
    - calib_df: 2019->2020 (used for CQR conformal calibration)
    - test_df: 2020->2021 (held out test set touched once)
    """
    train_mask = pairs_df["transition"].isin(FINAL_TEST_SPLIT["train_transitions"])
    calib_mask = pairs_df["transition"] == FINAL_TEST_SPLIT["calib_transition"]
    test_mask = pairs_df["transition"] == FINAL_TEST_SPLIT["test_transition"]

    train_df = pairs_df[train_mask].copy().reset_index(drop=True)
    calib_df = pairs_df[calib_mask].copy().reset_index(drop=True)
    test_df = pairs_df[test_mask].copy().reset_index(drop=True)

    return train_df, calib_df, test_df
