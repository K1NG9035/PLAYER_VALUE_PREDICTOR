
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class PersistenceBaseline:
    """Baseline 1: Predicts value_{t+1} = value_t."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PersistenceBaseline":
        return self

    def predict_eur(self, df: pd.DataFrame) -> np.ndarray:
        return df["value_eur"].values.astype(float)


class SegmentMedianGrowthBaseline:
    """
    Baseline 2: Segment-median growth by age_band x overall_band x position_group.
    Learned strictly on training data.
    """

    def __init__(self):
        self.median_growth_table_: dict[tuple, float] = {}
        self.global_median_growth_: float = 0.0

    def _get_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        s_df = pd.DataFrame(index=df.index)
        s_df["age_band"] = pd.cut(
            df["age"],
            bins=[0, 21, 25, 29, 100],
            labels=["<=21", "22-25", "26-29", "30+"],
        ).astype(str)
        s_df["overall_band"] = pd.cut(
            df["overall"],
            bins=[0, 60, 68, 75, 82, 100],
            labels=["<60", "60-68", "68-75", "75-82", "82+"],
        ).astype(str)
        s_df["position_group"] = df["position_group"].fillna("MID").astype(str)
        return s_df

    def fit(self, df: pd.DataFrame) -> "SegmentMedianGrowthBaseline":
        growth = df["target_b"]
        self.global_median_growth_ = float(growth.median())

        s_df = self._get_segments(df)
        s_df["growth"] = growth

        grouped = s_df.groupby(["age_band", "overall_band", "position_group"])["growth"].median()
        self.median_growth_table_ = grouped.to_dict()
        return self

    def predict_eur(self, df: pd.DataFrame) -> np.ndarray:
        s_df = self._get_segments(df)
        growths = []
        for idx in range(len(s_df)):
            key = (
                s_df["age_band"].iloc[idx],
                s_df["overall_band"].iloc[idx],
                s_df["position_group"].iloc[idx],
            )
            g = self.median_growth_table_.get(key, self.global_median_growth_)
            growths.append(g)

        growths = np.array(growths)
        pred_log_next = df["log_value"].values + growths
        return np.expm1(np.maximum(pred_log_next, 0.0))


class RidgeBaseline:
    """
    Baseline 3: Ridge regression on a compact feature set in log space.
    Scalers and imputers fit on training fold only.
    """

    COMPACT_FEATURES: ClassVar[list[str]] = [
        "overall",
        "potential",
        "potential_gap",
        "age",
        "log_value",
        "log_wage",
        "contract_years_remaining",
        "tenure_years",
        "pace",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "physic",
        "gk_composite",
        "outfield_composite",
        "overall_season_pct",
        "log_value_season_pct",
    ]

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame, target: str = "target_a") -> "RidgeBaseline":
        self.feature_cols = [c for c in self.COMPACT_FEATURES if c in df.columns]
        X = df[self.feature_cols].values
        y = df[target].values

        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)
        self.model.fit(X_scaled, y)
        self.target_type = target
        return self

    def predict_eur(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        preds = self.model.predict(X_scaled)

        if self.target_type == "target_a":
            # level
            return np.expm1(np.maximum(preds, 0.0))
        else:
            # growth
            log_next = df["log_value"].values + preds
            return np.expm1(np.maximum(log_next, 0.0))
