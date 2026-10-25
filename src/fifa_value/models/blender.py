import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class StackingBlender:
    """
    Non-negative linear stacking blender fit on time-respecting out-of-fold predictions.
    Enforces positive weights to prevent over-fitting and cancellation artifacts.
    """

    def __init__(self, normalize_weights: bool = True):
        self.normalize_weights = normalize_weights
        self.model = LinearRegression(positive=True, fit_intercept=True)
        self.weights_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.feature_names_: list[str] = []

    def fit(self, oof_preds_matrix: np.ndarray, y_true: np.ndarray, feature_names: list[str] | None = None) -> "StackingBlender":
        self.feature_names_ = feature_names or [f"model_{i}" for i in range(oof_preds_matrix.shape[1])]
        self.model.fit(oof_preds_matrix, y_true)
        self.weights_ = self.model.coef_
        self.intercept_ = float(self.model.intercept_)

        if self.normalize_weights and np.sum(self.weights_) > 0:
            self.weights_ = self.weights_ / np.sum(self.weights_)
            self.intercept_ = 0.0  # pure convex combination
        return self

    def predict(self, preds_matrix: np.ndarray) -> np.ndarray:
        if self.normalize_weights and self.weights_ is not None:
            return np.dot(preds_matrix, self.weights_) + self.intercept_
        return self.model.predict(preds_matrix)

    def get_weights_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "model": self.feature_names_,
            "weight": self.weights_,
        }).sort_values("weight", ascending=False).reset_index(drop=True)
