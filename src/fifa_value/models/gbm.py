from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from fifa_value.features.encoders import FrequencyEncoder

GBM_CAT_COLS = [
    "primary_position",
    "position_group",
    "preferred_foot",
    "body_type_clean",
    "club_name",
    "league_name",
    "nationality",
]

GBM_NUM_EXCLUDE = [
    "sofifa_id",
    "season",
    "short_name",
    "transition",
    "target_a",
    "target_b",
    "value_next",
    "wage_next",
    "is_zero_pair",
]


class LightGBMBranch:
    """LightGBM tabular model with support for Target A, Target B, and Quantile regression."""

    def __init__(
        self,
        target_name: str = "target_a",
        params: dict[str, Any] | None = None,
    ):
        self.target_name = target_name
        self.params = params or {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 1000,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model: lgb.LGBMRegressor | None = None
        self.quantile_models: dict[float, lgb.LGBMRegressor] = {}
        self.encoder = FrequencyEncoder(cat_cols=GBM_CAT_COLS)
        self.feature_cols: list[str] = []

    def _prepare_features(self, df: pd.DataFrame, fit_encoder: bool = False) -> pd.DataFrame:
        if fit_encoder:
            df_enc = self.encoder.fit_transform(df)
        else:
            df_enc = self.encoder.transform(df)

        feat_cols = [
            c for c in df_enc.columns
            if c not in GBM_NUM_EXCLUDE
            and c not in GBM_CAT_COLS
            and not c.endswith("_year")
            and df_enc[c].dtype in [np.float32, np.float64, np.int32, np.int64, int, float]
        ]
        if fit_encoder:
            self.feature_cols = feat_cols
        return df_enc[self.feature_cols]

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        fit_quantiles: bool = False,
    ) -> "LightGBMBranch":
        X_train = self._prepare_features(train_df, fit_encoder=True)
        y_train = train_df[self.target_name].values

        eval_set = None
        if val_df is not None:
            X_val = self._prepare_features(val_df, fit_encoder=False)
            y_val = val_df[self.target_name].values
            eval_set = [(X_val, y_val)]

        self.model = lgb.LGBMRegressor(**self.params)
        callbacks = [lgb.early_stopping(stopping_rounds=40, verbose=False)] if eval_set else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        if fit_quantiles:
            for alpha in [0.05, 0.50, 0.95]:
                q_params = self.params.copy()
                q_params["objective"] = "quantile"
                q_params["alpha"] = alpha
                q_params["metric"] = "quantile"
                q_model = lgb.LGBMRegressor(**q_params)
                q_model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
                self.quantile_models[alpha] = q_model

        return self

    def predict_target(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(df, fit_encoder=False)
        return self.model.predict(X)

    def predict_eur(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.predict_target(df)
        if self.target_name == "target_a":
            return np.expm1(np.maximum(preds, 0.0))
        else:
            log_next = df["log_value"].values + preds
            return np.expm1(np.maximum(log_next, 0.0))

    def predict_quantiles_eur(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (q05, q50, q95) predictions converted to euro space."""
        X = self._prepare_features(df, fit_encoder=False)
        q05 = self.quantile_models[0.05].predict(X)
        q50 = self.quantile_models[0.50].predict(X)
        q95 = self.quantile_models[0.95].predict(X)

        if self.target_name == "target_a":
            return (
                np.expm1(np.maximum(q05, 0.0)),
                np.expm1(np.maximum(q50, 0.0)),
                np.expm1(np.maximum(q95, 0.0)),
            )
        else:
            log_v = df["log_value"].values
            return (
                np.expm1(np.maximum(log_v + q05, 0.0)),
                np.expm1(np.maximum(log_v + q50, 0.0)),
                np.expm1(np.maximum(log_v + q95, 0.0)),
            )

    def get_feature_importances(self) -> pd.DataFrame:
        gains = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": gains,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


class CatBoostBranch:
    """CatBoost tabular model with native categorical features."""

    def __init__(
        self,
        target_name: str = "target_a",
        params: dict[str, Any] | None = None,
    ):
        self.target_name = target_name
        self.params = params or {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "early_stopping_rounds": 40,
            "verbose": 0,
            "thread_count": -1,
        }
        self.model: CatBoostRegressor | None = None
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.feature_cols: list[str] = []

    def _prepare_pool(self, df: pd.DataFrame, is_train: bool = False) -> Pool:
        df_copy = df.copy()

        if is_train:
            self.cat_cols = [c for c in GBM_CAT_COLS if c in df_copy.columns]
            self.num_cols = [
                c for c in df_copy.columns
                if c not in GBM_NUM_EXCLUDE
                and c not in self.cat_cols
                and df_copy[c].dtype in [np.float32, np.float64, np.int32, np.int64, int, float]
            ]
            self.feature_cols = self.cat_cols + self.num_cols

        for c in self.cat_cols:
            df_copy[c] = df_copy[c].fillna("Unknown").astype(str)

        X = df_copy[self.feature_cols]
        y = df_copy[self.target_name].values if self.target_name in df_copy.columns else None

        return Pool(data=X, label=y, cat_features=self.cat_cols)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "CatBoostBranch":
        train_pool = self._prepare_pool(train_df, is_train=True)
        val_pool = self._prepare_pool(val_df, is_train=False) if val_df is not None else None

        self.model = CatBoostRegressor(**self.params)
        self.model.fit(train_pool, eval_set=val_pool, use_best_model=bool(val_pool))
        return self

    def predict_target(self, df: pd.DataFrame) -> np.ndarray:
        pool = self._prepare_pool(df, is_train=False)
        return self.model.predict(pool)

    def predict_eur(self, df: pd.DataFrame) -> np.ndarray:
        preds = self.predict_target(df)
        if self.target_name == "target_a":
            return np.expm1(np.maximum(preds, 0.0))
        else:
            log_next = df["log_value"].values + preds
            return np.expm1(np.maximum(log_next, 0.0))
