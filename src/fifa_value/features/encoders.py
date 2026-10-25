
import pandas as pd

CAT_COLS = [
    "primary_position",
    "position_group",
    "preferred_foot",
    "body_type_clean",
    "club_name",
    "league_name",
    "nationality",
]


class FrequencyEncoder:
    """
    Fits frequency encoding on training data only.
    Maps rare or unseen categories in validation/test to 0.
    """

    def __init__(self, cat_cols: list[str] | None = None, min_freq: int = 5):
        self.cat_cols = cat_cols or CAT_COLS
        self.min_freq = min_freq
        self.freq_maps_: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame) -> "FrequencyEncoder":
        self.freq_maps_ = {}
        n = len(df)
        for col in self.cat_cols:
            if col in df.columns:
                counts = df[col].astype(str).value_counts()
                valid_counts = counts[counts >= self.min_freq]
                freqs = (valid_counts / n).to_dict()
                self.freq_maps_[col] = freqs
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.cat_cols:
            if col in self.freq_maps_:
                freq_map = self.freq_maps_[col]
                df_out[f"{col}_freq"] = (
                    df_out[col].astype(str).map(freq_map).fillna(0.0).astype(float)
                )
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class OrdinalCategoricalEncoder:
    """
    Maps categorical columns into integer indices for embedding layers or LightGBM categoricals.
    Index 0 is always reserved for UNKNOWN / PADDING.
    """

    def __init__(self, cat_cols: list[str] | None = None, max_categories: int = 200):
        self.cat_cols = cat_cols or CAT_COLS
        self.max_categories = max_categories
        self.category_maps_: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> "OrdinalCategoricalEncoder":
        self.category_maps_ = {}
        for col in self.cat_cols:
            if col in df.columns:
                top_cats = (
                    df[col]
                    .astype(str)
                    .value_counts()
                    .head(self.max_categories - 1)
                    .index.tolist()
                )
                cat_map = {cat: idx + 1 for idx, cat in enumerate(top_cats)}
                self.category_maps_[col] = cat_map
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col, cat_map in self.category_maps_.items():
            if col in df_out.columns:
                df_out[f"{col}_idx"] = (
                    df_out[col].astype(str).map(cat_map).fillna(0).astype(int)
                )
        return df_out
