import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEQ_NUMERIC_COLS = [
    "overall",
    "potential",
    "log_value",
    "log_wage",
    "age",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]

STATIC_NUMERIC_COLS = [
    "age",
    "overall",
    "potential_gap",
    "tenure_years",
    "contract_years_remaining",
    "is_clubless",
    "is_loaned",
    "is_gk",
    "work_rate_att",
    "work_rate_def",
    "n_tags",
    "n_traits",
    "overall_season_pct",
    "log_value_season_pct",
]


class PlayerSequenceDataset(Dataset):
    """
    PyTorch Dataset providing multi-season historical sequences:
    - x_seq: (L, D_seq) tensor
    - mask: (L,) boolean/float tensor (1 if season was observed, 0 if padded)
    - x_static: (D_static,) tensor of current season t attributes
    - y_a, y_b: targets (or NaNs for inference)
    """

    def __init__(
        self,
        x_seq: np.ndarray,
        mask: np.ndarray,
        x_static: np.ndarray,
        y_a: np.ndarray | None = None,
        y_b: np.ndarray | None = None,
    ):
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.x_static = torch.tensor(x_static, dtype=torch.float32)
        self.y_a = torch.tensor(y_a, dtype=torch.float32) if y_a is not None else None
        self.y_b = torch.tensor(y_b, dtype=torch.float32) if y_b is not None else None

    def __len__(self) -> int:
        return len(self.x_seq)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "x_seq": self.x_seq[idx],
            "mask": self.mask[idx],
            "x_static": self.x_static[idx],
        }
        if self.y_a is not None:
            item["y_a"] = self.y_a[idx]
        if self.y_b is not None:
            item["y_b"] = self.y_b[idx]
        return item


def build_sequence_arrays(
    pairs_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    history_length: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct numpy arrays for sequence modeling:
    For each pair at season t, extracts historical observations from seasons [t - L + 1, ..., t].
    Returns (x_seq, mask, x_static, y_a, y_b).
    """
    # Create lookup map for fast player-season attribute retrieval
    panel_lookup = panel_df.set_index(["sofifa_id", "season"])
    available_cols = [c for c in SEQ_NUMERIC_COLS if c in panel_lookup.columns]
    d_seq = len(available_cols)

    n_samples = len(pairs_df)
    x_seq = np.zeros((n_samples, history_length, d_seq), dtype=np.float32)
    mask = np.zeros((n_samples, history_length), dtype=np.float32)

    sofifa_ids = pairs_df["sofifa_id"].values
    t_seasons = pairs_df["season"].values

    # Pre-extract normalized matrix for lookup
    lookup_sub = panel_lookup[available_cols].fillna(0.0).copy()
    # Normalize features roughly to [0, 1] range for neural net stability
    lookup_sub["overall"] = lookup_sub["overall"] / 100.0
    lookup_sub["potential"] = lookup_sub["potential"] / 100.0
    lookup_sub["log_value"] = lookup_sub["log_value"] / 20.0
    lookup_sub["log_wage"] = lookup_sub["log_wage"] / 15.0
    lookup_sub["age"] = lookup_sub["age"] / 50.0
    for c in ["pace", "shooting", "passing", "dribbling", "defending", "physic"]:
        if c in lookup_sub.columns:
            lookup_sub[c] = lookup_sub[c] / 100.0

    lookup_dict = lookup_sub.to_dict(orient="index")

    for i in range(n_samples):
        pid = sofifa_ids[i]
        curr_t = t_seasons[i]
        for step_idx in range(history_length):
            hist_season = curr_t - (history_length - 1 - step_idx)
            key = (pid, hist_season)
            if key in lookup_dict:
                mask[i, step_idx] = 1.0
                row_vals = [lookup_dict[key][c] for c in available_cols]
                x_seq[i, step_idx] = row_vals

    # Static features from pairs_df at current season t
    static_cols = [c for c in STATIC_NUMERIC_COLS if c in pairs_df.columns]
    static_df = pairs_df[static_cols].fillna(0.0).copy()
    static_df["age"] = static_df["age"] / 50.0
    static_df["overall"] = static_df["overall"] / 100.0
    if "potential_gap" in static_df.columns:
        static_df["potential_gap"] = static_df["potential_gap"] / 20.0
    if "tenure_years" in static_df.columns:
        static_df["tenure_years"] = static_df["tenure_years"] / 15.0
    if "contract_years_remaining" in static_df.columns:
        static_df["contract_years_remaining"] = static_df["contract_years_remaining"] / 10.0

    x_static = static_df.values.astype(np.float32)

    y_a = pairs_df["target_a"].values.astype(np.float32) if "target_a" in pairs_df.columns else np.zeros(n_samples, dtype=np.float32)
    y_b = pairs_df["target_b"].values.astype(np.float32) if "target_b" in pairs_df.columns else np.zeros(n_samples, dtype=np.float32)

    return x_seq, mask, x_static, y_a, y_b
