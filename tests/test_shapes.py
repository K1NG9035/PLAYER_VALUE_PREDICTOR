import numpy as np
import pandas as pd
import torch

from fifa_value.features.sequences import build_sequence_arrays
from fifa_value.models.conformal import ConformalQuantileCalibrator
from fifa_value.models.seq_nn import SequencePlayerModel


def test_sequence_arrays_and_model_shapes():
    # Synthetic panel
    panel_rows = []
    for pid in [1, 2, 3]:
        for s in [2017, 2018, 2019, 2020]:
            panel_rows.append({
                "sofifa_id": pid,
                "season": s,
                "overall": 75,
                "potential": 80,
                "log_value": 14.0,
                "log_wage": 10.0,
                "age": 24,
                "pace": 70,
                "shooting": 65,
                "passing": 70,
                "dribbling": 72,
                "defending": 50,
                "physic": 68,
                "potential_gap": 5,
                "tenure_years": 2,
                "contract_years_remaining": 3,
                "is_clubless": 0,
                "is_loaned": 0,
                "is_gk": 0,
                "work_rate_att": 2,
                "work_rate_def": 2,
                "n_tags": 1,
                "n_traits": 2,
                "overall_season_pct": 0.5,
                "log_value_season_pct": 0.5,
                "target_a": 14.2,
                "target_b": 0.2,
            })
    panel_df = pd.DataFrame(panel_rows)
    pairs_df = panel_df[panel_df["season"] == 2019].copy()

    L = 4
    x_seq, mask, x_static, y_a, _y_b = build_sequence_arrays(pairs_df, panel_df, history_length=L)

    assert x_seq.shape == (3, L, 11)
    assert mask.shape == (3, L)
    assert len(x_static) == 3
    assert len(y_a) == 3

    # Pass through neural net
    model = SequencePlayerModel(d_seq=11, d_static=x_static.shape[1])
    out = model(torch.tensor(x_seq), torch.tensor(mask), torch.tensor(x_static))

    for key in ["pred_a", "pred_b", "q05", "q50", "q95"]:
        assert key in out
        assert out[key].shape == (3,)


def test_conformal_calibrator_ordering():
    y_true_log = np.array([12.0, 13.0, 14.0, 15.0, 16.0])
    q_low = y_true_log - 0.5
    q_high = y_true_log + 0.5

    calibrator = ConformalQuantileCalibrator(target_coverage=0.90)
    calibrator.fit(y_true_log, q_low, q_high)

    test_q_low = np.array([13.0, 14.0])
    test_q_high = np.array([14.0, 15.0])
    low_eur, high_eur = calibrator.predict_intervals_eur(test_q_low, test_q_high)

    assert len(low_eur) == 2
    assert len(high_eur) == 2
    assert np.all(low_eur <= high_eur)
