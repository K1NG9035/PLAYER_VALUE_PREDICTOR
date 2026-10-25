import pandas as pd
import pytest

from fifa_value.data.clean import clean_panel_data
from fifa_value.data.pairs import build_pairs
from fifa_value.features.engineer import engineer_features


def test_no_future_leakage_in_features():
    """
    Non-negotiable rule:
    Features for (player p, season t) must ONLY depend on seasons <= t.
    Perturbing season t+1 attributes must produce zero change in season t features.
    """
    # Create baseline synthetic history
    data_base = []
    for s in [2015, 2016, 2017, 2018]:
        data_base.append({
            "sofifa_id": 99999,
            "short_name": "Test Player",
            "season": s,
            "overall": 75 + (s - 2015),
            "potential": 85,
            "value_eur": 5000000.0 + (s - 2015) * 1000000.0,
            "wage_eur": 20000.0,
            "player_positions": "ST, CF",
            "club_name": "FC Test",
            "league_name": "Test League",
            "work_rate": "High/Medium",
            "body_type": "Normal",
            "real_face": "No",
            "contract_valid_until": "2022",
            "joined": "2014-07-01",
            "pace": 80.0,
            "shooting": 75.0,
            "passing": 70.0,
            "dribbling": 78.0,
            "defending": 35.0,
            "physic": 72.0,
            "international_reputation": 2,
        })
    df_base = pd.DataFrame(data_base)

    # Clean and engineer baseline
    clean_base = clean_panel_data(df_base)
    feat_base = engineer_features(clean_base)

    # Create perturbed version where season 2018 (t+1 for t=2017) is wildly altered
    data_perturbed = [row.copy() for row in data_base]
    # Alter 2018 values
    data_perturbed[3]["overall"] = 99
    data_perturbed[3]["value_eur"] = 150000000.0
    data_perturbed[3]["pace"] = 99.0
    data_perturbed[3]["shooting"] = 99.0

    df_perturbed = pd.DataFrame(data_perturbed)
    clean_perturbed = clean_panel_data(df_perturbed)
    feat_perturbed = engineer_features(clean_perturbed)

    # Compare features at season 2017 (t=2017)
    row_base_2017 = feat_base[feat_base["season"] == 2017].iloc[0]
    row_pert_2017 = feat_perturbed[feat_perturbed["season"] == 2017].iloc[0]

    # Exclude non-feature columns
    skip_cols = {"season", "sofifa_id"}
    test_cols = [c for c in feat_base.columns if c not in skip_cols and not c.startswith("target_")]

    for col in test_cols:
        v_base = row_base_2017[col]
        v_pert = row_pert_2017[col]
        if pd.isna(v_base) and pd.isna(v_pert):
            continue
        assert v_base == pytest.approx(v_pert), f"Leakage detected in feature '{col}': {v_base} vs {v_pert}"


def test_pairs_builder_structure():
    """Verify labeled pairs are generated correctly and 2021 has no label."""
    data = []
    for s in [2018, 2019, 2020, 2021]:
        data.append({
            "sofifa_id": 101,
            "short_name": "P1",
            "season": s,
            "overall": 80,
            "potential": 85,
            "value_eur": 10000000.0,
            "wage_eur": 50000.0,
            "player_positions": "CM",
            "club_name": "Club A",
            "league_name": "League A",
            "work_rate": "Medium/Medium",
            "body_type": "Normal",
            "real_face": "No",
            "contract_valid_until": "2023",
            "joined": "2017-01-01",
            "pace": 70.0,
            "shooting": 70.0,
            "passing": 80.0,
            "dribbling": 75.0,
            "defending": 65.0,
            "physic": 70.0,
            "international_reputation": 2,
        })
    df = pd.DataFrame(data)
    clean = clean_panel_data(df)
    feat = engineer_features(clean)
    labeled, inference = build_pairs(feat)

    # 2018->2019, 2019->2020, 2020->2021 are labeled (3 transitions)
    assert len(labeled) == 3
    assert set(labeled["transition"]) == {"2018->2019", "2019->2020", "2020->2021"}
    assert all(labeled["target_a"] > 0)

    # 2021 is in inference_df
    assert len(inference) == 1
    assert inference.iloc[0]["season"] == 2021
