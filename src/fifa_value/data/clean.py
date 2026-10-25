import re

import numpy as np
import pandas as pd

POSITION_GROUPS = {
    "GK": "GK",
    "CB": "DEF",
    "LCB": "DEF",
    "RCB": "DEF",
    "LB": "DEF",
    "RB": "DEF",
    "LWB": "DEF",
    "RWB": "DEF",
    "CDM": "MID",
    "LDM": "MID",
    "RDM": "MID",
    "CM": "MID",
    "LCM": "MID",
    "RCM": "MID",
    "CAM": "MID",
    "LAM": "MID",
    "RAM": "MID",
    "LM": "MID",
    "RM": "MID",
    "ST": "ATT",
    "CF": "ATT",
    "LF": "ATT",
    "RF": "ATT",
    "LW": "ATT",
    "RW": "ATT",
    "LS": "ATT",
    "RS": "ATT",
}

WORK_RATE_MAP = {
    "low": 1,
    "medium": 2,
    "med": 2,
    "high": 3,
}

POS_RATING_COLS = [
    "ls", "st", "rs", "lw", "lf", "cf", "rf", "rw",
    "lam", "cam", "ram", "lm", "lcm", "cm", "rcm", "rm",
    "lwb", "ldm", "cdm", "rdm", "rwb", "lb", "lcb", "cb", "rcb", "rb"
]


def parse_position_rating(val: str | None) -> float:
    """Parse string position rating e.g. '85+3', '72-1', or '80' to a numeric value."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if "+" in s:
        parts = s.split("+")
        try:
            return float(parts[0]) + float(parts[1])
        except (ValueError, IndexError):
            return np.nan
    elif "-" in s:
        parts = s.split("-")
        try:
            return float(parts[0]) - float(parts[1])
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_work_rate(val: str | None) -> tuple[float, float]:
    """Parse work_rate 'Attack/Defense' into ordinal integers (1=Low, 2=Medium, 3=High)."""
    if pd.isna(val):
        return np.nan, np.nan
    parts = str(val).split("/")
    if len(parts) == 2:
        att = WORK_RATE_MAP.get(parts[0].strip().lower(), 2)
        defense = WORK_RATE_MAP.get(parts[1].strip().lower(), 2)
        return float(att), float(defense)
    return np.nan, np.nan


def parse_year(val: str | None) -> float:
    """Extract a 4-digit year from date or year string."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    m = re.search(r"\b(19\d\d|20\d\d)\b", s)
    if m:
        return float(m.group(1))
    return np.nan


def clean_panel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw combined panel data for all seasons:
    - Parses string ratings, work rates, positions, contracts, dates
    - Creates goalkeeper and outfield composites
    - Builds indicators for missing values and zero values
    - Drops unusable columns (e.g. defending_marking which is 100% null in FIFA 21)
    """
    df = df.copy()

    # Drop columns that are completely missing in certain seasons or pure identifiers
    drop_cols = ["defending_marking", "player_url", "long_name"]
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Basic identity and targets
    df["sofifa_id"] = df["sofifa_id"].astype(int)
    df["season"] = df["season"].astype(int)
    df["short_name"] = df["short_name"].fillna("Unknown")

    # Financials
    df["value_eur"] = df["value_eur"].fillna(0.0).astype(float)
    df["wage_eur"] = df["wage_eur"].fillna(0.0).astype(float)
    df["log_value"] = np.log1p(np.maximum(df["value_eur"], 0.0))
    df["log_wage"] = np.log1p(np.maximum(df["wage_eur"], 0.0))
    df["is_zero_value"] = (df["value_eur"] == 0).astype(int)

    # Release clause
    if "release_clause_eur" in df.columns:
        df["has_release_clause"] = df["release_clause_eur"].notna().astype(int)
        df["release_clause_eur"] = df["release_clause_eur"].fillna(0.0).astype(float)
        df["log_release_clause"] = np.log1p(df["release_clause_eur"])
    else:
        df["has_release_clause"] = 0
        df["release_clause_eur"] = 0.0
        df["log_release_clause"] = 0.0

    # Club and loan status
    df["is_clubless"] = df["club_name"].isna().astype(int) if "club_name" in df.columns else 0
    df["club_name"] = df["club_name"].fillna("Free Agent") if "club_name" in df.columns else "Free Agent"
    df["league_name"] = df["league_name"].fillna("Unknown League") if "league_name" in df.columns else "Unknown League"
    df["is_loaned"] = df["loaned_from"].notna().astype(int) if "loaned_from" in df.columns else 0
    df["nationality"] = df["nationality"].fillna("Unknown") if "nationality" in df.columns else "Unknown"

    # Position parsing
    def extract_positions(pos_str):
        if pd.isna(pos_str):
            return "SUB", 1, "MID"
        pos_list = [p.strip() for p in str(pos_str).split(",") if p.strip()]
        if not pos_list:
            return "SUB", 1, "MID"
        prim = pos_list[0]
        n_pos = len(pos_list)
        grp = POSITION_GROUPS.get(prim, "MID")
        return prim, n_pos, grp

    if "player_positions" in df.columns:
        pos_info = [extract_positions(p) for p in df["player_positions"]]
        df["primary_position"] = [p[0] for p in pos_info]
        df["n_positions"] = [p[1] for p in pos_info]
        df["position_group"] = [p[2] for p in pos_info]
    else:
        df["primary_position"] = "SUB"
        df["n_positions"] = 1
        df["position_group"] = "MID"
    df["is_gk"] = (df["position_group"] == "GK").astype(int)

    # Work rates
    if "work_rate" in df.columns:
        work_rates = [parse_work_rate(w) for w in df["work_rate"]]
        df["work_rate_att"] = [w[0] for w in work_rates]
        df["work_rate_def"] = [w[1] for w in work_rates]
    else:
        df["work_rate_att"] = 2.0
        df["work_rate_def"] = 2.0
    df["work_rate_att"] = df["work_rate_att"].fillna(2.0)
    df["work_rate_def"] = df["work_rate_def"].fillna(2.0)

    # Contract and tenure
    if "contract_valid_until" in df.columns:
        contract_years = [parse_year(c) for c in df["contract_valid_until"]]
        df["contract_year"] = contract_years
        df["contract_years_remaining"] = df["contract_year"] - df["season"]
    else:
        df["contract_years_remaining"] = 0.0
    df["contract_years_remaining"] = df["contract_years_remaining"].fillna(0.0).clip(-1.0, 10.0)

    if "joined" in df.columns:
        joined_years = [parse_year(j) for j in df["joined"]]
        df["joined_year"] = joined_years
        df["tenure_years"] = (df["season"] - df["joined_year"]).fillna(0.0).clip(0.0, 25.0)
    else:
        df["tenure_years"] = 0.0

    # Parse position ratings
    for col in POS_RATING_COLS:
        if col in df.columns:
            df[f"pos_rating_{col}"] = [parse_position_rating(v) for v in df[col]]
            df.drop(columns=[col], inplace=True)

    # Tags and traits counts
    if "player_tags" in df.columns:
        df["n_tags"] = df["player_tags"].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )
    else:
        df["n_tags"] = 0

    if "player_traits" in df.columns:
        df["n_traits"] = df["player_traits"].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )
    else:
        df["n_traits"] = 0

    # Unified skill blocks & composites
    gk_cols = ["gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning"]
    for c in gk_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["gk_composite"] = df[[c for c in gk_cols if c in df.columns]].mean(axis=1).fillna(0.0)

    outfield_cols = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
    for c in outfield_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["outfield_composite"] = df[[c for c in outfield_cols if c in df.columns]].mean(axis=1).fillna(0.0)

    # Standard body type cleanup
    def clean_body_type(bt):
        if pd.isna(bt):
            return "Normal"
        s = str(bt).strip()
        if s in ["Normal", "Lean", "Stocky"]:
            return s
        return "Unique"

    if "body_type" in df.columns:
        df["body_type_clean"] = df["body_type"].apply(clean_body_type)
    else:
        df["body_type_clean"] = "Normal"

    # Real face flag
    if "real_face" in df.columns:
        df["has_real_face"] = (df["real_face"].astype(str).str.lower() == "yes").astype(int)
    else:
        df["has_real_face"] = 0

    # Ensure overall & potential are numeric
    df["overall"] = pd.to_numeric(df["overall"], errors="coerce").fillna(50).astype(int)
    df["potential"] = pd.to_numeric(df["potential"], errors="coerce").fillna(50).astype(int)
    df["potential_gap"] = df["potential"] - df["overall"]
    df["value_per_overall_point"] = df["value_eur"] / np.maximum(df["overall"], 1)

    # Sort strictly by player and season
    df = df.sort_values(["sofifa_id", "season"]).reset_index(drop=True)
    return df
