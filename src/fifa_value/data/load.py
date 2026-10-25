import zipfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def extract_archive_if_needed(
    archive_path: str = "archive.zip",
    target_dir: str = "data/raw",
    seasons: Sequence[int] = (15, 16, 17, 18, 19, 20, 21),
) -> None:
    """Extract CSV files from archive.zip into target_dir if not already present."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    missing_seasons = [s for s in seasons if not (target_path / f"players_{s}.csv").exists()]
    if not missing_seasons:
        return

    archive_file = Path(archive_path)
    if not archive_file.exists():
        raise FileNotFoundError(f"Archive file not found at {archive_path}")

    with zipfile.ZipFile(archive_file, "r") as zf:
        for member in zf.namelist():
            # Match players_YY.csv inside root or archive/ subfolder
            filename = Path(member).name
            if filename.startswith("players_") and filename.endswith(".csv"):
                dest = target_path / filename
                if not dest.exists():
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())


def load_raw_season(season: int, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load a single season CSV and attach a normalized 4-digit season column."""
    raw_path = Path(raw_dir)
    # Support both 2-digit (15) and 4-digit (2015) input
    short_s = season if season < 100 else season % 100
    year = 2000 + short_s if season < 100 else season

    file_path = raw_path / f"players_{short_s}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Season file not found: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    df["season"] = year
    return df


def load_all_seasons(
    seasons: Sequence[int] = (15, 16, 17, 18, 19, 20, 21),
    raw_dir: str = "data/raw",
) -> pd.DataFrame:
    """Load all specified season CSVs into a single combined panel dataframe."""
    dfs = []
    for s in seasons:
        df_s = load_raw_season(s, raw_dir=raw_dir)
        dfs.append(df_s)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["sofifa_id", "season"]).reset_index(drop=True)
    return combined
