from __future__ import annotations
from pathlib import Path
import re
from typing import List, Optional

import pandas as pd

def find_cell_metadata_csvs(root: Path) -> List[Path]:
    """
    Recursively find all `cell_metadata.csv` files that live directly inside a `CellMetadata` folder.
    Example match: /root/2025_06_25-13_30/CellMetadata/cell_metadata.csv
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    return sorted(
        p
        for p in root.rglob("cell_metadata.csv")
        if p.parent.name == "CellMetadata"
    )

def normalize_time_string(name: str, digits_only: bool = True) -> str:
    """
    Turn the timestamp-like folder name into the desired `time` string.
    By default, we keep only digits to satisfy “leaving only the string of numbers”.
    If you prefer to only drop underscores (and keep dashes), set digits_only=False.
    """
    if digits_only:
        return re.sub(r"\D", "", name)
    return name.replace("_", "")

def extract_time_from_parent(csv_path: Path, digits_only: bool = True) -> str:
    """
    Given .../<PARENT>/CellMetadata/cell_metadata.csv, return normalized `<PARENT>` as `time`.
    """
    parent_one_up = Path(csv_path).parent.parent.name
    return normalize_time_string(parent_one_up, digits_only=digits_only)

def load_and_tag_csv(csv_path: Path, time_value: str) -> pd.DataFrame:
    """
    Read a single CSV and insert a `time` column as the first column.
    """
    df = pd.read_csv(csv_path)
    df.insert(0, "time", time_value)
    return df

def collate_cell_metadata(
    root_dir: Path,
    output_path: Optional[Path] = None,
    digits_only: bool = True,
) -> pd.DataFrame:
    """
    Find all cell_metadata.csv files under `root_dir`, add the `time` column to each,
    concatenate them, and write the result to `output_path` (or default path).
    Returns the combined DataFrame.
    """
    csv_paths = find_cell_metadata_csvs(root_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No 'cell_metadata.csv' files found under {root_dir}")

    frames = []
    for p in csv_paths:
        time_value = extract_time_from_parent(p, digits_only=digits_only)
        frames.append(load_and_tag_csv(p, time_value))

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if output_path is None:
        output_path = Path(root_dir) / "cell_metadata_aggregated.csv"
    output_path = Path(output_path)
    combined.to_csv(output_path, index=False)

    print(
        f"Saved {len(combined)} rows from {len(csv_paths)} file(s) to '{output_path}'."
    )
    return combined

if __name__ == "__main__":
    # >>> Edit this path to point at your experiments root directory <<<
    ROOT_DIR = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp")

    # Optional: set a custom output location/filename, or leave as None to use the default
    OUTPUT_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\cell_metadata_aggregated.csv")  # e.g., Path(r"/path/to/save/cell_metadata_aggregated.csv")

    # If you prefer to only remove underscores (keep dashes), set to False.
    DIGITS_ONLY = True

    collate_cell_metadata(ROOT_DIR, OUTPUT_PATH, digits_only=DIGITS_ONLY)
