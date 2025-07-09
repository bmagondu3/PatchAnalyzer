# PatchAnalyzer/models/data_loader.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..utils.log import setup_logger
from ..utils.helpers import find_cell_metadata

logger = setup_logger(__name__)   # child logger


def load_metadata(folders: list[Path]) -> pd.DataFrame:
    """
    Load every *cell_metadata.csv* found in *folders* into one DataFrame,
    append src_dir + row_idx cols, return the concat result.
    """
    frames: list[pd.DataFrame] = []
    for d in folders:
        meta_csv = find_cell_metadata(d)
        if not meta_csv:
            logger.warning("No CellMetadata in %s – skipped", d)
            continue
        try:
            df = pd.read_csv(meta_csv, sep=";")
            df["src_dir"] = d
            df["row_idx"] = df.index
            frames.append(df)
            logger.info("Loaded %s rows from %s", len(df), meta_csv)
        except Exception as exc:
            logger.exception("Failed reading %s: %s", meta_csv, exc)

    if not frames:
        raise ValueError("None of the selected folders contained valid metadata.")
    return pd.concat(frames, ignore_index=True)
