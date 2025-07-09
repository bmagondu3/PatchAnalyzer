# PatchAnalyzer/utils/helpers.py
from __future__ import annotations
import re
from pathlib import Path

CELL_ID_RE = re.compile(r"cell[_\-]?(\d+)", re.IGNORECASE)


def find_cell_metadata(folder: Path) -> Path | None:
    """Return *CellMetadata/cell_metadata.csv* inside *folder* (if present)."""
    meta = folder / "CellMetadata" / "cell_metadata.csv"
    return meta if meta.exists() else None

def find_voltage_image(folder: Path, row_idx: int) -> Path | None:
    """
    Return the *first* VoltageProtocol_<row_idx>.png inside *folder*/VoltageProtocol.

    Example
    -------
    row_idx == 7  ➜  VoltageProtocol_7.png

    Parameters
    ----------
    folder
        The experiment folder (i.e. the same src_dir stored in meta_df).
    row_idx
        Original index of the CSV row (already stored in meta_df["row_idx"]).

    Returns
    -------
    pathlib.Path or None
        Path to the PNG if found, else None.
    """
    vp_dir = folder / "VoltageProtocol"
    if not vp_dir.exists():
        return None

    target = vp_dir / f"VoltageProtocol_{row_idx}.png"
    return target if target.exists() else None