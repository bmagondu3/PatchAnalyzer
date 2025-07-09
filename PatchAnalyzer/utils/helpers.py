# PatchAnalyzer/utils/helpers.py
from __future__ import annotations
import re
from pathlib import Path

_CELL_RE = re.compile(r"cell[_\-]?(\d+)", re.IGNORECASE)  # extract the id


def find_cell_metadata(folder: Path) -> Path | None:
    """Return *CellMetadata/cell_metadata.csv* inside *folder* (if present)."""
    meta = folder / "CellMetadata" / "cell_metadata.csv"
    return meta if meta.exists() else None


def find_voltage_image(folder: Path, cell_image_name: str) -> Path | None:
    """
    Locate VoltageProtocol_<id>.png (case-insensitive) in *folder*/VoltageProtocol.

    Only that exact pattern is accepted — no extra suffixes.
    """
    m = _CELL_RE.search(cell_image_name)
    if not m:
        return None
    cell_id = m.group(1)

    vp_dir = folder / "VoltageProtocol"
    if not vp_dir.exists():
        return None

    # Case-insensitive exact filename match
    target = f"voltageprotocol_{cell_id}.png"
    for f in vp_dir.iterdir():
        if f.is_file() and f.name.lower() == target:
            return f
    return None