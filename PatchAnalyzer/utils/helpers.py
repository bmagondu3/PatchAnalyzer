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

def find_current_image(folder: Path, cell_image_name: str) -> Path | None:
    """
    Locate *CurrentProtocol_<id>.png*  **OR**  *CurrentProtocol_<id>_anything.png*
    (case-insensitive) in *folder*/CurrentProtocol.
    """
    m = _CELL_RE.search(cell_image_name)
    if not m:
        return None
    cell_id = m.group(1)

    cp_dir = folder / "CurrentProtocol"
    if not cp_dir.exists():
        return None

    prefix = f"currentprotocol_{cell_id}".lower()   # no trailing “_”
    for f in cp_dir.glob("*.png"):
        fn = f.name.lower()
        if fn.startswith(prefix) and fn.endswith(".png"):
            return f
    return None



def find_holding_image(folder: Path, cell_image_name: str) -> Path | None:
    """
    Locate HoldingProtocol_<id>.png (case-insensitive) in *folder*/HoldingProtocol.

    Only that exact pattern is accepted — no extra suffixes.
    """
    m = _CELL_RE.search(cell_image_name)
    if not m:
        return None
    cell_id = m.group(1)

    hp_dir = folder / "HoldingProtocol"
    if not hp_dir.exists():
        return None

    # Case-insensitive exact filename match
    target = f"holdingprotocol_{cell_id}.png"
    for f in hp_dir.iterdir():
        if f.is_file() and f.name.lower() == target:
            return f
    return None