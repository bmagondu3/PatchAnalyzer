# PatchAnalyzer/models/ephys_loader.py
from __future__ import annotations
from pathlib import Path
import csv, re, numpy as np

# ── helpers ──────────────────────────────────────────────────────────────
_CELL_RE = re.compile(r"(\d+)")          # first number – used as cell id


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, command, response from one space-delimited CSV."""
    t, cmd, rsp = [], [], []
    with open(path, "r") as fh:
        for row in csv.reader(fh, delimiter=" "):
            if len(row) < 3:
                continue
            t.append(float(row[0]))
            cmd.append(float(row[1]))
            rsp.append(float(row[2]))
    return np.asarray(t), np.asarray(cmd), np.asarray(rsp)


# ── add this NEW helper at the end of the file ───────────────────────────
import fnmatch

def load_voltage_traces_for_indices(
    src_dir: Path,
    indices: list[int],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return every VoltageProtocol CSV whose filename starts with any of the
    supplied *indices*.

    Accepts both “…_<n>.csv” and “…_<n>_k.csv” (or any extra suffix).
    """
    vp_dir = src_dir / "VoltageProtocol"
    if not vp_dir.exists():
        return {}

    traces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    patterns = [f"VoltageProtocol_{i}*.csv" for i in indices]

    for csv_path in vp_dir.glob("*.csv"):
        name = csv_path.name
        if any(fnmatch.fnmatchcase(name, pat) for pat in patterns):
            traces[name] = _read_csv(csv_path)

    return traces

# ── Current-clamp loader – nested {cell_id ▶ current_pA ▶ trace} ─────────
import re as _re
import numpy as _np
from pathlib import Path

# Pattern:  CurrentProtocol_<cell>_<hex-colour>_<current>.csv
_REGEX = _re.compile(
    r"CurrentProtocol_(\d+)_#[0-9A-Fa-f]{6}_([+-]?\d+(?:\.\d+)?)\.csv$"
)

# *** NEW: convert V → pA  (400 pA / V) ***
C_CLAMP_PAPERV = 400.0           # pico-amps per Volt

def load_current_traces(
    src_dir:      Path,
    cell_indices: list[int] | tuple[int] | set[int],
) -> dict[int, dict[float, tuple[_np.ndarray, _np.ndarray, _np.ndarray]]]:
    """
    Return
        {cell_id: {current_inj_pA: (time, cmd, rsp)}}

    • **Every CSV is unique** by definition of <cell_hex_current>, so no
      clashes are expected; the last assignment wins if duplicates ever
      appear by mistake.
    • If *cell_indices* is empty, nothing is returned.
    """
    cp_dir = src_dir / "CurrentProtocol"
    if not cp_dir.exists():
        return {}

    wanted = {int(i) for i in cell_indices}
    traces: dict[int, dict[float, tuple[_np.ndarray, _np.ndarray, _np.ndarray]]] = {
        cid: {} for cid in wanted
    }

    for csv_path in cp_dir.glob("*.csv"):
        m = _REGEX.match(csv_path.name)
        if not m:
            continue

        cell_id = int(m.group(1))
        if cell_id not in wanted:
            continue

        current_pA = float(m.group(2))
        t, cmd, rsp = _read_csv(csv_path)
        cmd *= C_CLAMP_PAPERV            # <<< 400× scaling to pA
        traces[cell_id][current_pA] = (t, cmd, rsp)

    return {cid: d for cid, d in traces.items() if d}
