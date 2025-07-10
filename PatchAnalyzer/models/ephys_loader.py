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

