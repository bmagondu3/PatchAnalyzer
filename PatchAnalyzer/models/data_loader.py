# PatchAnalyzer/models/data_loader.py
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────
#  Standard lib & 3rd-party
# ────────────────────────────────────────────────────────────────────────
from pathlib import Path
import csv, fnmatch, re
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ────────────────────────────────────────────────────────────────────────
from ..utils.log import setup_logger
from ..utils.helpers import find_cell_metadata

logger = setup_logger(__name__)           # unified logger for the whole file

# ────────────────────────────────────────────────────────────────────────
#  Public Section A – metadata CSVs  (UNCHANGED)
# ────────────────────────────────────────────────────────────────────────
def _discover_protocol_indices(src_dir: Path) -> list[int]:
    """
    Return every integer index that appears in **VoltageProtocol_*.csv**
    or **CurrentProtocol_*.csv** files under *src_dir*.
    """
    indices: set[int] = set()

    vp_dir = src_dir / "VoltageProtocol"
    if vp_dir.exists():
        for p in vp_dir.glob("VoltageProtocol_*.csv"):
            m = re.search(r"VoltageProtocol_(\d+)", p.name)
            if m:
                indices.add(int(m.group(1)))

    cp_dir = src_dir / "CurrentProtocol"
    if cp_dir.exists():
        for p in cp_dir.glob("CurrentProtocol_*.csv"):
            m = re.search(r"CurrentProtocol_(\d+)_#", p.name)
            if m:
                indices.add(int(m.group(1)))

    return sorted(indices)

def load_metadata(folders: list[Path]) -> pd.DataFrame:
    """
    Load cell-metadata from each acquisition folder.

    Normal path
    -----------
    • If *cell_metadata.csv* exists → behave exactly as the original version.

    Fallback path (always on)
    -------------------------
    • If the file is missing, scan for **VoltageProtocol_*.csv** and/or
      **CurrentProtocol_*.csv** files.  Each discovered integer index becomes
      one synthetic metadata row so ephys data without metadata is still
      analyzable.

        index           : <protocol index>  (int64)
        stage_x / y / z : <protocol index>  (float64 dummy coordinates)
        image           : "cell_<index>.webp"  (keeps `_cell_id` regex working)
        group_label     : ""   (you may edit later)

    After concatenation, any remaining NaN values are replaced by the empty
    string so that `QTableWidgetItem` never receives a float('nan'), which
    Qt cannot construct from.
    """
    import numpy as np  # local import to avoid new top-level changes
    frames: list[pd.DataFrame] = []

    for d in folders:
        meta_csv = find_cell_metadata(d)

        # ── 1) Normal path: real metadata present ───────────────────────
        if meta_csv:
            try:
                df = pd.read_csv(meta_csv, sep=";")
                df["src_dir"] = d
                df["row_idx"] = df.index
                frames.append(df)
                logger.info("Loaded %s rows from %s", len(df), meta_csv)
            except Exception as exc:
                logger.exception("Failed reading %s: %s", meta_csv, exc)
            continue

        # ── 2) Fallback path: discover protocol indices ─────────────────
        indices = _discover_protocol_indices(d)
        if not indices:
            logger.warning(
                "No CellMetadata and no protocol CSVs in %s – skipped", d
            )
            continue

        fake_rows = [{  # synthetic metadata when original CSV is absent
            "index"      : idx,
            "stage_x"    : float(idx),
            "stage_y"    : float(idx),
            "stage_z"    : float(idx),
            "image"      : f"cell_{idx}.webp",
            "group_label": "",
        } for idx in indices]

        df = pd.DataFrame(fake_rows)
        df["src_dir"] = d
        df["row_idx"] = df.index
        frames.append(df)
        logger.info("Fallback: created %s synthetic rows for %s", len(df), d)

    if not frames:
        raise ValueError(
            "None of the selected folders contained metadata or protocol CSVs."
        )

    # ── Combine everything and purge NaNs (Qt dislikes float('nan')) ────
    meta_df = pd.concat(frames, ignore_index=True)
    meta_df.replace({np.nan: ""}, inplace=True)

    return meta_df

# ────────────────────────────────────────────────────────────────────────
#  Public Section B – Ephys CSV loaders  (VERBATIM from *ephys_loader.py*)
# ────────────────────────────────────────────────────────────────────────
# -------- low-level reader ------------------------------------------------
def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, command, response from one space-delimited CSV."""
    t, cmd, rsp = [], [], []
    with open(path, "r", newline="") as fh:
        for row in csv.reader(fh, delimiter=" "):
            if len(row) < 3:             # skip malformed lines
                continue
            t.append(float(row[0]))
            cmd.append(float(row[1]))
            rsp.append(float(row[2]))
    return np.asarray(t), np.asarray(cmd), np.asarray(rsp)


# -------- voltage-protocol ------------------------------------------------
def load_voltage_traces_for_indices(
    src_dir: Path,
    indices: list[int],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return every VoltageProtocol CSV whose filename starts with one of *indices*.
    Accepts both “…_<n>.csv” and “…_<n>_k.csv” (or any extra suffix).
    """
    vp_dir = src_dir / "VoltageProtocol"
    if not vp_dir.exists():
        return {}

    patterns = [f"VoltageProtocol_{i}*.csv" for i in indices]
    traces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for csv_path in vp_dir.glob("*.csv"):
        if any(fnmatch.fnmatchcase(csv_path.name, pat) for pat in patterns):
            traces[csv_path.name] = _read_csv(csv_path)

    return traces


# -------- current-protocol ------------------------------------------------
_REGEX = re.compile(
    r"CurrentProtocol_(\d+)_#[0-9A-Fa-f]{6}_([+-]?\d+(?:\.\d+)?)\.csv$"
)
# Add near the top (if not already present)
C_CLAMP_PAPERV = 400.0          # 400 pA / V

# Replace the whole function with this version
def load_current_traces(
    src_dir: Path,
    cell_indices: list[int] | tuple[int] | set[int],
    *,                       # keyword-only from here on
    thin: int | None = None  # e.g. 10_000 for plotting, None for analysis
) -> dict[int, dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]]:

    cp_dir = src_dir / "CurrentProtocol"
    if not cp_dir.exists():
        return {}

    wanted = {int(i) for i in cell_indices}
    traces: dict[int, dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        cid: {} for cid in wanted
    }

    for csv_path in cp_dir.glob("*.csv"):
        m = _REGEX.match(csv_path.name)
        if not m:
            continue

        cell_id   = int(m.group(1))
        current_pA = float(m.group(2))
        if cell_id not in wanted:
            continue

        t, cmd, rsp = _read_csv(csv_path)

        # ↓ subsample **only** if trace is long enough
        if thin and thin > 1 and len(t) > thin:
            t   = t[::thin]
            cmd = cmd[::thin]
            rsp = rsp[::thin]

        traces[cell_id][current_pA] = (t, cmd * C_CLAMP_PAPERV, rsp)

    return {cid: d for cid, d in traces.items() if d}




# ────────────────────────────────────────────────────────────────────────
#  Public Section C – **NEW**  CC-Sweep container & convenience loader
# ────────────────────────────────────────────────────────────────────────
@dataclass
class CCSweep:
    """
    Light container mirroring the old ``spike_params.Sweep`` dataclass.

    Parameters
    ----------
    time           : seconds
    response_mV    : membrane potential in millivolts
    command_pA     : injected current in pico-amps
    sample_rate    : sampling frequency in Hz
    """
    time:        np.ndarray
    response_mV: np.ndarray
    command_pA:  np.ndarray
    sample_rate: float


def load_current_sweeps(
    src_dir: Path,
    cell_indices: list[int] | tuple[int] | set[int],
) -> dict[int, list[CCSweep]]:
    """
    Thin wrapper around :func:`load_current_traces` that converts each
    (time, cmd, rsp) triple into a :class:`CCSweep`.

    The dict key is the *cell_id*; the list retains **all** currents.
    """
    raw = load_current_traces(src_dir, cell_indices)
    out: dict[int, list[CCSweep]] = {}

    for cid, sweeps in raw.items():
        objs: list[CCSweep] = []
        for (t, cmd_pA, rsp_V) in sweeps.values():
            sr = 1.0 / (t[1] - t[0])
            objs.append(
                CCSweep(
                    time=t,
                    response_mV=rsp_V * 1e3,      # V → mV
                    command_pA=cmd_pA,
                    sample_rate=sr,
                )
            )
        out[cid] = objs

    return out
