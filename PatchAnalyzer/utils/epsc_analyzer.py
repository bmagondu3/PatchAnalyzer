#!/usr/bin/env python3
"""
PSC data aggregator
-------------------
- Scans INPUT_DIR for Excel files (*.xlsx, *.xls), non-recursive.
- For each sheet:
    * Uses the 3rd-to-last column (or a column named "AD" if present) as the
      Interevent Interval (IEI) column (already computed in your sheets).
    * Also extracts:
        - instantaneous frequency from the 4th-to-last column
        - absolute peak amplitude from the 8th column (0-based index 7)
- Writes two CSVs:
    * Events CSV (OUTPUT_CSV): one row per event with IEI + extra metrics.
    * Summary CSV (<OUTPUT_CSV stem>_summary.csv): one row per sheet with mean/std.
- Keeps existing KEEP_ORDER behavior for sorting (stable mergesort) before use.
"""
from pathlib import Path
import pandas as pd
import re  # ← added

# ======================
# EDIT THESE CONSTANTS 👇
# ======================
INPUT_DIR = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats")               # <— change to your folder
OUTPUT_CSV = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\sEPSCs_by_sheet.csv")
KEEP_ORDER = True                        # False => sort before use; True => keep original row order
PREFERRED_COLUMN_NAME = "AD"              # If this header exists, use it; else use 3rd-to-last column
FILE_GLOBS = ("*.xlsx", "*.xls")          # Only Excel, per your request
# ======================

# ----------------------
# OPTIONAL POST-PROCESS
# ----------------------
# 1) Combine sheets that share the same numeric prefix (e.g., "2k2","2k3" -> "2")
GENERATE_BY_SHEETPREFIX = True
# 2) Collate to cells using cell_metadata_aggregated.csv (build sEPSCs_by_cell*.csv)
GENERATE_BY_CELL = True

# Path to cell metadata (used only if GENERATE_BY_CELL is True)
CELL_METADATA_CSV = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\cell_metadata_aggregated.csv")

# Regex to extract the sheet's leading digits (numeric prefix)
SHEET_PREFIX_REGEX = r'^(\d+)'
# Round stage coordinates to this many decimals before deduplication into cells
COORD_DECIMALS = 3

# Output filenames for optional artifacts (same folder as OUTPUT_CSV)
BY_SHEETPREFIX_CSV = OUTPUT_CSV.with_name("sEPSCs_by_sheetprefix.csv")
BY_SHEETPREFIX_SUMMARY_CSV = OUTPUT_CSV.with_name("sEPSCs_by_sheetprefix_summary.csv")
BY_CELL_CSV = OUTPUT_CSV.with_name("sEPSCs_by_cell.csv")
BY_CELL_SUMMARY_CSV = OUTPUT_CSV.with_name("sEPSCs_by_cell_summary.csv")

INPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
summary_rows = []  # NEW: accumulate one summary row per sheet

for pattern in FILE_GLOBS:
    for f in sorted(INPUT_DIR.glob(pattern)):
        try:
            x = pd.ExcelFile(f)
            for sheet in x.sheet_names:
                try:
                    df = pd.read_excel(f, sheet_name=sheet)

                    # --- Extra metrics from fixed positions ---
                    # 4th-to-last column: instantaneous frequency (numeric)
                    if df.shape[1] >= 4:
                        inst_series = pd.to_numeric(df.iloc[:, -4], errors="coerce")
                    else:
                        inst_series = pd.Series([float("nan")] * len(df), index=df.index)

                    # 8th column (index 7): absolute peak amplitude (numeric, absolute value)
                    if df.shape[1] > 7:
                        peak_series = pd.to_numeric(df.iloc[:, 7], errors="coerce").abs()
                    else:
                        peak_series = pd.Series([float("nan")] * len(df), index=df.index)

                    # --- Choose IEI column (already computed in your sheets) ---
                    col = PREFERRED_COLUMN_NAME if PREFERRED_COLUMN_NAME in df.columns else df.columns[-3]

                    # series without NA
                    s = df[col].dropna()

                    # NEW: IEI is already stored in the selected column — coerce to numeric
                    vals = pd.to_numeric(s, errors="coerce").dropna()
                    if not KEEP_ORDER:
                        # Stable sort keeps original index labels
                        vals = vals.sort_values(kind="mergesort")

                    # Use 'diffs' name to keep downstream logic style unchanged
                    diffs = vals
                    unit = "units"  # adjust if you know the concrete time unit (e.g., "seconds")

                    # Filter: drop IEI values < 1 and announce
                    ignored = diffs[diffs < 1]
                    if not ignored.empty:
                        print(f"Ignored {len(ignored)} IEI value(s) < 1 in file '{f.name}', sheet '{sheet}'.")
                    diffs = diffs[diffs >= 1]

                    # Summary stats (per sheet)
                    mean = float(diffs.mean()) if len(diffs) else float("nan")
                    std = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")

                    # Means/STD for extra metrics (aligned to IEI rows)
                    inst_on_rows = inst_series.reindex(diffs.index)
                    peak_on_rows = peak_series.reindex(diffs.index)
                    inst_mean = float(inst_on_rows.dropna().mean()) if inst_on_rows.notna().any() else float("nan")
                    inst_std  = float(inst_on_rows.dropna().std(ddof=1)) if inst_on_rows.notna().sum() > 1 else float("nan")
                    peak_mean = float(peak_on_rows.dropna().mean()) if peak_on_rows.notna().any() else float("nan")
                    peak_std  = float(peak_on_rows.dropna().std(ddof=1)) if peak_on_rows.notna().sum() > 1 else float("nan")

                    # One summary row per sheet (no duplication on events)
                    summary_rows.append({
                        "file": f.name,
                        "sheet": sheet,
                        "column_used": str(col),
                        "order": "original" if KEEP_ORDER else "sorted",
                        "unit": unit,
                        "n_intervals": int(len(diffs)),
                        "mean_interevent_interval": mean,
                        "std_interevent_interval": std,
                        "mean_instantaneous_frequency": inst_mean,
                        "std_instantaneous_frequency": inst_std,
                        "mean_peak_amplitude_abs": peak_mean,
                        "std_peak_amplitude_abs": peak_std,
                    })

                    # Per-event rows with aligned extra metrics
                    for idx, d in diffs.items():
                        inst_val = float(inst_series.loc[idx]) if pd.notna(inst_series.loc[idx]) else float("nan")
                        peak_val = float(peak_series.loc[idx]) if pd.notna(peak_series.loc[idx]) else float("nan")

                        rows.append({
                            "file": f.name,
                            "sheet": sheet,
                            "column_used": str(col),
                            "order": "original" if KEEP_ORDER else "sorted",
                            "unit": unit,
                            # In this dataset, event_value IS the IEI
                            "event_value": str(d),
                            "interevent_interval": float(d),
                            "instantaneous_frequency": inst_val,
                            "peak_amplitude_abs": peak_val,
                            "error": "",
                        })
                except Exception as e:
                    # Per-sheet error -> add an events row with error annotated
                    rows.append({
                        "file": f.name, "sheet": sheet, "column_used": "ERROR",
                        "order": "n/a", "unit": "n/a",
                        "event_value": "", "interevent_interval": float("nan"),
                        "instantaneous_frequency": float("nan"), "peak_amplitude_abs": float("nan"),
                        "error": f"{type(e).__name__}: {e}",
                    })
        except Exception as e:
            # File-open error -> add an events row with error annotated
            rows.append({
                "file": f.name, "sheet": "ERROR_OPEN", "column_used": "n/a",
                "order": "n/a", "unit": "n/a",
                "event_value": "", "interevent_interval": float("nan"),
                "instantaneous_frequency": float("nan"), "peak_amplitude_abs": float("nan"),
                "error": f"{type(e).__name__}: {e}",
            })

# Events CSV (compact: no repeated mean/std)
cols = [
    "file","sheet","column_used","order","unit",
    "event_value","interevent_interval","instantaneous_frequency","peak_amplitude_abs","error"
]
out = pd.DataFrame(rows, columns=cols)
out.to_csv(OUTPUT_CSV, index=False)

# Summary CSV (one row per sheet)
SUMMARY_CSV = OUTPUT_CSV.with_name(OUTPUT_CSV.stem + "_summary.csv")
summary_cols = [
    "file","sheet","column_used","order","unit","n_intervals",
    "mean_interevent_interval","std_interevent_interval",
    "mean_instantaneous_frequency","std_instantaneous_frequency",
    "mean_peak_amplitude_abs","std_peak_amplitude_abs"
]
pd.DataFrame(summary_rows, columns=summary_cols).to_csv(SUMMARY_CSV, index=False)

print(f"Wrote {len(out)} event rows to {OUTPUT_CSV}")
print(f"Wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}")

# ================================
# BY-CELL COLLATOR (TEMP FIXES) ✅
# ================================
# This block is self-contained and optional. It does not alter your _by_sheet outputs.
# Hardcoded session corrections for filename->time mapping are applied here.
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# ---- Toggle
GENERATE_BY_CELL = True   # set to False to disable this optional by-cell export

# ---- Inputs/Outputs
CELL_METADATA_CSV = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\cell_metadata_aggregated.csv")
BY_CELL_CSV = OUTPUT_CSV.with_name("sEPSCs_by_cell.csv")
BY_CELL_SUMMARY_CSV = OUTPUT_CSV.with_name("sEPSCs_by_cell_summary.csv")

# ---- Collation parameters
SHEET_PREFIX_REGEX = r'^(\d+)'
COORD_DECIMALS = 3

# ---- Hardcoded filename->time overrides (YYYYMMDDHHMM)
# 712 -> July 1, 2025; 722 -> July 2, 2025; 628 minute align.
HARDCODED_FILE_TIME_OVERRIDES = {
    "71251325.xlsx": "202507011325",
    "72251309.xlsx": "202507021309",
    "628251758.xlsx": "202506281759",
}

def _file_to_time_guess(file_str: str) -> str | None:
    """
    Best-effort parse from event filename (no extension) to 'YYYYMMDDHHMM'.
    Supports 8, 9, or 10-character stems used in your dataset.
    Assumes year 2025 for 1-digit 'Y' stems.
    Examples:
      '71251325'  -> 2025-07-12 13:25  (but will be overridden by hardcoded map for July 1)
      '628251758' -> 2025-06-28 17:58  (tolerance/override will handle 17:59)
    """
    s = Path(file_str).stem
    try:
        if len(s) == 8:   # M DD Y HH MM
            M, DD, Y, HH, MM = int(s[0]), int(s[1:3]), int(s[3]), int(s[4:6]), int(s[6:8])
            return f"2025{M:02d}{DD:02d}{HH:02d}{MM:02d}"
        if len(s) == 9:   # M DD YY HH MM
            M, DD, YY, HH, MM = int(s[0]), int(s[1:3]), int(s[3:5]), int(s[5:7]), int(s[7:9])
            return f"20{YY:02d}{M:02d}{DD:02d}{HH:02d}{MM:02d}"
        if len(s) == 10:  # 2-digit month cases
            M, DD, rest = int(s[0:2]), int(s[2:4]), s[4:]
            if len(rest) == 5:   # Y(1) HH(2) MM(2)
                Y, HH, MM = int(rest[0]), int(rest[1:3]), int(rest[3:5])
                return f"202{Y}{M:02d}{DD:02d}{HH:02d}{MM:02d}"
            if len(rest) == 6:   # YY(2) HH(2) MM(2)
                YY, HH, MM = int(rest[0:2]), int(rest[2:4]), int(rest[4:6])
                return f"20{YY:02d}{M:02d}{DD:02d}{HH:02d}{MM:02d}"
    except Exception:
        return None
    return None

def _nearest_meta_time_for_file(file_str: str, meta_times: set[str], tol_minutes: int = 1) -> str | None:
    """
    Returns a metadata 'time' string for this file by:
    1) applying hardcoded overrides (this run), else
    2) parsing a best-effort guess, then
    3) matching nearest minute within ±tol_minutes present in metadata.
    """
    # 1) hardcoded fixes win
    base = HARDCODED_FILE_TIME_OVERRIDES.get(Path(file_str).name)
    if base:
        return base

    # 2) best-effort guess
    guess = _file_to_time_guess(file_str)
    if not guess:
        return None

    # 3) ± tol minute match against available metadata times
    try:
        dt = datetime.strptime(guess, "%Y%m%d%H%M")
    except ValueError:
        return None
    for delta in range(-tol_minutes, tol_minutes + 1):
        cand = (dt + timedelta(minutes=delta)).strftime("%Y%m%d%H%M")
        if cand in meta_times:
            return cand
    return None

def _first_non_null(series):
    for v in series:
        if pd.notna(v):
            return v
    return float("nan")

if GENERATE_BY_CELL:
    try:
        meta_raw = pd.read_csv(CELL_METADATA_CSV)

        # Identify the packed metadata column: "index;stage_x;stage_y;stage_z;image"
        packed_cols = [c for c in meta_raw.columns if ";" in c]
        if not packed_cols:
            print("[BY_CELL] No packed metadata column found; skipping by-cell export.")
        else:
            packed = packed_cols[0]
            parts = meta_raw[packed].astype(str).str.split(";", expand=True)
            parts.columns = ["index", "stage_x", "stage_y", "stage_z", "image"]
            meta = pd.concat([meta_raw[["time"]].copy(), parts], axis=1)
            meta["time"] = meta["time"].astype(str)

            # Normalize coordinates
            for c in ["stage_x", "stage_y", "stage_z"]:
                meta[c] = pd.to_numeric(meta[c], errors="coerce").round(COORD_DECIMALS)

            # Build join keys
            ev = out.copy()
            pref = ev["sheet"].astype(str).str.extract(SHEET_PREFIX_REGEX)
            ev["sheet_prefix"] = pref[0].fillna(ev["sheet"].astype(str))
            meta["sheet_prefix"] = meta["index"].astype(str)

            # Map each event file to a metadata 'time' using overrides and ±1min tolerance
            meta_times = set(meta["time"].astype(str).unique())
            ev["time"] = ev["file"].map(lambda fn: _nearest_meta_time_for_file(fn, meta_times, tol_minutes=1))

            # Log any files that failed to map to a session time
            missed = ev["time"].isna()
            if missed.any():
                print("[BY_CELL] Unmatched files (no metadata time even after overrides/tolerance):",
                      sorted(ev.loc[missed, "file"].unique()))

            # Prepare metadata keyed by 'time'
            meta_small = meta[["time", "sheet_prefix", "stage_x", "stage_y", "stage_z"]].drop_duplicates()
            meta_small["coord_key"] = (
                meta_small["time"].astype(str) + "|" +
                meta_small["stage_x"].astype(str) + "|" +
                meta_small["stage_y"].astype(str) + "|" +
                meta_small["stage_z"].astype(str)
            )
            # Assign per-session cell IDs (cell_001, cell_002, ...)
            coord_to_uid = meta_small[["time", "coord_key"]].drop_duplicates().sort_values(["time", "coord_key"])
            coord_to_uid["cell_uid"] = coord_to_uid.groupby("time").cumcount().add(1).map(lambda i: f"cell_{i:03d}")
            meta_small = meta_small.merge(coord_to_uid, on=["time", "coord_key"], how="left")

            # Join events to cells on (time, sheet_prefix)
            ev2 = ev.merge(meta_small[["time", "sheet_prefix", "cell_uid"]],
                           on=["time", "sheet_prefix"], how="left")
            mapped = ev2[ev2["cell_uid"].notna()].copy()

            if mapped.empty:
                print("[BY_CELL] No events mapped to cells; check overrides.")
            else:
                # Replace 'sheet' with cell id and write
                mapped["sheet"] = mapped["cell_uid"].astype(str)
                mapped = mapped[out.columns]  # keep your original event columns
                mapped.to_csv(BY_CELL_CSV, index=False)

                # Per-cell summary (same fields you already export per sheet)
                meta_cols = mapped.groupby(["file", "sheet"]).agg(
                    column_used=("column_used", _first_non_null),
                    order=("order", _first_non_null),
                    unit=("unit", _first_non_null),
                ).reset_index()
                grp = mapped.groupby(["file", "sheet"], as_index=False).agg(
                    n_intervals=("interevent_interval", "size"),
                    mean_interevent_interval=("interevent_interval", "mean"),
                    std_interevent_interval=("interevent_interval", "std"),
                    mean_instantaneous_frequency=("instantaneous_frequency", "mean"),
                    std_instantaneous_frequency=("instantaneous_frequency", "std"),
                    mean_peak_amplitude_abs=("peak_amplitude_abs", "mean"),
                    std_peak_amplitude_abs=("peak_amplitude_abs", "std"),
                )
                summary_by_cell = pd.merge(meta_cols, grp, on=["file", "sheet"], how="left")
                summary_by_cell.to_csv(BY_CELL_SUMMARY_CSV, index=False)

                total = len(ev2)
                nmapped = len(mapped)
                print(f"[BY_CELL] Wrote {nmapped}/{total} mapped events to {BY_CELL_CSV} "
                      f"and summary to {BY_CELL_SUMMARY_CSV}.")
    except Exception as e:
        print(f"[BY_CELL] Error during cell collation: {type(e).__name__}: {e}")
