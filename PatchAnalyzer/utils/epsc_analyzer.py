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

# ======================
# EDIT THESE CONSTANTS 👇
# ======================
INPUT_DIR = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats")               # <— change to your folder
OUTPUT_CSV = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\sEPSCs_by_sheet.csv")
KEEP_ORDER = True                        # False => sort before use; True => keep original row order
PREFERRED_COLUMN_NAME = "AD"              # If this header exists, use it; else use 3rd-to-last column
FILE_GLOBS = ("*.xlsx", "*.xls")          # Only Excel, per your request
# ======================

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
