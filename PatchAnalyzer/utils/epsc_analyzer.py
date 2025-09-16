#!/usr/bin/env python3
"""
Interevent intervals per sheet (simple, no CLI)
------------------------------------------------
- Edit the constants below.
- Scans INPUT_DIR for Excel files (*.xlsx, *.xls), non-recursive.
- For each sheet, uses the 3rd-to-last column (or a column named "AD" if present).
- Computes interevent intervals:
    * If the column looks like datetimes -> intervals in seconds
    * Otherwise -> intervals in the column's numeric units
- Writes one combined CSV with all intervals + per-sheet mean/std.
"""
from pathlib import Path
import pandas as pd

# ======================
# EDIT THESE CONSTANTS 👇
# ======================
INPUT_DIR = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats")               # <— change to your folder
OUTPUT_CSV = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\interevent_intervals_by_sheet.csv")
KEEP_ORDER = False                        # False => sort before differencing; True => use original row order
PREFERRED_COLUMN_NAME = "AD"              # If this header exists, use it; else use 3rd-to-last column
FILE_GLOBS = ("*.xlsx", "*.xls")          # Only Excel, per your request
# ======================

INPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for pattern in FILE_GLOBS:
    for f in sorted(INPUT_DIR.glob(pattern)):
        try:
            x = pd.ExcelFile(f)
            for sheet in x.sheet_names:
                try:
                    df = pd.read_excel(f, sheet_name=sheet)

                    # choose column
                    col = PREFERRED_COLUMN_NAME if PREFERRED_COLUMN_NAME in df.columns else df.columns[-3]

                    # series without NA
                    s = df[col].dropna()

                    # try numeric first
                    num = pd.to_numeric(s, errors="coerce")
                    numeric_ratio = num.notna().mean() if len(s) else 0.0
                    is_numeric = pd.api.types.is_numeric_dtype(s)

                    if is_numeric or numeric_ratio >= 0.8:
                        vals = num.dropna()
                        if not KEEP_ORDER:
                            vals = vals.sort_values(kind="mergesort")
                        diffs = vals.diff().dropna()
                        unit = "units"
                    else:
                        # try datetime
                        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                        dt_ratio = dt.notna().mean() if len(s) else 0.0
                        if dt_ratio >= 0.5:
                            vals = dt.dropna()
                            if not KEEP_ORDER:
                                vals = vals.sort_values(kind="mergesort")
                            diffs = vals.diff().dropna().dt.total_seconds()
                            unit = "seconds"
                        else:
                            vals = num.dropna()
                            if not KEEP_ORDER:
                                vals = vals.sort_values(kind="mergesort")
                            diffs = vals.diff().dropna()
                            unit = "units"

                    mean = float(diffs.mean()) if len(diffs) else float("nan")
                    std = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")

                    later = vals.iloc[1:]
                    for v, d in zip(later, diffs):
                        rows.append({
                            "file": f.name,
                            "sheet": sheet,
                            "column_used": str(col),
                            "order": "original" if KEEP_ORDER else "sorted",
                            "unit": unit,
                            "event_value": str(v),
                            "interevent_interval": float(d),
                            "mean_for_sheet": mean,
                            "std_for_sheet": std,
                            "n_intervals": int(len(diffs)),
                        })
                except Exception as e:
                    rows.append({
                        "file": f.name, "sheet": sheet, "column_used": "ERROR",
                        "order": "n/a", "unit": "n/a",
                        "event_value": "", "interevent_interval": float("nan"),
                        "mean_for_sheet": float("nan"), "std_for_sheet": float("nan"), "n_intervals": 0,
                        "error": f"{type(e).__name__}: {e}",
                    })
        except Exception as e:
            rows.append({
                "file": f.name, "sheet": "ERROR_OPEN", "column_used": "n/a",
                "order": "n/a", "unit": "n/a",
                "event_value": "", "interevent_interval": float("nan"),
                "mean_for_sheet": float("nan"), "std_for_sheet": float("nan"), "n_intervals": 0,
                "error": f"{type(e).__name__}: {e}",
            })

cols = [
    "file","sheet","column_used","order","unit",
    "event_value","interevent_interval",
    "mean_for_sheet","std_for_sheet","n_intervals","error"
]
out = pd.DataFrame(rows, columns=cols)
out.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(out)} rows to {OUTPUT_CSV}")
