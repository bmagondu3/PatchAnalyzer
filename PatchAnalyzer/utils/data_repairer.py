#!/usr/bin/env python3
# stitch_by_fit.py
# “Fits-if” segment stitcher that preserves original timestamps,
# trims overlapped heads, skips duplicates, and inserts NaNs across true gaps.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob

# --- core helpers -------------------------------------------------------------

def find_monotonic_runs(t, min_len=50):
    dt = np.diff(t)
    segs, start = [], 0
    for i, ok in enumerate(dt > 0):
        if not ok:
            if i + 1 - start >= min_len:
                segs.append((start, i + 1))
            start = i + 1
    if len(t) - start >= min_len:
        segs.append((start, len(t)))
    return segs

def _edge_score(gap, y_jump, max_gap, value_weight):
    # normalized, time-first score: smaller is better
    g = 0.0 if max_gap <= 0 else min(max(gap / max_gap, 0.0), 1.0)
    return (1.0 - value_weight) * g + value_weight * y_jump

def remove_timestamp_outliers(
    df,
    mad_z_hi=12.0,
    iqr_k=6.0,
):
    """
    Remove obvious timestamp outliers that blow up the x-axis.
    Uses two robust guards (MAD-based z and IQR fence) and drops only rows
    that violate both, keeping removal conservative.
    """
    out = df.copy()
    t = out["timestamp"].to_numpy()

    med = np.nanmedian(t)
    mad = np.nanmedian(np.abs(t - med))
    if mad == 0 or not np.isfinite(mad):
        guard1 = np.zeros_like(t, dtype=bool)
    else:
        robust_z = 0.6745 * (t - med) / mad
        guard1 = robust_z > mad_z_hi

    q1, q3 = np.nanpercentile(t, [25, 75])
    iqr = q3 - q1
    high_cut = q3 + iqr_k * iqr
    guard2 = t > high_cut

    drop_mask = guard1 & guard2
    return out.loc[~drop_mask].reset_index(drop=True)

def enforce_time_limit(stitched, t_max_limit):
    """
    Ensure stitched data respects the cleaned terminal timestamp, preserving NaNs.
    """
    if t_max_limit is None or not np.isfinite(t_max_limit):
        return stitched
    keep = stitched["timestamp"].isna() | (stitched["timestamp"] <= t_max_limit)
    return stitched.loc[keep].reset_index(drop=True)

def stitch_by_fit(
    df,
    max_gap_factor=20.0,
    value_weight=0.2,
    min_len=50,
    use_col="command_mV",
    gap_nan_threshold_factor=3.0,
):
    """
    Build a stitched chain of monotonic runs using only:
      - x-axis continuity (small time gap)
      - edge-value continuity (small jump at boundary)
    Enhancements:
      - If next run overlaps in time but extends past current end, trim its head and stitch.
      - If next run is fully before or inside what we already placed, treat as redundant.
      - Insert a NaN separator when a real gap exists to avoid diagonal lines when plotting.

    Parameters
    ----------
    max_gap_factor : float
        Allowed time gap relative to the file's median positive Δt.
    value_weight : float ∈ [0,1]
        Importance of edge-value jump (secondary to time gap).
    min_len : int
        Minimum length of a monotonic run to consider.
    use_col : str
        Column to use for edge continuity ("command_mV" or "response_pA").
    gap_nan_threshold_factor : float
        If the gap between current end and next start exceeds this * Δt_median,
        insert a NaN row before appending (prevents diagonal connectors in plots).
    """
    assert use_col in df.columns, f"use_col must be one of {list(df.columns)}"
    t = df["timestamp"].to_numpy()
    y = df[use_col].to_numpy()

    segs = find_monotonic_runs(t, min_len=min_len)
    if not segs:
        return [], pd.DataFrame(columns=df.columns)

    # basic time scales
    dt_pos = np.diff(t); dt_pos = dt_pos[dt_pos > 0]
    med_dt = float(np.median(dt_pos)) if len(dt_pos) else 1.0
    eps = med_dt * 0.25    # tolerance for "strictly after" trimming
    max_gap = max_gap_factor * med_dt
    gap_nan_threshold = gap_nan_threshold_factor * med_dt

    # Greedy chain: start from the earliest run
    unused = list(range(len(segs)))
    chain = []
    s0 = min(unused, key=lambda i: t[segs[i][0]])
    chain.append(s0)
    unused.remove(s0)

    # We'll build the stitched result incrementally so we can trim overlaps
    parts = [df.iloc[segs[s0][0]:segs[s0][1]]]
    placed_end_time = t[segs[s0][1] - 1]
    placed_end_value = y[segs[s0][1] - 1]

    # bookkeeping: for diagnostics
    decisions = []

    while unused:
        # Collect candidates:
        # - Case A: non-overlapping forward (start >= placed_end_time + eps)
        # - Case B: overlapping but extends past placed_end_time -> can trim head
        # - Case C: fully redundant (end <= placed_end_time) -> skip
        cands = []
        for j in unused:
            sJ, eJ = segs[j]
            t_start = t[sJ]
            t_end   = t[eJ - 1]

            if t_end <= placed_end_time:  # fully behind or inside -> redundant
                continue

            if t_start >= placed_end_time + eps:
                # forward, real gap = t_start - placed_end_time
                gap = t_start - placed_end_time
                if gap > max_gap:
                    continue
                y_jump = abs(y[sJ] - placed_end_value)
                score = _edge_score(gap, y_jump, max_gap, value_weight)
                cands.append(("forward", score, gap, y_jump, j, sJ, eJ))
            else:
                # overlap: starts before current end but extends beyond
                # find first index strictly after placed_end_time
                s_trim = sJ + int(np.searchsorted(t[sJ:eJ], placed_end_time + eps, side="left"))
                if s_trim >= eJ or (eJ - s_trim) < max(2, min_len // 4):
                    # nothing meaningful left; treat as redundant
                    continue
                gap = 0.0  # by definition after trimming it abuts the boundary
                y_jump = abs(y[s_trim] - placed_end_value)
                score = _edge_score(gap, y_jump, max_gap, value_weight)
                cands.append(("overlap_trim", score, gap, y_jump, j, s_trim, eJ))

        if not cands:
            break

        # Choose the best candidate: smallest score, then smallest gap, then smallest y_jump
        cands.sort(key=lambda x: (x[1], x[2], x[3]))
        mode, score, gap, y_jump, jbest, s_use, e_use = cands[0]

        # Append with possible NaN separator on real gaps
        if mode == "forward" and gap > gap_nan_threshold:
            parts.append(pd.DataFrame({"timestamp": [np.nan], "command_mV": [np.nan], "response_pA": [np.nan]}))

        parts.append(df.iloc[s_use:e_use])

        # Update current end
        placed_end_time = t[e_use - 1]
        placed_end_value = y[e_use - 1]
        chain.append(jbest)
        unused.remove(jbest)

        decisions.append({
            "chosen_run_index": jbest,
            "mode": mode,
            "used_range": (s_use, e_use),
            "gap_s": float(gap),
            "y_jump": float(y_jump),
            "score": float(score),
        })

    stitched = pd.concat(parts, ignore_index=True)

    # Return the *original* run ranges for the chosen chain (for logging)
    chosen_runs = [segs[i] for i in chain]
    return chosen_runs, stitched

# --- batch processing ---------------------------------------------------------

def main():
    # 1️⃣  Edit this path only:
    # FOLDER = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Levi_Injury_exp\corrected\2025_10_31-12_41\VoltageProtocol"
    # FOLDER =r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Levi_Injury_exp\corrected\2025_10_31-11_29\VoltageProtocol"
    # FOLDER = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Levi_Injury_exp\corrected\2025_10_31-14_56\VoltageProtocol"
    # FOLDER = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Levi_Injury_exp\corrected\2025_11_03-12_31\VoltageProtocol"
    # FOLDER = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Levi_Injury_exp\corrected\2025_11_03-17_04\VoltageProtocol"
    # FOLDER = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Levi_Injury_exp\corrected\2025_10_31-17_58\VoltageProtocol"

    #Forest data
    # FOLDER = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Forest_HEK_exp\corrected\2025_11_02-19_41\VoltageProtocol"
    FOLDER = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Forest_HEK_exp\corrected\2025_11_02-21_00\VoltageProtocol"

    csvs = sorted(glob.glob(os.path.join(FOLDER, "*.csv")))
    if not csvs:
        print("No CSV files found in", FOLDER)
        return

    out_dir = os.path.join(os.path.dirname(FOLDER), "VoltageProtocol_stitched")
    os.makedirs(out_dir, exist_ok=True)

    for f in csvs:
        base_name = os.path.basename(f)
        base_name_no_ext = os.path.splitext(base_name)[0]
        out_csv = os.path.join(out_dir, base_name)
        out_png = os.path.join(out_dir, base_name_no_ext + ".png")

        print(f"\nProcessing {os.path.basename(f)} ...")

        # --- load robustly ------------------------------------------------
        try:
            df = pd.read_csv(f, sep=r"\s+", header=None, engine="python")
            if df.shape[1] < 3:
                # fallback to comma-separated
                df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f)
        df = df.iloc[:, :3]
        df.columns = ["timestamp", "command_mV", "response_pA"]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        # Remove absurd timestamp spikes and capture the cleaned end time.
        df = remove_timestamp_outliers(df, mad_z_hi=12.0, iqr_k=6.0)
        t_max_limit = float(df["timestamp"].max()) if not df.empty else None

        # --- stitch (time-first “fits-if”, with overlap trimming) ---------
        chosen, stitched = stitch_by_fit(
            df,
            max_gap_factor=50.0,      # allow longer pauses if present
            value_weight=0.15,        # light preference for edge-value continuity
            min_len=40,               # accept shorter runs too
            use_col="command_mV",
            gap_nan_threshold_factor=3.0,
        )
        print("  Chosen runs:", chosen)
        # Clamp stitched data to the cleaned timestamp limit.
        stitched = enforce_time_limit(stitched, t_max_limit)
        stitched.to_csv(
            out_csv,
            index=False,
            header=False,
            sep=" ",
            columns=["timestamp", "command_mV", "response_pA"],
        )

        # --- plot (NaNs prevent diagonals) --------------------------------
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(stitched["timestamp"], stitched["command_mV"], lw=1.2)
        ax[0].set_ylabel("Command (mV)")
        ax[0].set_title(base_name_no_ext)
        ax[1].plot(stitched["timestamp"], stitched["response_pA"], lw=1.2)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Response (pA)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

        print("  Saved:", out_csv, "and", out_png)

if __name__ == "__main__":
    main()
