import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal  # <-- NEW

# -------------------- CONFIG (edit as needed) --------------------
CSV_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\2025_07_01-13_25\HoldingProtocol\HoldingProtocol_4_k_3.csv")  # your file
OUTPUT_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\Tau_Trace.png")

# CSV_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\2025_06_29-12_42\HoldingProtocol\HoldingProtocol_4_k_3.csv")
# OUTPUT_PATH =Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\Ctrl_Trace.png")


LINEWIDTH = 1.0
FIGSIZE = (10, 4)

AMP_UNITS = "pA"
SCALEBAR_TIME_FRACTION = 0.10
SCALEBAR_AMP_FRACTION  = 0.18

# --- Force the scalebar values to be constant across figures ---
# Units: seconds for time, pA for amplitude (y’s are converted to pA below)
FORCED_SB_TIME = 10    # e.g. 0.050 for 50 ms; 10 means 10 s
FORCED_SB_AMP  = 10    # e.g. 50 for 50 pA
SCALEBAR_FONT_SIZE = 10

# --- Filtering controls ---
APPLY_FILTER = True
FILTER_CUTOFF_HZ = 2000.0
FILTER_ORDER = 8
FILTER_INITIAL = 'zero'   # 'zero' (Clampfit-like) or 'steady'

# --- Baseline controls (NEW) ---
BASELINE_WINDOW_SEC = 30.0     # use last 30 s to estimate baseline
BASELINE_METHOD = "median"     # "median" or "mean"
BASELINE_AFTER_FILTER = True   # baseline on filtered signal (recommended)

# --- Scalebar placement controls (right-side "└" style) ---
RIGHT_MARGIN_FRAC = 0.22
SB_X_IN_MARGIN    = 0.75
SB_Y_POS_FRAC     = 0.18

# --- Lock axes so scalebar has constant visual size (and data adapts) ---
LOCK_Y_TO_SB = True
Y_SCALE_MULT = 3.0   # half-span = this * FORCED_SB_AMP
Y_CENTER     = 0.0   # baseline will be shifted to 0 pA, so keep this at 0
# ---------------------------------------------------------------


def read_three_columns(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python", header=None)
    except Exception:
        df = pd.read_csv(path, header=None)
    df = df.iloc[:, :3].copy()
    df.columns = ["time", "y1", "y2"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


def nice_length(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    frac = x / 10**exp
    if frac < 1.5:   base = 1.0
    elif frac < 3.0: base = 2.0
    elif frac < 7.0: base = 5.0
    else:            base = 10.0
    return base * 10**exp


def time_label(seconds: float) -> str:
    if seconds < 1:
        val = seconds * 1000.0
        return f"{val:.0f} ms" if val >= 100 else (f"{val:.1f} ms" if val >= 10 else f"{val:.2f} ms")
    return f"{seconds:.0f} s" if seconds >= 100 else (f"{seconds:.1f} s" if seconds >= 10 else f"{seconds:.2f} s")


def amp_label(val: float) -> str:
    v = float(val)
    if abs(v) >= 100:   return f"{v:.0f} {AMP_UNITS}"
    if abs(v) >= 10:    return f"{v:.0f} {AMP_UNITS}"
    if abs(v) >= 1:     return f"{v:.1f} {AMP_UNITS}"
    if abs(v) >= 0.1:   return f"{v:.2f} {AMP_UNITS}"
    return f"{v:.3f} {AMP_UNITS}"


# -------------------- 8-pole Bessel LPF (Clampfit-like) --------------------
def bessel_lowpass_clampfit(y, fs, cutoff_hz, order=8, initial='zero'):
    cutoff_hz = float(min(cutoff_hz, 0.49*fs))
    try:
        sos = signal.bessel(order, cutoff_hz, btype='low', analog=False,
                            output='sos', norm='mag', fs=fs)
    except TypeError:
        Wn = cutoff_hz / (fs/2.0)
        sos = signal.bessel(order, Wn, btype='low', analog=False,
                            output='sos', norm='mag')
    if initial == 'steady':
        zi = signal.sosfilt_zi(sos)
        y_f, _ = signal.sosfilt(sos, y, zi=zi*y[0])
        return y_f
    else:
        return signal.sosfilt(sos, y)


def add_scalebar(ax, xmin, xmax, ymin, ymax, xrange, yrange, right_margin):
    sb_time = FORCED_SB_TIME
    sb_amp  = FORCED_SB_AMP
    x_corner = xmax + SB_X_IN_MARGIN * right_margin
    y_base   = ymin + SB_Y_POS_FRAC * yrange
    ax.plot([x_corner, x_corner], [y_base, y_base + sb_amp], color="black", linewidth=LINEWIDTH)
    ax.plot([x_corner - sb_time, x_corner], [y_base, y_base], color="black", linewidth=LINEWIDTH)
    ax.text(x_corner - sb_time/2, y_base - 0.045*yrange, time_label(sb_time),
            ha="center", va="top", fontsize=SCALEBAR_FONT_SIZE)
    ax.text(x_corner + 0.012*xrange, y_base + sb_amp/2, amp_label(sb_amp),
            ha="left", va="center", fontsize=SCALEBAR_FONT_SIZE)


def _estimate_fs_from_time(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    dt = np.median(np.diff(t[np.isfinite(t)]))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Non-increasing or invalid time vector; cannot estimate sampling rate.")
    fs_candidate = 1.0 / dt  # assumes seconds
    if fs_candidate < 1000.0:
        return 1000.0 / dt  # treat t as milliseconds -> convert to seconds
    return fs_candidate


def _is_time_in_ms(fs_est: float) -> bool:
    # Heuristic paired with _estimate_fs_from_time above
    return fs_est < 1000.0


def _baseline_from_last_window(t, y, window_sec, time_is_ms: bool, method="median"):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    xmax = float(np.nanmax(t))
    win = window_sec * (1000.0 if time_is_ms else 1.0)
    thresh = xmax - win
    idx = np.where(t >= thresh)[0]
    if idx.size < 1:  # fallback: use last 20% if window is empty
        n = len(t)
        idx = np.arange(max(0, int(0.8*n)), n)
    seg = y[idx]
    return (np.nanmedian(seg) if method == "median" else np.nanmean(seg))


def main():
    df = read_three_columns(CSV_PATH)
    t, y1, y2 = df["time"].to_numpy(), df["y1"].to_numpy(), df["y2"].to_numpy()

    # Convert to pA if original was nA
    y1 = y1 * 1000.0
    y2 = y2 * 1000.0

    # Sampling rate & time units
    fs = _estimate_fs_from_time(t)
    time_is_ms = _is_time_in_ms(fs)

    # Common x-range (for identical horizontal scalebar)
    xmin, xmax = float(np.nanmin(t)), float(np.nanmax(t))
    xrange = xmax - xmin if xmax > xmin else 1.0
    xm = 0.02 * xrange
    right_margin = RIGHT_MARGIN_FRAC * xrange

    # Prepare filtered signals if needed (we'll baseline these if configured)
    def process(y):
        y_f = bessel_lowpass_clampfit(y, fs=fs, cutoff_hz=FILTER_CUTOFF_HZ,
                                      order=FILTER_ORDER, initial=FILTER_INITIAL) if APPLY_FILTER else y
        # Baseline from last 30 s
        base = _baseline_from_last_window(t, y_f if BASELINE_AFTER_FILTER else y,
                                          BASELINE_WINDOW_SEC, time_is_ms, BASELINE_METHOD)
        return y_f - base  # shift so baseline is 0 pA

    y1_plot = process(y1)
    y2_plot = process(y2)

    # Lock y-lims to scalebar frame (constant visual size), centered at 0 pA
    if LOCK_Y_TO_SB:
        halfspan = Y_SCALE_MULT * float(FORCED_SB_AMP)
        ymin_fixed = float(Y_CENTER) - halfspan
        ymax_fixed = float(Y_CENTER) + halfspan
        yrange_fixed = ymax_fixed - ymin_fixed
    else:
        # Shared y-lims from (baseline-adjusted) plotted signals
        y_all = np.concatenate([y1_plot, y2_plot])
        ymin_fixed, ymax_fixed = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        yr = ymax_fixed - ymin_fixed if ymax_fixed > ymin_fixed else 1.0
        pad = 0.06 * yr
        ymin_fixed -= pad
        ymax_fixed += pad
        yrange_fixed = ymax_fixed - ymin_fixed

    # Plot each trace with identical axes so scalebar size is constant
    for y_plot, label in zip([y1_plot, y2_plot], ["y1", "y2"]):
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        ax.plot(t, y_plot, color="black", linewidth=LINEWIDTH)
        ax.set_axis_off()
        ax.set_xlim(xmin - xm, xmax + xm + right_margin)
        ax.set_ylim(ymin_fixed, ymax_fixed)

        add_scalebar(ax,
                     xmin=xmin, xmax=xmax,
                     ymin=ymin_fixed, ymax=ymax_fixed,
                     xrange=xrange, yrange=yrange_fixed,
                     right_margin=right_margin)

        outpath = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_{label}.png")
        fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
