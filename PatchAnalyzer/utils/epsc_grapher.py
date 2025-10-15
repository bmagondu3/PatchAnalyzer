import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

try:
    from .ephys_grapher2 import (
        _DEFAULT_SAVE_DIR,
        filter_3std,
        plot_box_with_scatter,
        set_save_dir_from,
        use_prism_style,
    )
except ImportError:
    import sys
    from pathlib import Path as _Path

    _file_path = _Path(__file__).resolve()
    _repo_root = _file_path.parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    try:
        from PatchAnalyzer.utils.ephys_grapher2 import (
            _DEFAULT_SAVE_DIR,
            filter_3std,
            plot_box_with_scatter,
            set_save_dir_from,
            use_prism_style,
        )
    except ImportError:
        from ephys_grapher2 import (
            _DEFAULT_SAVE_DIR,
            filter_3std,
            plot_box_with_scatter,
            set_save_dir_from,
            use_prism_style,
        )


# ---------- Helpers ----------
def ecdf_percent(x, ensure_last_point=True):
    """ECDF step data (x_sorted, y in %), guaranteed to reach 100%.

    If ensure_last_point=True we explicitly append the last (x, 100) to avoid any visual truncation
    due to step rendering or axis clipping.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    n = xs.size
    y = np.arange(1, n + 1) / n * 100.0
    if ensure_last_point and (xs.size > 0):
        # Ensure we have an explicit terminal point at 100%
        xs = np.concatenate([xs, [xs[-1]]])
        y = np.concatenate([y, [100.0]])
    return xs, y
def plot_ecdf(
    ctrl_values,
    exp_values,
    x_label="IEI (ms)",
    title="sEPSCs",
    ctrl_label="Ctrl",
    exp_label="Experimental",
    exp_color="#2ca02c",
    ctrl_color="black",
    xlim=(0, 1600),
    xticks=(0, 400,800,1200, 1600,2000,2400,2800),
    ylim=(0, 150),
    yticks=(0, 50, 100, 150),
    savepath=None,
    # NEW: outlier handling
    filter_outliers=True,
    center="mean",
):
    """Standalone ECDF plot (step) with optional ±3σ filtering applied per group.

    If filter_outliers=True, each group's ECDF is computed after a per-group ±3×STD filter.
    The CDF is guaranteed to hit 100%.
    """
    use_prism_style()
    fig = plt.figure(figsize=(6, 4.4))
    ax = fig.add_subplot(111)
    # Hide top/right spines for cleaner look
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)

    c = np.asarray(ctrl_values, float)
    e = np.asarray(exp_values, float)
    c = c[~np.isnan(c)]; e = e[~np.isnan(e)]
    if filter_outliers:
        c = filter_3std(c, center=center)
        e = filter_3std(e, center=center)

    xs_c, yp_c = ecdf_percent(c, ensure_last_point=True)
    xs_e, yp_e = ecdf_percent(e, ensure_last_point=True)

    # Guard empty
    if xs_c.size:
        ax.step(xs_c, yp_c, where="post", linewidth=3.0, color=ctrl_color, label=ctrl_label)
    if xs_e.size:
        ax.step(xs_e, yp_e, where="post", linewidth=3.0, color=exp_color, label=exp_label)

    ax.set_xlim(*xlim)
    # Auto-pick ticks from 1/2/5 × 10^n (→ 10s/100s/1000s) and end axis at the last tick
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, steps=[1, 2, 5, 10]))
    _xt = ax.get_xticks()
    if len(_xt):
        ax.set_xlim(ax.get_xlim()[0], float(_xt[-1]))

    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _yt = ax.get_yticks()
    if len(_yt):
        ax.set_ylim(ax.get_ylim()[0], float(_yt[-1]))

    ax.set_xlabel(x_label, labelpad=6)
    ax.set_ylabel("Relative frequency (%)", labelpad=6)
    ax.set_title(title, pad=12)
    leg = ax.legend(loc="lower right")
    if leg:
        # color legend text to match the lines (Ctrl first, then Experimental)
        for txt, col in zip(leg.get_texts(), [ctrl_color, exp_color]):
            txt.set_color(col)

    fig.tight_layout()
    if savepath:
        out_path = Path(savepath)
        if not out_path.is_absolute() and _DEFAULT_SAVE_DIR is not None:
            out_path = _DEFAULT_SAVE_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig, ax

def generate_synthetic_iei_event_trains(n_events_ctrl=1000, n_events_exp=1000, seed=7):
    rng = np.random.default_rng(seed)
    ctrl = rng.gamma(shape=2.2, scale=85, size=n_events_ctrl)
    exp  = rng.gamma(shape=2.2, scale=90, size=n_events_exp)
    return ctrl, exp

def generate_synthetic_cellwise_summary(n_ctrl=10, n_exp=11, seed=13):
    rng = np.random.default_rng(seed)
    ctrl_means = rng.normal(loc=180, scale=35, size=n_ctrl)
    exp_means  = rng.normal(loc=205, scale=40, size=n_exp)
    return ctrl_means, exp_means


# ---------- CSV parsing (real dataset) ----------
# NOTE: These functions are standalone and do NOT modify any existing plotting logic.
# They only prepare arrays for plot_ecdf() and plot_box_with_scatter().
import re
from typing import Callable, Dict, Optional

def _extract_numeric_id(filename: str) -> Optional[int]:
    """
    Return the last integer found in a filename (e.g., '629251656.xlsx' -> 629251656).
    If none is found, return None.
    """
    if not isinstance(filename, str):
        return None
    nums = re.findall(r"(\d+)", filename)
    return int(nums[-1]) if nums else None

def _default_is_experimental_prefix3(file_id: Optional[int], cutoff_prefix_inclusive: int = 629) -> bool:
    """
    Classify 'experimental' if the first 3 digits of the numeric id are strictly greater than 629.
    This matches your rule: after 629 -> (630..., 712..., 722...) are Experimental; 629..., 628..., 625... are Ctrl.
    """
    if file_id is None:
        return False  # Treat unknowns as control by default
    s = str(file_id)
    if len(s) < 3:
        return False
    return int(s[:3]) > cutoff_prefix_inclusive

def parse_interevent_csv(
    csv_path: str,
    filename_col: str = "file",
    sheet_col: str = "sheet",
    value_col: str = "event_value",
    mean_col: str = "mean_for_sheet",
    is_experimental: Optional[Callable[[Optional[int]], bool]] = None,
    # NEW: outlier handling for ECDF and box inputs
    filter_outliers=False,
    center="mean",
) -> Dict[str, object]:
    """
    Load the real dataset and split for ECDF and box/whisker.

    ECDF: uses every event_value per group (Control vs Experimental).
    Box/whisker: uses the provided per-sheet means (mean_for_sheet) and the sample
                 count equals the number of unique (filename, sheet) pairs per group.

    If filter_outliers=True, apply ±3σ per group separately **before** constructing the
    ECDF arrays and the sheet-level means arrays.

    Returns a dict with keys:
        'ecdf_ctrl', 'ecdf_exp'                   -> 1D float arrays of event-level values
        'box_ctrl_means', 'box_exp_means'        -> 1D float arrays of sheet-level means
        'n_ctrl', 'n_exp'                         -> ints (# unique sheets per group, post-filter if enabled)
        'sheet_means_df'                          -> DataFrame (optional inspection)
    """
    import pandas as pd  # local import keeps existing imports untouched

    # --- Load CSV ---
    df = pd.read_csv(csv_path)

    # Verify required columns; allow flexible second key ('sheet' or 'cell')
    if filename_col not in df.columns:
        raise KeyError(f"Required column '{filename_col}' not found in {csv_path}. Present: {list(df.columns)}")

    if sheet_col not in df.columns:
        # NEW: minimal compatibility layer — accept common aliases for the per-entity column
        for alt in ("cell", "Cell", "cell_id", "Cell_ID", "CellID"):
            if alt in df.columns:
                sheet_col = alt
                break
        else:
            raise KeyError(
                f"Required column '{sheet_col}' not found and no alias detected in {csv_path}. "
                f"Present: {list(df.columns)}"
            )


    # Choose a numeric event-values column for ECDF if the default is missing/non-numeric
    if value_col not in df.columns or not pd.api.types.is_numeric_dtype(df[value_col]):
        if "interevent_interval" in df.columns:
            value_col = "interevent_interval"
        else:
            raise KeyError(f"Could not locate a numeric event column. Tried '{value_col}'; have: {list(df.columns)}")

    # Ensure numeric types for selected columns
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # File id + group classification
    df["_file_id"] = df[filename_col].apply(_extract_numeric_id)
    if is_experimental is None:
        is_experimental = lambda fid: _default_is_experimental_prefix3(fid, cutoff_prefix_inclusive=629)
    df["_is_exp"] = df["_file_id"].apply(is_experimental)
    df["_group"]  = df["_is_exp"].map({True: "Experimental", False: "Ctrl"})

    # --- ECDF arrays (events per group) ---
    ecdf_ctrl_vals = df.loc[df["_group"] == "Ctrl", value_col].dropna().to_numpy(dtype=float)
    ecdf_exp_vals  = df.loc[df["_group"] == "Experimental", value_col].dropna().to_numpy(dtype=float)

    if filter_outliers:
        ecdf_ctrl_vals = filter_3std(ecdf_ctrl_vals, center=center)
        ecdf_exp_vals  = filter_3std(ecdf_exp_vals, center=center)

    # --- Box/whisker: per-sheet means ---
    # Prefer a precomputed mean column if present (old pipeline),
    # otherwise compute the mean from event-level columns (new analyzer outputs).
    if mean_col in df.columns and pd.api.types.is_numeric_dtype(df[mean_col]):
        sheet_means = (
            df.dropna(subset=[mean_col])
              .groupby([filename_col, sheet_col], as_index=False)[mean_col]
              .first()
              .rename(columns={mean_col: "mean_for_sheet"})
        )
    else:
        # Map mean_col like "mean_instantaneous_frequency" -> event column "instantaneous_frequency"
        event_for_mean = None
        if isinstance(mean_col, str) and mean_col.startswith("mean_"):
            candidate = mean_col[len("mean_"):]
            if candidate in df.columns:
                event_for_mean = candidate
        if event_for_mean is None:
            # Default to interevent interval if nothing else is specified/found
            for cand in ("interevent_interval", "instantaneous_frequency", "peak_amplitude_abs"):
                if cand in df.columns:
                    event_for_mean = cand; break
        if event_for_mean is None:
            raise KeyError(f"Cannot infer a column to average for per-sheet means. Available: {list(df.columns)}")
        sheet_means = (
            df.dropna(subset=[event_for_mean])
              .groupby([filename_col, sheet_col], as_index=False)[event_for_mean]
              .mean()
              .rename(columns={event_for_mean: "mean_for_sheet"})
        )

    # Add group label (based on filename) to the sheet-level table
    sheet_means["_file_id"] = sheet_means[filename_col].apply(_extract_numeric_id)
    sheet_means["_is_exp"]  = sheet_means["_file_id"].apply(is_experimental)
    sheet_means["_group"]   = sheet_means["_is_exp"].map({True: "Experimental", False: "Ctrl"})

    box_ctrl_means = sheet_means.loc[sheet_means["_group"] == "Ctrl", "mean_for_sheet"].to_numpy(dtype=float)
    box_exp_means  = sheet_means.loc[sheet_means["_group"] == "Experimental", "mean_for_sheet"].to_numpy(dtype=float)

    if filter_outliers:
        box_ctrl_means = filter_3std(box_ctrl_means, center=center)
        box_exp_means  = filter_3std(box_exp_means, center=center)

    n_ctrl = int(len(box_ctrl_means))
    n_exp  = int(len(box_exp_means))

    return {
        "ecdf_ctrl": ecdf_ctrl_vals,
        "ecdf_exp": ecdf_exp_vals,
        "box_ctrl_means": box_ctrl_means,
        "box_exp_means": box_exp_means,
        "n_ctrl": n_ctrl,
        "n_exp": n_exp,
        "sheet_means_df": sheet_means,
    }

if __name__ == "__main__":
    # data_path = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\sEPSCs_by_sheet.csv"
    data_path = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\stats\sEPSCs_by_cell.csv"
    set_save_dir_from(data_path)
    # Generate ECDF + box plots for IEI, instantaneous frequency, and peak amplitude
    configs = [
        {"tag": "IEI", "event_col": "interevent_interval", "label": "IEI (ms)"},
        {"tag": "InstFreq", "event_col": "instantaneous_frequency", "label": "Instantaneous frequency (Hz)"},
        {"tag": "PeakAmp", "event_col": "peak_amplitude_abs", "label": "Peak amplitude (abs)"}
    ]

    for cfg in configs:
        tag = cfg["tag"]; event_col = cfg["event_col"]; label = cfg["label"]
        data = parse_interevent_csv(
            data_path, value_col=event_col, mean_col=f"mean_{event_col}",
            filter_outliers=True, center="mean"
        )
        # Dynamic axis scaling for readability (p99 headroom)
        import numpy as _np
        all_events = _np.concatenate([data["ecdf_ctrl"], data["ecdf_exp"]]) if len(data["ecdf_ctrl"]) or len(data["ecdf_exp"]) else _np.array([])
        if all_events.size:
            p99 = _np.nanpercentile(all_events, 99)
            xlim = (0, float(p99 * 1.05))
            xticks = list(_np.linspace(xlim[0], xlim[1], 6))
        else:
            xlim = (0, 1); xticks = [0, 0.5, 1.0]

        fig_ecdf, _ = plot_ecdf(
            data["ecdf_ctrl"], data["ecdf_exp"],
            x_label=label, title=f"sEPSCs {tag}",
            ctrl_label="Ctrl", exp_label="Exp",
            xlim=xlim, xticks=xticks,
            savepath=f"ecdf_{tag}.png",
            filter_outliers=True, center="mean"
        )

        # Compute y-axis for box plot based on per-sheet means (p95 headroom)
        all_means = _np.concatenate([data["box_ctrl_means"], data["box_exp_means"]]) if len(data["box_ctrl_means"]) or len(data["box_exp_means"]) else _np.array([])
        if all_means.size:
            p95 = _np.nanpercentile(all_means, 95)
            ymax = float(p95 * 1.15 if p95 > 0 else 1.0)
            ylim = (0, ymax)
            yticks = list(_np.linspace(ylim[0], ylim[1], 5))
        else:
            ylim = (0, 1); yticks = [0, 0.5, 1.0]

        fig_box, _, stats = plot_box_with_scatter(
            data["box_ctrl_means"], data["box_exp_means"],
            y_label=label, title=f"sEPSCs {tag}",
            ctrl_label=f"Ctrl (n={data['n_ctrl']})",
            exp_label=f"Exp (n={data['n_exp']})",
            ylim=ylim, yticks=yticks,
            savepath=f"box_{tag}.png",
            filter_outliers=True, center="mean", report_filtered_n=True
        )
        print(f"{tag}: Welch t={stats['t']:.3g}, df={stats['df']:.1f}, p={stats['p']:.3g}")
