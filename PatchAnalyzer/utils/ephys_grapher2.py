from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    from math import erf, sqrt


_DEFAULT_SAVE_DIR: Path | None = None


def set_save_dir_from(csv_path: str) -> None:
    """Make relative figure save paths resolve to the CSV's directory."""
    global _DEFAULT_SAVE_DIR
    _DEFAULT_SAVE_DIR = Path(csv_path).resolve().parent


def use_prism_style(font_family: str = "DejaVu Sans", base_size: int = 12, axis_linewidth: float = 2.4) -> None:
    """Apply a Prism-like style with heavier axes/ticks."""
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": base_size,
            "axes.linewidth": axis_linewidth,
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": axis_linewidth,
            "ytick.major.width": axis_linewidth,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.frameon": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def filter_3std(x: Iterable[float], center: str = "mean") -> np.ndarray:
    """Return values within +/- 3 * STD of the chosen center (mean or median)."""
    arr = np.asarray(list(x), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return arr
    if center == "median":
        mu = float(np.median(arr))
    else:
        mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if sigma == 0.0:
        return arr.copy()
    lower, upper = mu - 3.0 * sigma, mu + 3.0 * sigma
    return arr[(arr >= lower) & (arr <= upper)]


def p_to_stars(p: float) -> str:
    """Map p-value to GraphPad-style asterisks."""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def welch_ttest(a: Iterable[float], b: Iterable[float]) -> Tuple[float, float, float]:
    """Welch's t-test (unequal variances). Returns t, df, p(two-sided)."""
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if _HAVE_SCIPY:
        t_stat, p_val = _stats.ttest_ind(
            arr_a,
            arr_b,
            equal_var=False,
            nan_policy="omit",
            alternative="two-sided",
        )
        var_a = np.var(arr_a, ddof=1)
        var_b = np.var(arr_b, ddof=1)
        n_a = len(arr_a)
        n_b = len(arr_b)
        df = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a**2) / (n_a**2 * (n_a - 1)) + (var_b**2) / (n_b**2 * (n_b - 1))
        )
        return float(t_stat), float(df), float(p_val)

    na, nb = len(arr_a), len(arr_b)
    mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))
    se = np.sqrt(var_a / na + var_b / nb)
    t_stat = (mean_a - mean_b) / se if se != 0 else 0.0
    df = (
        (var_a / na + var_b / nb) ** 2
        / ((var_a**2) / (na**2 * (na - 1)) + (var_b**2) / (nb**2 * (nb - 1)))
        if na > 1 and nb > 1
        else np.inf
    )
    z = abs(t_stat)
    phi = (1 + erf(z / sqrt(2))) / 2.0
    p_val = 2 * (1 - phi)
    return float(t_stat), float(df), float(p_val)


def _scatter_with_jitter(
    ax: plt.Axes,
    x_positions: Iterable[float],
    groups: Iterable[np.ndarray],
    colors: Iterable[str],
    jitter: float = 0.08,
    size_pts2: float = 45,
    edge_lw: float = 1.4,
) -> None:
    """Jittered open-circle points. size_pts2 is Matplotlib scatter 's' (points^2)."""
    rng = np.random.default_rng(42)
    for x, group_values, color in zip(x_positions, groups, colors):
        group_array = np.asarray(group_values, dtype=float)
        if group_array.size == 0:
            continue
        jittered = x + (rng.random(group_array.size) - 0.5) * 2 * jitter
        ax.scatter(
            jittered,
            group_array,
            s=size_pts2,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=edge_lw,
            zorder=3,
        )


def _draw_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=2.2, clip_on=False)
    dy = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.text((x1 + x2) / 2, y + h + 0.02 * dy, text, ha="center", va="bottom", fontsize=13)


def plot_box_with_scatter(
    ctrl_summary_values: Iterable[float],
    exp_summary_values: Iterable[float],
    y_label: str = "IEI (ms)",
    title: str = "sEPSCs",
    ctrl_label: str = "Ctrl",
    exp_label: str = "Experimental",
    ctrl_color: str = "#2ca02c",
    exp_color: str = "black",
    ylim: Tuple[float, float] = (0, 400),
    yticks: Tuple[float, ...] = (0, 400, 800, 1200, 1600, 2000),
    whisker_mode: str = "tukey",
    show_n: bool = True,
    savepath: str | None = None,
    filter_outliers: bool = True,
    center: str = "mean",
    report_filtered_n: bool = True,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, float]]:
    """Transparent boxes, open-circle dots, Welch t-test + bracket."""
    use_prism_style()
    fig = plt.figure(figsize=(3.8, 4.4))
    ax = fig.add_subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ctrl_vals = np.asarray(list(ctrl_summary_values), dtype=float)
    exp_vals = np.asarray(list(exp_summary_values), dtype=float)
    ctrl_vals = ctrl_vals[~np.isnan(ctrl_vals)]
    exp_vals = exp_vals[~np.isnan(exp_vals)]

    if filter_outliers:
        ctrl_vals = filter_3std(ctrl_vals, center=center)
        exp_vals = filter_3std(exp_vals, center=center)

    data = [ctrl_vals, exp_vals]
    colors = [ctrl_color, exp_color]
    whis = (0, 100) if whisker_mode == "minmax" else 1.5

    bp = ax.boxplot(
        data,
        whis=whis,
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="black", linewidth=3.0),
        boxprops=dict(linewidth=2.6, facecolor="none"),
        whiskerprops=dict(linewidth=2.4, color="black"),
        capprops=dict(linewidth=2.4, color="black"),
        flierprops=dict(
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor="black",
            alpha=1.0,
        ),
    )

    for idx, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor("none")
        patch.set_edgecolor(color)
        bp["medians"][idx].set_color(color)
        bp["medians"][idx].set_linewidth(3.0)
        for line in bp["whiskers"][2 * idx : 2 * idx + 2]:
            line.set_color(color)
            line.set_linewidth(2.4)
        for line in bp["caps"][2 * idx : 2 * idx + 2]:
            line.set_color(color)
            line.set_linewidth(2.4)

    for idx, flier in enumerate(bp["fliers"]):
        flier.set_marker("o")
        flier.set_markersize(6.5)
        flier.set_markerfacecolor("white")
        flier.set_markeredgecolor(colors[idx])
        flier.set_alpha(1.0)
        flier.set_linestyle("none")

    _scatter_with_jitter(ax, [1, 2], data, colors, jitter=0.08, size_pts2=45, edge_lw=1.4)

    ax.set_xlim(0.4, 2.6)
    ax.set_xticks([1, 2], labels=[ctrl_label, exp_label])
    labels = ax.get_xticklabels()
    if len(labels) >= 2:
        labels[0].set_color(ctrl_color)
        labels[1].set_color(exp_color)

    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ticks = ax.get_yticks()
    if len(ticks):
        lower, _upper = ax.get_ylim()
        ax.set_ylim(lower, float(ticks[-1]))
    ax.set_ylabel(y_label, labelpad=6)
    ax.set_title(title, pad=40)

    if show_n:
        n_ctrl, n_exp = len(data[0]), len(data[1])
        trans = ax.get_xaxis_transform()
        baseline = -0.12
        if filter_outliers and report_filtered_n:
            ax.text(
                1,
                baseline,
                f"n={n_ctrl} (3SD)",
                ha="center",
                va="top",
                transform=trans,
                clip_on=False,
                color=ctrl_color,
            )
            ax.text(
                2,
                baseline,
                f"n={n_exp} (3SD)",
                ha="center",
                va="top",
                transform=trans,
                clip_on=False,
                color=exp_color,
            )
        else:
            ax.text(
                1,
                baseline,
                f"n={n_ctrl}",
                ha="center",
                va="top",
                transform=trans,
                clip_on=False,
                color=ctrl_color,
            )
            ax.text(
                2,
                baseline,
                f"n={n_exp}",
                ha="center",
                va="top",
                transform=trans,
                clip_on=False,
                color=exp_color,
            )

    if len(data[0]) > 0 and len(data[1]) > 0:
        t_stat, df, p_val = welch_ttest(data[0], data[1])
        stars = p_to_stars(p_val)
    else:
        t_stat = df = p_val = np.nan
        stars = "na"

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    has_data = [len(d) > 0 for d in data]
    ymax = (
        np.nanmax([np.max(d) if len(d) else np.nan for d in data])
        if any(has_data)
        else y_max
    )
    _draw_sig_bracket(ax, 1, 2, ymax + 0.06 * y_range, h=0.015 * y_range, text=stars)

    fig.tight_layout()
    if savepath:
        out_path = Path(savepath)
        if not out_path.is_absolute() and _DEFAULT_SAVE_DIR is not None:
            out_path = _DEFAULT_SAVE_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    return fig, ax, {"t": t_stat, "df": df, "p": p_val, "stars": stars}


# ---------------------------------------------------------------------------
# Configuration mirrors the current-clamp grapher defaults.
# ---------------------------------------------------------------------------
CM_SOURCE = "VOLTAGE"  # "CURRENT" keeps CC Cm; "VOLTAGE" prefers VC Cm
V_CSV_PATH = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\McEachin_SH-SY5Y_exp\results\v_McEachin_SH-SY5Y.csv"
)
CSV_PATH = Path(
    r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\McEachin_SH-SY5Y_exp\results\c_McEachin_SH-SY5Y.csv"
)

BIN_STEP = 0.5  # pA / pF - bin width for F-I curve (kept for parity)
CM_RANGE = (20, 500)  # pF - keep cells with sensible Cm
TAU_MAX = 200  # ms - drop rows with absurd tau
I_RATIO_MAX = 60  # pA / pF - truncate extreme x-axis values

PASSIVE_PARAMS = ["RMP", "Tau", "Rm", "Rm_VC", "Cm", "Ra"]
ACTIVE_PARAMS = ["APpeak", "APhwdt", "threshold", "dVdt"]
ALL_PARAMS: Tuple[str, ...] = tuple(PASSIVE_PARAMS + ACTIVE_PARAMS)


@dataclass(frozen=True)
class ParamSpec:
    title: str
    y_label: str
    filename: str


PARAM_SPECS: Dict[str, ParamSpec] = {
    "RMP": ParamSpec("Resting membrane potential", "Membrane potential (mV)", "box_RMP.png"),
    "Tau": ParamSpec("Membrane tau", "Tau (ms)", "box_Tau.png"),
    "Rm": ParamSpec("Input resistance (CC)", "Input resistance (MOhm)", "box_Rm_CC.png"),
    "Rm_VC": ParamSpec("Membrane resistance (VC)", "Membrane resistance (MOhm)", "box_Rm_VC.png"),
    "Cm": ParamSpec("Membrane capacitance", "Capacitance (pF)", "box_Cm.png"),
    "Ra": ParamSpec("Access resistance", "Access resistance (MOhm)", "box_Ra.png"),
    "APpeak": ParamSpec("AP peak", "AP peak (mV)", "box_APpeak.png"),
    "APhwdt": ParamSpec("AP half-width", "AP half-width (ms)", "box_APhwdt.png"),
    "threshold": ParamSpec("Spike threshold", "Threshold (mV)", "box_threshold.png"),
    "dVdt": ParamSpec("Max dV/dt", "Max dV/dt (mV/s)", "box_dVdt.png"),
}


# ---------------------------------------------------------------------------
# Data loading follows the original helper logic (unit fixes included).
# ---------------------------------------------------------------------------
def load_cc_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).rename(
        columns={
            "Cell(UniqueID)": "UID",
            "Group Label": "Group",
            "Injected current (pA)": "Iinj",
            "RMP mV": "RMP",
            "Tau ms": "Tau",
            "Rm (Mohms)": "Rm",
            "Cm (pF)": "Cm",
            "Firing rate (Hz)": "FiringRate",
            "mean AP peak mV": "APpeak",
            "mean AP hwdt(ms)": "APhwdt",
            "dV/dt max (mV/s)": "dVdt",
        }
    )
    df["Group"] = df["Group"].astype(str).str.upper()
    df["Rm"] = df["Rm"] * 100.0  # convert MOhm to the corrected scale
    df["Cm"] = df["Cm"] / 100.0  # convert pF to the corrected scale
    return df


def load_vc_data(v_csv_path: Path) -> pd.DataFrame:
    vdf = pd.read_csv(v_csv_path).rename(
        columns={
            "UID": "UID",
            "group_label": "Group",
            "mean_Ra_MOhm": "Ra_VC",
            "mean_Rm_MOhm": "Rm_VC",
            "mean_Cm_pF": "Cm_VC",
        }
    )
    vdf["Group"] = vdf["Group"].astype(str).str.upper()
    numeric_cols = ["Ra_VC", "Rm_VC", "Cm_VC"]
    for col in numeric_cols:
        vdf[col] = pd.to_numeric(vdf[col].astype(str).str.replace("'", "", regex=False), errors="coerce")

    ra_ohm = vdf["Ra_VC"] * 1e6
    rm_ohm = vdf["Rm_VC"] * 1e6
    r_parallel = 1.0 / (1.0 / ra_ohm + 1.0 / rm_ohm)
    cm_farads = vdf["Cm_VC"] * 1e-12
    vdf["Tau_VC"] = cm_farads * r_parallel * 1e3  # convert to ms
    return vdf[["UID", "Group", "Ra_VC", "Rm_VC", "Cm_VC", "Tau_VC"]]


# ---------------------------------------------------------------------------
# Processing and aggregation.
# ---------------------------------------------------------------------------
def prepare_cell_means(
    cc_csv: Path,
    vc_csv: Path,
    cm_source: str = CM_SOURCE,
    cm_range: Tuple[float, float] = CM_RANGE,
    tau_max: float = TAU_MAX,
    i_ratio_max: float = I_RATIO_MAX,
    bin_step: float = BIN_STEP,
) -> Tuple[pd.DataFrame, List[str]]:
    cc_df = load_cc_data(cc_csv)
    vc_df = load_vc_data(vc_csv)

    vc_means = (
        vc_df.groupby(["UID", "Group"], as_index=False)[["Ra_VC", "Rm_VC", "Cm_VC", "Tau_VC"]]
        .mean()
    )
    cc_df = cc_df.merge(vc_means, on=["UID", "Group"], how="left")
    cc_df["Ra"] = cc_df["Ra_VC"]

    if cm_source.upper() == "VOLTAGE":
        cc_df["Cm_used"] = cc_df["Cm_VC"].fillna(cc_df["Cm"])
    else:
        cc_df["Cm_used"] = cc_df["Cm"]

    mask = cc_df["Cm"].between(*cm_range) & (cc_df["Tau"] <= tau_max)
    cc_df = cc_df.loc[mask].copy()

    cm_mean = cc_df.groupby("UID")["Cm_used"].transform("mean")
    cc_df["I_norm"] = cc_df["Iinj"] / cm_mean
    cc_df = cc_df[cc_df["I_norm"].between(-5.0, i_ratio_max)].copy()
    cc_df["I_bin"] = (cc_df["I_norm"] / bin_step).round() * bin_step

    cell_means = (
        cc_df.groupby(["UID", "Group"], as_index=False)[list(ALL_PARAMS)]
        .mean()
        .astype({param: float for param in ALL_PARAMS})
    )

    for param in ALL_PARAMS:
        mean_col = f"mean_{param}"
        cell_means[mean_col] = cell_means[param].astype(float)

    groups = sorted(cell_means["Group"].dropna().unique().tolist())
    return cell_means, groups


def _combined_values(arrays: Iterable[np.ndarray]) -> np.ndarray:
    stacked: List[np.ndarray] = []
    for arr in arrays:
        if arr.size:
            stacked.append(arr[~np.isnan(arr)])
    if not stacked:
        return np.empty(0)
    return np.concatenate(stacked)


def _compute_ylim(arrays: Iterable[np.ndarray]) -> Tuple[float, float]:
    combined = _combined_values(arrays)
    if combined.size == 0:
        return (-1.0, 1.0)
    data_min = float(np.nanmin(combined))
    data_max = float(np.nanmax(combined))
    if np.isclose(data_min, data_max, atol=1e-9):
        offset = max(abs(data_min) * 0.2, 1.0)
        return data_min - offset, data_max + offset
    padding = (data_max - data_min) * 0.12
    lower = data_min - padding
    upper = data_max + padding
    if data_min >= 0.0 and lower < 0.0:
        lower = 0.0
    return lower, upper


def _extract_group_arrays(cell_means: pd.DataFrame, param: str, groups: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups for plotting, received: {groups}")
    ctrl_group, exp_group = groups
    ctrl_vals = (
        cell_means.loc[cell_means["Group"] == ctrl_group, f"mean_{param}"]
        .dropna()
        .to_numpy(dtype=float)
    )
    exp_vals = (
        cell_means.loc[cell_means["Group"] == exp_group, f"mean_{param}"]
        .dropna()
        .to_numpy(dtype=float)
    )
    return ctrl_vals, exp_vals


def save_param_plot(
    cell_means: pd.DataFrame,
    groups: List[str],
    param: str,
    save_dir: Path | None = None,
) -> Dict[str, float]:
    ctrl_vals, exp_vals = _extract_group_arrays(cell_means, param, groups)
    spec = PARAM_SPECS[param]
    ylim = _compute_ylim([ctrl_vals, exp_vals])

    ctrl_label = f"{groups[0]} (n={len(ctrl_vals)})"
    exp_label = f"{groups[1]} (n={len(exp_vals)})"

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_target: str | Path = save_dir / spec.filename
    else:
        save_target = spec.filename

    fig, _, stats = plot_box_with_scatter(
        ctrl_vals,
        exp_vals,
        y_label=spec.y_label,
        title=spec.title,
        ctrl_label=ctrl_label,
        exp_label=exp_label,
        ylim=ylim,
        savepath=str(save_target),
        filter_outliers=True,
        center="mean",
        report_filtered_n=True,
    )

    plt.close(fig)
    print(f"{param}: Welch t={stats['t']:.3g}, df={stats['df']:.1f}, p={stats['p']:.3g}")
    return stats


def batch_plot_ephys_means(
    cc_csv: Path = CSV_PATH,
    vc_csv: Path = V_CSV_PATH,
    cm_source: str = CM_SOURCE,
    save_dir: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    cell_means, groups = prepare_cell_means(
        cc_csv=cc_csv,
        vc_csv=vc_csv,
        cm_source=cm_source,
        cm_range=CM_RANGE,
        tau_max=TAU_MAX,
        i_ratio_max=I_RATIO_MAX,
        bin_step=BIN_STEP,
    )

    if save_dir is None:
        set_save_dir_from(str(cc_csv))

    results: Dict[str, Dict[str, float]] = {}
    for param in ALL_PARAMS:
        stats = save_param_plot(
            cell_means=cell_means,
            groups=groups,
            param=param,
            save_dir=save_dir,
        )
        results[param] = stats
    return results


if __name__ == "__main__":
    batch_plot_ephys_means()
