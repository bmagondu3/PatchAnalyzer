from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# USER-TWEAKABLE CONSTANTS (kept same defaults)
# ─────────────────────────────────────────────────────────────────────────────
CM_SOURCE   = "VOLTAGE"   # "CURRENT" → keep CC Cm;  "VOLTAGE" → prefer VC Cm
V_CSV_PATH  = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Levi_control_cultured_exp\Cohort_2\v_levi_92525.csv")
CSV_PATH    = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Levi_control_cultured_exp\Cohort_2\c_levi_92525.csv")

BIN_STEP      = 0.5      # pA / pF – bin width for F–I curve
CM_RANGE      = (20, 500)  # pF      – keep cells with sensible Cm
TAU_MAX       = 200        # ms      – drop rows with absurd τ
I_RATIO_MAX   = 60         # pA / pF – truncate extreme x-axis values

# COLOUR SCHEME (legacy colors kept; dynamic palette used below)
DARK_GREEN  = "darkgreen"   # historical
LIGHT_GREEN = "lightgreen"  # historical
RED         = "#b30000"      # VC overlay

RNG = np.random.default_rng(42)       # reproducible jitter

# ─────────────────────────────────────────────────────────────────────────────
# 0 │ DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_cc_data(csv_path: Path) -> pd.DataFrame:
    """Load current-clamp summary CSV and apply unit fixes."""
    df = pd.read_csv(csv_path).rename(columns={
        "Cell(UniqueID)"       : "UID",
        "Group Label"          : "Group",
        "Injected current (pA)": "Iinj",
        "RMP mV"               : "RMP",
        "Tau ms"               : "Tau",
        "Rm (Mohms)"           : "Rm",
        "Cm (pF)"              : "Cm",
        "Firing rate (Hz)"     : "FiringRate",
        "mean AP peak mV"      : "APpeak",
        "mean AP hwdt(ms)"     : "APhwdt",
        "dV/dt max (mV/s)"     : "dVdt",
    })
    df["Group"] = df["Group"].astype(str).str.upper()
    # ----- unit-scaling requested -------------------------------------------
    df["Rm"] *= 100      # input-resistance ×100 → correct MΩ
    df["Cm"] /= 100     # capacitance ÷100     → correct pF
    return df

def load_vc_data(v_csv_path: Path) -> pd.DataFrame:
    """Load voltage-protocol summary CSV and compute τ."""
    vdf = pd.read_csv(v_csv_path).rename(columns={
        "UID"              : "UID",        # already correct in file
        "group_label"      : "Group",
        "mean_Ra_MOhm"     : "Ra_VC",
        "mean_Rm_MOhm"     : "Rm_VC",
        "mean_Cm_pF"       : "Cm_VC",
    })
    vdf["Group"] = vdf["Group"].astype(str).str.upper()
    numeric_cols = ["Ra_VC", "Rm_VC", "Cm_VC"]
    for col in numeric_cols:
        vdf[col] = pd.to_numeric(
            vdf[col].astype(str).str.replace("'", "", regex=False),
            errors="coerce",
        )
    # ----- τ  (τ = Cm × R_parallel) -----------------------------------------
    Ra_ohm = vdf["Ra_VC"] * 1e6
    Rm_ohm = vdf["Rm_VC"] * 1e6
    R_para = 1 / (1/Ra_ohm + 1/Rm_ohm)
    Cm_F   = vdf["Cm_VC"] * 1e-12
    vdf["Tau_VC"] = Cm_F * R_para * 1e3   # → ms
    return vdf[["UID", "Group", "Rm_VC", "Cm_VC", "Tau_VC"]]

# ─────────────────────────────────────────────────────────────────────────────
# Palette helper (readable categorical colors)
# ─────────────────────────────────────────────────────────────────────────────
def make_group_palette(groups):
    """Return a dict {group: color}, using a readable qualitative palette."""
    cmap = plt.get_cmap("Set2")  # designed for categorical data
    colors = [cmap(i % cmap.N) for i in range(len(groups))]
    return {g: c for g, c in zip(groups, colors)}

# ─────────────────────────────────────────────────────────────────────────────
# 1 │ PREPARE MASTER DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────
cc_df = load_cc_data(CSV_PATH)          # current-clamp summary (always)
vc_df = load_vc_data(V_CSV_PATH)        # voltage-protocol summary (always)

# Decide which Cm each row will use for the F–I normalisation
if CM_SOURCE.upper() == "VOLTAGE":
    # prefer voltage-protocol Cm when available
    cc_df = cc_df.merge(vc_df[["UID", "Cm_VC"]], on="UID", how="left")
    cc_df["Cm_used"] = cc_df["Cm_VC"].fillna(cc_df["Cm"])
else:  # "CURRENT"
    cc_df["Cm_used"] = cc_df["Cm"]

# ----- basic cleaning --------------------------------------------------------
mask  = (cc_df["Cm"].between(*CM_RANGE)) & (cc_df["Tau"] <= TAU_MAX)
cc_df = cc_df[mask].copy()

# ----- normalise current by per-cell mean Cm ---------------------------------
cm_mean = cc_df.groupby("UID")["Cm_used"].transform("mean")
cc_df["I_norm"] = cc_df["Iinj"] / cm_mean
cc_df = cc_df[cc_df["I_norm"].between(-5, I_RATIO_MAX)]
cc_df["I_bin"] = (cc_df["I_norm"] / BIN_STEP).round() * BIN_STEP

# Discover groups present and set colors
GROUPS = sorted(cc_df["Group"].dropna().unique().tolist())
colour_map = make_group_palette(GROUPS)

# ─────────────────────────────────────────────────────────────────────────────
# 2 │   F-I CURVE  (mean ± SD  + n)
# ─────────────────────────────────────────────────────────────────────────────
cell_avg = (
    cc_df.groupby(["UID", "Group", "I_bin"], as_index=False)["FiringRate"]
         .mean()
)
fi_stats = (
    cell_avg.groupby(["Group", "I_bin"])["FiringRate"]
            .agg(mean="mean", sd="std", n="count").reset_index()
)
fi_stats["sem"] = fi_stats["sd"] / np.sqrt(fi_stats["n"])

# ─────────────────────────────────────────────────────────────────────────────
# 3 │   PER-CELL PASSIVE / ACTIVE MEANS
# ─────────────────────────────────────────────────────────────────────────────
NUM_COLS = ["RMP", "Tau", "Rm", "Cm",
            "APpeak", "APhwdt", "threshold", "dVdt"]
cell_vals = (
    cc_df.groupby(["UID", "Group"], as_index=False)[NUM_COLS]
         .mean()
)

PASSIVE = ["RMP", "Tau", "Rm", "Cm"]
ACTIVE  = ["APpeak", "APhwdt", "threshold", "dVdt"]

# ----- VC overlay (group-level means) ----------------------------------------
vc_grp_mean = vc_df.groupby("Group").agg({
    "Rm_VC" : "mean",
    "Cm_VC" : "mean",
    "Tau_VC": "mean"
})

# ─────────────────────────────────────────────────────────────────────────────
# 4 │   PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), constrained_layout=True)
gs  = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[2, 1, 1])

# ―― PANEL C – F-I curve (group means ± SEM) ――――――――――――――――――――――――――――――
PLOT_STEPS   = np.arange(-2, 15,1)               # display only these bins
total_cells  = cell_vals.groupby("Group")["UID"].nunique()   # overall n

axC = fig.add_subplot(gs[0, 2:4])

for grp in GROUPS:
    rows = fi_stats[(fi_stats["Group"] == grp) &
                    (fi_stats["I_bin"].isin(PLOT_STEPS))].copy()
    if rows.empty:
        continue
    x, y, sem = rows["I_bin"], rows["mean"], rows["sem"].fillna(0.0)
    axC.errorbar(x, y, yerr=sem,
                 fmt="o", capsize=4, markersize=6,
                 color=colour_map[grp],
                 label=f"{grp} (n={int(total_cells.get(grp, 0))})")
    for xi, yi, ni in zip(x, y, rows["n"]):
        if np.isfinite(yi):
            axC.text(float(xi), float(yi),
                     f"n={int(ni)}",
                     color=colour_map[grp], fontsize=7, ha="center", va="bottom")

axC.set_xlabel("Injected current / capacitance  (pA per pF)")
axC.set_ylabel("Mean firing frequency  (Hz)")
axC.set_title("F–I curve  (group mean ± SEM)")
axC.set_xlim(PLOT_STEPS.min() - 1, PLOT_STEPS.max() + 1)
axC.grid(True, linestyle="--", alpha=.3)
axC.legend(frameon=False)

# ―― helper to tighten y-range to whiskers ――――――――――――――――――――――――――――――
def _tight_ylim(ax, data: pd.Series) -> None:
    data = pd.Series(data).dropna()
    if data.empty:
        return
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1 if q3 > q1 else (q1*0.1 if q1 != 0 else 1.0)
    ax.set_ylim(q1 - 1.5*iqr, q3 + 1.5*iqr)

# ―― helper to draw transparent box + jitter + VC overlay ――――――――――――――――
def _boxpanel(ax, param: str, colour_map: dict, groups: list) -> None:
    vc_label_added = False
    positions = list(range(len(groups)))
    for xpos, grp in enumerate(groups):
        vals = cell_vals.loc[cell_vals["Group"] == grp, param].dropna()
        if vals.empty:
            continue

        bp = ax.boxplot(vals, positions=[xpos], widths=.55,
                        patch_artist=True, medianprops=dict(color="black"))
        bp["boxes"][0].set_facecolor("none")
        for part in ("whiskers", "caps"):
            plt.setp(bp[part], color="black")

        jitter = RNG.normal(0, .08, len(vals))
        ax.scatter(xpos + jitter, vals, s=25,
                   color=colour_map[grp], edgecolors="black",
                   linewidths=.3, alpha=.9)

        # VC overlay (single red diamond)
        if param in ("Rm", "Cm", "Tau") and (grp in vc_grp_mean.index):
            overlay_val = {
                "Rm" : vc_grp_mean.loc[grp, "Rm_VC"],
                "Cm" : vc_grp_mean.loc[grp, "Cm_VC"],
                "Tau": vc_grp_mean.loc[grp, "Tau_VC"],
            }[param]
            ax.scatter(xpos, overlay_val, marker="D", s=70,
                       color=RED, zorder=5,
                       label=None if vc_label_added else "VC mean")
            vc_label_added = True

    _tight_ylim(ax, cell_vals[param])
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel({
        # passive
        "RMP"      : "mV",
        "Tau"      : "ms",
        "Rm"       : "MΩ",
        "Cm"       : "pF",
        # active (Panel E)
        "APpeak"   : "mV",
        "APhwdt"   : "ms",
        "threshold": "mV",
        "dVdt"     : "mV/s",
    }.get(param, ""))
    ax.set_title(param, fontsize=10)
    ax.grid(True, linestyle="--", alpha=.3)
    if not ax.legend_:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(frameon=False, fontsize=8)

# ―― Panel D – PASSIVE ―――――――――――――――――――――――――――――――――――――――――――――
for idx, prm in enumerate(PASSIVE):
    _boxpanel(fig.add_subplot(gs[1, idx]), prm, colour_map, GROUPS)

# ―― Panel E – ACTIVE ―――――――――――――――――――――――――――――――――――――――――――――
for idx, prm in enumerate(ACTIVE):
    _boxpanel(fig.add_subplot(gs[2, idx]), prm, colour_map, GROUPS)

plt.show()
