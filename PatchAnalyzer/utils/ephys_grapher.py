from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# USER-TWEAKABLE CONSTANTS (kept same defaults)
# ─────────────────────────────────────────────────────────────────────────────
CM_SOURCE   = "VOLTAGE"   # "CURRENT" → keep CC Cm;  "VOLTAGE" → prefer VC Cm
V_CSV_PATH  = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\McEachin_SH-SY5Y_exp\v_McEachin_SH-SY5Y.csv")
CSV_PATH    = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\McEachin_SH-SY5Y_exp\c_McEachin_SH-SY5Y.csv")

BIN_STEP      = 2    # pA / pF – bin width for F–I curve
CM_RANGE      = (20, 500)  # pF      – keep cells with sensible Cm
TAU_MAX       = 2000        # ms      – drop rows with absurd τ
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
    # Zero-fill missing firing rates so zero-current sweeps plot as flat points instead of gaps
    df["FiringRate"] = pd.to_numeric(df["FiringRate"], errors="coerce").fillna(0.0)
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

# Toggle to True when you want to inspect per-cell current densities in the console
DEBUG_PRINT_CURRENT_BINS = True
if DEBUG_PRINT_CURRENT_BINS:
    for (grp, uid), grp_df in cc_df.groupby(["Group", "UID"]):
        norm_vals = np.round(np.sort(grp_df["I_norm"].unique()), 4)
        bin_vals = np.round(np.sort(grp_df["I_bin"].unique()), 4)
        print(f"[F-I DEBUG] Group {grp}, UID {uid}")
        print(f"  I_norm values: {norm_vals}")
        print(f"  I_bin values : {bin_vals}")
        print()

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

# --- Prism-like global cosmetics (font + line widths) ---
plt.rcParams.update({
    "font.family": "Arial",      # Prism commonly uses Arial by default
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.6,
})

fig = plt.figure(figsize=(14, 9), constrained_layout=True)
gs  = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[2, 1, 1])

# ―― PANEL C – F-I curve (GraphPad‑like rendering) ―――――――――――――――――――――――――
# Colors chosen to emulate Prism defaults
PRISM_MAGENTA = "#C65A8E"   # close to Prism “magenta”
PRISM_BLACK   = "#000000"

# GraphPad Prism user guide recommends bold axis lines, long tick marks, and capped error bars
# Source: GraphPad Prism 9 User Guide, "Format XY Graphs" (https://www.graphpad.com/guides/prism/latest/user-guide/)
PRISM_LINE_LW       = 2.4
PRISM_MARKER_SIZE   = 8.0
PRISM_MARKER_EDGE   = 2.1
PRISM_CAP_SIZE      = 5
PRISM_CAP_THICKNESS = 1.8
PRISM_TICK_LENGTH   = 8
PRISM_TICK_WIDTH    = 1.8
PRISM_TICK_PAD      = 6

def _prism_colour(grp: str) -> str:
    g = str(grp).upper()
    if "APP" in g:
        return PRISM_MAGENTA
    if any(tag in g for tag in ("CTRL", "CONTROL", "CTR", "CTL", "WT")):
        return PRISM_BLACK
    return colour_map.get(grp, "#333333")

# Optional prettified labels for legend (falls back to raw group if missing)
LEGEND_LABEL_MAP = {"CTRL": "Ctrl", "CONTROL": "Ctrl", "HAPP": "hAPP"}

PLOT_STEPS   = np.arange(0, 14, 2)              # keep your original display bins
total_cells  = cell_vals.groupby("Group")["UID"].nunique()   # overall n

axC = fig.add_subplot(gs[0, 2:4])

# Axis cosmetics (Prism-like): only left/bottom spines, thicker lines, outward ticks
for side in ("top", "right"):
    axC.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axC.spines[side].set_linewidth(PRISM_TICK_WIDTH)
axC.tick_params(direction="out", width=PRISM_TICK_WIDTH,
                length=PRISM_TICK_LENGTH, pad=PRISM_TICK_PAD)

handles, labels = [], []
for grp in GROUPS:
    rows = fi_stats[(fi_stats["Group"] == grp) &
                    (fi_stats["I_bin"].isin(PLOT_STEPS))].copy()
    if rows.empty:
        continue
    rows = rows.sort_values("I_bin")
    x = rows["I_bin"].to_numpy()
    y = rows["mean"].to_numpy()
    sem = rows["sem"].fillna(0.0).to_numpy()
    col = _prism_colour(grp)
    grp_upper = str(grp).upper()
    marker_face = col if "APP" in grp_upper else "white"
    eb = axC.errorbar(
        x, y, yerr=sem,
        fmt="o-", lw=PRISM_LINE_LW, ms=PRISM_MARKER_SIZE,
        mfc=marker_face, mec=col, mew=PRISM_MARKER_EDGE,
        color=col, elinewidth=PRISM_CAP_THICKNESS,
        capsize=PRISM_CAP_SIZE, capthick=PRISM_CAP_THICKNESS,
        solid_capstyle="round",
    )
    eb.lines[0].set_solid_joinstyle("round")
    for cap in eb[1]:
        cap.set_color(col)
        cap.set_linewidth(PRISM_CAP_THICKNESS)
    label_txt = f"{LEGEND_LABEL_MAP.get(str(grp).upper(), str(grp))}   n={int(total_cells.get(grp, 0))}"
    handles.append(eb.lines[0])   # show line+marker in legend
    labels.append(label_txt)

axC.set_xlabel("Current Density (pA/pF)")
axC.set_ylabel("AP Frequency (Hz)")
axC.set_xlim(0, 16)
# axC.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
axC.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
axC.set_ylim(0, 8)
axC.set_yticks([0, 2, 4, 6, 8])

# Place legend inside axes like Prism and remove frame
axC.legend(handles, labels, loc="upper left", frameon=False, fontsize=11,
           handlelength=1.4, handletextpad=0.8)

# # ―― Panel D – PASSIVE ―――――――――――――――――――――――――――――――――――――――――――――
# for idx, prm in enumerate(PASSIVE):
#     _boxpanel(fig.add_subplot(gs[1, idx]), prm, colour_map, GROUPS)

# # ―― Panel E – ACTIVE ―――――――――――――――――――――――――――――――――――――――――――――
# for idx, prm in enumerate(ACTIVE):
#     _boxpanel(fig.add_subplot(gs[2, idx]), prm, colour_map, GROUPS)

plt.show()
