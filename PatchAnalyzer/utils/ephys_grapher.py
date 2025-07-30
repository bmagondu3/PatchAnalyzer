
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION (tweak here only if necessary)
# ──────────────────────────────────────────────────────────────────
# ── CAPACITANCE SOURCE ───────────────────────────────────────────
#    "CURRENT"  – use Cm values from the current‑clamp CSV (default)
#    "VOLTAGE"  – replace each cell’s Cm with its average from VprotRowan.csv
CM_SOURCE   = "VOLTAGE"            # or "VOLTAGE"
V_CSV_PATH  = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\VprotRowan.csv")
CSV_PATH = Path(r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Rowan_GFP_TAU_exp\CprotRowan.csv")
BIN_STEP      = 0.5             # pA/pF bins for F–I curve (panel C)
CM_RANGE      = (20, 500)       # keep cells whose Cm∈[20,500] pF
TAU_MAX       = 200             # ms – drop rows with Tau > 200
I_RATIO_MAX   = 60              # pA/pF – x‑axis limit for panel C
DARK_GREEN    = "darkgreen"     # Ctrl / GFP
LIGHT_GREEN   = "lightgreen"    # Tau

# ──────────────────────────────────────────────────────────────────
# 1 │ LOAD  &  BASIC TIDY
# ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

df = df.rename(columns={
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
df["Group"] = df["Group"].str.upper()        # ‘GFP’, ‘TAU’, …

# ──────────────────────────────────────────────────────────────────
# OPTIONAL: replace Cm with voltage‑protocol averages
# ──────────────────────────────────────────────────────────────────
if CM_SOURCE.upper() == "VOLTAGE":
    # 2a. Load the voltage‑protocol CSV
    vdf = pd.read_csv(V_CSV_PATH)

    # 2b. Ensure matching UID and Cm column names
    vdf = vdf.rename(columns={
        "Cell(UniqueID)": "UID",
        "Cm (pF)"       : "mean_Cm_pF"
    })

    # 2c. Average Cm per cell from the voltage‑protocol file
    cm_v_avg = vdf.groupby("UID")["mean_Cm_pF"].mean().reset_index()

    # 2d. Merge those averages into the current‑clamp dataframe
    df = df.merge(cm_v_avg, on="UID", how="left")

    # 2e. Choose voltage Cm when available, otherwise fall back
    df["Cm_used"] = df["mean_Cm_pF"].fillna(df["Cm"])
else:
    # default: keep using Cm from the current‑clamp file
    df["Cm_used"] = df["Cm"]

# ──────────────────────────────────────────────────────────────────
# 2 │ REMOVE CLEARLY BAD ROWS  (fixes wild axes)
# ──────────────────────────────────────────────────────────────────
df = df[
    (df["Cm_used"].between(*CM_RANGE)) &   # sensible capacitance
    (df["Tau"]     <= TAU_MAX)             # drop huge τ artefacts
].copy()

# ──────────────────────────────────────────────────────────────────
# 3 │ NORMALISE CURRENT WITH PER‑CELL Cm (AVERAGED)
# ──────────────────────────────────────────────────────────────────
cm_mean = df.groupby("UID")["Cm_used"].transform("mean")
df["I_norm"] = df["Iinj"] / cm_mean
df["I_bin"]  = (df["I_norm"] / BIN_STEP).round() * BIN_STEP

# limit to a practical x‑range for the plot
df = df[df["I_norm"].between(-5, I_RATIO_MAX)]


# ──────────────────────────────────────────────────────────────────
# 4 │ BUILD DATA FOR PANEL C  (mean‑of‑means ± 95 % CI)
# ──────────────────────────────────────────────────────────────────
cell_avg = (
    df.groupby(["UID", "Group", "I_bin"], as_index=False)["FiringRate"]
      .mean()
)

fi_stats = (
    cell_avg.groupby(["Group", "I_bin"])["FiringRate"]
            .agg(mean="mean", n="count", sd="std").reset_index()
)
fi_stats["sem"]   = fi_stats["sd"] / np.sqrt(fi_stats["n"])
fi_stats["ci_lo"] = fi_stats["mean"] - 1.96 * fi_stats["sem"]
fi_stats["ci_hi"] = fi_stats["mean"] + 1.96 * fi_stats["sem"]

max_I   = fi_stats["I_bin"].max()
xlim_up = max_I * 1.1

# ──────────────────────────────────────────────────────────────────
# 5 │   DATA FOR BOX PLOTS (panels D & E)
# ──────────────────────────────────────────────────────────────────
NUM_COLS  = ["RMP","Tau","Rm","Cm","APpeak","APhwdt","threshold","dVdt"]
cell_vals = (
    df.groupby(["UID", "Group"], as_index=False)[NUM_COLS]
      .mean()
)

cell_counts = cell_vals.groupby("Group")["UID"].nunique().to_dict()

PASSIVE = ["RMP","Tau","Rm","Cm"]
ACTIVE  = ["APpeak","APhwdt","threshold","dVdt"]

def set_whisker_ylim(ax, series):
    """Tight y‑range = [Q₁−1.5·IQR, Q₃+1.5·IQR] to avoid crazy axes."""
    q1, q3 = np.percentile(series.dropna(), [25, 75])
    iqr    = q3 - q1
    ax.set_ylim(q1 - 1.5*iqr, q3 + 1.5*iqr)

# ──────────────────────────────────────────────────────────────────
# 6 │   PLOT!   (row 0 = C,  row 1 = D,  row 2 = E)
# ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), constrained_layout=True)
gs  = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[2,1,1])

# ── PANEL C (split: points ±SD  |  line ±CI) ──────────────────────────
# reserve two axes: left (col 0‑1) and right (col 2‑3)
axC_pt  = fig.add_subplot(gs[0, 2:4])   # points ±SD
# axC_ln  = fig.add_subplot(gs[0, 2:4])   # line ±CI

# --- compute 110 % x‑limit once, reuse for both axes ------------------
max_I   = fi_stats["I_bin"].max()
xlim_up = max_I * 1.1

# --- sample counts for legend ----------------------------------------
cell_counts = cell_vals.groupby("Group")["UID"].nunique().to_dict()

# ---------------------------------------------------------------------
# C‑left: discrete points (–2 to 30 pA/pF in 2‑step bins)  ± SD
# ---------------------------------------------------------------------
steps = np.arange(-2, 32, 2)          # –2, 0, 2, … 30
for grp, colour in zip(["GFP","TAU"], [DARK_GREEN, LIGHT_GREEN]):
    rows = fi_stats[fi_stats["Group"] == grp]
    # keep only the requested bins
    rows = rows[rows["I_bin"].isin(steps)]
    x    = rows["I_bin"].to_numpy(float)
    y    = rows["mean"].to_numpy(float)
    sd   = rows["sd"].to_numpy(float)
    n    = cell_counts.get(grp, 0)
    axC_pt.errorbar(x, y, yerr=sd,
                    fmt='o', capsize=4, markersize=5,
                    color=colour, label=f"{grp} (n={n})")

axC_pt.set_title("C‑left. Mean ± SD (2 pA/pF steps)", fontweight="bold", fontsize=10)
axC_pt.set_xlabel("Injected current / capacitance  (pA per pF)")
axC_pt.set_ylabel("Mean firing frequency  (Hz)")
axC_pt.set_xlim(-4, 32)
axC_pt.legend(frameon=False)

# ---------------------------------------------------------------------
# C‑right: existing continuous line  ± 95 % CI
# ---------------------------------------------------------------------
# for grp, colour in zip(["GFP","TAU"], [DARK_GREEN, LIGHT_GREEN]):
#     sub = fi_stats[fi_stats["Group"] == grp].dropna(subset=["mean"])
#     x   = sub["I_bin"].to_numpy(float)
#     y   = sub["mean"].to_numpy(float)
#     lo  = sub["ci_lo"].to_numpy(float)
#     hi  = sub["ci_hi"].to_numpy(float)
#     n   = cell_counts.get(grp, 0)
#     axC_ln.plot(x, y,
#                 color=colour,
#                 label=f"{grp} (n={n})")
#     axC_ln.fill_between(x, lo, hi,
#                         color=colour,
#                         alpha=0.25)

# axC_ln.set_title("C‑right. Mean ± 95 % CI (continuous)", fontweight="bold", fontsize=10)
# axC_ln.set_xlabel("Injected current / capacitance  (pA per pF)")
# axC_ln.set_ylabel("Mean firing frequency  (Hz)")
# axC_ln.set_xlim(0, xlim_up)
# axC_ln.legend(frameon=False)

# 0) put this once near the top (so jitter is repeatable)
rng = np.random.default_rng(42)          # reproducible jitter
# ── Panel D  (4 separate axes) ───────────
for i, prm in enumerate(PASSIVE):
    ax = fig.add_subplot(gs[1, i])

    for grp, colour, xpos in zip(["GFP", "TAU"],
                                 [DARK_GREEN, LIGHT_GREEN],
                                 [0, 1]):
        vals = cell_vals.loc[cell_vals["Group"] == grp, prm].dropna()

        # box‐and‐whisker, but leave boxes transparent
        bp = ax.boxplot(vals,
                        positions=[xpos],
                        widths=0.55,
                        patch_artist=True,
                        medianprops=dict(color="black"))
        bp["boxes"][0].set_facecolor('none')    # ← clear box
        for part in ("whiskers", "caps"):
            plt.setp(bp[part], color="black")

        # overlay coloured dots
        jitter = rng.normal(0, 0.08, len(vals))
        ax.scatter(xpos + jitter, vals,
                   s=25,
                   color=colour,          # coloured fill
                   edgecolor="black",
                   linewidths=0.3,
                   alpha=0.9)

    set_whisker_ylim(ax, cell_vals[prm])
    ax.set_xticks([0, 1]);  ax.set_xticklabels(["Ctrl", "Tau"])
    ax.set_title(prm, fontsize=10);  ax.tick_params(axis="y", labelsize=8)

# ── Panel E  (4 separate axes) ───────────
for i, prm in enumerate(ACTIVE):
    ax = fig.add_subplot(gs[2, i])

    for grp, colour, xpos in zip(["GFP", "TAU"],
                                 [DARK_GREEN, LIGHT_GREEN],
                                 [0, 1]):
        vals = cell_vals.loc[cell_vals["Group"] == grp, prm].dropna()

        bp = ax.boxplot(vals,
                        positions=[xpos],
                        widths=0.55,
                        patch_artist=True,
                        medianprops=dict(color="black"))
        bp["boxes"][0].set_facecolor('none')  # ← clear box
        for part in ("whiskers", "caps"):
            plt.setp(bp[part],  color="black")

        jitter = rng.normal(0, 0.08, size=len(vals))
        ax.scatter(xpos + jitter, vals,
                   s=25,
                   color=colour,
                   edgecolor="black",
                   linewidths=0.3,
                   alpha=0.8)

    set_whisker_ylim(ax, cell_vals[prm])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ctrl", "Tau"])
    ax.set_title(prm, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)

plt.show()
