import os
import glob
import pandas as pd
import numpy as np
from pyabf.abfWriter import writeABF1

# --- conversion constants ---
C_CLAMP_AMP_PER_VOLT = 400 * 1e-12  # 400 pA per V (DAQ output)
C_CLAMP_VOLT_PER_VOLT = (1 * 1e-3) / (1e-3)  # 10 mV per V (DAQ input)
V_CLAMP_VOLT_PER_VOLT = (1 * 1e-3)  # 1 mV per V (DAQ output)
V_CLAMP_VOLT_PER_AMP = (1 * 1e-12)    # 1 mV per pA (DAQ input)

prottype = "holding"
num_channels = 1

# Folders
# csv_folder = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Test\2025_06_28-14_16\HoldingProtocol\splits"
csv_folder = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Test\2025_06_28-14_16\CurrentProtocol\temp"
# csv_folder = r"c:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\Data\Test\2025_06_28-14_16\VoltageProtocol\temp"
out_folder = csv_folder
os.makedirs(out_folder, exist_ok=True)

# 1) find & group CSVs by the text before the first '#'
all_csvs = glob.glob(os.path.join(csv_folder, "*.csv"))
groups = {}
for path in all_csvs:
    prefix = os.path.basename(path).split("#", 1)[0]
    groups.setdefault(prefix, []).append(path)

# 2) process each group
for prefix, paths in groups.items():
    paths.sort()
    raws_input = []
    raws_output = []
    times = None
    sample_rate = None

    for p in paths:
        df = pd.read_csv(
            p,
            sep=r"\s+", header=None,
            names=["time_s", "raw_current", "raw_voltage"],
            engine="python"
        )
        if times is None:
            times = df["time_s"].to_numpy()
            dt = times[1] - times[0]
            sample_rate = 1.0 / dt
        else:
            assert len(df) == len(times), f"Length mismatch in {p}"

        raws_input.append(df["raw_current"].to_numpy())
        raws_output.append(df["raw_voltage"].to_numpy())

    if prottype == "current":
        inputs_pa = [r *  C_CLAMP_AMP_PER_VOLT * 1e12 for r in raws_input]   # pA
        outputs_mv = [r * C_CLAMP_VOLT_PER_VOLT * 1e3 for r in raws_output] # mV
        out_state = "(voltage in mV)"
        in_state = "(current in pA)"
        C_stack = np.vstack(inputs_pa)
        R_stack = np.vstack(outputs_mv)
        c_unit = "pA"
        r_unit = "mV"

    elif prottype == "voltage":
        # inputs_mv = raws_input*1e3 # mV
        # outputs_pa = raws_output*1e9  # pA
        inputs_mV = [r / V_CLAMP_VOLT_PER_VOLT for r in raws_input] # V
        outputs_pa = [r / V_CLAMP_VOLT_PER_AMP for r in raws_output] # A
        in_state = "(voltage in mV)"
        out_state = "(current in pA)"
        C_stack = np.vstack(inputs_mV)
        R_stack = np.vstack(outputs_pa)
        c_unit = "mV"
        r_unit = "pA"
        # data converted already.
    
    elif prottype == "holding":
         inputs_mV = raws_output
         outputs_pa = raws_input
         inputs_mV = [r/V_CLAMP_VOLT_PER_VOLT for r in raws_output]
         outputs_pa = [r * 1000 for r in raws_input]
         in_state = "(voltage in mV)"
         out_state = "(current in pA)"
         C_stack = np.vstack(inputs_mV)
         R_stack = np.vstack(outputs_pa)
         c_unit = "mV"
         r_unit = "pA"
         # for some reason, the command is the third column on holding protocol data


if num_channels == 1:
        # write current ABF1 as *_Command.abf
        cmd_abf = os.path.join(out_folder, f"{prefix.rstrip('_')}_Command.abf")
        writeABF1(C_stack, cmd_abf, sample_rate, units=c_unit)
        print(f"Wrote {len(C_stack)} sweeps → {os.path.basename(cmd_abf)} {in_state}")

        # write voltage ABF1 as *_Response.abf
        resp_abf = os.path.join(out_folder, f"{prefix.rstrip('_')}_Response.abf")
        writeABF1(R_stack, resp_abf, sample_rate, units=r_unit)
        print(f"Wrote {len(R_stack)} sweeps → {os.path.basename(resp_abf)} {out_state}")
