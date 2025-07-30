"""
CSV ➜ ABF (two channels, ABF‑v1)
--------------------------------
• keeps your filename grouping PREFIX#NN.csv
• retains the exact scaling factors you specified
• writes one file per group:  CH0 = current (pA), CH1 = voltage (mV)
"""

import os
import glob
import struct
import pandas as pd
import numpy as np
from pyabf.abfWriter import writeABF1         # still available if you need 1‑ch writers

# ── conversion constants (unchanged) ───────────────────────────────────────────
C_CLAMP_AMP_PER_VOLT  = 400e-12      # 400 pA per DAQ‑V  (current path)
C_CLAMP_VOLT_PER_VOLT = 1000e-3      # 1000 mV per DAQ‑V (voltage path)

# ── I/O folders ───────────────────────────────────────────────────────────────
csv_folder = r"C:\Users\sa-forest\Documents\GitHub\PatchAnalyzer\testcode\CurrentProtocol"
out_folder = csv_folder
os.makedirs(out_folder, exist_ok=True)

# ── helper: ABF1 writer for TWO ADC channels ──────────────────────────────────
def writeABF1_dual(I_stack, V_stack, filename, sample_rate_hz):
    """
    Save current (pA) and voltage (mV) sweeps into a 2‑channel ABF‑v1 file
    compatible with Clampfit and pyABF.

    Parameters
    ----------
    I_stack, V_stack : np.ndarray  (nSweeps, nPoints)
        Current and voltage sweeps, already in physical units.
    filename         : str
        Output *.abf* file path.
    sample_rate_hz   : float
        Sampling frequency (Hz).
    """
    import struct
    import numpy as np

    # ───── basic checks ──────────────────────────────────────────────────
    assert I_stack.shape == V_stack.shape, "current and voltage arrays differ"
    sweeps, pts = I_stack.shape
    nCH         = 2
    BLOCK       = 512          # ABF1 fixed block size
    BPP         = 2            # int16

    # ───── interleave CH0,CH1,CH0,CH1 for every sweep ───────────────────
    inter = np.empty((sweeps, pts * nCH), dtype=np.float32)
    inter[:, 0::2] = I_stack        # channel 0 (current)
    inter[:, 1::2] = V_stack        # channel 1 (voltage)

    total_pts = inter.size
    data_blocks = int(np.ceil(total_pts * BPP / BLOCK))
    buf = bytearray((4 + data_blocks) * BLOCK)   # 4 header blocks + data
    pk  = struct.pack_into

    # ────── HEADER (field offsets follow pyABF 2.3.8 writer) ─────────────
    pk('4s', buf,   0, b'ABF ')      # FileSignature
    pk('f' , buf,   4, 1.83)         # fFileVersionNumber  (≥1.80 mandatory)
    pk('h' , buf,   8, 5)            # nOperationMode (5 = episodic)
    pk('i' , buf,  10, total_pts)    # lActualAcqLength   (all samples)
    pk('i' , buf,  16, sweeps)       # lActualEpisodes    (sweeps)
    pk('i' , buf,  40, 4)            # lDataSectionPtr    (block index)
    pk('h' , buf, 100, 0)            # nDataFormat (0=int16)
    pk('h' , buf, 120, nCH)          # nADCNumChannels
    pk('f' , buf, 122, 1e6 / sample_rate_hz)  # fADCSampleInterval (µs)
    pk('i' , buf, 138, pts)          # lNumSamplesPerEpisode

    # Resolution / range (matches official writer)
    pk('f', buf, 244, 10.0)          # fADCRange (±10 V DAQ)
    pk('i', buf, 252, 32768)         # lADCResolution (16‑bit)

    # ───── choose INT16 scaling so nothing clips ────────────────────────
    vmax       = float(np.abs(inter).max()) or 1.0
    INT_SCALE  = 32767 / vmax        # counts per physical‑unit

    # Clampfit/pyABF converts counts to physical by:
    #  gain = fInstrScale * fADCRange / lADCResolution
    # We want gain == 1/INT_SCALE  →  fInstrScale = (1/INT_SCALE)*32768/10
    instr_scale = (1.0 / INT_SCALE) * 32768.0 / 10.0

    for ch in range(16):            # populate all 16 possible channels
        pk('f', buf, 922  + ch*4, instr_scale if ch < 2 else 1.0)  # fInstrScale
        pk('f', buf, 1050 + ch*4, 1.0)                             # fSignalGain
        pk('f', buf, 730  + ch*4, 1.0)                             # fADCProgGain

    # channel unit strings
    pk('8s', buf, 602, b'pA')       # CH0 units
    pk('8s', buf, 610, b'mV')       # CH1 units

    # ───── convert data to int16 and write into buffer ──────────────────
    inter_i16 = np.round(inter * INT_SCALE).astype(np.int16)
    base = 4 * BLOCK
    for s, sweep in enumerate(inter_i16):
        offset = base + s * pts * nCH * BPP
        pk('<' + 'h'*len(sweep), buf, offset, *sweep)

    # ───── save file ────────────────────────────────────────────────────
    with open(filename, 'wb') as fh:
        fh.write(buf)


# ── 1) group CSVs by prefix ───────────────────────────────────────────────────
all_csvs = glob.glob(os.path.join(csv_folder, "*.csv"))
groups = {}
for path in all_csvs:
    prefix = os.path.basename(path).split("#", 1)[0]
    groups.setdefault(prefix, []).append(path)

# ── 2) process each group ─────────────────────────────────────────────────────
for prefix, paths in groups.items():
    paths.sort()
    raws_current, raws_voltage = [], []
    times, sample_rate = None, None

    for p in paths:
        df = pd.read_csv(
            p, sep=r"\s+", header=None,
            names=["time_s", "raw_current", "raw_voltage"],
            engine="python"
        )
        if times is None:
            times = df["time_s"].to_numpy()
            dt = times[1] - times[0]
            sample_rate = 1.0 / dt
        else:
            assert len(df) == len(times), f"Length mismatch in {p}"

        raws_current.append(df["raw_current"].to_numpy())
        raws_voltage.append(df["raw_voltage"].to_numpy())

    # scale to physical units
    currents_pa = [r * C_CLAMP_AMP_PER_VOLT * 1e12 for r in raws_current]  # pA
    voltages_mv = [r * C_CLAMP_VOLT_PER_VOLT * 1e3 for r in raws_voltage]  # mV

    I_stack = np.vstack(currents_pa)   # (nSweeps × nPoints)
    V_stack = np.vstack(voltages_mv)

    # write single two‑channel ABF
    combined_abf = os.path.join(out_folder, f"{prefix.rstrip('_')}_Combined.abf")
    writeABF1_dual(I_stack, V_stack, combined_abf, sample_rate)
    print(f"Wrote {len(I_stack)} sweeps → {os.path.basename(combined_abf)} (2 channels)")
