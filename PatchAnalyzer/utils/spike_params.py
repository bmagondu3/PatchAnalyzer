# PatchAnalyzer/utils/spike_params.py
"""
Current‑clamp firing‑rate curves and per‑spike metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import signal


# ─────────────────────────── little container ────────────────────────────
@dataclass
class Sweep:
    """One electrophysiology sweep (current‑clamp protocol)."""
    time:         np.ndarray     # seconds
    response_mV:  np.ndarray     # membrane voltage (mV)
    command_pA:   np.ndarray     # injected current (pA)
    sample_rate:  float          # Hz


# ─────────────────────────── F–I curve (sweep‑level) ─────────────────────
def calc_firing_curve(
    sweeps: list[Sweep],
) -> pd.DataFrame:
    """
    Return a DataFrame with *sweep, current_inj_pA, mean_firing_frequency_Hz*.

    Identical thresholds/prominences as patchAnalysis.calc_freq
    (prominence ≥ 12 mV on a 50‑pt boxcar‑smoothed trace).
    """
    rows = []
    for k, sw in enumerate(sweeps, 1):
        dt   = 1.0 / sw.sample_rate
        cmd  = sw.command_pA
        rsp  = sw.response_mV

        # identify current pulse (first negative edge relative to baseline)
        is_on = cmd < np.mean(cmd[:10])
        idxs  = np.where(is_on)[0]
        if idxs.size == 0:
            continue
        stim_start, stim_end = idxs[0], idxs[-1]
        stim_len = (stim_end - stim_start) * dt

        seg = rsp[stim_start: stim_end + int(0.5/dt)]          # +0.5 s buffer
        sm  = signal.convolve(seg, np.ones(50) / 50, mode="same")
        peaks, _ = signal.find_peaks(
            sm,
            prominence=12,
            width=(None, 1000),
            height=.01 * (seg.max() - seg.min()) + seg.min(),
        )
        n_spikes  = len(peaks)
        mean_freq = n_spikes / stim_len if stim_len else 0.0

        rows.append(dict(
            sweep                   = k,
            current_inj_pA          = float(cmd[stim_start + 5]),
            mean_firing_frequency_Hz= mean_freq,
        ))

    return pd.DataFrame(rows)


# ─────────────────────────── spike‑level metrics ─────────────────────────
def calc_spike_metrics(
    sweeps: list[Sweep],
    dvdt_threshold_mV_per_ms: float = 50.0,
) -> pd.DataFrame:
    """
    Return one row per spike with:

        sweep, spike_number, current_inj_pA,
        peak_mV, half_width_ms, AHP_mV, threshold_mV, dvdt_max_mV_per_ms
    """
    all_rows = []
    for s_idx, sw in enumerate(sweeps, 1):
        dt   = 1.0 / sw.sample_rate
        rsp  = sw.response_mV
        cmd  = sw.command_pA

        starts = np.where(np.diff(cmd) < 0)[0]
        ends   = np.where(np.diff(cmd) > 0)[0]
        if len(starts) < 2 or len(ends) < 2:
            continue
        stim_start = starts[1]
        stim_end   = ends[1]

        dvdt = np.gradient(rsp, dt * 1e3)       # mV/ms
        seg  = rsp[stim_start: stim_end + int(0.5/dt)]
        sm   = signal.convolve(seg, np.ones(50) / 50, mode="same")
        peaks, _ = signal.find_peaks(
            sm,
            prominence=25,
            height=.25 * (rsp.max() - rsp.min()) + rsp.min(),
        )
        prominences = signal.peak_prominences(sm, peaks)[0]
        peaks -= 5                               # replicate repo’s −5 shift

        for p_idx, pk in enumerate(peaks):
            pk_global = stim_start + pk
            win = int(0.003 / dt)                # ±3 ms

            # half‑width (mV >  peak − 0.5 × prominence)
            half_val = rsp[pk_global] - 0.5 * prominences[p_idx]
            local    = rsp[pk_global - win : pk_global + win]
            xloc = np.where(local > half_val)[0]
            if xloc.size < 2:
                continue
            half_w_ms = (xloc[-1] - xloc[0]) * dt * 1e3

            # threshold = last dv/dt < threshold before spike
            dv_seg = dvdt[pk_global - win : pk_global - win//10]
            lt = np.where(dv_seg < dvdt_threshold_mV_per_ms)[0]
            if lt.size == 0:
                continue
            thresh_idx = pk_global - win + lt[-1]
            thresh_mV  = rsp[thresh_idx]

            # dv/dt max within ±3 ms
            dv_max = dvdt[pk_global - win : pk_global + win].max()

            # after‑hyperpolarisation amplitude
            ahp = abs(rsp[pk_global : pk_global + int(0.006/dt)].min() - thresh_mV)

            all_rows.append(dict(
                sweep              = s_idx,
                spike_number       = p_idx,
                current_inj_pA     = float(cmd[stim_start + 5]),
                peak_mV            = rsp[pk_global],
                half_width_ms      = half_w_ms,
                AHP_mV             = ahp,
                threshold_mV       = thresh_mV,
                dvdt_max_mV_per_ms = dv_max,
            ))

    return pd.DataFrame(all_rows)
