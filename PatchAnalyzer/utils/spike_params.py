# PatchAnalyzer/utils/spike_params.py
"""
Current‑clamp firing‑rate curves and per‑spike metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import signal


# ─────────────────────────── internal helpers ────────────────────────────
def _find_test_pulse(cmd, edge_frac=0.05, min_pts=5):
    n_bl = max(1, int(0.05 * cmd.size))
    baseline = np.median(cmd[:n_bl])
    mad = np.median(np.abs(cmd[:n_bl] - baseline))
    sigma = 1.4826 * mad if mad else np.std(cmd[:n_bl])
    thr = max(1e-3, sigma * edge_frac * 100)
    idxs = np.where(np.abs(cmd - baseline) > thr)[0]
    if idxs.size == 0:
        return None, None
    runs = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
    runs = [r for r in runs if len(r) >= min_pts]
    return (int(runs[0][0]), int(runs[0][-1])) if runs else (None, None)


def _find_experimental_pulse(cmd, after_idx, min_pts=10):
    if after_idx is None or after_idx >= cmd.size - 1:
        return None, None
    post = cmd[after_idx : after_idx + max(1, int(0.02 * cmd.size))]
    baseline = np.median(post)
    thr = max(2.0, 0.05 * np.ptp(cmd))
    mask = (np.abs(cmd - baseline) > thr) & (np.arange(cmd.size) > after_idx)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return None, None
    runs = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
    runs = [r for r in runs if len(r) >= min_pts]
    return (int(runs[0][0]), int(runs[0][-1])) if runs else (None, None)


def _detect_spikes(trace_mV, sr_Hz, prom_mV=12, smooth_pts=50):
    sm = signal.convolve(trace_mV,
                         np.ones(smooth_pts) / smooth_pts,
                         mode="same")
    peaks, _ = signal.find_peaks(
        sm,
        prominence=prom_mV,
        height=trace_mV.min() + 0.05 * np.ptp(trace_mV),
    )
    return peaks


def _step_current_pA(cmd, s, e, clamp_gain=400.0):
    """Return baseline-subtracted step amplitude in pA (auto V→pA)."""
    if np.ptp(cmd) < 5:               # looks like Volts
        cmd = cmd * clamp_gain
    return float(np.mean(cmd[s:e]) - np.median(cmd[:s - 10]))
# ─────────────────────────── F–I curve (sweep-level) ─────────────────────
def calc_firing_curve(
    sweeps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    peak_prominence_mV: float = 12.0,
    smooth_pts:         int   = 50,
    normalize_by_Cm:    bool  = False,
    Cm_pF:              float | None = None,
    fail_fast:          bool  = False,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    sweeps
        List of *(time_s, command_pA, response_mV)* tuples – one per sweep.
        `response_mV` **must already be in millivolts** (same as notebook).
    normalize_by_Cm
        If True, adds a `current_inj_pApF` column (pA / pF).
    Cm_pF
        Membrane capacitance used for the normalisation.  Required iff
        `normalize_by_Cm` is True.
    """
    rows, skips = [], []

    for k, (time_s, cmd_pA, rsp_mV) in enumerate(sweeps, 1):
        try:
            ts, te = _find_test_pulse(cmd_pA)
            es, ee = _find_experimental_pulse(cmd_pA, te)
            if es is None:
                raise ValueError("no experimental pulse detected")

            sr = 1.0 / (time_s[1] - time_s[0])
            stim_len = (ee - es) / sr
            peaks = _detect_spikes(rsp_mV[es:ee], sr,
                                   prom_mV=peak_prominence_mV,
                                   smooth_pts=smooth_pts)

            mean_freq = len(peaks) / stim_len if stim_len else 0.0
            I_step_pA = _step_current_pA(cmd_pA, es, ee)

            row = dict(
                sweep                   = k,
                current_inj_pA          = I_step_pA,
                mean_firing_frequency_Hz= mean_freq,
            )
            if normalize_by_Cm:
                if not Cm_pF or Cm_pF <= 0:
                    raise ValueError("Cm_pF missing for normalisation")
                row["current_inj_pApF"] = I_step_pA / Cm_pF

            rows.append(row)

        except Exception as exc:
            skips.append(f"Sweep {k}: {exc}")
            if fail_fast:
                raise

    if skips:
        import warnings; warnings.warn("; ".join(skips))

    return pd.DataFrame(rows)
# ─────────────────────────── spike-level metrics ─────────────────────────
def calc_spike_metrics(
    sweeps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    dvdt_threshold_mV_per_ms: float = 50.0,
    normalize_by_Cm:          bool  = False,
    Cm_pF:                    float | None = None,
    half_width_window_ms:     float = 3.0,
    fail_fast:                bool  = False,
) -> pd.DataFrame:
    """
    Returns one row per spike with the same columns as before.

    * sweeps – list of (time_s, command_pA, response_mV) tuples.
    * response_mV must be in millivolts.
    """
    rows, skips = [], []

    for s_idx, (time_s, cmd_pA, rsp_mV) in enumerate(sweeps, 1):
        try:
            ts, te = _find_test_pulse(cmd_pA)
            es, ee = _find_experimental_pulse(cmd_pA, te)
            if es is None:
                raise ValueError("no experimental pulse detected")

            sr   = 1.0 / (time_s[1] - time_s[0])
            peaks = _detect_spikes(rsp_mV[es:ee], sr) + es
            if not len(peaks):
                continue

            dt   = 1.0 / sr
            dvdt = np.gradient(rsp_mV, dt * 1e3)
            win  = int(half_width_window_ms / 1000 / dt)

            I_step_pA = _step_current_pA(cmd_pA, es, ee)
            if normalize_by_Cm:
                if not Cm_pF or Cm_pF <= 0:
                    raise ValueError("Cm_pF missing for normalisation")
                I_step_pApF = I_step_pA / Cm_pF

            for p_num, pk in enumerate(peaks):
                prom = signal.peak_prominences(rsp_mV, [pk])[0][0]
                half_val = rsp_mV[pk] - 0.5 * prom
                local = rsp_mV[pk - win : pk + win]
                xloc = np.where(local > half_val)[0]
                if xloc.size < 2:
                    continue
                hw_ms = (xloc[-1] - xloc[0]) * dt * 1e3

                dv_seg = dvdt[pk - win : pk]
                below  = np.where(dv_seg < dvdt_threshold_mV_per_ms)[0]
                thr_i  = pk - win + below[-1] if below.size else pk
                thr_mV = rsp_mV[thr_i]

                dv_max = dvdt[pk - win : pk + win].max()
                ahp_mV = abs(rsp_mV[pk : pk + int(0.006 / dt)].min() - thr_mV)

                row = dict(
                    sweep                 = s_idx,
                    spike_number          = p_num,
                    current_inj_pA        = I_step_pA,
                    peak_mV               = rsp_mV[pk],
                    half_width_ms         = hw_ms,
                    AHP_mV                = ahp_mV,
                    threshold_mV          = thr_mV,
                    dvdt_max_mV_per_ms    = dv_max,
                )
                if normalize_by_Cm:
                    row["current_inj_pApF"] = I_step_pApF
                rows.append(row)

        except Exception as exc:
            skips.append(f"Sweep {s_idx}: {exc}")
            if fail_fast:
                raise

    if skips:
        import warnings; warnings.warn("; ".join(skips))

    return pd.DataFrame(rows)
