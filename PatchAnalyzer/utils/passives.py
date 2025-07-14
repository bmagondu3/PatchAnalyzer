# PatchAnalyzer/utils/passives.py
"""
Voltage‑step passive‑parameter extraction (Ra, Rm, Cm).

All voltages are assumed to be millivolts, currents pico‑amps.
The protocol step amplitude is fixed at +10 mV.

"""

from __future__ import annotations
import numpy as np
import scipy.optimize as opt

# ───────────────────────── helpers ──────────────────────────────────────
def _filter_data(T_ms: np.ndarray,
                 X_mV: np.ndarray,
                 Y_pA: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           tuple[float, int, float, int], float, float]:
    """
    Direct Python port of your `filter_data` method (unchanged logic).
    See original code for detailed doc‑string.    """
    X_dT = np.gradient(X_mV, T_ms)
    pos_i = int(np.argmax(X_dT))
    neg_i = int(np.argmin(X_dT))
    peak_i = int(np.argmax(Y_pA))

    peak_t  = float(T_ms[peak_i])
    npeak_t = float(T_ms[neg_i])

    pre_mean = float(Y_pA[:pos_i + 1].mean())

    sub_Y = Y_pA[peak_i:neg_i + 1]
    sub_T = T_ms[peak_i:neg_i + 1]
    sub_X = X_mV[peak_i:neg_i + 1]

    sub_grad = np.gradient(sub_Y, sub_T)
    close0   = np.where(np.isclose(sub_grad, 0, atol=1e-2))[0]
    if close0.size:
        zt = float(sub_T[int(close0[0])])
        msk = (T_ms >= zt) & (T_ms <= T_ms[neg_i])
        post_mean = float(Y_pA[msk].mean()) if msk.any() else None
    else:
        post_mean = None

    return (sub_Y, sub_T, sub_X,
            (peak_t, peak_i, npeak_t, neg_i),
            pre_mean, post_mean)


def _mono_exp(x, m, t, b):          # I(t) = m·e^(‑t·t) + b
    return m * np.exp(-t * x) + b


def _optimizer(T_ms, Y_pA, I_peak_pA, I_peak_t_ms, I_ss_pA):
    """
    Robust bounded least‑squares mono‑exponential fit.
    Returns  (m, t, b)  –  or (None, None, None) on failure.
    """
    y_nA = Y_pA * 1e-3
    m0   = I_peak_pA * 1e-3
    b0   = I_ss_pA   * 1e-3
    t0   = max(1.0 / max(I_peak_t_ms, 0.1), 0.01)
    p0   = (m0, t0, b0)

    bounds = ([-10.0, 1/100.0, -10.0],
              [ 10.0,   1/0.05,  10.0])

    def resid(p, x, y):
        return _mono_exp(x, *p) - y

    def jac(p, x, y):
        m, t, b = p
        e = np.exp(-t * x)
        return np.vstack((e, -m * x * e, np.ones_like(x))).T

    try:
        res = opt.least_squares(
            resid, p0, jac=jac, bounds=bounds,
            args=(T_ms, y_nA), loss="soft_l1",
            f_scale=1.0, max_nfev=400)
    except Exception:
        return None, None, None

    return (res.x * np.array([1e3, 1.0, 1e3])   # back to pA, 1/ms, pA
            if res.success and np.isfinite(res.x).all()
            else (None, None, None))


def _calc_params(tau_ms, V_step_mV,
                 I_peak_pA, I_pre_pA, I_ss_pA,
):
    I_d     = I_peak_pA - I_pre_pA
    I_d_ss  = I_ss_pA   - I_pre_pA

    if I_d == 0 or I_d_ss == 0:
        return None, None, None

    V_step_V = V_step_mV * 1e-3
    I_d_A    = I_d     * 1e-12
    I_d_ss_A = I_d_ss  * 1e-12

    Ra_ohm = V_step_V / I_d_A
    Rm_ohm = (V_step_V - Ra_ohm * I_d_ss_A) / I_d_ss_A

    tau_s  = tau_ms * 1e-3
    R_para = 1 / (1/Ra_ohm + 1/Rm_ohm)
    Cm_F   = tau_s / R_para

    return Ra_ohm * 1e-6, Rm_ohm * 1e-6, Cm_F * 1e12     # MΩ, MΩ, pF
# ───────────────────────── public API ────────────────────────────────────
def compute_passive_params(time_s: np.ndarray,
                           cmd_V:  np.ndarray,
                           rsp_A:  np.ndarray,
                           step_mV: float = 10.0,
):
    """
    Convenience wrapper – feed raw trace arrays, get
    (Ra_MΩ, Rm_MΩ, Cm_pF)  or (None, None, None).
    """
    if not (len(time_s) and len(cmd_V) and len(rsp_A)):
        return None, None, None
    try:
        T_ms = (time_s - time_s[0]) * 1000.0
        X_mV = cmd_V * 1000.0
        Y_pA = rsp_A * 1e12

        sub_Y, sub_T, sub_X, pp, I_pre, I_ss = _filter_data(T_ms, X_mV, Y_pA)

        peak_i = pp[1]
        I_pk   = Y_pA[min(peak_i + 1, len(Y_pA)-1)]
        t_pk   = T_ms[min(peak_i + 1, len(T_ms)-1)]

        m, t, b = _optimizer(sub_T - sub_T[0], sub_Y, I_pk, t_pk, I_ss)
        if m is None:
            return None, None, None

        tau = 1.0 / t
        return _calc_params(tau, step_mV, I_pk, I_pre, I_ss)
    except Exception:
        return None, None, None

# ─────────────────────────────────────────────────────────────────────────────
#  Current‑clamp (“CC”) passive parameters
#  Logic ported from   ephys_core.calc_pas_params_one_sweep
#  Units:  time [s] • command [pA] • response [mV]
# ─────────────────────────────────────────────────────────────────────────────
import numpy as _np
import scipy.optimize as _opt

def compute_cc_passive_params(time_s:  _np.ndarray,
                              cmd_pA:  _np.ndarray,
                              rsp_mV:  _np.ndarray,
                              I_hold_from_MT: float = 0.0,
):
    """
    Return a *dict* with 
        membrane_tau_ms, input_resistance_MOhm, membrane_capacitance_pF,
        resting_potential_mV, holding_current_pA

    Parameters
    ----------
    time_s : 1‑D array, seconds
    cmd_pA : injected current, pA (positive = depolarising)
    rsp_mV : membrane voltage, mV
    I_hold_from_MT : optional DC holding current measured in a MemTest

    Failure → all fields set to *None*.
    """
    try:
        dt = float(1.0 / (len(time_s) / (time_s[-1] - time_s[0])))

        # locate first negative‑going edge in command
        starts = _np.where(_np.diff(cmd_pA) < 0)[0]
        ends   = _np.where(_np.diff(cmd_pA) > 0)[0]
        if len(starts) == 0 or len(ends) == 0:
            raise RuntimeError("No current step detected")

        p_start, p_end = int(starts[0]), int(ends[0])

        V1 = float(_np.mean(rsp_mV[:p_start - 1]))
        V2 = float(_np.mean(rsp_mV[int(p_start + 0.1/dt): p_end]))

        I_hold = float(_np.mean(cmd_pA[:p_start - 10]))
        I_step = float(_np.mean(cmd_pA[p_start + 10: p_start + 110]) - I_hold)

        input_R = abs((V1 - V2) / I_step)      # MΩ (mV/pA  ==  MΩ)
        resting = V1 - (input_R * I_hold_from_MT)

        # fit mono‑exp over first 100 ms
        X = time_s[p_start : int(p_start + 0.1/dt)]
        Y = rsp_mV[p_start : int(p_start + 0.1/dt)]

        def _exp(x, m, t, b):
            return m * _np.exp(-t * x) + b

        m, t, b = _opt.curve_fit(
            _exp, X[::25], Y[::25],
            p0=(20.0, 10.0, rsp_mV[p_end]),
            maxfev=100_000
        )[0]

        tau_ms = (1 / t) * 1e3                       # ms
        Cm_pF  = (tau_ms / input_R)                  # pF   (ms / MΩ)

        return dict(
            membrane_tau_ms          = tau_ms,
            input_resistance_MOhm    = input_R,
            membrane_capacitance_pF  = Cm_pF,
            resting_potential_mV     = resting,
            holding_current_pA       = I_hold
        )
    except Exception:
        # any failure → return keys with None to keep GUI tolerant
        return dict(
            membrane_tau_ms          = None,
            input_resistance_MOhm    = None,
            membrane_capacitance_pF  = None,
            resting_potential_mV     = None,
            holding_current_pA       = None
        )
