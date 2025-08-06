#PatchAnalyzer/utils.py/ephys_analyzer.py
"""ephys_analyzer.py
====================
Unified electrophysiology‑analysis utilities.

This module consolidates the logic that used to live in
`PatchAnalyzer.utils.passives` **and** `PatchAnalyzer.utils.spike_params` into
_two_ classes that provide a clean, object‑oriented façade:

* **`VprotAnalyzer`**  – analysis of voltage‑step (voltage‑clamp) sweeps.
* **`CprotAnalyzer`**  – analysis of current‑clamp sweeps (passives and spikes).

Legacy functions in the original modules now proxy to these classes so that
existing notebooks/scripts keep running unchanged.

All numerical algorithms are _verbatim ports_ of the original code; no
behavioural changes were introduced.  Any differences would therefore be bugs –
please file an issue if you spot one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import scipy.optimize as opt
from scipy import signal
# for ignoring peak prominence of zero errors. comment out below 2 lines if you would like to view in console.
# ── EDIT PeakPropertyWarning START ──────────────────────
import warnings
try:                                # SciPy ≥1.9 keeps it internal
    from scipy.signal._peak_finding import PeakPropertyWarning
except ImportError:                 # fallback for older / future versions
    class PeakPropertyWarning(UserWarning):
        pass
warnings.filterwarnings("ignore", category=PeakPropertyWarning)
# ── EDIT PeakPropertyWarning END ────────────────────────


__all__ = [
    "VprotAnalyzer",
    "CprotAnalyzer",
]
# ---------------------------------------------------------------------------
#  Voltage‑step analysis  ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@dataclass
class VprotAnalyzer:
    """Extract *Ra*, *Rm* and *Cm* from voltage‑step sweeps.

    Parameters
    ----------
    step_mV
        Command‑step amplitude in **millivolts** (defaults to +10 mV).
    debug
        If *True*, each call to :py:meth:`fit_single_sweep` stores intermediate
        data under :pyattr:`last_debug` **and** returns it together with the
        tuple ``(Ra_MΩ, Rm_MΩ, Cm_pF)``.  Useful for unit‑tests or inspecting
        failed fits.
    """

    step_mV: float = 10.0
    debug: bool = False

    # last_debug is populated only when *debug* **or** *return_intermediates*
    # is True.  Its contents are implementation‑detail and may change.
    last_debug: dict[str, Any] = field(default_factory=dict, init=False)

    # ─────────────────────────────── public API ──────────────────────────
    # --------------------------------------------------------------------
    def fit_single_sweep(
        self,
        time_s: np.ndarray,
        cmd_V:  np.ndarray,
        rsp_A:  np.ndarray,
        *,
        step_mV: float | None = None,
        debug: bool = False,
        return_intermediates: bool = False,
    ):
        """
        Estimate passive parameters (Ra [MΩ], Rm [MΩ], Cm [pF]) from **one**
        voltage-step sweep.

        Parameters
        ----------
        time_s, cmd_V, rsp_A
            Sweep arrays: seconds • Volts • Amps.
        step_mV
            Optional per-call override of the command-step amplitude.
            Falls back to ``self.step_mV`` when *None* (default).
        debug
            When *True* prints short progress notes and enables extra checks.
        return_intermediates
            When *True* returns a tuple *(Ra, Rm, Cm, extras)* where *extras*
            is a ``dict`` containing intermediate arrays and fit results.
        """
        V_step = self.step_mV if step_mV is None else step_mV

        if not (len(time_s) and len(cmd_V) and len(rsp_A)):
            return (None, None, None, {}) if return_intermediates else (None, None, None)

        # ── unit conversions ────────────────────────────────────────────────
        T_ms = (time_s - time_s[0]) * 1e3       # s → ms
        X_mV = cmd_V * 1e3                      # V → mV
        Y_pA = rsp_A * 1e12                     # A → pA

        # ── data windowing & baseline ---------------------------------------
        sub_Y, sub_T, sub_X, pp, I_pre, I_ss = self._filter_data(T_ms, X_mV, Y_pA)
        peak_i = pp[1]
        I_pk   = Y_pA[min(peak_i + 1, len(Y_pA)-1)]
        t_pk   = T_ms[min(peak_i + 1, len(T_ms)-1)]

        # ── mono-exponential fit --------------------------------------------
        m, t, b = self._optimizer(sub_T - sub_T[0], sub_Y, I_pk, t_pk, I_ss)
        if m is None:
            return (None, None, None, {}) if return_intermediates else (None, None, None)

        tau_ms = 1.0 / t
        ra, rm, cm = self._calc_params(tau_ms, V_step, I_pk, I_pre, I_ss)

        if debug:
            print(f"[Vprot]  τ={tau_ms:.2f} ms • Ra={ra:.1f} MΩ • Rm={rm:.1f} MΩ • Cm={cm:.1f} pF")

        if return_intermediates:
            extras = dict(
                T_ms=T_ms, X_mV=X_mV, Y_pA=Y_pA,
                sub_T=sub_T, sub_Y=sub_Y,
                tau_ms=tau_ms, I_pk=I_pk, I_pre=I_pre, I_ss=I_ss,
            )
            return ra, rm, cm, extras

        return ra, rm, cm


    # --------------------------------------------------------------------
    def fit_cell(
        self,
        traces: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        aggregate: str = "mean",
        return_all: bool = False,
    ) -> tuple[float | None, float | None, float | None] | tuple[
        tuple[float | None, float | None, float | None], list[tuple[float, float, float]]
    ]:
        """Run :py:meth:`fit_single_sweep` on **every** sweep in *traces*.

        ``traces`` is expected to be the same dict structure returned by
        :pyfunc:`PatchAnalyzer.models.ephys_loader.load_voltage_traces_for_indices`.

        If at least one fit succeeds the *aggregate* (``"mean"`` or
        ``"median"``) is returned; otherwise every component is *None*.
        When *return_all* is *True*, the list of per‑sweep results is appended.
        """
        if aggregate not in {"mean", "median"}:
            raise ValueError("aggregate must be 'mean' or 'median'")

        results: list[tuple[float | None, float | None, float | None]] = []
        for t, cmd, rsp in traces.values():
            res = self.fit_single_sweep(t, cmd, rsp)
            results.append(res if isinstance(res, tuple) else res[0])

        ok = [r for r in results if all(v is not None for v in r)]
        if not ok:
            out = (None, None, None)
        else:
            agg = np.mean if aggregate == "mean" else np.median
            out = tuple(float(agg([r[i] for r in ok])) for i in range(3))  # type: ignore[arg-type]

        return (out, results) if return_all else out

    # ───────────────────────── internal helpers ────────────────────────
    # (verbatim logic from the original *passives.py*)
    # ------------------------------------------------------------------
    @staticmethod
    def _filter_data(
        T_ms: np.ndarray,
        X_mV: np.ndarray,
        Y_pA: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[float, int, float, int],
        float,
        float,
    ]:
        X_dT = np.gradient(X_mV, T_ms)
        pos_i = int(np.argmax(X_dT))
        neg_i = int(np.argmin(X_dT))
        peak_i = int(np.argmax(Y_pA))

        peak_t  = float(T_ms[peak_i])
        npeak_t = float(T_ms[neg_i])

        pre_mean = float(Y_pA[: pos_i + 1].mean())

        sub_Y = Y_pA[peak_i : neg_i + 1]
        sub_T = T_ms[peak_i : neg_i + 1]
        sub_X = X_mV[peak_i : neg_i + 1]

        sub_grad = np.gradient(sub_Y, sub_T)
        close0 = np.where(np.isclose(sub_grad, 0, atol=1e-2))[0]
        if close0.size:
            zt = float(sub_T[int(close0[0])])
            msk = (T_ms >= zt) & (T_ms <= T_ms[neg_i])
            post_mean = float(Y_pA[msk].mean()) if msk.any() else None
        else:
            post_mean = None

        return (
            sub_Y,
            sub_T,
            sub_X,
            (peak_t, peak_i, npeak_t, neg_i),
            pre_mean,
            post_mean,
        )

    @staticmethod
    def _mono_exp(x, m, t, b):
        return m * np.exp(-t * x) + b

    @classmethod
    def _optimizer(cls, T_ms, Y_pA, I_peak_pA, I_peak_t_ms, I_ss_pA):
        y_nA = Y_pA * 1e-3
        m0 = I_peak_pA * 1e-3
        b0 = I_ss_pA * 1e-3
        t0 = max(1.0 / max(I_peak_t_ms, 0.1), 0.01)
        p0 = (m0, t0, b0)

        bounds = ([-10.0, 1 / 100.0, -10.0], [10.0, 1 / 0.05, 10.0])

        def resid(p, x, y):
            return cls._mono_exp(x, *p) - y

        def jac(p, x, y):
            m, t, b = p
            e = np.exp(-t * x)
            return np.vstack((e, -m * x * e, np.ones_like(x))).T

        try:
            res = opt.least_squares(
                resid,
                p0,
                jac=jac,
                bounds=bounds,
                args=(T_ms, y_nA),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=400,
            )
        except Exception:
            return None, None, None

        return (
            res.x * np.array([1e3, 1.0, 1e3])
            if res.success and np.isfinite(res.x).all()
            else (None, None, None)
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _calc_params(
        tau_ms: float,
        V_step_mV: float,
        I_peak_pA: float,
        I_pre_pA: float,
        I_ss_pA: float,
    ) -> tuple[float, float, float] | tuple[None, None, None]:
        I_d = I_peak_pA - I_pre_pA
        I_d_ss = I_ss_pA - I_pre_pA

        if I_d == 0 or I_d_ss == 0:
            return None, None, None

        V_step_V = V_step_mV * 1e-3
        I_d_A = I_d * 1e-12
        I_d_ss_A = I_d_ss * 1e-12

        Ra_ohm = V_step_V / I_d_A
        Rm_ohm = (V_step_V - Ra_ohm * I_d_ss_A) / I_d_ss_A

        tau_s = tau_ms * 1e-3
        R_para = 1 / (1 / Ra_ohm + 1 / Rm_ohm)
        Cm_F = tau_s / R_para

        return Ra_ohm * 1e-6, Rm_ohm * 1e-6, Cm_F * 1e12  # MΩ, MΩ, pF

# ---------------------------------------------------------------------------
#  Current‑clamp analysis  ─────────────────────────────────────────────────-
# ---------------------------------------------------------------------------

@dataclass
class CprotAnalyzer:
    """Analyse current‑clamp sweeps – passive params & spikes.

    Parameters
    ----------
    clamp_gain
        Conversion factor from **Volts** applied by the stimulus generator to
        measured pico‑amps.  The default (``400 pA / V``) matches Axon
        Multiclamp 700.
    debug
        Same semantics as in :pyclass:`VprotAnalyzer`.
    """

    cclamp_gain: float = 400.0  # 400 pA/V
    vclamp_gain: float = 1000.0  # 1000 mV/V
    debug: bool = False
    last_debug: dict[str, Any] = field(default_factory=dict, init=False)


    # ───────────────────────── passive parameters ───────────────────────
    def passive_params(
        self,
        time_s: np.ndarray,
        cmd_V:  np.ndarray,
        rsp_V: np.ndarray,
        *,
        baseline_ms: float = 20.0,
        fit_window_ms: float = 80.0,
        min_step_pA: float = 5.0,
        return_intermediates: bool = False,
    ):
        # convert stimulus  V → pA
        cmd_pA = cmd_V * self.cclamp_gain
        rsp_mV = rsp_V * self.vclamp_gain    # V → mV

        dbg: dict[str, Any] = {}
        errors: list[str] = []

        # ── locate test-pulse
        ts, te = self._find_test_pulse(cmd_pA)
        if ts is None:
            raise RuntimeError("No test-pulse detected in this sweep.")

        dt       = float(time_s[1] - time_s[0])
        pre_pts  = int(baseline_ms / 1000 / dt)

        V_pre    = float(np.mean(rsp_mV[max(0, ts - pre_pts) : ts]))
        V_ss     = float(np.mean(rsp_mV[te - pre_pts : te]))
        dV_mV    = V_ss - V_pre

        I_hold   = float(np.mean(cmd_pA[max(0, ts - pre_pts) : ts]))
        I_step   = float(np.mean(cmd_pA[ts:te]) - I_hold)
        if abs(I_step) < min_step_pA:
            raise RuntimeError(f"Test-pulse amplitude < {min_step_pA} pA")

        Rin_GOhm = abs(dV_mV / I_step) # in GΩ
        Rin_MOhm = Rin_GOhm * 1e3 # now in MΩ
        tau_ms   = self._fit_tau(
            time_s[ts : ts + int(fit_window_ms / 1000 / dt)] - time_s[ts],
            rsp_mV[ts : ts + int(fit_window_ms / 1000 / dt)],
            V_ss,
            errors,
        )
        tau_s = tau_ms * 1e-3 # convert ms → seconds
        Rin_Ohm = Rin_MOhm*1e6 # convert MOhms →  Ohms
        Cm_F    = (tau_s / (Rin_Ohm)) if (Rin_GOhm and not np.isnan(tau_ms)) else np.nan 
        Cm_pF = Cm_F*1e12 # F → pF
        # -----------------------------------------------

        result = dict(
            membrane_tau_ms       = tau_ms,
            input_resistance_MOhm = Rin_MOhm,
            membrane_capacitance_pF = Cm_pF,
            resting_potential_mV  = V_pre,
            holding_current_pA    = I_hold,
        )

        if errors:
            raise RuntimeError("; ".join(errors))

        if self.debug or return_intermediates:
            dbg = dict(ts=ts, te=te, V_pre=V_pre, V_ss=V_ss, I_step_pA=I_step)
            self.last_debug = dbg
            return result, dbg

        return result

    # ───────────────────────── firing curve / spikes ────────────────────
    def firing_curve(
        self,
        sweeps: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
        **kwargs,
    ) -> 'pd.DataFrame':  # type: ignore[name-defined]
        """Compute the F–I curve (one row ≙ one sweep).

        The command arrays must be **Volts**.
        """
        import pandas as pd

        conv: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for t, cmd_V, rsp in sweeps:
            conv.append((t, cmd_V * self.cclamp_gain, rsp*self.vclamp_gain))  # V → mV
        return self._calc_firing_curve(conv, **kwargs)

    def spike_metrics(
        self,
        sweeps: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
        **kwargs,
    ) -> 'pd.DataFrame':  # type: ignore[name-defined]
        """Per‑spike metrics for every spike in *sweeps*.

        The command arrays must be **Volts**.
        """
        import pandas as pd

        conv: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for t, cmd_V, rsp in sweeps:
            conv.append((t, cmd_V * self.cclamp_gain, rsp*self.vclamp_gain))  # V → mV
        return self._calc_spike_metrics(conv, **kwargs)

    # ───────────────────────── internal helpers ────────────────────────
    # -------------------------- passive τ fit --------------------------
    @staticmethod
    def _fit_tau(X_rel_s: np.ndarray, Y_mV: np.ndarray, V_ss: float, errors: list[str]):
        """
        Fit the membrane voltage decay with a *single* exponential
        (no offset term).  The stimulus is assumed to be a hyper‑/depolarising
        step that has already been centred around the steady‑state level
        (V_ss).  This removes the redundant `b` parameter and allows the fit
        to use negative amplitudes, which is essential for correctly
        estimating τ for hyperpolarising steps.
        """
        # Convert relative time (seconds) → milliseconds
        X_ms = X_rel_s * 1e3

        # Remove the steady‑state voltage – the exponential will start from 0
        Y = Y_mV - V_ss

        try:
            with np.errstate(over="ignore", under="ignore"):
                # Fit m * exp(-k * x)  (2‑parameter model)
                popt, _ = opt.curve_fit(
                    lambda x, m, k: m * np.exp(-k * x),
                    X_ms,
                    Y,
                    p0=(Y[0], 10),                           # initial guess
                    bounds=([-np.inf, 0], [np.inf, 1e3]),      # m free (±∞), k ≥ 0
                    maxfev=20_000,
                )
            # τ = 1 / k  (k is in ms⁻¹)
            return 1.0 / popt[1]
        except Exception as exc:
            errors.append(f"τ‑fit failed: {exc}")
            return np.nan


    # -------------------------- shared helpers -------------------------
    @staticmethod
    def _find_test_pulse(cmd_pA: np.ndarray, edge_frac=0.05, min_pts=5):
        n_bl = max(1, int(0.05 * cmd_pA.size))
        baseline = np.median(cmd_pA[:n_bl])
        mad = np.median(np.abs(cmd_pA[:n_bl] - baseline))
        sigma = 1.4826 * mad if mad else np.std(cmd_pA[:n_bl])
        thr = max(1e-3, sigma * edge_frac * 100)
        idxs = np.where(np.abs(cmd_pA - baseline) > thr)[0]
        if idxs.size == 0:
            return None, None
        runs = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
        runs = [r for r in runs if len(r) >= min_pts]
        return (int(runs[0][0]), int(runs[0][-1])) if runs else (None, None)

    # -------------------------- F–I curve -------------------------------
    @staticmethod
    def _calc_firing_curve(
        sweeps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        peak_prominence_mV: float = 25.0,
        smooth_pts: int = 50,
        normalize_by_Cm: bool = False,
        Cm_pF: float | None = None,
        fail_fast: bool = False,
    ):
        import pandas as pd

        rows, skips = [], []
        for k, (time_s, cmd_pA, rsp_mV) in enumerate(sweeps, 1):
            try:
                ts, te = CprotAnalyzer._find_test_pulse(cmd_pA)
                es, ee = CprotAnalyzer._find_experimental_pulse(cmd_pA, te)
                if es is None:
                    raise ValueError("no experimental pulse detected")

                sr = 1.0 / (time_s[1] - time_s[0])
                stim_len = (ee - es) / sr
                peaks = CprotAnalyzer._detect_spikes(
                    rsp_mV[es:ee], sr, prom_mV=peak_prominence_mV, smooth_pts=smooth_pts
                )
                mean_freq = len(peaks) / stim_len if stim_len else 0.0
                I_step = CprotAnalyzer._step_current_pA(cmd_pA, es, ee)

                row = dict(
                    sweep=k,
                    current_inj_pA=I_step,
                    mean_firing_frequency_Hz=mean_freq,
                )
                if normalize_by_Cm:
                    if not Cm_pF or Cm_pF <= 0:
                        raise ValueError("Cm_pF missing for normalisation")
                    row["current_inj_pApF"] = I_step / Cm_pF

                rows.append(row)
            except Exception as exc:
                skips.append(f"Sweep {k}: {exc}")
                if fail_fast:
                    raise
        if skips:
            import warnings

            warnings.warn("; ".join(skips))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    @staticmethod
    def _calc_spike_metrics(
        sweeps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        dvdt_threshold_mV_per_ms: float = 50.0,
        normalize_by_Cm: bool = False,
        Cm_pF: float | None = None,
        half_width_window_ms: float = 3.0,
        fail_fast: bool = False,
    ):
        import pandas as pd

        rows, skips = [], []
        for s_idx, (time_s, cmd_pA, rsp_mV) in enumerate(sweeps, 1):

            try:
                ts, te = CprotAnalyzer._find_test_pulse(cmd_pA)
                es, ee = CprotAnalyzer._find_experimental_pulse(cmd_pA, te)
                if es is None:
                    raise ValueError("no experimental pulse detected")

                sr   = 1.0 / (time_s[1] - time_s[0])
                peaks = CprotAnalyzer._detect_spikes(rsp_mV[es:ee], sr) + es
                if not peaks.size:
                    continue

                dt     = 1.0 / sr
                dvdt   = np.gradient(rsp_mV, dt * 1e3)    # mV / ms
                win    = int(half_width_window_ms / 1000 / dt)
                I_step = CprotAnalyzer._step_current_pA(cmd_pA, es, ee)

                # ---------- metric extraction unchanged ----------
                if normalize_by_Cm:
                    print("normalize_by_Cm is set")
                    if not Cm_pF or Cm_pF <= 0:
                        raise ValueError("Cm_pF missing for normalisation")
                    I_step_pApF = I_step / Cm_pF

                for p_num, pk in enumerate(peaks):
                    prom     = signal.peak_prominences(rsp_mV, [pk])[0][0]
                    half_val = rsp_mV[pk] - 0.5 * prom
                    local    = rsp_mV[pk - win : pk + win]
                    xloc     = np.where(local > half_val)[0]
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
                        sweep               = s_idx,
                        spike_number        = p_num,
                        current_inj_pA      = I_step,
                        peak_mV             = rsp_mV[pk],
                        half_width_ms       = hw_ms,
                        AHP_mV              = ahp_mV,
                        threshold_mV        = thr_mV,
                        dvdt_max_mV_per_ms  = dv_max,
                    )
                    if normalize_by_Cm:
                        print("normalized injections being added")
                        row["current_inj_pApF"] = I_step_pApF
                    rows.append(row)
                # -------------------------------------------------
            except Exception as exc:
                skips.append(f"Sweep {s_idx}: {exc}")
                if fail_fast:
                    raise

        if skips:
            import warnings
            warnings.warn("; ".join(skips))
        return pd.DataFrame(rows)

    # --------------------- helper: experimental pulse ------------------
    @staticmethod
    def _find_experimental_pulse(cmd_pA: np.ndarray, after_idx: int | None, min_pts=10):
        if after_idx is None or after_idx >= cmd_pA.size - 1:
            return None, None
        post = cmd_pA[after_idx : after_idx + max(1, int(0.02 * cmd_pA.size))]
        baseline = np.median(post)
        thr = max(0.5, 0.02 * np.ptp(cmd_pA))     # recognise ±20 pA steps
        mask = (np.abs(cmd_pA - baseline) > thr) & (np.arange(cmd_pA.size) > after_idx)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            return None, None
        runs = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
        runs = [r for r in runs if len(r) >= min_pts]
        return (int(runs[0][0]), int(runs[0][-1])) if runs else (None, None)

    # --------------------------- spike detect --------------------------
    @staticmethod
    def _detect_spikes(trace_mV, sr_Hz, *, prom_mV=25, smooth_pts=50):
        sm = signal.convolve(trace_mV, np.ones(smooth_pts) / smooth_pts, mode="same")
        peaks, _ = signal.find_peaks(
            sm,
            prominence=prom_mV,
            height=trace_mV.min() + 0.05 * np.ptp(trace_mV),
        )
        return peaks

    # ---------------------------- helpers -----------------------------
    @staticmethod
    def _step_current_pA(cmd_pA: np.ndarray, s: int, e: int):
        """Baseline‑subtracted step amplitude in pico‑amps (no auto‑detect)."""
        return float(np.mean(cmd_pA[s:e]) - np.median(cmd_pA[: s - 10]))
