# PatchAnalyzer/utils/spike_params.py
"""
Legacy wrappers for current-clamp analyses (passives + spikes).

All real logic now lives in `PatchAnalyzer.utils.ephys_analyzer`
class `CprotAnalyzer`.  These thin functions proxy the calls so
existing notebooks remain functional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .ephys_analyzer import CprotAnalyzer

__all__ = [
    "compute_cc_passive_params",
    "calc_firing_curve",
    "calc_spike_metrics",
    "CprotAnalyzer",
]

# ---------------------------------------------------------------------------

_cprot = CprotAnalyzer()          # uses default clamp_gain = 400 pA / V


# --- passive parameters ----------------------------------------------------
def compute_cc_passive_params(
    time_s: np.ndarray,
    cmd_pA: np.ndarray,
    rsp_mV: np.ndarray,
    **kwargs,
) -> dict:
    """
    Wrapper for :py:meth:`CprotAnalyzer.passive_params`.

    Additional keyword arguments are forwarded unchanged.
    """
    return _cprot.passive_params(time_s, cmd_pA, rsp_mV, **kwargs)


# --- sweep-level F–I curve -------------------------------------------------
def calc_firing_curve(
    sweeps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper for :py:meth:`CprotAnalyzer.firing_curve`.

    • *sweeps* – list of (time_s, command_pA, response_mV) tuples.  
    • Extra kwargs (e.g. `normalize_by_Cm`) are forwarded verbatim.
    """
    return _cprot.firing_curve(sweeps, **kwargs)


# --- spike-level metrics ---------------------------------------------------
def calc_spike_metrics(
    sweeps: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    **kwargs,
) -> pd.DataFrame:
    """
    Wrapper for :py:meth:`CprotAnalyzer.spike_metrics`.

    Extra kwargs are forwarded directly to the underlying method.
    """
    return _cprot.spike_metrics(sweeps, **kwargs)



#--------------------------------------old code--------------------------------------

# PatchAnalyzer/utils/spike_params.py
"""
Current‑clamp firing‑rate curves and per‑spike metrics.
"""

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

