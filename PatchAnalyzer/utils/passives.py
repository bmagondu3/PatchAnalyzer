# PatchAnalyzer/utils/passives.py
"""
Legacy wrapper for voltage-step passive-parameter extraction.

The original implementation has been moved to
`PatchAnalyzer.utils.ephys_analyzer` (class `VprotAnalyzer`).

Importing this module keeps older notebooks/scripts working:
`compute_passive_params()` forwards to one shared `VprotAnalyzer` instance.

Remove these wrappers once all external code uses the class API directly.
"""
from __future__ import annotations

import numpy as np
from .ephys_analyzer import VprotAnalyzer

__all__ = ["compute_passive_params", "VprotAnalyzer"]

# ---------------------------------------------------------------------------

# one analyser reused for every call (stateless apart from optional debug logs)
_vprot = VprotAnalyzer()

def compute_passive_params(
    time_s: np.ndarray,
    cmd_V:  np.ndarray,
    rsp_A:  np.ndarray,
    *,
    step_mV: float = 10.0,
    debug: bool = False,
    return_intermediates: bool = False,
):
    """
    Wrapper around :py:meth:`VprotAnalyzer.fit_single_sweep`.

    Parameters
    ----------
    time_s, cmd_V, rsp_A
        Sweep arrays as before (seconds, Volts, Amps).
    step_mV
        Voltage-step amplitude (default ±10 mV).
    debug, return_intermediates
        Passed straight through to the analyser.
    """
    return _vprot.fit_single_sweep(
        time_s,
        cmd_V,
        rsp_A,
        step_mV=step_mV,
        debug=debug,
        return_intermediates=return_intermediates,
    )
