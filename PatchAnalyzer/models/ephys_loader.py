# DEPRECATED shim – will be removed in a future release
from warnings import warn
from .data_loader import (
    _read_csv,
    load_voltage_traces_for_indices,
    load_current_traces,
)

warn(
    "✅  'PatchAnalyzer.models.ephys_loader' is deprecated – "
    "import the same names from 'PatchAnalyzer.models.data_loader' instead.",
    DeprecationWarning,
    stacklevel=2,
)
