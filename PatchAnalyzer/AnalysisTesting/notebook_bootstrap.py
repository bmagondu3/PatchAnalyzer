"""
Utility helpers for running analysis notebooks without hard-coding
absolute paths or manually adjusting sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _analysis_dir() -> Path:
    """Return the directory that contains the analysis notebooks."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Workspace root that contains `PatchAnalyzer/` and `Data/`."""
    return _analysis_dir().parent.parent


def package_root() -> Path:
    """Return the importable package directory (`PatchAnalyzer`)."""
    return repo_root() / "PatchAnalyzer"


def data_root() -> Path:
    """Return the shared Data directory."""
    return repo_root() / "Data"


def ensure_repo_on_path() -> None:
    """
    Add the repository root to sys.path so `import PatchAnalyzer` works
    even when the notebook kernel starts in `PatchAnalyzer/AnalysisTesting`.
    """
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# Ensure imports succeed as soon as the helper is imported.
ensure_repo_on_path()

