# PatchAnalyzer/utils/log.py
from __future__ import annotations
import logging, sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "PatchAnalyzer", log_dir: Path | None = None) -> logging.Logger:
    """Create <logs/run-YYYYMMDD-HHMMSS.log> + stdout handler and return logger."""
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = log_dir / f"run-{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()      # avoid dupes if setup twice

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh  = logging.FileHandler(file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("Logging to %s", file)
    return logger
