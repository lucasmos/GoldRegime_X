import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler()
    # On Windows the default console encoding (cp1252) can't render emoji from
    # button labels — reconfigure stdout to UTF-8 so log lines never crash.
    try:
        if hasattr(sh.stream, "reconfigure"):
            sh.stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / "goldregimex.log", mode="a")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def log_regime_transition(logger, timestamp, old_state, new_state, state_names):
    logger.info(
        "REGIME CHANGE at %s: %s -> %s",
        timestamp,
        state_names.get(old_state, str(old_state)),
        state_names.get(new_state, str(new_state)),
    )


def log_trade_signal(logger, timestamp, direction, probability, hmm_state):
    logger.info(
        "SIGNAL at %s: %s (prob=%.3f, hmm_state=%d)",
        timestamp, direction, probability, hmm_state,
    )
