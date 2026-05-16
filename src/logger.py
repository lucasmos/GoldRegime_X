import io
import logging
import sys
from pathlib import Path
from typing import Optional

LOG_DIR = Path("logs")

# Loggers that always write to the main goldregimex.log (summaries + Telegram).
# All other loggers are redirected to the TF-specific log by reconfigure_for_tf().
_MAIN_LOG_NAMES = {"main", "notifier"}

_fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(name)s: %(message)s")


def _make_file_handler(path: Path, level: int = logging.INFO) -> logging.FileHandler:
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_fmt)
    return fh


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    # Build a UTF-8 stream for the console handler.
    # On Windows, sys.stderr defaults to cp1252 which can't encode emoji or
    # box-drawing characters used in log messages.  We create a fresh
    # TextIOWrapper on the raw buffer so encoding errors are replaced with '?'
    # rather than crashing the process.
    try:
        _stream = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        # sys.stderr has no .buffer (e.g. IDLE / certain test runners) —
        # fall back to the stream as-is; emoji may break but the process won't.
        _stream = sys.stderr

    sh = logging.StreamHandler(stream=_stream)
    sh.setLevel(level)
    sh.setFormatter(_fmt)
    logger.addHandler(sh)
    if name in _MAIN_LOG_NAMES or _active_tf_handler is None:
        # Main/notifier loggers and loggers created before any TF is set
        # write to the main goldregimex.log.
        logger.addHandler(_make_file_handler(LOG_DIR / "goldregimex.log", level))
    else:
        # Logger created after reconfigure_for_tf() — write to TF log only.
        logger.addHandler(_active_tf_handler)
    return logger


def reconfigure_for_tf(tf: str, level: int = logging.INFO) -> None:
    """Redirect all non-main loggers to logs/goldregimex_{tf}.log.

    Call this once at the start of any TF-specific command (optimize, train,
    live, etc.) after the TF is known.  The ``main`` and ``notifier`` loggers
    keep writing to ``goldregimex.log`` so summaries and Telegram messages are
    always there.  Every other logger (src.*) switches to the TF file so
    detailed output is isolated per timeframe.
    """
    LOG_DIR.mkdir(exist_ok=True)
    tf_log_path = LOG_DIR / f"goldregimex_{tf.upper()}.log"

    # One shared TF FileHandler for all redirected loggers.
    tf_fh = _make_file_handler(tf_log_path, level)

    # Iterate every logger that has already been instantiated.
    manager = logging.Logger.manager
    all_names = list(manager.loggerDict.keys())

    for name in all_names:
        if name in _MAIN_LOG_NAMES:
            continue
        lgr = logging.getLogger(name)
        # Remove ALL existing FileHandlers (main log + any previous TF log).
        # This ensures a clean switch when cycling through multiple TFs in one run.
        to_remove = [h for h in list(lgr.handlers) if isinstance(h, logging.FileHandler)]
        for h in to_remove:
            h.close()
            lgr.removeHandler(h)
        lgr.addHandler(tf_fh)

    # Ensure future loggers created during this run also write to the TF file.
    # We do this by monkeypatching setup_logger's default file path for this process.
    global _active_tf_handler, _score_fh
    _active_tf_handler = tf_fh

    # Score-only log: one clean line per trial for easy tail -f monitoring.
    score_log_path = LOG_DIR / f"goldregimex_{tf.upper()}_scores.log"
    _score_fh = _make_file_handler(score_log_path, level)


# Module-level slot for the active TF handler so setup_logger can pick it up
# for loggers created *after* reconfigure_for_tf() is called.
_active_tf_handler: Optional[logging.FileHandler] = None

# Separate per-TF score log — one line per trial for easy progress tracking.
_score_fh: Optional[logging.FileHandler] = None


def append_trial_score(message: str) -> None:
    """Append a single score line to logs/goldregimex_{tf}_scores.log.

    Called once per trial by the optimizer so the scores log stays clean and
    easy to ``tail -f`` without HMM/XGB noise.
    """
    if _score_fh is None:
        return
    record = logging.LogRecord(
        name="scores", level=logging.INFO, pathname="", lineno=0,
        msg=message, args=(), exc_info=None,
    )
    _score_fh.emit(record)


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
