"""Multi-asset data consolidator.

Merges MT5 CSV exports from ``data/raw/`` into sorted master files at
``data/processed/``.  One master file is produced per asset per trading
timeframe so each training pipeline uses a series whose bar frequency
matches the XAUUSD bars it is merged onto.

Supported assets:
    USDCHF  — intraday DXY proxy (~0.85 DXY correlation)
    XAGUSD  — silver (cross-commodity regime signal)
    XTIUSD  — WTI crude oil (macro risk-on/off signal)
    US500   — S&P 500 (equity-gold correlation)
    USDJPY  — JPY carry trade (safe-haven proxy)

Raw CSV naming convention (place in data/raw/):
    {ASSET}_H1.csv                      — hourly bars (H1)
    {ASSET}_M15_<dates>.csv             — 15-min bars (M15)
    {ASSET}_M5_<dates>.csv              — 5-min bars  (M5)

Usage:
    python main.py --mode consolidate
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.logger import setup_logger

logger = setup_logger(__name__)

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# ── Per-asset per-TF configuration ───────────────────────────────────────────
# primary:  preferred single-file source (checked first)
# glob:     fallback pattern when primary is absent
# exclude:  substrings that disqualify a glob match (avoid mixing TFs)
# output:   filename inside PROCESSED_DIR

ASSET_CONFIGS: dict[str, dict[str, dict]] = {
    "USDCHF": {
        "H1":  {"primary": "USDCHF_H1.csv",     "glob": "*USDCHF*.csv",     "exclude": ["m5", "m15"], "output": "USDCHF_master.csv"},
        "M15": {"primary": None,                  "glob": "USDCHF_M15_*.csv", "exclude": [],            "output": "USDCHF_master_M15.csv"},
        "M5":  {"primary": None,                  "glob": "USDCHF_M5_*.csv",  "exclude": [],            "output": "USDCHF_master_M5.csv"},
    },
    "XAGUSD": {
        "H1":  {"primary": "XAGUSD_H1.csv",      "glob": "*XAGUSD*.csv",     "exclude": ["m5", "m15"], "output": "XAGUSD_master.csv"},
        "M15": {"primary": None,                  "glob": "XAGUSD_M15_*.csv", "exclude": [],            "output": "XAGUSD_master_M15.csv"},
        "M5":  {"primary": None,                  "glob": "XAGUSD_M5_*.csv",  "exclude": [],            "output": "XAGUSD_master_M5.csv"},
    },
    "XTIUSD": {
        "H1":  {"primary": "XTIUSD_H1.csv",      "glob": "*XTIUSD*.csv",     "exclude": ["m5", "m15"], "output": "XTIUSD_master.csv"},
        "M15": {"primary": None,                  "glob": "XTIUSD_M15_*.csv", "exclude": [],            "output": "XTIUSD_master_M15.csv"},
        "M5":  {"primary": None,                  "glob": "XTIUSD_M5_*.csv",  "exclude": [],            "output": "XTIUSD_master_M5.csv"},
    },
    "US500": {
        "H1":  {"primary": "US500_H1.csv",        "glob": "*US500*.csv",      "exclude": ["m5", "m15"], "output": "US500_master.csv"},
        "M15": {"primary": None,                  "glob": "US500_M15_*.csv",  "exclude": [],            "output": "US500_master_M15.csv"},
        "M5":  {"primary": None,                  "glob": "US500_M5_*.csv",   "exclude": [],            "output": "US500_master_M5.csv"},
    },
    "USDJPY": {
        "H1":  {"primary": "USDJPY_H1.csv",       "glob": "*USDJPY*.csv",     "exclude": ["m5", "m15"], "output": "USDJPY_master.csv"},
        "M15": {"primary": None,                  "glob": "USDJPY_M15_*.csv", "exclude": [],            "output": "USDJPY_master_M15.csv"},
        "M5":  {"primary": None,                  "glob": "USDJPY_M5_*.csv",  "exclude": [],            "output": "USDJPY_master_M5.csv"},
    },
}

# XAUUSD raw file locations (for detect_common_start_date)
_XAUUSD_RAW_BY_TF = {
    "H1":  Path("data/raw/XAU_1h_data.csv"),
    "M15": Path("data/raw/XAU_m15_data.csv"),
    "M5":  Path("data/raw/XAU_5m_data.csv"),
}

# Legacy path constants kept for backwards compat (callers that reference them directly)
MASTER_PATH     = PROCESSED_DIR / "USDCHF_master.csv"
MASTER_PATH_M15 = PROCESSED_DIR / "USDCHF_master_M15.csv"
MASTER_PATH_M5  = PROCESSED_DIR / "USDCHF_master_M5.csv"


# ── CSV reader ────────────────────────────────────────────────────────────────

def _read_asset_csv(f: Path) -> pd.DataFrame | None:
    """Read a single asset CSV, auto-detecting delimiter and header presence.

    Handles three formats exported from MT5:
    - History Center tab-delimited format with ``<DATE>`` + ``<TIME>`` columns
    - Named-column exports (``Date;Open;High;Low;Close;Volume`` header)
    - Headerless exports where the first column is a raw datetime string
    """
    try:
        first_line = f.read_text(encoding="utf-8", errors="replace").splitlines()[0]

        # MT5 History Center: tab-delimited with angle-bracket column names
        if "<DATE>" in first_line and "<TIME>" in first_line:
            df = pd.read_csv(f, sep="\t")
            df.columns = [c.strip("<>").lower() for c in df.columns]
            df["Date"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                format="%Y.%m.%d %H:%M:%S",
            )
            df.set_index("Date", inplace=True)
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                    "close": "Close", "tickvol": "Volume"})
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep]
            if pd.api.types.is_datetime64_any_dtype(df.index):
                logger.info("Loaded %d rows (MT5 History format) from %s", len(df), f.name)
                return df
            logger.warning("Could not parse dates in MT5 History format %s — skipping.", f.name)
            return None

        sep = ";" if ";" in first_line else ","

        # Headerless export — first char is a digit
        if first_line[0].isdigit():
            df = pd.read_csv(
                f, sep=sep, header=None,
                names=["Date", "Open", "High", "Low", "Close", "Volume"],
                parse_dates=["Date"],
            )
            df.set_index("Date", inplace=True)
            if pd.api.types.is_datetime64_any_dtype(df.index):
                logger.info("Loaded %d rows (headerless) from %s", len(df), f.name)
                return df
            logger.warning("Could not parse dates in headerless %s — skipping.", f.name)
            return None

        # Normal file with header row — try several date formats
        for fmt in ["%m/%d/%Y", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d", None]:
            try:
                kwargs: dict = dict(sep=sep, parse_dates=["Date"])
                if fmt:
                    kwargs["date_format"] = fmt
                df = pd.read_csv(f, **kwargs)
                df.set_index("Date", inplace=True)
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    logger.info("Loaded %d rows from %s", len(df), f.name)
                    return df
            except Exception:
                continue
        logger.warning("Could not parse dates in %s — skipping.", f.name)
        return None

    except Exception as exc:
        logger.warning("Failed to read %s: %s — skipping.", f.name, exc)
        return None


# Legacy alias so old imports still work
_read_usdchf_csv = _read_asset_csv


def _consolidate_files(files: list[Path], out_path: Path, label: str) -> pd.DataFrame:
    """Merge a list of asset CSV files into a single sorted master CSV."""
    frames = []
    for f in files:
        df = _read_asset_csv(f)
        if df is not None:
            frames.append(df)

    if not frames:
        logger.error("All %s source files failed to load — no master produced.", label)
        return pd.DataFrame()

    merged = pd.concat(frames)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.sort_index(inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path)
    logger.info(
        "%s master saved: %d rows -> %s  (source: %s)",
        label, len(merged), out_path.name, ", ".join(f.name for f in files),
    )
    return merged


# ── Generic consolidation entry point ────────────────────────────────────────

def consolidate_asset(
    asset: str,
    tf: str,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Build the master CSV for one asset and timeframe.

    Looks up the raw CSV in *raw_dir* using the patterns in ASSET_CONFIGS,
    merges, deduplicates, and writes to data/processed/.

    Args:
        asset: Asset symbol, e.g. ``"USDCHF"``, ``"XAGUSD"``, ``"US500"``.
        tf:    Timeframe string: ``"H1"``, ``"M15"``, or ``"M5"``.

    Returns:
        Merged DataFrame (OHLCV, DatetimeIndex) or empty DataFrame if no
        source files were found.
    """
    tf    = tf.upper()
    asset = asset.upper()
    asset_cfg = ASSET_CONFIGS.get(asset)
    if asset_cfg is None:
        logger.warning("No config for asset=%s — skipping.", asset)
        return pd.DataFrame()

    tf_cfg = asset_cfg.get(tf)
    if tf_cfg is None:
        logger.warning("No config for asset=%s tf=%s — skipping.", asset, tf)
        return pd.DataFrame()

    out_path = PROCESSED_DIR / tf_cfg["output"]

    # Prefer the primary single-file source
    primary_name = tf_cfg.get("primary")
    if primary_name and (raw_dir / primary_name).exists():
        files = [raw_dir / primary_name]
        logger.info("Using %s as sole %s %s source.", primary_name, asset, tf)
    else:
        # Fallback: glob for any matching files in raw_dir
        exclude = [e.lower() for e in tf_cfg.get("exclude", [])]
        files = sorted([
            f for f in raw_dir.glob(tf_cfg["glob"])
            if not any(e in f.name.lower() for e in exclude)
        ])
        if not files:
            # Last-resort: case-insensitive glob
            files = sorted([
                f for f in raw_dir.glob(tf_cfg["glob"].lower())
                if not any(e in f.name.lower() for e in exclude)
            ])
        if not files:
            suffix = "" if tf == "H1" else f"_{tf}_<dates>"
            logger.warning(
                "No %s %s CSV found in %s. "
                "Export from MT5 History Center and save as data/raw/%s%s.csv, "
                "then re-run --mode consolidate.",
                asset, tf, raw_dir, asset, suffix,
            )
            return pd.DataFrame()

    return _consolidate_files(files, out_path, f"{asset} {tf}")


# ── Utility helpers ───────────────────────────────────────────────────────────

def _detect_bar_frequency(df: pd.DataFrame) -> str:
    """Infer bar frequency from a DataFrame's DatetimeIndex.

    Returns one of ``"M5"``, ``"M15"``, ``"H1"``, ``"D1"``, or ``"unknown"``.
    """
    if len(df) < 2:
        return "unknown"
    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    if len(diffs) == 0:
        return "unknown"
    median_sec = float(diffs.median())
    if median_sec <= 360:
        return "M5"
    if median_sec <= 1800:
        return "M15"
    if median_sec <= 7200:
        return "H1"
    if median_sec <= 90000:
        return "D1"
    return "unknown"


def detect_common_start_date(tf: str) -> pd.Timestamp | None:
    """Return the latest-of-earliest-dates across all assets for *tf*.

    Scans XAUUSD raw file and all new-asset processed master files.  Assets
    whose master file is absent are skipped (graceful degradation — they simply
    won't be merged for those bars).  Only assets present in BOTH sets are
    considered for the intersection.

    Logs each asset's start date so the user can see which asset is limiting.

    Returns None when no files can be read at all.
    """
    tf = tf.upper()
    earliest: dict[str, pd.Timestamp] = {}

    # XAUUSD raw file
    xauusd_path = _XAUUSD_RAW_BY_TF.get(tf)
    if xauusd_path and xauusd_path.exists():
        try:
            sample = pd.read_csv(
                xauusd_path, sep=";", parse_dates=["Date"],
                date_format="%Y.%m.%d %H:%M", nrows=1,
            )
            t = pd.Timestamp(sample["Date"].iloc[0])
            earliest["XAUUSD"] = t
            logger.info("XAUUSD [%s] starts: %s", tf, t.date())
        except Exception as exc:
            logger.debug("Could not read XAUUSD start date: %s", exc)

    # Processed master files for all other assets
    for asset, asset_cfg in ASSET_CONFIGS.items():
        tf_cfg = asset_cfg.get(tf)
        if tf_cfg is None:
            continue
        path = PROCESSED_DIR / tf_cfg["output"]
        if not path.exists():
            logger.debug("%s [%s] master not found — skipping from start-date scan.", asset, tf)
            continue
        try:
            row = pd.read_csv(path, nrows=1, parse_dates=[0])
            t = pd.Timestamp(row.iloc[0, 0])
            earliest[asset] = t
            logger.info("%s [%s] starts: %s", asset, tf, t.date())
        except Exception as exc:
            logger.debug("Could not read %s start date: %s", asset, exc)

    if not earliest:
        return None

    intersection = max(earliest.values())
    limiting = [a for a, t in earliest.items() if t == intersection]
    logger.info(
        "Common start date [%s]: %s  (limiting asset(s): %s)",
        tf, intersection.date(), ", ".join(limiting),
    )
    return intersection


# ── Legacy USDCHF wrappers (kept for backwards compatibility) ─────────────────

def consolidate_usdchf(
    raw_dir:  Path = RAW_DIR,
    out_path: Path = MASTER_PATH,
) -> pd.DataFrame:
    """Build the H1 USDCHF master CSV — thin wrapper around consolidate_asset."""
    return consolidate_asset("USDCHF", "H1", raw_dir=raw_dir)


def consolidate_usdchf_m15(
    raw_dir:  Path = RAW_DIR,
    out_path: Path = MASTER_PATH_M15,
) -> pd.DataFrame:
    """Build the M15 USDCHF master CSV — thin wrapper around consolidate_asset."""
    return consolidate_asset("USDCHF", "M15", raw_dir=raw_dir)


def consolidate_usdchf_m5(
    raw_dir:  Path = RAW_DIR,
    out_path: Path = MASTER_PATH_M5,
) -> pd.DataFrame:
    """Build the M5 USDCHF master CSV — thin wrapper around consolidate_asset."""
    return consolidate_asset("USDCHF", "M5", raw_dir=raw_dir)
