"""Shared lifecycle contract for GoldRegime_X.

Single source of truth for per-TF ATR trail configuration consumed by both
live execution (mt5_trader.py) and the backtester (backtester.py).

Design rules
------------
* All activation thresholds are in USD floating P&L.  R-multiple fields
  do not exist in this module by design.
* Cent-account normalization is centralised here via to_usd() and
  floating_pnl_usd() so every call-site is explicit about the unit
  being USD, not raw broker-currency.
* LIFECYCLE_CONFIG is the canonical reference; ATR_TRAIL_CONFIG in
  mt5_trader.py and backtester.py are rebuilt from this dict (or
  delegated to config_for_tf) so there is no conflicting duplicate.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.risk_manager import CENT_MULTIPLIER


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass — one frozen instance per timeframe
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LifecycleConfig:
    """Per-timeframe lifecycle parameters.

    All monetary threshold fields are denominated in USD floating P&L.
    No R-multiple fields exist here by design.

    Attributes
    ----------
    activation_pnl_usd:
        The floating P&L in USD that triggers Phase 1
        (break-even lock + optional partial close).
    trail_mult:
        Phase 2 ATR trail distance multiplier (ratchet SL).
    partial_close:
        When True, halve position size at activation.
    scalp_target_usd:
        M5 only — close the full position when floating P&L reaches
        this USD level.  None for H1/M15.
    recycle:
        M5 only — allow re-entry in the same regime direction after
        a scalp close without waiting for a regime change.
    """
    activation_pnl_usd: float
    trail_mult: float
    partial_close: bool
    scalp_target_usd: float | None = None
    recycle: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Canonical config — single source of truth
# ─────────────────────────────────────────────────────────────────────────────

LIFECYCLE_CONFIG: dict[str, LifecycleConfig] = {
    "H1":  LifecycleConfig(
        activation_pnl_usd=1.50,
        trail_mult=2.5,
        partial_close=True,
    ),
    "M15": LifecycleConfig(
        activation_pnl_usd=1.50,
        trail_mult=1.5,
        partial_close=True,
    ),
    "M5":  LifecycleConfig(
        activation_pnl_usd=1.00,
        trail_mult=1.5,
        partial_close=False,
        scalp_target_usd=4.00,
        recycle=True,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Access helper
# ─────────────────────────────────────────────────────────────────────────────

def config_for_tf(tf: str) -> LifecycleConfig:
    """Return the LifecycleConfig for *tf* (case-insensitive).

    Raises ValueError for unknown timeframes so misconfiguration is
    caught at startup rather than silently falling back to wrong defaults.
    """
    key = tf.upper()
    if key not in LIFECYCLE_CONFIG:
        raise ValueError(
            f"Missing lifecycle config for timeframe: {key!r}. "
            f"Valid values: {sorted(LIFECYCLE_CONFIG)}"
        )
    return LIFECYCLE_CONFIG[key]


# ─────────────────────────────────────────────────────────────────────────────
# USD conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_usd(value_raw: float, broker: str) -> float:
    """Normalize a raw broker monetary value to USD.

    On Headway Cent accounts all monetary values are reported in cUSD
    (cents).  Dividing by CENT_MULTIPLIER (100) converts to real USD.
    For standard accounts the value is returned unchanged.
    """
    if broker == "headway_cent":
        return value_raw / CENT_MULTIPLIER
    return float(value_raw)


def floating_pnl_usd(raw_open_pnl: float, broker: str) -> float:
    """Convert raw open P&L from broker currency to USD.

    Semantically identical to to_usd(); named explicitly for use at
    trigger-check sites so the intent — this value is compared against a
    USD threshold — is unambiguous at the call site.
    """
    return to_usd(raw_open_pnl, broker)
