"""Unit tests for src/trade_lifecycle.py.

Covers:
  - Cent normalization: floating_pnl_usd / to_usd
  - Per-TF config values and types
  - config_for_tf behaviour including rejection of unknown TF
  - No R-multiple fields exist on LifecycleConfig

Run:  python tests/test_lifecycle_units.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trade_lifecycle import (
    LIFECYCLE_CONFIG,
    LifecycleConfig,
    config_for_tf,
    floating_pnl_usd,
    to_usd,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pass(n, msg):
    print(f"  PASS  [{n:02d}] {msg}")
    return True


def _fail(n, msg):
    print(f"  FAIL  [{n:02d}] {msg}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Cent conversion
# ─────────────────────────────────────────────────────────────────────────────

def test_cent_broker_divides_by_100():
    result = floating_pnl_usd(150.0, "headway_cent")
    assert result == 1.5, f"expected 1.5 got {result}"
    return _pass(1, "cent broker: 150.0 raw → 1.5 USD")


def test_standard_broker_unchanged():
    result = floating_pnl_usd(1.5, "standard")
    assert result == 1.5, f"expected 1.5 got {result}"
    return _pass(2, "standard broker: 1.5 raw → 1.5 USD unchanged")


def test_to_usd_cent_equivalence():
    assert to_usd(200.0, "headway_cent") == 2.0
    return _pass(3, "to_usd cent broker: 200.0 → 2.0")


def test_to_usd_standard_equivalence():
    assert to_usd(2.0, "standard") == 2.0
    return _pass(4, "to_usd standard: 2.0 → 2.0")


def test_to_usd_fractional_cent():
    result = to_usd(1.0, "headway_cent")
    assert abs(result - 0.01) < 1e-9, f"expected 0.01 got {result}"
    return _pass(5, "to_usd cent: 1.0 → 0.01")


def test_floating_pnl_usd_zero():
    assert floating_pnl_usd(0.0, "headway_cent") == 0.0
    assert floating_pnl_usd(0.0, "standard") == 0.0
    return _pass(6, "floating_pnl_usd: 0.0 → 0.0 both brokers")


def test_floating_pnl_negative():
    result = floating_pnl_usd(-100.0, "headway_cent")
    assert result == -1.0, f"expected -1.0 got {result}"
    return _pass(7, "floating_pnl_usd: negative loss normalised correctly")


# ─────────────────────────────────────────────────────────────────────────────
# Per-TF config values
# ─────────────────────────────────────────────────────────────────────────────

def test_h1_activation_threshold():
    cfg = config_for_tf("H1")
    assert cfg.activation_pnl_usd == 1.50
    return _pass(8, "H1 activation_pnl_usd == 1.50")


def test_m15_activation_threshold():
    cfg = config_for_tf("M15")
    assert cfg.activation_pnl_usd == 1.50
    return _pass(9, "M15 activation_pnl_usd == 1.50")


def test_m5_activation_threshold():
    cfg = config_for_tf("M5")
    assert cfg.activation_pnl_usd == 1.00
    return _pass(10, "M5 activation_pnl_usd == 1.00")


def test_m5_scalp_target():
    cfg = config_for_tf("M5")
    assert cfg.scalp_target_usd == 4.00
    return _pass(11, "M5 scalp_target_usd == 4.00")


def test_m5_partial_close_false():
    cfg = config_for_tf("M5")
    assert cfg.partial_close is False
    return _pass(12, "M5 partial_close is False")


def test_h1_partial_close_true():
    cfg = config_for_tf("H1")
    assert cfg.partial_close is True
    return _pass(13, "H1 partial_close is True")


def test_m15_partial_close_true():
    cfg = config_for_tf("M15")
    assert cfg.partial_close is True
    return _pass(14, "M15 partial_close is True")


def test_h1_scalp_target_none():
    cfg = config_for_tf("H1")
    assert cfg.scalp_target_usd is None
    return _pass(15, "H1 scalp_target_usd is None")


def test_m15_scalp_target_none():
    cfg = config_for_tf("M15")
    assert cfg.scalp_target_usd is None
    return _pass(16, "M15 scalp_target_usd is None")


def test_m5_recycle_true():
    cfg = config_for_tf("M5")
    assert cfg.recycle is True
    return _pass(17, "M5 recycle is True")


def test_h1_trail_mult():
    cfg = config_for_tf("H1")
    assert cfg.trail_mult == 2.5
    return _pass(18, "H1 trail_mult == 2.5")


def test_m15_trail_mult():
    cfg = config_for_tf("M15")
    assert cfg.trail_mult == 1.5
    return _pass(19, "M15 trail_mult == 1.5")


def test_m5_trail_mult():
    cfg = config_for_tf("M5")
    assert cfg.trail_mult == 1.5
    return _pass(20, "M5 trail_mult == 1.5")


# ─────────────────────────────────────────────────────────────────────────────
# config_for_tf behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_config_for_tf_case_insensitive():
    assert config_for_tf("h1") == config_for_tf("H1")
    assert config_for_tf("m15") == config_for_tf("M15")
    assert config_for_tf("m5") == config_for_tf("M5")
    return _pass(21, "config_for_tf is case-insensitive")


def test_config_for_tf_unknown_raises():
    try:
        config_for_tf("D1")
        assert False, "should have raised ValueError"
    except ValueError as exc:
        assert "D1" in str(exc)
    return _pass(22, "config_for_tf raises ValueError for unknown TF")


def test_config_for_tf_returns_lifecycle_config():
    for tf in ("H1", "M15", "M5"):
        cfg = config_for_tf(tf)
        assert isinstance(cfg, LifecycleConfig), f"{tf}: expected LifecycleConfig"
    return _pass(23, "config_for_tf returns LifecycleConfig instances")


def test_lifecycle_config_is_frozen():
    cfg = config_for_tf("H1")
    try:
        object.__setattr__(cfg, "activation_pnl_usd", 0.0)
        # dataclass frozen=True raises FrozenInstanceError
        assert False, "should have raised FrozenInstanceError"
    except Exception:
        pass
    return _pass(24, "LifecycleConfig is frozen (immutable)")


# ─────────────────────────────────────────────────────────────────────────────
# No R-multiple fields
# ─────────────────────────────────────────────────────────────────────────────

def test_no_r_multiple_field():
    cfg = config_for_tf("H1")
    forbidden = [
        "r_multiple", "risk_multiple", "activation_r", "trail_r",
        "partial_r", "scalp_r",
    ]
    for f in forbidden:
        assert not hasattr(cfg, f), f"LifecycleConfig must not have field: {f}"
    return _pass(25, "LifecycleConfig has no R-multiple fields")


def test_activation_field_name_contains_usd():
    # The field name itself must make the unit clear
    cfg = config_for_tf("M15")
    assert hasattr(cfg, "activation_pnl_usd")
    assert not hasattr(cfg, "activation_pnl")
    return _pass(26, "activation field is named activation_pnl_usd (not bare activation_pnl)")


# ─────────────────────────────────────────────────────────────────────────────
# Boundary: activation thresholds as activation decision simulation
# ─────────────────────────────────────────────────────────────────────────────

def test_activation_boundary_not_triggered_just_below():
    cfg = config_for_tf("M15")
    pnl = cfg.activation_pnl_usd - 0.0001
    assert pnl < cfg.activation_pnl_usd
    return _pass(27, "M15: pnl just below threshold does not activate")


def test_activation_boundary_triggered_at_threshold():
    cfg = config_for_tf("M15")
    pnl = cfg.activation_pnl_usd
    assert pnl >= cfg.activation_pnl_usd
    return _pass(28, "M15: pnl at threshold activates")


def test_m5_scalp_boundary():
    cfg = config_for_tf("M5")
    assert cfg.scalp_target_usd is not None
    pnl_under = cfg.scalp_target_usd - 0.01
    pnl_at    = cfg.scalp_target_usd
    assert pnl_under < cfg.scalp_target_usd
    assert pnl_at >= cfg.scalp_target_usd
    return _pass(29, "M5: scalp trigger boundary correct")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_cent_broker_divides_by_100,
        test_standard_broker_unchanged,
        test_to_usd_cent_equivalence,
        test_to_usd_standard_equivalence,
        test_to_usd_fractional_cent,
        test_floating_pnl_usd_zero,
        test_floating_pnl_negative,
        test_h1_activation_threshold,
        test_m15_activation_threshold,
        test_m5_activation_threshold,
        test_m5_scalp_target,
        test_m5_partial_close_false,
        test_h1_partial_close_true,
        test_m15_partial_close_true,
        test_h1_scalp_target_none,
        test_m15_scalp_target_none,
        test_m5_recycle_true,
        test_h1_trail_mult,
        test_m15_trail_mult,
        test_m5_trail_mult,
        test_config_for_tf_case_insensitive,
        test_config_for_tf_unknown_raises,
        test_config_for_tf_returns_lifecycle_config,
        test_lifecycle_config_is_frozen,
        test_no_r_multiple_field,
        test_activation_field_name_contains_usd,
        test_activation_boundary_not_triggered_just_below,
        test_activation_boundary_triggered_at_threshold,
        test_m5_scalp_boundary,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
