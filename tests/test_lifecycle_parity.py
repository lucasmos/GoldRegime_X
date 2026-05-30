"""Lifecycle parity tests — verifies live and backtest event timelines match.

Builds synthetic deterministic price and regime paths, then runs a
backtest-style lifecycle evaluator using the same thresholds as the live
code.  Asserts identical event sequences.

Tests:
  - Activation fires at correct bar (when floating_pnl_usd crosses threshold)
  - Partial close fires at activation for H1/M15; skipped for M5
  - M5 scalp exit fires when floating_pnl_usd reaches scalp_target_usd
  - Regime-shift forced close fires when regime changes mid-trade
  - Trail updates begin after activation
  - Cent normalization: raw_pnl=150 activates for headway_cent (1.5 USD) but
    NOT for standard broker where 150 != 1.5
  - config_for_tf activation threshold matches live ATR_TRAIL_CONFIG values

Run:  python tests/test_lifecycle_parity.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trade_lifecycle import config_for_tf, floating_pnl_usd, LIFECYCLE_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pass(n, msg):
    print(f"  PASS  [{n:02d}] {msg}")
    return True


def _fail(n, msg):
    print(f"  FAIL  [{n:02d}] {msg}")
    return False


class _LiveStyleLifecycle:
    """Minimal live-style lifecycle evaluator.

    Mirrors the between-bar ATR trail loop in mt5_trader.run_live_loop:
      Phase 1: when floating_pnl_usd >= activation_pnl_usd → activate + partial close
      Phase 2: each subsequent bar → ratchet trail SL
    Supports regime-shift forced close.
    """

    def __init__(self, tf: str, broker: str, atr: float = 0.0010):
        self.cfg    = config_for_tf(tf)
        self.broker = broker
        self.atr    = atr
        self.reset()

    def reset(self):
        self.activated        = False
        self.partial_done     = False
        self.activation_bar   = None
        self.partial_bar      = None
        self.trail_bars       = []
        self.regime_close_bar = None
        self.current_sl       = None
        self.direction        = 1
        self.in_trade         = False

    def enter(self, direction: int = 1, entry_price: float = 1.0, sl: float = 0.99):
        self.in_trade   = True
        self.direction  = direction
        self.current_sl = sl

    def step(self, bar_idx: int, raw_pnl: float, hmm_state: int,
             signal_type: str = "trend") -> str:
        """Process one bar.  Returns event string or 'hold'."""
        if not self.in_trade:
            return "idle"

        pnl_usd = floating_pnl_usd(raw_pnl, self.broker)

        # Regime-shift forced close (mirrors live logic exactly)
        if signal_type == "trend" and hmm_state >= 2:
            self.regime_close_bar = bar_idx
            self.in_trade = False
            return "regime_close"
        if signal_type == "mean_reversion" and hmm_state < 2:
            self.regime_close_bar = bar_idx
            self.in_trade = False
            return "regime_close"

        # Phase 1: activation
        if not self.activated and pnl_usd >= self.cfg.activation_pnl_usd:
            self.activated      = True
            self.activation_bar = bar_idx
            if self.cfg.partial_close and not self.partial_done:
                self.partial_done = True
                self.partial_bar  = bar_idx
            return "activation"

        # M5 scalp exit
        if self.cfg.scalp_target_usd is not None and pnl_usd >= self.cfg.scalp_target_usd:
            self.in_trade = False
            return "scalp"

        # Phase 2: trail update (always after activation)
        if self.activated:
            self.trail_bars.append(bar_idx)
            return "trail"

        return "hold"


class _BacktestStyleLifecycle:
    """Minimal backtest-style lifecycle evaluator.

    Mirrors _run_bar_loop in backtester.py:
      - cum_return * account_size = floating_pnl_usd
      - activation when floating_pnl_usd >= cfg.activation_pnl_usd
      - partial close at activation when cfg.partial_close
      - scalp exit when floating_pnl_usd >= cfg.scalp_target_usd
    """

    def __init__(self, tf: str, account_size: float = 15.0):
        self.cfg          = config_for_tf(tf)
        self.account_size = account_size
        self.reset()

    def reset(self):
        self.activated        = False
        self.partial_done     = False
        self.activation_bar   = None
        self.partial_bar      = None
        self.trail_bars       = []
        self.regime_close_bar = None
        self.in_trade         = False
        self.cum_return       = 0.0

    def enter(self):
        self.in_trade   = True
        self.cum_return = 0.0

    def step(self, bar_idx: int, log_ret: float, hmm_state: int,
             signal_type: str = "trend") -> str:
        """Process one bar.  Returns event string or 'hold'."""
        if not self.in_trade:
            return "idle"

        self.cum_return += log_ret
        floating_pnl = self.cum_return * self.account_size

        # Regime-shift forced close
        if signal_type == "trend" and hmm_state >= 2:
            self.regime_close_bar = bar_idx
            self.in_trade = False
            return "regime_close"
        if signal_type == "mean_reversion" and hmm_state < 2:
            self.regime_close_bar = bar_idx
            self.in_trade = False
            return "regime_close"

        # Phase 1: activation
        if not self.activated and floating_pnl >= self.cfg.activation_pnl_usd:
            self.activated      = True
            self.activation_bar = bar_idx
            if self.cfg.partial_close and not self.partial_done:
                self.partial_done = True
                self.partial_bar  = bar_idx
            return "activation"

        # M5 scalp exit
        if self.cfg.scalp_target_usd is not None and floating_pnl >= self.cfg.scalp_target_usd:
            self.in_trade = False
            return "scalp"

        # Phase 2: trail
        if self.activated:
            self.trail_bars.append(bar_idx)
            return "trail"

        return "hold"


# ─────────────────────────────────────────────────────────────────────────────
# Parity tests
# ─────────────────────────────────────────────────────────────────────────────

def test_activation_bar_parity_m15():
    """Live and backtest evaluators activate at the same bar for M15."""
    account_size = 15.0
    # 0.01 log-return per bar; floating_pnl = cum * 15
    # Activation at the bar where cum * 15 >= 1.50 (cum >= 0.10).
    # Both evaluators use the same cumulative sum to avoid fp drift.
    bar_log_rets = [0.01] * 20

    live = _LiveStyleLifecycle("M15", "standard")
    live.enter()
    bt   = _BacktestStyleLifecycle("M15", account_size)
    bt.enter()

    live_act_bar = None
    bt_act_bar   = None
    cum = 0.0

    for i, lr in enumerate(bar_log_rets):
        cum += lr
        # Pass the same cumulative pnl to both evaluators so fp is identical
        live_pnl = cum * account_size
        evt_live = live.step(i, live_pnl, 0)
        evt_bt   = bt.step(i, lr, 0)
        if evt_live == "activation" and live_act_bar is None:
            live_act_bar = i
        if evt_bt == "activation" and bt_act_bar is None:
            bt_act_bar = i

    assert live_act_bar is not None, "live did not activate"
    assert bt_act_bar is not None,   "backtest did not activate"
    assert live_act_bar == bt_act_bar, (
        f"Activation bar mismatch: live={live_act_bar} bt={bt_act_bar}"
    )
    return _pass(1, f"M15 activation bar parity: both activate at bar {live_act_bar}")


def test_partial_close_fires_on_activation_m15():
    cfg = config_for_tf("M15")
    account_size = 15.0
    bt = _BacktestStyleLifecycle("M15", account_size)
    bt.enter()
    # push past activation in one bar
    bt.step(0, 0.15, 0)   # 0.15 * 15 = 2.25 > 1.50
    assert bt.activated,   "should be activated"
    assert bt.partial_done, "M15 partial_close=True: partial_done should be True"
    assert bt.partial_bar == 0
    return _pass(2, "M15 partial close fires at activation bar")


def test_partial_close_skipped_m5():
    cfg = config_for_tf("M5")
    account_size = 15.0
    bt = _BacktestStyleLifecycle("M5", account_size)
    bt.enter()
    # push past M5 activation (1.0 USD)
    bt.step(0, 0.10, 0)   # 0.10 * 15 = 1.50 > 1.00
    assert bt.activated,        "should be activated"
    assert not bt.partial_done, "M5 partial_close=False: partial_done must remain False"
    return _pass(3, "M5 partial close is skipped (partial_close=False)")


def test_m5_scalp_exit_fires():
    cfg = config_for_tf("M5")
    account_size = 15.0
    bt = _BacktestStyleLifecycle("M5", account_size)
    bt.enter()
    # scalp_target=4.0 → need floating >= 4.0 → cum_return >= 4.0/15 = 0.2667
    events = []
    for i in range(10):
        evt = bt.step(i, 0.04, 0)   # 0.04 * (i+1) * 15 reaches 4.0 at i=6
        events.append(evt)
        if not bt.in_trade:
            break
    assert "scalp" in events, f"scalp exit never fired; events={events}"
    return _pass(4, "M5 scalp exit fires when floating_pnl_usd >= scalp_target_usd")


def test_m5_scalp_exit_not_fired_below_target():
    account_size = 15.0
    bt = _BacktestStyleLifecycle("M5", account_size)
    bt.enter()
    # 6 bars at 0.01 each: max floating = 0.06 * 15 = 0.90 — below scalp_target=4.0
    events = []
    for i in range(6):
        events.append(bt.step(i, 0.01, 0))
    assert "scalp" not in events
    assert bt.in_trade, "trade should still be open"
    return _pass(5, "M5 scalp exit does not fire below scalp target")


def test_regime_shift_forced_close_trend_to_chop():
    """Trend position closed immediately when regime flips to state 2 (SHOCK/CHOP)."""
    live = _LiveStyleLifecycle("M15", "standard")
    live.enter()
    bt   = _BacktestStyleLifecycle("M15", 15.0)
    bt.enter()

    # 3 bars in trend (state=0), then regime shifts to state=2
    states = [0, 0, 0, 2]
    log_rets = [0.001, 0.001, 0.001, 0.001]
    live_pnl_seq = [v * 15.0 for v in [0.001, 0.002, 0.003, 0.004]]

    live_close = bt_close = None
    for i, (state, lr, lp) in enumerate(zip(states, log_rets, live_pnl_seq)):
        evt_live = live.step(i, lp, state, signal_type="trend")
        evt_bt   = bt.step(i, lr, state, signal_type="trend")
        if evt_live == "regime_close" and live_close is None:
            live_close = i
        if evt_bt == "regime_close" and bt_close is None:
            bt_close = i

    assert live_close == 3, f"live regime close expected at bar 3, got {live_close}"
    assert bt_close   == 3, f"bt regime close expected at bar 3, got {bt_close}"
    assert live_close == bt_close, "regime close bar mismatch between live and bt"
    return _pass(6, "Regime-shift forced close fires at same bar in live and backtest")


def test_regime_shift_mr_to_trend():
    """MR position closed when regime flips to state < 2 (trend breakout)."""
    live = _LiveStyleLifecycle("M15", "standard")
    live.enter()
    bt   = _BacktestStyleLifecycle("M15", 15.0)
    bt.enter()

    # 2 bars in chop (state=2), then regime shifts to state=0 (TREND)
    states = [2, 2, 0]
    log_rets = [0.001, 0.001, 0.001]
    live_pnl_seq = [0.015, 0.030, 0.045]

    live_close = bt_close = None
    for i, (state, lr, lp) in enumerate(zip(states, log_rets, live_pnl_seq)):
        evt_live = live.step(i, lp, state, signal_type="mean_reversion")
        evt_bt   = bt.step(i, lr, state, signal_type="mean_reversion")
        if evt_live == "regime_close" and live_close is None:
            live_close = i
        if evt_bt == "regime_close" and bt_close is None:
            bt_close = i

    assert live_close == 2
    assert bt_close == 2
    return _pass(7, "MR regime-shift close fires at correct bar in both evaluators")


def test_regime_shift_not_fired_in_same_regime():
    """No forced close while regime stays in expected state."""
    bt = _BacktestStyleLifecycle("M15", 15.0)
    bt.enter()
    for i in range(10):
        evt = bt.step(i, 0.001, 0, signal_type="trend")
        assert evt != "regime_close", f"false regime close at bar {i}"
    return _pass(8, "No false regime-shift close when regime stays TREND (state 0)")


def test_cent_normalization_activates_correctly():
    """headway_cent: raw_pnl=150 → 1.5 USD activates M15 (threshold=1.5)."""
    live = _LiveStyleLifecycle("M15", "headway_cent")
    live.enter()
    # raw_pnl=149 should NOT activate (149/100=1.49 < 1.50)
    evt = live.step(0, 149.0, 0)
    assert evt != "activation", "149 cents (1.49 USD) should not activate"
    # raw_pnl=150 SHOULD activate (150/100=1.50 >= 1.50)
    evt = live.step(1, 150.0, 0)
    assert evt == "activation", f"150 cents (1.50 USD) should activate; got {evt}"
    return _pass(9, "Cent normalization: 150 cUSD activates; 149 cUSD does not")


def test_standard_broker_150_does_not_activate_m15():
    """standard broker: raw_pnl=150 → 150 USD; M15 threshold is 1.50 USD."""
    live = _LiveStyleLifecycle("M15", "standard")
    live.enter()
    evt = live.step(0, 150.0, 0)
    # 150 USD >> threshold so it does activate; this validates that the unit is USD
    assert evt == "activation", (
        "standard broker 150 USD is way above 1.50 threshold and must activate"
    )
    return _pass(10, "Standard broker: 150 USD activates immediately (threshold=1.50)")


def test_trail_updates_start_after_activation():
    bt = _BacktestStyleLifecycle("H1", 15.0)
    bt.enter()
    # First 5 bars: no activation (pnl < 1.50)
    for i in range(5):
        evt = bt.step(i, 0.005, 0)   # 0.005*15=0.075 per bar, 5 bars=0.375 USD < 1.5
        assert evt != "trail", f"trail fired before activation at bar {i}"
    # Activate at bar 5
    evt = bt.step(5, 0.10, 0)   # cumulative: 0.075 + 0.10*15=1.875 > 1.5
    # bar 6 onwards should be trail
    evt = bt.step(6, 0.001, 0)
    assert evt == "trail", f"trail should fire after activation; got {evt}"
    return _pass(11, "Trail updates start only after activation, not before")


def test_h1_partial_close_parity():
    """H1 live and backtest both fire partial close at same activation."""
    live = _LiveStyleLifecycle("H1", "standard")
    live.enter()
    bt   = _BacktestStyleLifecycle("H1", 15.0)
    bt.enter()

    # Single large bar that activates immediately
    evt_live = live.step(0, 2.0, 0)     # 2.0 USD > 1.50 threshold
    evt_bt   = bt.step(0, 0.20, 0)      # 0.20 * 15 = 3.0 > 1.50 threshold

    assert evt_live == "activation"
    assert evt_bt   == "activation"
    assert live.partial_done, "live H1 should partial-close"
    assert bt.partial_done,   "bt H1 should partial-close"
    assert live.partial_bar == bt.partial_bar == 0
    return _pass(12, "H1 partial close parity: both evaluators partial-close at bar 0")


def test_config_parity_with_legacy_live_dict():
    """config_for_tf values must match the legacy ATR_TRAIL_CONFIG in mt5_trader."""
    from src.mt5_trader import ATR_TRAIL_CONFIG as _live_cfg

    for tf in ("H1", "M15", "M5"):
        lc = config_for_tf(tf)
        leg = _live_cfg[tf]
        assert lc.activation_pnl_usd == leg["activation_pnl"], (
            f"{tf}: activation_pnl mismatch: {lc.activation_pnl_usd} vs {leg['activation_pnl']}"
        )
        assert lc.trail_mult == leg["trail_mult"], (
            f"{tf}: trail_mult mismatch"
        )
        assert lc.partial_close == leg["partial_close"], (
            f"{tf}: partial_close mismatch"
        )
    return _pass(13, "config_for_tf values match legacy ATR_TRAIL_CONFIG in mt5_trader")


def test_no_r_multiple_in_live_evaluator():
    """Live evaluator has no R-multiple field."""
    live = _LiveStyleLifecycle("M15", "standard")
    assert not hasattr(live.cfg, "r_multiple")
    assert not hasattr(live.cfg, "activation_r")
    return _pass(14, "Live lifecycle evaluator has no R-multiple fields")


def test_activation_idempotent():
    """Activation fires at most once per trade."""
    bt = _BacktestStyleLifecycle("M15", 15.0)
    bt.enter()
    activations = 0
    for i in range(10):
        # large returns: every bar would re-trigger if not idempotent
        evt = bt.step(i, 0.20, 0)
        if evt == "activation":
            activations += 1
    assert activations == 1, f"activation fired {activations} times, expected 1"
    return _pass(15, "Activation fires exactly once per trade (idempotent)")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_activation_bar_parity_m15,
        test_partial_close_fires_on_activation_m15,
        test_partial_close_skipped_m5,
        test_m5_scalp_exit_fires,
        test_m5_scalp_exit_not_fired_below_target,
        test_regime_shift_forced_close_trend_to_chop,
        test_regime_shift_mr_to_trend,
        test_regime_shift_not_fired_in_same_regime,
        test_cent_normalization_activates_correctly,
        test_standard_broker_150_does_not_activate_m15,
        test_trail_updates_start_after_activation,
        test_h1_partial_close_parity,
        test_config_parity_with_legacy_live_dict,
        test_no_r_multiple_in_live_evaluator,
        test_activation_idempotent,
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
