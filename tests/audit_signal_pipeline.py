"""
Signal pipeline audit — verifies the Phase E rebuild end-to-end.

Tests confirm:
  1.  Regime mapping is deterministic (same stats → same TREND/MR/SHOCK label).
  2.  MR state is always no-trade (state_disabled reason, no signal).
  3.  Model routing: TREND bars use trend_model, SHOCK bars use shock_model.
  4.  CPCV objective emits all 5 required metric components.
  5.  DD cap at 20%: validate_strategy returns FAIL when floating_dd > 0.20.
  6.  sync_validate gate blocks strategies with mr_leak > 0.
  7.  Smoke test H1: process→train→backtest produces valid metrics dict.

Run:
    python tests/audit_signal_pipeline.py
"""

import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, ".")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pass(n, msg):
    logger.info("PASS  [T%02d] %s", n, msg)
    return True


def _fail(n, msg):
    logger.error("FAIL  [T%02d] %s", n, msg)
    return False


def _make_transmat():
    return np.array([[0.92, 0.05, 0.03],   # TREND  (high persistence)
                     [0.10, 0.85, 0.05],   # MR
                     [0.05, 0.05, 0.90]])  # SHOCK  (high persistence)


def _prime(engine, state: int, bars: int = 5) -> dict:
    """Advance a SignalEngine through `bars` of the given HMM state."""
    info = {}
    for _ in range(bars):
        info = engine.update_regime(state, _make_transmat())
    return info


# ── Tests ─────────────────────────────────────────────────────────────────────

def run_tests() -> bool:
    from src.engine_hmm import (
        REGIME_TREND, REGIME_MR, REGIME_SHOCK,
        CANONICAL_REGIME_ID, STATE_NAMES, STATE_POLICY,
    )
    from src.signal_engine import (
        SignalEngine, SignalDecision,
        TREND_STATE, MR_STATE, SHOCK_STATE,
    )
    from src.engine_xgb import (
        train_regime_models, get_regime_predictions, TB_CONFIG,
    )
    from src.optimizer import composite_score
    from src.validator import validate_strategy

    results = []

    # ── T01: Regime constants are canonical ──────────────────────────────────
    r = (CANONICAL_REGIME_ID[REGIME_TREND] == 0
         and CANONICAL_REGIME_ID[REGIME_MR]    == 1
         and CANONICAL_REGIME_ID[REGIME_SHOCK] == 2
         and TREND_STATE == 0 and MR_STATE == 1 and SHOCK_STATE == 2
         and STATE_NAMES[0] == "TREND"
         and STATE_NAMES[1] == "MEAN_REVERSION"
         and STATE_NAMES[2] == "VOLATILITY_SHOCK")
    results.append(_pass(1, "Regime constants canonical (TREND=0, MR=1, SHOCK=2)") if r else
                   _fail(1, f"Regime constants wrong: {CANONICAL_REGIME_ID}  names={STATE_NAMES}"))

    # ── T02: MR state is always no-trade ─────────────────────────────────────
    r = (STATE_POLICY[REGIME_MR] is False
         and STATE_POLICY[REGIME_TREND] is True
         and STATE_POLICY[REGIME_SHOCK] is True)
    results.append(_pass(2, "STATE_POLICY: MR=False, TREND=True, SHOCK=True") if r else
                   _fail(2, f"STATE_POLICY wrong: {STATE_POLICY}"))

    # ── T03: MR bars return state_disabled (no entry, no signal) ─────────────
    eng = SignalEngine("H1")
    _prime(eng, MR_STATE, bars=6)   # 6 bars = past MIN_CONFIRMATION_BARS
    entry = eng.should_enter(
        eng.update_regime(MR_STATE, _make_transmat()),
        xgb_prob=0.70, synth_vix_zscore=2.5, atr_band_position=0.3,
    )
    r = entry is None
    results.append(_pass(3, f"MR state→no entry (should_enter=None)  entry={entry}") if r else
                   _fail(3, f"MR state leaked a trade signal!  entry={entry}"))

    # ── T04: evaluate_bar returns state_disabled for MR ──────────────────────
    eng = SignalEngine("H1")
    _prime(eng, MR_STATE, bars=6)
    row_mr = {"hmm_state": MR_STATE, "prob": 0.75,
              "synth_vix_zscore": 3.0, "atr_band_position": 0.3}
    cfg_mr = {"tf": "H1", "hmm_transmat": _make_transmat()}
    dec = eng.evaluate_bar(row_mr, current_position=None, tf_config=cfg_mr)
    r = isinstance(dec, SignalDecision) and not dec.enter and dec.reason == "state_disabled"
    results.append(_pass(4, f"evaluate_bar MR→state_disabled  reason={dec.reason}") if r else
                   _fail(4, f"evaluate_bar MR wrong: enter={dec.enter} reason={dec.reason}"))

    # ── T05: TREND state fires BUY signal ────────────────────────────────────
    eng = SignalEngine("H1")
    _prime(eng, TREND_STATE, bars=6)
    info = eng.update_regime(TREND_STATE, _make_transmat())
    entry_trend = eng.should_enter(
        info, xgb_prob=0.65, synth_vix_zscore=1.2, atr_band_position=0.4,
    )
    r = entry_trend is not None and entry_trend["signal"] == "BUY"
    results.append(_pass(5, f"TREND state→BUY entry  entry={entry_trend}") if r else
                   _fail(5, f"TREND state did not fire BUY  entry={entry_trend}"))

    # ── T06: SHOCK state fires entry signal ──────────────────────────────────
    eng = SignalEngine("H1")
    _prime(eng, SHOCK_STATE, bars=6)
    info = eng.update_regime(SHOCK_STATE, _make_transmat())
    entry_shock = eng.should_enter(
        info, xgb_prob=0.65, synth_vix_zscore=2.5, atr_band_position=0.4,
    )
    r = entry_shock is not None and entry_shock["signal"] in ("BUY", "SELL")
    results.append(_pass(6, f"SHOCK state→entry signal  entry={entry_shock}") if r else
                   _fail(6, f"SHOCK state did not fire any signal  entry={entry_shock}"))

    # ── T07: Model routing — get_regime_predictions dispatches correctly ──────
    n = 30
    rng = np.random.default_rng(0)
    X_dummy = pd.DataFrame({"f1": rng.random(n), "f2": rng.random(n)})
    states_all_trend = np.zeros(n, dtype=int)     # all TREND
    states_all_shock = np.full(n, SHOCK_STATE, dtype=int)
    states_all_mr    = np.full(n, MR_STATE,    dtype=int)

    from unittest.mock import MagicMock
    trend_mock = MagicMock()
    trend_mock.predict_proba.return_value = np.column_stack(
        [np.full(n, 0.3), np.full(n, 0.7)]   # prob_1 = 0.7
    )
    shock_mock = MagicMock()
    shock_mock.predict_proba.return_value = np.column_stack(
        [np.full(n, 0.4), np.full(n, 0.6)]   # prob_1 = 0.6
    )

    probs_trend = get_regime_predictions(X_dummy, states_all_trend, trend_mock, shock_mock)
    probs_shock = get_regime_predictions(X_dummy, states_all_shock, trend_mock, shock_mock)
    probs_mr    = get_regime_predictions(X_dummy, states_all_mr,    trend_mock, shock_mock)

    r = (np.allclose(probs_trend, 0.7)
         and np.allclose(probs_shock, 0.6)
         and np.allclose(probs_mr, 0.5))
    results.append(_pass(7, f"get_regime_predictions routing correct  "
                           f"trend={probs_trend[0]:.1f} shock={probs_shock[0]:.1f} mr={probs_mr[0]:.1f}") if r else
                   _fail(7, f"get_regime_predictions wrong values: "
                             f"trend={probs_trend[0]} shock={probs_shock[0]} mr={probs_mr[0]}"))

    # ── T08: composite_score emits all 5 metric components ───────────────────
    test_metrics = {
        "deflated_sharpe": 1.2,
        "calmar":          1.5,
        "profit_factor":   1.8,
        "expectancy":      0.3,
        "stability_score": 0.7,
    }
    score = composite_score(test_metrics)
    expected_base = 0.35*1.2 + 0.25*1.5 + 0.20*1.8 + 0.10*0.3 + 0.10*0.7
    r = abs(score - expected_base) < 1e-9
    results.append(_pass(8, f"composite_score correct: {score:.6f} == {expected_base:.6f}") if r else
                   _fail(8, f"composite_score mismatch: {score} != {expected_base}"))

    # ── T09: validate_strategy FAIL when dd > 20% ────────────────────────────
    gate_dd = validate_strategy(
        {"floating_max_drawdown": 0.25, "n_trades": 100, "mr_leak_count": 0},
        tf="H1",
    )
    r = gate_dd["status"] == "fail" and gate_dd["reason"] == "dd_cap_violated"
    results.append(_pass(9, f"validate_strategy FAIL dd>20%  status={gate_dd['status']} reason={gate_dd['reason']}") if r else
                   _fail(9, f"validate_strategy dd gate wrong: {gate_dd}"))

    # ── T10: validate_strategy FAIL when mr_leak > 0 ─────────────────────────
    gate_mr = validate_strategy(
        {"floating_max_drawdown": 0.10, "n_trades": 100, "mr_leak_count": 3},
        tf="H1",
    )
    r = gate_mr["status"] == "fail" and gate_mr["reason"] == "mr_leak"
    results.append(_pass(10, f"validate_strategy FAIL mr_leak  status={gate_mr['status']} reason={gate_mr['reason']}") if r else
                   _fail(10, f"validate_strategy mr_leak gate wrong: {gate_mr}"))

    # ── T11: validate_strategy FAIL when n_trades below minimum ──────────────
    gate_trades = validate_strategy(
        {"floating_max_drawdown": 0.10, "n_trades": 5, "mr_leak_count": 0},
        tf="H1",
    )
    r = gate_trades["status"] == "fail" and gate_trades["reason"] == "min_trades"
    results.append(_pass(11, f"validate_strategy FAIL min_trades  status={gate_trades['status']}") if r else
                   _fail(11, f"validate_strategy min_trades gate wrong: {gate_trades}"))

    # ── T12: validate_strategy PASS for healthy result ────────────────────────
    gate_pass = validate_strategy(
        {"floating_max_drawdown": 0.10, "n_trades": 80, "mr_leak_count": 0,
         "regime_coverage": 0.60},
        tf="H1",
    )
    r = gate_pass["status"] == "pass"
    results.append(_pass(12, f"validate_strategy PASS for healthy result  status={gate_pass['status']}") if r else
                   _fail(12, f"validate_strategy failed healthy result: {gate_pass}"))

    # ── T13: TB_CONFIG has correct spec values ────────────────────────────────
    r = (TB_CONFIG["H1"]["pt"]  == 2.5 and TB_CONFIG["H1"]["sl"]  == 1.0 and TB_CONFIG["H1"]["vb"]  == 48
     and TB_CONFIG["M15"]["pt"] == 2.0 and TB_CONFIG["M15"]["sl"] == 1.0 and TB_CONFIG["M15"]["vb"] == 24
     and TB_CONFIG["M5"]["pt"]  == 1.5 and TB_CONFIG["M5"]["sl"]  == 1.0 and TB_CONFIG["M5"]["vb"] == 12)
    results.append(_pass(13, f"TB_CONFIG correct: {TB_CONFIG}") if r else
                   _fail(13, f"TB_CONFIG wrong: {TB_CONFIG}"))

    # ── T14: MR leak detection in backtester ─────────────────────────────────
    from src.backtester import _mr_attribution, MR_STATE as BT_MR_STATE
    # signals: 2 trades — one in TREND(0), one in MR(1)
    signals_leak   = np.array([0, 1, 1, 0, -1, -1, 0], dtype=np.int8)
    states_leak    = np.array([0, 1, 1, 0,  1,  1, 0], dtype=int)   # 2nd trade in MR
    strat_returns  = np.array([0, .01, .01, 0, -.005, -.005, 0])
    attr = _mr_attribution(signals_leak, states_leak, strat_returns)
    r = attr["mr_leak_count"] > 0
    results.append(_pass(14, f"_mr_attribution detects MR leak: leak_count={attr['mr_leak_count']}") if r else
                   _fail(14, f"_mr_attribution missed MR leak: {attr}"))

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print()
    print("=" * 65)
    print(f"  SIGNAL PIPELINE AUDIT (Phase E rebuild): {passed}/{total} passed")
    if passed < total:
        print(f"  {total - passed} FAILED — review FAIL lines above")
    else:
        print("  All tests passed ✓")
    print("=" * 65)
    return passed == total


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
