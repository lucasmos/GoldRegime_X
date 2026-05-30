"""
tests/test_three_state_contract.py
===================================
Patch5 — Three-State Regime Contract Enforcement.

Verifies that exactly three semantic states (TREND=0, MEAN_REVERSION=1,
VOLATILITY_SHOCK=2) are enforced across optimizer, engine_hmm, engine_xgb,
and main.py, and that any attempt to use 4+ states raises immediately.

All tests are self-contained and require no external data or models.
"""

import sys
import types
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── helpers ──────────────────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0


def _pass(n: int, msg: str) -> str:
    global _PASS
    _PASS += 1
    print(f"  PASS  [{n:02d}] {msg}")
    return "pass"


def _fail(n: int, msg: str) -> str:
    global _FAIL
    _FAIL += 1
    print(f"  FAIL  [{n:02d}] {msg}")
    return "fail"


def _check(n: int, condition: bool, pass_msg: str, fail_msg: str = "") -> str:
    return _pass(n, pass_msg) if condition else _fail(n, fail_msg or pass_msg)


# ── Test 1-3: canonical constants in engine_hmm ──────────────────────────────

def test_canonical_ids():
    from src.engine_hmm import CANONICAL_REGIME_ID, REGIME_TREND, REGIME_MR, REGIME_SHOCK
    _check(1, CANONICAL_REGIME_ID[REGIME_TREND] == 0,  "TREND id == 0")
    _check(2, CANONICAL_REGIME_ID[REGIME_MR]    == 1,  "MEAN_REVERSION id == 1")
    _check(3, CANONICAL_REGIME_ID[REGIME_SHOCK] == 2,  "VOLATILITY_SHOCK id == 2")


# ── Test 4-5: exactly 3 keys in canonical maps ───────────────────────────────

def test_canonical_map_size():
    from src.engine_hmm import CANONICAL_REGIME_ID, STATE_NAMES
    _check(4, len(CANONICAL_REGIME_ID) == 3, "CANONICAL_REGIME_ID has exactly 3 entries")
    _check(5, len(STATE_NAMES) == 3,          "STATE_NAMES has exactly 3 entries")


# ── Test 6: _assert_canonical_states passes {0,1,2} ──────────────────────────

def test_assert_canonical_states_ok():
    from src.engine_hmm import _assert_canonical_states
    try:
        _assert_canonical_states(np.array([0, 1, 2, 0, 1]), "test_ok")
        _pass(6, "_assert_canonical_states accepts {{0,1,2}}")
    except RuntimeError as e:
        _fail(6, f"_assert_canonical_states wrongly raised: {e}")


# ── Test 7: _assert_canonical_states raises on state 3 ───────────────────────

def test_assert_canonical_states_raises_on_3():
    from src.engine_hmm import _assert_canonical_states
    try:
        _assert_canonical_states(np.array([0, 1, 2, 3]), "test_bad")
        _fail(7, "_assert_canonical_states should have raised on state 3")
    except RuntimeError:
        _pass(7, "_assert_canonical_states raises RuntimeError on state id 3")


# ── Test 8: _assert_canonical_states raises on state 4 ───────────────────────

def test_assert_canonical_states_raises_on_4():
    from src.engine_hmm import _assert_canonical_states
    try:
        _assert_canonical_states(np.array([0, 1, 4]), "test_bad4")
        _fail(8, "_assert_canonical_states should have raised on state 4")
    except RuntimeError:
        _pass(8, "_assert_canonical_states raises RuntimeError on state id 4")


# ── Test 9-10: resolve_n_states rejects non-3 ────────────────────────────────

def test_resolve_n_states_accepts_3():
    from src.optimizer import resolve_n_states
    result = resolve_n_states("H1", {"n_states": 3})
    _check(9, result == 3, "resolve_n_states returns 3 for n_states=3")


def test_resolve_n_states_rejects_4():
    from src.optimizer import resolve_n_states
    try:
        resolve_n_states("H1", {"n_states": 4})
        _fail(10, "resolve_n_states should raise on n_states=4")
    except ValueError:
        _pass(10, "resolve_n_states raises ValueError on n_states=4")


# ── Test 11: resolve_n_states rejects 2 ──────────────────────────────────────

def test_resolve_n_states_rejects_2():
    from src.optimizer import resolve_n_states
    try:
        resolve_n_states("M15", {"n_states": 2})
        _fail(11, "resolve_n_states should raise on n_states=2")
    except ValueError:
        _pass(11, "resolve_n_states raises ValueError on n_states=2")


# ── Test 12: resolve_n_states defaults to 3 when key absent ──────────────────

def test_resolve_n_states_default():
    from src.optimizer import resolve_n_states
    result = resolve_n_states("M5", {})
    _check(12, result == 3, "resolve_n_states defaults to 3 when n_states absent")


# ── Test 13-14: _enforce_three_states ────────────────────────────────────────

def test_enforce_three_states_ok():
    from src.optimizer import _enforce_three_states
    n = _enforce_three_states({"n_states": 3}, "test_ctx")
    _check(13, n == 3, "_enforce_three_states returns 3 for valid input")


def test_enforce_three_states_raises():
    from src.optimizer import _enforce_three_states
    try:
        _enforce_three_states({"n_states": 4}, "test_bad_ctx")
        _fail(14, "_enforce_three_states should raise on n_states=4")
    except ValueError:
        _pass(14, "_enforce_three_states raises ValueError on n_states=4")


# ── Test 15: SEARCH_SPACES n_states pinned to (3,3,"int") ────────────────────

def test_search_spaces_n_states_pinned():
    from src.optimizer import SEARCH_SPACES
    ok = True
    for tf_key, space in SEARCH_SPACES.items():
        if "n_states" in space:
            lo, hi, kind = space["n_states"]
            if not (lo == 3 and hi == 3 and kind == "int"):
                ok = False
    _check(15, ok, "SEARCH_SPACES n_states is (3,3,'int') in all TF spaces")


# ── Test 16: backtester CHOP_STATE == 2 (== SHOCK_STATE) ─────────────────────

def test_backtester_chop_state():
    from src.backtester import CHOP_STATE, SHOCK_STATE
    _check(16, CHOP_STATE == 2 and SHOCK_STATE == 2,
           "backtester CHOP_STATE == SHOCK_STATE == 2")


# ── Test 17: main.py resolve_n_states rejects 4 ──────────────────────────────

def test_main_resolve_n_states_rejects_4():
    import main as m
    try:
        m.resolve_n_states("H1", {"n_states": 4})
        _fail(17, "main.resolve_n_states should raise on n_states=4")
    except ValueError:
        _pass(17, "main.resolve_n_states raises ValueError on n_states=4")


# ── Test 18: STATE_NAMES_3 reverse-maps 0→TREND, 1→MEAN_REVERSION, 2→VOLATILITY_SHOCK

def test_state_names_3_labels():
    from src.engine_hmm import STATE_NAMES_3
    _check(18,
           STATE_NAMES_3[0] == "TREND" and
           STATE_NAMES_3[1] == "MEAN_REVERSION" and
           STATE_NAMES_3[2] == "VOLATILITY_SHOCK",
           "STATE_NAMES_3 maps 0→TREND, 1→MEAN_REVERSION, 2→VOLATILITY_SHOCK")


# ── Test 19: STATE_POLICY allows TREND and SHOCK, blocks MR ──────────────────

def test_state_policy():
    from src.engine_hmm import STATE_POLICY, REGIME_TREND, REGIME_MR, REGIME_SHOCK
    _check(19,
           STATE_POLICY[REGIME_TREND] is True and
           STATE_POLICY[REGIME_MR]    is False and
           STATE_POLICY[REGIME_SHOCK] is True,
           "STATE_POLICY: TREND=True, MEAN_REVERSION=False, VOLATILITY_SHOCK=True")


# ── Test 20: _assert_canonical_states empty array is OK ──────────────────────

def test_assert_canonical_states_empty():
    from src.engine_hmm import _assert_canonical_states
    try:
        _assert_canonical_states(np.array([], dtype=int), "empty")
        _pass(20, "_assert_canonical_states accepts empty array")
    except RuntimeError as e:
        _fail(20, f"_assert_canonical_states raised on empty: {e}")


# ── Test 21: _assert_canonical_states with only state 0 ──────────────────────

def test_assert_canonical_states_single_state():
    from src.engine_hmm import _assert_canonical_states
    try:
        _assert_canonical_states(np.array([0, 0, 0]), "single")
        _pass(21, "_assert_canonical_states accepts all-0 array")
    except RuntimeError as e:
        _fail(21, f"unexpected error: {e}")


# ── Test 22: _canonical_frozenset contains exactly {0,1,2} ───────────────────

def test_canonical_frozenset():
    from src.engine_hmm import _CANONICAL_STATE_IDS
    _check(22, _CANONICAL_STATE_IDS == frozenset({0, 1, 2}),
           "_CANONICAL_STATE_IDS == frozenset({0,1,2})")


# ── Test 23: optimizer HARD_TRADE_FLOORS defined for H1, M15, M5 ─────────────

def test_hard_trade_floors_tfs():
    from src.optimizer import HARD_TRADE_FLOORS
    _check(23,
           "H1" in HARD_TRADE_FLOORS and
           "M15" in HARD_TRADE_FLOORS and
           "M5" in HARD_TRADE_FLOORS,
           "HARD_TRADE_FLOORS defined for H1, M15, M5")


# ── Test 24: resolve_n_states all TFs return 3 ───────────────────────────────

def test_resolve_n_states_all_tfs():
    from src.optimizer import resolve_n_states
    results = {tf: resolve_n_states(tf, {"n_states": 3}) for tf in ["H1", "M15", "M5"]}
    _check(24, all(v == 3 for v in results.values()),
           "resolve_n_states returns 3 for H1, M15, M5")


# ── Test 25: SEARCH_SPACES contains H1, M15, M5 ──────────────────────────────

def test_search_spaces_tfs():
    from src.optimizer import SEARCH_SPACES
    _check(25,
           "H1" in SEARCH_SPACES and
           "M15" in SEARCH_SPACES and
           "M5" in SEARCH_SPACES,
           "SEARCH_SPACES contains H1, M15, M5 entries")


# ── Test 26: no 4-state field names in engine_xgb FEATURE_COLS ───────────────

def test_engine_xgb_no_4state_features():
    from src.engine_xgb import FEATURE_COLS
    bad = [f for f in FEATURE_COLS if "state_3" in f.lower() or "state_4" in f.lower()]
    _check(26, len(bad) == 0,
           f"FEATURE_COLS contains no state_3/state_4 fields (found: {bad})")


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_canonical_ids,
        test_canonical_map_size,
        test_assert_canonical_states_ok,
        test_assert_canonical_states_raises_on_3,
        test_assert_canonical_states_raises_on_4,
        test_resolve_n_states_accepts_3,
        test_resolve_n_states_rejects_4,
        test_resolve_n_states_rejects_2,
        test_resolve_n_states_default,
        test_enforce_three_states_ok,
        test_enforce_three_states_raises,
        test_search_spaces_n_states_pinned,
        test_backtester_chop_state,
        test_main_resolve_n_states_rejects_4,
        test_state_names_3_labels,
        test_state_policy,
        test_assert_canonical_states_empty,
        test_assert_canonical_states_single_state,
        test_canonical_frozenset,
        test_hard_trade_floors_tfs,
        test_resolve_n_states_all_tfs,
        test_search_spaces_tfs,
        test_engine_xgb_no_4state_features,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            _fail(0, f"{t.__name__} raised unexpectedly: {e}")

    total = _PASS + _FAIL
    print(f"\n{_PASS}/{total} tests passed")
    sys.exit(0 if _FAIL == 0 else 1)
