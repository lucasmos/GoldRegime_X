"""Smoke test that exercises every phase of pipeline_observability.

Run with:  python3 -m pytest tests/test_observability_smoke.py -q
Or just:   python3 tests/test_observability_smoke.py

Generates a synthetic 2-timeframe pipeline with realistic drop-off at each
stage and verifies that all artifacts are produced and non-empty.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the sibling `shared/` package importable when run as a script.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from shared.pipeline_observability import PipelineObservability  # noqa: E402


def _run_synthetic_pipeline(obs: PipelineObservability, tf: str, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base_ts = pd.Timestamp("2025-01-06 00:00:00")
    for i in range(n):
        cid = int(i) + (0 if tf == "M15" else 1_000_000)
        ts = base_ts + pd.Timedelta(minutes=(15 if tf == "M15" else 5) * i)
        strategy = "Trend Pullback" if i % 3 else "Breakout"
        obs.record_generation(cid, tf, ts, strategy)

        # Session gate: ~35% rejected outside London_NY
        in_session = 7 <= ts.hour < 21
        obs.record_session(cid, tf, in_session,
                           reason="" if in_session else "Outside London_NY")
        if not in_session:
            continue

        # TBM gate: ~15% rejected
        tbm_ok = rng.random() > 0.15
        obs.record_tbm(cid, tf, tbm_ok, reason="" if tbm_ok else "Label=NaN")
        if not tbm_ok:
            continue

        # HMM: M15 spends more time in state 2 (untradeable)
        if tf == "M15":
            state = int(rng.choice([0, 1, 2], p=[0.25, 0.30, 0.45]))
        else:
            state = int(rng.choice([0, 1, 2], p=[0.35, 0.40, 0.25]))
        hmm_ok = state != 2
        obs.record_hmm(cid, tf, hmm_ok, state=state,
                       reason="" if hmm_ok else "State 2 not tradeable")
        if not hmm_ok:
            continue

        # Probability gate
        p = float(rng.beta(2, 5))
        thr = 0.55 if tf == "M15" else 0.50
        prob_ok = p >= thr
        obs.record_probability(cid, tf, prob_ok, probability=p, threshold=thr)
        if not prob_ok:
            continue

        # Risk gate
        risk_ok = rng.random() > 0.05
        obs.record_risk(cid, tf, risk_ok, reason="" if risk_ok else "Spread cap")
        if not risk_ok:
            continue

        obs.mark_executed(cid, tf)

    # HMM snapshot from the ledger's recorded states
    states = [t.hmm_state for t in obs.ledger.all_for_tf(tf) if t.hmm_state >= 0]
    if states:
        obs.record_hmm_inference(tf, states=states)

    # Probability snapshots (synthetic)
    obs.record_probability_snapshot(tf, "raw", rng.beta(2, 5, size=200))
    obs.record_probability_snapshot(tf, "post_hmm", rng.beta(2.5, 4, size=140))
    obs.record_probability_snapshot(tf, "post_threshold", rng.beta(4, 3, size=45))

    # Feature drift (synthetic)
    is_df = pd.DataFrame({
        "rsi5": rng.normal(50, 10, 500),
        "atr14": rng.normal(1.0, 0.2, 500),
        "volatility_20": rng.normal(0.005, 0.001, 500),
    })
    oos_df = pd.DataFrame({
        "rsi5": rng.normal(52, 12, 200),
        "atr14": rng.normal(1.5, 0.3, 200),   # drift
        "volatility_20": rng.normal(0.007, 0.002, 200),  # drift
    })
    obs.record_feature_distributions(tf, is_df, oos_df,
                                     feature_cols=["rsi5", "atr14", "volatility_20"])

    # Reconciliation
    execd = sum(1 for t in obs.ledger.all_for_tf(tf) if t.executed)
    obs.record_reconciliation(tf, {
        "Generated": len(obs.ledger.all_for_tf(tf)),
        "Exported":  len(obs.ledger.all_for_tf(tf)),
        "Imported":  len(obs.ledger.all_for_tf(tf)),
        "Processed": execd + 3,   # simulate loss between processed & executed
        "Executed":  execd,
    })

    obs.record_model_hash(tf, model_hash="deadbeefcafe%s" % tf.lower())


def main() -> None:
    out = _ROOT / "tests" / "_smoke_out"
    if out.exists():
        shutil.rmtree(out)

    obs = PipelineObservability(
        output_dir=out,
        expected_session_by_tf={"M15": "London_NY", "M5": "London_NY"},
    )
    _run_synthetic_pipeline(obs, "M15", n=800, seed=1)
    _run_synthetic_pipeline(obs, "M5", n=2400, seed=2)

    result = obs.finalize(
        integrity_flags={
            "Candidate Integrity":  "PASS",
            "Model Integrity":      "PASS",
            "Train/OOS Separation": "PASS",
        },
        verbose=True,
    )

    # Assertions -- every artifact exists and is non-empty.
    for key, path in result.items():
        if key in ("run_id", "survival_gap_warnings"):
            continue
        if path is None:
            continue
        p = Path(path)
        assert p.exists(), "missing artifact: %s" % p
        assert p.stat().st_size > 0, "empty artifact: %s" % p

    # Decision log contains rows and every FAIL has a non-empty reason.
    decisions = pd.read_csv(out / "candidate_decisions.csv")
    assert len(decisions) > 0
    fails = decisions[decisions["decision"] == "FAIL"]
    assert (fails["reason"].astype(str).str.strip() != "").all(), \
        "a FAIL row is missing an explicit reason"

    # Survival matrix has both timeframes and monotone non-increasing counts.
    matrix = pd.read_csv(out / "stage_survival.csv")
    for tf in ("M15", "M5"):
        col = "%s Remaining" % tf
        assert col in matrix.columns
        vals = matrix[col].tolist()
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)), \
            "%s survival is not monotone non-increasing" % tf

    # Audit JSON parses and has the shape we advertised.
    with open(out / "pipeline_audit.json") as fh:
        payload = json.load(fh)
    for key in ("run_id", "integrity_flags", "survival_matrix",
                "per_timeframe", "total_candidates", "dashboard"):
        assert key in payload, "missing key in audit json: %s" % key
    for tf in ("M15", "M5"):
        assert tf in payload["per_timeframe"]
        tf_block = payload["per_timeframe"][tf]
        assert tf_block["generated"] > 0
        assert tf_block["hmm_snapshot"] is not None
        assert tf_block["threshold"] is not None

    # Lost trades report has entries (M15 should be starved).
    lost = (out / "lost_trades_m15.txt").read_text()
    assert "Candidate" in lost

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
