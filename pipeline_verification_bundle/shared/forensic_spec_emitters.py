"""
Forensic spec emitters for GoldRegime_X pipeline observability.

This module implements the per-timeframe report artifacts required by the
FINAL IMPLEMENTATION PROMPT (Complete Forensic Observability):

    Phase 5  -> feature_drift_report.csv   (PSI, KS, Mean/Std/Skew/Kurtosis shift, SAFE/WARNING/CRITICAL)
    Phase 6  -> hmm_regime_report_M15.csv / _M5.csv
    Phase 7  -> probability_report_M15.csv / _M5.csv (+ summary stats)
    Phase 11 -> survival_analysis.csv
    Phase 12 -> pipeline waterfall (text)
    Phase 13 -> model_registry.json
    Phase 15 -> marginal-loss root-cause table
    Phase 17 -> self-verification table

These are pure functions operating on the tested CandidateLedger / HMMSnapshot
data structures defined in pipeline_observability.py.  They are imported and
called from PipelineObservability.finalize().
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# The canonical spec stage list (Phase 11 / 12 / 15).
SPEC_SURVIVAL_STAGES: Tuple[str, ...] = (
    "Generated", "FeatureEngineering", "Session", "TBM",
    "HMM", "Probability", "Risk", "Executed",
)
SPEC_STAGE_LABEL: Dict[str, str] = {
    "Generated": "Generated",
    "FeatureEngineering": "Feature Engineering",
    "Session": "Session",
    "TBM": "TBM",
    "HMM": "HMM",
    "Probability": "Probability",
    "Risk": "Risk",
    "Executed": "Executed",
}

# Full internal stage order (mirrors STAGE_ORDER in pipeline_observability.py).
_INTERNAL_STAGE_ORDER: Tuple[str, ...] = (
    "Generated", "FeatureEngineering", "Session", "TBM",
    "HMM", "Probability", "Threshold", "Risk", "Executed",
)


def _mkparent(path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# =========================================================================
# PHASE 11 -- Survival Analysis
# =========================================================================
def survival_counts_for_tf(ledger, tf: str) -> Dict[str, int]:
    """Monotone survival count per spec stage for one timeframe.

    A candidate 'survives through' stage S iff it was never rejected, or its
    rejection stage comes strictly after S in the canonical order.
    """
    traces = ledger.all_for_tf(tf)
    order_index = {s: i for i, s in enumerate(_INTERNAL_STAGE_ORDER)}
    counts: Dict[str, int] = {}
    total = len(traces)
    for stage in SPEC_SURVIVAL_STAGES:
        if stage == "Generated":
            counts[stage] = total
            continue
        if stage == "Executed":
            counts[stage] = sum(1 for t in traces if t.executed)
            continue
        si = order_index[stage]
        surviving = 0
        for t in traces:
            rej = getattr(t, "rejection_stage", None)
            if rej is None:
                surviving += 1
            elif order_index.get(rej, len(_INTERNAL_STAGE_ORDER)) > si:
                surviving += 1
        counts[stage] = surviving
    return counts


def write_survival_analysis_csv(
    ledger, path, tfs: Sequence[str] = ("M15", "M5"),
) -> Tuple[Path, Optional[str], float]:
    """Phase 11: Stage | M15 Count | M15 % | M5 Count | M5 % | Difference.

    Returns (path, max_divergence_stage, max_divergence_pp).
    """
    tfs = [str(t).upper() for t in tfs]
    counts = {tf: survival_counts_for_tf(ledger, tf) for tf in tfs}
    gen = {tf: max(counts[tf]["Generated"], 0) for tf in tfs}
    rows: List[Dict[str, Any]] = []
    best_stage: Optional[str] = None
    best_gap = -1.0
    for stage in SPEC_SURVIVAL_STAGES:
        row: Dict[str, Any] = {"Stage": SPEC_STAGE_LABEL[stage]}
        pcts: Dict[str, float] = {}
        for tf in tfs:
            c = counts[tf][stage]
            g = gen[tf]
            pct = (c / g * 100.0) if g else 0.0
            row["%s Count" % tf] = int(c)
            row["%s %%" % tf] = round(pct, 2)
            pcts[tf] = pct
        if len(tfs) >= 2:
            diff = round(pcts[tfs[0]] - pcts[tfs[1]], 2)
        else:
            diff = 0.0
        row["Difference"] = diff
        if stage != "Generated" and abs(diff) > best_gap:
            best_gap = abs(diff)
            best_stage = SPEC_STAGE_LABEL[stage]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["Max Divergence"] = [">>> MAX" if r == best_stage else "" for r in df["Stage"]]
    p = _mkparent(path)
    df.to_csv(p, index=False)
    return p, best_stage, (best_gap if best_gap >= 0 else 0.0)


# =========================================================================
# PHASE 12 -- Pipeline Waterfall (text)
# =========================================================================
def render_pipeline_waterfall(ledger, tf: str) -> str:
    tf = str(tf).upper()
    counts = survival_counts_for_tf(ledger, tf)
    gen = max(counts["Generated"], 1)
    parts = []
    for stage in SPEC_SURVIVAL_STAGES:
        c = counts[stage]
        pct = (c / gen * 100.0)
        parts.append("%s: %d (%.1f%%)" % (SPEC_STAGE_LABEL[stage], c, pct))
    return ("[%s] " % tf) + "  \u2193  ".join(parts)


# =========================================================================
# PHASE 15 -- Marginal-Loss Root Cause Analysis
# =========================================================================
def build_marginal_loss_table(ledger, tf: str) -> Tuple[pd.DataFrame, Optional[str], float]:
    """Stage | Remaining | Lost | Marginal Loss %.  Returns (df, bottleneck, loss%)."""
    tf = str(tf).upper()
    counts = survival_counts_for_tf(ledger, tf)
    rows: List[Dict[str, Any]] = []
    prev: Optional[int] = None
    bottleneck: Optional[str] = None
    worst = -1.0
    for stage in SPEC_SURVIVAL_STAGES:
        remaining = counts[stage]
        if prev is None:
            lost = 0
            marg = 0.0
        else:
            lost = prev - remaining
            marg = (lost / prev * 100.0) if prev > 0 else 0.0
        rows.append({
            "Stage": SPEC_STAGE_LABEL[stage],
            "Remaining": int(remaining),
            "Lost": int(lost),
            "Marginal Loss %": round(marg, 2),
        })
        if stage != "Generated" and marg > worst:
            worst = marg
            bottleneck = SPEC_STAGE_LABEL[stage]
        prev = remaining
    return pd.DataFrame(rows), bottleneck, (worst if worst >= 0 else 0.0)


def write_marginal_loss_csv(ledger, path, tf: str) -> Path:
    df, _, _ = build_marginal_loss_table(ledger, tf)
    p = _mkparent(path)
    df.to_csv(p, index=False)
    return p


# =========================================================================
# PHASE 6 -- HMM Regime Report (per timeframe)
# =========================================================================
def write_hmm_regime_report(ledger, snapshot, tf, path) -> Path:
    """Per-state occupancy, transition matrix, mean duration, trades
    generated/accepted/rejected, and mean XGB probability for one timeframe.

    A trace with no recorded HMM state is bucketed under 'unassigned' so the
    report is never empty when candidates exist.
    """
    tf = str(tf).upper()
    traces = ledger.all_for_tf(tf)

    def _label(v):
        if v is None:
            return "unassigned"
        try:
            iv = int(v)
        except Exception:
            return "unassigned"
        return iv if iv >= 0 else "unassigned"

    # Collect state labels from snapshot + ledger.
    labels: List[Any] = []
    seen = set()
    if snapshot is not None:
        for s in sorted(snapshot.state_counts.keys()):
            if int(s) not in seen:
                labels.append(int(s)); seen.add(int(s))
    for t in traces:
        lab = _label(getattr(t, "hmm_state", None))
        key = ("u" if lab == "unassigned" else lab)
        if key not in seen:
            labels.append(lab); seen.add(key)
    if not labels:
        labels = ["unassigned"]

    tmat = snapshot.transition_matrix if snapshot is not None else []
    n_states = len(tmat) if tmat else 0

    rows: List[Dict[str, Any]] = []
    for lab in labels:
        if lab == "unassigned":
            s_traces = [t for t in traces if _label(getattr(t, "hmm_state", None)) == "unassigned"]
        else:
            s_traces = [t for t in traces if _label(getattr(t, "hmm_state", None)) == lab]
        generated = len(s_traces)
        accepted = sum(1 for t in s_traces if getattr(t, "hmm_pass", False))
        rejected = generated - accepted
        probs = [float(t.xgb_probability) for t in s_traces
                 if getattr(t, "xgb_probability", None) is not None
                 and not (isinstance(t.xgb_probability, float) and math.isnan(t.xgb_probability))]
        mean_prob = round(float(np.mean(probs)), 6) if probs else float("nan")
        occ = float("nan")
        dwell = float("nan")
        if snapshot is not None and isinstance(lab, int):
            occ = round(float(snapshot.state_occupancy_pct.get(lab, float("nan"))), 4)
            dwell = round(float(snapshot.mean_dwell_time.get(lab, float("nan"))), 4)
        row: Dict[str, Any] = {
            "HMM State": lab,
            "Occupancy %": occ,
            "Mean Regime Duration": dwell,
            "Trades Generated": generated,
            "Trades Accepted": accepted,
            "Trades Rejected": rejected,
            "Mean XGB Probability": mean_prob,
        }
        if tmat and isinstance(lab, int) and lab < n_states:
            for j in range(n_states):
                row["P(->%d)" % j] = round(float(tmat[lab][j]), 6)
        rows.append(row)

    base_cols = ["HMM State", "Occupancy %", "Mean Regime Duration",
                 "Trades Generated", "Trades Accepted", "Trades Rejected",
                 "Mean XGB Probability"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=base_cols)
    p = _mkparent(path)
    df.to_csv(p, index=False)
    return p


# =========================================================================
# PHASE 7 -- Probability Report (per timeframe) + summary stats
# =========================================================================
def _prob_summary(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        keys = ("n", "min", "q1", "median", "mean", "q3", "p95", "p99", "max")
        return {k: (0.0 if k == "n" else float("nan")) for k in keys}
    return {
        "n": float(a.size),
        "min": float(np.min(a)),
        "q1": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "q3": float(np.quantile(a, 0.75)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def write_probability_report(ledger, tf, path) -> Tuple[Path, Dict[str, float]]:
    """Per-candidate probability rows + returns summary-stats dict.

    Columns: Candidate ID | Strategy | HMM State | Raw Probability |
             Threshold | Accepted.
    """
    tf = str(tf).upper()
    traces = ledger.all_for_tf(tf)
    rows: List[Dict[str, Any]] = []
    probs: List[float] = []
    for t in traces:
        raw = getattr(t, "xgb_probability", None)
        thr = getattr(t, "threshold", None)
        if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
            probs.append(float(raw))
        rows.append({
            "Candidate ID": t.candidate_id,
            "Strategy": t.strategy,
            "HMM State": (t.hmm_state if t.hmm_state is not None else ""),
            "Raw Probability": (round(float(raw), 6) if raw is not None and not (isinstance(raw, float) and math.isnan(raw)) else ""),
            "Threshold": (round(float(thr), 6) if thr is not None and not (isinstance(thr, float) and math.isnan(thr)) else ""),
            "Accepted": bool(getattr(t, "threshold_pass", False)),
        })
    cols = ["Candidate ID", "Strategy", "HMM State", "Raw Probability", "Threshold", "Accepted"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    p = _mkparent(path)
    df.to_csv(p, index=False)
    return p, _prob_summary(probs)


# =========================================================================
# PHASE 13 -- Model Registry
# =========================================================================
def write_model_registry_json(
    path,
    tfs: Sequence[str],
    uuid_report: Optional[Dict[str, Any]] = None,
    model_components: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_hashes: Optional[Dict[str, str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    manifest_data: Optional[Dict[str, Any]] = None,
) -> Path:
    uuid_report = uuid_report or {}
    model_components = model_components or {}
    feature_hashes = feature_hashes or {}
    thresholds = thresholds or {}
    manifest_data = manifest_data or {}

    registry: Dict[str, Any] = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "pipeline_version": manifest_data.get("pipeline_version", "unknown"),
        "timeframes": {},
    }
    for tf in [str(t).upper() for t in tfs]:
        comp = dict(model_components.get(tf, {}))
        uinfo = uuid_report.get(tf, {}) if isinstance(uuid_report, dict) else {}
        uuids = uinfo.get("uuids", {}) if isinstance(uinfo, dict) else {}
        entry = {
            "model_uuid": comp.get("model_uuid") or uuids.get("training"),
            "hmm_uuid": comp.get("hmm_uuid"),
            "xgboost_uuid": comp.get("xgboost_uuid"),
            "feature_hash": comp.get("feature_hash") or feature_hashes.get(tf) or manifest_data.get("feature_hash"),
            "threshold_version": comp.get("threshold_version") or (
                ("thr-%.4f" % thresholds[tf]) if tf in thresholds and thresholds[tf] is not None else None
            ),
            "model_hash": comp.get("model_hash") or manifest_data.get("model_hash"),
            "uuid_consistency": uinfo.get("status", "MISSING") if isinstance(uinfo, dict) else "MISSING",
        }
        present = sum(1 for v in (entry["model_uuid"], entry["hmm_uuid"],
                                  entry["xgboost_uuid"], entry["feature_hash"],
                                  entry["threshold_version"]) if v)
        entry["status"] = "PASS" if present >= 3 else ("PARTIAL" if present else "MISSING")
        registry["timeframes"][tf] = entry
    p = _mkparent(path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, default=str)
    return p


# =========================================================================
# PHASE 5 -- Feature Drift with Skew + Kurtosis and SAFE/WARNING/CRITICAL
# =========================================================================
def _safe_moment(arr: np.ndarray, order: int) -> float:
    a = arr[~np.isnan(arr)]
    if a.size < 2:
        return float("nan")
    mu = a.mean()
    sd = a.std()
    if sd == 0:
        return 0.0
    z = (a - mu) / sd
    if order == 3:
        return float(np.mean(z ** 3))
    if order == 4:
        return float(np.mean(z ** 4) - 3.0)  # excess kurtosis
    return float("nan")


def feature_drift_report_full(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    feature_columns: Sequence[str],
    psi_fn,
    ks_fn,
    psi_warn: float = 0.10,
    psi_crit: float = 0.25,
    ks_warn: float = 0.10,
) -> pd.DataFrame:
    """Phase 5: PSI, KS, Mean/Std/Skew/Kurtosis shift + SAFE/WARNING/CRITICAL."""
    rows: List[Dict[str, Any]] = []
    for f in feature_columns:
        if f not in is_df.columns or f not in oos_df.columns:
            continue
        is_v = np.asarray(is_df[f], dtype=float)
        oos_v = np.asarray(oos_df[f], dtype=float)
        psi = psi_fn(is_v, oos_v)
        ks = ks_fn(is_v, oos_v)
        is_c = is_v[~np.isnan(is_v)]
        oos_c = oos_v[~np.isnan(oos_v)]
        mean_shift = float(oos_c.mean() - is_c.mean()) if is_c.size and oos_c.size else float("nan")
        std_shift = float(oos_c.std() - is_c.std()) if is_c.size and oos_c.size else float("nan")
        skew_shift = _safe_moment(oos_v, 3) - _safe_moment(is_v, 3)
        kurt_shift = _safe_moment(oos_v, 4) - _safe_moment(is_v, 4)
        if not math.isnan(psi) and psi >= psi_crit:
            drift_class = "CRITICAL"
        elif (not math.isnan(psi) and psi >= psi_warn) or (not math.isnan(ks) and ks >= ks_warn):
            drift_class = "WARNING"
        else:
            drift_class = "SAFE"
        rows.append({
            "feature": f,
            "psi": round(psi, 6) if not math.isnan(psi) else float("nan"),
            "ks": round(ks, 6) if not math.isnan(ks) else float("nan"),
            "mean_shift": round(mean_shift, 6) if not math.isnan(mean_shift) else float("nan"),
            "std_shift": round(std_shift, 6) if not math.isnan(std_shift) else float("nan"),
            "skew_shift": round(skew_shift, 6) if not math.isnan(skew_shift) else float("nan"),
            "kurtosis_shift": round(kurt_shift, 6) if not math.isnan(kurt_shift) else float("nan"),
            "drift_class": drift_class,
        })
    cols = ["feature", "psi", "ks", "mean_shift", "std_shift",
            "skew_shift", "kurtosis_shift", "drift_class"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


# =========================================================================
# PHASE 17 -- Self-verification table
# =========================================================================
REQUIRED_ARTIFACTS: Tuple[str, ...] = (
    "candidate_decisions.csv",
    "pipeline_audit.json",
    "feature_drift_report.csv",
    "hmm_regime_report_M15.csv",
    "hmm_regime_report_M5.csv",
    "probability_report_M15.csv",
    "probability_report_M5.csv",
    "session_audit.csv",
    "candidate_integrity.csv",
    "top100_rejected_M15.csv",
    "top100_rejected_M5.csv",
    "survival_analysis.csv",
    "model_registry.json",
)


def _has_data(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            return bool(obj)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            return len(df) > 0
        return path.stat().st_size > 0
    except Exception:
        return False


def build_self_verification_table(
    output_dir,
    manifest_path: Optional[Any] = None,
) -> pd.DataFrame:
    """Phase 17: Artifact | Exists | Contains Data for every required file."""
    out = Path(output_dir)
    rows: List[Dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        p = out / name
        exists = p.exists()
        rows.append({
            "Artifact": name,
            "Exists": "YES" if exists else "NO",
            "Contains Data": "YES" if _has_data(p) else "NO",
        })
    # pipeline_manifest.json lives at reports/ root, not observability/.
    if manifest_path is not None:
        mp = Path(manifest_path)
        rows.append({
            "Artifact": "pipeline_manifest.json",
            "Exists": "YES" if mp.exists() else "NO",
            "Contains Data": "YES" if _has_data(mp) else "NO",
        })
    return pd.DataFrame(rows)
