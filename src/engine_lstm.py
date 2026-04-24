"""
LSTM Regime Classifier for GoldRegime_X.

Predicts HMM state labels (Bull/Bear/Chop) from 100-bar sequences.
Ensembled with the HMM at inference time to give early warning of regime
transitions before the HMM commits to a new state.

Architecture:
    Input  (100, n_feats)
    → LSTM(64, return_sequences=True) + Dropout(0.3)
    → LSTM(32) + Dropout(0.3)
    → Dense(16, relu) + Dropout(0.2)
    → Dense(n_states, softmax, name='regime_output')

Save layout:
    models/lstm/{TF}_{broker}/lstm_regime_classifier.keras
    models/lstm/{TF}_{broker}/lstm_feature_scaler.pkl
    models/lstm/{TF}_{broker}/lstm_metadata.json
"""

import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.logger import setup_logger

logger = setup_logger(__name__)

# Backwards-compat: engine_xgb.py defines its own LSTM_CONTEXT_COLS; this stub
# prevents ImportError if anything still references the old name.
LSTM_CONTEXT_COLS: list[str] = []

LSTM_WINDOW   = 100
LSTM_N_STATES = 4     # default; overridden per model by n_states arg
STATE_NAMES   = {0: "Bull", 1: "Bear", 2: "Chop_Low", 3: "Chop_High"}


def get_lstm_dir(tf: str, broker: str = "headway_cent") -> Path:
    return Path(f"models/lstm/{tf.upper()}_{broker}")


# ── Feature helpers ────────────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LSTM-specific derived columns on a copy of df."""
    df = df.copy()

    if "rsi" in df.columns:
        df["rsi_normalized"] = df["rsi"] / 100.0
    else:
        df["rsi_normalized"] = 0.5

    if "Volume" in df.columns:
        vol_ma = df["Volume"].rolling(20, min_periods=1).mean()
        df["volume_ratio"] = (df["Volume"] / vol_ma.replace(0, np.nan).fillna(1)).clip(0.1, 5.0)
    else:
        df["volume_ratio"] = 1.0

    if "Close" in df.columns:
        ma     = df["Close"].rolling(20, min_periods=1).mean()
        std    = df["Close"].rolling(20, min_periods=1).std().fillna(0)
        bb_rng = (4 * std).replace(0, np.nan)
        df["bb_position"] = ((df["Close"] - (ma - 2 * std)) / bb_rng).clip(0, 1).fillna(0.5)

        sma50  = df["Close"].rolling(50, min_periods=1).mean()
        df["dist_from_sma50"] = ((df["Close"] - sma50) / sma50).fillna(0.0)
    else:
        df["bb_position"]    = 0.5
        df["dist_from_sma50"] = 0.0

    return df


_INPUT_COLUMNS = [
    "log_return",
    "volatility",
    "rsi_normalized",
    "atr_normalized",
    "volume_ratio",
    "bb_position",
    "gmm_vol_cluster",
    "dist_from_sma50",
]


# ── Model class ───────────────────────────────────────────────────────────────

class LSTMRegimeClassifier:
    """LSTM that classifies market regime from bar sequences.

    Trained to predict HMM state labels.  At inference time the prediction is
    ensembled with the HMM state to detect regime transitions early.
    """

    def __init__(self, sequence_length: int = LSTM_WINDOW, n_states: int = LSTM_N_STATES):
        self.sequence_length = sequence_length
        self.n_states        = n_states
        self._model          = None
        self.feature_scaler  = None
        self.input_columns   = list(_INPUT_COLUMNS)

    # ── Architecture ──────────────────────────────────────────────────────────

    def _build_model(self, n_feats: int):
        from keras import Input
        from keras.layers import LSTM, Dense, Dropout
        from keras.models import Model
        from keras.optimizers import Adam

        inp = Input(shape=(self.sequence_length, n_feats), name="price_sequence")
        x   = LSTM(64, return_sequences=True,  name="lstm_1")(inp)
        x   = Dropout(0.3, name="dropout_1")(x)
        x   = LSTM(32, return_sequences=False, name="lstm_2")(x)
        x   = Dropout(0.3, name="dropout_2")(x)
        x   = Dense(16, activation="relu", name="dense_1")(x)
        x   = Dropout(0.2, name="dropout_3")(x)
        out = Dense(self.n_states, activation="softmax", name="regime_output")(x)

        model = Model(inputs=inp, outputs=out, name="lstm_regime_classifier")
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Training ──────────────────────────────────────────────────────────────

    def _prepare_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (X, y_onehot, y_raw) from a featurised DataFrame."""
        from keras.utils import to_categorical
        from sklearn.preprocessing import StandardScaler

        df = _add_derived_features(df)

        if "hmm_state" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'hmm_state' column. "
                "Pass HMM states via load_data_with_hmm_labels() before training."
            )

        active_cols = [c for c in self.input_columns if c in df.columns]
        df = df.dropna(subset=active_cols + ["hmm_state"])

        # Clamp labels to valid range
        df = df[df["hmm_state"].isin(range(self.n_states))].copy()

        logger.info("LSTM regime classifier — %d bars  states: %s", len(df),
                    dict(df["hmm_state"].value_counts().sort_index()))

        for s in range(self.n_states):
            cnt = (df["hmm_state"] == s).sum()
            logger.info("  State %d (%s): %d bars (%.1f%%)",
                        s, STATE_NAMES.get(s, "?"), cnt, cnt / len(df) * 100)

        feat  = df[active_cols].values.astype(np.float64)
        feat  = np.nan_to_num(feat, nan=0.0)
        self.feature_scaler = StandardScaler()
        feat  = self.feature_scaler.fit_transform(feat).astype(np.float32)

        labels = df["hmm_state"].values.astype(int)

        X, y = [], []
        for i in range(self.sequence_length, len(feat)):
            X.append(feat[i - self.sequence_length: i])
            y.append(labels[i])

        X       = np.array(X, dtype=np.float32)
        y_raw   = np.array(y, dtype=np.int32)
        y_onehot = to_categorical(y_raw, num_classes=self.n_states)

        logger.info("Sequences: %d  shape: %s", len(X), X.shape)
        return X, y_onehot, y_raw

    def fit(
        self,
        df: pd.DataFrame,
        tf: str = "H1",
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.1,
    ):
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau

        X, y_onehot, y_raw = self._prepare_sequences(df)

        self._model = self._build_model(X.shape[2])

        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history = self._model.fit(
            X, y_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        val_acc      = history.history["val_accuracy"][-1]
        train_acc    = history.history["accuracy"][-1]
        majority_pct = np.bincount(y_raw, minlength=self.n_states).max() / len(y_raw)

        logger.info("LSTM regime classifier training complete [%s]:", tf)
        logger.info("  train accuracy : %.4f (%.1f%%)", train_acc, train_acc * 100)
        logger.info("  val accuracy   : %.4f (%.1f%%)", val_acc, val_acc * 100)
        logger.info("  baseline       : %.4f (%.1f%%)", majority_pct, majority_pct * 100)

        if val_acc > majority_pct * 1.05:
            logger.info(
                "  LSTM beats baseline by %.1f%%",
                (val_acc / majority_pct - 1) * 100,
            )
        else:
            logger.warning(
                "  LSTM does not beat baseline — may not add value. "
                "Consider more data or re-optimising the HMM."
            )

        return history

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, df_recent: pd.DataFrame) -> Optional[Dict[int, float]]:
        """Return state probability dict from recent bars.

        Needs at least ``sequence_length`` rows; pads with zeros if fewer.
        Returns None when the model output is collapsed (range < 0.05) —
        callers should treat None as "LSTM unreliable, use HMM only".
        Returns uniform distribution on unexpected error.
        """
        if self._model is None:
            return {s: 1.0 / self.n_states for s in range(self.n_states)}

        try:
            df_p = _add_derived_features(df_recent)
            active_cols = [c for c in self.input_columns if c in df_p.columns]

            feat = df_p[active_cols].values.astype(np.float64)
            feat = np.nan_to_num(feat, nan=0.0)
            if self.feature_scaler is not None:
                feat = self.feature_scaler.transform(feat)

            # Pad if shorter than window
            if len(feat) < self.sequence_length:
                pad  = np.zeros((self.sequence_length - len(feat), feat.shape[1]))
                feat = np.vstack([pad, feat])
            feat = feat[-self.sequence_length:].astype(np.float32)

            probs = self._model.predict(feat[np.newaxis], verbose=0)[0]

            # Detect collapsed model — output range near zero means all classes
            # are equally likely, which is indistinguishable from random.
            if float(probs.max() - probs.min()) < 0.05:
                logger.warning(
                    "[LSTM COLLAPSED] probs=%s — near-uniform output, model unreliable. "
                    "Retrain with --mode train_lstm.", probs,
                )
                return None

            return {int(i): float(probs[i]) for i in range(self.n_states)}

        except Exception as exc:
            logger.debug("LSTM predict_proba error: %s", exc)
            return {s: 1.0 / self.n_states for s in range(self.n_states)}

    def predict_state(self, df_recent: pd.DataFrame) -> Optional[int]:
        probs = self.predict_proba(df_recent)
        if probs is None:
            return None
        return max(probs, key=probs.get)

    def ensemble_predict(
        self,
        df_recent: pd.DataFrame,
        hmm_state: int,
        lstm_weight: float = 0.3,
    ) -> Tuple[int, Dict]:
        """Ensemble LSTM probabilities with the HMM state.

        Returns (ensemble_state, confidence_info).

        confidence_info keys:
            agreement       bool   — LSTM and HMM agree on state
            lstm_confidence float  — LSTM's max prob
            transition_risk float  — 0→1, higher = regime likely changing
            lstm_state      int
            hmm_state       int
            ensemble_state  int
            lstm_probs      dict
            lstm_status     str    — 'OK' | 'COLLAPSED'
        """
        lstm_probs = self.predict_proba(df_recent)

        # Collapsed LSTM — return HMM-only fallback with zero influence
        if lstm_probs is None:
            return hmm_state, {
                "agreement":       True,   # don't penalise HMM on broken LSTM
                "lstm_confidence": 0.0,
                "transition_risk": 0.0,    # never tighten Z on a broken model
                "lstm_state":      hmm_state,
                "hmm_state":       hmm_state,
                "ensemble_state":  hmm_state,
                "lstm_probs":      {},
                "ensemble_scores": {s: 1.0 if s == hmm_state else 0.0 for s in range(self.n_states)},
                "lstm_status":     "COLLAPSED",
            }

        lstm_state      = max(lstm_probs, key=lstm_probs.get)
        lstm_confidence = lstm_probs[lstm_state]
        agreement       = (lstm_state == hmm_state)

        # transition_risk: only meaningful when LSTM is CONFIDENT about a
        # DIFFERENT state than HMM.  Agreement (or low confidence) → no risk.
        if lstm_state != hmm_state and lstm_confidence > 0.50:
            # LSTM disagrees AND is confident — genuine transition signal
            transition_risk = lstm_confidence
        elif lstm_state != hmm_state and lstm_confidence > 0.35:
            # LSTM disagrees but uncertain — mild risk, won't reach 0.5 gate
            transition_risk = 0.30
        else:
            # Agreement, or LSTM not confident enough to matter
            transition_risk = 0.0

        ensemble_scores = {}
        for s in range(self.n_states):
            hmm_score = 1.0 if s == hmm_state else 0.0
            ensemble_scores[s] = (
                (1 - lstm_weight) * hmm_score
                + lstm_weight * lstm_probs.get(s, 0.0)
            )
        ensemble_state = max(ensemble_scores, key=ensemble_scores.get)

        info = {
            "agreement":       agreement,
            "lstm_confidence": lstm_confidence,
            "transition_risk": transition_risk,
            "lstm_state":      lstm_state,
            "hmm_state":       hmm_state,
            "ensemble_state":  ensemble_state,
            "lstm_probs":      lstm_probs,
            "ensemble_scores": ensemble_scores,
            "lstm_status":     "OK",
        }
        return ensemble_state, info

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, dirpath) -> None:
        path = Path(dirpath)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "lstm_regime_classifier.keras"))
        joblib.dump(self.feature_scaler, path / "lstm_feature_scaler.pkl")
        meta = {
            "sequence_length": self.sequence_length,
            "n_states":        self.n_states,
            "input_columns":   self.input_columns,
        }
        (path / "lstm_metadata.json").write_text(json.dumps(meta, indent=2))
        logger.info("LSTM regime classifier saved: %s", path)

    @classmethod
    def load(cls, dirpath) -> "LSTMRegimeClassifier":
        from keras.models import load_model as _load_model

        path = Path(dirpath)
        meta = json.loads((path / "lstm_metadata.json").read_text())

        inst = cls(
            sequence_length=meta["sequence_length"],
            n_states=meta["n_states"],
        )
        inst.input_columns  = meta["input_columns"]
        inst._model         = _load_model(str(path / "lstm_regime_classifier.keras"))
        inst.feature_scaler = joblib.load(path / "lstm_feature_scaler.pkl")

        logger.info(
            "LSTM regime classifier loaded: %s  (n_states=%d)", path, inst.n_states
        )
        return inst


# ── Standalone helpers ────────────────────────────────────────────────────────

def load_lstm_classifier(
    tf: str, broker: str = "headway_cent"
) -> Optional["LSTMRegimeClassifier"]:
    """Load a saved LSTMRegimeClassifier or return None if not found."""
    path = get_lstm_dir(tf, broker)
    if not (path / "lstm_regime_classifier.keras").exists():
        logger.debug("LSTM regime classifier not found at %s.", path)
        return None
    try:
        return LSTMRegimeClassifier.load(path)
    except Exception as exc:
        logger.warning(
            "Failed to load LSTM classifier from %s: %s — running HMM-only.", path, exc
        )
        return None
