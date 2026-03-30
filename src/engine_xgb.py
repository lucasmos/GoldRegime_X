import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import onnx
from pathlib import Path
from sklearn.metrics import accuracy_score
from src.logger import setup_logger

logger = setup_logger(__name__)

ONNX_PATH = Path("models/xgb_model.onnx")
XGB_PKL_PATH = Path("models/xgb_model.pkl")

FEATURE_COLS = ["hmm_state", "rsi_slope", "atr_normalized", "prev_log_return"]


def prepare_features(df: pd.DataFrame, hmm_states: np.ndarray):
    df = df.copy()
    df["hmm_state"] = hmm_states
    df["prev_log_return"] = df["log_return"].shift(1)
    y = (df["log_return"].shift(-1) > 0).astype(int).rename("target")

    X = df[FEATURE_COLS]
    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]
    df_aligned = df.loc[X.index]

    logger.info("Features prepared: %d samples, %d features", len(X), len(FEATURE_COLS))
    return X, y, df_aligned


def train_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    n_estimators: int = 200,
    subsample: float = 0.8,
    min_child_weight: int = 5,
    gamma: float = 1.0,
    reg_alpha: float = 0.1,
    colsample_bytree: float = 0.8,
    train_ratio: float = 0.8,
):
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))

    logger.info("XGB Train Acc: %.4f | Test Acc: %.4f", train_acc, test_acc)
    logger.info("Feature importance: %s", importance)

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "feature_importance": importance,
        "split_idx": split_idx,
    }
    return model, metrics


def get_predictions(model: xgb.XGBClassifier, X: pd.DataFrame):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities


def save_xgb(model: xgb.XGBClassifier, metrics: dict = None, path: Path = XGB_PKL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics or {}}, path)
    logger.info("XGB model saved to %s", path)


def load_xgb(path: Path = XGB_PKL_PATH):
    data = joblib.load(path)
    if isinstance(data, dict) and "model" in data:
        model = data["model"]
        metrics = data.get("metrics", {})
    else:
        # Backwards compat: old saves stored just the model
        model = data
        metrics = {
            "feature_importance": dict(zip(FEATURE_COLS, model.feature_importances_)),
        }
    return model, metrics


def _strip_zipmap(onnx_model):
    """Remove ZipMap node from ONNX graph, exposing the raw float probability tensor.

    onnxmltools converts XGBoost probabilities as ZipMap (sequence of maps).
    MT5's OnnxRun expects a plain float32 tensor.  This surgery replaces the
    ZipMap output with the raw tensor it is wrapping, making the model fully
    compatible with MT5's OnnxSetOutputShape / OnnxRun API.
    """
    import onnx as _onnx

    graph = onnx_model.graph

    float_tensor_name  = None
    zipmap_output_name = None
    nodes_to_keep      = []

    for node in graph.node:
        if node.op_type == "ZipMap":
            float_tensor_name  = node.input[0]
            zipmap_output_name = node.output[0]
        else:
            nodes_to_keep.append(node)

    if float_tensor_name is None:
        return onnx_model  # no ZipMap present — nothing to do

    new_outputs = []
    for output in graph.output:
        if output.name == zipmap_output_name:
            new_outputs.append(
                _onnx.helper.make_tensor_value_info(
                    float_tensor_name,
                    _onnx.TensorProto.FLOAT,
                    None,   # shape inferred at runtime
                )
            )
        else:
            new_outputs.append(output)

    new_graph = _onnx.helper.make_graph(
        nodes_to_keep,
        graph.name,
        list(graph.input),
        new_outputs,
        list(graph.initializer),
    )
    new_model = _onnx.helper.make_model(
        new_graph, opset_imports=onnx_model.opset_import
    )
    new_model.ir_version = onnx_model.ir_version
    return new_model


def export_onnx(model: xgb.XGBClassifier, n_features: int = 4, path: Path = ONNX_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    # onnxmltools requires feature names in 'f%d' pattern
    # Clone the model's booster with generic feature names
    import copy
    model_copy = copy.deepcopy(model)
    model_copy.get_booster().feature_names = [f"f{i}" for i in range(n_features)]

    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model_copy, initial_types=initial_type)
    onnx_model = _strip_zipmap(onnx_model)   # expose float tensor output for MT5

    onnx.save_model(onnx_model, str(path))
    onnx.checker.check_model(onnx_model)

    inputs    = [i.name for i in onnx_model.graph.input]
    outputs   = [o.name for o in onnx_model.graph.output]
    n_classes = model.n_classes_
    logger.info(
        "ONNX exported to %s | inputs: %s | outputs: %s | n_classes=%d",
        path, inputs, outputs, n_classes,
    )
    print(f"\n  ONNX export OK — n_classes={n_classes}. "
          f"Set NStates={n_classes} in the MT5 EA inputs.\n")
