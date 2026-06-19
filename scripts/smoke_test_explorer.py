"""Smoke test Explorer notebook integration (cells 1-10 core logic)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

nb = json.loads((ROOT / "notebooks" / "GoldRegimeX_Explorer.ipynb").read_text(encoding="utf-8"))
ns = {"__name__": "__main__", "Path": Path, "display": print}
for idx in range(1, 11):
    code = "".join(nb["cells"][idx]["source"])
    if idx == 8:
        code = code.replace("@njit(cache=True)", "@njit(cache=False)")
    exec(compile(code, f"Explorer_cell_{idx}.py", "exec"), ns)

# Data load cell 11
exec(compile("".join(nb["cells"][11]["source"]), "Explorer_cell_11.py", "exec"), ns)

grid = ns["build_coarse_grid"]()
assert len(grid) == 5, f"expected 5 coarse combos, got {len(grid)}"
print("coarse grid size:", len(grid))

# Truncate train set for faster smoke CPCV
m5_small = ns["m5_train"].iloc[-12000:].copy()
m15_small = ns["m15_train"].loc[ns["m15_train"].index <= m5_small.index[-1]].copy()

result = ns["evaluate_combo_cpcv"](
    m5_df=m5_small,
    m15_df=m15_small,
    xgb_threshold=0.45,
    n_blocks=ns["COARSE_CPCV_N_BLOCKS"],
    k_val_blocks=ns["COARSE_CPCV_K_VAL"],
    embargo_bars=int(ns["EMBARGO_HOURS"] * (ns["BARS_PER_DAY"]["M5"] / 24.0)),
)
print("single CPCV result:", result)
assert result["n_paths"] > 0, "CPCV produced zero paths"
assert result["mean_sharpe"] == result["mean_sharpe"], "mean_sharpe is NaN"

feat = ns["build_features"](m5_small.iloc[-2000:], m15_small, ns["EXEC_TF"])
required_cols = {"m15_ema50", "session_mask_london", "regime_code", "tb_label", "atr14"}
missing = required_cols - set(feat.columns)
assert not missing, f"build_features missing columns: {missing}"

print("SMOKE TEST PASSED")
