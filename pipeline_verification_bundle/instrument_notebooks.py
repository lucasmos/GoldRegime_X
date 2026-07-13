"""instrument_notebooks.py -- one-command observability injector.

Usage from repo root:

    python pipeline_verification_bundle/instrument_notebooks.py

What it does:

    * Backs up each target notebook to <name>.orig.ipynb (only on first run).
    * Inserts a small "observability init" cell immediately after the bootstrap
      cell (cell index 1) if not already present.
    * Appends a "observability finalize" cell at the end of the notebook
      (unless already present).
    * Idempotent -- safe to re-run after editing notebooks; existing hook
      cells are detected by their tag `pipeline_observability_v1` and updated
      in place rather than duplicated.

Zero-touch guarantee: every original cell is preserved byte-for-byte. This
script only inserts two additional cells; it never modifies your model,
strategy, HMM, XGBoost, or CPCV code.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List

DEFAULT_TARGETS = [
    "pipeline_verification_bundle/GoldRegimeX_Explorer_fixed.ipynb",
    "pipeline_verification_bundle/Strategy_Tester_fixed.ipynb",
]

HOOK_TAG = "pipeline_observability_v1"

INIT_CELL_SOURCE = [
    "# --- Pipeline Observability init (tag: pipeline_observability_v1) -------------\n",
    "# Auto-inserted by instrument_notebooks.py. Do not edit manually --\n",
    "# re-run the injector to update.\n",
    "from pathlib import Path\n",
    "\n",
    "try:\n",
    "    from pipeline_verification_bundle.shared.pipeline_observability import (\n",
    "        PipelineObservability,\n",
    "    )\n",
    "except ImportError:\n",
    "    # sys.path may not include the repo root yet in some kernels.\n",
    "    import sys as _sys\n",
    "    _sys.path.insert(0, str(Path.cwd()))\n",
    "    from pipeline_verification_bundle.shared.pipeline_observability import (\n",
    "        PipelineObservability,\n",
    "    )\n",
    "\n",
    "OBS_OUTPUT_DIR = Path(\"reports/observability\")\n",
    "OBS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "obs = PipelineObservability(\n",
    "    output_dir=OBS_OUTPUT_DIR,\n",
    "    expected_session_by_tf={\"M15\": \"London_NY\", \"M5\": \"London_NY\"},\n",
    "    material_survival_gap_pct=10.0,\n",
    "    lost_trade_limit=100,\n",
    "    lost_trade_tf=\"M15\",\n",
    ")\n",
    "print(f\"[observability] run_id={obs.run_id} -> {OBS_OUTPUT_DIR}\")\n",
]

FINALIZE_CELL_SOURCE = [
    "# --- Pipeline Observability finalize (tag: pipeline_observability_v1) ---------\n",
    "# Auto-inserted by instrument_notebooks.py. Do not edit manually.\n",
    "# If a PipelineProfiler `profiler` object is defined above (traceability\n",
    "# layer), it is auto-imported into the observability ledger so the two\n",
    "# systems produce a single consistent audit trail.\n",
    "\n",
    "try:\n",
    "    _existing_profiler = profiler  # from the traceability layer\n",
    "except NameError:\n",
    "    _existing_profiler = None\n",
    "\n",
    "if _existing_profiler is not None:\n",
    "    obs.import_from_profiler(_existing_profiler)\n",
    "\n",
    "# Wire in integrity results from pipeline_verification if present.\n",
    "_integrity = {}\n",
    "for _flag_name, _var_name in [\n",
    "    (\"Candidate Integrity\",  \"candidate_integrity_result\"),\n",
    "    (\"Model Integrity\",      \"model_integrity_result\"),\n",
    "    (\"Train/OOS Separation\", \"train_oos_separation_result\"),\n",
    "]:\n",
    "    if _var_name in dir():\n",
    "        _val = eval(_var_name)\n",
    "        _status = getattr(_val, \"status\", None) or (\"PASS\" if _val else \"FAIL\")\n",
    "        _integrity[_flag_name] = str(_status)\n",
    "\n",
    "_result = obs.finalize(integrity_flags=_integrity or None, verbose=True)\n",
    "print(\"\\nObservability artifacts written to:\")\n",
    "for _k, _v in _result.items():\n",
    "    if _v is None or _k in (\"run_id\", \"survival_gap_warnings\"):\n",
    "        continue\n",
    "    print(f\"  {_k}: {_v}\")\n",
]


def _cell_is_hook(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return HOOK_TAG in src


def _new_hook_cell(source_lines: List[str], cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"tags": [HOOK_TAG]},
        "outputs": [],
        "source": source_lines,
    }


def instrument_notebook(path: Path, *, dry_run: bool = False) -> dict:
    if not path.exists():
        return {"path": str(path), "status": "skipped_missing"}

    with open(path, "r", encoding="utf-8") as fh:
        nb = json.load(fh)

    cells = nb.get("cells", [])
    n_before = len(cells)

    # Locate any existing hook cells.
    hook_indices = [i for i, c in enumerate(cells) if _cell_is_hook(c)]

    # ---- Build fresh hook cells --------------------------------------------
    init_cell = _new_hook_cell(INIT_CELL_SOURCE, cell_id="obs_init_v1")
    finalize_cell = _new_hook_cell(FINALIZE_CELL_SOURCE, cell_id="obs_finalize_v1")

    if hook_indices:
        # Update in-place: replace first hook with init, last with finalize;
        # delete any others.
        first = hook_indices[0]
        last = hook_indices[-1]
        cells[first] = init_cell
        if last != first:
            cells[last] = finalize_cell
        # Remove any intermediate hook duplicates (walk in reverse).
        for idx in reversed(hook_indices[1:-1]):
            del cells[idx]
        action = "updated"
    else:
        # Fresh injection: init right after cell 0, finalize at the end.
        insert_at = 1 if len(cells) >= 1 else 0
        cells.insert(insert_at, init_cell)
        cells.append(finalize_cell)
        action = "inserted"

    nb["cells"] = cells

    # Backup on first modification only.
    backup_path = path.with_suffix(".orig.ipynb")
    made_backup = False
    if not backup_path.exists() and not dry_run:
        shutil.copy2(path, backup_path)
        made_backup = True

    if not dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")

    return {
        "path": str(path),
        "status": action,
        "cells_before": n_before,
        "cells_after": len(cells),
        "backup_created": made_backup,
        "backup_path": str(backup_path),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", default=DEFAULT_TARGETS,
        help="Notebook paths to instrument. Defaults to the Explorer and Strategy Tester notebooks.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing files.")
    args = parser.parse_args()

    print("Instrumenting notebooks with pipeline_observability hooks:")
    any_missing = False
    for p in args.paths:
        result = instrument_notebook(Path(p), dry_run=args.dry_run)
        print(f"  {result['path']}: {result['status']}", end="")
        if result["status"] == "skipped_missing":
            any_missing = True
            print("  (file not found)")
            continue
        print(f"  cells {result['cells_before']} -> {result['cells_after']}", end="")
        if result.get("backup_created"):
            print(f"  backup={result['backup_path']}", end="")
        if result.get("dry_run"):
            print("  [dry-run: no write]", end="")
        print()

    if any_missing:
        print("\nOne or more notebook paths were missing. Pass explicit paths, e.g.:")
        print("  python pipeline_verification_bundle/instrument_notebooks.py path/to/notebook.ipynb")
        return 2
    print("\nDone. Restart the notebook kernel and run all cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
