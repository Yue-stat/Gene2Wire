"""Complete notebook diagnostics and read-only reload of existing result exports.

Reloading a result is deliberately separate from resuming fitting: an old export
retains its own scientific settings and code identity. No dataset is downloaded,
no checkpoint is changed, and results from different runs are never combined.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


_CONTEXT = ("dataset", "sharing_strength", "analysis", "mechanism", "loss_rate",
            "calibration_fraction", "calibration_spec", "model")
_CONFIG = ("kind", "rank", "shared_l2", "residual_l2", "use_target_features", "target_l2")
_TABLE_ORDER = ("aggregate", "per_repetition", "selected", "tuning", "metrics",
                "per_target", "reliability", "detection", "detection_per_target",
                "detection_reliability", "thinning_audit", "paired_audit", "failures",
                "simulation_intervals", "metrics_long")


def configure_full_display() -> None:
    """Keep every table row, column and cell value visible in the notebook."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_seq_items", None)
    pd.set_option("display.width", None)
    pd.set_option("display.expand_frame_repr", False)


def load_export_artifacts(export_dir: str | Path):
    """Read one explicit 0908 run directory; never pick the newest run silently.

    The saved manifest remains the authoritative description of these results,
    even if the currently imported core or notebook settings have changed.
    Incomplete runs are rejected because their final aggregate tables may not
    exist yet. Their checkpoints can still be resumed via the fitting pipeline.
    """
    from .pipeline import Artifacts

    directory = Path(export_dir).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest.json in {directory}. Supply the exact exported run directory "
            "ending in its run ID, not the parent dataset or checkpoint directory."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid result manifest: {manifest_path}")
    if manifest.get("completed") is False:
        raise ValueError(f"{directory} is an incomplete run. Resume it before loading final results.")
    declared = manifest.get("table_files")
    if declared is None:
        declared = [path.name for path in sorted(directory.glob("*.csv"))]
    if not isinstance(declared, list) or not declared:
        raise ValueError(f"No result CSV tables declared or found in {directory}")
    tables = {}
    for filename in declared:
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".csv"):
            raise ValueError(f"Invalid result table filename in {manifest_path}: {filename!r}")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"A declared result table is missing: {path}")
        try:
            tables[path.stem] = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            # Empty failure/diagnostic tables are intentionally exported too.
            tables[path.stem] = pd.DataFrame()
    if "metrics" not in tables:
        raise ValueError(f"No metrics.csv found in {directory}; this is not a complete experiment export")
    return Artifacts(tables=tables, export_dir=directory, manifest=manifest)


def load_existing_exports(export_dirs: Mapping[str, str | Path | None], *, expected_labels):
    """Load the explicit notebook dataset/panel mapping, including both BARseq panels."""
    expected = tuple(expected_labels)
    if not isinstance(export_dirs, Mapping) or set(export_dirs) != set(expected):
        raise ValueError(f"EXISTING_EXPORT_DIRS must contain exactly these keys: {list(expected)}")
    missing = [label for label in expected if export_dirs[label] is None or str(export_dirs[label]).strip() == ""]
    if missing:
        raise ValueError(
            "RESULTS_ONLY=True needs an explicit completed export directory for "
            + ", ".join(missing) + ". Set EXISTING_EXPORT_DIRS in the settings cell."
        )
    artifacts = {label: load_export_artifacts(export_dirs[label]) for label in expected}
    for label, result in artifacts.items():
        datasets = result.manifest.get("datasets", [])
        names = {str(item.get("name")) for item in datasets if isinstance(item, dict)}
        # A1/M1 are notebook labels; the pipeline names are BARseq A1/BARseq M1.
        allowed = {"A1": {"BARseq A1"}, "M1": {"BARseq M1"}}.get(label, {label})
        if names and not names.issubset(allowed):
            raise ValueError(f"Export for {label!r} contains datasets {sorted(names)}, expected {sorted(allowed)}")
    return artifacts


def _counts(frame: pd.DataFrame, columns: list[str], count_name: str) -> pd.DataFrame:
    if frame.empty or not columns:
        return pd.DataFrame()
    return frame.groupby(columns, dropna=False, observed=True).size().rename(count_name).reset_index()


def _boolean_status(values: pd.Series) -> pd.Series:
    # CSV reloads can represent mixed boolean/NA columns as object. Unknown is
    # never treated as optimizer failure or silently converted into True.
    return values.map(lambda value: "converged" if str(value).lower() == "true"
                      else "not_converged" if str(value).lower() == "false" else "not_recorded")


def diagnostic_summaries(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Summaries of recorded evidence only; detailed original tables are unchanged."""
    output = {}
    selected = tables.get("selected", pd.DataFrame())
    tuning = tables.get("tuning", pd.DataFrame())
    if not selected.empty:
        context = [column for column in _CONTEXT if column in selected]
        config = [column for column in ("selected_structure", *_CONFIG) if column in selected]
        output["selected_configuration_frequency"] = _counts(selected, context + config, "selected_count")
    for name, frame, flag in (("final_fit_convergence", selected, "final_converged"),
                              ("candidate_fit_convergence", tuning, "converged")):
        if frame.empty:
            output[name] = pd.DataFrame()
            continue
        status = frame.copy()
        status["convergence_status"] = (_boolean_status(status[flag]) if flag in status
                                         else "not_recorded")
        groups = [column for column in _CONTEXT if column in status] + ["convergence_status"]
        output[name] = _counts(status, groups, "fit_count")
    if not tuning.empty:
        groups = [column for column in _CONTEXT if column in tuning]
        config = [column for column in _CONFIG if column in tuning]
        if config:
            # Counts represent actual recorded trials, including repeat/fold
            # occurrences. They are not inferred Cartesian search budgets.
            output["recorded_candidate_coverage"] = _counts(tuning, groups + config, "trial_occurrences")
    return output


def display_diagnostics(artifacts, *, label: str | None = None,
                        display_fn: Callable[[Any], Any] | None = None) -> None:
    """Show all exports plus diagnostic summaries, without pandas truncation."""
    configure_full_display()
    if display_fn is None:
        from IPython.display import display
        display_fn = display
    print(f"\n{label or 'Experiment'} — complete diagnostics", flush=True)
    print(f"Export directory: {artifacts.export_dir}", flush=True)
    print("Saved run manifest (authoritative settings for these results):", flush=True)
    def json_default(value):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)
    print(json.dumps(artifacts.manifest, indent=2, default=json_default, ensure_ascii=False), flush=True)
    inventory = pd.DataFrame([{"table": name, "rows": len(frame), "columns": len(frame.columns)}
                              for name, frame in artifacts.tables.items()])
    print("Result-table inventory:", flush=True)
    display_fn(inventory)
    for name, frame in diagnostic_summaries(artifacts.tables).items():
        print(f"\n{name}: {len(frame)} rows", flush=True)
        if frame.empty:
            print("No recorded rows available for this summary.", flush=True)
        display_fn(frame)
    order = [name for name in _TABLE_ORDER if name in artifacts.tables]
    order.extend(name for name in artifacts.tables if name not in order)
    for name in order:
        frame = artifacts.tables[name]
        print(f"\n{name}: {len(frame)} rows × {len(frame.columns)} columns", flush=True)
        if frame.empty:
            print("This exported table has zero rows.", flush=True)
        display_fn(frame)
    if "failures" not in artifacts.tables:
        print("No failures table was recorded; absence does not establish that every fit succeeded.", flush=True)
