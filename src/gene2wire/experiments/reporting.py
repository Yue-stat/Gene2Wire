"""Compact notebook diagnostics and read-only reload of existing result exports.

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


def configure_compact_display() -> None:
    """Display complete small summary tables; detailed exports stay on disk."""
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.max_seq_items", 20)
    pd.set_option("display.width", 140)
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


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["analysis"].eq("primary")] if "analysis" in frame else frame


def _endpoint(frame: pd.DataFrame) -> pd.DataFrame:
    """Natural or largest recorded loss within each scenario/model, without averaging."""
    frame = _primary(frame)
    if frame.empty or "loss_rate" not in frame:
        return frame
    groups = [name for name in _CONTEXT if name in frame and name != "loss_rate"]
    rates = pd.to_numeric(frame["loss_rate"], errors="coerce")
    if not groups:
        return frame.loc[rates.eq(rates.max()) | rates.isna()]
    # Rates are selected separately per model so an endpoint-only baseline
    # remains visible; the recorded rate is printed rather than assumed 80%.
    maximum = frame.assign(_rate=rates).groupby(groups, dropna=False, observed=True)["_rate"].transform("max")
    return frame.loc[rates.eq(maximum) | rates.isna()]


def _final_status(frame: pd.DataFrame) -> pd.Series:
    flags = frame.get("final_converged", pd.Series(index=frame.index, dtype=object))
    # Older Qiao exports used the shorter key. Unknown RF status stays unknown.
    if "converged" in frame:
        flags = flags.combine_first(frame["converged"])
    return _boolean_status(flags)


def _configuration_signature(row: pd.Series) -> str:
    fields = ("selected_structure", "kind", "rank", "shared_l2", "residual_l2",
              "target_l2", "l2", "objective", "n_estimators", "min_samples_leaf",
              "max_features", "max_depth")
    short = {"selected_structure": "kind", "rank": "K", "shared_l2": "sh",
             "residual_l2": "res", "target_l2": "tgt", "n_estimators": "trees",
             "min_samples_leaf": "leaf", "max_features": "features", "max_depth": "depth"}
    bits = []
    for field in fields:
        if field not in row:
            continue
        if field == "kind" and pd.notna(row.get("selected_structure")):
            continue
        value = row[field]
        if pd.isna(value):
            if field == "max_depth" and str(row.get("model", "")).startswith("RF"):
                value = "unlimited"
            else:
                continue
        bits.append(f"{short.get(field, field)}={value}")
    return ", ".join(bits) or "not recorded"


def compact_summaries(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Small, explicitly scoped summaries; no raw trial/per-target/event tables."""
    output = {}
    aggregate = tables.get("aggregate", pd.DataFrame())
    if not aggregate.empty:
        endpoint = _endpoint(aggregate)
        columns = [name for name in ("dataset", "sharing_strength", "mechanism", "loss_rate", "model",
            "macro_auprc", "macro_log_loss", "macro_hidden_recall_at_h", "macro_brier",
            "macro_predicted_prevalence", "macro_reference_prevalence") if name in endpoint]
        for name in ("calibration_fraction", "calibration_spec"):
            if name in endpoint and endpoint[name].nunique(dropna=False) > 1:
                columns.insert(columns.index("model") if "model" in columns else 0, name)
        output["primary_endpoint_metrics"] = endpoint.loc[:, columns].reset_index(drop=True)
    selected = _primary(tables.get("selected", pd.DataFrame()))
    tuning = _primary(tables.get("tuning", pd.DataFrame()))
    groups = [name for name in ("dataset", "sharing_strength", "model") if name in selected]
    if not selected.empty and groups:
        rows = []
        for key, frame in selected.groupby(groups, dropna=False, observed=True):
            row = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
            row["final_fits"] = len(frame)
            status = _final_status(frame)
            row.update({name: int(status.eq(name).sum()) for name in ("converged", "not_converged", "not_recorded")})
            frequency = frame.apply(_configuration_signature, axis=1).value_counts(sort=True)
            row["unique_selected_configs"] = len(frequency)
            row["most_selected_config"] = frequency.index[0]
            row["mode_count"] = int(frequency.iloc[0])
            rows.append(row)
        output["selection_and_convergence_all_primary_rates"] = pd.DataFrame(rows)
    all_selected = tables.get("selected", pd.DataFrame())
    all_tuning = tables.get("tuning", pd.DataFrame())
    status_rows = []
    for label, frame, statuses in (
        ("final fits", all_selected, _final_status(all_selected)),
        ("candidate trials", all_tuning, _boolean_status(all_tuning.get("converged",
            pd.Series(index=all_tuning.index, dtype=object))))):
        if not frame.empty:
            status_rows.append({"scope": "all scenarios", "records": label, "count": len(frame),
                **{name: int(statuses.eq(name).sum()) for name in
                   ("converged", "not_converged", "not_recorded")}})
    if status_rows:
        output["convergence_all_scenarios"] = pd.DataFrame(status_rows)
    incomplete = all_selected.loc[_final_status(all_selected).eq("not_converged")]
    if not incomplete.empty:
        columns = [name for name in (*_CONTEXT, "repetition", "outer_fold", "rank",
            "final_iterations", "validation_observed_log_loss") if name in incomplete]
        output["nonconverged_final_fits_first_10_see_selected_csv"] = incomplete.loc[:, columns].head(10)
    if not tuning.empty:
        group_cols = [name for name in ("dataset", "sharing_strength", "model") if name in tuning]
        if group_cols:
            rows = []
            for key, frame in tuning.groupby(group_cols, dropna=False, observed=True):
                row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
                row["recorded_trials"] = len(frame)
                for column, target in (("rank", "unique_ranks"), ("shared_l2", "unique_shared_penalties"),
                                       ("residual_l2", "unique_residual_penalties"), ("l2", "unique_bilinear_penalties")):
                    if column in frame:
                        row[target] = frame[column].nunique(dropna=True)
                status = _boolean_status(frame["converged"]) if "converged" in frame else pd.Series("not_recorded", index=frame.index)
                row["nonconverged_trials"] = int(status.eq("not_converged").sum())
                rows.append(row)
            output["candidate_coverage_all_primary_rates"] = pd.DataFrame(rows)
    # Failure details must remain visible even when outside the primary analysis.
    failures = tables.get("failures", pd.DataFrame())
    if not failures.empty:
        output["failed_units_first_10_see_failures_csv"] = failures.head(10).copy()
    return output


def display_diagnostics(artifacts, *, label: str | None = None,
                        display_fn: Callable[[Any], Any] | None = None,
                        full: bool = False) -> None:
    """Show useful endpoint/selection diagnostics; opt in to every raw export."""
    if display_fn is None:
        from IPython.display import display
        display_fn = display
    if full:
        return _display_full_diagnostics(artifacts, label=label, display_fn=display_fn)
    configure_compact_display()
    print(f"\n{label or 'Experiment'} — concise diagnostics", flush=True)
    print(f"All result tables, predictions and settings: {artifacts.export_dir}", flush=True)
    print("Metrics: largest recorded primary loss rate for each model (or natural paired labels). "
          "Selection and convergence: all primary rates, folds and repetitions.", flush=True)
    summaries = compact_summaries(artifacts.tables)
    for name, frame in summaries.items():
        print(f"\n{name}", flush=True)
        display_fn(frame)
    if "primary_endpoint_metrics" not in summaries:
        print("No aggregate endpoint table was recorded; inspect metrics.csv for the original per-unit results.", flush=True)
    failures = artifacts.tables.get("failures")
    if failures is None:
        print("Failure status was not recorded in this export.", flush=True)
    elif failures.empty:
        print("Recorded failed units: 0.", flush=True)
    else:
        print(f"Recorded failed units: {len(failures)}; see failures.csv for complete details.", flush=True)
    print("Set SHOW_FULL_DIAGNOSTICS=True to display every saved table and the full manifest.", flush=True)


def _display_full_diagnostics(artifacts, *, label: str | None = None,
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
