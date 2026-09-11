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


# Optional coordinates distinguish the additional block-masking experiment.
# Fitting records use training_block_fraction; evaluation records additionally
# use block_fraction/evaluation_scope because one full-panel fit can be scored
# on several distinct held-out target panels. Only existing columns are used,
# so diagnostics of historical standard exports keep their original shape.
_BLOCK_CONTEXT = ("experiment", "group_mode", "training_panel", "training_block_fraction",
                  "block_fraction", "evaluation_scope")
_OVERLAP_CONTEXT = ("panel_design", "arm", "requested_overlap", "actual_overlap", "panel_size")
_CONTEXT = ("dataset", "sharing_strength", "analysis", "mechanism", "loss_rate",
            "calibration_fraction", "calibration_spec", *_BLOCK_CONTEXT,
            *_OVERLAP_CONTEXT, "model")
_CONFIG = ("kind", "rank", "shared_l2", "residual_l2", "use_target_features", "target_l2")
_TABLE_ORDER = ("aggregate", "per_repetition", "selected", "tuning", "metrics",
                "per_target", "reliability", "detection", "detection_per_target",
                "detection_reliability", "thinning_audit", "paired_audit", "failures",
                "simulation_intervals", "metrics_long", "worst_panel", "overlap_contrasts")


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
        # BARseq notebook labels carry the panel design suffix while the
        # pipeline manifest keeps the biological dataset name.
        if str(label) == "A1" or str(label).startswith("BARseq A1"):
            allowed = {"BARseq A1"}
        elif str(label) == "M1" or str(label).startswith("BARseq M1"):
            allowed = {"BARseq M1"}
        else:
            allowed = {label}
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


def joint_selection_diagnostics(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Audit the candidates *inside each recorded Joint search*, without fitting.

    Compare converged finite validation scores, as the tuner does. No separate
    independent/MIRT search is merged into the candidate set, and no test metric
    is read. Positive ``direct_minus_selected_validation_loss`` favors the
    recorded selection over its own direct endpoint. It is not a test-set gain
    or a confidence interval. Missing convergence is unknown, never success.

    Old exports remain usable: missing fields produce explicit evidence status
    and NaNs. If repetition/fold metadata is absent, ``unit_scope`` makes the
    coarser recorded grouping explicit. Original tables/exports are not changed.
    """
    tuning = tables.get("tuning", pd.DataFrame())
    if tuning.empty or "model" not in tuning:
        return pd.DataFrame()
    tuning = tuning.loc[tuning["model"].isin(("Joint", "PU-Joint", "Reference+PU-Joint"))]
    if tuning.empty:
        return pd.DataFrame()
    groups = [column for column in (*_CONTEXT, "repetition", "outer_fold",
                                   "supervision", "supervision_mode") if column in tuning]
    selected = tables.get("selected", pd.DataFrame())
    selection_aligned = not selected.empty and all(column in selected for column in groups)

    def stable_key(key):
        values = key if isinstance(key, tuple) else (key,)
        return tuple(None if pd.isna(value) else value for value in values)

    selection_groups = ({stable_key(key): frame for key, frame in
                         selected.groupby(groups, dropna=False, observed=True)}
                        if selection_aligned else {})
    rows = []
    for key, frame in tuning.groupby(groups, dropna=False, observed=True):
        row = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        row["unit_scope"] = ("fold_repetition" if {"repetition", "outer_fold"}.issubset(groups)
                             else "recorded_context_only")
        score = pd.to_numeric(frame.get("validation_loss",
            pd.Series(index=frame.index, dtype=float)), errors="coerce")
        status = _boolean_status(frame.get("converged", pd.Series(index=frame.index, dtype=object)))
        eligible = status.eq("converged") & np.isfinite(score)
        kind = frame.get("kind", pd.Series(index=frame.index, dtype=object))
        row["recorded_candidates"] = len(frame)
        row["eligible_candidates"] = int(eligible.sum())
        row["convergence_unknown_candidates"] = int(status.eq("not_recorded").sum())
        row["unknown_family_candidates"] = int((~kind.isin(("direct", "lowrank", "joint"))).sum())
        missing = [label for field, label in (("converged", "convergence not recorded"),
                   ("kind", "structure not recorded"), ("validation_loss", "validation loss not recorded"))
                   if field not in frame]
        row["candidate_evidence_status"] = ("; ".join(missing) if missing else
            "no converged finite candidates" if not eligible.any() else "recorded")
        for family in ("direct", "lowrank", "joint"):
            in_family = kind.eq(family)
            available = eligible & in_family
            row[f"{family}_recorded_candidates"] = int(in_family.sum())
            row[f"{family}_eligible_candidates"] = int(available.sum())
            row[f"{family}_minimum_validation_loss"] = score[available].min()
        row["best_recorded_validation_loss"] = score[eligible].min()
        row["selected_family"] = None
        row["selected_validation_loss"] = np.nan
        matched = selection_groups.get(stable_key(key), pd.DataFrame())
        if selected.empty:
            selection_status = "selection not recorded"
        elif not selection_aligned:
            selection_status = "selection unit metadata missing"
        elif matched.empty:
            selection_status = "matching selection not recorded"
        elif len(matched) != 1:
            selection_status = "multiple selection rows for recorded unit"
        else:
            choice = matched.iloc[0]
            family = choice.get("selected_structure")
            if pd.isna(family):
                family = choice.get("kind")
            row["selected_family"] = (family if isinstance(family, str)
                                      and family in ("direct", "lowrank", "joint") else None)
            recorded_score = pd.to_numeric(pd.Series([choice.get("validation_observed_log_loss")]),
                                           errors="coerce").iloc[0]
            if np.isfinite(recorded_score):
                row["selected_validation_loss"] = float(recorded_score)
            selection_status = ("recorded" if row["selected_family"] is not None
                and np.isfinite(row["selected_validation_loss"]) else "selection fields missing")
        row["selection_evidence_status"] = selection_status
        row["direct_minus_selected_validation_loss"] = (
            row["direct_minimum_validation_loss"] - row["selected_validation_loss"])
        row["selected_minus_best_recorded_validation_loss"] = (
            row["selected_validation_loss"] - row["best_recorded_validation_loss"])
        rows.append(row)
    return pd.DataFrame(rows)


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
    joint = joint_selection_diagnostics(tables)
    if not joint.empty:
        output["joint_selection_diagnostics"] = joint
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
        columns = [name for name in ("dataset", "sharing_strength", "mechanism", "loss_rate",
            *_BLOCK_CONTEXT, *_OVERLAP_CONTEXT, "model",
            "macro_auprc", "macro_log_loss", "macro_hidden_recall_at_h", "macro_brier",
            "macro_predicted_prevalence", "macro_reference_prevalence") if name in endpoint]
        for name in ("calibration_fraction", "calibration_spec"):
            if name in endpoint and endpoint[name].nunique(dropna=False) > 1:
                columns.insert(columns.index("model") if "model" in columns else 0, name)
        output["primary_endpoint_metrics"] = endpoint.loc[:, columns].reset_index(drop=True)
    selected = _primary(tables.get("selected", pd.DataFrame()))
    tuning = _primary(tables.get("tuning", pd.DataFrame()))
    groups = [name for name in ("dataset", "sharing_strength", *_BLOCK_CONTEXT,
                                *_OVERLAP_CONTEXT, "model")
              if name in selected]
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
        group_cols = [name for name in ("dataset", "sharing_strength", *_BLOCK_CONTEXT,
                                        *_OVERLAP_CONTEXT, "model")
                      if name in tuning]
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


_IMPORTANT_MODELS = ("PU", "PU-MIRT", "PU-Joint", "Reference+PU-Joint")
_PU_MODELS = _IMPORTANT_MODELS[:3]
_ASSAY_MODELS = ("Logistic", "MIRT", "Joint")
_REPORT_LIMITS = {
    "primary_endpoint_metrics": 48,
    "pu_curves": 48,
    "per_animal_measured_metrics": 48,
    "selected_hyperparameters_by_repetition": 60,
    "selection_and_convergence": 36,
    "joint_validation_selection": 36,
    "detection_calibration": 24,
    "matched_full_panel_control": 36,
    "overlap_contrasts": 60,
    "failed_units": 12,
}


def _overlap_contrast_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the paired overlap contrast across repetitions for display.

    ``overlap_contrasts.csv`` intentionally keeps one row per repetition. A
    concise notebook table must not hide that repetition identifier while
    showing raw rows, so this display-only summary reports the equal-weight
    mean, paired between-repetition SD, and number of recorded repetitions.
    """
    required = {"model", "metric", "actual_overlap",
                "joint_or_mirt_minus_pu", "difference_vs_100pct_overlap"}
    if frame.empty or not required.issubset(frame):
        return pd.DataFrame()
    values = ("joint_or_mirt_minus_pu", "difference_vs_100pct_overlap")
    numeric = frame.copy()
    for name in values:
        numeric[name] = pd.to_numeric(numeric[name], errors="coerce")
    groups = list(dict.fromkeys(
        name for name in (*_CONTEXT, "metric", "actual_overlap")
        if name in numeric
    ))
    if not groups:
        return pd.DataFrame()
    grouped = numeric.groupby(groups, dropna=False, observed=True)
    result = grouped[list(values)].mean().rename(columns={
        "joint_or_mirt_minus_pu": "mean_advantage_vs_PU",
        "difference_vs_100pct_overlap": "mean_R_vs_100pct",
    })
    result["sd_R_vs_100pct"] = grouped["difference_vs_100pct_overlap"].std(ddof=1)
    if "repetition" in numeric:
        result["n_repetitions"] = grouped["repetition"].nunique()
    else:
        result["n_repetitions"] = grouped.size()
    return result.reset_index()


def _merge_worst_panel_metrics(endpoint: pd.DataFrame,
                               worst_panel: pd.DataFrame) -> pd.DataFrame:
    """Attach fold-then-repetition averaged worst-panel metrics to endpoints."""
    values = [name for name in (
        "worst_panel_macro_auprc", "worst_panel_macro_log_loss",
        "worst_panel_macro_brier",
    ) if name in worst_panel]
    if endpoint.empty or worst_panel.empty or not values:
        return endpoint
    worst = _report_scope(worst_panel, endpoint=True)
    worst = _equal_repetition_mean(worst, values)
    keys = [name for name in _CONTEXT if name in endpoint and name in worst]
    if not keys or worst.duplicated(keys).any():
        return endpoint
    return endpoint.merge(worst[keys + values], on=keys, how="left", validate="many_to_one")


def _assay_only(frame: pd.DataFrame) -> bool:
    """Require an explicit assay-only mechanism; never infer it from model names."""
    return (not frame.empty and "mechanism" in frame
            and frame["mechanism"].notna().all()
            and frame["mechanism"].eq("assay_only").all())


def _important_models(frame: pd.DataFrame) -> tuple[str, ...]:
    return _ASSAY_MODELS if _assay_only(frame) else _IMPORTANT_MODELS


def _report_scope(frame: pd.DataFrame, *, endpoint: bool = False,
                  masked: bool = True) -> pd.DataFrame:
    """Primary, blocked evaluation; endpoint includes the largest block fraction."""
    frame = _primary(frame)
    if frame.empty:
        return frame
    if "evaluation_scope" in frame:
        frame = frame.loc[frame.evaluation_scope.eq("blocked")]
    if masked and "training_panel" in frame:
        frame = frame.loc[frame.training_panel.eq("masked")]
    if endpoint:
        for coordinate in ("block_fraction", "training_block_fraction", "loss_rate"):
            if coordinate not in frame:
                continue
            ignored = {coordinate}
            if coordinate in ("block_fraction", "training_block_fraction"):
                ignored.update(("block_fraction", "training_block_fraction"))
            groups = [name for name in _CONTEXT if name in frame and name not in ignored]
            values = pd.to_numeric(frame[coordinate], errors="coerce")
            maximum = (frame.assign(_coordinate=values).groupby(groups, dropna=False,
                observed=True)["_coordinate"].transform("max") if groups else values.max())
            frame = frame.loc[values.eq(maximum) | values.isna()]
    return frame


def _equal_repetition_mean(frame: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    """Average folds inside repetitions, then give each repetition equal weight."""
    if frame.empty or not values:
        return pd.DataFrame()
    groups = [name for name in (*_CONTEXT, "probability_semantics") if name in frame]
    numeric = frame.copy()
    for name in values:
        numeric[name] = pd.to_numeric(numeric[name], errors="coerce")
    if "repetition" in numeric:
        per_rep = numeric.groupby(groups + ["repetition"], dropna=False,
                                 observed=True)[values].mean().reset_index()
    else:
        per_rep = numeric
    return (per_rep.groupby(groups, dropna=False, observed=True)[values].mean().reset_index()
            if groups else pd.DataFrame([per_rep[values].mean()]))


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    choices = ["macro_auprc", "macro_log_loss", "macro_brier",
               "hidden_recall_at_h", "macro_hidden_recall_at_h", "macro_predicted_prevalence",
               "macro_reference_prevalence"]
    if "block_fraction" in frame or "training_panel" in frame or _assay_only(frame):
        choices = [name for name in choices if "hidden_recall" not in name]
    return [name for name in choices if name in frame]


def _report_view(frame: pd.DataFrame, values: list[str], *, extra=()) -> pd.DataFrame:
    """Keep varying coordinates explicit; record fixed coordinates in table metadata."""
    coordinates = [name for name in (*_CONTEXT, "probability_semantics") if name in frame]
    # The training fraction is redundant with the evaluation fraction for a
    # masked fit and is zero for the full control. Preserve it in configuration
    # tables, where no evaluation fraction is present.
    if "block_fraction" in coordinates and "training_block_fraction" in coordinates:
        coordinates.remove("training_block_fraction")
    fixed, retained = {}, []
    for name in coordinates:
        if name != "model" and frame[name].nunique(dropna=False) == 1:
            fixed[name] = frame[name].iloc[0]
        else:
            retained.append(name)
    selected = list(dict.fromkeys([*retained, *extra, *values]))
    result = frame.loc[:, [name for name in selected if name in frame]].copy()
    # A large user-specified scenario grid can vary more coordinates than fit
    # on one screen. Pack those coordinates, rather than silently drop them.
    if len(result.columns) > 14:
        pack = [name for name in retained if name not in ("model", "sharing_strength",
                "loss_rate", "block_fraction")]
        if pack:
            result.insert(0, "scenario", result[pack].apply(
                lambda row: "; ".join(f"{name}={row[name]}" for name in pack), axis=1))
            result = result.drop(columns=pack)
    result.attrs["fixed_coordinates"] = fixed
    return result


def _bounded_report(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Hard row/column/text limits with context-balanced, explicitly labeled previews."""
    original = len(frame)
    metadata = dict(frame.attrs)
    if original > limit:
        strata = [name for name in ("dataset", "sharing_strength", "mechanism",
                  "calibration_fraction", "calibration_spec", "training_panel", "scenario")
                  if name in frame]
        groups = (list(frame.groupby(strata, dropna=False, observed=True, sort=False).indices.values())
                  if strata else [np.arange(original)])
        # Round-robin contexts prevents a head() preview containing only rho=0
        # or the first dataset. Within a context, important models come first.
        if "model" in frame:
            priority = {name: index for index, name in enumerate(
                (*_important_models(frame), *_ASSAY_MODELS))}
            balanced = []
            for indices in groups:
                model_rows = {}
                for index in indices:
                    model_rows.setdefault(str(frame.iloc[index]["model"]), []).append(index)
                names = sorted(model_rows, key=lambda model: priority.get(model, 99))
                # Interleave models before later repetitions; otherwise a
                # twenty-repetition PU group can consume the whole preview.
                balanced.append([model_rows[name][position]
                    for position in range(max(map(len, model_rows.values())))
                    for name in names if position < len(model_rows[name])])
            groups = balanced
        chosen = []
        for position in range(max(map(len, groups))):
            for indices in groups:
                if position < len(indices):
                    chosen.append(indices[position])
                    if len(chosen) == limit:
                        break
            if len(chosen) == limit:
                break
        frame = frame.iloc[chosen].copy()
        metadata["preview_contexts"] = min(len(groups), limit)
        metadata["total_contexts"] = len(groups)
    else:
        frame = frame.copy()
    omitted_columns = list(frame.columns[14:])
    frame = frame.iloc[:, :14]
    clipped_cells = 0
    for name in frame.select_dtypes(include="object"):
        def clip(value):
            nonlocal clipped_cells
            if isinstance(value, str) and len(value) > 500:
                clipped_cells += 1
                return value[:477] + " … [see exported CSV]"
            return value
        frame[name] = frame[name].map(clip)
    frame.attrs = {**metadata, "total_rows": original, "omitted_rows": original-len(frame),
                   "omitted_columns": omitted_columns, "clipped_cells": clipped_cells}
    return frame.reset_index(drop=True)


def _selected_by_repetition(selected: pd.DataFrame) -> pd.DataFrame:
    selected = _report_scope(selected, endpoint=True)
    if selected.empty or "model" not in selected:
        return pd.DataFrame()
    selected = selected.loc[selected.model.isin(_important_models(selected))]
    groups = [name for name in (*_CONTEXT, "repetition") if name in selected]
    rows = []
    for key, frame in selected.groupby(groups, dropna=False, observed=True, sort=False):
        row = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        frame = frame.sort_values("outer_fold") if "outer_fold" in frame else frame
        configs = []
        for _, choice in frame.iloc[:3].iterrows():
            fold = choice.get("outer_fold", "not recorded")
            config = _configuration_signature(choice)
            status = _final_status(pd.DataFrame([choice])).iloc[0]
            configs.append(f"fold {fold}: {config}" + (" [not converged]" if status == "not_converged" else ""))
        if len(frame) > 3:
            configs.append(f"+ {len(frame)-3} more fold records in selected.csv")
        row.update(fold_configurations=" | ".join(configs), recorded_folds=len(frame))
        rows.append(row)
    return pd.DataFrame(rows)


def _selection_health(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    selected = _report_scope(tables.get("selected", pd.DataFrame()), masked=False)
    tuning = _report_scope(tables.get("tuning", pd.DataFrame()), masked=False)
    if selected.empty or "model" not in selected:
        return pd.DataFrame()
    selected = selected.loc[selected.model.isin(_important_models(selected))]
    groups = [name for name in ("dataset", "sharing_strength", "mechanism", "calibration_fraction",
        "calibration_spec", *_OVERLAP_CONTEXT, "training_panel", "model") if name in selected]
    rows = []
    for key, frame in selected.groupby(groups, dropna=False, observed=True, sort=False):
        row = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        status = _final_status(frame)
        family = frame.get("selected_structure", frame.get("kind", pd.Series(index=frame.index, dtype=object)))
        if "kind" in frame:
            family = family.fillna(frame.kind)
        frequency = frame.apply(_configuration_signature, axis=1).value_counts()
        row.update(final_fits=len(frame), final_not_converged=int(status.eq("not_converged").sum()),
            final_status_unknown=int(status.eq("not_recorded").sum()),
            selected_families="; ".join(f"{kind}: {int(family.eq(kind).sum())}"
                for kind in ("direct", "lowrank", "joint") if family.eq(kind).any()) or "not recorded",
            modal_configuration=frequency.index[0], modal_count=int(frequency.iloc[0]))
        candidates = tuning
        for name, value in zip(groups, key if isinstance(key, tuple) else (key,)):
            if name not in candidates:
                candidates = pd.DataFrame()
                break
            candidates = candidates.loc[candidates[name].isna() if pd.isna(value) else candidates[name].eq(value)]
        row["candidate_trials"] = len(candidates)
        flags = _boolean_status(candidates.get("converged", pd.Series(index=candidates.index, dtype=object)))
        row["candidate_not_converged"] = int(flags.eq("not_converged").sum())
        row["recorded_boundary_hits"] = _recorded_boundary_hits(frame, candidates)
        rows.append(row)
    return pd.DataFrame(rows)


def _recorded_boundary_hits(selected: pd.DataFrame, tuning: pd.DataFrame) -> str:
    """Count selected edges of the recorded same-family search, per fit unit.

    A one-value search provides no boundary evidence. Rank zero is an exact
    endpoint rather than a too-small-rank warning. These are diagnostics of
    evaluated candidates, not claims about an unrecorded Cartesian grid.
    """
    if tuning.empty:
        return "not recorded"
    units = [name for name in (*_CONTEXT, "repetition", "outer_fold") if name in selected and name in tuning]
    def key(values):
        return tuple(None if pd.isna(value) else value for value in values)
    lookup = {key(group if isinstance(group, tuple) else (group,)): frame
              for group, frame in tuning.groupby(units, dropna=False, observed=True)} if units else {(): tuning}
    counts = {}
    for _, choice in selected.iterrows():
        candidates = lookup.get(key(choice[name] for name in units), pd.DataFrame())
        family = choice.get("selected_structure", choice.get("kind"))
        if pd.isna(family):
            family = choice.get("kind")
        if "kind" in candidates and isinstance(family, str):
            candidates = candidates.loc[candidates.kind.eq(family)]
        for field, short in (("rank", "K max"), ("shared_l2", "shared"),
                             ("residual_l2", "residual"), ("target_l2", "target")):
            if field not in candidates or field not in choice:
                continue
            value = pd.to_numeric(pd.Series([choice[field]]), errors="coerce").iloc[0]
            options = pd.to_numeric(candidates[field], errors="coerce").dropna().unique()
            if len(options) < 2 or not np.isfinite(value) or (field == "rank" and value <= 0):
                continue
            sides = (("", options.max()),) if field == "rank" else ((" min", options.min()), (" max", options.max()))
            for suffix, edge in sides:
                if value == edge:
                    label = short + suffix
                    counts[label] = counts.get(label, 0) + 1
    return "; ".join(f"{name}: {count}" for name, count in counts.items()) or "none among varying recorded candidates"


def _matched_panel_control(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Paired masking loss and sharing difference-in-differences, then fold/rep means.

    Positive masking_loss means the masked model did worse. Positive
    extra_sharing_gain means panel masking increased that model's advantage
    over independent PU, on the identical test entries within each fold/rep.
    No difference is constructed from unmatched aggregate rows.
    """
    metrics = _report_scope(tables.get("metrics", pd.DataFrame()), masked=False)
    required = {"training_panel", "model", "repetition", "outer_fold", "block_fraction"}
    if metrics.empty or not required.issubset(metrics):
        return pd.DataFrame()
    assay_only = _assay_only(metrics)
    baseline_name = "Logistic" if assay_only else "PU"
    structure_models = ("Logistic", "MIRT", "Joint") if assay_only else _PU_MODELS
    metrics = metrics.loc[metrics.model.isin(structure_models)]
    values = [name for name in ("macro_auprc", "macro_log_loss", "macro_brier") if name in metrics]
    groups = [name for name in (*_CONTEXT, "repetition", "outer_fold") if name in metrics
              and name not in ("training_panel", "training_block_fraction")]
    masked = metrics.loc[metrics.training_panel.eq("masked"), groups+values]
    full = metrics.loc[metrics.training_panel.eq("full"), groups+values]
    if masked.empty or full.empty or masked.duplicated(groups).any() or full.duplicated(groups).any():
        return pd.DataFrame()
    paired = masked.merge(full, on=groups, how="inner", suffixes=("_masked", "_full"), validate="one_to_one")
    loss_columns = []
    for name in values:
        short = name.removeprefix("macro_")
        column = f"masking_loss_{short}"
        sign = -1 if name == "macro_auprc" else 1
        paired[column] = sign * (paired[f"{name}_masked"]-paired[f"{name}_full"])
        loss_columns.append(column)
    baseline_groups = [name for name in groups if name != "model"]
    baseline = paired.loc[paired.model.eq(baseline_name), baseline_groups+loss_columns]
    paired = paired.merge(baseline, on=baseline_groups, how="left", suffixes=("", "_PU"), validate="many_to_one")
    difference_columns = []
    for name in loss_columns:
        column = name.replace("masking_loss_", "extra_sharing_gain_")
        paired[column] = paired[f"{name}_PU"]-paired[name]
        difference_columns.append(column)
    return _equal_repetition_mean(paired, loss_columns+difference_columns)


def notebook_summaries(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Bound notebook output to 8 tables, 300 rows and 14 columns per table.

    Every table states its scope. Raw exports remain the complete record;
    over-limit grids receive a context-balanced preview, never an unlabeled
    truncation. Configuration values are actual selected values, not means.
    """
    output = {}
    aggregate = tables.get("aggregate", pd.DataFrame())
    if aggregate.empty:
        raw = tables.get("metrics", pd.DataFrame())
        aggregate = _equal_repetition_mean(raw, _metric_columns(raw))
    assay_only = _assay_only(aggregate)
    if not aggregate.empty:
        endpoint = _report_scope(aggregate, endpoint=True)
        endpoint = _merge_worst_panel_metrics(
            endpoint, tables.get("worst_panel", pd.DataFrame()))
        if not endpoint.empty:
            endpoint_values = _metric_columns(endpoint)
            endpoint_values.extend(name for name in (
                "worst_panel_macro_auprc", "worst_panel_macro_log_loss",
                "worst_panel_macro_brier",
            ) if name in endpoint)
            output["primary_endpoint_metrics"] = _report_view(endpoint, endpoint_values)
        curves = _report_scope(aggregate)
        if "model" in curves:
            curves = curves.loc[curves.model.isin(_PU_MODELS)]
        if "block_fraction" in curves:
            curves = _endpoint(curves)  # all blocks; largest positive loss within each block
        if not curves.empty and not assay_only:
            output["pu_curves"] = _report_view(curves, _metric_columns(curves)[:4])
        if assay_only:
            animals = tables.get("per_animal_aggregate", pd.DataFrame())
            if not animals.empty:
                output["per_animal_measured_metrics"] = _report_view(animals,
                    [name for name in ("macro_auprc", "macro_log_loss", "macro_brier",
                     "macro_auroc", "macro_predicted_prevalence", "macro_reference_prevalence")
                     if name in animals], extra=("animal",))
    selections = _selected_by_repetition(tables.get("selected", pd.DataFrame()))
    if not selections.empty:
        output["selected_hyperparameters_by_repetition"] = _report_view(selections,
            ["recorded_folds", "fold_configurations"], extra=("repetition",))
    health = _selection_health(tables)
    if not health.empty:
        values = [name for name in health if name not in _CONTEXT]
        output["selection_and_convergence"] = _report_view(health, values)
    joint = tables.get("joint_selection_diagnostics", pd.DataFrame())
    if joint.empty:
        joint = joint_selection_diagnostics(tables)
    joint = _report_scope(joint, endpoint=True)
    if not joint.empty:
        groups = [name for name in _CONTEXT if name in joint]
        values = [name for name in ("direct_minus_selected_validation_loss",
                  "selected_minus_best_recorded_validation_loss") if name in joint]
        joint_summary = _equal_repetition_mean(joint, values)
        if not joint_summary.empty:
            output["joint_validation_selection"] = _report_view(joint_summary, values)
    detection = _report_scope(tables.get("detection", pd.DataFrame()), endpoint=True)
    values = [name for name in ("detection_log_loss", "detection_brier", "detection_mean",
        "detection_observed_rate", "sensitivity_mae", "sensitivity_rmse") if name in detection]
    if not detection.empty and values and not assay_only:
        output["detection_calibration"] = _report_view(_equal_repetition_mean(detection, values), values)
    controls = _matched_panel_control(tables)
    if not controls.empty:
        # When the optional thinning grid is enabled, keep this diagnostic at
        # its largest loss, while retaining every evaluation block fraction.
        controls = _endpoint(controls)
        values = [name for name in controls if name.startswith(("masking_loss_", "extra_sharing_gain_"))]
        output["matched_full_panel_control"] = _report_view(controls, values)
    contrasts = tables.get("overlap_contrasts", pd.DataFrame())
    if not contrasts.empty:
        contrast_summary = _overlap_contrast_summary(contrasts)
        values = [name for name in ("metric", "mean_advantage_vs_PU",
                                    "mean_R_vs_100pct", "sd_R_vs_100pct",
                                    "n_repetitions") if name in contrast_summary]
        if not contrast_summary.empty and values:
            output["overlap_contrasts"] = _report_view(
                contrast_summary, values, extra=("actual_overlap",))
    failures = tables.get("failures", pd.DataFrame())
    if not failures.empty:
        values = [name for name in ("stage", "reason", "error", "exception") if name in failures]
        output["failed_units"] = _report_view(failures, values, extra=("repetition", "outer_fold"))
    return {name: _bounded_report(frame, _REPORT_LIMITS[name]) for name, frame in output.items()}


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
    protocol = artifacts.manifest.get("protocol", {})
    assay_only = any(_assay_only(artifacts.tables.get(name, pd.DataFrame()))
                     for name in ("aggregate", "metrics"))
    if isinstance(protocol, Mapping):
        saved_settings = {name: protocol[name] for name in (
            "n_outer_folds", "n_repetitions", "use_location", "use_target_features",
            "paired_fraction", "strategy", "candidate_budget", "supervision_profile") if name in protocol}
        if assay_only:
            saved_settings.pop("paired_fraction", None)
        if saved_settings:
            print(f"Saved run settings (authoritative for these results): {saved_settings}", flush=True)
    runtime = {name: artifacts.manifest[name] for name in (
        "parallel_unit", "requested_n_jobs", "effective_n_jobs", "scheduled_tasks",
        "planned_model_evaluations", "cached_model_evaluations", "reused_fit_model_evaluations",
        "new_or_mixed_model_evaluations", "unknown_fit_model_evaluations") if name in artifacts.manifest}
    if runtime:
        print(f"Execution settings recorded in this export: {runtime}", flush=True)
    if assay_only:
        print("Measured-assay metrics give folds equal weight within each repetition, then repetitions equal weight. "
              "Only naturally measured entries are evaluated. Unassayed-target predictions are unvalidated "
              "assay-score extrapolations, not measured anatomical connectivity. "
              "No paired reference or detector calibration is used. No fold-based confidence intervals.", flush=True)
    else:
        print("Metric means give folds equal weight within each repetition, then repetitions equal weight. "
              "Endpoints use each model's largest recorded primary loss (or natural paired labels); "
              "block experiments use the largest block fraction and masked training. "
              "PU curves retain all primary loss rates, or all blocks at the largest positive loss. "
              "Selection/convergence summarizes all primary rates. No fold-based confidence intervals.", flush=True)
    summaries = notebook_summaries(artifacts.tables)
    for name, frame in summaries.items():
        total = frame.attrs.get("total_rows", len(frame))
        print(f"\n{name}: showing {len(frame)}/{total} rows", flush=True)
        fixed = frame.attrs.get("fixed_coordinates", {})
        if fixed:
            print("Fixed coordinates: " + "; ".join(f"{key}={value}" for key, value in fixed.items()), flush=True)
        if frame.attrs.get("omitted_rows", 0):
            print(f"Bounded, context-balanced preview: {frame.attrs['omitted_rows']} rows omitted; "
                  f"{frame.attrs.get('preview_contexts', 1)}/{frame.attrs.get('total_contexts', 1)} contexts shown. "
                  "Use the complete CSV exports for all configurations/contexts.", flush=True)
        if frame.attrs.get("omitted_columns") or frame.attrs.get("clipped_cells"):
            print("Some columns or long cell text are omitted here; complete values remain in the CSV exports.", flush=True)
        if name == "joint_validation_selection":
            print("Validation-only audit: positive direct-minus-selected favors the selected candidate; "
                  "selected-minus-best measures its gap from the best converged recorded candidate. "
                  "Unknown evidence remains NaN; these are not test gains.", flush=True)
        if name == "matched_full_panel_control":
            print("Matched rep/fold/test entries: positive masking_loss means masking hurt. "
                  "Positive extra_sharing_gain means masking increased the model's advantage over PU; "
                  "zero means no additional sharing gain attributable to panel masking. "
                  "Values average paired differences, not unmatched aggregates.", flush=True)
        # Explicit limits apply even after a previous cell enabled full pandas
        # display. The frames themselves are bounded, including custom display functions.
        with pd.option_context("display.max_rows", None, "display.max_columns", None,
                               "display.max_colwidth", None, "display.precision", 6):
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
    selected = artifacts.tables.get("selected", pd.DataFrame())
    tuning = artifacts.tables.get("tuning", pd.DataFrame())
    for name, frame, status in (("final-fit", selected, _final_status(selected)),
        ("candidate", tuning, _boolean_status(tuning.get("converged",
         pd.Series(index=tuning.index, dtype=object))))):
        if not frame.empty:
            print(f"All scenarios, {name} records: {len(frame)}; "
                  f"not converged: {int(status.eq('not_converged').sum())}; "
                  f"status not recorded: {int(status.eq('not_recorded').sum())}.", flush=True)
    print("Compact output is capped at 8 tables / 300 rows / 14 columns. "
          "Set SHOW_FULL_DIAGNOSTICS=True to display every saved table and the full manifest.", flush=True)


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
