"""Results-only analysis helpers embedded in measurement-degradation V3 notebooks.

This module contains plotting and evaluation code only.  It never fits a model,
changes a checkpoint, or writes into a completed experiment export.  The V3
builder embeds :func:`notebook_source` into each generated notebook so the
notebooks remain self-contained when opened outside a repository checkout.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from gene2wire.experiments.measurement_plotting import (
    FULL_MODEL_ORDER,
    KEY_MODELS,
    MODEL_COLORS,
)
from gene2wire.experiments.pipeline import slug


FUNKY_METRICS = (
    ("macro_auprc", "AUPRC", "circle", True),
    ("macro_log_loss", "Log loss", "bar", False),
    ("macro_auroc", "AUROC", "circle", True),
    ("macro_brier", "Brier", "bar", False),
)

REFERENCE_PROBABILITY_PREFIX = "reference"
EXTRA_INFORMATION_MODELS = frozenset({
    "Reference-only",
    "Reference+PU",
    "Reference+PU-MIRT",
    "Reference+PU-Joint",
    "RF-reference",
    "RF-mixed",
})


def _as_frame(value, *, name: str, required: Sequence[str] = ()) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = set(required).difference(value.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    return value.copy()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _roles(frame: pd.DataFrame, role: str) -> pd.Series:
    return frame["condition_roles"].astype(str).str.contains(role, regex=False)


def _reference_probability(frame: pd.DataFrame) -> pd.Series:
    if "probability_semantics" not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)
    return frame["probability_semantics"].fillna("").astype(str).str.startswith(
        REFERENCE_PROBABILITY_PREFIX
    )


def _ordered_models(values: Sequence[str], *, key_models=KEY_MODELS) -> list[str]:
    present = list(dict.fromkeys(map(str, values)))
    priority = {model: index for index, model in enumerate(FULL_MODEL_ORDER)}
    key_priority = {model: index for index, model in enumerate(key_models)}
    return sorted(
        present,
        key=lambda model: (
            0 if model in key_priority else 1,
            key_priority.get(model, priority.get(model, len(priority))),
            model,
        ),
    )


def _models_for_mode(frame: pd.DataFrame, mode: str, key_models=KEY_MODELS) -> list[str]:
    if mode not in {"key", "full"}:
        raise ValueError("mode must be 'key' or 'full'")
    present = set(frame["model"].dropna().astype(str))
    if mode == "key":
        # Keep the declared rows even when a model failed every condition.  A
        # fully absent model must appear as NA instead of silently vanishing.
        models = list(dict.fromkeys(map(str, key_models)))
    else:
        extras = sorted(present.difference(FULL_MODEL_ORDER))
        models = [*FULL_MODEL_ORDER, *extras]
    if not models:
        raise ValueError(f"No {mode} models are present in the plotting table")
    return models


def _model_color(model: str, index: int):
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    palette = plt.get_cmap("tab20")
    return palette(index % palette.N)


def plot_degradation_pair(
    table: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    mode: str = "key",
    key_models: Sequence[str] = KEY_MODELS,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot Macro-AUPRC and reference-probability log loss side by side.

    The degradation coordinate always runs from 100% at the left toward 0% at
    the right.  Ranking metrics remain available for every model.  The proper
    log-loss panel omits observed/mixed-probability models rather than treating
    a detection probability as a reference probability.
    """
    frame = _as_frame(
        table,
        name="degradation table",
        required=(x_col, "model", "macro_auprc", "macro_log_loss"),
    )
    if frame.empty:
        raise ValueError("degradation table must contain at least one row")
    frame[x_col] = _numeric(frame, x_col)
    frame["macro_auprc"] = _numeric(frame, "macro_auprc")
    frame["macro_log_loss"] = _numeric(frame, "macro_log_loss")
    frame = frame.loc[np.isfinite(frame[x_col]) & frame[x_col].between(0, 1)]
    models = _models_for_mode(frame, mode, key_models)
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.5), constrained_layout=True)
    specifications = (
        ("macro_auprc", "Macro AUPRC ↑", pd.Series(True, index=frame.index)),
        ("macro_log_loss", "Macro log loss ↓", _reference_probability(frame)),
    )
    for axis, (metric, ylabel, semantic_mask) in zip(axes, specifications):
        plotting = frame.loc[semantic_mask & np.isfinite(frame[metric])].copy()
        summary = (
            plotting.groupby(["model", x_col], observed=True, sort=False)[metric]
            .mean()
            .reset_index()
        )
        for index, model in enumerate(models):
            selected = summary.loc[summary["model"].astype(str).eq(model)].sort_values(
                x_col, ascending=False
            )
            if selected.empty:
                continue
            axis.plot(
                selected[x_col],
                selected[metric],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=model,
                color=_model_color(model, index),
            )
        axis.set_xlim(1.0, 0.0)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.set_xlabel(f"{x_label} (100% → 0%)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            frameon=False,
            loc="outside lower center",
            ncol=min(4, max(1, len(labels))),
        )
    figure.suptitle(title or f"{x_label} degradation — {mode} models")
    return figure, axes


def augment_projection_budget_metrics(
    table: pd.DataFrame,
    *,
    per_target: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add confirmation yield and union-reference false-positive measures.

    Macro ratios are computed from the per-target table.  Pooled ratios are
    explicitly named ``micro_*``.  A union-negative selection is an assay-
    reference false positive, not proof of a biological false projection.
    """
    required = (
        "dataset",
        "repetition",
        "model",
        "budget_fraction",
        "selected_candidates",
        "candidate_count",
        "amplification_confirmed_positive_count",
        "recovered_positive_count",
    )
    summary = _as_frame(table, name="Projection budget table", required=required)
    targets = _as_frame(
        pd.DataFrame() if per_target is None else per_target,
        name="Projection per-target budget table",
    )
    for frame in (summary, targets):
        if frame.empty:
            continue
        missing = set(required).difference(frame.columns)
        if missing:
            raise ValueError(
                "Projection per-target budget table is missing required columns: "
                f"{sorted(missing)}"
            )
        for column in (
            "budget_fraction",
            "selected_candidates",
            "candidate_count",
            "amplification_confirmed_positive_count",
            "recovered_positive_count",
        ):
            frame[column] = _numeric(frame, column)
        frame["amplification_reference_false_positive_count"] = (
            frame["selected_candidates"] - frame["recovered_positive_count"]
        )
        frame["amplification_reference_negative_count"] = (
            frame["candidate_count"]
            - frame["amplification_confirmed_positive_count"]
        )
        frame["amplification_confirmed_false_negative_count"] = (
            frame["amplification_confirmed_positive_count"]
            - frame["recovered_positive_count"]
        )
        frame["amplification_reference_true_negative_count"] = (
            frame["amplification_reference_negative_count"]
            - frame["amplification_reference_false_positive_count"]
        )
        selected = frame["selected_candidates"].to_numpy(dtype=float)
        negatives = frame["amplification_reference_negative_count"].to_numpy(dtype=float)
        candidates = frame["candidate_count"].to_numpy(dtype=float)
        positives = frame["amplification_confirmed_positive_count"].to_numpy(dtype=float)
        recovered = frame["recovered_positive_count"].to_numpy(dtype=float)
        false_positive = frame[
            "amplification_reference_false_positive_count"
        ].to_numpy(dtype=float)
        precision = np.divide(
            recovered,
            selected,
            out=np.full_like(recovered, np.nan),
            where=selected > 0,
        )
        recall = np.divide(
            recovered,
            positives,
            out=np.full_like(recovered, np.nan),
            where=positives > 0,
        )
        prevalence = np.divide(
            positives,
            candidates,
            out=np.full_like(recovered, np.nan),
            where=candidates > 0,
        )
        frame["amplification_confirmed_precision"] = precision
        frame["amplification_confirmed_recall"] = recall
        frame["amplification_reference_fdr"] = 1.0 - precision
        frame["amplification_reference_fpr"] = np.divide(
            false_positive,
            negatives,
            out=np.full_like(recovered, np.nan),
            where=negatives > 0,
        )
        frame["amplification_confirmed_f1"] = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.full_like(recovered, np.nan),
            where=np.isfinite(precision + recall) & ((precision + recall) > 0),
        )
        frame["amplification_confirmed_lift"] = np.divide(
            precision,
            prevalence,
            out=np.full_like(recovered, np.nan),
            where=prevalence > 0,
        )

    if not targets.empty:
        keys = ["dataset", "repetition", "model", "budget_fraction"]
        ratios = (
            "amplification_confirmed_recall",
            "amplification_confirmed_precision",
            "amplification_reference_fdr",
            "amplification_reference_fpr",
            "amplification_confirmed_f1",
            "amplification_confirmed_lift",
        )
        macro = targets.groupby(keys, dropna=False, observed=True)[list(ratios)].mean()
        summary = summary.drop(columns=[name for name in ratios if name in summary], errors="ignore")
        summary = summary.merge(macro.reset_index(), on=keys, how="left", validate="one_to_one")

    # Aggregate-count ratios are always retained with a micro prefix.
    selected = summary["selected_candidates"].to_numpy(dtype=float)
    candidates = summary["candidate_count"].to_numpy(dtype=float)
    positives = summary["amplification_confirmed_positive_count"].to_numpy(dtype=float)
    recovered = summary["recovered_positive_count"].to_numpy(dtype=float)
    negatives = candidates - positives
    false_positive = selected - recovered
    micro_precision = np.divide(
        recovered, selected, out=np.full_like(recovered, np.nan), where=selected > 0
    )
    micro_recall = np.divide(
        recovered, positives, out=np.full_like(recovered, np.nan), where=positives > 0
    )
    prevalence = np.divide(
        positives, candidates, out=np.full_like(recovered, np.nan), where=candidates > 0
    )
    summary["micro_amplification_confirmed_precision"] = micro_precision
    summary["micro_amplification_confirmed_recall"] = micro_recall
    summary["micro_amplification_reference_fdr"] = 1.0 - micro_precision
    summary["micro_amplification_reference_fpr"] = np.divide(
        false_positive,
        negatives,
        out=np.full_like(recovered, np.nan),
        where=negatives > 0,
    )
    summary["micro_amplification_confirmed_f1"] = np.divide(
        2 * micro_precision * micro_recall,
        micro_precision + micro_recall,
        out=np.full_like(recovered, np.nan),
        where=np.isfinite(micro_precision + micro_recall)
        & ((micro_precision + micro_recall) > 0),
    )
    summary["micro_amplification_confirmed_lift"] = np.divide(
        micro_precision,
        prevalence,
        out=np.full_like(recovered, np.nan),
        where=prevalence > 0,
    )
    return summary, targets


def plot_projection_budget_metrics(
    table: pd.DataFrame,
    *,
    models: Sequence[str] = KEY_MODELS,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot amplification-confirmed recovery and reference FP trade-offs."""
    frame = _as_frame(
        table,
        name="Projection budget metrics",
        required=(
            "model",
            "budget_fraction",
            "amplification_confirmed_recall",
            "amplification_confirmed_precision",
            "amplification_reference_fdr",
            "amplification_reference_fpr",
            "amplification_confirmed_f1",
            "amplification_confirmed_lift",
        ),
    )
    order = [model for model in models if model in set(frame["model"].astype(str))]
    if not order:
        raise ValueError("Projection budget metrics contain none of the requested models")
    specifications = (
        ("amplification_confirmed_recall", "Confirmed recall ↑", True),
        ("amplification_confirmed_precision", "Confirmation precision ↑", True),
        ("amplification_confirmed_f1", "Confirmed F1 ↑", True),
        ("amplification_confirmed_lift", "Enrichment over random ↑", False),
        ("amplification_reference_fdr", "Union-reference FDR ↓", True),
        ("amplification_reference_fpr", "Union-reference FPR ↓", True),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=True)
    for axis, (metric, ylabel, percentage) in zip(axes.flat, specifications):
        plotting = frame.copy()
        plotting[metric] = _numeric(plotting, metric)
        summary = (
            plotting.groupby(["model", "budget_fraction"], observed=True)[metric]
            .mean()
            .reset_index()
        )
        for index, model in enumerate(order):
            selected = summary.loc[summary["model"].astype(str).eq(model)].sort_values(
                "budget_fraction"
            )
            axis.plot(
                selected["budget_fraction"],
                selected[metric],
                marker="o",
                linewidth=1.7,
                markersize=4,
                label=model,
                color=_model_color(model, index),
            )
        axis.set_xlabel("Top-ranked candidate budget")
        axis.set_ylabel(ylabel)
        axis.set_xlim(left=0.0)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        if percentage:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="outside lower center",
        ncol=min(4, len(labels)),
    )
    figure.suptitle(title or "Projection-TAGs amplification-reference recovery")
    return figure, axes


def _metric_vector(truth: np.ndarray, probability: np.ndarray | None,
                   ranking: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=int)
    ranking = np.asarray(ranking, dtype=float)
    both = len(truth) > 0 and np.unique(truth).size == 2
    result = {
        "auprc": float(average_precision_score(truth, ranking)) if both else np.nan,
        "auroc": float(roc_auc_score(truth, ranking)) if both else np.nan,
        "brier": np.nan,
        "log_loss": np.nan,
        "predicted_prevalence": np.nan,
        "reference_prevalence": float(np.mean(truth)) if len(truth) else np.nan,
    }
    if probability is not None:
        values = np.asarray(probability, dtype=float)
        if values.shape != truth.shape or not np.all(np.isfinite(values)):
            raise ValueError("Candidate probabilities must be finite and aligned")
        values = np.clip(values, 1e-9, 1 - 1e-9)
        result.update(
            brier=float(np.mean((truth - values) ** 2)),
            log_loss=float(np.mean(-truth * np.log(values) - (1 - truth) * np.log1p(-values))),
            predicted_prevalence=float(np.mean(values)),
        )
    return result


def _natural_semantics(metrics: pd.DataFrame) -> dict[str, str]:
    required = {"model", "mechanism", "condition_roles", "probability_semantics"}
    if not required.issubset(metrics):
        raise ValueError(
            "metrics.csv must record model probability_semantics for natural recovery"
        )
    frame = metrics.loc[
        metrics["mechanism"].astype(str).eq("natural")
        & _roles(metrics, "natural_recovery")
    ]
    result = {}
    for model, values in frame.groupby("model", observed=True)["probability_semantics"]:
        unique = tuple(sorted(set(values.dropna().astype(str))))
        if len(unique) != 1:
            raise ValueError(
                f"Natural recovery has ambiguous probability semantics for {model}: {unique}"
            )
        result[str(model)] = unique[0]
    return result


def _posterior_after_nondetection(reference_probability, sensitivity):
    p = np.asarray(reference_probability, dtype=float)
    e = np.asarray(sensitivity, dtype=float)
    if p.shape != e.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(e)):
        raise ValueError("Reference probability and sensitivity must be finite and aligned")
    if np.any((p < 0) | (p > 1)) or np.any((e < 0) | (e > 1)):
        raise ValueError("Reference probability and sensitivity must lie in [0, 1]")
    denominator = 1.0 - e * p
    return np.divide(
        (1.0 - e) * p,
        denominator,
        out=np.full_like(p, np.nan),
        where=denominator > 1e-12,
    )


def derive_projection_natural_metrics(
    artifacts,
    *,
    budgets: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
) -> dict[str, pd.DataFrame]:
    """Re-evaluate saved natural OOF candidates without fitting or export writes.

    For a reference-probability model, standard non-detection changes the
    candidate probability from ``p`` to ``h=(1-e)p/(1-ep)``.  Ranking-only
    observed/mixed models retain their saved score but receive no Brier or log
    loss.  The returned tables are newly allocated in memory.
    """
    metrics = _as_frame(
        artifacts.tables.get("metrics", pd.DataFrame()),
        name="metrics.csv",
        required=("model", "mechanism", "condition_roles", "probability_semantics"),
    )
    semantics = _natural_semantics(metrics)
    model_lookup = {slug(model): model for model in semantics}
    export_dir = Path(artifacts.export_dir)
    units = export_dir / "units"
    if not units.is_dir():
        raise FileNotFoundError(
            f"Projection natural re-evaluation needs saved OOF units below {units}"
        )
    budgets = tuple(float(value) for value in budgets)
    if not budgets or len(set(budgets)) != len(budgets) or any(
        not np.isfinite(value) or value <= 0 or value > 1 for value in budgets
    ):
        raise ValueError("budgets must contain distinct fractions in (0, 1]")

    manifest = getattr(artifacts, "manifest", {}) or {}
    measurement_protocol = manifest.get("measurement_protocol") or {}
    saved_protocol = manifest.get("protocol") or {}
    try:
        expected_panel_seed_count = int(measurement_protocol["n_panel_seeds"])
        expected_outer_fold_count = int(saved_protocol["n_outer_folds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Projection natural re-evaluation needs manifest n_panel_seeds and "
            "n_outer_folds to prove OOF completeness"
        ) from error
    if expected_panel_seed_count < 1 or expected_outer_fold_count < 1:
        raise ValueError("Manifest panel-seed and outer-fold counts must be positive")
    expected_seed_folds = {
        (panel_seed, outer_fold)
        for panel_seed in range(expected_panel_seed_count)
        for outer_fold in range(expected_outer_fold_count)
    }

    print(
        f"[V3 Projection] discover natural-recovery OOF files below {units}",
        flush=True,
    )
    expected_contexts: dict[tuple[str, int], set[tuple[int, int]]] = {}
    prediction_paths: dict[tuple[str, int, str, int, int], Path] = {}
    for audit_path in sorted(units.glob("*/audit.json")):
        context = json.loads(audit_path.read_text(encoding="utf-8"))
        if context.get("mechanism") != "natural" or "natural_recovery" not in str(
            context.get("condition_roles", "")
        ):
            continue
        dataset = str(context.get("dataset"))
        # Real-data ``repetition`` is the panel seed; ``data_repetition`` is the
        # independent data unit. Natural full-panel copies are collapsed only
        # after their arrays are verified identical.
        repetition = int(context.get(
            "data_repetition", context.get("repetition", 0)
        ))
        panel_seed = int(context.get("panel_seed", context.get("repetition", 0)))
        outer_fold = int(context.get("outer_fold", -1))
        expected_contexts.setdefault((dataset, repetition), set()).add(
            (panel_seed, outer_fold)
        )
        for prediction_path in sorted(audit_path.parent.glob("*_predictions.npz")):
            model = model_lookup.get(prediction_path.name[: -len("_predictions.npz")])
            if model is None:
                continue
            key = (dataset, repetition, model, panel_seed, outer_fold)
            if key in prediction_paths:
                raise ValueError(
                    f"Duplicate natural prediction file for {key}: "
                    f"{prediction_paths[key]} and {prediction_path}"
                )
            prediction_paths[key] = prediction_path
    if not prediction_paths:
        raise ValueError("No saved Projection-TAGs natural-recovery OOF candidates were found")
    print(
        f"[V3 Projection] found {len(prediction_paths)} prediction files across "
        f"{len(expected_contexts)} data-repetition context(s) and "
        f"{len(semantics)} model(s)",
        flush=True,
    )
    for (dataset, repetition), contexts in sorted(expected_contexts.items()):
        if contexts != expected_seed_folds:
            raise ValueError(
                "Natural-recovery OOF audit contexts are incomplete for "
                f"{dataset}, data repetition={repetition}; "
                f"missing={sorted(expected_seed_folds.difference(contexts))}, "
                f"extra={sorted(contexts.difference(expected_seed_folds))}"
            )

    def load_candidate_arrays(path: Path, model: str) -> dict[str, object]:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        required_arrays = {
            "prediction", "reference", "observed", "source_measured",
            "cell_ids", "target_ids",
        }
        missing = required_arrays.difference(arrays)
        if missing:
            raise ValueError(
                f"{path} lacks natural-recovery arrays: {sorted(missing)}"
            )
        prediction = np.asarray(arrays["prediction"], dtype=float)
        reference = np.asarray(arrays["reference"], dtype=bool)
        observed = np.asarray(arrays["observed"], dtype=bool)
        measured = np.asarray(arrays["source_measured"], dtype=bool)
        if not (
            prediction.shape == reference.shape == observed.shape == measured.shape
        ):
            raise ValueError(f"Natural-recovery arrays have different shapes in {path}")
        cells = np.asarray(arrays["cell_ids"]).astype(str)
        targets = np.asarray(arrays["target_ids"]).astype(str)
        if cells.shape != (len(prediction),) or targets.shape != (prediction.shape[1],):
            raise ValueError(f"Natural-recovery IDs do not align in {path}")
        semantics_value = semantics[model]
        if semantics_value.startswith(REFERENCE_PROBABILITY_PREFIX):
            if "estimated_sensitivity" not in arrays:
                raise ValueError(
                    f"{path} lacks sensitivity required for h-posterior scoring"
                )
            probability = _posterior_after_nondetection(
                prediction, np.asarray(arrays["estimated_sensitivity"], dtype=float)
            )
            ranking = probability
            ranking_source = "h_posterior_after_standard_nondetection"
        else:
            probability = None
            ranking = np.asarray(arrays.get("ranking_score", prediction), dtype=float)
            ranking_source = "saved_ranking_only"
        if ranking.shape != prediction.shape:
            raise ValueError(f"{path} has a ranking score with the wrong shape")
        candidates = measured & ~observed
        if not np.all(np.isfinite(ranking[candidates])) or (
            probability is not None and not np.all(np.isfinite(probability[candidates]))
        ):
            raise ValueError(
                "Undefined/nonfinite natural-candidate score; this commonly means "
                f"p=e=1 in {path}"
            )
        return {
            "cells": cells,
            "targets": targets,
            "truth": reference,
            "candidates": candidates,
            "ranking": ranking,
            "probability": probability,
            "semantics": semantics_value,
            "ranking_source": ranking_source,
        }

    def assert_panel_copy(canonical, duplicate, *, context: str) -> None:
        for field in ("cells", "targets", "truth", "candidates"):
            if not np.array_equal(canonical[field], duplicate[field]):
                raise ValueError(
                    f"Natural panel-seed copies disagree in {field}: {context}"
                )
        if not np.array_equal(
            canonical["ranking"], duplicate["ranking"], equal_nan=True
        ):
            raise ValueError(f"Natural panel-seed rankings disagree: {context}")
        canonical_probability = canonical["probability"]
        duplicate_probability = duplicate["probability"]
        if (canonical_probability is None) != (duplicate_probability is None):
            raise ValueError(f"Natural panel-seed probability semantics disagree: {context}")
        if canonical_probability is not None and not np.array_equal(
            canonical_probability, duplicate_probability, equal_nan=True
        ):
            raise ValueError(f"Natural panel-seed probabilities disagree: {context}")

    for (dataset, repetition), contexts in sorted(expected_contexts.items()):
        for model in sorted(semantics):
            recorded = {
                (panel_seed, outer_fold)
                for (current_dataset, current_repetition, current_model,
                     panel_seed, outer_fold) in prediction_paths
                if (current_dataset, current_repetition, current_model)
                == (dataset, repetition, model)
            }
            if recorded != contexts:
                raise ValueError(
                    "Natural-recovery prediction files are incomplete for "
                    f"{dataset}, data repetition={repetition}, model={model}; "
                    f"missing={sorted(contexts.difference(recorded))}, "
                    f"extra={sorted(recorded.difference(contexts))}"
                )

    per_target_rows = []
    budget_target_rows = []
    audit_rows = []
    reference_candidates: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    seen_models_by_target: dict[tuple[str, int, str], set[str]] = {}
    work = [
        (dataset, repetition, model)
        for dataset, repetition in sorted(expected_contexts)
        for model in sorted(semantics)
    ]
    for work_index, (dataset, repetition, model) in enumerate(work, start=1):
        contexts = expected_contexts[(dataset, repetition)]
        panel_seeds = sorted({panel_seed for panel_seed, _ in contexts})
        canonical_seed = panel_seeds[0]
        folds = sorted({outer_fold for panel_seed, outer_fold in contexts
                        if panel_seed == canonical_seed})
        for panel_seed in panel_seeds[1:]:
            other_folds = sorted({outer_fold for current_seed, outer_fold in contexts
                                  if current_seed == panel_seed})
            if other_folds != folds:
                raise ValueError(
                    f"Natural panel seeds have different fold sets: {dataset}, "
                    f"data repetition={repetition}, seed={panel_seed}"
                )
        print(
            f"[V3 Projection {work_index}/{len(work)}] verify panel copies and "
            f"evaluate {dataset}, data repetition={repetition}, model={model}",
            flush=True,
        )
        cells_by_target: dict[str, list[np.ndarray]] = {}
        truth_by_target: dict[str, list[np.ndarray]] = {}
        ranking_by_target: dict[str, list[np.ndarray]] = {}
        probability_by_target: dict[str, list[np.ndarray]] = {}
        ranking_source = None
        duplicate_candidates = 0
        for outer_fold in folds:
            canonical = load_candidate_arrays(
                prediction_paths[(
                    dataset, repetition, model, canonical_seed, outer_fold
                )],
                model,
            )
            if ranking_source is None:
                ranking_source = str(canonical["ranking_source"])
            elif ranking_source != canonical["ranking_source"]:
                raise ValueError(f"Natural folds mix ranking sources for {model}")
            for panel_seed in panel_seeds[1:]:
                duplicate = load_candidate_arrays(
                    prediction_paths[(
                        dataset, repetition, model, panel_seed, outer_fold
                    )],
                    model,
                )
                assert_panel_copy(
                    canonical,
                    duplicate,
                    context=(f"{dataset}, data repetition={repetition}, model={model}, "
                             f"fold={outer_fold}, panel seed={panel_seed}"),
                )
                duplicate_candidates += int(np.asarray(
                    canonical["candidates"], dtype=bool
                ).sum())
            for column, target in enumerate(canonical["targets"]):
                selected = np.asarray(canonical["candidates"], dtype=bool)[:, column]
                cells_by_target.setdefault(str(target), []).append(
                    np.asarray(canonical["cells"])[selected]
                )
                truth_by_target.setdefault(str(target), []).append(
                    np.asarray(canonical["truth"], dtype=bool)[selected, column]
                )
                ranking_by_target.setdefault(str(target), []).append(
                    np.asarray(canonical["ranking"], dtype=float)[selected, column]
                )
                if canonical["probability"] is not None:
                    probability_by_target.setdefault(str(target), []).append(
                        np.asarray(canonical["probability"], dtype=float)[selected, column]
                    )

        model_candidate_count = 0
        for target in sorted(cells_by_target):
            cells = np.concatenate(cells_by_target[target]).astype(str)
            truth = np.concatenate(truth_by_target[target]).astype(bool)
            ranking = np.concatenate(ranking_by_target[target]).astype(float)
            probability = (
                np.concatenate(probability_by_target[target]).astype(float)
                if target in probability_by_target else None
            )
            identity_order = np.argsort(cells, kind="stable")
            cells = cells[identity_order]
            truth = truth[identity_order]
            ranking = ranking[identity_order]
            if probability is not None:
                probability = probability[identity_order]
            if len(np.unique(cells)) != len(cells):
                raise ValueError(
                    "A natural candidate appears in multiple outer folds in the "
                    f"canonical panel seed: {dataset}, data repetition={repetition}, "
                    f"model={model}, target={target}"
                )
            reference_key = (dataset, repetition, target)
            if reference_key not in reference_candidates:
                reference_candidates[reference_key] = (cells.copy(), truth.copy())
            else:
                reference_cells, reference_truth = reference_candidates[reference_key]
                if not (
                    np.array_equal(cells, reference_cells)
                    and np.array_equal(truth, reference_truth)
                ):
                    raise ValueError(
                        "Natural candidate identities/reference labels differ across models "
                        f"for {dataset}, data repetition={repetition}, target={target}"
                    )
            seen_models_by_target.setdefault(reference_key, set()).add(model)
            model_candidate_count += len(truth)
            scores = _metric_vector(truth, probability, ranking)
            per_target_rows.append({
                "dataset": dataset,
                "repetition": repetition,
                "model": model,
                "target": target,
                "evaluation_scope": "hidden_candidate",
                "mechanism": "natural",
                "condition_roles": "natural_recovery",
                "probability_semantics": semantics[model],
                "candidate_probability": (
                    "h_after_standard_nondetection" if probability is not None else "NA"
                ),
                "ranking_source": ranking_source,
                "candidate_count": len(truth),
                "amplification_confirmed_positive_count": int(truth.sum()),
                "amplification_reference_negative_count": int((~truth).sum()),
                "source_panel_seed_count": len(panel_seeds),
                "duplicate_candidates_collapsed": len(truth) * (len(panel_seeds) - 1),
                **scores,
            })
            ranked_order = np.lexsort((cells, -ranking))
            ranked_truth = truth[ranked_order]
            for budget in budgets:
                selected_count = (
                    max(1, int(np.ceil(budget * len(ranked_truth))))
                    if len(ranked_truth) else 0
                )
                budget_target_rows.append({
                    "dataset": dataset,
                    "repetition": repetition,
                    "model": model,
                    "target": target,
                    "budget_fraction": budget,
                    "selected_candidates": selected_count,
                    "candidate_count": len(ranked_truth),
                    "amplification_confirmed_positive_count": int(truth.sum()),
                    "recovered_positive_count": int(
                        ranked_truth[:selected_count].sum()
                    ),
                    "ranking_source": ranking_source,
                })
        audit_rows.append({
            "dataset": dataset,
            "data_repetition": repetition,
            "model": model,
            "probability_semantics": semantics[model],
            "canonical_panel_seed": canonical_seed,
            "source_panel_seed_count": len(panel_seeds),
            "outer_fold_count": len(folds),
            "prediction_file_count": len(panel_seeds) * len(folds),
            "candidate_count": model_candidate_count,
            "duplicate_candidates_collapsed": duplicate_candidates,
            "candidate_identity_check": "matched_across_models",
            "panel_seed_copy_check": "identical",
        })

    for key, models in seen_models_by_target.items():
        if models != set(semantics):
            raise ValueError(
                f"Natural candidate target {key} is missing models: "
                f"{sorted(set(semantics).difference(models))}"
            )

    print(
        "[V3 Projection] candidate identities matched; aggregate macro and "
        "budget metrics",
        flush=True,
    )

    per_target = pd.DataFrame(per_target_rows)
    group_keys = [
        "dataset",
        "repetition",
        "model",
        "evaluation_scope",
        "mechanism",
        "condition_roles",
        "probability_semantics",
        "candidate_probability",
        "ranking_source",
    ]
    metric_names = (
        "auprc",
        "auroc",
        "brier",
        "log_loss",
        "predicted_prevalence",
        "reference_prevalence",
    )
    summary_rows = []
    for key, frame in per_target.groupby(group_keys, dropna=False, observed=True):
        prefix = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        row = {
            **prefix,
            "n_targets": int(len(frame)),
            "n_nonempty_targets": int((frame["candidate_count"] > 0).sum()),
            "n_evaluable_targets": int(np.isfinite(
                pd.to_numeric(frame["auprc"], errors="coerce")
            ).sum()),
            "candidate_count": int(frame["candidate_count"].sum()),
            "amplification_confirmed_positive_count": int(
                frame["amplification_confirmed_positive_count"].sum()
            ),
            "amplification_reference_negative_count": int(
                frame["amplification_reference_negative_count"].sum()
            ),
            "scanned_prediction_files": len(prediction_paths),
            "source_panel_seed_count": int(frame["source_panel_seed_count"].max()),
            "duplicate_candidates_collapsed": int(
                frame["duplicate_candidates_collapsed"].sum()
            ),
        }
        for metric in metric_names:
            row[f"macro_{metric}"] = float(
                pd.to_numeric(frame[metric], errors="coerce").mean()
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    budget_per_target = pd.DataFrame(budget_target_rows)
    budget_keys = ["dataset", "repetition", "model", "budget_fraction"]
    budget_summary = (
        budget_per_target.groupby(budget_keys, dropna=False, observed=True)[
            [
                "selected_candidates",
                "candidate_count",
                "amplification_confirmed_positive_count",
                "recovered_positive_count",
            ]
        ]
        .sum()
        .reset_index()
    )
    ranking_sources = (
        budget_per_target.groupby(budget_keys, dropna=False, observed=True)["ranking_source"]
        .agg(lambda values: ";".join(sorted(set(values.astype(str)))))
        .reset_index()
    )
    budget_summary = budget_summary.merge(
        ranking_sources, on=budget_keys, how="left", validate="one_to_one"
    )
    # The legacy column is retained so existing plot/report code can consume
    # the V3 table.  augment_projection_budget_metrics replaces it with the
    # target-macro value below.
    budget_summary["amplification_confirmed_recall"] = np.nan
    budget_summary, budget_per_target = augment_projection_budget_metrics(
        budget_summary, per_target=budget_per_target
    )
    print(
        f"[V3 Projection] complete: {len(summary)} model summary row(s), "
        f"{len(per_target)} model-target row(s), "
        f"{len(budget_summary)} budget row(s)",
        flush=True,
    )
    return {
        "summary": summary,
        "per_target": per_target,
        "budget": budget_summary,
        "budget_per_target": budget_per_target,
        "audit": pd.DataFrame(audit_rows),
    }


def plot_projection_natural_comparison(
    table: pd.DataFrame,
    *,
    models: Sequence[str] = KEY_MODELS,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Show candidate-pool ranking and h-probability accuracy."""
    frame = _as_frame(
        table,
        name="Projection natural candidate metrics",
        required=("model", "macro_auprc", "macro_auroc", "macro_brier", "macro_log_loss"),
    )
    order = [model for model in models if model in set(frame["model"].astype(str))]
    if not order:
        raise ValueError("Projection natural metrics contain none of the requested models")
    specifications = (
        ("macro_auprc", "Candidate AUPRC ↑"),
        ("macro_log_loss", "Candidate h-log loss ↓"),
        ("macro_auroc", "Candidate AUROC ↑"),
        ("macro_brier", "Candidate h-Brier ↓"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.8), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, specifications):
        values = (
            frame.assign(**{metric: _numeric(frame, metric)})
            .groupby("model", observed=True)[metric]
            .mean()
            .reindex(order)
        )
        y = np.arange(len(order))
        finite = np.isfinite(values.to_numpy(dtype=float))
        axis.scatter(
            values.to_numpy(dtype=float)[finite],
            y[finite],
            s=58,
            color=[_model_color(order[index], index) for index in np.flatnonzero(finite)],
            zorder=3,
        )
        for index in np.flatnonzero(~finite):
            axis.text(0.5, index, "NA", transform=axis.get_yaxis_transform(),
                      ha="center", va="center", color="#777777")
        axis.set_yticks(y, order)
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.7)
        axis.spines[["top", "right", "left"]].set_visible(False)
    figure.suptitle(
        title
        or "Projection-TAGs standard-negative candidates — amplification-reference evaluation"
    )
    return figure, axes


def _coverage_label(prefix: str, requested: float, actual: float) -> str:
    requested_label = f"{100 * requested:g}%"
    if np.isclose(requested, actual, atol=5e-4):
        return f"{prefix}{requested_label}"
    return f"{prefix}{requested_label}→{100 * actual:.1f}%"


def _facet_values(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    if "sharing_strength" not in frame or frame["sharing_strength"].isna().all():
        return [("all", pd.Series(True, index=frame.index, dtype=bool))]
    numeric = pd.to_numeric(frame["sharing_strength"], errors="coerce")
    values = sorted(numeric.dropna().unique())
    return [
        (f"sharing={value:g}", np.isclose(numeric, value)) for value in values
    ]


def prepare_measurement_funky_table(
    per_repetition: pd.DataFrame,
    *,
    measurement_protocol: Mapping[str, object],
    projection_natural: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the fixed scenario × metric table used by both V3 summaries."""
    required = (
        "model",
        "condition_roles",
        "evaluation_scope",
        "mechanism",
        "gene_requested_coverage",
        "gene_coverage",
        "target_requested_coverage",
        "target_coverage",
        "positive_retention",
    )
    frame = _as_frame(per_repetition, name="per_repetition.csv", required=required)
    if frame.empty:
        raise ValueError("per_repetition.csv is empty")
    for column in (
        "gene_requested_coverage",
        "gene_coverage",
        "target_requested_coverage",
        "target_coverage",
        "positive_retention",
    ):
        frame[column] = _numeric(frame, column)
    anchor_gene = float(measurement_protocol["anchor_gene_coverage"])
    anchor_target = float(measurement_protocol["anchor_target_coverage"])
    anchor_retention = float(measurement_protocol["anchor_retention"])
    metric_columns = [specification[0] for specification in FUNKY_METRICS]
    missing_metrics = set(metric_columns).difference(frame)
    if missing_metrics:
        raise ValueError(
            f"per_repetition.csv lacks funky-summary metrics: {sorted(missing_metrics)}"
        )
    records = []
    condition_records = []

    def append_conditions(source: pd.DataFrame, family: str, coordinates, order_start: int):
        for offset, (condition, condition_label, selected) in enumerate(coordinates):
            for metric_order, (metric, metric_label, glyph, higher) in enumerate(
                FUNKY_METRICS
            ):
                condition_records.append({
                    "family": family,
                    "family_order": order_start,
                    "condition": condition,
                    "condition_label": condition_label,
                    "condition_order": order_start + offset,
                    "metric": metric,
                    "metric_label": metric_label,
                    "metric_order": metric_order,
                    "glyph": glyph,
                    "higher_is_better": higher,
                })
            if selected.empty:
                continue
            for facet, facet_mask in _facet_values(selected):
                current = selected.loc[facet_mask]
                if current.empty:
                    continue
                for model, model_rows in current.groupby("model", observed=True):
                    semantics_values = tuple(sorted(set(
                        model_rows.get(
                            "probability_semantics", pd.Series("unknown", index=model_rows.index)
                        ).dropna().astype(str)
                    )))
                    semantics = semantics_values[0] if len(semantics_values) == 1 else "mixed_records"
                    for metric_order, (metric, metric_label, glyph, higher) in enumerate(
                        FUNKY_METRICS
                    ):
                        value = float(pd.to_numeric(model_rows[metric], errors="coerce").mean())
                        if metric in {"macro_brier", "macro_log_loss"} and not semantics.startswith(
                            REFERENCE_PROBABILITY_PREFIX
                        ):
                            value = np.nan
                        records.append({
                            "facet": facet,
                            "model": str(model),
                            "family": family,
                            "family_order": order_start,
                            "condition": condition,
                            "condition_label": condition_label,
                            "condition_order": order_start + offset,
                            "metric": metric,
                            "metric_label": metric_label,
                            "metric_order": metric_order,
                            "glyph": glyph,
                            "higher_is_better": higher,
                            "value": value,
                            "probability_semantics": semantics,
                            "n_rows_averaged": int(len(model_rows)),
                        })

    primary = frame.loc[
        frame["evaluation_scope"].astype(str).eq("native_reference")
        & frame["mechanism"].astype(str).eq("assay_target_sar")
    ].copy()
    full = primary.loc[_roles(primary, "full_control")]
    append_conditions(
        primary,
        "Full control",
        [("full", "G100/T100/P100", full)],
        0,
    )

    retention = primary.loc[
        _roles(primary, "retention_curve")
        & np.isclose(primary["gene_requested_coverage"], anchor_gene)
        & np.isclose(primary["target_requested_coverage"], anchor_target)
    ]
    retention_conditions = []
    declared_retentions = tuple(map(float, measurement_protocol["retentions"]))
    for value in sorted(declared_retentions, reverse=True):
        retention_conditions.append((
            f"retention_{value:.12g}",
            f"P{100 * value:g}%",
            retention.loc[np.isclose(retention["positive_retention"], value)],
        ))
    append_conditions(retention, "Positive retention", retention_conditions, 10)

    coverage = primary.loc[
        _roles(primary, "coverage_heatmap")
        & np.isclose(primary["positive_retention"], anchor_retention)
    ]
    joint_conditions = []
    declared_gene_coverages = tuple(map(float, measurement_protocol["gene_coverages"]))
    declared_target_coverages = tuple(map(float, measurement_protocol["target_coverages"]))
    for gene_requested in sorted(declared_gene_coverages, reverse=True):
        for target_requested in sorted(declared_target_coverages, reverse=True):
            selected = coverage.loc[
                np.isclose(coverage["gene_requested_coverage"], gene_requested)
                & np.isclose(
                    coverage["target_requested_coverage"], target_requested
                )
            ]
            gene_actual = (
                float(selected["gene_coverage"].mean())
                if not selected.empty else gene_requested
            )
            target_actual = (
                float(selected["target_coverage"].mean())
                if not selected.empty else target_requested
            )
            joint_conditions.append((
                f"coverage_g{gene_requested:.12g}_t{target_requested:.12g}",
                (
                    f"{_coverage_label('G', gene_requested, gene_actual)} / "
                    f"{_coverage_label('T', target_requested, target_actual)}"
                ),
                selected,
            ))
    append_conditions(
        coverage, "Gene × target coverage", joint_conditions, 20
    )

    scar = frame.loc[
        frame["evaluation_scope"].astype(str).eq("native_reference")
        & frame["mechanism"].astype(str).eq("scar")
        & _roles(frame, "matched_uniform_control")
    ]
    if bool(measurement_protocol.get("include_matched_uniform", True)):
        append_conditions(
            scar,
            "Matched SCAR",
            [("matched_scar", f"SCAR P{100 * anchor_retention:g}%", scar)],
            40,
        )

    if projection_natural is not None and not projection_natural.empty:
        natural = _as_frame(
            projection_natural,
            name="Projection natural V3 metrics",
            required=(
                "model",
                "evaluation_scope",
                "mechanism",
                "condition_roles",
                *metric_columns,
            ),
        )
        natural = natural.loc[
            natural["evaluation_scope"].astype(str).eq("hidden_candidate")
            & natural["mechanism"].astype(str).eq("natural")
            & _roles(natural, "natural_recovery")
        ]
        append_conditions(
            natural,
            "Amplification-confirmed recovery",
            [("natural_candidates", "standard D=0", natural)],
            50,
        )

    observed = pd.DataFrame(records)
    if observed.empty:
        raise ValueError("No declared measurement scenarios were available for the funky summary")
    catalog = (
        pd.DataFrame(condition_records)
        .drop_duplicates(["condition", "metric"])
        .sort_values(["family_order", "condition_order", "metric_order"], kind="stable")
        .reset_index(drop=True)
    )
    facets = [facet for facet, _ in _facet_values(frame)]
    present_models = set(observed["model"].astype(str))
    models = [*FULL_MODEL_ORDER, *sorted(present_models.difference(FULL_MODEL_ORDER))]
    skeleton = pd.MultiIndex.from_product(
        [facets, models, catalog.index],
        names=["facet", "model", "catalog_index"],
    ).to_frame(index=False)
    skeleton = skeleton.merge(
        catalog.rename_axis("catalog_index").reset_index(),
        on="catalog_index",
        how="left",
        validate="many_to_one",
    ).drop(columns="catalog_index")
    values = observed[[
        "facet", "model", "condition", "metric", "value",
        "probability_semantics", "n_rows_averaged",
    ]]
    if values.duplicated(["facet", "model", "condition", "metric"]).any():
        raise ValueError("Funky summary has duplicate model × condition × metric rows")
    result = skeleton.merge(
        values,
        on=["facet", "model", "condition", "metric"],
        how="left",
        validate="one_to_one",
    )
    result["probability_semantics"] = result["probability_semantics"].fillna(
        "not_available"
    )
    result["n_rows_averaged"] = result["n_rows_averaged"].fillna(0).astype(int)
    # One common metric range per dataset/facet makes key and full figures use
    # identical normalization and avoids magnifying tiny within-cell deltas.
    eligible = ~result["model"].isin(EXTRA_INFORMATION_MODELS)
    result["desirability"] = np.nan
    for (facet, metric), indices in result.groupby(
        ["facet", "metric"], dropna=False, observed=True
    ).groups.items():
        indices = pd.Index(indices)
        reference = result.loc[indices[eligible.loc[indices] & np.isfinite(
            pd.to_numeric(result.loc[indices, "value"], errors="coerce")
        )], "value"].astype(float)
        if reference.empty:
            continue
        lower, upper = float(reference.min()), float(reference.max())
        values = pd.to_numeric(result.loc[indices, "value"], errors="coerce")
        if np.isclose(lower, upper):
            normalized = pd.Series(0.5, index=indices, dtype=float)
        else:
            normalized = (values - lower) / (upper - lower)
        higher = bool(result.loc[indices, "higher_is_better"].iloc[0])
        if not higher:
            normalized = 1.0 - normalized
        result.loc[indices, "desirability"] = normalized.clip(0, 1)
    return result.sort_values(
        ["facet", "family_order", "condition_order", "metric_order", "model"],
        kind="stable",
    ).reset_index(drop=True)


def _row_group(model: str) -> str:
    if model in {"Logistic", "MIRT", "Joint"}:
        return "Non-PU baselines"
    if model in {"PU", "PU-MIRT", "PU-Joint"}:
        return "Gene2Wire PU family"
    if model.startswith("Reference"):
        return "Reference-label controls"
    if model.startswith("RF-"):
        return "Random-forest controls"
    if model.startswith("Qiao"):
        return "Qiao baselines"
    if model in {"GenEML-adapted", "Inductive-PU-MC", "SAR-PU"}:
        return "External PU comparators"
    return "Other"


def plot_measurement_funky_summary(
    table: pd.DataFrame,
    *,
    mode: str,
    key_models: Sequence[str] = KEY_MODELS,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Render a dependency-free funkyheatmap-style benchmark summary.

    Circles encode ranking metrics and bars encode proper losses.  Glyph size,
    fill and length all point in the favorable direction.  Exact raw values are
    overlaid in the key view; the full view keeps glyphs legible.
    """
    frame = _as_frame(
        table,
        name="funky summary table",
        required=(
            "facet",
            "model",
            "family",
            "family_order",
            "condition",
            "condition_label",
            "condition_order",
            "metric",
            "metric_label",
            "metric_order",
            "glyph",
            "higher_is_better",
            "value",
            "desirability",
        ),
    )
    models = _models_for_mode(frame, mode, key_models)
    frame = frame.loc[frame["model"].isin(models)].copy()
    facets = list(dict.fromkeys(frame["facet"].astype(str)))
    columns = (
        frame[[
            "family",
            "family_order",
            "condition",
            "condition_label",
            "condition_order",
            "metric",
            "metric_label",
            "metric_order",
            "glyph",
        ]]
        .drop_duplicates()
        .sort_values(["family_order", "condition_order", "metric_order"], kind="stable")
        .reset_index(drop=True)
    )
    column_keys = list(zip(columns["condition"], columns["metric"]))
    width = max(12.0, 2.8 + 0.52 * len(columns))
    per_axis_height = 1.8 + 0.43 * len(models)
    figure, axes = plt.subplots(
        len(facets),
        1,
        figsize=(width, per_axis_height * len(facets)),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]
    cmap_circle = plt.get_cmap("Blues")
    cmap_bar = plt.get_cmap("Purples")
    family_colors = ("#F3F6FA", "#FFF7ED", "#F3FAF7", "#F8F3FA", "#FFF4F4", "#F4F4F4")
    for facet_index, (axis, facet) in enumerate(zip(axes, facets)):
        current = frame.loc[frame["facet"].astype(str).eq(facet)]
        lookup = current.set_index(["model", "condition", "metric"], drop=False)
        # Alternate softly colored family bands and label each block once.
        for family_index, (family, family_columns) in enumerate(
            columns.groupby("family", sort=False, observed=True)
        ):
            positions = family_columns.index.to_numpy(dtype=int)
            start, end = positions.min() - 0.5, positions.max() + 0.5
            axis.axvspan(start, end, color=family_colors[family_index % len(family_colors)], zorder=0)
            axis.text(
                (start + end) / 2,
                1.13,
                family,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
            if start > -0.5:
                axis.axvline(start, color="white", linewidth=2.0, zorder=1)
        for condition, condition_columns in columns.groupby(
            "condition", sort=False, observed=True
        ):
            positions = condition_columns.index.to_numpy(dtype=int)
            start, end = positions.min() - 0.5, positions.max() + 0.5
            axis.axvline(start, color="#CCCCCC", linewidth=0.65, zorder=1)
            axis.text(
                (start + end) / 2,
                1.035,
                str(condition_columns["condition_label"].iloc[0]),
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=35,
            )
        for row, model in enumerate(models):
            for column, (condition, metric) in enumerate(column_keys):
                key = (model, condition, metric)
                if key not in lookup.index:
                    axis.add_patch(Rectangle(
                        (column - 0.34, row - 0.27), 0.68, 0.54,
                        facecolor="#E6E6E6", edgecolor="white", linewidth=0.5,
                        zorder=2,
                    ))
                    axis.text(column, row, "NA", ha="center", va="center",
                              fontsize=5.5, color="#777777", zorder=4)
                    continue
                value_row = lookup.loc[key]
                if isinstance(value_row, pd.DataFrame):
                    value_row = value_row.iloc[0]
                value = float(value_row["value"]) if np.isfinite(value_row["value"]) else np.nan
                desirability = (
                    float(value_row["desirability"])
                    if np.isfinite(value_row["desirability"])
                    else np.nan
                )
                if not np.isfinite(value) or not np.isfinite(desirability):
                    axis.add_patch(Rectangle(
                        (column - 0.34, row - 0.27), 0.68, 0.54,
                        facecolor="#E6E6E6", edgecolor="white", linewidth=0.5,
                        zorder=2,
                    ))
                    axis.text(column, row, "NA", ha="center", va="center",
                              fontsize=5.5, color="#777777", zorder=4)
                    continue
                if value_row["glyph"] == "circle":
                    axis.scatter(
                        [column],
                        [row],
                        s=26 + 250 * desirability,
                        color=[cmap_circle(0.22 + 0.75 * desirability)],
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=3,
                    )
                else:
                    axis.add_patch(Rectangle(
                        (column - 0.36, row - 0.20), 0.72, 0.40,
                        facecolor="#ECECEC", edgecolor="white", linewidth=0.4,
                        zorder=2,
                    ))
                    axis.add_patch(Rectangle(
                        (column - 0.36, row - 0.20), 0.72 * desirability, 0.40,
                        facecolor=cmap_bar(0.24 + 0.70 * desirability),
                        edgecolor="none", zorder=3,
                    ))
                if mode == "key":
                    axis.text(
                        column,
                        row,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=5.2,
                        color="black" if desirability < 0.73 else "white",
                        zorder=4,
                    )
        axis.set_xlim(-0.5, len(columns) - 0.5)
        axis.set_ylim(len(models) - 0.5, -0.5)
        axis.set_yticks(np.arange(len(models)), models)
        axis.set_xticks(
            np.arange(len(columns)),
            [
                f"{label}{'↑' if higher else '↓'}"
                for label, higher in zip(
                    columns["metric_label"],
                    columns.merge(
                        frame[["metric", "higher_is_better"]].drop_duplicates(),
                        on="metric", how="left"
                    )["higher_is_better"],
                )
            ],
            rotation=90,
            fontsize=6.5,
        )
        axis.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        axis.tick_params(axis="y", length=0)
        axis.set_title(facet if facet != "all" else "", loc="left", pad=58, fontsize=10)
        for spine in axis.spines.values():
            spine.set_visible(False)
        # Row-family separators preserve model identities without sorting by score.
        previous = None
        for row, model in enumerate(models):
            current_group = _row_group(model)
            if previous is not None and current_group != previous:
                axis.axhline(row - 0.5, color="#BDBDBD", linewidth=0.8, zorder=4)
            previous = current_group
    figure.suptitle(
        title or f"Measurement degradation — {mode} models (funkyheatmap-style)",
        fontsize=13,
    )
    return figure, axes


def notebook_source() -> str:
    """Return this module without its loader-only function and future import."""
    source = Path(__file__).read_text(encoding="utf-8")
    source = source.replace("from __future__ import annotations\n\n", "", 1)
    marker = "\ndef notebook_source() -> str:\n"
    if marker not in source:
        raise RuntimeError("Could not locate notebook_source boundary")
    return source.split(marker, 1)[0].rstrip() + "\n"


__all__ = [
    "FUNKY_METRICS",
    "augment_projection_budget_metrics",
    "derive_projection_natural_metrics",
    "plot_degradation_pair",
    "plot_measurement_funky_summary",
    "plot_projection_budget_metrics",
    "plot_projection_natural_comparison",
    "prepare_measurement_funky_table",
]
