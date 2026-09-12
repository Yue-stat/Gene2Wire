"""Plots for the combined measurement-degradation experiments.

The functions in this module consume tidy data frames rather than experiment
artifact objects.  This keeps plotting independent of the runner and makes the
scientific contrasts explicit at the call site.  Replicate rows are averaged
within the coordinates displayed by a figure; callers must subset any other
scientific axes before plotting.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


KEY_MODELS = (
    "PU",
    "PU-MIRT",
    "PU-Joint",
    "GenEML-adapted",
    "Inductive-PU-MC",
    "SAR-PU",
)

FULL_MODEL_ORDER = (
    "Logistic",
    "MIRT",
    "Joint",
    "PU",
    "PU-MIRT",
    "PU-Joint",
    "Reference-only",
    "Reference+PU",
    "Reference+PU-MIRT",
    "Reference+PU-Joint",
    "RF-observed",
    "RF-reference",
    "RF-mixed",
    "Qiao-ID-squared",
    "Qiao-ID-logit",
    "Qiao-squared",
    "Qiao-logit",
    "GenEML-adapted",
    "Inductive-PU-MC",
    "SAR-PU",
)

MODEL_COLORS = {
    "Logistic": "#737373",
    "MIRT": "#67AA94",
    "Joint": "#E5A07B",
    "PU": "#0072B2",
    "PU-MIRT": "#009E73",
    "PU-Joint": "#D55E00",
    "GenEML-adapted": "#CC79A7",
    "Inductive-PU-MC": "#56B4E9",
    "SAR-PU": "#E69F00",
}


def _table(value: pd.DataFrame, required: Sequence[str], *, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = set(required).difference(value.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if value.empty:
        raise ValueError(f"{name} must contain at least one row")
    return value.copy()


def _labels(frame: pd.DataFrame, column: str, *, name: str) -> pd.Series:
    values = frame[column]
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name}.{column} must contain nonempty labels")
    return values.astype(str)


def _numeric(frame: pd.DataFrame, column: str, *, name: str,
             probability: bool = False) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}.{column} must contain finite numeric values")
    if probability and not values.between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{name}.{column} must lie in [0, 1]")
    return values.astype(float)


def _figure_axis(ax: Axes | None, *, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    if ax is None:
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        return figure, axis
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes")
    return ax.figure, ax


def _ordered_models(models: Sequence[str]) -> list[str]:
    present = list(dict.fromkeys(map(str, models)))
    priority = {model: index for index, model in enumerate(FULL_MODEL_ORDER)}
    return sorted(present, key=lambda model: (priority.get(model, len(priority)), model))


def _model_color(model: str, index: int) -> object:
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    palette = plt.get_cmap("tab20")
    return palette(index % palette.N)


def plot_retention_auprc(
    table: pd.DataFrame,
    *,
    mode: str = "key",
    key_models: Sequence[str] = KEY_MODELS,
    retention_col: str = "retention",
    model_col: str = "model",
    metric_col: str = "macro_auprc",
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot mean macro-AUPRC against positive-label retention.

    ``mode='key'`` shows the predeclared PU/structured/new-comparator subset;
    ``mode='full'`` shows every model present in ``table``.  Multiple rows at
    the same model/retention coordinate are treated as repetitions and averaged.
    """
    name = "retention table"
    frame = _table(table, (retention_col, model_col, metric_col), name=name)
    frame[retention_col] = _numeric(
        frame, retention_col, name=name, probability=True)
    frame[metric_col] = _numeric(frame, metric_col, name=name, probability=True)
    frame[model_col] = _labels(frame, model_col, name=name)
    if mode not in {"key", "full"}:
        raise ValueError("mode must be 'key' or 'full'")
    if mode == "key":
        requested = tuple(map(str, key_models))
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("key_models must contain distinct model names")
        frame = frame.loc[frame[model_col].isin(requested)].copy()
        models = [model for model in requested if model in set(frame[model_col])]
        if not models:
            raise ValueError("retention table contains none of the requested key models")
    else:
        models = _ordered_models(frame[model_col])

    summary = (frame.groupby([model_col, retention_col], observed=True, sort=False)[metric_col]
               .mean().reset_index())
    figure, axis = _figure_axis(ax, figsize=(7.0, 4.3))
    for index, model in enumerate(models):
        selected = summary.loc[summary[model_col].eq(model)].sort_values(retention_col)
        if selected.empty:
            continue
        axis.plot(
            selected[retention_col], selected[metric_col], marker="o",
            linewidth=1.8, markersize=4.5, label=model,
            color=_model_color(model, index),
        )
    axis.set_xlabel("Positive-label retention")
    axis.set_ylabel("Macro AUPRC ↑")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2 if len(models) > 6 else 1)
    axis.set_title(title or "Positive-retention degradation")
    return figure, axis


def plot_gene_target_brier_heatmap(
    table: pd.DataFrame,
    *,
    baseline_model: str = "PU",
    comparison_model: str = "PU-MIRT",
    gene_coverage_col: str = "gene_coverage",
    target_coverage_col: str = "target_coverage",
    model_col: str = "model",
    metric_col: str = "macro_brier",
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "RdBu",
) -> tuple[Figure, Axes]:
    """Plot fixed-baseline Brier minus PU-MIRT Brier on the coverage grid.

    The baseline name is supplied once for the whole grid; this function never
    chooses a different comparator cell by cell.  Positive values favor the
    comparison model.  Missing design cells remain visible as ``NA``.
    """
    name = "heatmap table"
    required = (gene_coverage_col, target_coverage_col, model_col, metric_col)
    frame = _table(table, required, name=name)
    frame[gene_coverage_col] = _numeric(
        frame, gene_coverage_col, name=name, probability=True)
    frame[target_coverage_col] = _numeric(
        frame, target_coverage_col, name=name, probability=True)
    frame[metric_col] = _numeric(frame, metric_col, name=name, probability=True)
    frame[model_col] = _labels(frame, model_col, name=name)
    if not str(baseline_model).strip() or not str(comparison_model).strip():
        raise ValueError("baseline_model and comparison_model must be nonempty")
    if baseline_model == comparison_model:
        raise ValueError("baseline_model and comparison_model must differ")
    available = set(frame[model_col])
    absent = {baseline_model, comparison_model}.difference(available)
    if absent:
        raise ValueError(f"heatmap table is missing requested models: {sorted(absent)}")

    selected = frame.loc[frame[model_col].isin((baseline_model, comparison_model))]
    summary = (selected.groupby(
        [gene_coverage_col, target_coverage_col, model_col], observed=True
    )[metric_col].mean().unstack(model_col))
    contrast = summary.get(baseline_model) - summary.get(comparison_model)
    genes = np.sort(frame[gene_coverage_col].unique())
    targets = np.sort(frame[target_coverage_col].unique())
    grid = contrast.unstack(gene_coverage_col).reindex(index=targets, columns=genes)
    values = grid.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("heatmap table has no cell containing both requested models")
    limit = max(float(np.max(np.abs(finite))), np.finfo(float).eps)

    figure, axis = _figure_axis(ax, figsize=(6.8, 4.8))
    colormap = plt.get_cmap(cmap).with_extremes(bad="#E5E5E5")
    image = axis.imshow(
        np.ma.masked_invalid(values), origin="lower", aspect="auto",
        cmap=colormap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(np.arange(len(genes)), [f"{100 * value:g}%" for value in genes])
    axis.set_yticks(np.arange(len(targets)), [f"{100 * value:g}%" for value in targets])
    axis.set_xlabel("Gene coverage")
    axis.set_ylabel("Target coverage")
    axis.set_title(title or f"{baseline_model} − {comparison_model} macro Brier")
    for row in range(len(targets)):
        for column in range(len(genes)):
            value = values[row, column]
            text = "NA" if not np.isfinite(value) else f"{value:+.3f}"
            color = "white" if np.isfinite(value) and abs(value) > 0.55 * limit else "black"
            axis.text(column, row, text, ha="center", va="center", color=color, fontsize=9)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label(
        f"{baseline_model} Brier − {comparison_model} Brier\n"
        f"positive favors {comparison_model}"
    )
    return figure, axis


def plot_accuracy_panel_sensitivity(
    table: pd.DataFrame,
    *,
    model_col: str = "model",
    accuracy_col: str = "macro_auprc",
    sensitivity_col: str = "panel_sensitivity",
    models: Sequence[str] | None = None,
    annotate: bool = True,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot each model's mean accuracy against cross-panel sensitivity."""
    name = "accuracy-sensitivity table"
    frame = _table(table, (model_col, accuracy_col, sensitivity_col), name=name)
    frame[model_col] = _labels(frame, model_col, name=name)
    frame[accuracy_col] = _numeric(frame, accuracy_col, name=name, probability=True)
    frame[sensitivity_col] = _numeric(frame, sensitivity_col, name=name)
    if models is not None:
        requested = tuple(map(str, models))
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("models must contain distinct model names")
        frame = frame.loc[frame[model_col].isin(requested)].copy()
        if frame.empty:
            raise ValueError("accuracy-sensitivity table contains none of the requested models")
        order = [model for model in requested if model in set(frame[model_col])]
    else:
        order = _ordered_models(frame[model_col])
    summary = frame.groupby(model_col, observed=True)[
        [accuracy_col, sensitivity_col]].mean().reindex(order)

    figure, axis = _figure_axis(ax, figsize=(6.5, 4.7))
    for index, (model, row) in enumerate(summary.iterrows()):
        axis.scatter(
            row[sensitivity_col], row[accuracy_col], s=56,
            color=_model_color(model, index), label=model, zorder=3,
        )
        if annotate:
            axis.annotate(model, (row[sensitivity_col], row[accuracy_col]),
                          xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Cross-panel prediction sensitivity")
    axis.set_ylabel("Macro AUPRC ↑")
    axis.set_ylim(bottom=0.0)
    axis.grid(color="#E6E6E6", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    if not annotate:
        axis.legend(frameon=False)
    axis.set_title(title or "Accuracy versus panel sensitivity")
    return figure, axis


def plot_projection_tags_budget_recall(
    table: pd.DataFrame,
    *,
    budget_col: str = "budget_fraction",
    model_col: str = "model",
    recall_col: str = "amplification_confirmed_recall",
    models: Sequence[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot amplification-confirmed positive recall at ranked-cell budgets."""
    name = "Projection-TAGs budget-recall table"
    frame = _table(table, (budget_col, model_col, recall_col), name=name)
    frame[budget_col] = _numeric(frame, budget_col, name=name, probability=True)
    frame[recall_col] = _numeric(frame, recall_col, name=name, probability=True)
    frame[model_col] = _labels(frame, model_col, name=name)
    if models is not None:
        requested = tuple(map(str, models))
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("models must contain distinct model names")
        frame = frame.loc[frame[model_col].isin(requested)].copy()
        if frame.empty:
            raise ValueError("budget-recall table contains none of the requested models")
        order = [model for model in requested if model in set(frame[model_col])]
    else:
        order = _ordered_models(frame[model_col])
    summary = (frame.groupby([model_col, budget_col], observed=True, sort=False)[recall_col]
               .mean().reset_index())

    figure, axis = _figure_axis(ax, figsize=(7.0, 4.3))
    for index, model in enumerate(order):
        selected = summary.loc[summary[model_col].eq(model)].sort_values(budget_col)
        axis.plot(
            selected[budget_col], selected[recall_col], marker="o",
            linewidth=1.8, markersize=4.5, label=model,
            color=_model_color(model, index),
        )
    axis.set_xlabel("Top-ranked prediction budget")
    axis.set_ylabel("Amplification-confirmed positive recall ↑")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.0)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2 if len(order) > 6 else 1)
    axis.set_title(title or "Projection-TAGs hidden-positive recovery")
    return figure, axis


# Longer aliases keep notebook calls readable while retaining compact public names.
plot_retention_auprc_curve = plot_retention_auprc
plot_accuracy_panel_sensitivity_scatter = plot_accuracy_panel_sensitivity
plot_projection_tags_budget_recall_curve = plot_projection_tags_budget_recall


__all__ = [
    "KEY_MODELS",
    "FULL_MODEL_ORDER",
    "plot_retention_auprc",
    "plot_retention_auprc_curve",
    "plot_gene_target_brier_heatmap",
    "plot_accuracy_panel_sensitivity",
    "plot_accuracy_panel_sensitivity_scatter",
    "plot_projection_tags_budget_recall",
    "plot_projection_tags_budget_recall_curve",
]
