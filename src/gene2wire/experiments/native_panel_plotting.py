"""PDF notebook figures for measured assays with naturally incomplete panels.

Measured-entry evaluation and unassayed-entry extrapolation are always separate.
This module never fills unknown assay outcomes or pools different model settings.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .plotting import (_STYLE, _BENCHMARK_COLORS, _BENCHMARK_LABELS,
                       _benchmark_models, _save_display, _slug)


_STRUCTURES = ("Logistic", "MIRT", "Joint")
_METRICS = {
    "macro_auprc": "Measured-assay macro AUPRC ↑",
    "macro_log_loss": "Measured-assay macro log loss ↓",
    "macro_brier": "Measured-assay macro Brier ↓",
    "macro_auroc": "Measured-assay macro AUROC ↑",
}
_CORE_METRICS = ("macro_auprc", "macro_log_loss", "macro_brier")


def _warn(message):
    warnings.warn(message, UserWarning, stacklevel=3)


def _unique(frame, keys, label):
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate {label} rows for {keys}; keep distinct experiment settings separate")


def _required(frame, keys, label):
    if frame.empty:
        return False
    missing = set(keys).difference(frame.columns)
    if missing:
        _warn(f"Skipping {label}: missing columns {sorted(missing)}")
        return False
    return True


def _measured_matrix(panel):
    if not _required(panel, ("animal", "target", "measured"), "native panel availability"):
        return pd.DataFrame()
    _unique(panel, ["animal", "target"], "native panel")
    frame = panel.copy()
    # Do not use astype(bool): the CSV string "False" would become True.
    frame["measured"] = frame["measured"].map(
        lambda value: {"true": 1., "false": 0., "1": 1., "0": 0.,
                       "1.0": 1., "0.0": 0.}.get(str(value).lower(), np.nan))
    if frame["measured"].isna().any():
        _warn("Some native panel entries have unknown availability; they remain blank")
    animals = sorted(frame["animal"].dropna().unique(), key=str)
    targets = sorted(frame["target"].dropna().unique(), key=str)
    return frame.pivot(index="animal", columns="target", values="measured").reindex(
        index=animals, columns=targets)


def _availability_figure(matrix, title):
    figure, axis = plt.subplots(figsize=(max(9., .43 * len(matrix.columns)),
                                         2.2 + .42 * len(matrix)))
    cmap = ListedColormap(["#EAEAEA", "#3C789E"])
    cmap.set_bad("white")
    axis.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), cmap=cmap,
                vmin=0., vmax=1., aspect="auto", interpolation="nearest")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(matrix)), matrix.index)
    axis.set_xticks(np.arange(-.5, len(matrix.columns), 1.), minor=True)
    axis.set_yticks(np.arange(-.5, len(matrix), 1.), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_xlabel("Projection target")
    axis.set_ylabel("Animal")
    axis.set_title(f"{title}: native assay coverage", loc="left", pad=12)
    axis.legend(handles=[Patch(facecolor="#3C789E", label="Measured"),
                         Patch(facecolor="#EAEAEA", label="Not assayed")],
                loc="upper center", bbox_to_anchor=(.5, 1.27), ncol=2, frameon=False)
    figure.text(.5, .015, "Availability follows the experimental panel; observed zero does not mean unassayed.",
                ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0., .06, 1., .97))
    return figure


def _finite_metric(frame, metric):
    return metric in frame and np.isfinite(pd.to_numeric(frame[metric], errors="coerce")).any()


def _bar_axis(axis, frame, metric, models):
    values = pd.to_numeric(frame.set_index("model")[metric], errors="coerce").reindex(models)
    valid = np.isfinite(values.to_numpy(float))
    for index in np.flatnonzero(valid):
        model = models[index]
        value = float(values.iloc[index])
        axis.barh(index, value, color=_BENCHMARK_COLORS.get(model, "#999999"), height=.64)
        axis.annotate(f"{value:.4f}", (value, index), xytext=(4, 0),
                      textcoords="offset points", va="center", fontsize=8)
    axis.set_yticks(np.arange(len(models)), [_BENCHMARK_LABELS.get(m, m) for m in models])
    axis.set_ylim(len(models)-.5, -.5)
    high = float(values[valid].max()) if valid.any() else 0.
    axis.set_xlim(0., max(.05, high * 1.22))
    axis.set_title(_METRICS[metric], loc="left", pad=10)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#E5E5E5", linewidth=.6)
    axis.tick_params(axis="y", length=0)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))


def _measured_figure(frame, title, metrics):
    if not _required(frame, ("model",), "measured-assay comparison"):
        return None
    _unique(frame, ["model"], "measured aggregate")
    metrics = tuple(metric for metric in metrics if _finite_metric(frame, metric))
    if not metrics:
        _warn("Skipping measured-assay comparison: no finite requested metric")
        return None
    models = _benchmark_models(frame)
    if not models:
        return None
    figure, axes = plt.subplots(1, len(metrics), squeeze=False,
                               figsize=(4.9 * len(metrics), max(3.3, .36 * len(models) + 1.5)))
    for axis, metric in zip(axes[0], metrics):
        _bar_axis(axis, frame, metric, models)
    figure.suptitle(f"{title}: prediction on held-out measured entries", y=.985, fontsize=12)
    figure.text(.5, .01, "Means over folds and repetitions; unassayed targets do not enter these metrics.",
                ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0., .07, 1., .93), w_pad=2.)
    return figure


def _animal_figure(frame, title):
    if not _required(frame, ("animal", "model"), "per-animal measured comparison"):
        return None
    _unique(frame, ["animal", "model"], "per-animal aggregate")
    metrics = [metric for metric in ("macro_auprc", "macro_log_loss") if _finite_metric(frame, metric)]
    if not metrics:
        _warn("Skipping per-animal comparison: no finite AUPRC or log loss")
        return None
    animals = sorted(frame["animal"].dropna().unique(), key=str)
    models = _benchmark_models(frame)
    if not models or not animals:
        return None
    figure, axes = plt.subplots(1, len(metrics), squeeze=False,
                               figsize=(6. * len(metrics), 4.1 + .25 * ((len(models)-1)//4)))
    width = .8 / len(models)
    for axis, metric in zip(axes[0], metrics):
        for index, model in enumerate(models):
            selected = frame.loc[frame.model.eq(model)].set_index("animal")
            values = pd.to_numeric(selected[metric], errors="coerce").reindex(animals).to_numpy(float)
            x = np.arange(len(animals)) + (index - (len(models)-1)/2.) * width
            valid = np.isfinite(values)
            axis.bar(x[valid], values[valid], width=width * .95,
                     color=_BENCHMARK_COLORS.get(model, "#999999"),
                     label=_BENCHMARK_LABELS.get(model, model))
        axis.set_xticks(np.arange(len(animals)), animals)
        axis.set_ylim(bottom=0.)
        axis.set_title(_METRICS[metric], loc="left", pad=10)
        axis.set_xlabel("Animal")
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E5E5E5", linewidth=.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .92),
                  ncol=min(4, len(models)), frameon=False)
    figure.suptitle(f"{title}: within-animal held-out measured entries", y=.995, fontsize=12)
    figure.text(.5, .01, "Each animal is evaluated on its own measured target panel; target sets differ across animals.",
                ha="center", fontsize=9, color="#555555")
    legend_space = .13 + .045 * ((len(models)-1)//4)
    figure.tight_layout(rect=(0., .06, 1., 1.-legend_space))
    return figure


def _prediction_figure(frame, matrix, title):
    if not _required(frame, ("animal", "target", "model", "mean_prediction"),
                     "unassayed prediction heatmaps") or matrix.empty:
        return None
    _unique(frame, ["animal", "target", "model"], "unassayed prediction summary")
    if not set(frame["animal"]).issubset(matrix.index) or not set(frame["target"]).issubset(matrix.columns):
        raise ValueError("Unassayed prediction summary contains animals/targets absent from the native panel")
    models = [model for model in _STRUCTURES if frame["model"].eq(model).any()]
    data = {}
    availability = matrix.to_numpy(float)
    for model in models:
        selected = frame.loc[frame.model.eq(model)].copy()
        selected["mean_prediction"] = pd.to_numeric(selected["mean_prediction"], errors="coerce")
        values = selected.pivot(index="animal", columns="target", values="mean_prediction").reindex(
            index=matrix.index, columns=matrix.columns).to_numpy(float, copy=True)
        # Measured entries and unknown availability must never appear as off-panel predictions.
        values[availability != 0.] = np.nan
        finite = np.isfinite(values)
        if ((values[finite] < 0.) | (values[finite] > 1.)).any():
            raise ValueError("Predicted assay-positive scores must be probabilities in [0, 1]")
        if finite.any():
            data[model] = values
    if not data:
        _warn("Skipping unassayed heatmaps: no finite off-panel predictions for Logistic/MIRT/Joint")
        return None
    figure, axes = plt.subplots(len(data), 1, squeeze=False,
                                figsize=(max(10., .43*len(matrix.columns)), 2.1 * len(data) + 1.9))
    score_cmap = plt.get_cmap("viridis").copy()
    score_cmap.set_bad((1., 1., 1., 0.))
    coverage_cmap = ListedColormap(["white", "#CFCFCF"])
    coverage_cmap.set_bad("white")
    for row, ((model, values), axis) in enumerate(zip(data.items(), axes[:, 0])):
        axis.imshow(np.ma.masked_invalid(availability), cmap=coverage_cmap, vmin=0., vmax=1.,
                    aspect="auto", interpolation="nearest")
        plotted = axis.imshow(np.ma.masked_invalid(values), cmap=score_cmap, vmin=0., vmax=1.,
                              aspect="auto", interpolation="nearest")
        axis.set_yticks(np.arange(len(matrix)), matrix.index)
        axis.set_ylabel(model)
        axis.set_xticks(np.arange(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns if row == len(data)-1 else [""] * len(matrix.columns),
                             rotation=55, ha="right")
        axis.tick_params(axis="both", length=0)
    axes[-1, 0].set_xlabel("Projection target")
    figure.suptitle(f"{title}: unassayed-target predictions — unvalidated extrapolation", y=.985, fontsize=12)
    figure.text(.5, .045, "Gray: measured entries (excluded). White: no prediction or unknown availability.\n"
                "Colors show predicted assay-positive scores; no unassayed outcome is available for validation.",
                ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0., .14, .90, .95), h_pad=1.4)
    color_axis = figure.add_axes([.915, .28, .018, .54])
    figure.colorbar(plotted, cax=color_axis, label="Mean predicted assay-positive score")
    return figure


def plot_native_panel_results(artifacts, output_dir: str | Path, *, show: bool = True,
                              include_auroc: bool = False) -> dict[str, Path]:
    """Display and save coverage, measured evaluation and unvalidated predictions.

    Each artifact must contain one experiment configuration. Duplicate aggregates
    are rejected rather than silently averaged; nonfinite metrics remain absent.
    All exports are PDFs, and displayed figures are closed to bound notebook memory.
    """
    tables = artifacts.tables
    aggregate = tables.get("aggregate", pd.DataFrame())
    if not aggregate.empty and "mechanism" in aggregate and not aggregate["mechanism"].eq("assay_only").all():
        raise ValueError("Native panel figures require assay_only results, without synthetic thinning or paired references")
    names = aggregate["dataset"].dropna().unique() if "dataset" in aggregate else []
    if len(names) > 1:
        raise ValueError("Plot one native-panel dataset at a time")
    title = str(names[0]) if len(names) else "Natural target panels"
    destination = Path(output_dir)
    metrics = _CORE_METRICS + (("macro_auroc",) if include_auroc else ())
    paths = {}
    matrix = _measured_matrix(tables.get("native_panel", pd.DataFrame()))
    with plt.rc_context(_STYLE):
        factories = {
            "native_panel_coverage": lambda: _availability_figure(matrix, title) if not matrix.empty else None,
            "measured_assay_results": lambda: _measured_figure(aggregate, title, metrics),
            "per_animal_measured_results": lambda: _animal_figure(
                tables.get("per_animal_aggregate", pd.DataFrame()), title),
            "unassayed_predictions": lambda: _prediction_figure(
                tables.get("unassayed_summary", pd.DataFrame()), matrix, title),
        }
        for name, factory in factories.items():
            figure = factory()
            if figure is not None:
                path = destination / f"{_slug(title)}__{name}.pdf"
                _save_display(figure, path, show)
                paths[name] = path
    return paths
