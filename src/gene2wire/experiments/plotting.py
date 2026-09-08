"""Notebook-displayed paper figures with PDF-only exports.

All panels select the primary experiment before aggregation. Real-data curves
are descriptive means; simulation intervals, when supplied, use the generated
dataset as the independent unit. Calibration and mechanism controls never get
averaged into the main Technical-SAR curves.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


MODEL_ORDER = ("Logistic", "MIRT", "Joint", "PU", "PU-MIRT", "PU-Joint")
PU_MODELS = ("PU", "PU-MIRT", "PU-Joint")
SIMULATION_MODELS = ("Logistic", "PU", "PU-MIRT", "PU-Joint")
MODEL_LABELS = {"Logistic": "Logistic", "MIRT": "MIRT", "Joint": "Joint",
                "PU": "PU logistic", "PU-MIRT": "PU-MIRT", "PU-Joint": "PU-Joint"}
COLORS = {"Logistic": "#737373", "MIRT": "#67AA94", "Joint": "#E5A07B",
          "PU": "#0072B2", "PU-MIRT": "#009E73", "PU-Joint": "#D55E00"}
MARKERS = {"Logistic": "x", "MIRT": "s", "Joint": "^",
           "PU": "o", "PU-MIRT": "s", "PU-Joint": "^"}
PRIMARY_METRICS = ("macro_auprc", "macro_log_loss", "hidden_recall_at_h")
_GROUPS = ("dataset", "sharing_strength", "analysis", "mechanism", "loss_rate",
           "calibration_fraction", "calibration_spec", "model", "probability_semantics")
_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 10, "axes.linewidth": .7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "results"


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "analysis" in result:
        result = result.loc[result["analysis"].eq("primary")]
    if "calibration_spec" in result:
        result = result.loc[result["calibration_spec"].eq("correct")]
    if "mechanism" in result:
        result = result.loc[result["mechanism"].isin(["technical_sar", "natural"])]
    return result


def _aggregate(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "aggregate" in tables and not tables["aggregate"].empty:
        return _primary(tables["aggregate"])
    frame = _primary(tables.get("per_repetition", tables.get("metrics", pd.DataFrame())))
    if frame.empty:
        return frame
    groups = [key for key in _GROUPS if key in frame]
    numeric = [key for key in frame.select_dtypes(include="number")
               if key not in set(groups) | {"repetition", "outer_fold"}]
    if "repetition" in frame:
        # Fold means first, then repeat means; no biological CI is constructed.
        frame = frame.groupby(groups + ["repetition"], observed=True, dropna=False)[numeric].mean().reset_index()
    return frame.groupby(groups, observed=True, dropna=False)[numeric].mean().reset_index()


def _style_axis(axis, *, loss_axis: bool = True) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E6E6E6", linewidth=.6)
    axis.tick_params(length=3, width=.7)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if loss_axis:
        axis.set_xlim(-.025, .825)
        axis.set_xticks([0, .2, .4, .6, .8])
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.set_xlabel("Positive-label loss")


def _metric_title(metric: str, *, simulation: bool = False) -> str:
    return {
        "macro_auprc": "Macro AUPRC ↑",
        "macro_log_loss": ("True-projection log loss ↓" if simulation else "Reference log loss ↓"),
        "hidden_recall_at_h": "Hidden Recall@$H$ ↑",
        "macro_predicted_prevalence": "Mean target prevalence",
    }[metric]


def _curve(axis, frame: pd.DataFrame, metric: str, models: tuple[str, ...],
           intervals: pd.DataFrame | None = None) -> bool:
    any_data = False
    for model in models:
        if metric not in frame or "loss_rate" not in frame:
            continue
        selected = frame.loc[frame["model"].eq(model), ["loss_rate", metric]].copy()
        selected["loss_rate"] = pd.to_numeric(selected["loss_rate"], errors="coerce")
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        selected = selected.replace([np.inf, -np.inf], np.nan).dropna()
        if metric.startswith("hidden_"):
            selected = selected.loc[selected["loss_rate"] > 0]
        selected = selected.groupby("loss_rate", as_index=False, observed=True)[metric].mean().sort_values("loss_rate")
        if selected.empty:
            continue
        x, y = selected["loss_rate"].to_numpy(float), selected[metric].to_numpy(float)
        axis.plot(x, y, color=COLORS[model], marker=MARKERS[model], markersize=4.5,
                  markeredgewidth=.7, linewidth=1.8,
                  linestyle="-" if model.startswith("PU") else "--",
                  label=MODEL_LABELS[model], zorder=3)
        any_data = True
        if intervals is not None and not intervals.empty:
            required = {"model", "metric", "loss_rate", "ci95_low", "ci95_high", "n"}
            if required.issubset(intervals.columns):
                band = intervals.loc[intervals["model"].eq(model) & intervals["metric"].eq(metric)].copy()
                band = band.loc[pd.to_numeric(band["n"], errors="coerce") >= 2]
                if metric.startswith("hidden_"):
                    band = band.loc[pd.to_numeric(band["loss_rate"], errors="coerce") > 0]
                band = band.replace([np.inf, -np.inf], np.nan).dropna(subset=["loss_rate", "ci95_low", "ci95_high"])
                if band["loss_rate"].duplicated().any():
                    raise ValueError("Duplicate primary simulation interval rows: distinguish scientific settings before plotting")
                band = band.sort_values("loss_rate")
                if not band.empty:
                    axis.fill_between(band["loss_rate"].to_numpy(float), band["ci95_low"].to_numpy(float),
                                      band["ci95_high"].to_numpy(float), color=COLORS[model],
                                      alpha=.12, linewidth=0, zorder=1)
    if not any_data:
        axis.text(.5, .5, "No defined estimates", transform=axis.transAxes,
                  ha="center", va="center", color="#777777", fontsize=9)
    _style_axis(axis)
    return any_data


def _legend(figure, axes, models: tuple[str, ...]) -> None:
    found = {}
    for axis in np.asarray(axes, dtype=object).ravel():
        handles, labels = axis.get_legend_handles_labels()
        found.update(zip(labels, handles))
    order = [MODEL_LABELS[model] for model in models if MODEL_LABELS[model] in found]
    if order:
        figure.legend([found[label] for label in order], order, frameon=False,
                      loc="upper center", bbox_to_anchor=(.5, .995),
                      ncol=min(len(order), 6), handlelength=2.1, columnspacing=1.5)


def _save_display(figure, destination: Path, show: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="pdf", bbox_inches="tight", pad_inches=.08,
                   metadata={"Creator": "Gene2Wire shared 0908 plotting"})
    if show:
        plt.show()
    # A displayed notebook figure remains in the output; closing prevents large
    # multi-dataset runs from retaining every matplotlib canvas in memory.
    plt.close(figure)


def _simulation_main(frame, intervals, destination, *, show, models=SIMULATION_MODELS):
    rhos = sorted(pd.to_numeric(frame["sharing_strength"], errors="coerce").dropna().unique())
    if not rhos:
        return
    figure, axes = plt.subplots(len(rhos), 3, squeeze=False,
                                figsize=(11.6, 2.5 * len(rhos) + .5))
    for row, rho in enumerate(rhos):
        selected = frame.loc[np.isclose(pd.to_numeric(frame["sharing_strength"], errors="coerce"), rho)]
        bands = intervals
        if bands is not None and not bands.empty and "sharing_strength" in bands:
            bands = bands.loc[np.isclose(pd.to_numeric(bands["sharing_strength"], errors="coerce"), rho)]
        for column, metric in enumerate(PRIMARY_METRICS):
            axis = axes[row, column]
            _curve(axis, selected, metric, models, bands)
            axis.set_title(f"{chr(65 + row * 3 + column)}   {_metric_title(metric, simulation=True)}", loc="left", pad=9)
            if column == 0:
                axis.set_ylabel(rf"Sharing $\rho={rho:g}$")
            if row != len(rhos) - 1:
                axis.set_xlabel("")
    _legend(figure, axes, models)
    figure.tight_layout(rect=(0, 0, 1, .945), h_pad=1.8, w_pad=2.)
    _save_display(figure, destination, show)


def _real_curves(frame, destination, *, show, models=PU_MODELS):
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
    for column, metric in enumerate(PRIMARY_METRICS):
        _curve(axes[column], frame, metric, models)
        axes[column].set_title(f"{chr(65 + column)}   {_metric_title(metric)}", loc="left", pad=9)
    _legend(figure, axes, models)
    figure.tight_layout(rect=(0, 0, 1, .88), w_pad=2.)
    _save_display(figure, destination, show)


def _dot_panel(axis, frame, metric, *, models=MODEL_ORDER):
    present = [model for model in models if model in set(frame["model"])]
    axis.set_yticks(np.arange(len(present)), [MODEL_LABELS[model] for model in present])
    any_data = False
    for index, model in enumerate(present):
        if metric not in frame:
            continue
        values = pd.to_numeric(frame.loc[frame["model"].eq(model), metric], errors="coerce")
        value = float(values.mean()) if values.notna().any() else np.nan
        if np.isfinite(value):
            axis.scatter(value, index, color=COLORS[model], marker=MARKERS[model],
                         s=43, linewidths=.9, zorder=3)
            any_data = True
    axis.invert_yaxis()
    _style_axis(axis, loss_axis=False)
    # These are categorical y positions; do not replace them with a numeric locator.
    axis.set_yticks(np.arange(len(present)), [MODEL_LABELS[model] for model in present])
    axis.grid(False, axis="y")
    axis.grid(axis="x", color="#E6E6E6", linewidth=.6)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if not any_data:
        axis.text(.5, .5, "No defined estimates", transform=axis.transAxes,
                  ha="center", va="center", color="#777777", fontsize=9)


def _paired_audit(axis, audit):
    required = {"target", "standard_positive_count", "reference_positive_count"}
    if not required.issubset(audit.columns):
        raise ValueError(f"paired_audit requires columns {sorted(required)}")
    audit = audit.copy()
    standard = pd.to_numeric(audit["standard_positive_count"], errors="raise")
    reference = pd.to_numeric(audit["reference_positive_count"], errors="raise")
    if (standard.lt(0).any() or reference.lt(0).any() or standard.gt(reference).any()
        or standard.isna().any() or reference.isna().any()):
        raise ValueError("Paired-audit counts must satisfy 0 <= standard <= reference")
    audit["relative_detection"] = np.divide(standard, reference,
        out=np.full(len(audit), np.nan), where=reference.to_numpy() > 0)
    audit = audit.sort_values("relative_detection", na_position="last")
    for index, row in enumerate(audit.itertuples(index=False)):
        value = row.relative_detection
        if np.isfinite(value):
            axis.hlines(index, 0, value, color="#BEBEBE", linewidth=1.4)
            axis.scatter(value, index, color=COLORS["PU"], s=30, zorder=3)
            axis.annotate(f"{int(row.standard_positive_count):,}/{int(row.reference_positive_count):,}",
                          (value, index), xytext=(7, 0), textcoords="offset points",
                          va="center", fontsize=8, color="#555555")
    pooled = float(standard.sum() / reference.sum()) if reference.sum() > 0 else np.nan
    if np.isfinite(pooled):
        axis.axvline(pooled, color="#555555", linestyle="--", linewidth=1)
        axis.text(.98, .97, f"Pooled = {pooled:.1%}", transform=axis.transAxes,
                  ha="right", va="top", fontsize=9)
    axis.set_yticks(np.arange(len(audit)), audit["target"].astype(str))
    axis.invert_yaxis()
    axis.set_xlim(0, 1.13)
    axis.set_xticks([0, .25, .5, .75, 1.])
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.set_xlabel("Standard positives / union-reference positives")
    axis.grid(axis="x", color="#E6E6E6", linewidth=.6)
    axis.set_axisbelow(True)


def _natural_main(frame, audit, destination, *, show):
    if audit is not None and not audit.empty:
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        _paired_audit(axes[0, 0], audit)
        axes[0, 0].set_title("A   Paired-label audit", loc="left", pad=9)
        for axis, metric, letter in zip((axes[0, 1], axes[1, 0], axes[1, 1]),
                                       ("macro_log_loss", "macro_auprc", "hidden_recall_at_h"), "BCD"):
            _dot_panel(axis, frame, metric)
            axis.set_xlabel(_metric_title(metric))
            axis.set_title(f"{letter}   Natural paired evaluation", loc="left", pad=9)
        figure.tight_layout(h_pad=2.2, w_pad=2.8)
    else:
        figure, axes = plt.subplots(1, 3, figsize=(11.6, 3.8))
        for axis, metric, letter in zip(axes, PRIMARY_METRICS, "ABC"):
            _dot_panel(axis, frame, metric)
            axis.set_xlabel(_metric_title(metric))
            axis.set_title(f"{letter}   Natural paired evaluation", loc="left", pad=9)
        figure.tight_layout(w_pad=2.5)
    _save_display(figure, destination, show)


def _probability_dashboard(frame, destination, *, show, simulation=False, natural=False):
    rhos = (sorted(pd.to_numeric(frame["sharing_strength"], errors="coerce").dropna().unique())
            if simulation else [None])
    figure, axes = plt.subplots(len(rhos), 2, squeeze=False,
                                figsize=(9.0, 2.6 * len(rhos) + .65))
    for row, rho in enumerate(rhos):
        selected = (frame.loc[np.isclose(pd.to_numeric(frame["sharing_strength"], errors="coerce"), rho)]
                    if rho is not None else frame)
        for column, metric in enumerate(("macro_log_loss", "macro_predicted_prevalence")):
            axis = axes[row, column]
            if natural:
                _dot_panel(axis, selected, metric)
                axis.set_xlabel(_metric_title(metric))
            else:
                _curve(axis, selected, metric, MODEL_ORDER)
            axis.set_title(f"{chr(65 + row * 2 + column)}   {_metric_title(metric, simulation=simulation)}", loc="left", pad=9)
            if rho is not None and column == 0:
                axis.set_ylabel(rf"Sharing $\rho={rho:g}$")
            if metric == "macro_predicted_prevalence" and "macro_reference_prevalence" in selected:
                if natural:
                    value = pd.to_numeric(selected["macro_reference_prevalence"], errors="coerce").mean()
                    if np.isfinite(value):
                        axis.axvline(value, color="#222222", linestyle=":", linewidth=1.3)
                        axis.text(.98, .97, "Dotted: reference", transform=axis.transAxes,
                                  ha="right", va="top", fontsize=8, color="#555555")
                else:
                    truth = selected[["loss_rate", "macro_reference_prevalence"]].dropna().groupby("loss_rate", as_index=False).mean()
                    if not truth.empty:
                        axis.plot(truth["loss_rate"], truth["macro_reference_prevalence"],
                                  linestyle=":", linewidth=1.5, color="#222222", zorder=4)
                        axis.text(.98, .97, "Dotted: reference", transform=axis.transAxes,
                                  ha="right", va="top", fontsize=8, color="#555555")
    if not natural:
        _legend(figure, axes, MODEL_ORDER)
    figure.tight_layout(rect=(0, 0, 1, .91 if len(rhos) == 1 else .95), h_pad=1.8, w_pad=2.8)
    _save_display(figure, destination, show)


def plot_results(artifacts: Any, output_dir: str | Path, *, show: bool = True,
                 include_all_models: bool = False) -> dict[str, Path]:
    """Display notebook figures and export PDF files, returning their paths.

    ``artifacts`` provides ``tables`` and ``manifest`` attributes, as returned
    by the shared pipeline. Empty result tables produce an empty mapping. No
    artificial loss-rate axis is fabricated for naturally paired datasets.
    """
    frame = _aggregate(artifacts.tables)
    if frame.empty:
        return {}
    if "model" not in frame:
        raise ValueError("Result tables must identify the fitted model")
    manifest = getattr(artifacts, "manifest", {}) or {}
    simulation = (manifest.get("uncertainty_unit") == "generated_dataset" or
                  ("sharing_strength" in frame and pd.to_numeric(frame["sharing_strength"], errors="coerce").notna().any()))
    directory, paths = Path(output_dir), {}
    intervals = _primary(artifacts.tables.get("simulation_intervals", pd.DataFrame()))
    audit = artifacts.tables.get("paired_audit", pd.DataFrame())
    with plt.rc_context(_STYLE):
        if simulation:
            main = directory / "simulation_results_0908.pdf"
            _simulation_main(frame, intervals, main, show=show)
            paths["simulation_primary"] = main
            probability = directory / "simulation_probability_0908.pdf"
            _probability_dashboard(frame, probability, show=show, simulation=True)
            paths["simulation_probability"] = probability
            if include_all_models:
                all_models = directory / "simulation_all_models_0908.pdf"
                _simulation_main(frame, intervals, all_models, show=show, models=MODEL_ORDER)
                paths["simulation_all_models"] = all_models
        else:
            datasets = list(frame["dataset"].drop_duplicates()) if "dataset" in frame else ["real_data"]
            for dataset in datasets:
                selected = frame.loc[frame["dataset"].eq(dataset)] if "dataset" in frame else frame
                natural = ("mechanism" in selected and selected["mechanism"].eq("natural").all()) or (
                    "loss_rate" not in selected or pd.to_numeric(selected["loss_rate"], errors="coerce").isna().all())
                stem = _slug(dataset)
                main = directory / f"{stem}_results_0908.pdf"
                if natural:
                    dataset_audit = audit.loc[audit["dataset"].eq(dataset)] if not audit.empty and "dataset" in audit else audit
                    _natural_main(selected, dataset_audit, main, show=show)
                else:
                    _real_curves(selected, main, show=show)
                paths[f"{stem}_primary"] = main
                probability = directory / f"{stem}_probability_0908.pdf"
                _probability_dashboard(selected, probability, show=show, natural=natural)
                paths[f"{stem}_probability"] = probability
                if include_all_models and not natural:
                    all_models = directory / f"{stem}_all_models_0908.pdf"
                    _real_curves(selected, all_models, show=show, models=MODEL_ORDER)
                    paths[f"{stem}_all_models"] = all_models
    return paths
