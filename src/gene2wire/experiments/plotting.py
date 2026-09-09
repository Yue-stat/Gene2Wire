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


# Diagnostic figures deliberately discover models from the exported results.
# Keep MODEL_ORDER above fixed: changing it would alter the existing paper plots.
_BENCHMARK_ORDER = MODEL_ORDER + (
    "Reference-only", "Reference+PU", "Logistic-rescaled",
    "RF-observed", "RF-reference", "Prevalence-observed", "Prevalence-reference",
    "Qiao-squared", "Qiao-logit",
)
_BENCHMARK_LABELS = {**MODEL_LABELS,
    "Reference-only": "Reference-only logistic",
    "Reference+PU": "Reference + PU logistic",
    "Logistic-rescaled": "Logistic + sensitivity rescaling",
    "RF-observed": "RF (observed labels)",
    "RF-reference": "RF (paired references)",
    "Prevalence-observed": "Prevalence (observed labels)",
    "Prevalence-reference": "Prevalence (paired references)",
    "Qiao-squared": "Qiao bilinear (squared error)",
    "Qiao-logit": "Qiao bilinear (logistic)",
}
_BENCHMARK_COLORS = {**COLORS,
    "Reference-only": "#9467BD", "Reference+PU": "#332288",
    "Logistic-rescaled": "#CC79A7", "RF-observed": "#AA4499",
    "RF-reference": "#882255", "Prevalence-observed": "#AA8833",
    "Prevalence-reference": "#665522", "Qiao-squared": "#44AA99",
    "Qiao-logit": "#117733",
}
_BENCHMARK_MARKERS = {**MARKERS,
    "Reference-only": "D", "Reference+PU": "P", "Logistic-rescaled": "*",
    "RF-observed": "v", "RF-reference": "D", "Prevalence-observed": "X",
    "Prevalence-reference": "P", "Qiao-squared": "<", "Qiao-logit": ">",
}
_BENCHMARK_SCOPES = ("dataset", "sharing_strength", "analysis", "mechanism",
                     "calibration_fraction", "calibration_spec")


def _benchmark_aggregate(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Keep every analysis, averaging folds within each repetition first."""
    for source in ("aggregate", "per_repetition", "metrics"):
        frame = tables.get(source, pd.DataFrame())
        if not frame.empty:
            frame = frame.copy()
            break
    else:
        return pd.DataFrame()
    if "model" not in frame:
        raise ValueError("Benchmark result tables must identify the fitted model")
    if source == "aggregate":
        return frame
    groups = [key for key in _GROUPS if key in frame]
    numeric = [key for key in frame.select_dtypes(include="number")
               if key not in set(groups) | {"repetition", "outer_fold"}]
    if "repetition" in frame:
        frame = frame.groupby(groups + ["repetition"], observed=True, dropna=False)[numeric].mean().reset_index()
    return frame.groupby(groups, observed=True, dropna=False)[numeric].mean().reset_index()


def _benchmark_models(frame: pd.DataFrame) -> tuple[str, ...]:
    available = set(frame["model"].dropna().astype(str))
    return tuple(model for model in _BENCHMARK_ORDER if model in available) + tuple(
        sorted(available.difference(_BENCHMARK_ORDER)))


def _benchmark_style(model: str, models: tuple[str, ...]) -> dict[str, Any]:
    # Unknown future baselines remain visible and receive deterministic styles.
    index = models.index(model)
    return {"color": _BENCHMARK_COLORS.get(model, plt.get_cmap("tab20")(index % 20)),
            "marker": _BENCHMARK_MARKERS.get(model, ("o", "s", "D", "v", "P")[index % 5]),
            "label": _BENCHMARK_LABELS.get(model, model),
            "linewidth": 2.4 if model == "PU-Joint" else 1.3,
            "markersize": 6 if model == "PU-Joint" else 4.8,
            "zorder": 5 if model == "PU-Joint" else 3}


def _benchmark_metric_title(metric: str, *, simulation: bool) -> str:
    extra = {"macro_auroc": "Macro AUROC ↑", "macro_brier": "Brier score ↓",
             "macro_ece": "Expected calibration error ↓",
             "macro_prevalence_absolute_error": "Prevalence absolute error ↓"}
    if metric in PRIMARY_METRICS or metric == "macro_predicted_prevalence":
        return _metric_title(metric, simulation=simulation)
    return extra.get(metric, metric.replace("_", " ").capitalize())


def _benchmark_curve(axis, frame, metric, models):
    rates = np.sort(pd.to_numeric(frame["loss_rate"], errors="coerce").dropna().unique())
    if metric.startswith("hidden_"):
        rates = rates[rates > 0]
    any_data = False
    for model in models:
        style = _benchmark_style(model, models)
        selected = frame.loc[frame["model"].eq(model), ["loss_rate", metric]].copy()
        selected["loss_rate"] = pd.to_numeric(selected["loss_rate"], errors="coerce")
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        # Reindexing inserts NaNs at missing rates and breaks any interpolated
        # segment. In particular, 0%/80%-only RF fits are disconnected points.
        values = selected.set_index("loss_rate")[metric].reindex(rates).to_numpy(float)
        # Pandas copy-on-write can expose a read-only NumPy view. A functional
        # replacement handles both writable arrays and immutable views without
        # modifying the caller's table or relying on pandas' copy policy.
        values = np.where(np.isfinite(values), values, np.nan)
        count = int(np.isfinite(values).sum())
        if not count:
            continue
        linestyle = "-" if model.startswith("PU") else "--"
        if count < 3:
            linestyle = "None"
        axis.plot(rates, values, **style, linestyle=linestyle, markeredgewidth=.7)
        any_data = True
    _style_axis(axis)
    if not any_data:
        axis.text(.5, .5, "No defined estimates", transform=axis.transAxes,
                  ha="center", va="center", color="#777777", fontsize=9)


def _benchmark_dots(axis, frame, metric, models):
    any_data = False
    for index, model in enumerate(models):
        values = pd.to_numeric(frame.loc[frame["model"].eq(model), metric], errors="coerce")
        if len(values) != 1:
            raise ValueError("Benchmark dot panels require one estimate per model and scientific setting")
        value = float(values.iloc[0])
        if np.isfinite(value):
            style = _benchmark_style(model, models)
            axis.plot(value, index, **style, linestyle="None", markeredgewidth=.7)
            any_data = True
    _style_axis(axis, loss_axis=False)
    axis.set_yticks(np.arange(len(models)), [_BENCHMARK_LABELS.get(model, model) for model in models])
    axis.invert_yaxis()
    axis.set_ylim(len(models) - .5, -.5)
    axis.grid(False, axis="y")
    axis.grid(axis="x", color="#E6E6E6", linewidth=.6)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if not any_data:
        axis.text(.5, .5, "No defined estimates", transform=axis.transAxes,
                  ha="center", va="center", color="#777777", fontsize=9)


def plot_benchmark_results(artifacts: Any, output_dir: str | Path, *,
                           show: bool = True,
                           metrics: tuple[str, ...] = PRIMARY_METRICS) -> dict[str, Path]:
    """Add diagnostic PDFs comparing PU-Joint with every available model.

    Call this *after* :func:`plot_results` to retain existing paper figures.
    Output lives under ``output_dir/all_benchmarks``. A separate figure is
    drawn for each dataset, sharing strength, analysis, mechanism, paired
    fraction and calibration specification: no controls are averaged into
    primary estimates. Three or more measured loss rates produce curves;
    endpoint-only models have disconnected markers. Natural and single-rate
    comparisons use categorical dot panels. All exported model names are
    discovered dynamically, so optional and future baselines are included.

    Diagnostic means follow the same fold-then-repetition aggregation as the
    main plots. They carry no confidence intervals; formal simulation intervals
    remain in the preserved main plots and exported tables.
    """
    frame = _benchmark_aggregate(artifacts.tables)
    if frame.empty:
        return {}
    missing = set(metrics).difference(frame.columns)
    if missing:
        raise ValueError(f"Benchmark metrics are missing from the results: {sorted(missing)}")
    if not metrics:
        raise ValueError("Select at least one benchmark metric")
    scope_columns = [key for key in _BENCHMARK_SCOPES if key in frame]
    grouping = (frame.groupby(scope_columns, observed=True, dropna=False, sort=True)
                if scope_columns else [((), frame)])
    directory = Path(output_dir) / "all_benchmarks"
    paths = {}
    with plt.rc_context(_STYLE):
        for scope_key, selected in grouping:
            key = scope_key if isinstance(scope_key, tuple) else (scope_key,)
            scope = dict(zip(scope_columns, key))
            # Loss rate is the only scientific coordinate allowed to vary in
            # a figure. A duplicated model/rate often means that outputs from
            # distinct information budgets or probability meanings were mixed.
            identifiers = ["model"] + (["loss_rate"] if "loss_rate" in selected else [])
            if selected.duplicated(identifiers).any():
                raise ValueError("Duplicate model/loss estimates within a benchmark setting; "
                                 "keep distinct scientific configurations in separate artifacts")
            models = _benchmark_models(selected)
            if not models:
                continue
            finite_rates = (pd.to_numeric(selected["loss_rate"], errors="coerce").dropna().unique()
                            if "loss_rate" in selected else np.array([]))
            natural = scope.get("mechanism") == "natural" or len(finite_rates) == 0
            curves = not natural and len(finite_rates) > 1
            simulation = pd.notna(scope.get("sharing_strength", np.nan))
            title_parts = []
            for label, value in scope.items():
                if pd.isna(value):
                    continue
                if label == "sharing_strength":
                    title_parts.append(rf"$\rho={float(value):g}$")
                elif label == "calibration_fraction":
                    title_parts.append(f"paired = {float(value):.0%}")
                else:
                    title_parts.append(str(value).replace("_", " "))
            if not curves and len(finite_rates) == 1:
                title_parts.append(f"loss = {float(finite_rates[0]):.0%}")
            title = " · ".join(title_parts)
            stem = "__".join(f"{label}_{_slug(value)}" for label, value in scope.items() if pd.notna(value)) or "results"
            destination = directory / f"{stem}__all_benchmarks_0908.pdf"
            count = len(metrics)
            if curves:
                legend_columns = min(4, max(1, int(4.2 * count / 2.8)))
                legend_rows = int(np.ceil(len(models) / legend_columns))
                figure, axes = plt.subplots(1, count, squeeze=False,
                    figsize=(4.2 * count, 3.5 + .27 * legend_rows))
                for axis, metric in zip(axes[0], metrics):
                    _benchmark_curve(axis, selected, metric, models)
                    axis.set_title(_benchmark_metric_title(metric, simulation=simulation), loc="left", pad=9)
                from matplotlib.lines import Line2D
                handles = [Line2D([], [], **_benchmark_style(model, models), linestyle="None")
                           for model in models]
                legend = figure.legend(handles, [_BENCHMARK_LABELS.get(model, model) for model in models],
                              loc="upper center", bbox_to_anchor=(.5, .94), frameon=False,
                              ncol=min(legend_columns, len(models)), fontsize=8.5, columnspacing=1.5,
                              handlelength=1.2)
                figure.suptitle(title, fontsize=10, y=.995)
                figure.text(.5, .012, "Points show evaluated loss rates; endpoint-only fits are not connected. "
                            "Means over folds and repetitions; no diagnostic confidence intervals.",
                            ha="center", va="bottom", fontsize=8, color="#555555")
                figure.canvas.draw()
                legend_bottom = legend.get_window_extent(figure.canvas.get_renderer()).transformed(
                    figure.transFigure.inverted()).y0
                figure.subplots_adjust(left=.065, right=.985, bottom=.18,
                                       top=legend_bottom - .10, wspace=.33)
            else:
                figure, axes = plt.subplots(1, count, squeeze=False,
                    figsize=(5.6 * count, max(3.4, .30 * len(models) + 1.1)))
                for axis, metric in zip(axes[0], metrics):
                    _benchmark_dots(axis, selected, metric, models)
                    axis.set_xlabel(_benchmark_metric_title(metric, simulation=simulation))
                figure.suptitle(title, fontsize=10, y=.995)
                figure.text(.5, .01, "Means over folds and repetitions; no diagnostic confidence intervals.",
                            ha="center", va="bottom", fontsize=8, color="#555555")
                figure.tight_layout(rect=(0, .04, 1, .94), w_pad=2.)
            _save_display(figure, destination, show)
            paths[stem] = destination
    return paths


_INFORMATION_ARMS = ("Reference-only", "PU", "Reference+PU")
_INFORMATION_LABELS = {"Reference-only": "Reference-only",
                       "PU": "Calibrated PU",
                       "Reference+PU": "Reference + PU"}


def plot_information_budget_results(artifacts: Any, output_dir: str | Path, *,
                                    show: bool = True) -> dict[str, Path]:
    """Compare three independent-logistic arms at the same paired budget.

    Additional 1-by-3 PDF figures show the primary natural evaluation or the
    80%-loss endpoint. Each scientific setting gets its own figure, including
    each simulation sharing strength. ``PU`` is the calibrated independent
    predictor, not PU-MIRT or PU-Joint. Missing arms/undefined metrics are
    explicitly annotated; they are never replaced by another model or rate.
    Means follow the same fold-then-repetition aggregation as the main plots.
    This function does not overwrite or replace any existing figure.
    """
    frame = _primary(_benchmark_aggregate(artifacts.tables))
    if frame.empty:
        return {}
    frame = frame.loc[frame["model"].isin(_INFORMATION_ARMS)].copy()
    natural = (frame["mechanism"].eq("natural") if "mechanism" in frame
               else pd.Series(False, index=frame.index))
    endpoint = (np.isclose(pd.to_numeric(frame["loss_rate"], errors="coerce"), .8)
                if "loss_rate" in frame else np.zeros(len(frame), dtype=bool))
    frame = frame.loc[natural | endpoint]
    if frame.empty:
        return {}
    scope_columns = [key for key in _BENCHMARK_SCOPES if key in frame]
    grouping = (frame.groupby(scope_columns, observed=True, dropna=False, sort=True)
                if scope_columns else [((), frame)])
    directory, paths = Path(output_dir) / "information_budget", {}
    with plt.rc_context(_STYLE):
        for scope_key, selected in grouping:
            key = scope_key if isinstance(scope_key, tuple) else (scope_key,)
            scope = dict(zip(scope_columns, key))
            if selected.duplicated("model").any():
                raise ValueError("Duplicate model estimates within an information-budget setting; "
                                 "keep distinct scientific configurations in separate artifacts")
            simulation = pd.notna(scope.get("sharing_strength", np.nan))
            title_parts = [str(scope.get("dataset", "Results")).replace("_", " ")]
            if simulation:
                title_parts.append(rf"$\rho={float(scope['sharing_strength']):g}$")
            fraction = scope.get("calibration_fraction", np.nan)
            if pd.notna(fraction):
                title_parts.append(f"paired = {float(fraction):.0%}")
            title_parts.append("natural paired evaluation" if scope.get("mechanism") == "natural"
                               else "positive-label loss = 80%")
            figure, axes = plt.subplots(1, len(PRIMARY_METRICS), figsize=(11.6, 3.05))
            for column, (axis, metric) in enumerate(zip(axes, PRIMARY_METRICS)):
                for position, model in enumerate(_INFORMATION_ARMS):
                    estimates = selected.loc[selected["model"].eq(model)]
                    value = (pd.to_numeric(estimates[metric], errors="coerce").iloc[0]
                             if not estimates.empty and metric in estimates else np.nan)
                    if np.isfinite(value):
                        style = _benchmark_style(model, _INFORMATION_ARMS)
                        style["label"] = _INFORMATION_LABELS[model]
                        style["markersize"] = 7
                        axis.plot(float(value), position, **style, linestyle="None",
                                  markeredgewidth=.7)
                    else:
                        message = "Not run" if estimates.empty else "Undefined / unavailable"
                        axis.text(.5, position, message, transform=axis.get_yaxis_transform(),
                                  ha="center", va="center", fontsize=8, color="#888888")
                _style_axis(axis, loss_axis=False)
                axis.set_yticks(np.arange(len(_INFORMATION_ARMS)),
                    [_INFORMATION_LABELS[model] for model in _INFORMATION_ARMS]
                    if column == 0 else [""] * len(_INFORMATION_ARMS))
                axis.set_ylim(len(_INFORMATION_ARMS) - .5, -.5)
                axis.grid(False, axis="y")
                axis.grid(axis="x", color="#E6E6E6", linewidth=.6)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
                axis.set_xlabel(_metric_title(metric, simulation=simulation))
            figure.suptitle(" · ".join(title_parts), fontsize=10, y=.98)
            figure.text(.5, .015, "Independent logistic predictors; same paired-reference budget. "
                        "Means over folds and repetitions; no diagnostic confidence intervals.",
                        ha="center", va="bottom", fontsize=8, color="#555555")
            figure.tight_layout(rect=(0, .07, 1, .92), w_pad=2.)
            stem = "__".join(f"{label}_{_slug(value)}" for label, value in scope.items()
                              if pd.notna(value)) or "results"
            destination = directory / f"{stem}__information_budget_0908.pdf"
            _save_display(figure, destination, show)
            paths[stem] = destination
    return paths
