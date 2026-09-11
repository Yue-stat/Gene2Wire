"""PDF and notebook curves for group-by-target block masking experiments.

The horizontal coordinate describes target panels, not positive-label loss.
Only outcomes in blocked group-target pairs of held-out cells are evaluated.
Full-panel controls are shown in separate rows on the identical evaluation
subsets; changing a subset can change a control curve without changing its fit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .plotting import (
    PU_MODELS,
    _BENCHMARK_LABELS,
    _INFORMATION_ARMS,
    _STYLE,
    _benchmark_legend_handles,
    _benchmark_linestyle,
    _benchmark_metric_title,
    _benchmark_models,
    _benchmark_style,
    _save_display,
    _slug,
    _style_axis,
)


BLOCK_METRICS = ("macro_auprc", "macro_log_loss", "macro_brier")
_BLOCK_SCOPES = ("dataset", "sharing_strength", "experiment", "group_mode", "analysis", "mechanism",
                 "loss_rate", "calibration_fraction", "calibration_spec",
                 "block_grouping")
_PANELS = ("masked", "full")
_PANEL_LABELS = {"masked": "Masked training panel",
                 "full": "Full training panel control"}
_CATEGORY_LABELS = {
    "primary": "Shared structure under missing target panels",
    "all_benchmarks": "All benchmark methods",
    "detection_only_benchmarks": "Predictors trained on detection labels",
    "information_budget": "Same paired-reference budget",
}


def _block_aggregate(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Preserve panel and scientific settings, averaging folds then repetitions."""
    for source in ("aggregate", "per_repetition", "metrics"):
        original = tables.get(source, pd.DataFrame())
        if not original.empty:
            frame = original.copy()
            break
    else:
        return pd.DataFrame()
    required = {"model", "training_panel", "evaluation_scope", "block_fraction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Block result coordinates are missing: {sorted(missing)}")
    # Observed-panel and per-group summaries are distinct evaluation questions.
    frame = frame.loc[frame["evaluation_scope"].eq("blocked")].copy()
    if frame.empty:
        return frame
    unexpected = set(frame["training_panel"].dropna()).difference(_PANELS)
    if unexpected or frame["training_panel"].isna().any():
        raise ValueError("Block training_panel must be 'masked' or 'full'")
    fraction = pd.to_numeric(frame["block_fraction"], errors="coerce")
    if not (np.isfinite(fraction) & fraction.ge(0) & fraction.le(1)).all():
        raise ValueError("block_fraction must be finite and in [0, 1]")
    frame["block_fraction"] = fraction
    # Zero blocking is a valid experiment coordinate, but provides no held-out
    # group-target entries to score. Keep it in exports, not in these curves.
    frame = frame.loc[frame["block_fraction"].gt(0)].copy()
    if frame.empty:
        return frame
    if source == "aggregate":
        return frame
    coordinates = [*_BLOCK_SCOPES, "model", "training_panel", "block_fraction",
                   "evaluation_scope", "probability_semantics"]
    groups = [key for key in coordinates if key in frame]
    metrics = [key for key in BLOCK_METRICS if key in frame]
    if not metrics:
        raise ValueError("Block result tables contain no supported evaluation metrics")
    if "repetition" in frame:
        frame = frame.groupby(groups + ["repetition"], observed=True, dropna=False)[
            metrics].mean().reset_index()
    return frame.groupby(groups, observed=True, dropna=False)[metrics].mean().reset_index()


def _block_curve(axis, frame: pd.DataFrame, metric: str, models: tuple[str, ...],
                 fractions: np.ndarray) -> None:
    """Keep gaps visible and retain the existing information-use line styles."""
    any_data = False
    for model in models:
        selected = frame.loc[frame["model"].eq(model), ["block_fraction", metric]]
        values = pd.to_numeric(selected.set_index("block_fraction")[metric],
                               errors="coerce").reindex(fractions).to_numpy(float)
        values = np.where(np.isfinite(values), values, np.nan)
        if not np.isfinite(values).any():
            continue
        axis.plot(fractions, values, **_benchmark_style(model, models),
                  linestyle=_benchmark_linestyle(model), markeredgewidth=.7)
        any_data = True
    _style_axis(axis, loss_axis=False)
    axis.set_xticks(fractions)
    span = float(fractions[-1] - fractions[0])
    padding = max(.025, .04 * span)
    axis.set_xlim(max(0., float(fractions[0]) - padding),
                  min(1.025, float(fractions[-1]) + padding))
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.set_xlabel("Targets with masked group blocks")
    if not any_data:
        axis.text(.5, .5, "No defined estimates", transform=axis.transAxes,
                  ha="center", va="center", color="#777777", fontsize=9)


def _categories(frame: pd.DataFrame, include_benchmarks: bool):
    available = _benchmark_models(frame)
    # Single-assay native-panel runs declare their supervision profile in the
    # exported mechanism coordinate.  Do not infer assay-only status merely
    # because Logistic/MIRT/Joint rows happen to be present in a mixed table.
    assay_primary = ("Logistic", "MIRT", "Joint")
    assay_only = (
        "mechanism" in frame
        and frame["mechanism"].notna().all()
        and frame["mechanism"].eq("assay_only").all()
    )
    primary_order = assay_primary if assay_only else PU_MODELS
    primary = tuple(model for model in primary_order if model in available)
    if primary:
        yield "primary", primary
    if not include_benchmarks:
        return
    if available:
        yield "all_benchmarks", available
    detection_only = tuple(model for model in available
                           if not model.startswith(("Reference-only", "Reference+PU"))
                           and model not in {"RF-reference", "RF-mixed"})
    if detection_only:
        yield "detection_only_benchmarks", detection_only
    if set(_INFORMATION_ARMS).issubset(available):
        yield "information_budget", _INFORMATION_ARMS


def _scope_title(scope: dict[str, Any]) -> str:
    parts = [str(scope.get("dataset", "Block masking")).replace("_", " ")]
    rho = scope.get("sharing_strength", np.nan)
    if pd.notna(rho):
        parts.append(rf"$\rho={float(rho):g}$")
    mechanism = scope.get("mechanism")
    if mechanism is not None and pd.notna(mechanism):
        parts.append(str(mechanism).replace("_", " "))
    loss = scope.get("loss_rate", np.nan)
    if pd.notna(loss):
        parts.append(f"positive-label loss = {float(loss):.0%}")
    paired = scope.get("calibration_fraction", np.nan)
    if pd.notna(paired):
        parts.append(f"paired = {float(paired):.0%}")
    for key in ("calibration_spec", "group_mode", "block_grouping"):
        value = scope.get(key)
        if value is not None and pd.notna(value):
            parts.append(str(value).replace("_", " "))
    return " · ".join(parts)


def plot_block_results(artifacts: Any, output_dir: str | Path, *,
                       show: bool = True,
                       include_benchmarks: bool = True) -> dict[str, Path]:
    """Plot blocked-test performance, saving PDF files and optionally displaying.

    Each figure has AUPRC, log loss and Brier columns, and separate masked/full
    training-panel rows. Shared x/y limits make the controls comparable. Every
    dataset, rho, detection mechanism, positive-label-loss rate and calibration
    setting gets separate figures. No confidence intervals are inferred from
    folds or repeated artificial masks. Missing model/fraction estimates stay
    missing; no horizontal model-comparison dot panels are created. Zero block
    fraction has no blocked outcomes to score, so it is omitted from curves;
    exports containing only that coordinate return an empty path dictionary.

    Four comparisons are emitted when their methods are available: the three
    primary PU methods, all benchmarks, predictors that do not train directly
    on paired-reference labels, and the three matched-information arms.
    Detection-only predictors can still use paired references for calibration.
    Hidden-positive metrics are intentionally absent: unassayed entries do not
    imply an observed non-detection, and predictions are scored as p, not h.
    """
    frame = _block_aggregate(artifacts.tables)
    if frame.empty:
        return {}
    missing = set(BLOCK_METRICS).difference(frame.columns)
    if missing:
        raise ValueError(f"Block evaluation metrics are missing: {sorted(missing)}")
    scope_columns = [key for key in _BLOCK_SCOPES if key in frame]
    grouping = (frame.groupby(scope_columns, observed=True, dropna=False, sort=True)
                if scope_columns else [((), frame)])
    paths: dict[str, Path] = {}
    with plt.rc_context(_STYLE):
        for scope_key, selected in grouping:
            key = scope_key if isinstance(scope_key, tuple) else (scope_key,)
            scope = dict(zip(scope_columns, key))
            coordinates = ["training_panel", "model", "block_fraction"]
            if selected.duplicated(coordinates).any():
                raise ValueError("Duplicate block estimates within a scientific setting; "
                                 "keep configurations and probability semantics separate")
            fractions = np.sort(selected["block_fraction"].unique())
            panels = tuple(panel for panel in _PANELS
                           if selected["training_panel"].eq(panel).any())
            stem = "__".join(f"{label}_{_slug(value)}" for label, value in scope.items()
                              if pd.notna(value)) or "results"
            simulation = pd.notna(scope.get("sharing_strength", np.nan))
            for category, models in _categories(selected, include_benchmarks):
                legend_columns = min(4, len(models))
                legend_rows = int(np.ceil(len(models) / legend_columns))
                figure, axes = plt.subplots(len(panels), len(BLOCK_METRICS),
                    squeeze=False, sharex=True, sharey="col",
                    figsize=(12.6, 3.0 * len(panels) + 1.2 + .27 * legend_rows))
                for row, panel in enumerate(panels):
                    panel_frame = selected.loc[selected["training_panel"].eq(panel)]
                    for column, metric in enumerate(BLOCK_METRICS):
                        axis = axes[row, column]
                        _block_curve(axis, panel_frame, metric, models, fractions)
                        axis.set_title(_benchmark_metric_title(metric, simulation=simulation),
                                       loc="left", pad=9)
                        if column == 0:
                            axis.set_ylabel(_PANEL_LABELS[panel])
                        if row < len(panels) - 1:
                            axis.set_xlabel("")
                handles = _benchmark_legend_handles(axes.ravel(), models)
                legend = figure.legend(handles,
                    [_BENCHMARK_LABELS.get(model, model) for model in models],
                    loc="upper center", bbox_to_anchor=(.5, .93), frameon=False,
                    ncol=legend_columns, fontsize=8.5, columnspacing=1.5, handlelength=2.5)
                figure.suptitle(f"{_scope_title(scope)}\n{_CATEGORY_LABELS[category]}",
                                fontsize=10, y=.995)
                footer = "Evaluation uses blocked group–target pairs in held-out cells.\n"
                if set(panels) == set(_PANELS):
                    footer = "Both rows evaluate the same blocked group–target pairs in held-out cells.\n"
                if "full" in panels:
                    footer += ("Full-panel fits are reused; their curves can change as the "
                               "evaluation subset changes.\n")
                footer += ("Solid: PU or paired-reference information; dashed: neither. "
                           "Descriptive means; no confidence intervals.")
                if category == "detection_only_benchmarks":
                    footer += " Paired references may calibrate detection."
                figure.text(.5, .012, footer, ha="center", va="bottom",
                            fontsize=8, color="#555555")
                figure.canvas.draw()
                legend_bottom = legend.get_window_extent(figure.canvas.get_renderer()).transformed(
                    figure.transFigure.inverted()).y0
                figure.subplots_adjust(left=.075, right=.985, bottom=.14,
                                       top=legend_bottom - .05, wspace=.30, hspace=.38)
                destination = Path(output_dir) / "block_masking" / category / f"{stem}__{category}.pdf"
                _save_display(figure, destination, show)
                paths[f"{stem}__{category}"] = destination
    return paths
