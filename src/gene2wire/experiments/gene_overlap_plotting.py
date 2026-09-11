"""Compact, explicitly titled figures for gene-panel overlap experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = {"PU": "#555555", "PU-MIRT": "#0072B2", "PU-Joint": "#D55E00"}
CONTROL_COLORS = {
    "intersection": "#009E73",
    "separate_panel": "#CC79A7",
    "union": MODEL_COLORS["PU"],
    "all_gene_oracle": "#E69F00",
}
METRICS = (("macro_auprc", "Macro AUPRC ↑"),
           ("macro_log_loss", "Macro log loss ↓"),
           ("macro_brier", "Macro Brier ↓"))


def _save_show(fig, path: Path, display: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if display:
        plt.show()
    else:
        plt.close(fig)


def _line_or_points(ax, x, y, *, label, color):
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    if len(np.unique(x)) == 1:
        ax.scatter(x, y, s=50, label=label, color=color)
    else:
        ax.plot(x, y, marker="o", linewidth=2, label=label, color=color)


def _condition_values(frame: pd.DataFrame, column: str) -> list[float | None]:
    """Return explicit simulation conditions without treating NaN as a level."""
    if column not in frame:
        return [None]
    values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
    return sorted(float(value) for value in values) or [None]


def _select_condition(frame: pd.DataFrame, column: str, value: float | None) -> pd.DataFrame:
    if value is None or column not in frame:
        if column not in frame:
            return frame
        numeric = pd.to_numeric(frame[column], errors="coerce")
        return frame.loc[numeric.isna()]
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[np.isclose(numeric, value, rtol=0, atol=1e-12)]


def _design_values(frame: pd.DataFrame) -> list[str | None]:
    if "panel_design" not in frame:
        return [None]
    values = frame["panel_design"].dropna().astype(str).unique()
    return sorted(values.tolist()) or [None]


def _label(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame:
        return default
    values = frame[column].dropna().astype(str).unique()
    return str(values[0]) if len(values) == 1 else default


def _rho_suffix(rho: float | None) -> str:
    if rho is None:
        return ""
    text = f"{rho:g}".replace("-", "m").replace(".", "p")
    return f"_rho{text}"


def _design_suffix(design: str | None) -> str:
    return "" if design is None else f"_{design}"


def _curve(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one finite value per x coordinate for a plotted curve."""
    if frame.empty or "actual_overlap" not in frame or metric not in frame:
        return pd.DataFrame(columns=("x", "y"))
    x = pd.to_numeric(frame["actual_overlap"], errors="coerce")
    y = pd.to_numeric(frame[metric], errors="coerce")
    finite = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    if not np.any(finite):
        return pd.DataFrame(columns=("x", "y"))
    data = pd.DataFrame({"x": x[finite], "y": y[finite]})
    # A curve must never contain multiple points at one x.  Duplicate rows can
    # occur in old exports when an extra scenario coordinate was not displayed;
    # averaging them here is only a plotting fallback, while raw CSVs remain
    # untouched and the summary writer validates scientific contrasts.
    return data.groupby("x", as_index=False, observed=True)["y"].mean().sort_values("x")


def _title(dataset: str, design: str | None, rho: float | None, topic: str) -> str:
    design_label = "all panel designs" if design is None else f"{design} panels"
    rho_label = "" if rho is None else f"; sharing strength ρ={rho:g}"
    return f"{dataset} — {design_label}{rho_label} — {topic}"


def _make_axes(title: str, *, relative: bool = False):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    fig.suptitle(title, fontsize=13)
    for ax, (metric, metric_title) in zip(axes, METRICS):
        ax.set_title(metric_title, fontsize=10)
        ax.set_xlabel("Actual gene-panel overlap ω")
        ax.set_ylabel("Relative gain" if relative else "Score")
        ax.grid(alpha=.2)
    return fig, axes


def _refresh_invalid_contrasts(artifacts):
    """Regenerate derived contrasts for old RESULTS_ONLY exports when needed."""
    contrasts = artifacts.tables.get("overlap_contrasts", pd.DataFrame())
    value_column = "difference_vs_100pct_overlap"
    valid = (not contrasts.empty and value_column in contrasts and
             pd.to_numeric(contrasts[value_column], errors="coerce").notna().all())
    if not valid and "per_repetition" in artifacts.tables:
        from .gene_overlap import rebuild_overlap_summaries
        rebuild_overlap_summaries(artifacts)
        contrasts = artifacts.tables.get("overlap_contrasts", pd.DataFrame())
    return contrasts


def plot_overlap_results(artifacts, figure_dir, *, prefix="gene_overlap", display=True):
    """Plot methods, controls and R_M, with one clearly labeled figure per rho."""
    figure_dir = Path(figure_dir)
    aggregate = artifacts.tables.get("aggregate", pd.DataFrame())
    outputs = []
    if aggregate.empty or "arm" not in aggregate or "model" not in aggregate:
        return outputs

    dataset = _label(aggregate, "dataset", "Gene-panel overlap")
    primary = aggregate.loc[
        aggregate["arm"].eq("union") & aggregate["model"].isin(MODEL_COLORS)
    ].copy()
    for design in _design_values(primary):
        if design is None or "panel_design" not in primary:
            design_frame = primary
        else:
            design_frame = primary.loc[primary["panel_design"].astype(str).eq(design)]
        for rho in _condition_values(design_frame, "sharing_strength"):
            frame = _select_condition(design_frame, "sharing_strength", rho)
            if frame.empty:
                continue
            fig, axes = _make_axes(_title(dataset, design, rho, "union-model performance"))
            for ax, (metric, _) in zip(axes, METRICS):
                for model, model_frame in frame.groupby("model", observed=True):
                    curve = _curve(model_frame, metric)
                    if not curve.empty:
                        _line_or_points(ax, curve["x"], curve["y"],
                                        label=model, color=MODEL_COLORS[model])
            axes[0].legend(frameon=False)
            path = figure_dir / (
                f"{prefix}{_design_suffix(design)}{_rho_suffix(rho)}_primary.pdf"
            )
            _save_show(fig, path, display)
            outputs.append(path)

    controls = aggregate.loc[aggregate["model"].eq("PU")].copy()
    if not controls.empty:
        for design in _design_values(controls):
            if design is None or "panel_design" not in controls:
                design_frame = controls
            else:
                design_frame = controls.loc[controls["panel_design"].astype(str).eq(design)]
            for rho in _condition_values(design_frame, "sharing_strength"):
                frame = _select_condition(design_frame, "sharing_strength", rho)
                if frame.empty or "arm" not in frame:
                    continue
                fig, axes = _make_axes(_title(dataset, design, rho, "PU control arms"))
                for ax, (metric, _) in zip(axes, METRICS):
                    for arm, arm_frame in frame.groupby("arm", observed=True):
                        curve = _curve(arm_frame, metric)
                        if not curve.empty:
                            _line_or_points(
                                ax, curve["x"], curve["y"], label=str(arm),
                                color=CONTROL_COLORS.get(str(arm), "#777777"),
                            )
                axes[0].legend(frameon=False, fontsize=8)
                path = figure_dir / (
                    f"{prefix}{_design_suffix(design)}{_rho_suffix(rho)}_controls.pdf"
                )
                _save_show(fig, path, display)
                outputs.append(path)

    contrasts = _refresh_invalid_contrasts(artifacts)
    if not contrasts.empty and "difference_vs_100pct_overlap" in contrasts:
        for design in _design_values(contrasts):
            if design is None or "panel_design" not in contrasts:
                design_frame = contrasts
            else:
                design_frame = contrasts.loc[contrasts["panel_design"].astype(str).eq(design)]
            for rho in _condition_values(design_frame, "sharing_strength"):
                frame = _select_condition(design_frame, "sharing_strength", rho)
                if frame.empty:
                    continue
                values = pd.to_numeric(frame["difference_vs_100pct_overlap"], errors="coerce")
                frame = frame.loc[values.notna()].copy()
                frame["difference_vs_100pct_overlap"] = values.loc[frame.index]
                if frame.empty:
                    continue
                summary = frame.groupby(["model", "metric", "actual_overlap"],
                                        dropna=False, observed=True)[
                    "difference_vs_100pct_overlap"].mean().reset_index()
                fig, axes = _make_axes(
                    _title(dataset, design, rho, "fragmentation contrast R_M(ω)"),
                    relative=True,
                )
                for ax, (metric, _) in zip(axes, METRICS):
                    metric_frame = summary.loc[summary["metric"].eq(metric)]
                    for model, model_frame in metric_frame.groupby("model", observed=True):
                        curve = _curve(model_frame, "difference_vs_100pct_overlap")
                        if not curve.empty:
                            _line_or_points(
                                ax, curve["x"], curve["y"], label=model,
                                color=MODEL_COLORS.get(model, "black"),
                            )
                    ax.axhline(0, color="black", linewidth=1, alpha=.5)
                axes[0].legend(frameon=False)
                path = figure_dir / (
                    f"{prefix}{_design_suffix(design)}{_rho_suffix(rho)}"
                    "_fragmentation_contrast.pdf"
                )
                _save_show(fig, path, display)
                outputs.append(path)
    return outputs
