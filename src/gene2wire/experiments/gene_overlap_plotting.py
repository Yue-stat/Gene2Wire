"""Compact PDF figures for partial gene-panel overlap experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = {"PU": "#555555", "PU-MIRT": "#0072B2", "PU-Joint": "#D55E00"}
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


def plot_overlap_results(artifacts, figure_dir, *, prefix="gene_overlap", display=True):
    """Plot main methods, controls and fragmentation-specific contrasts."""
    figure_dir = Path(figure_dir)
    aggregate = artifacts.tables.get("aggregate", pd.DataFrame())
    outputs = []
    if aggregate.empty:
        return outputs

    primary = aggregate[(aggregate.get("arm") == "union") &
                        aggregate["model"].isin(MODEL_COLORS)].copy()
    designs = [None]
    if "panel_design" in primary:
        designs = list(primary["panel_design"].dropna().unique())
    for design in designs:
        frame = primary if design is None else primary[primary["panel_design"] == design]
        if frame.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for ax, (metric, ylabel) in zip(axes, METRICS):
            for model, model_frame in frame.groupby("model", observed=True):
                if metric in model_frame:
                    _line_or_points(ax, model_frame["actual_overlap"], model_frame[metric],
                                    label=model, color=MODEL_COLORS.get(model, "black"))
            ax.set(xlabel="Actual gene-panel overlap", ylabel=ylabel)
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False)
        suffix = "" if design is None else f"_{design}"
        path = figure_dir / f"{prefix}{suffix}_primary.pdf"
        _save_show(fig, path, display)
        outputs.append(path)

    controls = aggregate[aggregate["model"] == "PU"].copy()
    if not controls.empty and "arm" in controls:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for ax, (metric, ylabel) in zip(axes, METRICS):
            for arm, frame in controls.groupby("arm", observed=True):
                if metric in frame:
                    _line_or_points(ax, frame["actual_overlap"], frame[metric],
                                    label=str(arm), color=None)
            ax.set(xlabel="Actual gene-panel overlap", ylabel=ylabel)
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False, fontsize=8)
        path = figure_dir / f"{prefix}_controls.pdf"
        _save_show(fig, path, display)
        outputs.append(path)

    contrasts = artifacts.tables.get("overlap_contrasts", pd.DataFrame())
    if not contrasts.empty:
        summary = contrasts.groupby(["model", "metric", "actual_overlap"], observed=True)[
            "difference_vs_100pct_overlap"].mean().reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for ax, (metric, ylabel) in zip(axes, METRICS):
            frame = summary[summary["metric"] == metric]
            for model, model_frame in frame.groupby("model", observed=True):
                _line_or_points(ax, model_frame["actual_overlap"],
                                model_frame["difference_vs_100pct_overlap"],
                                label=model, color=MODEL_COLORS.get(model, "black"))
            ax.axhline(0, color="black", linewidth=1, alpha=.5)
            ax.set(xlabel="Actual gene-panel overlap", ylabel=f"Relative gain: {ylabel}")
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False)
        path = figure_dir / f"{prefix}_fragmentation_contrast.pdf"
        _save_show(fig, path, display)
        outputs.append(path)
    return outputs

