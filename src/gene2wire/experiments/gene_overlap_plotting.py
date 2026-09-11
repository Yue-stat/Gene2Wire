"""Compact, explicitly titled figures for gene-panel overlap experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = {"PU": "#555555", "PU-MIRT": "#0072B2", "PU-Joint": "#D55E00"}
CONTROL_COLORS = {
    "intersection": "#009E73",
    "disjoint_coefficient": "#CC79A7",
    "union": MODEL_COLORS["PU"],
    "all_gene_oracle": "#E69F00",
}
ABLATION_COLORS = {"shared_a": "#0072B2", "separate_a": "#D55E00"}
METRICS = (("macro_auprc", "Macro AUPRC ↑"),
           ("macro_log_loss", "Macro log loss ↓"),
           ("macro_brier", "Macro Brier ↓"))
SCENARIO_COLUMNS = (
    "analysis", "mechanism", "loss_rate", "calibration_fraction",
    "calibration_spec",
)


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


def _native(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _scenario_slices(frame: pd.DataFrame):
    """Yield complete observation scenarios so curves are never averaged together."""
    columns = [column for column in SCENARIO_COLUMNS if column in frame]
    if not columns:
        yield {}, frame
        return
    grouped = frame.groupby(columns, dropna=False, observed=True, sort=True)
    for values, selected in grouped:
        values = values if isinstance(values, tuple) else (values,)
        yield {column: _native(value) for column, value in zip(columns, values)}, selected


def _scenario_label(scenario: dict) -> str:
    mechanism = scenario.get("mechanism")
    rate = scenario.get("loss_rate")
    analysis = scenario.get("analysis")
    if mechanism == "natural":
        label = "Natural paired observation"
    elif rate is not None and np.isclose(float(rate), 0.0, rtol=0, atol=1e-12):
        label = "Primary: no added positive-label loss"
    elif rate is not None:
        mechanism_label = ({"technical_sar": "Technical-SAR"}.get(
            mechanism, str(mechanism or "synthetic").replace("_", "-").title()))
        label = f"Sensitivity: {mechanism_label}; {100 * float(rate):g}% added positive-label loss"
    elif analysis is not None:
        label = str(analysis).replace("_", " ").title()
    else:
        return ""
    calibration = scenario.get("calibration_fraction")
    specification = scenario.get("calibration_spec")
    if calibration is not None and not np.isclose(float(calibration), .2, rtol=0, atol=1e-12):
        label += f"; calibration={100 * float(calibration):g}%"
    if specification not in (None, "correct"):
        label += f"; calibration spec={specification}"
    return label


def _scenario_suffix(scenario: dict) -> str:
    """Use a stable suffix for every non-default overlap observation scenario."""
    mechanism = scenario.get("mechanism")
    rate = scenario.get("loss_rate")
    analysis = scenario.get("analysis")
    specification = scenario.get("calibration_spec")
    calibration = scenario.get("calibration_fraction")
    is_default = (
        analysis in (None, "primary")
        and mechanism in (None, "technical_sar")
        and (rate is None or np.isclose(float(rate), 0.0, rtol=0, atol=1e-12))
        and specification in (None, "correct")
        and (calibration is None or np.isclose(float(calibration), .2, rtol=0, atol=1e-12))
    )
    if is_default:
        return ""
    if mechanism == "natural":
        pieces = ["natural"]
    elif rate is not None and float(rate) > 0:
        pieces = ["sensitivity", str(mechanism or "synthetic"), f"loss{100 * float(rate):g}"]
    else:
        pieces = [str(analysis or "scenario"), str(mechanism or "unknown")]
    if calibration is not None and not np.isclose(float(calibration), .2, rtol=0, atol=1e-12):
        pieces.append(f"cal{100 * float(calibration):g}")
    if specification not in (None, "correct"):
        pieces.append(str(specification))
    return "_" + "_".join(piece.replace(".", "p").replace("-", "_") for piece in pieces)


def _scientific_slices(frame: pd.DataFrame):
    for design in _design_values(frame):
        design_frame = (frame if design is None or "panel_design" not in frame else
                        frame.loc[frame["panel_design"].astype(str).eq(design)])
        for rho in _condition_values(design_frame, "sharing_strength"):
            rho_frame = _select_condition(design_frame, "sharing_strength", rho)
            for scenario, scenario_frame in _scenario_slices(rho_frame):
                if not scenario_frame.empty:
                    yield design, rho, scenario, scenario_frame


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
    if data["x"].duplicated().any():
        raise ValueError(
            "A plotted curve contains multiple rows at one overlap. A scientific "
            "scenario coordinate is missing from the figure grouping."
        )
    return data.sort_values("x")


def _title(dataset: str, design: str | None, rho: float | None, topic: str,
           scenario: dict | None = None) -> str:
    design_label = "all panel designs" if design is None else f"{design} panels"
    rho_label = "" if rho is None else f"; sharing strength ρ={rho:g}"
    scenario_label = _scenario_label(scenario or {})
    scenario_text = "" if not scenario_label else f" — {scenario_label}"
    return f"{dataset} — {design_label}{rho_label}{scenario_text} — {topic}"


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
        # RESULTS_ONLY plotting is read-only with respect to the completed
        # export.  Rebuild an obsolete derived table in memory only.
        rebuild_overlap_summaries(artifacts, persist=False)
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
    for design, rho, scenario, frame in _scientific_slices(primary):
        fig, axes = _make_axes(
            _title(dataset, design, rho, "union-model performance", scenario))
        for ax, (metric, _) in zip(axes, METRICS):
            for model, model_frame in frame.groupby("model", observed=True):
                curve = _curve(model_frame, metric)
                if not curve.empty:
                    _line_or_points(ax, curve["x"], curve["y"],
                                    label=model, color=MODEL_COLORS[model])
        axes[0].legend(frameon=False)
        path = figure_dir / (
            f"{prefix}{_design_suffix(design)}{_rho_suffix(rho)}"
            f"{_scenario_suffix(scenario)}_primary.pdf"
        )
        _save_show(fig, path, display)
        outputs.append(path)

    controls = aggregate.loc[aggregate["model"].eq("PU")].copy()
    if not controls.empty:
        for design, rho, scenario, frame in _scientific_slices(controls):
            if "arm" not in frame or not set(frame["arm"].astype(str)).difference({"union"}):
                continue
            fig, axes = _make_axes(
                _title(dataset, design, rho, "PU control arms", scenario))
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
                f"{prefix}{_design_suffix(design)}{_rho_suffix(rho)}"
                f"{_scenario_suffix(scenario)}_controls.pdf"
            )
            _save_show(fig, path, display)
            outputs.append(path)

    ablation = aggregate.loc[
        aggregate["arm"].isin(ABLATION_COLORS)
        & aggregate["model"].eq("PU-MIRT")
    ].copy()
    for design, rho, scenario, frame in _scientific_slices(ablation):
        fig, axes = _make_axes(
            _title(dataset, design, rho,
                   "matched Shared-A versus Separate-A MIRT", scenario))
        for ax, (metric, _) in zip(axes, METRICS):
            for arm, arm_frame in frame.groupby("arm", observed=True):
                curve = _curve(arm_frame, metric)
                if not curve.empty:
                    _line_or_points(
                        ax, curve["x"], curve["y"],
                        label=("Shared-A" if arm == "shared_a" else "Separate-A"),
                        color=ABLATION_COLORS[str(arm)],
                    )
        axes[0].legend(frameon=False)
        path = figure_dir / (
            f"{prefix}{_design_suffix(design)}{_rho_suffix(rho)}"
            f"{_scenario_suffix(scenario)}_shared_A_ablation.pdf"
        )
        _save_show(fig, path, display)
        outputs.append(path)

    contrasts = _refresh_invalid_contrasts(artifacts)
    if not contrasts.empty and "difference_vs_100pct_overlap" in contrasts:
        for design, rho, scenario, frame in _scientific_slices(contrasts):
            values = pd.to_numeric(frame["difference_vs_100pct_overlap"], errors="coerce")
            frame = frame.loc[values.notna()].copy()
            frame["difference_vs_100pct_overlap"] = values.loc[frame.index]
            if frame.empty:
                continue
            summary = frame.groupby(["model", "metric", "actual_overlap"],
                                    dropna=False, observed=True)[
                "difference_vs_100pct_overlap"].mean().reset_index()
            fig, axes = _make_axes(
                _title(dataset, design, rho, "fragmentation contrast R_M(ω)", scenario),
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
                f"{prefix}{_design_suffix(design)}{_rho_suffix(rho)}{_scenario_suffix(scenario)}"
                "_fragmentation_contrast.pdf"
            )
            _save_show(fig, path, display)
            outputs.append(path)

    shared_contrasts = artifacts.tables.get("shared_a_contrasts", pd.DataFrame())
    if not shared_contrasts.empty and "difference_vs_100pct_overlap" in shared_contrasts:
        for design, rho, scenario, frame in _scientific_slices(shared_contrasts):
            values = pd.to_numeric(frame["difference_vs_100pct_overlap"], errors="coerce")
            metric_frame = frame.loc[values.notna()].copy()
            metric_frame["difference_vs_100pct_overlap"] = values.loc[metric_frame.index]
            if metric_frame.empty:
                continue
            summary = metric_frame.groupby(
                ["metric", "actual_overlap"], dropna=False, observed=True
            )["difference_vs_100pct_overlap"].mean().reset_index()
            fig, axes = _make_axes(
                _title(dataset, design, rho,
                       "target-loading sharing contrast Q_A(ω)", scenario),
                relative=True,
            )
            for ax, (metric, _) in zip(axes, METRICS):
                curve = _curve(
                    summary.loc[summary["metric"].eq(metric)],
                    "difference_vs_100pct_overlap",
                )
                if not curve.empty:
                    _line_or_points(
                        ax, curve["x"], curve["y"],
                        label="Shared-A − Separate-A",
                        color="#7B3294",
                    )
                ax.axhline(0, color="black", linewidth=1, alpha=.5)
            axes[0].legend(frameon=False)
            path = figure_dir / (
                f"{prefix}{_design_suffix(design)}{_rho_suffix(rho)}"
                f"{_scenario_suffix(scenario)}_shared_A_contrast.pdf"
            )
            _save_show(fig, path, display)
            outputs.append(path)
    return outputs
