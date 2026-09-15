"""Plot-only helpers for measurement-degradation V4 notebooks.

The helpers consume exported ``per_repetition`` tables.  They do not fit a
model or read a checkpoint.  Internal model identifiers remain unchanged for
provenance; publication-facing names are applied only while plotting.

``funkyheatmappy`` is imported lazily by :func:`render_v4_funkyheatmap`, so the
table preparation and normalization contracts can be tested without that
optional notebook dependency.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    """A public plot metric and its exported-table semantics."""

    key: str
    column: str
    label: str
    higher_is_better: bool
    requires_reference_probability: bool = False


@dataclass(frozen=True)
class CurveFamilySpec:
    """One predeclared V4 measurement-degradation curve."""

    key: str
    role: str
    label: str
    x_label: str
    descending: bool


METRIC_SPECS: Mapping[str, MetricSpec] = {
    "auprc": MetricSpec("auprc", "macro_auprc", "Macro AUPRC ↑", True),
    "auroc": MetricSpec("auroc", "macro_auroc", "Macro AUROC ↑", True),
    "log_loss": MetricSpec(
        "log_loss", "macro_log_loss", "Macro log loss ↓", False, True
    ),
    "brier": MetricSpec(
        "brier", "macro_brier", "Macro Brier score ↓", False, True
    ),
    "hidden_recall": MetricSpec(
        "hidden_recall",
        "hidden_recall_at_h",
        "On-panel hidden Recall@$H$ ↑",
        True,
    ),
}

# This is the only tuple a notebook user needs to edit to add/remove the five
# predeclared metrics from both line figures and the funky heatmap.
DEFAULT_V4_METRICS = (
    "auprc",
    "auroc",
    "log_loss",
    "brier",
    "hidden_recall",
)

CURVE_FAMILY_SPECS: Mapping[str, CurveFamilySpec] = {
    "positive_label_loss": CurveFamilySpec(
        "positive_label_loss",
        "positive_label_loss_only",
        "Positive-label loss only",
        "Positive-label loss",
        False,
    ),
    "gene_coverage": CurveFamilySpec(
        "gene_coverage",
        "gene_coverage_only",
        "Gene coverage only",
        "Gene coverage",
        True,
    ),
    "target_coverage": CurveFamilySpec(
        "target_coverage",
        "target_coverage_only",
        "Target coverage only",
        "Target coverage",
        True,
    ),
    "combined_masks": CurveFamilySpec(
        "combined_masks",
        "combined_mask_curve",
        "All three masks",
        "Harmonic mean retained measurement",
        True,
    ),
}

DEFAULT_V4_CURVE_FAMILIES = tuple(CURVE_FAMILY_SPECS)

# Funkyheatmap shows the nine one-factor physical scenarios exactly once.  The
# combined harmonic-mean line summarizes the full factorial, but duplicating
# all 45 conditions in the scorecard would obscure its mechanism-by-level view.
DEFAULT_V4_FUNKY_FAMILIES = (
    "positive_label_loss",
    "gene_coverage",
    "target_coverage",
)

FUNKY_FAMILY_LABELS = {
    "positive_label_loss": "Positive-label loss only",
    "gene_coverage": "Gene coverage",
    "target_coverage": "Target coverage",
}

GENE2WIRE_FAMILY_MODELS = ("PU", "PU-MIRT", "PU-Joint")
EXTERNAL_CURVE_MODELS = (
    "PU-Joint",
    "GenEML-authors-mask",
    "Inductive-PU-MC",
)
FUNKY_MODELS = (
    "PU-Joint",
    "GenEML-authors-mask",
    "Inductive-PU-MC",
    "SAR-PU",
)

MODEL_DISPLAY_LABELS = {
    "PU": "PU",
    "PU-MIRT": "PU-MIRT",
    "PU-Joint": "Gene2Wire",
    "GenEML": "GenEML",
    "GenEML-adapted": "GenEML",
    "GenEML-authors": "GenEML",
    "GenEML-authors-mask": "GenEML",
    "Inductive-PU-MC": "PU matrix completion",
    "ShiftIMC-adapted": "PU matrix completion",
    "PU-matrix-completion": "PU matrix completion",
    "SAR-PU": "SAR-PU",
    "SAR-PU-authors": "SAR-PU",
}

MODEL_STYLES = {
    "PU": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#0072B2",
    },
    "PU-MIRT": {
        "color": "#009E73",
        "marker": "s",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#009E73",
    },
    "PU-Joint": {
        "color": "#D55E00",
        "marker": "^",
        "linestyle": "-",
        "markerfacecolor": "#D55E00",
        "markeredgecolor": "#D55E00",
    },
    "GenEML-adapted": {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#CC79A7",
    },
    "GenEML": {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#CC79A7",
    },
    "GenEML-authors": {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#CC79A7",
    },
    "GenEML-authors-mask": {
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#CC79A7",
    },
    "Inductive-PU-MC": {
        "color": "#56B4E9",
        "marker": "P",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#56B4E9",
    },
    "ShiftIMC-adapted": {
        "color": "#56B4E9",
        "marker": "P",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#56B4E9",
    },
    "PU-matrix-completion": {
        "color": "#56B4E9",
        "marker": "P",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#56B4E9",
    },
    "SAR-PU": {
        "color": "#E69F00",
        "marker": "X",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#E69F00",
    },
    "SAR-PU-authors": {
        "color": "#E69F00",
        "marker": "X",
        "linestyle": "--",
        "markerfacecolor": "white",
        "markeredgecolor": "#E69F00",
    },
}

FUNKY_BLUE = "#2F80ED"

_PHYSICAL_CONDITION_COLUMNS = (
    "gene_requested_coverage",
    "target_requested_coverage",
    "positive_retention",
)


def _as_frame(
    value: pd.DataFrame, *, name: str, required: Sequence[str] = ()
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = set(required).difference(value.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    return value.copy()


def _metric_specs(metrics: Sequence[str]) -> tuple[MetricSpec, ...]:
    keys = tuple(map(str, metrics))
    if not keys:
        raise ValueError("metrics must contain at least one metric key")
    if len(set(keys)) != len(keys):
        raise ValueError("metrics must not contain duplicate keys")
    unknown = sorted(set(keys).difference(METRIC_SPECS))
    if unknown:
        raise ValueError(
            f"Unknown V4 metric keys {unknown}; choose from {sorted(METRIC_SPECS)}"
        )
    return tuple(METRIC_SPECS[key] for key in keys)


def _family_spec(family: str) -> CurveFamilySpec:
    try:
        return CURVE_FAMILY_SPECS[str(family)]
    except KeyError as error:
        raise ValueError(
            f"Unknown V4 curve family {family!r}; "
            f"choose from {list(CURVE_FAMILY_SPECS)}"
        ) from error


def _role_mask(frame: pd.DataFrame, role: str) -> pd.Series:
    return frame["condition_roles"].fillna("").astype(str).str.contains(
        role, regex=False
    )


def _is_reference_probability(frame: pd.DataFrame) -> pd.Series:
    if "probability_semantics" not in frame:
        raise ValueError(
            "Proper-score plots require the exported probability_semantics column"
        )
    return frame["probability_semantics"].fillna("").astype(str).str.startswith(
        "reference"
    )


def _add_facets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if (
        "sharing_strength" not in result
        or pd.to_numeric(result["sharing_strength"], errors="coerce").isna().all()
    ):
        result["facet"] = "all"
        result["facet_order"] = 0
        return result
    sharing = pd.to_numeric(result["sharing_strength"], errors="coerce")
    if sharing.isna().any():
        raise ValueError(
            "A V4 plotting table cannot mix finite and missing sharing_strength values"
        )
    levels = sorted(sharing.unique())
    order = {float(level): index for index, level in enumerate(levels)}
    result["facet"] = [f"sharing={value:g}" for value in sharing]
    result["facet_order"] = [order[float(value)] for value in sharing]
    return result


def _validate_inferred_v4_factorial(frame: pd.DataFrame) -> None:
    """Require every observed facet/repetition to contain the full V4 grid.

    Plot-only helpers do not receive the manifest, so they infer the three
    declared coordinate grids from the complete table and then require their
    Cartesian product in every facet/repetition.  A failed model may be absent;
    a physical condition may not silently disappear for every model.
    """

    coordinate_columns = list(_PHYSICAL_CONDITION_COLUMNS)
    grids: list[tuple[float, ...]] = []
    for column in coordinate_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not len(values) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"V4 per_repetition schedule has a missing/nonfinite {column}"
            )
        if np.any(values <= 0.0) or np.any(values > 1.0):
            raise ValueError(
                f"V4 per_repetition schedule has an invalid {column}; "
                "retained fractions must lie in (0, 1]"
            )
        grids.append(tuple(sorted(set(np.round(values, 12)))))

    expected = set(product(*grids))
    missing_records: list[str] = []
    for unit in (
        frame[["facet", "repetition"]]
        .drop_duplicates()
        .sort_values(["facet", "repetition"], kind="stable")
        .itertuples(index=False)
    ):
        selected = frame.loc[
            frame["facet"].eq(unit.facet)
            & frame["repetition"].eq(unit.repetition),
            coordinate_columns,
        ]
        actual = {
            tuple(np.round(np.asarray(record, dtype=float), 12))
            for record in selected.drop_duplicates().itertuples(
                index=False, name=None
            )
        }
        missing = sorted(expected.difference(actual))
        if missing:
            preview = ", ".join(
                f"G{_percent(gene)}/T{_percent(target)}/R{_percent(retention)}"
                for gene, target, retention in missing[:5]
            )
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
            missing_records.append(
                f"{unit.facet}, repetition {unit.repetition:g}: {preview}{suffix}"
            )
    if missing_records:
        raise ValueError(
            "V4 per_repetition schedule is incomplete; missing declared "
            "factorial conditions: " + " | ".join(missing_records)
        )


def _base_measurement_rows(per_repetition: pd.DataFrame) -> pd.DataFrame:
    required = (
        "model",
        "repetition",
        "condition_roles",
        "evaluation_scope",
        "mechanism",
        "gene_requested_coverage",
        "target_requested_coverage",
        "positive_retention",
        "probability_semantics",
    )
    frame = _as_frame(
        per_repetition, name="per_repetition.csv", required=required
    )
    if frame.empty:
        raise ValueError("per_repetition.csv is empty")
    frame = frame.loc[
        frame["evaluation_scope"].astype(str).eq("native_reference")
        & frame["mechanism"].astype(str).eq("assay_target_sar")
    ].copy()
    if frame.empty:
        raise ValueError(
            "per_repetition.csv has no native-reference assay_target_sar rows"
        )
    for column in (
        "repetition",
        "gene_requested_coverage",
        "target_requested_coverage",
        "positive_retention",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["repetition"].isna().any():
        raise ValueError("per_repetition.csv contains a missing repetition identifier")
    frame["model"] = frame["model"].astype(str)
    frame = _add_facets(frame)
    _validate_inferred_v4_factorial(frame)
    return frame


def harmonic_mean_retained_measurement(
    gene_coverage, target_coverage, positive_retention
) -> np.ndarray:
    """Harmonic mean of three requested retained-measurement fractions.

    The retained fractions, rather than the loss fractions, are used because a
    harmonic mean of losses collapses to zero whenever any one mechanism is
    unmasked.  Inputs must be finite and in ``(0, 1]``.
    """

    values = np.column_stack(
        [
            np.asarray(gene_coverage, dtype=float),
            np.asarray(target_coverage, dtype=float),
            np.asarray(positive_retention, dtype=float),
        ]
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(values > 1):
        raise ValueError("Retained-measurement fractions must be finite in (0, 1]")
    return 3.0 / np.sum(1.0 / values, axis=1)


def prepare_v4_curve_table(
    per_repetition: pd.DataFrame,
    *,
    family: str,
    metrics: Sequence[str] = DEFAULT_V4_METRICS,
) -> pd.DataFrame:
    """Select one V4 curve and attach its declared theoretical x coordinate."""

    specification = _family_spec(family)
    metric_specs = _metric_specs(metrics)
    frame = _base_measurement_rows(per_repetition)
    missing_metrics = sorted(
        {metric.column for metric in metric_specs}.difference(frame.columns)
    )
    if missing_metrics:
        raise ValueError(
            f"per_repetition.csv lacks requested V4 metrics: {missing_metrics}"
        )
    frame = frame.loc[_role_mask(frame, specification.role)].copy()
    if frame.empty:
        raise ValueError(
            f"No native-reference rows have V4 condition role {specification.role!r}"
        )
    for metric in metric_specs:
        frame[metric.column] = pd.to_numeric(frame[metric.column], errors="coerce")

    if specification.key == "positive_label_loss":
        frame["plot_x"] = 1.0 - frame["positive_retention"]
    elif specification.key == "gene_coverage":
        frame["plot_x"] = frame["gene_requested_coverage"]
    elif specification.key == "target_coverage":
        frame["plot_x"] = frame["target_requested_coverage"]
    else:
        frame["plot_x"] = harmonic_mean_retained_measurement(
            frame["gene_requested_coverage"],
            frame["target_requested_coverage"],
            frame["positive_retention"],
        )
    frame["plot_x"] = pd.to_numeric(frame["plot_x"], errors="coerce")
    if not frame["plot_x"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"{family} produced an invalid theoretical x coordinate")
    frame["curve_family"] = specification.key
    frame["curve_family_label"] = specification.label
    frame["plot_x_label"] = specification.x_label
    return frame


def _physical_condition_label(record: Mapping[str, object]) -> str:
    """Return an audit label made only from requested/theoretical settings."""

    return "/".join(
        (
            f"G{_percent(float(record['gene_requested_coverage']))}",
            f"T{_percent(float(record['target_requested_coverage']))}",
            f"R{_percent(float(record['positive_retention']))}",
        )
    )


def _curve_point_label(specification: CurveFamilySpec, value: float) -> str:
    prefix = {
        "positive_label_loss": "Loss ",
        "gene_coverage": "G",
        "target_coverage": "T",
        "combined_masks": "H",
    }[specification.key]
    return f"{prefix}{_percent(float(value))}"


def _finite_metric_values(frame: pd.DataFrame, metric: MetricSpec) -> pd.Series:
    values = pd.to_numeric(frame[metric.column], errors="coerce").astype(float)
    if metric.requires_reference_probability:
        values = values.where(_is_reference_probability(frame), np.nan)
    return values.where(np.isfinite(values), np.nan)


def prepare_v4_curve_comparison(
    per_repetition: pd.DataFrame,
    *,
    family: str,
    models: Sequence[str],
    metrics: Sequence[str] = DEFAULT_V4_METRICS,
) -> pd.DataFrame:
    """Build fail-closed, repetition-paired means for one curve family.

    A physical condition contributes to a repetition only when every requested
    model has a finite value for that metric.  Physical conditions mapping to
    the same x coordinate (notably symmetric combined-mask conditions) are
    averaged *inside* that repetition on this common support.  The final mean
    then uses only repetitions for which every requested model has a finite
    aggregate.  Rows for unavailable points and models are retained with a
    missing ``value`` and explicit matched/expected audit counts.
    """

    specification = _family_spec(family)
    metric_specs = _metric_specs(metrics)
    model_order = tuple(map(str, models))
    if not model_order or len(set(model_order)) != len(model_order):
        raise ValueError("models must contain distinct internal model identifiers")

    frame = prepare_v4_curve_table(
        per_repetition, family=family, metrics=metrics
    )
    base = _base_measurement_rows(per_repetition)
    facet_repetitions = (
        base[["facet", "facet_order", "repetition"]]
        .drop_duplicates()
        .sort_values(["facet_order", "repetition"], kind="stable")
    )
    conditions = (
        frame[[*_PHYSICAL_CONDITION_COLUMNS, "plot_x"]]
        .drop_duplicates()
        .sort_values(
            ["plot_x", *_PHYSICAL_CONDITION_COLUMNS], kind="stable"
        )
    )
    # V4 declares the same physical schedule for every simulation sharing
    # regime.  Taking the global condition catalog lets a fully failed
    # condition in one regime remain visible as N/A instead of disappearing.
    schedule = facet_repetitions.merge(conditions, how="cross")
    schedule["physical_condition"] = [
        _physical_condition_label(record)
        for record in schedule.to_dict("records")
    ]

    actual = frame.loc[frame["model"].isin(model_order)].copy()
    actual_keys = [
        "facet",
        "repetition",
        "model",
        *_PHYSICAL_CONDITION_COLUMNS,
    ]
    if actual.duplicated(actual_keys).any():
        duplicates = actual.loc[
            actual.duplicated(actual_keys, keep=False), actual_keys
        ].drop_duplicates()
        raise ValueError(
            "V4 curves require one per-repetition row per model and physical "
            f"condition; duplicates include {duplicates.head(3).to_dict('records')}"
        )

    point_keys = ["facet", "facet_order", "plot_x"]
    point_audit = (
        schedule.groupby(point_keys, dropna=False, observed=True, sort=False)
        .agg(
            n_expected_repetitions=("repetition", "nunique"),
            n_expected_physical_units=("repetition", "size"),
            n_physical_conditions_at_x=("physical_condition", "nunique"),
            physical_conditions_at_x=(
                "physical_condition",
                lambda values: "; ".join(dict.fromkeys(map(str, values))),
            ),
        )
        .reset_index()
    )
    model_frame = pd.DataFrame(
        {"model": model_order, "model_order": range(len(model_order))}
    )
    summary_records: list[pd.DataFrame] = []
    physical_unit_keys = [
        "facet",
        "facet_order",
        "repetition",
        "plot_x",
        *_PHYSICAL_CONDITION_COLUMNS,
    ]
    repetition_point_keys = [
        "facet",
        "facet_order",
        "plot_x",
        "repetition",
    ]

    for metric_index, metric in enumerate(metric_specs):
        cells = schedule.merge(model_frame, how="cross")
        actual_values = actual[
            [*actual_keys, metric.column, "probability_semantics"]
        ].copy()
        actual_values["metric_value"] = _finite_metric_values(
            actual_values, metric
        )
        actual_values = actual_values.drop(columns=[metric.column])
        cells = cells.merge(
            actual_values,
            on=actual_keys,
            how="left",
            validate="one_to_one",
        )
        cells["metric_value"] = pd.to_numeric(
            cells["metric_value"], errors="coerce"
        )
        cells["finite"] = np.isfinite(cells["metric_value"])

        complete_units = (
            cells.groupby(
                physical_unit_keys,
                dropna=False,
                observed=True,
                sort=False,
            )["finite"]
            .sum()
            .eq(len(model_order))
            .rename("physical_unit_complete")
            .reset_index()
        )
        expected_per_repetition = (
            schedule.groupby(
                repetition_point_keys,
                dropna=False,
                observed=True,
                sort=False,
            )
            .size()
            .rename("n_expected_physical_conditions")
            .reset_index()
        )
        complete_per_repetition = (
            complete_units.loc[complete_units["physical_unit_complete"]]
            .groupby(
                repetition_point_keys,
                dropna=False,
                observed=True,
                sort=False,
            )
            .size()
            .rename("n_complete_physical_conditions")
            .reset_index()
        )
        repetition_support = expected_per_repetition.merge(
            complete_per_repetition,
            on=repetition_point_keys,
            how="left",
            validate="one_to_one",
        )
        repetition_support["n_complete_physical_conditions"] = (
            repetition_support["n_complete_physical_conditions"]
            .fillna(0)
            .astype(int)
        )
        repetition_support["all_physical_conditions_complete"] = (
            repetition_support["n_complete_physical_conditions"].eq(
                repetition_support["n_expected_physical_conditions"]
            )
        )
        cells = cells.merge(
            complete_units,
            on=physical_unit_keys,
            how="left",
            validate="many_to_one",
        ).merge(
            repetition_support[
                [*repetition_point_keys, "all_physical_conditions_complete"]
            ],
            on=repetition_point_keys,
            how="left",
            validate="many_to_one",
        )
        matched_cells = cells.loc[
            cells["physical_unit_complete"]
            & cells["all_physical_conditions_complete"]
        ].copy()
        per_repetition = (
            matched_cells.groupby(
                [*repetition_point_keys, "model", "model_order"],
                dropna=False,
                observed=True,
                sort=False,
            )["metric_value"]
            .mean()
            .reset_index()
        )
        repetition_complete = (
            per_repetition.groupby(
                repetition_point_keys,
                dropna=False,
                observed=True,
                sort=False,
            )
            .agg(
                n_models=("model", "nunique"),
                n_finite=("metric_value", lambda values: int(np.isfinite(values).sum())),
            )
            .reset_index()
        )
        repetition_complete["repetition_complete"] = (
            repetition_complete["n_models"].eq(len(model_order))
            & repetition_complete["n_finite"].eq(len(model_order))
        )
        per_repetition = per_repetition.merge(
            repetition_complete[
                [*repetition_point_keys, "repetition_complete"]
            ],
            on=repetition_point_keys,
            how="left",
            validate="many_to_one",
        )
        per_repetition = per_repetition.loc[
            per_repetition["repetition_complete"]
        ]

        means = (
            per_repetition.groupby(
                [*point_keys, "model", "model_order"],
                dropna=False,
                observed=True,
                sort=False,
            )["metric_value"]
            .mean()
            .rename("value")
            .reset_index()
        )
        matched_repetitions = (
            per_repetition[repetition_point_keys]
            .drop_duplicates()
            .groupby(point_keys, dropna=False, observed=True, sort=False)
            .size()
            .rename("n_matched_repetitions")
            .reset_index()
        )
        used_physical_units = complete_units.merge(
            repetition_support[
                [*repetition_point_keys, "all_physical_conditions_complete"]
            ],
            on=repetition_point_keys,
            how="left",
            validate="many_to_one",
        )
        matched_physical = (
            used_physical_units.loc[
                used_physical_units["physical_unit_complete"]
                & used_physical_units["all_physical_conditions_complete"]
            ]
            .groupby(point_keys, dropna=False, observed=True, sort=False)
            .size()
            .rename("n_matched_physical_units")
            .reset_index()
        )
        model_finite = (
            cells.groupby(
                [*point_keys, "model", "model_order"],
                dropna=False,
                observed=True,
                sort=False,
            )["finite"]
            .sum()
            .rename("n_model_finite_physical_units")
            .reset_index()
        )

        metric_summary = point_audit.merge(model_frame, how="cross")
        metric_summary = metric_summary.merge(
            means,
            on=[*point_keys, "model", "model_order"],
            how="left",
            validate="one_to_one",
        )
        metric_summary = metric_summary.merge(
            matched_repetitions,
            on=point_keys,
            how="left",
            validate="many_to_one",
        ).merge(
            matched_physical,
            on=point_keys,
            how="left",
            validate="many_to_one",
        ).merge(
            model_finite,
            on=[*point_keys, "model", "model_order"],
            how="left",
            validate="one_to_one",
        )
        for column in (
            "n_matched_repetitions",
            "n_matched_physical_units",
            "n_model_finite_physical_units",
        ):
            metric_summary[column] = metric_summary[column].fillna(0).astype(int)

        incomplete_records = []
        for key, group in metric_summary.groupby(
            point_keys, dropna=False, observed=True, sort=False
        ):
            expected = int(group["n_expected_physical_units"].iloc[0])
            incomplete = group.loc[
                group["n_model_finite_physical_units"].lt(expected)
            ]
            incomplete_records.append(
                {
                    **dict(zip(point_keys, key if isinstance(key, tuple) else (key,))),
                    "incomplete_models": "; ".join(
                        f"{MODEL_DISPLAY_LABELS.get(row.model, row.model)} "
                        f"({int(row.n_model_finite_physical_units)}/{expected})"
                        for row in incomplete.itertuples()
                    ),
                }
            )
        metric_summary = metric_summary.merge(
            pd.DataFrame(incomplete_records),
            on=point_keys,
            how="left",
            validate="many_to_one",
        )
        metric_summary["metric"] = metric.key
        metric_summary["metric_column"] = metric.column
        metric_summary["metric_label"] = metric.label
        metric_summary["metric_order"] = metric_index
        metric_summary["higher_is_better"] = metric.higher_is_better
        metric_summary["n_rows_averaged"] = metric_summary[
            "n_matched_repetitions"
        ]
        metric_summary["availability"] = np.select(
            [
                metric_summary["n_matched_repetitions"].eq(0),
                metric_summary["n_matched_repetitions"].lt(
                    metric_summary["n_expected_repetitions"]
                )
                | metric_summary["n_matched_physical_units"].lt(
                    metric_summary["n_expected_physical_units"]
                ),
            ],
            ["N/A", "partial"],
            default="complete",
        )
        summary_records.append(metric_summary)

    result = pd.concat(summary_records, ignore_index=True)
    result["curve_family"] = specification.key
    result["curve_family_label"] = specification.label
    result["plot_x_label"] = specification.x_label
    return result.sort_values(
        ["facet_order", "metric_order", "plot_x", "model_order"],
        ascending=[True, True, not specification.descending, True],
        kind="stable",
    ).reset_index(drop=True)


def _model_style(model: str, index: int) -> dict[str, object]:
    if model in MODEL_STYLES:
        return dict(MODEL_STYLES[model])
    palette = plt.get_cmap("tab10")
    return {
        "color": palette(index % palette.N),
        "marker": "o",
        "linestyle": "--",
    }


def _set_curve_x_axis(axis, frame: pd.DataFrame, specification: CurveFamilySpec):
    values = np.sort(pd.to_numeric(frame["plot_x"], errors="coerce").dropna().unique())
    if specification.key == "positive_label_loss":
        upper = float(values.max()) if len(values) else 1.0
        margin = 0.025 if upper > 0 else 0.0
        axis.set_xlim(-margin, min(1.0, upper + margin))
    else:
        axis.set_xlim(1.0, 0.0)
    if 0 < len(values) <= 8:
        axis.set_xticks(values)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlabel(specification.x_label)


def _report_curve_comparison(
    comparison: pd.DataFrame,
    *,
    specification: CurveFamilySpec,
    model_order: Sequence[str],
) -> None:
    point_columns = [
        "facet",
        "plot_x",
        "metric",
        "metric_label",
        "n_expected_repetitions",
        "n_matched_repetitions",
        "n_expected_physical_units",
        "n_matched_physical_units",
        "n_physical_conditions_at_x",
        "physical_conditions_at_x",
        "incomplete_models",
        "availability",
    ]
    points = comparison[point_columns].drop_duplicates()
    model_labels = ", ".join(
        MODEL_DISPLAY_LABELS.get(model, model) for model in model_order
    )
    available = int(points["n_matched_repetitions"].gt(0).sum())
    print(
        f"[V4 paired curves] {specification.label}: exact models "
        f"[{model_labels}]; {available}/{len(points)} facet/x/metric points "
        "have at least one repetition complete for every model.",
        flush=True,
    )
    collisions = points.loc[
        points["n_physical_conditions_at_x"].gt(1),
        [
            "facet",
            "plot_x",
            "n_physical_conditions_at_x",
            "physical_conditions_at_x",
        ],
    ].drop_duplicates()
    for row in collisions.itertuples():
        print(
            f"[V4 paired curves] {row.facet}, "
            f"{_curve_point_label(specification, row.plot_x)} aggregates "
            f"{int(row.n_physical_conditions_at_x)} physical conditions inside "
            f"each repetition: {row.physical_conditions_at_x}.",
            flush=True,
        )
    gaps = points.loc[points["availability"].ne("complete")]
    for row in gaps.itertuples():
        print(
            f"[V4 paired curves] {row.availability}: {row.facet}, "
            f"{_curve_point_label(specification, row.plot_x)}, "
            f"{row.metric_label}; matched repetitions "
            f"{int(row.n_matched_repetitions)}/"
            f"{int(row.n_expected_repetitions)}, matched physical units "
            f"{int(row.n_matched_physical_units)}/"
            f"{int(row.n_expected_physical_units)}; incomplete models: "
            f"{row.incomplete_models or 'none'}.",
            flush=True,
        )


def plot_v4_metric_curves(
    per_repetition: pd.DataFrame,
    *,
    family: str,
    models: Sequence[str],
    metrics: Sequence[str] = DEFAULT_V4_METRICS,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot one V4 family with sharing regimes as rows and metrics as columns."""

    specification = _family_spec(family)
    metric_specs = _metric_specs(metrics)
    model_order = tuple(map(str, models))
    if not model_order or len(set(model_order)) != len(model_order):
        raise ValueError("models must contain distinct internal model identifiers")
    frame = prepare_v4_curve_table(per_repetition, family=family, metrics=metrics)
    comparison = prepare_v4_curve_comparison(
        per_repetition,
        family=family,
        models=model_order,
        metrics=metrics,
    )
    facets = (
        comparison[["facet", "facet_order"]]
        .drop_duplicates()
        .sort_values("facet_order", kind="stable")["facet"]
        .tolist()
    )
    figure, axes = plt.subplots(
        len(facets),
        len(metric_specs),
        figsize=(3.8 * len(metric_specs), 3.25 * len(facets) + 0.55),
        squeeze=False,
        constrained_layout=True,
    )
    legend_handles = {
        model: Line2D(
            [],
            [],
            linewidth=1.8,
            markersize=4.5,
            markeredgewidth=0.7,
            label=MODEL_DISPLAY_LABELS.get(model, model),
            **_model_style(model, index),
        )
        for index, model in enumerate(model_order)
    }
    for row_index, facet in enumerate(facets):
        facet_frame = frame.loc[frame["facet"].eq(facet)]
        for column_index, metric in enumerate(metric_specs):
            axis = axes[row_index, column_index]
            summary = comparison.loc[
                comparison["facet"].eq(facet)
                & comparison["metric"].eq(metric.key)
            ]
            plotted = False
            for model_index, model in enumerate(model_order):
                selected = summary.loc[summary["model"].eq(model)].sort_values(
                    "plot_x", ascending=not specification.descending
                )
                selected = selected.loc[np.isfinite(selected["value"])]
                if selected.empty:
                    continue
                style = _model_style(model, model_index)
                axis.plot(
                    selected["plot_x"],
                    selected["value"],
                    linewidth=1.8,
                    markersize=4.5,
                    markeredgewidth=0.7,
                    label=MODEL_DISPLAY_LABELS.get(model, model),
                    zorder=3,
                    **style,
                )
                plotted = True
            if not plotted:
                axis.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
                )
            else:
                availability = summary[
                    [
                        "plot_x",
                        "n_matched_repetitions",
                        "n_expected_repetitions",
                        "availability",
                    ]
                ].drop_duplicates()
                for point in availability.loc[
                    availability["availability"].ne("complete")
                ].itertuples():
                    label = (
                        "N/A"
                        if int(point.n_matched_repetitions) == 0
                        else f"n={int(point.n_matched_repetitions)}/"
                        f"{int(point.n_expected_repetitions)}"
                    )
                    axis.text(
                        float(point.plot_x),
                        0.025,
                        label,
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#777777",
                    )
            _set_curve_x_axis(axis, facet_frame, specification)
            axis.set_ylabel(metric.label)
            axis.grid(axis="y", color="#E6E6E6", linewidth=0.65)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                axis.set_title(metric.label)
        if facet != "all":
            axes[row_index, 0].annotate(
                facet,
                xy=(-0.23, 0.5),
                xycoords="axes fraction",
                ha="right",
                va="center",
                fontweight="bold",
            )
    handles = [legend_handles[model] for model in model_order]
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(handles),
        frameon=False,
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.88))
    figure.suptitle(title or specification.label, fontsize=13, y=0.985)
    # Notebook callers can save this exact audit table next to the figure.  It
    # also makes incomplete support inspectable without reverse-engineering
    # matplotlib artists.
    figure.v4_curve_audit = comparison.copy()
    _report_curve_comparison(
        comparison, specification=specification, model_order=model_order
    )
    return figure, axes


def plot_v4_metric_grid(
    per_repetition: pd.DataFrame,
    *,
    family: str,
    models: Sequence[str],
    metrics: Sequence[str] = DEFAULT_V4_METRICS,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Public notebook alias for :func:`plot_v4_metric_curves`."""

    return plot_v4_metric_curves(
        per_repetition,
        family=family,
        models=models,
        metrics=metrics,
        title=title,
    )


def _percent(value: float) -> str:
    percent = 100.0 * float(value)
    if np.isclose(percent, round(percent), rtol=0.0, atol=1e-9):
        return f"{round(percent):.0f}%"
    return f"{percent:.1f}%"


def _condition_columns(
    frame: pd.DataFrame, specification: CurveFamilySpec
) -> pd.DataFrame:
    coordinates = [
        "gene_requested_coverage",
        "target_requested_coverage",
        "positive_retention",
        "plot_x",
    ]
    conditions = frame[coordinates].drop_duplicates().copy()
    if specification.key == "positive_label_loss":
        conditions["condition"] = [
            f"positive_loss_{value:.12g}" for value in conditions["plot_x"]
        ]
        conditions["condition_label"] = [
            f"Loss {_percent(value)}" for value in conditions["plot_x"]
        ]
        conditions = conditions.sort_values("plot_x", ascending=True, kind="stable")
    elif specification.key == "gene_coverage":
        conditions["condition"] = [
            f"gene_{value:.12g}"
            for value in conditions["gene_requested_coverage"]
        ]
        conditions["condition_label"] = [
            f"G{_percent(value)}"
            for value in conditions["gene_requested_coverage"]
        ]
        conditions = conditions.sort_values(
            "gene_requested_coverage", ascending=False, kind="stable"
        )
    elif specification.key == "target_coverage":
        conditions["condition"] = [
            f"target_{value:.12g}"
            for value in conditions["target_requested_coverage"]
        ]
        conditions["condition_label"] = [
            f"T{_percent(value)}"
            for value in conditions["target_requested_coverage"]
        ]
        conditions = conditions.sort_values(
            "target_requested_coverage", ascending=False, kind="stable"
        )
    else:
        conditions["condition"] = [
            f"combined_g{gene:.12g}_t{target:.12g}_p{retention:.12g}"
            for gene, target, retention in zip(
                conditions["gene_requested_coverage"],
                conditions["target_requested_coverage"],
                conditions["positive_retention"],
            )
        ]
        conditions["condition_label"] = [
            f"H{_percent(harmonic)}" for harmonic in conditions["plot_x"]
        ]
        conditions = conditions.sort_values(
            [
                "plot_x",
                "gene_requested_coverage",
                "target_requested_coverage",
                "positive_retention",
            ],
            ascending=[False, False, False, False],
            kind="stable",
        )
    conditions["condition_order"] = np.arange(len(conditions), dtype=int)
    return conditions


def _partition_funky_scenarios(
    per_repetition: pd.DataFrame,
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Assign each physical V4 scenario to exactly one funky column family."""

    metric_specs = _metric_specs(metrics)
    frame = _base_measurement_rows(per_repetition)
    missing_metrics = sorted(
        {metric.column for metric in metric_specs}.difference(frame.columns)
    )
    if missing_metrics:
        raise ValueError(
            f"per_repetition.csv lacks requested V4 metrics: {missing_metrics}"
        )
    for metric in metric_specs:
        frame[metric.column] = pd.to_numeric(frame[metric.column], errors="coerce")

    gene = frame["gene_requested_coverage"].to_numpy(dtype=float)
    target = frame["target_requested_coverage"].to_numpy(dtype=float)
    retention = frame["positive_retention"].to_numpy(dtype=float)
    full_gene = np.isclose(gene, 1.0)
    full_target = np.isclose(target, 1.0)
    full_retention = np.isclose(retention, 1.0)

    # Priority is scientifically meaningful: the all-full control belongs only
    # to the positive-loss family, followed by the two literal one-factor
    # coverage slices.  No physical triple is copied across families.
    positive = full_gene & full_target
    gene_only = (~full_gene) & full_target & full_retention
    target_only = full_gene & (~full_target) & full_retention
    selected = positive | gene_only | target_only
    frame = frame.loc[selected].copy()
    positive = positive[selected]
    gene_only = gene_only[selected]
    target_only = target_only[selected]
    frame["family"] = np.select(
        [positive, gene_only, target_only],
        [
            "positive_label_loss",
            "gene_coverage",
            "target_coverage",
        ],
        default="",
    )
    frame["family_label"] = frame["family"].map(FUNKY_FAMILY_LABELS)

    records = []
    coordinates = (
        frame[
            [
                "family",
                "family_label",
                "gene_requested_coverage",
                "target_requested_coverage",
                "positive_retention",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    for family_index, family in enumerate(DEFAULT_V4_FUNKY_FAMILIES):
        family_coordinates = coordinates.loc[coordinates["family"].eq(family)].copy()
        if family == "positive_label_loss":
            family_coordinates["condition"] = [
                f"positive_loss_{1.0 - value:.12g}"
                for value in family_coordinates["positive_retention"]
            ]
            family_coordinates["condition_label"] = [
                f"Loss {_percent(1.0 - value)}"
                for value in family_coordinates["positive_retention"]
            ]
            family_coordinates = family_coordinates.sort_values(
                "positive_retention", ascending=False, kind="stable"
            )
        elif family == "gene_coverage":
            family_coordinates["condition"] = [
                f"gene_{value:.12g}"
                for value in family_coordinates["gene_requested_coverage"]
            ]
            family_coordinates["condition_label"] = [
                f"G{_percent(value)}"
                for value in family_coordinates["gene_requested_coverage"]
            ]
            family_coordinates = family_coordinates.sort_values(
                "gene_requested_coverage", ascending=False, kind="stable"
            )
        elif family == "target_coverage":
            family_coordinates["condition"] = [
                f"target_{value:.12g}"
                for value in family_coordinates["target_requested_coverage"]
            ]
            family_coordinates["condition_label"] = [
                f"T{_percent(value)}"
                for value in family_coordinates["target_requested_coverage"]
            ]
            family_coordinates = family_coordinates.sort_values(
                "target_requested_coverage", ascending=False, kind="stable"
            )
        family_coordinates["family_order"] = family_index
        family_coordinates["condition_order"] = np.arange(
            len(family_coordinates), dtype=int
        )
        records.append(family_coordinates)
    catalog = pd.concat(records, ignore_index=True)
    join_columns = [
        "family",
        "family_label",
        "gene_requested_coverage",
        "target_requested_coverage",
        "positive_retention",
    ]
    frame = frame.merge(
        catalog,
        on=join_columns,
        how="left",
        validate="many_to_one",
    )
    scenario_columns = [
        "gene_requested_coverage",
        "target_requested_coverage",
        "positive_retention",
    ]
    assigned = frame[
        [*scenario_columns, "family", "condition"]
    ].drop_duplicates()
    if assigned.duplicated(scenario_columns).any():
        raise RuntimeError("A physical V4 scenario was assigned to multiple funky families")
    return frame


def prepare_v4_funky_table(
    per_repetition: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_V4_METRICS,
    models: Sequence[str] = FUNKY_MODELS,
    families: Sequence[str] = DEFAULT_V4_FUNKY_FAMILIES,
) -> pd.DataFrame:
    """Create a repetition-paired raw model-by-scenario scorecard.

    A physical metric column uses the intersection of repetitions having a
    finite value for *every* requested model.  Every model's raw ``value`` is
    averaged on that identical support.  If the intersection is empty, every
    model in the physical column remains N/A.  Normalization is delayed until
    the exact displayed model set and simulation sharing facet are known.
    """

    metric_specs = _metric_specs(metrics)
    model_order = tuple(map(str, models))
    if not model_order or len(set(model_order)) != len(model_order):
        raise ValueError("models must contain distinct internal model identifiers")
    family_order = tuple(map(str, families))
    if not family_order or len(set(family_order)) != len(family_order):
        raise ValueError("families must contain distinct V4 funky-family keys")
    unknown_families = sorted(set(family_order).difference(DEFAULT_V4_FUNKY_FAMILIES))
    if unknown_families:
        raise ValueError(f"Unknown V4 funky families: {unknown_families}")
    frame = _partition_funky_scenarios(
        per_repetition,
        metrics=metrics,
    )
    frame = frame.loc[frame["family"].isin(family_order)].copy()
    remap_order = {family: index for index, family in enumerate(family_order)}
    frame["family_order"] = frame["family"].map(remap_order).astype(int)
    catalog = (
        frame[
            [
                "family",
                "family_label",
                "family_order",
                "condition",
                "condition_label",
                "condition_order",
                "gene_requested_coverage",
                "target_requested_coverage",
                "positive_retention",
            ]
        ]
        .drop_duplicates(["family", "condition"])
        .sort_values(["family_order", "condition_order"], kind="stable")
        .reset_index(drop=True)
    )
    facet_repetitions = (
        frame[["facet", "facet_order", "repetition"]]
        .drop_duplicates()
        .sort_values(["facet_order", "repetition"], kind="stable")
    )
    facets = facet_repetitions[["facet", "facet_order"]].drop_duplicates()
    schedule = facet_repetitions.merge(catalog, how="cross")
    skeleton_records = []
    for facet_record in facets.to_dict("records"):
        for model_index, model in enumerate(model_order):
            for condition in catalog.to_dict("records"):
                for metric_index, metric in enumerate(metric_specs):
                    skeleton_records.append(
                        {
                            **facet_record,
                            "model": model,
                            "model_order": model_index,
                            "model_label": MODEL_DISPLAY_LABELS.get(model, model),
                            **condition,
                            "metric": metric.key,
                            "metric_column": metric.column,
                            "metric_label": metric.label,
                            "metric_order": metric_index,
                            "higher_is_better": metric.higher_is_better,
                            "requires_reference_probability": (
                                metric.requires_reference_probability
                            ),
                        }
                    )
    skeleton = pd.DataFrame(skeleton_records)

    actual = frame.loc[frame["model"].isin(model_order)].copy()
    actual_keys = ["facet", "repetition", "model", "family", "condition"]
    if actual.duplicated(actual_keys).any():
        duplicates = actual.loc[
            actual.duplicated(actual_keys, keep=False), actual_keys
        ].drop_duplicates()
        raise ValueError(
            "V4 funkyheatmap requires one per-repetition row per model and "
            f"physical condition; duplicates include "
            f"{duplicates.head(3).to_dict('records')}"
        )

    model_frame = pd.DataFrame(
        {"model": model_order, "model_order": range(len(model_order))}
    )
    physical_keys = [
        "facet",
        "facet_order",
        "family",
        "condition",
        "repetition",
    ]
    column_keys = [
        "facet",
        "facet_order",
        "family",
        "condition",
    ]
    observed_records: list[pd.DataFrame] = []
    for metric in metric_specs:
        cells = schedule.merge(model_frame, how="cross")
        actual_values = actual[
            [*actual_keys, metric.column, "probability_semantics"]
        ].copy()
        actual_values["metric_value"] = _finite_metric_values(
            actual_values, metric
        )
        actual_values = actual_values.drop(columns=[metric.column])
        cells = cells.merge(
            actual_values,
            on=actual_keys,
            how="left",
            validate="one_to_one",
        )
        cells["metric_value"] = pd.to_numeric(
            cells["metric_value"], errors="coerce"
        )
        cells["finite"] = np.isfinite(cells["metric_value"])
        complete = (
            cells.groupby(
                physical_keys,
                dropna=False,
                observed=True,
                sort=False,
            )["finite"]
            .sum()
            .eq(len(model_order))
            .rename("repetition_complete")
            .reset_index()
        )
        cells = cells.merge(
            complete,
            on=physical_keys,
            how="left",
            validate="many_to_one",
        )
        matched = cells.loc[cells["repetition_complete"]].copy()
        means = (
            matched.groupby(
                [*column_keys, "model", "model_order"],
                dropna=False,
                observed=True,
                sort=False,
            )["metric_value"]
            .mean()
            .rename("value")
            .reset_index()
        )
        matched_counts = (
            complete.loc[complete["repetition_complete"]]
            .groupby(column_keys, dropna=False, observed=True, sort=False)
            .size()
            .rename("n_matched_repetitions")
            .reset_index()
        )
        expected_counts = (
            schedule.groupby(column_keys, dropna=False, observed=True, sort=False)
            ["repetition"]
            .nunique()
            .rename("n_expected_repetitions")
            .reset_index()
        )
        model_finite = (
            cells.groupby(
                [*column_keys, "model", "model_order"],
                dropna=False,
                observed=True,
                sort=False,
            )["finite"]
            .sum()
            .rename("n_model_finite_repetitions")
            .reset_index()
        )
        semantics = (
            cells.loc[cells["probability_semantics"].notna()]
            .groupby(
                [*column_keys, "model", "model_order"],
                dropna=False,
                observed=True,
                sort=False,
            )["probability_semantics"]
            .agg(lambda values: tuple(sorted(set(map(str, values)))))
            .reset_index()
        )
        ambiguous = semantics.loc[semantics["probability_semantics"].map(len).gt(1)]
        if not ambiguous.empty:
            raise ValueError(
                "Ambiguous probability semantics for V4 funky cells: "
                f"{ambiguous.head(3).to_dict('records')}"
            )
        semantics["probability_semantics"] = semantics[
            "probability_semantics"
        ].map(lambda values: values[0] if values else "not_available")

        metric_result = skeleton.loc[skeleton["metric"].eq(metric.key)].copy()
        metric_result = metric_result.merge(
            means,
            on=[*column_keys, "model", "model_order"],
            how="left",
            validate="one_to_one",
        ).merge(
            expected_counts,
            on=column_keys,
            how="left",
            validate="many_to_one",
        ).merge(
            matched_counts,
            on=column_keys,
            how="left",
            validate="many_to_one",
        ).merge(
            model_finite,
            on=[*column_keys, "model", "model_order"],
            how="left",
            validate="one_to_one",
        ).merge(
            semantics,
            on=[*column_keys, "model", "model_order"],
            how="left",
            validate="one_to_one",
        )
        for column in (
            "n_matched_repetitions",
            "n_model_finite_repetitions",
        ):
            metric_result[column] = metric_result[column].fillna(0).astype(int)
        metric_result["probability_semantics"] = metric_result[
            "probability_semantics"
        ].fillna("not_available")
        incomplete_records = []
        for key, group in metric_result.groupby(
            column_keys, dropna=False, observed=True, sort=False
        ):
            expected = int(group["n_expected_repetitions"].iloc[0])
            incomplete = group.loc[
                group["n_model_finite_repetitions"].lt(expected)
            ]
            incomplete_records.append(
                {
                    **dict(zip(column_keys, key)),
                    "incomplete_models": "; ".join(
                        f"{MODEL_DISPLAY_LABELS.get(row.model, row.model)} "
                        f"({int(row.n_model_finite_repetitions)}/{expected})"
                        for row in incomplete.itertuples()
                    ),
                }
            )
        metric_result = metric_result.merge(
            pd.DataFrame(incomplete_records),
            on=column_keys,
            how="left",
            validate="many_to_one",
        )
        metric_result["n_rows_averaged"] = metric_result[
            "n_matched_repetitions"
        ]
        metric_result["availability"] = np.select(
            [
                metric_result["n_matched_repetitions"].eq(0),
                metric_result["n_matched_repetitions"].lt(
                    metric_result["n_expected_repetitions"]
                ),
            ],
            ["N/A", "partial"],
            default="complete",
        )
        observed_records.append(metric_result)

    result = pd.concat(observed_records, ignore_index=True)
    return result.sort_values(
        [
            "facet_order",
            "family_order",
            "condition_order",
            "metric_order",
            "model_order",
        ],
        kind="stable",
    ).reset_index(drop=True)


def normalize_v4_funky_table(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize utility inside each sharing/condition/metric cell.

    Losses are negated before scaling.  Every non-tied comparison with at least
    two finite models therefore contains an exact zero and one.  A genuine tie
    receives 0.5 for every tied model; ties are never broken by row order.
    """

    required = (
        "facet",
        "family",
        "condition",
        "metric",
        "higher_is_better",
        "value",
    )
    result = _as_frame(scorecard, name="V4 funky scorecard", required=required)
    result["desirability"] = np.nan
    groups = result.groupby(
        ["facet", "family", "condition", "metric"],
        dropna=False,
        observed=True,
        sort=False,
    ).groups
    for group_key, indices in groups.items():
        indices = pd.Index(indices)
        values = pd.to_numeric(result.loc[indices, "value"], errors="coerce")
        finite_indices = values.index[np.isfinite(values.to_numpy(dtype=float))]
        # A funky column is an exact-model comparison.  Never rescale over a
        # smaller surviving subset when any requested model is unavailable.
        if len(finite_indices) != len(indices):
            continue
        directions = result.loc[finite_indices, "higher_is_better"].astype(bool).unique()
        if len(directions) != 1:
            raise ValueError(f"Inconsistent metric direction in funky cell {group_key}")
        utility = values.loc[finite_indices].astype(float)
        if not bool(directions[0]):
            utility = -utility
        lower = float(utility.min())
        upper = float(utility.max())
        span = upper - lower
        if span == 0.0:
            normalized = pd.Series(0.5, index=finite_indices, dtype=float)
        else:
            normalized = (utility - lower) / span
        result.loc[finite_indices, "desirability"] = normalized.clip(0.0, 1.0)
    return result


def build_funkyheatmappy_inputs(scorecard: pd.DataFrame) -> dict[str, object]:
    """Build native funkyheatmappy inputs without importing the package."""

    required = (
        "facet",
        "facet_order",
        "model",
        "model_order",
        "model_label",
        "family",
        "family_label",
        "family_order",
        "condition",
        "condition_label",
        "condition_order",
        "metric",
        "metric_label",
        "metric_order",
        "value",
        "higher_is_better",
    )
    normalized = normalize_v4_funky_table(
        _as_frame(scorecard, name="V4 funky scorecard", required=required)
    )
    catalog = (
        normalized[
            [
                "family",
                "family_label",
                "family_order",
                "condition",
                "condition_label",
                "condition_order",
                "metric",
                "metric_label",
                "metric_order",
            ]
        ]
        .drop_duplicates(["family", "condition", "metric"])
        .sort_values(
            ["family_order", "condition_order", "metric_order"], kind="stable"
        )
        .reset_index(drop=True)
    )
    catalog["plot_id"] = [f"metric_{index:03d}" for index in range(len(catalog))]
    condition_keys = list(
        dict.fromkeys(zip(catalog["family"], catalog["condition"]))
    )
    condition_groups = {
        key: f"condition_{index:03d}" for index, key in enumerate(condition_keys)
    }
    catalog["group_id"] = [
        condition_groups[(family, condition)]
        for family, condition in zip(catalog["family"], catalog["condition"])
    ]
    normalized = normalized.merge(
        catalog[["family", "condition", "metric", "plot_id"]],
        on=["family", "condition", "metric"],
        how="left",
        validate="many_to_one",
    )

    available = normalized.groupby("plot_id", observed=True)["desirability"].apply(
        lambda values: np.isfinite(pd.to_numeric(values, errors="coerce")).any()
    )
    dropped_plot_ids = tuple(available.index[~available].astype(str))
    dropped_catalog = catalog.loc[catalog["plot_id"].isin(dropped_plot_ids)].copy()
    dropped_columns = tuple(
        f"{record['family_label']} / {record['condition_label']} / "
        f"{record['metric_label']}"
        for record in dropped_catalog.to_dict("records")
    )
    kept_plot_ids = tuple(available.index[available].astype(str))
    if not kept_plot_ids:
        raise ValueError("Every configured funkyheatmap column is unavailable")
    catalog = catalog.loc[catalog["plot_id"].isin(kept_plot_ids)].copy()
    normalized = normalized.loc[normalized["plot_id"].isin(kept_plot_ids)].copy()

    rows = (
        scorecard[["facet", "facet_order", "model", "model_order", "model_label"]]
        .drop_duplicates(["facet", "model"])
        .sort_values(["facet_order", "model_order"], kind="stable")
        .reset_index(drop=True)
    )
    rows["id"] = [f"row_{index:03d}" for index in range(len(rows))]
    normalized = normalized.merge(
        rows[["facet", "model", "id"]],
        on=["facet", "model"],
        how="left",
        validate="many_to_one",
    )
    if normalized.duplicated(["id", "plot_id"]).any():
        raise ValueError("Duplicate V4 funky cells cannot be pivoted")
    wide = normalized.pivot(index="id", columns="plot_id", values="desirability")
    wide = wide.reindex(index=rows["id"], columns=catalog["plot_id"])
    # Keep NaN as NaN.  For ``funkyrect``, zero is a legitimate small circle,
    # not a blank cell.
    plot_data = wide.reset_index().merge(
        rows[["id", "model_label"]], on="id", how="left", validate="one_to_one"
    )
    plot_data = plot_data[["id", "model_label", *catalog["plot_id"].tolist()]]
    plot_data.columns.name = None

    if rows["facet"].nunique() > 1:
        group_lookup = {
            facet: f"row_group_{index:03d}"
            for index, facet in enumerate(rows["facet"].drop_duplicates())
        }
        row_info = pd.DataFrame(
            {"group": [group_lookup[facet] for facet in rows["facet"]]},
            index=pd.Index(rows["id"].astype(str), name="id"),
        )
        row_groups = pd.DataFrame(
            [
                {"level1": facet, "group": group_id}
                for facet, group_id in group_lookup.items()
            ]
        )
    else:
        row_info = pd.DataFrame(
            {"group": pd.Series([np.nan] * len(rows), dtype=object)},
            index=pd.Index(rows["id"].astype(str), name="id"),
        )
        row_groups = None

    column_records = [
        {
            "id": "model_label",
            "group": np.nan,
            "name": "",
            "geom": "text",
            "palette": np.nan,
            "width": 7.0,
            "legend": False,
            "ha": 0.0,
        }
    ]
    for record in catalog.to_dict("records"):
        column_records.append(
            {
                "id": record["plot_id"],
                "group": record["group_id"],
                "name": record["metric_label"],
                "geom": "funkyrect",
                "palette": "metric_blue",
                "width": 1.0,
                "legend": False,
                "ha": np.nan,
            }
        )
    column_info = pd.DataFrame(column_records)
    # funkyheatmappy 0.7 mutates these columns in-place.  Object dtype avoids
    # pandas' LossySetitemError, and explicit data-column ids avoid its mixed
    # string/NaN validator bug.
    column_info["id_color"] = column_info["id"].astype(object)
    column_info["id_size"] = column_info["id"].astype(object)
    column_info.index = pd.Index(column_info["id"].astype(str), name="id")

    condition_catalog = catalog.drop_duplicates(
        ["family", "condition"], keep="first"
    )
    column_groups = pd.DataFrame(
        [
            {
                "level1": str(record["family_label"]),
                "level2": str(record["condition_label"]),
                "group": str(record["group_id"]),
                "palette": "metric_blue",
            }
            for record in condition_catalog.to_dict("records")
        ]
    )
    return {
        "plot_data": plot_data,
        "column_info": column_info,
        "row_info": row_info,
        "column_groups": column_groups,
        "row_groups": row_groups,
        "palettes": {"metric_blue": [FUNKY_BLUE]},
        # funkyheatmappy 0.7 fails with zero enabled legends, so keep one
        # compact shape legend.  The palette is deliberately one color.
        "legends": [
            {
                "title": "Relative performance (larger is better)",
                "palette": "metric_blue",
                "enabled": True,
                "geom": "funkyrect",
                "labels": ["minimum", "", "maximum"],
                "size": [0.0, 0.5, 1.0],
            }
        ],
        "catalog": catalog.reset_index(drop=True),
        "normalized_scorecard": normalized.reset_index(drop=True),
        "dropped_all_na_plot_ids": dropped_plot_ids,
        "dropped_all_na_columns": dropped_columns,
    }


def render_v4_funkyheatmap(
    scorecard: pd.DataFrame,
    *,
    title: str,
) -> Figure:
    """Render the four-model scorecard with native funkyheatmappy 0.7."""

    try:
        import funkyheatmappy
        from funkyheatmappy.position_arguments import position_arguments
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Install funkyheatmappy==0.7.0 before rendering the V4 scorecard"
        ) from error

    inputs = build_funkyheatmappy_inputs(scorecard)
    audit_columns = [
        "facet",
        "family_label",
        "condition_label",
        "metric_label",
        "n_matched_repetitions",
        "n_expected_repetitions",
        "incomplete_models",
        "availability",
    ]
    audit = scorecard[audit_columns].drop_duplicates()
    complete = int(audit["availability"].eq("complete").sum())
    print(
        f"[V4 paired funkyheatmap] {complete}/{len(audit)} "
        "sharing/condition/metric columns have all requested models on "
        "complete matched repetitions.",
        flush=True,
    )
    for row in audit.loc[audit["availability"].ne("complete")].itertuples():
        print(
            f"[V4 paired funkyheatmap] {row.availability}: {row.facet}, "
            f"{row.family_label} / {row.condition_label} / {row.metric_label}; "
            f"matched repetitions {int(row.n_matched_repetitions)}/"
            f"{int(row.n_expected_repetitions)}; incomplete models: "
            f"{row.incomplete_models or 'none'}.",
            flush=True,
        )
    dropped = inputs["dropped_all_na_columns"]
    if dropped:
        print(
            "Funkyheatmap omitted globally all-NA columns: " + ", ".join(dropped),
            flush=True,
        )
    figure = funkyheatmappy.funky_heatmap(
        data=inputs["plot_data"],
        column_info=inputs["column_info"],
        row_info=inputs["row_info"],
        column_groups=inputs["column_groups"],
        row_groups=inputs["row_groups"],
        palettes=inputs["palettes"],
        legends=inputs["legends"],
        position_args=position_arguments(
            row_height=0.9,
            row_space=0.10,
            row_bigspace=0.55,
            col_width=1.0,
            col_space=0.08,
            col_bigspace=0.45,
            col_annot_offset=4.2,
            col_annot_angle=45,
            expand_xmin=0.2,
            expand_xmax=0.8,
            expand_ymin=0.2,
            expand_ymax=0.4,
        ),
        scale_column=False,
        add_abc=False,
    )
    n_columns = len(inputs["catalog"])
    n_rows = len(inputs["plot_data"])
    figure.set_size_inches(
        max(14.0, 5.5 + 0.50 * n_columns),
        max(5.0, 3.0 + 0.44 * n_rows),
        forward=True,
    )
    figure.suptitle(title, fontsize=14, y=1.01)
    return figure


def plot_v4_funky_heatmap(
    scorecard: pd.DataFrame,
    *,
    title: str,
) -> Figure:
    """Public notebook alias for :func:`render_v4_funkyheatmap`."""

    return render_v4_funkyheatmap(scorecard, title=title)


def notebook_source() -> str:
    """Return this module without its loader-only function and future import."""

    source = Path(__file__).read_text(encoding="utf-8")
    source = source.replace("from __future__ import annotations\n\n", "", 1)
    marker = "\ndef notebook_source() -> str:\n"
    if marker not in source:
        raise RuntimeError("Could not locate notebook_source boundary")
    return source.split(marker, 1)[0].rstrip() + "\n"


__all__ = [
    "CURVE_FAMILY_SPECS",
    "DEFAULT_V4_CURVE_FAMILIES",
    "DEFAULT_V4_FUNKY_FAMILIES",
    "DEFAULT_V4_METRICS",
    "EXTERNAL_CURVE_MODELS",
    "FUNKY_MODELS",
    "GENE2WIRE_FAMILY_MODELS",
    "METRIC_SPECS",
    "MODEL_DISPLAY_LABELS",
    "build_funkyheatmappy_inputs",
    "harmonic_mean_retained_measurement",
    "normalize_v4_funky_table",
    "plot_v4_funky_heatmap",
    "plot_v4_metric_grid",
    "plot_v4_metric_curves",
    "prepare_v4_curve_comparison",
    "prepare_v4_curve_table",
    "prepare_v4_funky_table",
    "render_v4_funkyheatmap",
]
