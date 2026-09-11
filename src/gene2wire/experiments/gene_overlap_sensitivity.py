"""Optional, isolated sensitivity analysis for simulation gene-panel overlap.

The primary overlap experiment deliberately has no added positive-label loss.
This module prepares a *separate* Technical-SAR 80% run without changing those
primary views or settings.  Only the union arm and the three PU models needed
for ``R_M(omega)`` are retained, so controls are not needlessly refit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .contracts import ExperimentDataset
from .protocol import Settings


TECH_SAR_80_LOSS_RATE = 0.8
TECH_SAR_80_LABEL = "Simulation — Technical-SAR 80% sensitivity"
TECH_SAR_80_EXPORT_NAME = "simulation_gene_overlap_0910_tech_sar80_sensitivity"
TECH_SAR_80_MODELS = ("PU", "PU-MIRT", "PU-Joint")


@dataclass(frozen=True)
class GeneOverlapSensitivityPlan:
    """Fully specified inputs for one separate sensitivity export."""

    views: tuple[ExperimentDataset, ...]
    settings: Settings
    label: str
    export_name: str
    expected_scenario_units: int
    expected_model_evaluations: int


def _zero_only(rates: Sequence[float]) -> bool:
    return len(rates) == 1 and np.isclose(float(rates[0]), 0.0, rtol=0, atol=1e-12)


def _sensitivity_view(view: ExperimentDataset) -> ExperimentDataset:
    context = dict(view.metadata.get("experiment_context", {}))
    if context.get("experiment") != "gene_overlap":
        raise ValueError("Tech-SAR sensitivity requires gene-overlap views")
    if context.get("arm") != "union":
        raise ValueError("Tech-SAR sensitivity accepts only union-arm views")
    if view.metadata.get("independent_unit") != "generated_dataset":
        raise ValueError("Tech-SAR overlap sensitivity is simulation-only")
    if view.natural_observed is not None:
        raise ValueError("Natural paired observations cannot be relabeled as synthetic Tech-SAR")

    context.update({
        "observation_profile": "technical_sar_80_sensitivity",
        "sensitivity_loss_rate": TECH_SAR_80_LOSS_RATE,
    })
    metadata = dict(view.metadata)
    metadata.update({
        "experiment_context": context,
        "model_allowlist": TECH_SAR_80_MODELS,
        "sensitivity_analysis": {
            "role": "secondary",
            "mechanism": "technical_sar",
            "positive_label_loss": TECH_SAR_80_LOSS_RATE,
            "primary_export_is_separate": True,
        },
    })
    result = replace(view, metadata=metadata)
    result.validate()
    return result


def prepare_tech_sar80_sensitivity(
    views: Sequence[ExperimentDataset],
    primary_settings: Settings,
    *,
    enabled: bool = False,
    export_name: str = TECH_SAR_80_EXPORT_NAME,
    primary_export_name: str = "simulation_gene_overlap_0910",
) -> GeneOverlapSensitivityPlan | None:
    """Prepare a separate union-only Technical-SAR 80% simulation run.

    ``enabled=False`` is intentionally a no-op, keeping the expensive
    sensitivity disabled by default.  When enabled, the function requires a
    zero-loss primary configuration, selects its union views, and returns new
    metadata/settings rather than mutating either input.
    """
    if not enabled:
        return None
    if not _zero_only(primary_settings.loss_rates):
        raise ValueError("Primary gene-overlap settings must contain only loss_rate=0")
    if str(export_name) == str(primary_export_name):
        raise ValueError("Sensitivity and primary runs must use different export names")
    if not views:
        raise ValueError("At least one simulation overlap view is required")

    union = []
    for view in views:
        context = view.metadata.get("experiment_context", {})
        if context.get("experiment") != "gene_overlap":
            raise ValueError("Every supplied view must belong to the gene-overlap experiment")
        if view.metadata.get("independent_unit") != "generated_dataset":
            raise ValueError("Tech-SAR overlap sensitivity is simulation-only")
        if view.natural_observed is not None:
            raise ValueError("Projection-TAGs/natural-observation views are not eligible")
        if context.get("arm") == "union":
            union.append(_sensitivity_view(view))
    if not union:
        raise ValueError("The primary simulation views contain no union arm")

    sensitivity_settings = replace(
        primary_settings,
        loss_rates=(TECH_SAR_80_LOSS_RATE,),
        run_mechanism_controls=False,
        run_calibration_controls=False,
    )
    scenario_units = len(union) * sensitivity_settings.n_outer_folds
    model_evaluations = scenario_units * len(TECH_SAR_80_MODELS)
    return GeneOverlapSensitivityPlan(
        views=tuple(union),
        settings=sensitivity_settings,
        label=TECH_SAR_80_LABEL,
        export_name=str(export_name),
        expected_scenario_units=scenario_units,
        expected_model_evaluations=model_evaluations,
    )
