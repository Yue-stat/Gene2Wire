"""Combined gene-panel, target-panel, and positive-censoring experiments.

This module is the only orchestration layer used by the canonical
``notebooks/measurement_degradation`` entry points.  Panel membership is fixed
before any outcome is inspected.  The native target mask remains the headline
test scope; its intersection with the artificial target panel is the only mask
given to calibration and learners.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..seeds import stable_seed
from .contracts import (ExperimentDataset, FeatureSet, Fold,
                        MeasurementEvaluationSpec)
from .io import atomic_json
from .measurement_design import (
    DEFAULT_ASSAYS,
    apply_target_panel,
    assign_virtual_assays,
    audit_measurement_support,
    draw_cyclic_panel_family,
    make_observation_plan,
    panel_design_tables,
    row_panel_mask,
)
from .pipeline import Artifacts, _atomic_csv, _execute, slug
from .protocol import Settings, fingerprint


GENE_COVERAGE_GRID = (1.0, 5.0 / 6.0, 2.0 / 3.0, 0.5)
TARGET_COVERAGE_GRID = (1.0, 2.0 / 3.0, 0.5)
RETENTION_GRID = (1.0, 0.75, 0.50, 0.25, 0.10)


def _probability_grid(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if (not result or len(set(result)) != len(result)
            or any(not np.isfinite(value) or value <= 0 or value > 1
                   for value in result)):
        raise ValueError(f"{name} must contain distinct probabilities in (0,1]")
    return result


def _contains(values: Sequence[float], target: float) -> bool:
    return any(np.isclose(value, target, rtol=0, atol=1e-12) for value in values)


@dataclass(frozen=True)
class MeasurementConfig:
    """Predeclared first-round condition grid from the masking specification."""

    gene_coverages: tuple[float, ...] = GENE_COVERAGE_GRID
    target_coverages: tuple[float, ...] = TARGET_COVERAGE_GRID
    retentions: tuple[float, ...] = RETENTION_GRID
    anchor_gene_coverage: float = 2.0 / 3.0
    anchor_target_coverage: float = 2.0 / 3.0
    anchor_retention: float = 0.50
    assay_ids: tuple[str, ...] = DEFAULT_ASSAYS
    heterogeneity_delta: float = 1.0
    include_matched_uniform: bool = True
    include_natural_recovery: bool = True
    n_panel_seeds: int = 5

    def __post_init__(self) -> None:
        genes = _probability_grid(self.gene_coverages, "gene_coverages")
        targets = _probability_grid(self.target_coverages, "target_coverages")
        retentions = _probability_grid(self.retentions, "retentions")
        assays = tuple(map(str, self.assay_ids))
        if len(assays) != 3 or len(set(assays)) != 3 or any(not value for value in assays):
            raise ValueError("Measurement degradation currently requires three assay IDs")
        if min(genes + targets) < 1 / 3 - 1e-12:
            raise ValueError("Union-preserving three-assay coverage cannot be below one third")
        for grid, anchor, name in (
            (genes, self.anchor_gene_coverage, "anchor_gene_coverage"),
            (targets, self.anchor_target_coverage, "anchor_target_coverage"),
            (retentions, self.anchor_retention, "anchor_retention"),
        ):
            if not _contains(grid, float(anchor)):
                raise ValueError(f"{name} must occur in its declared grid")
        if not _contains(genes, 1.0) or not _contains(targets, 1.0) or not _contains(retentions, 1.0):
            raise ValueError("Every grid must include its 100% control")
        if not np.isfinite(self.heterogeneity_delta) or self.heterogeneity_delta < 0:
            raise ValueError("heterogeneity_delta must be finite and nonnegative")
        if not isinstance(self.include_matched_uniform, bool) or not isinstance(
                self.include_natural_recovery, bool):
            raise TypeError("Control switches must be boolean")
        if (isinstance(self.n_panel_seeds, bool)
                or not isinstance(self.n_panel_seeds, (int, np.integer))
                or int(self.n_panel_seeds) < 1):
            raise ValueError("n_panel_seeds must be a positive integer")
        object.__setattr__(self, "gene_coverages", genes)
        object.__setattr__(self, "target_coverages", targets)
        object.__setattr__(self, "retentions", retentions)
        object.__setattr__(self, "assay_ids", assays)
        object.__setattr__(self, "n_panel_seeds", int(self.n_panel_seeds))


@dataclass(frozen=True)
class _FixedFolds:
    folds: tuple[Fold, ...]

    def __call__(self, n_outer_folds: int, seed: int) -> tuple[Fold, ...]:
        del seed
        if int(n_outer_folds) != len(self.folds):
            raise ValueError("Measurement views were built for a different fold count")
        return self.folds


@dataclass
class MaskedGeneFeatureBuilder:
    """Fold-fitted values plus explicit gene-observation and assay indicators."""

    gene_matrix: np.ndarray
    gene_names: tuple[str, ...]
    observed_mask: np.ndarray
    row_assays: np.ndarray
    source_builder: object
    panel_metadata: Mapping[str, object]
    n_gene_components: int | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.gene_matrix)
        mask = np.asarray(self.observed_mask)
        assays = np.asarray(self.row_assays).astype(str)
        if matrix.ndim != 2 or mask.shape != matrix.shape or assays.shape != (len(matrix),):
            raise ValueError("Gene matrix, observation mask and assay labels must align")
        if len(self.gene_names) != matrix.shape[1] or not np.all(np.isfinite(matrix)):
            raise ValueError("The measurement gene pool must be finite and named")
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError("Gene observation mask must be binary")
        if self.n_gene_components is not None and (
                isinstance(self.n_gene_components, bool)
                or not isinstance(self.n_gene_components, (int, np.integer))
                or int(self.n_gene_components) < 1):
            raise ValueError("n_gene_components must be a positive integer or None")
        self.gene_matrix = matrix
        self.observed_mask = mask.astype(bool)
        self.row_assays = assays

    def __call__(self, train_rows, use_location=False, use_target_features=False):
        train = np.asarray(train_rows)
        if (train.ndim != 1 or not np.issubdtype(train.dtype, np.integer)
                or not len(train) or len(np.unique(train)) != len(train)
                or np.any(train < 0) or np.any(train >= len(self.gene_matrix))):
            raise ValueError("train_rows must be unique aligned integer indices")
        if not isinstance(use_location, bool) or not isinstance(use_target_features, bool):
            raise TypeError("Feature switches must be boolean")
        n_cells, n_genes = self.gene_matrix.shape
        center = np.empty(n_genes, dtype=float)
        scale = np.empty(n_genes, dtype=float)
        values = np.zeros((n_cells, n_genes), dtype=float)
        unsupported = []
        for gene in range(n_genes):
            visible_train = train[self.observed_mask[train, gene]]
            if not len(visible_train):
                unsupported.append(self.gene_names[gene])
                continue
            source = self.gene_matrix[visible_train, gene].astype(float)
            center[gene] = float(np.mean(source))
            scale[gene] = float(np.std(source))
            if scale[gene] < 1e-12:
                scale[gene] = 1.0
            visible = self.observed_mask[:, gene]
            values[visible, gene] = (
                self.gene_matrix[visible, gene] - center[gene]
            ) / scale[gene]
        if unsupported:
            raise ValueError(
                "Training rows contain no visible value for genes: "
                + ", ".join(unsupported[:10])
            )

        value_metadata = {}
        if self.n_gene_components is None:
            value_features = values
            value_names = tuple(f"value::{name}" for name in self.gene_names)
        else:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            n_components = min(
                int(self.n_gene_components), n_genes, len(train) - 1)
            if n_components < 1:
                raise ValueError("PCA requires at least two training rows")
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=0,
            ).fit(values[train])
            value_features = pca.transform(values)
            pc_scaler = StandardScaler().fit(value_features[train])
            value_features = pc_scaler.transform(value_features)
            value_names = tuple(
                f"masked_gene_PC_{index + 1}" for index in range(n_components)
            )
            value_metadata = {
                "n_gene_components_requested": int(self.n_gene_components),
                "n_gene_components_used": n_components,
                "pca_explained_variance_ratio": (
                    pca.explained_variance_ratio_.astype(float).tolist()),
                "pca_random_state": 0,
                "pca_fit_rows": train.astype(int).tolist(),
            }

        n_value_features = value_features.shape[1]
        blocks: dict[str, tuple[int, ...]] = {
            "gene_value": tuple(range(n_value_features)),
            "gene_observed": tuple(range(
                n_value_features, n_value_features + n_genes)),
        }
        x = np.column_stack((value_features, self.observed_mask.astype(float)))
        names = value_names + tuple(f"observed::{name}" for name in self.gene_names)
        assay_levels = tuple(sorted(set(self.row_assays)))
        # The first level is represented by both dummy columns being zero.
        if len(assay_levels) != 3:
            raise ValueError("Every feature view must contain all three virtual assays")
        assay_design = np.column_stack([
            (self.row_assays == assay).astype(float) for assay in assay_levels[1:]
        ])
        blocks["assay"] = tuple(range(x.shape[1], x.shape[1] + assay_design.shape[1]))
        x = np.column_stack((x, assay_design))
        names += tuple(f"assay::{assay}" for assay in assay_levels[1:])

        target = None
        source_metadata = {}
        if use_location or use_target_features:
            original = self.source_builder(train, use_location, use_target_features)
            source_metadata = dict(original.metadata)
            if use_location:
                indices = tuple(original.feature_blocks.get("location", ()))
                if not indices:
                    raise ValueError("USE_LOCATION=True but the source adapter has no location block")
                location = np.asarray(original.X)[:, indices]
                blocks["location"] = tuple(range(x.shape[1], x.shape[1] + location.shape[1]))
                x = np.column_stack((x, location))
                original_names = tuple(original.feature_names[index] for index in indices)
                names += tuple(f"location::{name}" for name in original_names)
            if use_target_features:
                if original.Y_target is None:
                    raise ValueError("USE_TARGET_FEATURES=True but target descriptors are unavailable")
                target = np.asarray(original.Y_target).copy()

        return FeatureSet(
            X=x,
            feature_blocks=blocks,
            Y_target=target,
            feature_names=names,
            metadata={
                **dict(self.panel_metadata),
                "preprocessing": (
                    "assay-wide mask -> visible-training-only standardization -> "
                    "missing standardized value 0 -> "
                    + ("train-only PCA -> " if self.n_gene_components is not None else "")
                    + "explicit observation mask + assay ID"
                ),
                "preprocessing_fit_rows": train.astype(int).tolist(),
                "gene_center": center.tolist(),
                "gene_scale": scale.tolist(),
                "missing_gene_value": 0.0,
                "missing_is_distinct_from_observed_zero": True,
                **value_metadata,
                "source_adapter_metadata": source_metadata,
            },
        )


def _strata(dataset: ExperimentDataset) -> tuple[np.ndarray, str]:
    for name in ("animal", "sample", "animal_id", "sample_id", "slice"):
        if name in dataset.groups:
            return np.asarray(dataset.groups[name]).astype(str), name
    return np.repeat("all", len(dataset.cell_ids)), "all_cells"


def _canonical_coverage(family, requested: float) -> float:
    draw = family.draw(requested)
    return (float(requested) if draw.duplicate_of_coverage is None
            else float(draw.duplicate_of_coverage))


def _condition_registry(config: MeasurementConfig, *, include_natural: bool):
    registry: dict[tuple[float, float, float | None, str], set[str]] = {}

    def add(gene, target, retention, mechanism, role):
        key = (float(gene), float(target),
               None if retention is None else float(retention), str(mechanism))
        registry.setdefault(key, set()).add(str(role))

    add(1.0, 1.0, 1.0, "assay_target_sar", "full_control")
    for retention in config.retentions:
        add(config.anchor_gene_coverage, config.anchor_target_coverage,
            retention, "assay_target_sar", "retention_curve")
    for gene in config.gene_coverages:
        for target in config.target_coverages:
            add(gene, target, config.anchor_retention,
                "assay_target_sar", "coverage_heatmap")
    if config.include_matched_uniform:
        add(config.anchor_gene_coverage, config.anchor_target_coverage,
            config.anchor_retention, "scar", "matched_uniform_control")
    if include_natural and config.include_natural_recovery:
        add(1.0, 1.0, None, "natural", "natural_recovery")
    return registry


def _panel_audit(prefix, dimension, family) -> dict[str, pd.DataFrame]:
    result = {}
    for name, table in panel_design_tables(family).items():
        result[f"measurement_{dimension}_panel_{name}"] = table.assign(
            **prefix, dimension=dimension)
    membership = []
    for requested, draw in family.draws.items():
        for assay_index, assay in enumerate(draw.assay_ids):
            for item_index, item in enumerate(draw.item_ids):
                membership.append({
                    **prefix, "dimension": dimension,
                    "requested_coverage": requested,
                    "actual_coverage": draw.actual_coverage,
                    "assay": assay, "item": item,
                    "measured": bool(draw.incidence[assay_index, item_index]),
                })
    result[f"measurement_{dimension}_panel_membership"] = pd.DataFrame(membership)
    return result


def _information_access(prefix, use_target_features: bool) -> pd.DataFrame:
    from .qiao import qiao_model_names

    rows = []
    def add(model, labels, exposure, paired, reference_training, target_input):
        rows.append({
            **prefix, "model": model, "training_labels": labels,
            "exposure_source": exposure, "uses_paired_calibration": paired,
            "uses_reference_labels_for_training": reference_training,
            "target_input": target_input, "uses_true_generator_e": False,
            "uses_test_reference": False,
        })
    target_input = "declared_target_features" if use_target_features else "target_identity"
    for model in ("Logistic", "MIRT", "Joint"):
        add(model, "observed", "none", False, False, target_input)
    for model in ("PU", "PU-MIRT", "PU-Joint"):
        add(model, "observed", "paired_estimated", True, False, target_input)
    add("Reference-only", "paired_reference", "none", True, True, target_input)
    for model in ("Reference+PU", "Reference+PU-MIRT", "Reference+PU-Joint"):
        add(model, "paired_reference+observed", "paired_estimated", True, True, target_input)
    add("RF-observed", "observed", "none", False, False, "cell_features")
    add("RF-reference", "paired_reference", "none", True, True, "cell_features")
    add("RF-mixed", "paired_reference+observed", "none", True, True, "cell_features")
    for model in qiao_model_names(use_target_features):
        add(model, "observed", "none", False, False, target_input)
    # The current GenEML/SAR wrappers model target identity and assay-target
    # exposure directly; only ShiftIMC consumes optional target descriptors.
    add("GenEML-adapted", "observed", "model_estimated_assay_target", False, False,
        "target_identity")
    add("Inductive-PU-MC", "observed", "paired_estimated", True, False, target_input)
    add("SAR-PU", "observed", "model_estimated_assay_target", False, False,
        "target_identity")
    return pd.DataFrame(rows)


def _combine_tables(collection: Sequence[Mapping[str, pd.DataFrame]]):
    keys = sorted(set().union(*(tables.keys() for tables in collection)))
    return {key: pd.concat([tables[key] for tables in collection if key in tables],
                           ignore_index=True) for key in keys}


def build_measurement_views(
    dataset: ExperimentDataset,
    settings: Settings,
    config: MeasurementConfig,
    *,
    repetition: int,
    panel_seed: int | None = None,
) -> tuple[list[ExperimentDataset], dict[str, pd.DataFrame]]:
    """Create all unique first-round views for one data/panel repetition."""
    dataset.validate()
    if dataset.gene_matrix is None or not dataset.gene_names:
        raise ValueError("Measurement degradation requires a declared source gene pool")
    if len(dataset.gene_names) < 3 or len(dataset.target_ids) < 3:
        raise ValueError("Three virtual assays require at least three genes and targets")
    folds = tuple(dataset.split_builder(settings.n_outer_folds, settings.seed))
    for fold in folds:
        fold.validate(len(dataset.cell_ids))
    strata, stratum_kind = _strata(dataset)
    if (isinstance(repetition, bool) or not isinstance(repetition, (int, np.integer))
            or int(repetition) < 0):
        raise ValueError("repetition must be a nonnegative integer")
    repetition = int(repetition)
    if panel_seed is None:
        panel_seed = repetition
    if (isinstance(panel_seed, bool) or not isinstance(panel_seed, (int, np.integer))
            or int(panel_seed) < 0):
        raise ValueError("panel_seed must be a nonnegative integer")
    panel_seed = int(panel_seed)

    # Real-data panel repeats change panel membership but keep cell assignment,
    # censoring draws, paired rows and model initialization fixed. Simulation
    # repeats redraw those objects per generated dataset, with multiple panel
    # seeds nested inside that independent repetition.
    independent = dataset.metadata.get("independent_unit") == "generated_dataset"
    data_repetition = repetition if independent else 0
    assignment_rep = data_repetition
    assays = assign_virtual_assays(
        dataset.cell_ids, strata,
        seed=stable_seed(settings.seed, "measurement_assays", dataset.name, assignment_rep),
        assay_ids=config.assay_ids,
    )
    gene_family = draw_cyclic_panel_family(
        dataset.gene_names, config.gene_coverages,
        seed=stable_seed(
            settings.seed, "measurement_genes", dataset.name,
            *( (data_repetition, panel_seed) if independent else (panel_seed,) ),
        ),
        assay_ids=config.assay_ids,
    )
    target_family = draw_cyclic_panel_family(
        dataset.target_ids, config.target_coverages,
        seed=stable_seed(
            settings.seed, "measurement_targets", dataset.name,
            *( (data_repetition, panel_seed) if independent else (panel_seed,) ),
        ),
        assay_ids=config.assay_ids,
    )
    censor_rep = data_repetition
    observation_plan = make_observation_plan(
        assays, dataset.target_ids,
        seed=stable_seed(settings.seed, "measurement_censoring", dataset.name, censor_rep),
        assay_ids=config.assay_ids,
        heterogeneity_delta=config.heterogeneity_delta,
    )

    prefix = {"dataset": dataset.name, "panel_seed": panel_seed,
              "data_repetition": data_repetition,
              "sharing_strength": dataset.metadata.get("sharing_strength")}
    tables = {
        **_panel_audit(prefix, "gene", gene_family),
        **_panel_audit(prefix, "target", target_family),
        "measurement_assignment": pd.DataFrame({
            **{key: value for key, value in prefix.items() if key != "sharing_strength"},
            "cell_id": dataset.cell_ids, "virtual_assay": assays,
            "stratum": strata, "stratum_kind": stratum_kind,
        }),
        "information_access": _information_access(
            prefix, settings.use_target_features),
    }
    registry = _condition_registry(
        config, include_natural=dataset.natural_observed is not None)
    normalized: dict[tuple[float, float, float | None, str], set[str]] = {}
    for (gene, target, retention, mechanism), roles in registry.items():
        key = (_canonical_coverage(gene_family, gene),
               _canonical_coverage(target_family, target), retention, mechanism)
        normalized.setdefault(key, set()).update(roles)

    grouped: dict[tuple[float, float], list[dict]] = {}
    scenario_rows = []
    for (gene, target, retention, mechanism), roles in sorted(
            normalized.items(), key=lambda item: tuple(str(value) for value in item[0])):
        scenario = {
            "analysis": "primary",
            "mechanism": mechanism,
            "loss_rate": None if retention is None else float(1.0 - retention),
            "positive_retention": retention,
            "calibration_fraction": settings.paired_fraction,
            "calibration_spec": "correct",
            "condition_roles": "+".join(sorted(roles)),
        }
        grouped.setdefault((gene, target), []).append(scenario)
        scenario_rows.append({**prefix, "gene_requested_coverage": gene,
                              "target_requested_coverage": target, **scenario})
    tables["measurement_scenarios"] = pd.DataFrame(scenario_rows)

    views: list[ExperimentDataset] = []
    support_summary, support_counts, support_assays, support_pairs = [], [], [], []
    gene_target_support, unsupported_gene_target_pairs = [], []
    native = np.asarray(dataset.measured, dtype=bool)
    for (gene_coverage, target_coverage), schedule in grouped.items():
        gene_draw = gene_family.draw(gene_coverage)
        target_draw = target_family.draw(target_coverage)
        gene_mask = row_panel_mask(assays, gene_draw)
        fit_measured = apply_target_panel(native, assays, target_draw)
        panel_id = fingerprint({
            "dataset": dataset.name, "data_repetition": data_repetition,
            "panel_seed": panel_seed,
            "gene": gene_coverage, "target": target_coverage,
            "gene_incidence": gene_draw.incidence.astype(int).tolist(),
            "target_incidence": target_draw.incidence.astype(int).tolist(),
        })
        context = {
            "experiment": "measurement_degradation",
            "panel_seed": panel_seed, "data_repetition": data_repetition,
            "panel_id": panel_id,
            "gene_requested_coverage": float(gene_coverage),
            "gene_coverage": float(gene_draw.actual_coverage),
            "gene_overlap": float(gene_draw.mean_pairwise_overlap),
            "gene_panel_size": int(gene_draw.panel_size),
            "gene_union_coverage": float(gene_draw.union_coverage),
            "target_requested_coverage": float(target_coverage),
            "target_coverage": float(target_draw.actual_coverage),
            "target_overlap": float(target_draw.mean_pairwise_overlap),
            "target_panel_size": int(target_draw.panel_size),
            "target_union_coverage": float(target_draw.union_coverage),
        }
        features = MaskedGeneFeatureBuilder(
            np.asarray(dataset.gene_matrix), tuple(dataset.gene_names), gene_mask,
            assays, dataset.feature_builder,
            {**context, "source_gene_pool_size": len(dataset.gene_names)},
            n_gene_components=dataset.metadata.get("n_gene_components"),
        )
        view = replace(
            dataset,
            feature_builder=features,
            split_builder=_FixedFolds(folds),
            groups={**dataset.groups, "virtual_assay": assays},
            training_measured=fit_measured,
            virtual_assays=assays,
            observation_plan=observation_plan,
            measurement_evaluation=MeasurementEvaluationSpec(
                reference=np.asarray(dataset.reference, dtype=bool),
                source_measured=native,
                assays=assays,
            ),
            metadata={
                **dataset.metadata,
                "experiment_repetition": repetition,
                "model_seed": data_repetition,
                "experiment_context": context,
                "measurement_scenarios": tuple(schedule),
                "measurement_stratum_kind": stratum_kind,
                "measurement_design_reads_outcomes": False,
            },
        )
        view.validate()
        views.append(view)
        for fold in folds:
            for role, rows in (("train", fold.train_rows),
                               ("validation", fold.validation_rows),
                               ("test", fold.test_rows)):
                audit = audit_measurement_support(
                    fit_measured, assays, dataset.target_ids,
                    rows=rows, assay_ids=config.assay_ids)
                row_prefix = {**prefix, **context, "outer_fold": fold.outer_fold,
                              "split": role}
                # A complete gene union and a complete target union do not by
                # themselves guarantee that every gene-target relationship is
                # observed in at least one cell.  Audit that joint support on
                # each authorized fold/split without inspecting outcomes.
                if tuple(gene_draw.assay_ids) != tuple(audit.assay_ids):
                    raise RuntimeError(
                        "Gene and target support audits use different assay orders")
                # Gene membership is assay-wide, so summing the fold-local
                # assay-by-target counts is exactly equivalent to a cell-level
                # G-by-N @ N-by-T product and avoids that much larger operation.
                joint_counts = (
                    gene_draw.incidence.astype(np.int64).T
                    @ audit.measurement_counts.astype(np.int64)
                )
                joint_supported = joint_counts > 0
                unsupported = np.argwhere(~joint_supported)
                total_pairs = int(joint_supported.size)
                supported_pairs = int(joint_supported.sum())
                gene_target_support.append({
                    **row_prefix,
                    "gene_target_pair_count": total_pairs,
                    "supported_gene_target_pair_count": supported_pairs,
                    "unsupported_gene_target_pair_count": int(len(unsupported)),
                    "unsupported_gene_target_pair_fraction": float(
                        len(unsupported) / total_pairs),
                    "all_gene_target_pairs_supported": bool(not len(unsupported)),
                    "minimum_co_measured_cells": int(joint_counts.min()),
                    "median_co_measured_cells": float(np.median(joint_counts)),
                    "maximum_co_measured_cells": int(joint_counts.max()),
                })
                if role == "train":
                    for gene_index, target_index in unsupported:
                        unsupported_gene_target_pairs.append({
                            **row_prefix,
                            "gene": dataset.gene_names[int(gene_index)],
                            "target": dataset.target_ids[int(target_index)],
                            "co_measured_cell_count": 0,
                        })
                support_summary.append({
                    **row_prefix, "n_rows": audit.n_rows,
                    "effective_union_coverage": audit.union_coverage,
                    "graph_connected": audit.graph_connected,
                    "n_components": audit.n_components,
                    "unsupported_target_count": len(audit.unsupported_items),
                    "single_assay_target_count": len(audit.single_assay_items),
                    "unsupported_targets": ";".join(audit.unsupported_items),
                })
                for assay_index, assay in enumerate(audit.assay_ids):
                    support_assays.append({
                        **row_prefix,
                        "virtual_assay": assay,
                        "effective_target_count": int(
                            audit.effective_incidence[assay_index].sum()),
                        "effective_target_coverage": float(
                            audit.per_assay_coverage[assay_index]),
                        "measured_entry_count": int(
                            audit.measurement_counts[assay_index].sum()),
                    })
                    for target_index, target in enumerate(audit.item_ids):
                        support_counts.append({
                            **row_prefix, "virtual_assay": assay, "target": target,
                            "measured_entries": int(audit.measurement_counts[
                                assay_index, target_index]),
                            "effective_support": bool(audit.effective_incidence[
                                assay_index, target_index]),
                        })
                for first in range(len(audit.assay_ids)):
                    for second in range(first + 1, len(audit.assay_ids)):
                        count = int(audit.co_measurement_counts[first, second])
                        support_pairs.append({
                            **row_prefix,
                            "assay_a": audit.assay_ids[first],
                            "assay_b": audit.assay_ids[second],
                            "co_measured_target_count": count,
                            "connected_edge": bool(count > 0),
                        })
    tables["measurement_support"] = pd.DataFrame(support_summary)
    tables["measurement_support_counts"] = pd.DataFrame(support_counts)
    tables["measurement_support_assays"] = pd.DataFrame(support_assays)
    tables["measurement_support_pairs"] = pd.DataFrame(support_pairs)
    tables["measurement_gene_target_support"] = pd.DataFrame(gene_target_support)
    unsupported_frame = pd.DataFrame(unsupported_gene_target_pairs)
    if unsupported_frame.empty:
        summary_metrics = {
            "gene_target_pair_count", "supported_gene_target_pair_count",
            "unsupported_gene_target_pair_count",
            "unsupported_gene_target_pair_fraction",
            "all_gene_target_pairs_supported", "minimum_co_measured_cells",
            "median_co_measured_cells", "maximum_co_measured_cells",
        }
        prefix_columns = [
            column for column in tables["measurement_gene_target_support"].columns
            if column not in summary_metrics
        ]
        unsupported_frame = pd.DataFrame(columns=[
            *prefix_columns, "gene", "target", "co_measured_cell_count",
        ])
    tables["measurement_unsupported_gene_target_pairs"] = unsupported_frame
    return views, tables


def _manifest(config: MeasurementConfig) -> dict:
    return {
        "experiment": "measurement_degradation",
        "measurement_protocol": asdict(config),
        "panel_design": "three-assay nested cyclic equal-coverage panels with complete union",
        "gene_mask": "assay-wide before visible-training-only standardization; value+mask+assay inputs",
        "target_mask": "W_fit = W_native AND assay-target incidence; off-panel is NA, never negative",
        "positive_censoring": "fold-local per-target assay-heterogeneous logistic retention",
        "matched_control": "uniform censoring exactly matched to heterogeneous retained training positives",
        "headline_scope": "fixed W_native outer-test reference for every condition",
        "comparator_contract": "same splits, masks, inputs and candidate budget; no evaluation truth in fitting",
    }


def _add_heatmap_contrasts(artifacts: Artifacts) -> None:
    """Choose one baseline using development losses, then freeze it for the grid."""
    tuning = artifacts.tables.get("tuning", pd.DataFrame())
    metrics = artifacts.tables.get("metrics", pd.DataFrame())
    candidates = ("PU", "PU-Joint", "RF-reference")
    required = {"model", "validation_loss", "condition_roles", "outer_fold", "repetition"}
    selection = pd.DataFrame()
    if required.issubset(tuning):
        frame = tuning.loc[
            tuning["condition_roles"].astype(str).str.contains("coverage_heatmap", regex=False)
            & tuning["model"].isin(candidates)
        ].copy()
        frame["validation_loss"] = pd.to_numeric(frame["validation_loss"], errors="coerce")
        frame = frame.loc[np.isfinite(frame["validation_loss"])]
        if not frame.empty:
            units = [column for column in (
                "dataset", "sharing_strength", "mechanism", "panel_seed",
                "repetition", "outer_fold", "gene_requested_coverage",
                "gene_coverage", "target_requested_coverage", "target_coverage",
                "positive_retention", "model"
            ) if column in frame]
            best = frame.groupby(units, dropna=False, observed=True)["validation_loss"].min()
            selection = (best.groupby("model").mean().sort_values()
                         .rename("mean_development_validation_loss").reset_index())
            selection["selected"] = False
            selection.loc[selection.index[0], "selected"] = True
    if selection.empty:
        selection = pd.DataFrame({
            "model": ["PU"], "mean_development_validation_loss": [np.nan],
            "selected": [True], "fallback_reason": ["no comparable finite tuning rows"],
        })
    baseline = str(selection.loc[selection["selected"], "model"].iloc[0])
    artifacts.tables["heatmap_baseline_selection"] = selection

    required_metrics = {"model", "condition_roles", "evaluation_scope",
                        "gene_coverage", "target_coverage", "macro_brier"}
    contrast = pd.DataFrame()
    if required_metrics.issubset(metrics):
        frame = metrics.loc[
            metrics["condition_roles"].astype(str).str.contains("coverage_heatmap", regex=False)
            & metrics["evaluation_scope"].eq("native_reference")
            & metrics["model"].isin((baseline, "PU-MIRT"))
        ].copy()
        if not frame.empty:
            # Average folds/panel seeds only within a fully identified
            # scientific condition.  In particular, simulation sharing
            # strengths and separate datasets must never be silently pooled
            # into one heatmap cell.
            coordinate_candidates = (
                "dataset", "sharing_strength", "experiment", "analysis",
                "mechanism", "loss_rate", "positive_retention",
                "calibration_fraction", "calibration_spec", "evaluation_scope",
                "condition_roles", "gene_requested_coverage", "gene_coverage",
                "gene_overlap", "gene_panel_size", "gene_union_coverage",
                "target_requested_coverage", "target_coverage", "target_overlap",
                "target_panel_size", "target_union_coverage",
            )
            coordinates = [name for name in coordinate_candidates if name in frame]
            # The required actual coverages are the minimal coordinates for
            # older artifacts which predate the richer context columns.
            for name in ("gene_coverage", "target_coverage"):
                if name not in coordinates:
                    coordinates.append(name)
            summary = (frame.groupby(coordinates + ["model"], dropna=False,
                                     observed=True)["macro_brier"]
                       .mean().unstack("model"))
            if baseline in summary and "PU-MIRT" in summary:
                contrast = summary.reset_index()
                contrast["baseline_model"] = baseline
                contrast["comparison_model"] = "PU-MIRT"
                contrast["baseline_macro_brier"] = summary[baseline].to_numpy()
                contrast["comparison_macro_brier"] = summary["PU-MIRT"].to_numpy()
                contrast["brier_contrast"] = (
                    contrast["baseline_macro_brier"]
                    - contrast["comparison_macro_brier"]
                )
                if "probability_semantics" in frame:
                    semantics = (frame.groupby(
                        coordinates + ["model"], dropna=False, observed=True
                    )["probability_semantics"].agg(
                        lambda values: ";".join(sorted(set(
                            values.dropna().astype(str))))
                    ).unstack("model"))
                    semantics = semantics.reindex(summary.index)
                    contrast["baseline_probability_semantics"] = (
                        semantics[baseline].to_numpy())
                    contrast["comparison_probability_semantics"] = (
                        semantics["PU-MIRT"].to_numpy())
    artifacts.tables["heatmap_contrasts"] = contrast
    for name in ("heatmap_baseline_selection", "heatmap_contrasts"):
        _atomic_csv(artifacts.tables[name], artifacts.export_dir / f"{name}.csv")


def _prediction_units(artifacts: Artifacts):
    """Yield saved predictions with their evaluation-only unit context."""
    metrics = artifacts.tables.get("metrics", pd.DataFrame())
    model_lookup = {
        slug(model): str(model) for model in metrics.get("model", pd.Series(dtype=str))
        .dropna().astype(str).unique()
    }
    units = artifacts.export_dir / "units"
    if not units.is_dir():
        return
    for audit_path in sorted(units.glob("*/audit.json")):
        context = json.loads(audit_path.read_text())
        for path in sorted(audit_path.parent.glob("*_predictions.npz")):
            stem = path.name[:-len("_predictions.npz")]
            model = model_lookup.get(stem)
            if model is None:
                continue
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            yield context, model, arrays


def _add_panel_stability(artifacts: Artifacts) -> None:
    """Compute same-cell prediction SD across nested panel seeds."""
    columns = [
        "dataset", "sharing_strength", "repetition",
        "gene_requested_coverage",
        "gene_coverage", "target_requested_coverage", "target_coverage",
        "positive_retention", "mechanism", "condition_roles", "model",
        "probability_semantics", "is_reference_probability",
        "panel_sensitivity", "macro_auprc", "n_panel_seeds",
        "n_common_native_pairs",
    ]
    collected: dict[tuple, dict[int, dict[tuple[str, str], float]]] = {}
    for context, model, arrays in _prediction_units(artifacts) or ():
        roles = str(context.get("condition_roles", ""))
        if "coverage_heatmap" not in roles:
            continue
        required = {"prediction", "cell_ids", "target_ids", "source_measured"}
        if not required.issubset(arrays):
            continue
        key = (
            str(context.get("dataset")), context.get("sharing_strength"),
            int(context.get("data_repetition", 0)),
            float(context.get("gene_requested_coverage",
                              context["gene_coverage"])),
            float(context["gene_coverage"]),
            float(context.get("target_requested_coverage",
                              context["target_coverage"])),
            float(context["target_coverage"]),
            float(context["positive_retention"]),
            str(context.get("mechanism")), roles, model,
        )
        panel_seed = int(context["panel_seed"])
        destination = collected.setdefault(key, {}).setdefault(panel_seed, {})
        prediction = np.asarray(arrays["prediction"], dtype=float)
        measured = np.asarray(arrays["source_measured"], dtype=bool)
        cells = np.asarray(arrays["cell_ids"]).astype(str)
        targets = np.asarray(arrays["target_ids"]).astype(str)
        for row, column in zip(*np.nonzero(measured)):
            pair = (cells[row], targets[column])
            if pair in destination:
                raise ValueError("A cell-target prediction appears in multiple outer folds")
            destination[pair] = float(prediction[row, column])

    metrics = artifacts.tables.get("metrics", pd.DataFrame())
    rows = []
    for key, seeds in collected.items():
        (dataset, sharing, data_repetition,
         gene_requested_coverage, gene_coverage,
         target_requested_coverage, target_coverage, positive_retention,
         mechanism, roles, model) = key
        if len(seeds) < 2:
            continue
        common = set.intersection(*(set(values) for values in seeds.values()))
        if not common:
            continue
        ordered = sorted(common)
        matrix = np.asarray([[values[pair] for pair in ordered]
                             for _, values in sorted(seeds.items())], dtype=float)
        subset_mask = (
            metrics["dataset"].eq(dataset)
            & metrics["model"].eq(model)
            & metrics["evaluation_scope"].eq("native_reference")
            & metrics["mechanism"].eq(mechanism)
            & np.isclose(pd.to_numeric(metrics["gene_coverage"], errors="coerce"),
                         gene_coverage)
            & np.isclose(pd.to_numeric(metrics["target_coverage"], errors="coerce"),
                         target_coverage)
            & np.isclose(pd.to_numeric(metrics["positive_retention"], errors="coerce"),
                         positive_retention)
        )
        if "gene_requested_coverage" in metrics:
            subset_mask &= np.isclose(pd.to_numeric(
                metrics["gene_requested_coverage"], errors="coerce"),
                gene_requested_coverage)
        if "target_requested_coverage" in metrics:
            subset_mask &= np.isclose(pd.to_numeric(
                metrics["target_requested_coverage"], errors="coerce"),
                target_requested_coverage)
        if "condition_roles" in metrics:
            subset_mask &= metrics["condition_roles"].astype(str).eq(roles)
        if "sharing_strength" in metrics:
            subset_mask &= (metrics["sharing_strength"].isna() if pd.isna(sharing)
                            else metrics["sharing_strength"].eq(sharing))
        if "data_repetition" in metrics:
            subset_mask &= pd.to_numeric(
                metrics["data_repetition"], errors="coerce").eq(data_repetition)
        subset = metrics.loc[subset_mask]
        semantics = (subset.get("probability_semantics", pd.Series(dtype=str))
                     .dropna().astype(str).unique())
        if len(semantics) > 1:
            raise ValueError(
                "A panel-stability condition mixes probability semantics for "
                f"{model}: {sorted(semantics)}")
        probability_semantics = str(semantics[0]) if len(semantics) else "unknown"
        rows.append({
            "dataset": dataset, "sharing_strength": sharing,
            "repetition": data_repetition,
            "gene_requested_coverage": gene_requested_coverage,
            "gene_coverage": gene_coverage,
            "target_requested_coverage": target_requested_coverage,
            "target_coverage": target_coverage,
            "positive_retention": positive_retention,
            "mechanism": mechanism, "condition_roles": roles, "model": model,
            "probability_semantics": probability_semantics,
            "is_reference_probability": (
                probability_semantics == "reference"
                or probability_semantics.startswith("reference_p")
            ),
            "panel_sensitivity": float(np.mean(np.std(matrix, axis=0, ddof=0))),
            "macro_auprc": float(pd.to_numeric(
                subset.get("macro_auprc", pd.Series(dtype=float)), errors="coerce"
            ).mean()),
            "n_panel_seeds": len(seeds), "n_common_native_pairs": len(common),
        })
    artifacts.tables["panel_stability"] = pd.DataFrame(rows, columns=columns)
    _atomic_csv(artifacts.tables["panel_stability"],
                artifacts.export_dir / "panel_stability.csv")


def _add_projection_budget_recall(artifacts: Artifacts) -> None:
    """Pool OOF standard negatives and rank the same fraction per target."""
    budgets = (0.01, 0.05, 0.10, 0.20)
    columns = [
        "dataset", "repetition", "model", "budget_fraction",
        "selected_candidates", "candidate_count",
        "amplification_confirmed_positive_count", "recovered_positive_count",
        "amplification_confirmed_recall", "micro_amplification_confirmed_recall",
        "n_targets", "n_evaluable_targets", "ranking_source", "aggregation",
    ]
    per_target_columns = [
        "dataset", "repetition", "model", "target", "budget_fraction",
        "selected_candidates", "candidate_count",
        "amplification_confirmed_positive_count", "recovered_positive_count",
        "amplification_confirmed_recall", "ranking_source", "aggregation",
    ]
    pooled: dict[tuple[str, int, str], dict] = {}
    for context, model, arrays in _prediction_units(artifacts) or ():
        if (context.get("mechanism") != "natural"
                or "natural_recovery" not in str(context.get("condition_roles", ""))):
            continue
        required = {"prediction", "reference", "observed", "source_measured",
                    "cell_ids", "target_ids"}
        if not required.issubset(arrays):
            continue
        prediction = np.asarray(arrays["prediction"], dtype=float)
        reference = np.asarray(arrays["reference"], dtype=bool)
        observed = np.asarray(arrays["observed"], dtype=bool)
        measured = np.asarray(arrays["source_measured"], dtype=bool)
        if not (prediction.shape == reference.shape == observed.shape == measured.shape):
            raise ValueError("Projection recovery arrays must have identical matrix shapes")
        cells = np.asarray(arrays["cell_ids"]).astype(str)
        targets = np.asarray(arrays["target_ids"]).astype(str)
        if cells.shape != (prediction.shape[0],) or targets.shape != (prediction.shape[1],):
            raise ValueError("Projection recovery cell/target IDs must align with predictions")
        ranking_source = "ranking_score" if "ranking_score" in arrays else "prediction"
        score_matrix = np.asarray(arrays.get("ranking_score", prediction), dtype=float)
        if score_matrix.shape != prediction.shape:
            raise ValueError("Projection recovery ranking scores must align with predictions")
        key = (str(context.get("dataset")), int(context.get("repetition")), model)
        destination = pooled.setdefault(
            key, {"seen": set(), "candidates": {}, "ranking_sources": set()})
        destination["ranking_sources"].add(ranking_source)
        for row, column in zip(*np.nonzero(measured)):
            pair = (cells[row], targets[column])
            if pair in destination["seen"]:
                raise ValueError(
                    "A cell-target prediction appears in multiple outer folds "
                    f"for {key}: {pair}")
            destination["seen"].add(pair)
            if not observed[row, column]:
                score = float(score_matrix[row, column])
                if not np.isfinite(score):
                    raise ValueError(
                        f"Nonfinite Projection recovery ranking score for {pair}")
                destination["candidates"][pair] = (
                    score, bool(reference[row, column]))

    rows, target_rows = [], []
    for (dataset, repetition, model), values in sorted(pooled.items()):
        if len(values["ranking_sources"]) != 1:
            raise ValueError(
                "A pooled OOF model mixes ranking_score and prediction ranking "
                f"sources: {(dataset, repetition, model)}")
        ranking_source = next(iter(values["ranking_sources"]))
        by_target: dict[str, list[tuple[str, float, bool]]] = {}
        for (cell, target), (score, truth) in values["candidates"].items():
            by_target.setdefault(target, []).append((cell, score, truth))
        for budget in budgets:
            current_target_rows = []
            for target, candidates in sorted(by_target.items()):
                # Stable cell-ID tie breaking makes results independent of fold
                # completion and filesystem order.
                candidates = sorted(candidates, key=lambda value: (-value[1], value[0]))
                candidate_count = len(candidates)
                selected = min(
                    candidate_count,
                    max(1, int(np.ceil(budget * candidate_count))),
                ) if candidate_count else 0
                positives = int(sum(truth for _, _, truth in candidates))
                recovered = int(sum(
                    truth for _, _, truth in candidates[:selected]))
                recall = recovered / positives if positives else np.nan
                row = {
                    "dataset": dataset, "repetition": repetition, "model": model,
                    "target": target, "budget_fraction": budget,
                    "selected_candidates": selected,
                    "candidate_count": candidate_count,
                    "amplification_confirmed_positive_count": positives,
                    "recovered_positive_count": recovered,
                    "amplification_confirmed_recall": recall,
                    "ranking_source": ranking_source,
                    "aggregation": "pooled_oof_within_repetition_per_target",
                }
                current_target_rows.append(row)
                target_rows.append(row)
            finite_recalls = [row["amplification_confirmed_recall"]
                              for row in current_target_rows
                              if np.isfinite(row["amplification_confirmed_recall"])]
            positives = sum(row["amplification_confirmed_positive_count"]
                            for row in current_target_rows)
            recovered = sum(row["recovered_positive_count"]
                            for row in current_target_rows)
            rows.append({
                "dataset": dataset, "repetition": repetition, "model": model,
                "budget_fraction": budget,
                "selected_candidates": sum(row["selected_candidates"]
                                           for row in current_target_rows),
                "candidate_count": sum(row["candidate_count"]
                                       for row in current_target_rows),
                "amplification_confirmed_positive_count": positives,
                "recovered_positive_count": recovered,
                "amplification_confirmed_recall": (
                    float(np.mean(finite_recalls)) if finite_recalls else np.nan),
                "micro_amplification_confirmed_recall": (
                    recovered / positives if positives else np.nan),
                "n_targets": len(current_target_rows),
                "n_evaluable_targets": len(finite_recalls),
                "ranking_source": ranking_source,
                "aggregation": "pooled_oof_within_repetition",
            })
    artifacts.tables["projection_budget_recall"] = pd.DataFrame(rows, columns=columns)
    artifacts.tables["projection_budget_recall_per_target"] = pd.DataFrame(
        target_rows, columns=per_target_columns)
    _atomic_csv(artifacts.tables["projection_budget_recall"],
                artifacts.export_dir / "projection_budget_recall.csv")
    _atomic_csv(artifacts.tables["projection_budget_recall_per_target"],
                artifacts.export_dir / "projection_budget_recall_per_target.csv")


def _finish(artifacts: Artifacts) -> Artifacts:
    _add_heatmap_contrasts(artifacts)
    _add_panel_stability(artifacts)
    _add_projection_budget_recall(artifacts)
    files = set(artifacts.manifest.get("table_files", ()))
    files.update({"heatmap_baseline_selection.csv", "heatmap_contrasts.csv",
                  "panel_stability.csv", "projection_budget_recall.csv",
                  "projection_budget_recall_per_target.csv"})
    artifacts.manifest["table_files"] = sorted(files)
    atomic_json(artifacts.manifest, artifacts.export_dir / "manifest.json")
    return artifacts


def run_measurement_experiment(
    dataset: ExperimentDataset,
    settings: Settings,
    config: MeasurementConfig,
    *,
    checkpoint_dir,
    export_dir,
    progress=True,
    progress_interval=60.0,
    progress_level="summary",
    worker_status=None,
) -> Artifacts:
    """Run the configured outcome-blind panel seeds for one real dataset."""
    if dataset.metadata.get("independent_unit") == "generated_dataset":
        raise ValueError("Use run_measurement_simulation_experiments for simulation")
    views, audits = [], []
    for panel_seed in range(config.n_panel_seeds):
        current, tables = build_measurement_views(
            dataset, settings, config, repetition=panel_seed,
            panel_seed=panel_seed)
        views.extend(current)
        audits.append(tables)
    artifacts = _execute(
        views, settings, checkpoint_dir, export_dir,
        progress=progress, progress_interval=progress_interval,
        progress_level=progress_level, worker_status=worker_status,
        export_name=slug(dataset.name) + "_measurement_degradation",
        supplementary_tables=_combine_tables(audits),
        manifest_extra=_manifest(config),
    )
    return _finish(artifacts)


def run_measurement_simulation_experiments(
    settings: Settings,
    config: MeasurementConfig,
    *,
    raw_cache_dir,
    checkpoint_dir,
    export_dir,
    sharing_strengths=(0.0, 0.5, 1.0),
    simulation_options=None,
    progress=True,
    progress_interval=60.0,
    progress_level="summary",
    worker_status=None,
) -> Artifacts:
    """Generate independent simulation repetitions and apply the same protocol."""
    from .datasets.simulation import generate_simulation

    strengths = tuple(float(value) for value in sharing_strengths)
    if not strengths or len(set(strengths)) != len(strengths):
        raise ValueError("sharing_strengths must be nonempty and distinct")
    options = dict(simulation_options or {})
    if ("truth_uses_location" in options
            and bool(options["truth_uses_location"]) != settings.use_location):
        raise ValueError("Simulation truth_uses_location must match USE_LOCATION")
    options["truth_uses_location"] = settings.use_location
    Path(raw_cache_dir).mkdir(parents=True, exist_ok=True)
    views, audits = [], []
    for strength in strengths:
        for repetition in range(settings.n_repetitions):
            dataset = generate_simulation(
                repetition, strength, seed=settings.seed, **options)
            for panel_seed in range(config.n_panel_seeds):
                current, tables = build_measurement_views(
                    dataset, settings, config, repetition=repetition,
                    panel_seed=panel_seed)
                views.extend(current)
                audits.append(tables)
    artifacts = _execute(
        views, settings, checkpoint_dir, export_dir,
        progress=progress, progress_interval=progress_interval,
        progress_level=progress_level, worker_status=worker_status,
        export_name="simulation_measurement_degradation",
        supplementary_tables=_combine_tables(audits),
        manifest_extra={**_manifest(config),
                        "simulation_sharing_strengths": strengths,
                        "simulation_options": options},
    )
    return _finish(artifacts)


def preview_measurement_experiment(
    dataset: ExperimentDataset,
    settings: Settings,
    config: MeasurementConfig,
    *,
    repetition: int = 0,
) -> dict:
    """Return exact views and audit tables without fitting a model."""
    views, tables = build_measurement_views(
        dataset, settings, config, repetition=repetition)
    return {"views": views, "tables": tables}


__all__ = [
    "GENE_COVERAGE_GRID",
    "RETENTION_GRID",
    "TARGET_COVERAGE_GRID",
    "MaskedGeneFeatureBuilder",
    "MeasurementConfig",
    "build_measurement_views",
    "preview_measurement_experiment",
    "run_measurement_experiment",
    "run_measurement_simulation_experiments",
]
