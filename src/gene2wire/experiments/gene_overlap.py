"""Leak-safe partial gene-panel overlap experiments.

Only cell-gene inputs are hidden. Projection targets, outcomes, assay masks and
outer splits remain unchanged. Missing expression is centered to the fold's
visible training mean and represented by zero after standardization; hidden raw
values are never read while fitting preprocessing statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..seeds import stable_seed
from .contracts import ExperimentDataset, FeatureSet, Fold
from .pipeline import Artifacts, _execute, _atomic_csv
from .io import atomic_json
from .protocol import Settings


PRIMARY_MODELS = ("PU", "PU-MIRT", "PU-Joint")
PANEL_LABELS = ("A", "B")


@dataclass(frozen=True)
class PanelFitSpec:
    """JSON-safe orchestration contract for one gene-overlap arm.

    ``single`` arms enter the ordinary shared pipeline once. ``partitioned``
    arms require one fit per panel followed by prediction recombination in the
    original outer-test order.  Detection pooling is stated separately because
    a no-coefficient-pooling control need not also change the observation
    model.  The experiment module only declares this contract; the pipeline is
    responsible for executing it without exposing evaluation labels.
    """

    arm: str
    feature_layout: str
    model_allowlist: tuple[str, ...]
    fit_mode: str = "single"
    detector_pooling: str = "pooled"
    partition_group: str | None = None
    partition_feature_blocks: tuple[tuple[str, str], ...] = ()
    lowrank_feature_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.feature_layout not in {"union", "intersection", "disjoint", "all_gene"}:
            raise ValueError("Unknown overlap feature layout")
        if self.fit_mode not in {"single", "partitioned"}:
            raise ValueError("fit_mode must be single or partitioned")
        if self.detector_pooling not in {"pooled", "by_partition"}:
            raise ValueError("detector_pooling must be pooled or by_partition")
        partitioned = self.fit_mode == "partitioned"
        if partitioned != bool(self.partition_group):
            raise ValueError("partitioned fits require exactly one partition_group")
        if partitioned:
            blocks = dict(self.partition_feature_blocks)
            if set(blocks) != set(PANEL_LABELS) or any(not value for value in blocks.values()):
                raise ValueError("partitioned fits require one feature block for panels A and B")
        elif self.partition_feature_blocks:
            raise ValueError("single fits cannot declare partition feature blocks")
        if self.lowrank_feature_groups and (
            self.feature_layout != "disjoint" or self.model_allowlist != ("PU-MIRT",)
        ):
            raise ValueError("lowrank feature groups are defined only for disjoint PU-MIRT")

    def to_dict(self) -> dict:
        result = {
            "fit_mode": self.fit_mode,
            "detector_pooling": self.detector_pooling,
            "combine_test_predictions": self.fit_mode == "partitioned",
            "requires_partitioned_executor": self.fit_mode == "partitioned",
        }
        if self.fit_mode == "partitioned":
            result.update({
                "group": self.partition_group,
                "labels": list(PANEL_LABELS),
                "feature_blocks": dict(self.partition_feature_blocks),
                "component_coordinate": "fit_panel",
            })
        if self.lowrank_feature_groups:
            result["lowrank_feature_groups"] = list(self.lowrank_feature_groups)
        return result


def canonical_overlap_arm(arm: str) -> str:
    """Return a declared v2 arm name; archived semantics are never aliased.

    In v1, ``separate_panel`` meant one fit with disjoint coefficients.  That
    behavior is now named ``disjoint_coefficient``; reusing the old meaning
    under the same name would make saved results scientifically ambiguous.
    """
    value = str(arm)
    if value not in {
        "union", "intersection", "disjoint_coefficient",
        "shared_a", "separate_a", "independent_panel", "all_gene_oracle",
    }:
        raise ValueError(f"Unknown overlap arm {arm!r}")
    return value


def panel_fit_spec(arm: str) -> PanelFitSpec:
    """Declare feature, model, fit-partition and detector semantics by arm."""
    arm = canonical_overlap_arm(arm)
    ordinary = {
        "union": PanelFitSpec(arm, "union", PRIMARY_MODELS),
        "intersection": PanelFitSpec(arm, "intersection", ("PU",)),
        "disjoint_coefficient": PanelFitSpec(arm, "disjoint", ("PU",)),
        "shared_a": PanelFitSpec(arm, "disjoint", ("PU-MIRT",)),
        "separate_a": PanelFitSpec(
            arm, "disjoint", ("PU-MIRT",),
            lowrank_feature_groups=("panel_A_gene", "panel_B_gene"),
        ),
        "all_gene_oracle": PanelFitSpec(arm, "all_gene", ("PU",)),
    }
    if arm in ordinary:
        return ordinary[arm]
    return PanelFitSpec(
        arm, "disjoint", ("PU",), fit_mode="partitioned",
        detector_pooling="by_partition", partition_group="overlap_panel",
        partition_feature_blocks=(("A", "panel_A_gene"), ("B", "panel_B_gene")),
    )


@dataclass(frozen=True)
class PanelDraw:
    panel_a: tuple[int, ...]
    panel_b: tuple[int, ...]
    common: tuple[int, ...]
    requested_overlap: float

    @property
    def panel_size(self) -> int:
        return len(self.panel_a)

    @property
    def actual_overlap(self) -> float:
        return len(self.common) / self.panel_size


def validate_overlap_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid or len(set(grid)) != len(grid) or any(not np.isfinite(x) or x < 0 or x > 1 for x in grid):
        raise ValueError("OVERLAP_GRID must be a nonempty list of unique values in [0, 1]")
    return grid


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + .5))


def draw_gene_panels(n_genes: int, panel_size: int, overlap_grid: Sequence[float],
                     *, seed: int) -> dict[float, PanelDraw]:
    """Draw nested equal-budget panels without consulting projection labels."""
    grid = validate_overlap_grid(overlap_grid)
    if isinstance(panel_size, bool) or not isinstance(panel_size, (int, np.integer)):
        raise TypeError("panel_size must be an integer")
    if not 1 <= int(panel_size) <= int(n_genes):
        raise ValueError("panel_size must lie between one and the gene count")
    counts = [_round_half_up(int(panel_size) * overlap) for overlap in grid]
    if len(set(counts)) != len(counts):
        raise ValueError("OVERLAP_GRID maps to duplicate overlap-gene counts; use a coarser grid")
    if 2 * int(panel_size) - min(counts) > int(n_genes):
        raise ValueError("The requested panel size/overlap requires more genes than available")
    order = np.random.default_rng(seed).permutation(int(n_genes))
    panel_a_order = order[:int(panel_size)]
    outside_order = order[int(panel_size):]
    output: dict[float, PanelDraw] = {}
    for requested, count in zip(grid, counts):
        panel_a = tuple(sorted(map(int, panel_a_order)))
        panel_b = tuple(sorted(map(int, np.r_[panel_a_order[:count], outside_order[:int(panel_size)-count]])))
        common = tuple(sorted(set(panel_a).intersection(panel_b)))
        output[requested] = PanelDraw(panel_a, panel_b, common, requested)
    return output


def crossed_panel_assignment(strata: Sequence[object], *, seed: int) -> np.ndarray:
    """Balance pseudo-panels within every outcome-independent stratum."""
    values = np.asarray(strata).astype(str)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("strata must be a one-dimensional cell-aligned vector")
    panels = np.empty(len(values), dtype="U1")
    rng = np.random.default_rng(seed)
    for stratum in np.unique(values):
        rows = np.flatnonzero(values == stratum)
        shuffled = rng.permutation(rows)
        panels[shuffled[::2]] = "A"
        panels[shuffled[1::2]] = "B"
    if set(panels) != {"A", "B"}:
        raise ValueError("Panel assignment must contain both A and B")
    return panels


def composite_strata(dataset: ExperimentDataset, names: Sequence[str]) -> np.ndarray:
    if not names or any(name not in dataset.groups for name in names):
        raise ValueError("Every requested crossed-panel stratum must exist in dataset.groups")
    columns = [np.asarray(dataset.groups[name]).astype(str) for name in names]
    return np.asarray(["|".join(items) for items in zip(*columns)], dtype=str)


def aligned_panel_assignment(groups: Sequence[object], mapping: Mapping[object, str]) -> np.ndarray:
    values = np.asarray(groups)
    result = np.asarray([mapping.get(value, mapping.get(str(value), "")) for value in values], dtype="U1")
    if np.any(~np.isin(result, ["A", "B"])) or set(result) != {"A", "B"}:
        raise ValueError("Every aligned group must map to A or B and both panels must occur")
    return result


def partition_rows(rows: Sequence[int], panels: Sequence[object],
                   labels: Sequence[str] = PANEL_LABELS) -> dict[str, np.ndarray]:
    """Split authorized row indices by panel without inspecting any outcome."""
    panel_values = np.asarray(panels).astype(str)
    selected = np.asarray(rows)
    if panel_values.ndim != 1:
        raise ValueError("panels must be one-dimensional")
    if (selected.ndim != 1 or (selected.size and not np.issubdtype(selected.dtype, np.integer))
            or np.any(selected < 0) or np.any(selected >= len(panel_values))
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("rows must be unique in-range integer indices")
    requested = tuple(map(str, labels))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("labels must be a nonempty unique sequence")
    output = {label: np.sort(selected[panel_values[selected] == label].astype(int))
              for label in requested}
    empty = [label for label, values in output.items() if not len(values)]
    if empty:
        raise ValueError(f"Every fit partition must occur in this split; empty: {empty}")
    if sum(map(len, output.values())) != len(selected):
        unexpected = sorted(set(panel_values[selected]).difference(requested))
        raise ValueError(f"Selected rows contain undeclared panel labels: {unexpected}")
    return output


def validate_partition_fold(fold: Fold, panels: Sequence[object],
                            labels: Sequence[str] = PANEL_LABELS) -> None:
    """Fail before fitting when a panel is absent from any outer-fold role."""
    for role, rows in (("train", fold.train_rows), ("validation", fold.validation_rows),
                       ("test", fold.test_rows)):
        try:
            partition_rows(rows, panels, labels)
        except ValueError as error:
            raise ValueError(f"Partitioned fit has an invalid {role} split: {error}") from error


def panel_feature_subset(features: FeatureSet, label: str,
                         spec: PanelFitSpec | Mapping[str, object]) -> FeatureSet:
    """Select one panel's K gene columns for a partitioned component fit.

    The panel nuisance column is intentionally excluded: a component model's
    ordinary target intercept already supplies its panel-specific intercept.
    """
    fit = spec.to_dict() if isinstance(spec, PanelFitSpec) else dict(spec)
    if fit.get("fit_mode") != "partitioned":
        raise ValueError("panel_feature_subset requires a partitioned fit spec")
    label = str(label)
    blocks = fit.get("feature_blocks")
    if not isinstance(blocks, Mapping) or label not in blocks:
        raise ValueError(f"No feature block is declared for panel {label!r}")
    block_name = str(blocks[label])
    if block_name not in features.feature_blocks:
        raise ValueError(f"FeatureSet does not contain required block {block_name!r}")
    columns = tuple(map(int, features.feature_blocks[block_name]))
    if not columns:
        raise ValueError("A partitioned component cannot have zero gene features")
    names = (() if not features.feature_names else
             tuple(features.feature_names[index] for index in columns))
    metadata = {**dict(features.metadata), "fit_panel": label,
                "source_feature_block": block_name}
    return replace(
        features, X=np.asarray(features.X)[:, columns],
        feature_blocks={"gene_panel": tuple(range(len(columns)))},
        feature_names=names, metadata=metadata,
        X_nuisance=None, nuisance_names=(),
    )


def combine_partition_predictions(test_rows: Sequence[int], panels: Sequence[object],
                                  predictions: Mapping[str, np.ndarray],
                                  labels: Sequence[str] = PANEL_LABELS) -> np.ndarray:
    """Recombine component predictions in the original outer-test row order."""
    test = np.asarray(test_rows)
    by_panel = partition_rows(test, panels, labels)
    if set(map(str, predictions)) != set(map(str, labels)):
        raise ValueError("predictions must contain exactly the declared panel labels")
    n_targets = None
    result = None
    test_positions = {int(row): index for index, row in enumerate(test)}
    for label in map(str, labels):
        value = np.asarray(predictions[label], dtype=float)
        rows = by_panel[label]
        if value.ndim != 2 or value.shape[0] != len(rows):
            raise ValueError(f"Panel {label} prediction rows do not align to its test cells")
        if n_targets is None:
            n_targets = value.shape[1]
            result = np.full((len(test), n_targets), np.nan, dtype=float)
        if value.shape[1] != n_targets or not np.all(np.isfinite(value)):
            raise ValueError("Panel predictions must have matching finite target columns")
        positions = np.asarray([test_positions[int(row)] for row in rows], dtype=int)
        result[positions] = value
    assert result is not None
    if not np.all(np.isfinite(result)):
        raise ValueError("Every outer-test row must receive exactly one finite prediction")
    return result


def paired_group_folds(dataset: ExperimentDataset, group_name: str,
                       pairs: Sequence[Sequence[object]]) -> ExperimentDataset:
    """Use one complete A/B group pair for each train/validation/test role."""
    if group_name not in dataset.groups or len(pairs) != 3:
        raise ValueError("paired_group_folds requires three pairs from a declared group")
    groups = np.asarray(dataset.groups[group_name])
    normalized = tuple(tuple(pair) for pair in pairs)
    flat = [value for pair in normalized for value in pair]
    if any(len(pair) != 2 for pair in normalized) or len(set(map(str, flat))) != 6:
        raise ValueError("pairs must contain six distinct groups")
    if set(map(str, np.unique(groups))) != set(map(str, flat)):
        raise ValueError("pairs must cover every dataset group exactly once")

    def split_builder(n_outer_folds=3, seed=0):
        del seed
        if int(n_outer_folds) != 3:
            raise ValueError("The paired A/B animal design is fixed at three outer folds")
        output = []
        for outer in range(3):
            test_groups = normalized[outer]
            validation_groups = normalized[(outer + 1) % 3]
            train_groups = normalized[(outer + 2) % 3]
            fold = Fold(
                outer, np.flatnonzero(np.isin(groups, train_groups)),
                np.flatnonzero(np.isin(groups, validation_groups)),
                np.flatnonzero(np.isin(groups, test_groups)),
                {"split": "paired_group_train_validation_test",
                 "train_groups": list(map(str, train_groups)),
                 "validation_groups": list(map(str, validation_groups)),
                 "test_groups": list(map(str, test_groups))},
            )
            fold.validate(len(groups))
            output.append(fold)
        return tuple(output)
    result = replace(dataset, split_builder=split_builder)
    result.validate()
    return result


def common_target_dataset(dataset: ExperimentDataset, group_name: str) -> ExperimentDataset:
    """Keep only targets whose measurement mask covers every cell in every group."""
    if group_name not in dataset.groups:
        raise ValueError(f"Unknown target-coverage group {group_name!r}")
    groups = np.asarray(dataset.groups[group_name])
    keep = np.ones(len(dataset.target_ids), dtype=bool)
    for group in np.unique(groups):
        keep &= np.asarray(dataset.measured)[groups == group].all(axis=0)
    if np.sum(keep) < 2:
        raise ValueError("Fewer than two targets have identical assay coverage across all groups")
    indices = np.flatnonzero(keep)
    original_features = dataset.feature_builder

    def features(train_rows, use_location=False, use_target_features=False):
        result = original_features(train_rows, use_location, use_target_features)
        target = None if result.Y_target is None else np.asarray(result.Y_target)[indices]
        return replace(result, Y_target=target)

    metadata = dict(dataset.metadata)
    if (isinstance(metadata.get("target_audit"), list)
            and len(metadata["target_audit"]) == len(dataset.target_ids)):
        metadata["source_target_audit"] = metadata["target_audit"]
        metadata["target_audit"] = [metadata["target_audit"][index]
                                    for index in indices]
    metadata["common_target_audit"] = {
        "group": group_name, "source_target_count": len(dataset.target_ids),
        "retained_target_count": int(np.sum(keep)),
        "retained_targets": [dataset.target_ids[index] for index in indices],
    }
    result = replace(
        dataset, reference=np.asarray(dataset.reference)[:, indices],
        measured=np.asarray(dataset.measured)[:, indices],
        target_ids=tuple(dataset.target_ids[index] for index in indices),
        natural_observed=(None if dataset.natural_observed is None else
                          np.asarray(dataset.natural_observed)[:, indices]),
        feature_builder=features, metadata=metadata,
    )
    result.validate()
    return result


def _availability(panels: np.ndarray, draw: PanelDraw, n_genes: int) -> np.ndarray:
    available = np.zeros((len(panels), n_genes), dtype=bool)
    available[np.ix_(panels == "A", np.asarray(draw.panel_a, dtype=int))] = True
    available[np.ix_(panels == "B", np.asarray(draw.panel_b, dtype=int))] = True
    return available


def _visible_standardize(raw: np.ndarray, availability: np.ndarray,
                         train_rows: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    columns = tuple(map(int, columns))
    output = np.zeros((len(raw), len(columns)), dtype=float)
    train_mask = np.zeros(len(raw), dtype=bool)
    train_mask[np.asarray(train_rows, dtype=int)] = True
    for out_index, gene_index in enumerate(columns):
        fit = train_mask & availability[:, gene_index]
        if not np.any(fit):
            raise ValueError(f"Gene {gene_index} has no visible outer-training cells")
        fit_values = raw[fit, gene_index]
        center = float(np.mean(fit_values))
        scale = float(np.std(fit_values))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        visible = availability[:, gene_index]
        # Deliberately index only visible values: hidden raw expression never
        # enters preprocessing, even transiently before being overwritten.
        output[visible, out_index] = (raw[visible, gene_index] - center) / scale
    return output


def _panel_feature_builder(
    dataset: ExperimentDataset,
    panels: np.ndarray,
    draw: PanelDraw,
    arm: str,
    nuisance_groups: Sequence[str] = (),
):
    if dataset.gene_matrix is None:
        raise ValueError("The dataset adapter does not expose a gene_matrix")
    raw = np.asarray(dataset.gene_matrix, dtype=float)
    n_genes = raw.shape[1]
    spec = panel_fit_spec(arm)

    def builder(train_rows, use_location=False, use_target_features=False):
        if use_location or use_target_features:
            raise ValueError("Gene-overlap primary notebooks require USE_LOCATION=False and USE_TARGET_FEATURES=False")
        train = np.asarray(train_rows, dtype=int)
        if train.ndim != 1 or not len(train) or len(np.unique(train)) != len(train):
            raise ValueError("Training rows must be a nonempty unique integer vector")
        # Derive this compact boolean mask on demand rather than retaining one
        # full cell-by-gene array in every deferred SPIDER view closure.
        available = _availability(panels, draw, n_genes)
        nuisance_columns = [(panels == "B").astype(float)]
        nuisance_names = ["overlap_panel[B]"]
        for group_name in nuisance_groups:
            if group_name not in dataset.groups:
                raise ValueError(f"Unknown nuisance group {group_name!r}")
            values = np.asarray(dataset.groups[group_name]).astype(str)
            levels = sorted(np.unique(values), key=str)
            for level in levels[1:]:
                nuisance_columns.append((values == level).astype(float))
                nuisance_names.append(f"{group_name}[{level}]")
        nuisance = np.column_stack(nuisance_columns)
        if spec.feature_layout == "union":
            columns = tuple(sorted(set(draw.panel_a).union(draw.panel_b)))
            x_gene = _visible_standardize(raw, available, train, columns)
            names = tuple(dataset.gene_names[index] for index in columns)
            blocks = {"gene_union": tuple(range(len(columns)))}
        elif spec.feature_layout == "intersection":
            columns = draw.common
            if columns:
                x_gene = _visible_standardize(raw, np.ones_like(available), train, columns)
                names = tuple(dataset.gene_names[index] for index in columns)
            else:
                x_gene = np.zeros((len(raw), 1), dtype=float)
                names = ("no_common_gene_intercept_proxy",)
            blocks = {"gene_intersection": tuple(range(x_gene.shape[1]))}
        elif spec.feature_layout == "disjoint":
            pieces, names_list = [], []
            for label, columns in (("A", draw.panel_a), ("B", draw.panel_b)):
                panel_available = np.zeros_like(available)
                panel_available[np.ix_(panels == label, np.asarray(columns, dtype=int))] = True
                pieces.append(_visible_standardize(raw, panel_available, train, columns))
                names_list.extend(f"panel_{label}:{dataset.gene_names[index]}" for index in columns)
            x_gene = np.column_stack(pieces)
            names = tuple(names_list)
            blocks = {"panel_A_gene": tuple(range(draw.panel_size)),
                      "panel_B_gene": tuple(range(draw.panel_size, 2 * draw.panel_size))}
        elif spec.feature_layout == "all_gene":
            columns = tuple(range(n_genes))
            x_gene = _visible_standardize(raw, np.ones_like(available), train, columns)
            names = tuple(dataset.gene_names)
            blocks = {"all_gene_oracle": tuple(range(n_genes))}
        else:
            raise RuntimeError(f"Unhandled overlap feature layout {spec.feature_layout!r}")
        return FeatureSet(
            X=x_gene, feature_blocks=blocks, feature_names=names,
            metadata={"preprocessing": "visible-training-only standardization; hidden values are mean-zero",
                      "panel_A_genes": [dataset.gene_names[i] for i in draw.panel_a],
                      "panel_B_genes": [dataset.gene_names[i] for i in draw.panel_b],
                      "common_genes": [dataset.gene_names[i] for i in draw.common],
                      "arm": arm, "fit_rows": train.tolist()},
            X_nuisance=nuisance, nuisance_names=tuple(nuisance_names),
        )
    return builder


def overlap_view(dataset: ExperimentDataset, panels: Sequence[str], draw: PanelDraw, *,
                 repetition: int, panel_design: str, arm: str,
                 nuisance_groups: Sequence[str] = ()) -> ExperimentDataset:
    arm = canonical_overlap_arm(arm)
    spec = panel_fit_spec(arm)
    panels = np.asarray(panels).astype(str)
    if panels.shape != (len(dataset.cell_ids),) or set(panels) != {"A", "B"}:
        raise ValueError("panels must align to cells and contain A/B")
    context = {
        "experiment": "gene_overlap", "panel_design": panel_design, "arm": arm,
        "requested_overlap": float(draw.requested_overlap),
        "actual_overlap": float(draw.actual_overlap), "panel_size": draw.panel_size,
        "rank_cap": draw.panel_size, "fit_mode": spec.fit_mode,
        "detector_pooling": spec.detector_pooling,
        "nuisance_groups": list(map(str, nuisance_groups)),
    }
    metadata = dict(dataset.metadata)
    metadata.update({"experiment_context": context, "experiment_repetition": int(repetition),
                     "evaluation_group": "overlap_panel",
                     "model_allowlist": spec.model_allowlist,
                     "tuning_rank_cap": draw.panel_size,
                     "panel_fit_spec": spec.to_dict(),
                     "fit_partition": (spec.to_dict()
                                       if spec.fit_mode == "partitioned" else None),
                     "model_lowrank_feature_groups": spec.lowrank_feature_groups,
                     "gene_overlap": {"panel_A": list(draw.panel_a), "panel_B": list(draw.panel_b),
                                      "common": list(draw.common)}})
    groups = {**dataset.groups, "overlap_panel": panels}
    result = replace(dataset, feature_builder=_panel_feature_builder(
                         dataset, panels, draw, arm, nuisance_groups),
                     groups=groups, metadata=metadata)
    result.validate()
    return result


def build_overlap_views(dataset: ExperimentDataset, overlap_grid: Sequence[float], *,
                        panel_size: int | None = None, n_repetitions: int = 5,
                        panel_design: str = "crossed", strata: Sequence[str] = ("slice",),
                        aligned_mapping: Mapping[object, str] | None = None,
                        aligned_group: str = "animal",
                        include_controls: bool = True,
                        include_shared_a_ablation: bool = True,
                        include_independent_panel_sensitivity: bool = False,
                        nuisance_groups: Sequence[str] = (),
                        seed: int = 20260908) -> tuple[ExperimentDataset, ...]:
    dataset.validate()
    if dataset.gene_matrix is None:
        raise ValueError("Gene-overlap experiments require adapter gene_matrix/gene_names")
    p = np.asarray(dataset.gene_matrix).shape[1]
    k = p // 2 if panel_size is None else int(panel_size)
    grid = validate_overlap_grid(overlap_grid)
    if not any(np.isclose(value, 1.0, rtol=0, atol=1e-12) for value in grid):
        raise ValueError(
            "OVERLAP_GRID must include 1.0 so R_M(omega) has its matched "
            "same-K reference endpoint"
        )
    views = []
    repetitions = ((int(dataset.metadata["repetition"]),)
                   if dataset.metadata.get("independent_unit") == "generated_dataset"
                   and "repetition" in dataset.metadata
                   else range(int(n_repetitions)))
    for repetition in repetitions:
        if panel_design == "crossed":
            panels = crossed_panel_assignment(
                composite_strata(dataset, strata),
                seed=stable_seed(seed, dataset.name, panel_design, "assignment", repetition))
        elif panel_design == "animal_aligned":
            if aligned_mapping is None or aligned_group not in dataset.groups:
                raise ValueError("animal_aligned design requires an aligned group mapping")
            panels = aligned_panel_assignment(dataset.groups[aligned_group], aligned_mapping)
        else:
            raise ValueError("panel_design must be crossed or animal_aligned")
        draws = draw_gene_panels(p, k, grid,
            # Gene panels are paired across crossed/aligned designs.  Only the
            # cell-to-panel assignment is allowed to depend on panel_design.
            seed=stable_seed(seed, dataset.name, "genes", repetition))
        for requested in grid:
            draw = draws[requested]
            views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                      panel_design=panel_design, arm="union",
                                      nuisance_groups=nuisance_groups))
            if include_controls:
                views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                          panel_design=panel_design, arm="intersection",
                                          nuisance_groups=nuisance_groups))
                views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                          panel_design=panel_design, arm="disjoint_coefficient",
                                          nuisance_groups=nuisance_groups))
                if include_shared_a_ablation:
                    views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                              panel_design=panel_design, arm="shared_a",
                                              nuisance_groups=nuisance_groups))
                    views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                              panel_design=panel_design, arm="separate_a",
                                              nuisance_groups=nuisance_groups))
                if include_independent_panel_sensitivity:
                    views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                              panel_design=panel_design, arm="independent_panel",
                                              nuisance_groups=nuisance_groups))
        if include_controls:
            # Reuse the actual same-K endpoint draw in audit metadata.  The
            # oracle feature builder itself exposes all P genes.
            oracle_draw = next(
                draw for requested, draw in draws.items()
                if np.isclose(requested, 1.0, rtol=0, atol=1e-12)
            )
            views.append(overlap_view(dataset, panels, oracle_draw, repetition=repetition,
                                      panel_design=panel_design, arm="all_gene_oracle",
                                      nuisance_groups=nuisance_groups))
    return tuple(views)


_OVERLAP_SCENARIO_CONTEXT = (
    "dataset", "sharing_strength", "analysis", "mechanism", "loss_rate",
    "calibration_fraction", "calibration_spec", "experiment", "group_mode",
    "training_panel", "training_block_fraction", "evaluation_scope",
    "block_fraction", "panel_design", "panel_size", "probability_semantics",
)


def shared_a_contrasts(per_repetition: pd.DataFrame) -> pd.DataFrame:
    """Pair Shared-A and Separate-A scores within every repetition/scenario.

    Metric signs are normalized so positive always favors Shared-A.  The
    endpoint-subtracted value asks whether Shared-A becomes relatively more
    useful as gene overlap falls, beyond its advantage at 100% overlap.
    """
    required = {"arm", "model", "actual_overlap", "repetition"}
    if per_repetition.empty or not required.issubset(per_repetition):
        return pd.DataFrame()
    selected = per_repetition.loc[
        per_repetition["arm"].isin(("shared_a", "separate_a"))
        & per_repetition["model"].eq("PU-MIRT")
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    context = [column for column in (*_OVERLAP_SCENARIO_CONTEXT, "repetition")
               if column in selected]
    rows = []
    for values, frame in selected.groupby(context, dropna=False, observed=True):
        prefix = dict(zip(context, values if isinstance(values, tuple) else (values,)))
        shared = frame.loc[frame.arm.eq("shared_a")].set_index("actual_overlap")
        separate = frame.loc[frame.arm.eq("separate_a")].set_index("actual_overlap")
        if shared.index.has_duplicates or separate.index.has_duplicates:
            raise ValueError(
                "Shared-A contrasts require one row per arm/scenario/repetition/overlap"
            )
        overlaps = sorted(set(shared.index).intersection(separate.index))
        if not any(np.isclose(overlap, 1.0) for overlap in overlaps):
            continue
        endpoint_overlap = next(overlap for overlap in overlaps if np.isclose(overlap, 1.0))
        for metric, direction in (("macro_auprc", 1.), ("macro_log_loss", -1.),
                                  ("macro_brier", -1.)):
            if metric not in shared or metric not in separate:
                continue
            endpoint = direction * (
                shared.at[endpoint_overlap, metric] - separate.at[endpoint_overlap, metric]
            )
            for overlap in overlaps:
                advantage = direction * (shared.at[overlap, metric] - separate.at[overlap, metric])
                rows.append({
                    **prefix, "metric": metric, "actual_overlap": overlap,
                    "shared_a_minus_separate_a": advantage,
                    "difference_vs_100pct_overlap": advantage - endpoint,
                })
    return pd.DataFrame(rows)


def _write_overlap_summaries(artifacts: Artifacts, *, persist: bool = True) -> None:
    per_group = artifacts.tables.get("per_group", pd.DataFrame())
    if not per_group.empty:
        keys = [column for column in (
            *_OVERLAP_SCENARIO_CONTEXT, "arm", "requested_overlap", "actual_overlap",
            "model", "repetition", "outer_fold",
        ) if column in per_group]
        metrics = [column for column in ("macro_auprc", "macro_log_loss", "macro_brier")
                   if column in per_group]
        worst_rows = []
        for values, frame in per_group.groupby(keys, dropna=False, observed=True):
            prefix = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
            row = dict(prefix)
            for metric in metrics:
                row[f"worst_panel_{metric}"] = (frame[metric].min() if metric == "macro_auprc"
                                                    else frame[metric].max())
            worst_rows.append(row)
        artifacts.tables["worst_panel"] = pd.DataFrame(worst_rows)
        if persist:
            _atomic_csv(artifacts.tables["worst_panel"], artifacts.export_dir / "worst_panel.csv")

    per_rep = artifacts.tables.get("per_repetition", pd.DataFrame())
    required = {"arm", "model", "actual_overlap", "repetition"}
    if per_rep.empty or not required.issubset(per_rep):
        return
    # Keep every fixed scenario coordinate in the matching key.  In particular,
    # simulation artifacts contain several sharing strengths; dropping rho makes
    # a baseline lookup return a Series instead of one scalar and corrupts R_M.
    context = [column for column in (*_OVERLAP_SCENARIO_CONTEXT, "repetition")
               if column in per_rep]
    union = per_rep[per_rep["arm"] == "union"].copy()
    rows = []
    for values, frame in union.groupby(context, dropna=False, observed=True):
        prefix = dict(zip(context, values if isinstance(values, tuple) else (values,)))
        baseline = frame[frame["model"] == "PU"].set_index("actual_overlap")
        endpoints = frame[np.isclose(frame["actual_overlap"], 1.0)].set_index("model")
        if baseline.index.has_duplicates or endpoints.index.has_duplicates:
            raise ValueError(
                "Gene-overlap summaries require one PU baseline and one model endpoint "
                "per scenario/repetition/actual_overlap; include all scenario keys."
            )
        if "PU" not in endpoints.index:
            continue
        for _, candidate in frame[frame["model"].isin(("PU-MIRT", "PU-Joint"))].iterrows():
            overlap = candidate["actual_overlap"]
            if overlap not in baseline.index or candidate["model"] not in endpoints.index:
                continue
            for metric, direction in (("macro_auprc", 1.), ("macro_log_loss", -1.), ("macro_brier", -1.)):
                if metric not in frame:
                    continue
                current = direction * (candidate[metric] - baseline.at[overlap, metric])
                endpoint = direction * (
                    endpoints.at[candidate["model"], metric] - endpoints.at["PU", metric]
                )
                rows.append({**prefix, "model": candidate["model"], "metric": metric,
                             "actual_overlap": overlap, "joint_or_mirt_minus_pu": current,
                             "difference_vs_100pct_overlap": current - endpoint})
    artifacts.tables["overlap_contrasts"] = pd.DataFrame(rows)
    if persist:
        _atomic_csv(artifacts.tables["overlap_contrasts"], artifacts.export_dir / "overlap_contrasts.csv")
    shared = shared_a_contrasts(per_rep)
    if not shared.empty:
        artifacts.tables["shared_a_contrasts"] = shared
        if persist:
            _atomic_csv(shared, artifacts.export_dir / "shared_a_contrasts.csv")


def rebuild_overlap_summaries(artifacts: Artifacts, *, persist: bool = True) -> Artifacts:
    """Rebuild overlap-specific tables and persist them in an existing export.

    This is intentionally safe for ``RESULTS_ONLY`` notebooks: it only reads
    the saved per-group/per-repetition tables and rewrites derived summaries;
    it never launches model fitting.
    """
    _write_overlap_summaries(artifacts, persist=persist)
    table_files = {str(name) + ".csv" for name in artifacts.tables}
    artifacts.manifest["table_files"] = sorted(table_files)
    if persist:
        atomic_json(artifacts.manifest, artifacts.export_dir / "manifest.json")
    return artifacts


def validate_overlap_artifact(
    artifacts: Artifacts,
    *,
    expected_panel_design: str | None = None,
    expected_overlap_grid: Sequence[float] | None = None,
    require_release_v2: bool = False,
    tech_sar80: bool = False,
) -> None:
    """Reject a RESULTS_ONLY directory from the wrong overlap experiment.

    Dataset names alone cannot distinguish A1 crossed from A1 animal-aligned,
    and an unrelated historical export can have the same name.  Validation is
    deliberately based on the saved manifest and result coordinates; it never
    opens raw data or launches fitting.
    """

    manifest = dict(artifacts.manifest)
    if manifest.get("experiment") != "gene_overlap":
        raise ValueError("Saved export is not a gene-overlap experiment")
    contexts = []
    for entry in manifest.get("datasets", []):
        metadata = entry.get("metadata", {}) if isinstance(entry, Mapping) else {}
        context = metadata.get("experiment_context", {})
        if context:
            contexts.append(context)
    if not contexts or any(context.get("experiment") != "gene_overlap"
                           for context in contexts):
        raise ValueError("Saved export lacks valid gene-overlap dataset contexts")
    designs = {str(context.get("panel_design")) for context in contexts}
    if expected_panel_design is not None and designs != {str(expected_panel_design)}:
        raise ValueError(
            f"Saved export panel design {sorted(designs)} does not match "
            f"{expected_panel_design!r}"
        )

    aggregate = artifacts.tables.get("aggregate", pd.DataFrame())
    if aggregate.empty:
        raise ValueError("Saved gene-overlap export has no aggregate results")
    if expected_overlap_grid is not None:
        expected_grid = validate_overlap_grid(expected_overlap_grid)
        if "requested_overlap" not in aggregate or "arm" not in aggregate:
            raise ValueError("Saved export lacks requested overlap/arm coordinates")
        union_grid = sorted(pd.to_numeric(
            aggregate.loc[aggregate["arm"].eq("union"), "requested_overlap"],
            errors="coerce",
        ).dropna().unique())
        if len(union_grid) != len(expected_grid) or not np.allclose(
            union_grid, expected_grid, rtol=0, atol=1e-12
        ):
            raise ValueError(
                f"Saved export overlap grid {union_grid} does not match "
                f"{list(expected_grid)}"
            )
    if tech_sar80:
        required = {"analysis", "mechanism", "loss_rate", "arm"}
        if not required.issubset(aggregate):
            raise ValueError("Tech-SAR sensitivity export lacks scenario coordinates")
        rates = pd.to_numeric(aggregate["loss_rate"], errors="coerce").dropna()
        if (set(aggregate["analysis"].dropna().astype(str))
                != {"positive_label_loss_sensitivity"}
                or set(aggregate["mechanism"].dropna().astype(str)) != {"technical_sar"}
                or rates.empty or not np.all(np.isclose(rates, .8, rtol=0, atol=1e-12))
                or set(aggregate["arm"].dropna().astype(str)) != {"union"}):
            raise ValueError("Saved export is not the isolated union-only Tech-SAR 80% sensitivity")
    else:
        rates = (pd.to_numeric(aggregate["loss_rate"], errors="coerce").dropna()
                 if "loss_rate" in aggregate else pd.Series(dtype=float))
        if not rates.empty and not np.all(np.isclose(rates, 0., rtol=0, atol=1e-12)):
            raise ValueError("Primary gene-overlap export contains added positive-label loss")

    if not require_release_v2:
        return
    if not bool(manifest.get("completed")):
        raise ValueError("Saved gene-overlap export is not marked complete")
    protocol = manifest.get("protocol", {})
    nuisance_l2 = protocol.get("nuisance_l2") if isinstance(protocol, Mapping) else None
    if nuisance_l2 is None or not np.isclose(
        float(nuisance_l2), 1e-4, rtol=0, atol=1e-12
    ):
        raise ValueError(
            "Saved export does not use the v2 release nuisance_l2=1e-4 contract"
        )
    required_columns = {"arm", "model", "requested_overlap", "actual_overlap"}
    if not required_columns.issubset(aggregate):
        raise ValueError("Saved export lacks v2 arm/model/overlap coordinates")
    arm_models = {
        str(arm): set(frame["model"].dropna().astype(str))
        for arm, frame in aggregate.groupby("arm", dropna=False, observed=True)
    }
    if tech_sar80:
        expected_arm_models = {"union": set(PRIMARY_MODELS)}
    else:
        expected_arm_models = {
            "union": set(PRIMARY_MODELS),
            "intersection": {"PU"},
            "disjoint_coefficient": {"PU"},
            "shared_a": {"PU-MIRT"},
            "separate_a": {"PU-MIRT"},
            "all_gene_oracle": {"PU"},
        }
    if arm_models != expected_arm_models:
        raise ValueError(
            "Saved export does not contain the complete v2 arm/model contract; "
            f"found {arm_models}"
        )
    expected_grid = sorted(pd.to_numeric(
        aggregate.loc[aggregate["arm"].eq("union"), "requested_overlap"],
        errors="coerce",
    ).dropna().unique())
    repeated_arms = {"union"} if tech_sar80 else {
        "union", "intersection", "disjoint_coefficient", "shared_a", "separate_a",
    }
    for arm in repeated_arms:
        actual_grid = sorted(pd.to_numeric(
            aggregate.loc[aggregate["arm"].eq(arm), "requested_overlap"],
            errors="coerce",
        ).dropna().unique())
        if len(actual_grid) != len(expected_grid) or not np.allclose(
            actual_grid, expected_grid, rtol=0, atol=1e-12
        ):
            raise ValueError(f"Saved export has an incomplete overlap grid for arm {arm}")
    if not any(np.isclose(expected_grid, 1.0, rtol=0, atol=1e-12)):
        raise ValueError("Saved v2 export lacks the required 100% same-K endpoint")
    if artifacts.tables.get("overlap_contrasts", pd.DataFrame()).empty:
        raise ValueError("Saved v2 export lacks the R_M overlap contrast table")
    if (not tech_sar80
            and artifacts.tables.get("shared_a_contrasts", pd.DataFrame()).empty):
        raise ValueError("Saved v2 export lacks the Shared-A/Separate-A contrast table")


def run_gene_overlap_experiment(views: Sequence[ExperimentDataset], settings: Settings, *,
                                checkpoint_dir: str | Path, export_dir: str | Path,
                                export_name: str, progress: bool = True,
                                progress_interval: float = 60., progress_level: str = "summary",
                                worker_status=None) -> Artifacts:
    """Run overlap views through the shared model/tuning/evaluation pipeline."""
    if settings.run_random_forest:
        raise ValueError("Gene-overlap notebooks intentionally require RUN_RANDOM_FOREST=False")
    if settings.run_information_controls or settings.run_qiao:
        raise ValueError("Overlap controls are explicit views; disable information/Qiao controls")
    partitioned = [view for view in views if view.metadata.get("fit_partition")]
    if partitioned:
        from . import pipeline as pipeline_module
        if not getattr(pipeline_module, "SUPPORTS_PARTITIONED_PANEL_FITS", False):
            arms = sorted({view.metadata["experiment_context"]["arm"] for view in partitioned})
            raise RuntimeError(
                "Partitioned gene-overlap arms require the partition-aware pipeline executor; "
                f"unsupported requested arms: {arms}. Disable "
                "include_independent_panel_sensitivity until that executor is available."
            )
    artifacts = _execute(
        tuple(views), settings, checkpoint_dir=checkpoint_dir, export_dir=export_dir,
        progress=progress, progress_interval=progress_interval, progress_level=progress_level,
        worker_status=worker_status, export_name=export_name,
        manifest_extra={"experiment": "gene_overlap", "targets_unchanged": True,
                        "missing_expression_representation": "visible-train standardized; hidden=0"},
    )
    return rebuild_overlap_summaries(artifacts)
