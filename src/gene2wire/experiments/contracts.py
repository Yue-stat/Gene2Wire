"""Dataset adapters expose raw outcomes, explicit splits, and train-fitted features.

The experiment builder owns reference access. These objects are never passed to
the model/tuner: only reference-free DatasetBundle views reach the core.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Fold:
    outer_fold: int
    train_rows: np.ndarray
    validation_rows: np.ndarray
    test_rows: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, n_cells: int) -> None:
        parts = [np.asarray(x, dtype=int) for x in
                 (self.train_rows, self.validation_rows, self.test_rows)]
        if any(x.ndim != 1 or len(x) == 0 for x in parts):
            raise ValueError("Every split role must contain cells")
        rows = np.concatenate(parts)
        if not np.array_equal(np.sort(rows), np.arange(n_cells)):
            raise ValueError("Each fold must partition all cells exactly once")


@dataclass(frozen=True)
class FeatureSet:
    X: np.ndarray
    feature_blocks: Mapping[str, Sequence[int]]
    Y_target: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSpec:
    """Evaluation-only labels/support; never passed to preprocessing or learners.

    A separate object is essential for structural block experiments: fitting
    arrays contain only visible entries, while this object retains the original
    assay reference solely for scoring held-out cells after fitting.
    """
    reference: np.ndarray
    masks: Mapping[float, np.ndarray]
    groups: np.ndarray
    source_measured: np.ndarray

    def validate(self, shape: tuple[int, int]) -> None:
        z, w = np.asarray(self.reference), np.asarray(self.source_measured)
        if z.shape != shape or w.shape != shape or not np.all(np.isin(z, [0, 1])):
            raise ValueError("Evaluation reference/source mask must be aligned binary arrays")
        if not np.all(np.isin(w, [0, 1])) or np.any(z.astype(bool) & ~w.astype(bool)):
            raise ValueError("Evaluation positives cannot occur outside original assays")
        if np.asarray(self.groups).shape != (shape[0],) or not self.masks:
            raise ValueError("Evaluation groups and masks must be supplied")
        for fraction, mask in self.masks.items():
            mask = np.asarray(mask)
            if not 0 <= float(fraction) <= 1 or mask.shape != shape or not np.all(np.isin(mask, [0, 1])):
                raise ValueError("Blocked evaluation masks must be aligned binary arrays")
            if np.any(mask.astype(bool) & ~w.astype(bool)):
                raise ValueError("Blocked evaluation cannot include naturally unassayed entries")


@dataclass
class ExperimentDataset:
    name: str
    reference: np.ndarray
    measured: np.ndarray
    cell_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    # Feature builder(train_rows, use_location, use_target_features) -> all-row X.
    feature_builder: Callable[[np.ndarray, bool, bool], FeatureSet]
    # Split builder(n_outer_folds, seed) -> complete outer partitions.
    split_builder: Callable[[int, int], Sequence[Fold]]
    groups: Mapping[str, np.ndarray] = field(default_factory=dict)
    natural_observed: np.ndarray | None = None
    platform: np.ndarray | None = None
    technical_score: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evaluation: EvaluationSpec | None = None
    # Optional fold-independent gene-scale matrix used by panel-overlap views.
    # Values must be on the transform scale immediately before standardization
    # (for example log1p counts). It is never passed directly to a learner.
    gene_matrix: np.ndarray | None = None
    gene_names: tuple[str, ...] = ()

    def validate(self) -> None:
        z, w = np.asarray(self.reference), np.asarray(self.measured)
        if z.ndim != 2 or z.shape != w.shape:
            raise ValueError("reference and measured must be matching matrices")
        if not np.all(np.isin(z, [0, 1])) or not np.all(np.isin(w, [0, 1])):
            raise ValueError("reference and measured must be binary")
        if np.any(z.astype(bool) & ~w.astype(bool)):
            raise ValueError("Reference positives cannot occur off-panel")
        if len(self.cell_ids) != z.shape[0] or len(set(self.cell_ids)) != z.shape[0]:
            raise ValueError("Cell IDs must be unique and aligned")
        if len(self.target_ids) != z.shape[1] or len(set(self.target_ids)) != z.shape[1]:
            raise ValueError("Target IDs must be unique and aligned")
        if self.gene_matrix is not None:
            x = np.asarray(self.gene_matrix)
            if x.ndim != 2 or x.shape[0] != z.shape[0] or not np.all(np.isfinite(x)):
                raise ValueError("gene_matrix must be a finite cell-by-gene matrix")
            if len(self.gene_names) != x.shape[1] or len(set(self.gene_names)) != x.shape[1]:
                raise ValueError("gene_names must be unique and aligned with gene_matrix")
        for name, values in self.groups.items():
            if np.asarray(values).shape != (z.shape[0],):
                raise ValueError(f"Group {name!r} is not aligned to cells")
        if self.evaluation is not None:
            self.evaluation.validate(z.shape)
        if self.natural_observed is not None:
            d = np.asarray(self.natural_observed)
            if d.shape != z.shape or not np.all(np.isin(d, [0, 1])):
                raise ValueError("Natural observed labels must be aligned and binary")
            if np.any(d.astype(bool) & ~z.astype(bool)):
                raise ValueError("The PU paired reference must contain every detection")
