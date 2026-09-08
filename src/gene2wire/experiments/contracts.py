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
        for name, values in self.groups.items():
            if np.asarray(values).shape != (z.shape[0],):
                raise ValueError(f"Group {name!r} is not aligned to cells")
        if self.natural_observed is not None:
            d = np.asarray(self.natural_observed)
            if d.shape != z.shape or not np.all(np.isin(d, [0, 1])):
                raise ValueError("Natural observed labels must be aligned and binary")
            if np.any(d.astype(bool) & ~z.astype(bool)):
                raise ValueError("The PU paired reference must contain every detection")
