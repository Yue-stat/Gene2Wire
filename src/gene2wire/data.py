"""Validated data contract shared by every dataset adapter.

The core intentionally does not infer training labels from biological truth.  The
observed PU label ``S_observed``, the measured-entry mask ``W_measured``, and any
reference truth are separate arrays with explicit semantics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _binary(name: str, value: Any) -> BoolArray:
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{name} must be binary")
    result = np.array(arr, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


def _readonly_float(value: Any) -> Array:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _ids(name: str, values: Sequence[Any] | None, size: int, prefix: str) -> tuple[str, ...]:
    result = tuple(str(x) for x in values) if values is not None else tuple(f"{prefix}{i}" for i in range(size))
    if len(result) != size:
        raise ValueError(f"{name} must have length {size}, got {len(result)}")
    if len(set(result)) != size:
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True)
class DatasetBundle:
    """Canonical cell-by-target dataset.

    Parameters
    ----------
    X_cell:
        Numeric cell features, shape ``(n_cells, n_features)``.
    S_observed:
        Observed positive indicators.  Zero means unlabeled, not confirmed
        negative, on measured entries.
    W_measured:
        True only where the cell-target entry was assayed.  Off-panel entries
        never enter a loss or metric.
    Y_target:
        Optional target-level covariates, shape ``(n_targets, n_covariates)``.
        Models configured with ``use_target_features=True`` add the fixed-target
        term ``(X D) Y_target.T``.
    Z_reference/reference_mask:
        Optional evaluation-only reference truth and its availability mask.
        These arrays must not be passed into model tuning.
    """

    X_cell: Any
    S_observed: Any
    W_measured: Any
    Y_target: Any | None = None
    Z_reference: Any | None = None
    reference_mask: Any | None = None
    cell_ids: Sequence[Any] | None = None
    target_ids: Sequence[Any] | None = None
    groups: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    feature_blocks: Mapping[str, Sequence[int]] = field(default_factory=dict)
    semantics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = _readonly_float(self.X_cell)
        s = _binary("S_observed", self.S_observed)
        w = _binary("W_measured", self.W_measured)
        if x.ndim != 2 or s.ndim != 2 or w.ndim != 2:
            raise ValueError("X_cell, S_observed, and W_measured must be 2-D")
        if s.shape != w.shape:
            raise ValueError(f"S_observed shape {s.shape} != W_measured shape {w.shape}")
        if x.shape[0] != s.shape[0]:
            raise ValueError("X_cell and label matrices have different cell counts")
        if not np.all(np.isfinite(x)):
            raise ValueError("X_cell contains non-finite values")
        if np.any(s & ~w):
            raise ValueError("S_observed=1 is forbidden where W_measured=0")
        if not np.any(w):
            raise ValueError("W_measured contains no measured entries")

        n_cells, n_targets = s.shape
        y_target = None
        if self.Y_target is not None:
            y_target = _readonly_float(self.Y_target)
            if y_target.ndim != 2 or y_target.shape[0] != n_targets:
                raise ValueError("Y_target must have shape (n_targets, n_covariates)")
            if not np.all(np.isfinite(y_target)):
                raise ValueError("Y_target contains non-finite values")

        z_reference = None
        reference_mask = None
        if self.Z_reference is None and self.reference_mask is not None:
            raise ValueError("reference_mask requires Z_reference")
        if self.Z_reference is not None:
            z_reference = _binary("Z_reference", self.Z_reference)
            if z_reference.shape != s.shape:
                raise ValueError("Z_reference must match S_observed shape")
            reference_mask = (
                np.ones_like(w, dtype=bool)
                if self.reference_mask is None
                else _binary("reference_mask", self.reference_mask)
            )
            if reference_mask.shape != s.shape:
                raise ValueError("reference_mask must match S_observed shape")

        cell_ids = _ids("cell_ids", self.cell_ids, n_cells, "cell_")
        target_ids = _ids("target_ids", self.target_ids, n_targets, "target_")

        groups: dict[str, NDArray[np.object_]] = {}
        for name, values in self.groups.items():
            arr = np.array(values, dtype=object, copy=True)
            if arr.ndim != 1 or len(arr) != n_cells:
                raise ValueError(f"group {name!r} must have length {n_cells}")
            arr.setflags(write=False)
            groups[str(name)] = arr

        feature_blocks: dict[str, tuple[int, ...]] = {}
        used: set[int] = set()
        for name, indices in self.feature_blocks.items():
            idx = tuple(int(i) for i in indices)
            if len(set(idx)) != len(idx):
                raise ValueError(f"feature block {name!r} has duplicate indices")
            if any(i < 0 or i >= x.shape[1] for i in idx):
                raise ValueError(f"feature block {name!r} has an out-of-range index")
            overlap = used.intersection(idx)
            if overlap:
                raise ValueError(f"feature block {name!r} overlaps previous blocks at {sorted(overlap)}")
            used.update(idx)
            feature_blocks[str(name)] = idx

        object.__setattr__(self, "X_cell", x)
        object.__setattr__(self, "S_observed", s)
        object.__setattr__(self, "W_measured", w)
        object.__setattr__(self, "Y_target", y_target)
        object.__setattr__(self, "Z_reference", z_reference)
        object.__setattr__(self, "reference_mask", reference_mask)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "groups", MappingProxyType(groups))
        object.__setattr__(self, "feature_blocks", MappingProxyType(feature_blocks))
        object.__setattr__(self, "semantics", MappingProxyType(deepcopy(dict(self.semantics))))
        object.__setattr__(self, "metadata", MappingProxyType(deepcopy(dict(self.metadata))))

    @property
    def n_cells(self) -> int:
        return int(self.X_cell.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X_cell.shape[1])

    @property
    def n_targets(self) -> int:
        return int(self.S_observed.shape[1])

    def subset_rows(self, rows: Sequence[int] | NDArray[np.bool_]) -> "DatasetBundle":
        """Return a row subset while preserving target-level metadata."""

        idx = np.asarray(rows)
        return DatasetBundle(
            X_cell=self.X_cell[idx],
            S_observed=self.S_observed[idx],
            W_measured=self.W_measured[idx],
            Y_target=self.Y_target,
            Z_reference=None if self.Z_reference is None else self.Z_reference[idx],
            reference_mask=None if self.reference_mask is None else self.reference_mask[idx],
            cell_ids=np.asarray(self.cell_ids, dtype=object)[idx].tolist(),
            target_ids=self.target_ids,
            groups={name: values[idx] for name, values in self.groups.items()},
            feature_blocks=self.feature_blocks,
            semantics=self.semantics,
            metadata=self.metadata,
        )

    def without_reference(self) -> "DatasetBundle":
        """Drop evaluation-only truth before giving data to a tuner."""

        return DatasetBundle(
            X_cell=self.X_cell,
            S_observed=self.S_observed,
            W_measured=self.W_measured,
            Y_target=self.Y_target,
            cell_ids=self.cell_ids,
            target_ids=self.target_ids,
            groups=self.groups,
            feature_blocks=self.feature_blocks,
            semantics=self.semantics,
            metadata=self.metadata,
        )
