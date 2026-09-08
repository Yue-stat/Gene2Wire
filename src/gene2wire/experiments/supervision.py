"""Compile authorized reference and detection supervision for the core loss.

The core's DatasetBundle.Z_reference remains evaluation-only.  Callers must
explicitly supply paired_reference and paired_rows; nothing is inferred from an
evaluation field.  The compiled bundle contains no evaluation references.
"""

from __future__ import annotations

import numpy as np

from ..data import DatasetBundle
from .observation import _rows


SUPERVISION_MODES = ("observed", "calibrated_pu", "reference_only", "reference_plus_pu")


def compile_training_bundle(
    bundle: DatasetBundle,
    train_rows,
    paired_rows=(),
    paired_reference=None,
    estimated_sensitivity=None,
    mode: str = "calibrated_pu",
) -> tuple[DatasetBundle, np.ndarray]:
    """Return a reference-free training bundle and effective detection matrix.

    Arrays retain the input row order and shape.  Entries outside train_rows are
    masked and have label zero.  Reference-only uses all assayed positive *and*
    negative reference labels in paired cells.  Reference+PU uses reference BCE
    on C (exposure=1) and observed PU BCE on O, without duplicating C's outcome.

    ``paired_reference`` may have shape (N,T) or (len(paired_rows),T).  Only the
    paired measured entries are read; other entries may deliberately be NaN.
    Projection-side information access is therefore explicit at this boundary.
    """
    if mode not in SUPERVISION_MODES:
        raise ValueError(f"mode must be one of {SUPERVISION_MODES}")
    shape = (bundle.n_cells, bundle.n_targets)
    train = _rows(train_rows, bundle.n_cells, "train_rows")
    paired = _rows(paired_rows, bundle.n_cells, "paired_rows")
    if not len(train):
        raise ValueError("train_rows cannot be empty")
    if np.any(~np.isin(paired, train)):
        raise ValueError("paired_rows must be a subset of the authorized train_rows")
    train_mask = np.zeros(bundle.n_cells, dtype=bool)
    train_mask[train] = True
    paired_mask = np.zeros(bundle.n_cells, dtype=bool)
    paired_mask[paired] = True
    measured = np.asarray(bundle.W_measured)
    allowed = measured & train_mask[:, None]
    c = measured & paired_mask[:, None]
    if not np.any(allowed):
        raise ValueError("Authorized training rows contain no assayed entries")
    labels = np.zeros(shape, dtype=bool)
    labels[allowed] = bundle.S_observed[allowed]
    exposure = np.ones(shape, dtype=float)
    if mode in {"calibrated_pu", "reference_plus_pu"}:
        if estimated_sensitivity is None:
            raise ValueError(f"{mode} requires estimated_sensitivity")
        estimated = np.asarray(estimated_sensitivity, dtype=float)
        if estimated.shape != shape:
            raise ValueError("estimated_sensitivity must have shape (N,T)")
        pu_entries = allowed if mode == "calibrated_pu" else allowed & ~c
        values = estimated[pu_entries]
        if not np.all(np.isfinite(values)) or np.any((values <= 0) | (values > 1)):
            raise ValueError("PU training sensitivities must be finite in (0,1]")
        exposure[pu_entries] = values
    if mode in {"reference_only", "reference_plus_pu"}:
        if paired_reference is None:
            raise ValueError("Reference supervision requires explicit paired_reference; evaluation truth is never used automatically")
        if not np.any(c):
            raise ValueError("Reference supervision requires assayed paired entries")
        reference = np.asarray(paired_reference)
        if reference.shape == shape:
            paired_values = reference[paired]
        elif reference.shape == (len(paired), bundle.n_targets):
            paired_values = reference
        else:
            raise ValueError("paired_reference must have shape (N,T) or (len(paired_rows),T)")
        paired_measured = measured[paired]
        values = paired_values[paired_measured]
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("Authorized paired reference labels must be binary")
        if np.any(bundle.S_observed[paired][paired_measured] & (values == 0)):
            raise ValueError("Observed paired positives must be reference positive under the PU assumption")
        paired_labels = np.zeros((len(paired), bundle.n_targets), dtype=bool)
        paired_labels[paired_measured] = values.astype(bool)
        labels[paired] = paired_labels
        exposure[c] = 1.0
        if mode == "reference_only":
            allowed = c
            labels[~allowed] = False
    compiled = DatasetBundle(
        X_cell=bundle.X_cell,
        S_observed=labels,
        W_measured=allowed,
        Y_target=bundle.Y_target,
        cell_ids=bundle.cell_ids,
        target_ids=bundle.target_ids,
        groups=bundle.groups,
        feature_blocks=bundle.feature_blocks,
        semantics={
            **dict(bundle.semantics), "supervision_mode": mode,
            "reference_access": "explicit-authorized-paired-training-only",
            "prediction_semantics": "q" if mode == "observed" else "p",
        },
        metadata={
            **dict(bundle.metadata), "training_row_count": int(len(train)),
            "paired_training_row_count": int(len(paired)),
            "paired_training_entry_count": int(c.sum()),
            "training_entry_count": int(allowed.sum()),
        },
    )
    exposure.setflags(write=False)
    return compiled, exposure
