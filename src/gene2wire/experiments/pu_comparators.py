"""Generic positive-unlabeled comparators for measurement degradation studies.

The implementations in this module deliberately accept only learner-authorized
arrays.  In particular, none of the fitting or tuning functions accepts a clean
reference outcome, a hidden-positive indicator, or a generator propensity.

The three model families need explicit names because the Gene2Wire observation
contract is more general than the original papers:

``GenEML-adapted``
    Point-estimate EM for the contextual-exposure generative model of Jain,
    Modhe, and Rai (ICML 2017).  The paper permits dyad-level exposure
    covariates.  This implementation additionally excludes entries with known
    ``W_measured=0`` from every sufficient statistic.

``ShiftIMC-adapted``
    A bounded, factorized inductive version of the shifted PU matrix-completion
    loss of Hsieh, Natarajan, and Dhillon (ICML 2015).  It supports a measured
    mask and entry-specific estimated exposure.  The sigmoid parameterization
    and factorized nuclear-norm surrogate are adaptations of the paper's linear,
    box-constrained estimator.

``SAR-EM-adapted`` (legacy)
    A historical clean-room multi-target adaptation.  It is retained as a
    low-level compatibility API but is not the SAR-PU benchmark implementation:
    the measurement pipeline uses the pinned authors' kernel through
    :mod:`gene2wire.experiments.sarpu_authors`.

Every tuner selects a configuration by measured-entry observed-label log loss.
The returned fits expose reference probability ``p``, exposure ``e`` (when the
model estimates it), and observed probability ``q=e*p`` separately so callers
cannot accidentally evaluate one probability as another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from ..checkpoint import canonical_json
from ..seeds import stable_seed


Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_EPS = 1e-9
_MAX_COMPARATOR_CANDIDATES = 32


def _array2d(value: Any, name: str, *, allow_zero_columns: bool = False) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or (not allow_zero_columns and result.shape[1] == 0):
        qualifier = (
            "finite two-dimensional"
            if allow_zero_columns
            else "nonempty finite two-dimensional"
        )
        raise ValueError(f"{name} must be a {qualifier} array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _labels(observed: Any, measured: Any, n_rows: int) -> tuple[Array, BoolArray]:
    raw_s = np.asarray(observed)
    raw_w = np.asarray(measured)
    if raw_s.ndim != 2 or raw_s.shape[0] != n_rows or raw_w.shape != raw_s.shape:
        raise ValueError("observed and measured must align as cell-by-target matrices")
    if not np.all(np.isin(raw_w, (0, 1))):
        raise ValueError("measured must be binary")
    w = raw_w.astype(bool)
    if not np.any(w):
        raise ValueError("at least one measured entry is required")
    if np.any((raw_s == 1) & ~w):
        raise ValueError("an observed positive cannot occur where measured is false")
    if not np.all(np.isin(raw_s[w], (0, 1))):
        raise ValueError("observed labels must be binary on measured entries")
    s = np.where(w, raw_s, 0).astype(np.float64)
    return s, w


def _propensity_array(value: Any | None, shape: tuple[int, int]) -> Array:
    if value is None:
        # Target-identity-coded propensity effects.  The fitting functions add an
        # unpenalized intercept, giving one global exposure per target.
        n_rows, n_targets = shape
        result = np.zeros((n_rows, n_targets, max(0, n_targets - 1)), dtype=np.float64)
        for target in range(1, n_targets):
            result[:, target, target - 1] = 1.0
        return result
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 3 or result.shape[:2] != shape:
        raise ValueError("propensity_design must have shape (n_cells, n_targets, n_covariates)")
    if not np.all(np.isfinite(result)):
        raise ValueError("propensity_design contains non-finite values")
    return result


def _exposure(value: Any, shape: tuple[int, int], measured: BoolArray) -> Array:
    try:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    except ValueError as error:
        raise ValueError("exposure must broadcast to the cell-by-target shape") from error
    values = result[measured]
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(values > 1):
        raise ValueError("exposure must be finite in (0, 1] on measured entries")
    result[~measured] = 1.0
    return result


def _prediction_exposure(value: Any, shape: tuple[int, int]) -> Array:
    try:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    except ValueError as error:
        raise ValueError("exposure must broadcast to the cell-by-target shape") from error
    if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(result > 1):
        raise ValueError("prediction exposure must be finite in [0, 1]")
    return result


def _augment_intercept(value: Array) -> Array:
    return np.column_stack((np.ones(len(value), dtype=np.float64), value))


def _augment_propensity(value: Array) -> Array:
    return np.concatenate(
        (np.ones((*value.shape[:2], 1), dtype=np.float64), value), axis=2
    )


def _observed_log_loss(observed: Array, probability: Array, measured: BoolArray) -> float:
    q = np.asarray(probability, dtype=np.float64)[measured]
    if not np.all(np.isfinite(q)) or np.any(q < 0) or np.any(q > 1):
        raise ValueError("observed probabilities must be finite in [0, 1]")
    q = np.clip(q, _EPS, 1 - _EPS)
    y = observed[measured]
    return float(np.mean(-y * np.log(q) - (1 - y) * np.log1p(-q)))


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class AssayTargetPropensityEncoder:
    """Outcome-blind encoder for a saturated assay-by-target propensity design.

    Categories and continuous-variable scaling are learned from development
    design data only.  No label argument exists.  Reference coding avoids an
    overcomplete design after fitting functions add their intercept.
    """

    assay_levels: tuple[str, ...]
    target_ids: tuple[str, ...]
    row_covariate_names: tuple[str, ...]
    row_center: Array
    row_scale: Array
    interact_row_covariates_with_target: bool = True

    @classmethod
    def fit(
        cls,
        assay_ids: Sequence[Any],
        target_ids: Sequence[Any],
        row_covariates: Any | None = None,
        row_covariate_names: Sequence[Any] = (),
        *,
        interact_row_covariates_with_target: bool = True,
    ) -> "AssayTargetPropensityEncoder":
        assays = np.asarray(tuple(map(str, assay_ids)), dtype=object)
        targets = tuple(map(str, target_ids))
        if assays.ndim != 1 or len(assays) == 0:
            raise ValueError("assay_ids must be a nonempty one-dimensional sequence")
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("target_ids must be nonempty and unique")
        levels = tuple(sorted(set(map(str, assays))))
        if row_covariates is None:
            covariates = np.empty((len(assays), 0), dtype=np.float64)
        else:
            covariates = _array2d(row_covariates, "row_covariates", allow_zero_columns=True)
            if len(covariates) != len(assays):
                raise ValueError("row_covariates must align with assay_ids")
        names = tuple(map(str, row_covariate_names))
        if len(names) != covariates.shape[1] or len(set(names)) != len(names):
            raise ValueError("row_covariate_names must be unique and align with columns")
        center = covariates.mean(axis=0) if covariates.shape[1] else np.empty(0)
        scale = covariates.std(axis=0) if covariates.shape[1] else np.empty(0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(
            levels,
            targets,
            names,
            np.asarray(center, dtype=np.float64),
            np.asarray(scale, dtype=np.float64),
            bool(interact_row_covariates_with_target),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        target_effects = tuple(f"target={target}" for target in self.target_ids[1:])
        assay_effects = tuple(f"assay={assay}" for assay in self.assay_levels[1:])
        interactions = tuple(
            f"assay={assay}:target={target}"
            for assay in self.assay_levels[1:]
            for target in self.target_ids[1:]
        )
        names = (*target_effects, *assay_effects, *interactions, *self.row_covariate_names)
        if self.interact_row_covariates_with_target:
            names += tuple(
                f"{name}:target={target}"
                for name in self.row_covariate_names
                for target in self.target_ids[1:]
            )
        return names

    def transform(
        self,
        assay_ids: Sequence[Any],
        target_ids: Sequence[Any],
        row_covariates: Any | None = None,
    ) -> Array:
        assays = np.asarray(tuple(map(str, assay_ids)), dtype=object)
        targets = tuple(map(str, target_ids))
        if assays.ndim != 1:
            raise ValueError("assay_ids must be one-dimensional")
        if targets != self.target_ids:
            raise ValueError("target_ids must match the fitted propensity schema exactly")
        unknown = sorted(set(map(str, assays)).difference(self.assay_levels))
        if unknown:
            raise ValueError(f"unknown assay levels: {unknown}")
        if self.row_covariate_names:
            covariates = _array2d(row_covariates, "row_covariates")
            if covariates.shape != (len(assays), len(self.row_covariate_names)):
                raise ValueError("row_covariates do not match the fitted propensity schema")
            covariates = (covariates - self.row_center) / self.row_scale
        else:
            if row_covariates is not None and np.asarray(row_covariates).size:
                raise ValueError("row_covariates were supplied to a schema without them")
            covariates = np.empty((len(assays), 0), dtype=np.float64)

        n_rows, n_targets = len(assays), len(targets)
        columns: list[Array] = []
        for target in range(1, n_targets):
            column = np.zeros((n_rows, n_targets), dtype=np.float64)
            column[:, target] = 1.0
            columns.append(column)
        for assay in self.assay_levels[1:]:
            columns.append(np.broadcast_to((assays == assay)[:, None], (n_rows, n_targets)))
        for assay in self.assay_levels[1:]:
            selected = assays == assay
            for target in range(1, n_targets):
                column = np.zeros((n_rows, n_targets), dtype=np.float64)
                column[selected, target] = 1.0
                columns.append(column)
        for column in range(covariates.shape[1]):
            columns.append(np.broadcast_to(covariates[:, column, None], (n_rows, n_targets)))
        if self.interact_row_covariates_with_target:
            for column in range(covariates.shape[1]):
                for target in range(1, n_targets):
                    value = np.zeros((n_rows, n_targets), dtype=np.float64)
                    value[:, target] = covariates[:, column]
                    columns.append(value)
        result = (
            np.stack(columns, axis=2)
            if columns
            else np.empty((n_rows, n_targets, 0), dtype=np.float64)
        )
        if result.shape[2] != len(self.feature_names):
            raise RuntimeError("internal propensity feature-name mismatch")
        return result


@dataclass(frozen=True)
class PerTargetAssayPropensityEncoder:
    """Outcome-blind propensity designs for independent binary SAR fits.

    The authors' SAR-PU routine fits one binary model at a time. Feeding a
    slice of :class:`AssayTargetPropensityEncoder` to such a fit is incorrect:
    target main effects become duplicate intercepts and assay-by-target terms
    duplicate assay main effects. This encoder instead learns a separate
    centered-orthonormal assay/QC schema on the measured rows of every target.

    The returned tensor is padded to a common third dimension so it retains
    the public ``(cell, target, feature)`` wrapper API. Active columns are
    always packed first. Padding is identically zero and the SAR wrapper
    removes it (and defensively removes any other linearly dependent column)
    before calling the upstream optimizer.
    """

    assay_levels: tuple[str, ...]
    target_ids: tuple[str, ...]
    row_covariate_names: tuple[str, ...]
    target_assay_levels: tuple[tuple[str, ...], ...]
    target_qc_indices: tuple[tuple[int, ...], ...]
    row_center: Array
    row_scale: Array
    n_features: int

    @staticmethod
    def _assay_contrast_basis(n_levels: int) -> Array:
        """Return a deterministic Helmert basis for the centered level space.

        The columns are orthonormal and orthogonal to the all-ones vector.
        Consequently, relabeling/reordering assay categories changes the design
        only by an orthogonal rotation, leaving an L2-penalized fit invariant.
        """

        if n_levels < 1:
            return np.empty((0, 0), dtype=np.float64)
        basis = np.zeros((n_levels, n_levels - 1), dtype=np.float64)
        for column in range(n_levels - 1):
            count = column + 1
            scale = np.sqrt(count * (count + 1.0))
            basis[:count, column] = 1.0 / scale
            basis[count, column] = -count / scale
        return basis

    @classmethod
    def _assay_design(
        cls,
        assay_ids: Sequence[Any],
        levels: Sequence[str],
    ) -> Array:
        assays = tuple(map(str, assay_ids))
        fitted_levels = tuple(map(str, levels))
        if not fitted_levels:
            return np.empty((len(assays), 0), dtype=np.float64)
        lookup = {level: index for index, level in enumerate(fitted_levels)}
        basis = cls._assay_contrast_basis(len(fitted_levels))
        result = np.zeros((len(assays), basis.shape[1]), dtype=np.float64)
        for row, assay in enumerate(assays):
            position = lookup.get(assay)
            if position is not None:
                result[row] = basis[position]
        return result

    @classmethod
    def fit(
        cls,
        assay_ids: Sequence[Any],
        target_ids: Sequence[Any],
        measured: Any,
        row_covariates: Any | None = None,
        row_covariate_names: Sequence[Any] = (),
    ) -> "PerTargetAssayPropensityEncoder":
        assays = np.asarray(tuple(map(str, assay_ids)), dtype=object)
        targets = tuple(map(str, target_ids))
        if assays.ndim != 1 or len(assays) == 0:
            raise ValueError("assay_ids must be a nonempty one-dimensional sequence")
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("target_ids must be nonempty and unique")
        raw_measured = np.asarray(measured)
        if raw_measured.shape != (len(assays), len(targets)):
            raise ValueError("measured must align with assay_ids and target_ids")
        if not np.all(np.isin(raw_measured, (0, 1))):
            raise ValueError("measured must be binary")
        included = raw_measured.astype(bool)

        if row_covariates is None:
            covariates = np.empty((len(assays), 0), dtype=np.float64)
        else:
            covariates = _array2d(
                row_covariates, "row_covariates", allow_zero_columns=True
            )
            if len(covariates) != len(assays):
                raise ValueError("row_covariates must align with assay_ids")
        names = tuple(map(str, row_covariate_names))
        if len(names) != covariates.shape[1] or len(set(names)) != len(names):
            raise ValueError("row_covariate_names must be unique and align with columns")

        centers = np.zeros((len(targets), covariates.shape[1]), dtype=np.float64)
        scales = np.ones_like(centers)
        target_levels: list[tuple[str, ...]] = []
        target_qc: list[tuple[int, ...]] = []
        feature_counts: list[int] = []
        for target in range(len(targets)):
            rows = included[:, target]
            if not np.any(rows):
                # Keep an aligned zero-width schema. The downstream authors'
                # wrapper will report this target as unsupported without
                # preventing unrelated comparators from running.
                target_levels.append(())
                target_qc.append(())
                feature_counts.append(0)
                continue
            levels = tuple(sorted(set(map(str, assays[rows]))))
            target_levels.append(levels)
            assay_design = cls._assay_design(assays[rows], levels)
            base = np.column_stack(
                (np.ones(int(rows.sum()), dtype=np.float64), assay_design)
            )
            base_rank = int(np.linalg.matrix_rank(base))
            if base_rank != base.shape[1]:
                raise RuntimeError(
                    "centered-orthonormal assay design is not full rank"
                )

            if covariates.shape[1]:
                target_values = covariates[rows]
                centers[target] = target_values.mean(axis=0)
                raw_scale = target_values.std(axis=0)
                scales[target] = np.where(raw_scale < 1e-12, 1.0, raw_scale)
                standardized = (target_values - centers[target]) / scales[target]
            else:
                standardized = np.empty((int(rows.sum()), 0), dtype=np.float64)

            # Preserve declared QC order and retain a column only when it adds
            # an estimable direction beyond the intercept and assay contrasts.
            # This decision uses W and covariates only, never PU outcomes.
            selected: list[int] = []
            working = base
            rank = base_rank
            for column in range(standardized.shape[1]):
                proposed = np.column_stack((working, standardized[:, column]))
                proposed_rank = int(np.linalg.matrix_rank(proposed))
                if proposed_rank == rank + 1:
                    selected.append(column)
                    working = proposed
                    rank = proposed_rank
            target_qc.append(tuple(selected))
            feature_counts.append(len(levels) - 1 + len(selected))

        return cls(
            assay_levels=tuple(sorted(set(map(str, assays)))),
            target_ids=targets,
            row_covariate_names=names,
            target_assay_levels=tuple(target_levels),
            target_qc_indices=tuple(target_qc),
            row_center=centers,
            row_scale=scales,
            n_features=max(feature_counts, default=0),
        )

    @property
    def active_feature_counts(self) -> tuple[int, ...]:
        return tuple(
            max(0, len(levels) - 1) + len(qc)
            for levels, qc in zip(self.target_assay_levels, self.target_qc_indices)
        )

    @property
    def feature_names_by_target(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(
                "assay_centered_orthonormal_v1"
                + canonical_json(levels)
                + f"::{column}"
                for column in range(max(0, len(levels) - 1))
            )
            + tuple(self.row_covariate_names[index] for index in qc)
            for levels, qc in zip(self.target_assay_levels, self.target_qc_indices)
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Stable tensor-slot names; semantic names are target-specific."""
        return tuple(
            f"per_target_propensity_slot={index}" for index in range(self.n_features)
        )

    @property
    def schema_labels(self) -> tuple[tuple[str, ...], ...]:
        """Rectangular target-specific schema for checkpoint identities."""
        return tuple(
            names
            + tuple("<padding>" for _ in range(self.n_features - len(names)))
            for names in self.feature_names_by_target
        )

    def transform(
        self,
        assay_ids: Sequence[Any],
        target_ids: Sequence[Any],
        row_covariates: Any | None = None,
        *,
        required_mask: Any | None = None,
    ) -> Array:
        assays = np.asarray(tuple(map(str, assay_ids)), dtype=object)
        targets = tuple(map(str, target_ids))
        if assays.ndim != 1:
            raise ValueError("assay_ids must be one-dimensional")
        if targets != self.target_ids:
            raise ValueError("target_ids must match the fitted propensity schema exactly")
        if required_mask is None:
            # Preserve the strict legacy transform contract when the caller
            # supplies no row/target support information. Callers that know W
            # can permit wholly off-support rows by passing required_mask.
            unknown = sorted(set(map(str, assays)).difference(self.assay_levels))
            if unknown:
                raise ValueError(f"unknown assay levels: {unknown}")
            required = np.zeros((len(assays), len(targets)), dtype=bool)
        else:
            raw_required = np.asarray(required_mask)
            if raw_required.shape != (len(assays), len(targets)):
                raise ValueError(
                    "required_mask must align with assay_ids and target_ids"
                )
            if not np.all(np.isin(raw_required, (0, 1))):
                raise ValueError("required_mask must be binary")
            required = raw_required.astype(bool)
        if self.row_covariate_names:
            covariates = _array2d(row_covariates, "row_covariates")
            if covariates.shape != (len(assays), len(self.row_covariate_names)):
                raise ValueError("row_covariates do not match the fitted propensity schema")
        else:
            if row_covariates is not None and np.asarray(row_covariates).size:
                raise ValueError("row_covariates were supplied to a schema without them")
            covariates = np.empty((len(assays), 0), dtype=np.float64)

        result = np.zeros(
            (len(assays), len(targets), self.n_features), dtype=np.float64
        )
        for target, (levels, qc_indices) in enumerate(
            zip(self.target_assay_levels, self.target_qc_indices)
        ):
            unseen_required = required[:, target] & ~np.isin(assays, levels)
            if np.any(unseen_required):
                missing = sorted(set(map(str, assays[unseen_required])))
                raise ValueError(
                    "required assay levels were unseen on fitted W support for "
                    f"target {self.target_ids[target]!r}: {missing}"
                )
            position = 0
            assay_design = self._assay_design(assays, levels)
            if assay_design.shape[1]:
                stop = position + assay_design.shape[1]
                result[:, target, position:stop] = assay_design
                position = stop
            if qc_indices:
                standardized = (
                    covariates - self.row_center[target]
                ) / self.row_scale[target]
                for column in qc_indices:
                    result[:, target, position] = standardized[:, column]
                    position += 1
            if position != self.active_feature_counts[target]:
                raise RuntimeError("internal per-target propensity schema mismatch")
        return result


def _fit_soft_logistic(
    design: Array,
    outcome: Array,
    weight: Array,
    *,
    l2: float,
    initial: Array | None,
    maxiter: int,
    tolerance: float,
) -> tuple[Array, bool, int, float, str]:
    """Fit a logistic model to binary or fractional outcomes and case weights."""

    x = _array2d(design, "logistic design")
    y = np.asarray(outcome, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if y.shape != (len(x),) or w.shape != y.shape:
        raise ValueError("logistic outcomes and weights must align with design rows")
    if not np.all(np.isfinite(y)) or np.any(y < 0) or np.any(y > 1):
        raise ValueError("soft logistic outcomes must lie in [0, 1]")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("soft logistic weights must be nonnegative with positive total")
    penalty = _positive_float(l2, "l2")
    beta0 = (
        np.zeros(x.shape[1], dtype=np.float64)
        if initial is None
        else np.asarray(initial, dtype=np.float64)
    )
    if beta0.shape != (x.shape[1],) or not np.all(np.isfinite(beta0)):
        raise ValueError("initial logistic coefficients have the wrong shape")
    normalizer = float(np.sum(w))

    def objective(beta: Array) -> tuple[float, Array]:
        eta = x @ beta
        loss = float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) / normalizer)
        gradient = x.T @ (w * (expit(eta) - y)) / normalizer
        loss += 0.5 * penalty * float(beta[1:] @ beta[1:])
        gradient[1:] += penalty * beta[1:]
        return loss, gradient

    fitted = minimize(
        objective,
        beta0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maxiter),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
            "maxls": 30,
        },
    )
    if not np.isfinite(fitted.fun) or not np.all(np.isfinite(fitted.x)):
        raise FloatingPointError("soft logistic optimization produced non-finite values")
    return (
        np.asarray(fitted.x, dtype=np.float64),
        bool(fitted.success),
        int(fitted.nit),
        float(fitted.fun),
        str(fitted.message),
    )


@dataclass(frozen=True)
class ComparatorTrial:
    model: str
    index: int
    config: Mapping[str, Any]
    validation_observed_log_loss: float
    converged: bool
    iterations: int
    objective_value: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "config": dict(self.config)}


@dataclass(frozen=True)
class ComparatorTuningResult:
    model: str
    selected_fit: Any
    selected_config: Mapping[str, Any]
    trials: tuple[ComparatorTrial, ...]
    selection_metric: str = "observed_log_loss"


@dataclass(frozen=True)
class GenEMLFit:
    map_weights: Array
    target_factors: Array
    exposure_coefficients: Array
    rank: int
    n_features: int
    n_targets: int
    n_propensity_features: int
    default_target_exposure: bool
    l2_u: float
    l2_v: float
    l2_map: float
    l2_exposure: float
    converged: bool
    iterations: int
    objective_value: float
    optimizer_failures: int
    final_change: float

    @property
    def model_name(self) -> str:
        return "GenEML-adapted"

    def predict_reference(self, X: Any) -> Array:
        x = _array2d(X, "X")
        if x.shape[1] != self.n_features:
            raise ValueError("prediction X does not match the fitted feature count")
        latent = _augment_intercept(x) @ self.map_weights
        return expit(latent @ self.target_factors.T)

    def predict_exposure(self, X: Any, propensity_design: Any | None = None) -> Array:
        x = _array2d(X, "X")
        if x.shape[1] != self.n_features:
            raise ValueError("prediction X does not match the fitted feature count")
        if (propensity_design is None) != self.default_target_exposure:
            raise ValueError("prediction propensity design must use the fitted schema")
        phi = _propensity_array(propensity_design, (len(x), self.n_targets))
        if phi.shape[2] != self.n_propensity_features:
            raise ValueError("prediction propensity design does not match the fitted schema")
        return expit(np.einsum("ntr,r->nt", _augment_propensity(phi), self.exposure_coefficients))

    def predict_observed(self, X: Any, propensity_design: Any | None = None) -> Array:
        return self.predict_reference(X) * self.predict_exposure(X, propensity_design)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "selected_structure": "contextual_exposure_lowrank",
            "rank": self.rank,
            "l2_u": self.l2_u,
            "l2_v": self.l2_v,
            "l2_map": self.l2_map,
            "l2_exposure": self.l2_exposure,
            "final_converged": self.converged,
            "final_iterations": self.iterations,
            "final_objective": self.objective_value,
            "optimizer_failures": self.optimizer_failures,
            "final_change": self.final_change,
            "probability_semantics": "reference_p_with_model_exposure",
            "adaptations": "known-W exclusion|contextual exposure Python3 point-EM",
        }


def fit_geneml_adapted(
    X: Any,
    observed: Any,
    measured: Any,
    propensity_design: Any | None = None,
    *,
    rank: int = 2,
    l2_u: float = 1e-2,
    l2_v: float = 1e-2,
    l2_map: float = 1e-2,
    l2_exposure: float = 1e-2,
    maxiter: int = 100,
    tolerance: float = 1e-5,
    seed: int = 0,
) -> GenEMLFit:
    """Fit the contextual GenEML point-estimate equations on measured entries."""

    x = _array2d(X, "X")
    s, w = _labels(observed, measured, len(x))
    phi_raw = _propensity_array(propensity_design, s.shape)
    phi = _augment_propensity(phi_raw)
    n_rows, n_targets = s.shape
    rank = _positive_int(rank, "rank")
    if rank > min(x.shape[1] + 1, n_targets):
        raise ValueError("rank exceeds the augmented cell-feature/target cap")
    penalties = tuple(
        _positive_float(value, name)
        for value, name in (
            (l2_u, "l2_u"),
            (l2_v, "l2_v"),
            (l2_map, "l2_map"),
            (l2_exposure, "l2_exposure"),
        )
    )
    l2_u, l2_v, l2_map, l2_exposure = penalties
    maxiter = _positive_int(maxiter, "maxiter")
    tolerance = _positive_float(tolerance, "tolerance")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")

    xa = _augment_intercept(x)
    active_rows = np.any(w, axis=1)
    rng = np.random.default_rng(int(seed))
    map_weights = rng.normal(0.0, 0.03, size=(xa.shape[1], rank))
    latent = xa @ map_weights
    target_factors = rng.normal(0.0, 0.03, size=(n_targets, rank))
    # The paper's default Beta(1,1) exposure initialization has mean 0.5.
    exposure_coefficients = np.zeros(phi.shape[2], dtype=np.float64)
    previous_p = expit(latent @ target_factors.T)
    previous_mu = expit(np.einsum("ntr,r->nt", phi, exposure_coefficients))
    optimizer_failures = 0
    final_change = float("inf")
    converged = False

    identity = np.eye(rank, dtype=np.float64)
    for iteration in range(1, maxiter + 1):
        eta = latent @ target_factors.T
        p = expit(eta)
        mu = expit(np.einsum("ntr,r->nt", phi, exposure_coefficients))
        posterior_exposure = np.divide(
            mu * (1 - p),
            np.maximum(1 - mu * p, np.finfo(float).tiny),
        )
        posterior_exposure[s == 1] = 1.0
        posterior_exposure[~w] = 0.0
        omega = np.full_like(eta, 0.25)
        nonzero_eta = np.abs(eta) >= 1e-8
        omega[nonzero_eta] = (
            np.tanh(0.5 * eta[nonzero_eta]) / (2 * eta[nonzero_eta])
        )
        kappa = s - 0.5

        for row in np.flatnonzero(active_rows):
            weights = posterior_exposure[row] * omega[row] * w[row]
            precision = l2_u * identity + target_factors.T @ (weights[:, None] * target_factors)
            rhs = l2_u * (xa[row] @ map_weights)
            rhs += target_factors.T @ (posterior_exposure[row] * kappa[row] * w[row])
            latent[row] = np.linalg.solve(precision, rhs)

        for target in range(n_targets):
            weights = posterior_exposure[:, target] * omega[:, target] * w[:, target]
            precision = l2_v * identity + latent.T @ (weights[:, None] * latent)
            rhs = latent.T @ (
                posterior_exposure[:, target] * kappa[:, target] * w[:, target]
            )
            target_factors[target] = np.linalg.solve(precision, rhs)

        active_x = xa[active_rows]
        active_latent = latent[active_rows]
        map_precision = l2_u * (active_x.T @ active_x) + l2_map * np.eye(xa.shape[1])
        map_rhs = l2_u * (active_x.T @ active_latent)
        map_weights = np.linalg.solve(map_precision, map_rhs)

        flat = w.ravel()
        exposure_coefficients, success, _, _, _ = _fit_soft_logistic(
            phi.reshape(-1, phi.shape[2])[flat],
            posterior_exposure.ravel()[flat],
            np.ones(int(np.sum(flat)), dtype=np.float64),
            l2=l2_exposure,
            initial=exposure_coefficients,
            maxiter=max(20, min(200, maxiter)),
            tolerance=max(tolerance, 1e-8),
        )
        optimizer_failures += int(not success)

        eta = latent @ target_factors.T
        p = expit(eta)
        mu = expit(np.einsum("ntr,r->nt", phi, exposure_coefficients))
        final_change = max(
            float(np.max(np.abs(p[w] - previous_p[w]))),
            float(np.max(np.abs(mu[w] - previous_mu[w]))),
        )
        previous_p, previous_mu = p.copy(), mu.copy()
        if final_change < tolerance:
            converged = True
            break

    # Predictions for new cells use the feature-conditioned prior mean Bx,
    # exactly as in the GenEML inductive prediction rule.
    predicted_p = expit((xa @ map_weights) @ target_factors.T)
    predicted_mu = expit(np.einsum("ntr,r->nt", phi, exposure_coefficients))
    objective = _observed_log_loss(s, predicted_p * predicted_mu, w)
    return GenEMLFit(
        map_weights=np.asarray(map_weights, dtype=np.float64),
        target_factors=np.asarray(target_factors, dtype=np.float64),
        exposure_coefficients=np.asarray(exposure_coefficients, dtype=np.float64),
        rank=rank,
        n_features=x.shape[1],
        n_targets=n_targets,
        n_propensity_features=phi_raw.shape[2],
        default_target_exposure=propensity_design is None,
        l2_u=l2_u,
        l2_v=l2_v,
        l2_map=l2_map,
        l2_exposure=l2_exposure,
        converged=converged,
        iterations=iteration,
        objective_value=objective,
        optimizer_failures=optimizer_failures,
        final_change=final_change,
    )


def shift_imc_unbiased_entry_loss(
    probability: Any, observed: Any, exposure: Any
) -> Array:
    """Return the generalized ShiftIMC loss before masking or averaging.

    For constant ``e=1-rho`` this equals Eq. (5) of Hsieh et al.  Its
    expectation over one-sided positive censoring is ordinary squared loss to
    the clean binary outcome.
    """

    p = np.asarray(probability, dtype=np.float64)
    s = np.asarray(observed, dtype=np.float64)
    e = np.asarray(exposure, dtype=np.float64)
    try:
        p, s, e = np.broadcast_arrays(p, s, e)
    except ValueError as error:
        raise ValueError("probability, observed, and exposure must broadcast") from error
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probability must lie in [0, 1]")
    if not np.all(np.isin(s, (0, 1))):
        raise ValueError("observed must be binary")
    if not np.all(np.isfinite(e)) or np.any(e <= 0) or np.any(e > 1):
        raise ValueError("exposure must lie in (0, 1]")
    return p**2 + (s / e) * (1 - 2 * p)


def _target_ids(
    value: Sequence[Any] | None,
    n_targets: int,
) -> tuple[str, ...]:
    identifiers = (
        tuple(map(str, range(n_targets)))
        if value is None
        else tuple(map(str, value))
    )
    if len(identifiers) != n_targets or len(set(identifiers)) != n_targets:
        raise ValueError("target_ids must be unique and align with target columns")
    return identifiers


def _target_design(
    value: Any | None,
    n_targets: int,
    target_ids: Sequence[Any] | None = None,
) -> tuple[Array, str, bool, tuple[str, ...]]:
    identifiers = _target_ids(target_ids, n_targets)
    if value is None:
        # A positional identity assigns random initialization rows to arbitrary
        # input-column positions.  Key the basis columns by stable target IDs
        # instead.  The one-hot basis is already full rank, so adding a target-
        # side intercept would create the redundant [1, I] parameterization.
        levels = tuple(sorted(identifiers))
        lookup = {identifier: index for index, identifier in enumerate(levels)}
        result = np.zeros((n_targets, n_targets), dtype=np.float64)
        result[np.arange(n_targets), [lookup[item] for item in identifiers]] = 1.0
        return result, "known_target_identity", False, identifiers
    target = _array2d(value, "target_design")
    if target.shape[0] != n_targets:
        raise ValueError("target_design must align with target columns")
    return target, "target_features", True, identifiers


@dataclass(frozen=True)
class ShiftIMCFit:
    cell_loading: Array
    target_loading: Array
    target_design: Array
    target_input_kind: str
    target_ids: tuple[str, ...]
    augment_target_intercept: bool
    rank: int
    l2: float
    n_features: int
    converged: bool
    iterations: int
    objective_value: float
    optimizer_message: str
    n_measured: int

    @property
    def model_name(self) -> str:
        return "ShiftIMC-adapted"

    def predict_reference(
        self,
        X: Any,
        target_design: Any | None = None,
        target_ids: Sequence[Any] | None = None,
    ) -> Array:
        x = _array2d(X, "X")
        if x.shape[1] != self.n_features:
            raise ValueError("prediction X does not match the fitted feature count")
        if target_design is None:
            if target_ids is None:
                target = self.target_design
            elif self.target_input_kind == "known_target_identity":
                requested_values = tuple(target_ids)
                requested = _target_ids(requested_values, len(requested_values))
                if set(requested) != set(self.target_ids):
                    raise ValueError(
                        "identity target_ids must match the fitted target set"
                    )
                levels = tuple(sorted(self.target_ids))
                lookup = {
                    identifier: index for index, identifier in enumerate(levels)
                }
                target = np.zeros((len(requested), len(levels)), dtype=np.float64)
                target[
                    np.arange(len(requested)),
                    [lookup[item] for item in requested],
                ] = 1.0
            else:
                requested = _target_ids(target_ids, len(self.target_ids))
                # JSON checkpoint restoration materializes tuples as lists.
                # Compare canonical tuples so an otherwise identical restored
                # feature-design fit does not reject aligned target IDs.
                if requested != tuple(self.target_ids):
                    raise ValueError(
                        "target feature prediction requires an aligned target_design"
                    )
                target = self.target_design
        else:
            target = _array2d(target_design, "target_design")
            if target.shape[1] != self.target_design.shape[1]:
                raise ValueError("prediction target design does not match fitted columns")
            if target_ids is not None:
                _target_ids(target_ids, target.shape[0])
        target_features = (
            _augment_intercept(target)
            if self.augment_target_intercept
            else target
        )
        score = (_augment_intercept(x) @ self.cell_loading) @ (
            target_features @ self.target_loading
        ).T
        return expit(score)

    def predict_observed(
        self,
        X: Any,
        exposure: Any,
        target_design: Any | None = None,
        target_ids: Sequence[Any] | None = None,
    ) -> Array:
        p = self.predict_reference(X, target_design, target_ids)
        return p * _prediction_exposure(exposure, p.shape)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "selected_structure": "bounded_factorized_inductive_shift",
            "rank": self.rank,
            "l2": self.l2,
            "target_input_kind": self.target_input_kind,
            "target_ids": list(self.target_ids),
            "augment_target_intercept": self.augment_target_intercept,
            "final_converged": self.converged,
            "final_iterations": self.iterations,
            "final_objective": self.objective_value,
            "optimizer_message": self.optimizer_message,
            "n_measured": self.n_measured,
            "probability_semantics": "reference_p_with_supplied_exposure",
            "adaptations": (
                "known-W exclusion|entry-specific e|sigmoid bound|"
                "factorized nuclear surrogate|canonical target-ID identity"
            ),
        }


def fit_shift_imc_adapted(
    X: Any,
    observed: Any,
    measured: Any,
    exposure: Any,
    target_design: Any | None = None,
    *,
    target_ids: Sequence[Any] | None = None,
    rank: int = 2,
    l2: float = 1e-2,
    maxiter: int = 500,
    tolerance: float = 1e-8,
    seed: int = 0,
) -> ShiftIMCFit:
    """Fit bounded factorized ShiftIMC using only measured PU entries."""

    x = _array2d(X, "X")
    s, w = _labels(observed, measured, len(x))
    e = _exposure(exposure, s.shape, w)
    target, target_kind, augment_target_intercept, fitted_target_ids = _target_design(
        target_design, s.shape[1], target_ids
    )
    xa = _augment_intercept(x)
    ya = _augment_intercept(target) if augment_target_intercept else target
    rank = _positive_int(rank, "rank")
    if rank > min(xa.shape[1], ya.shape[1], s.shape[1]):
        raise ValueError("rank exceeds the augmented inductive feature cap")
    l2 = _positive_float(l2, "l2")
    maxiter = _positive_int(maxiter, "maxiter")
    tolerance = _positive_float(tolerance, "tolerance")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    a0 = rng.normal(0.0, 0.03, size=(xa.shape[1], rank))
    b0 = rng.normal(0.0, 0.03, size=(ya.shape[1], rank))
    theta0 = np.concatenate((a0.ravel(), b0.ravel()))
    count = int(np.sum(w))

    def unpack(theta: Array) -> tuple[Array, Array]:
        split = xa.shape[1] * rank
        return theta[:split].reshape(xa.shape[1], rank), theta[split:].reshape(ya.shape[1], rank)

    def objective(theta: Array) -> tuple[float, Array]:
        a, b = unpack(theta)
        u, v = xa @ a, ya @ b
        p = expit(u @ v.T)
        loss = float(np.sum(np.where(w, shift_imc_unbiased_entry_loss(p, s, e), 0.0)) / count)
        loss += 0.5 * l2 * float(np.sum(a**2) + np.sum(b**2))
        grad_eta = np.where(w, 2 * (p - s / e) * p * (1 - p) / count, 0.0)
        grad_a = xa.T @ (grad_eta @ v) + l2 * a
        grad_b = ya.T @ (grad_eta.T @ u) + l2 * b
        return loss, np.concatenate((grad_a.ravel(), grad_b.ravel()))

    fitted = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": tolerance, "gtol": tolerance, "maxls": 30},
    )
    if not np.isfinite(fitted.fun) or not np.all(np.isfinite(fitted.x)):
        raise FloatingPointError("ShiftIMC optimization produced non-finite values")
    a, b = unpack(np.asarray(fitted.x, dtype=np.float64))
    return ShiftIMCFit(
        a,
        b,
        target.copy(),
        target_kind,
        fitted_target_ids,
        augment_target_intercept,
        rank,
        l2,
        x.shape[1],
        bool(fitted.success),
        int(fitted.nit),
        float(fitted.fun),
        str(fitted.message),
        count,
    )


@dataclass(frozen=True)
class SAREMFit:
    classifier_coefficients: Array
    propensity_coefficients: Array
    n_features: int
    n_targets: int
    n_propensity_features: int
    default_target_exposure: bool
    l2_classifier: float
    l2_propensity: float
    local_certainty: float
    converged: bool
    iterations: int
    objective_value: float
    convergence_slope: float
    optimizer_failures: int

    @property
    def model_name(self) -> str:
        return "SAR-EM-adapted"

    def predict_reference(self, X: Any) -> Array:
        x = _array2d(X, "X")
        if x.shape[1] != self.n_features:
            raise ValueError("prediction X does not match the fitted feature count")
        return expit(_augment_intercept(x) @ self.classifier_coefficients)

    def predict_exposure(self, X: Any, propensity_design: Any | None = None) -> Array:
        x = _array2d(X, "X")
        if x.shape[1] != self.n_features:
            raise ValueError("prediction X does not match the fitted feature count")
        if (propensity_design is None) != self.default_target_exposure:
            raise ValueError("prediction propensity design must use the fitted schema")
        phi = _propensity_array(propensity_design, (len(x), self.n_targets))
        if phi.shape[2] != self.n_propensity_features:
            raise ValueError("prediction propensity design does not match the fitted schema")
        return expit(np.einsum("ntr,r->nt", _augment_propensity(phi), self.propensity_coefficients))

    def predict_observed(self, X: Any, propensity_design: Any | None = None) -> Array:
        return self.predict_reference(X) * self.predict_exposure(X, propensity_design)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "selected_structure": "independent_target_logistic_sar_em",
            "rank": 0,
            "l2_classifier": self.l2_classifier,
            "l2_propensity": self.l2_propensity,
            "local_certainty": self.local_certainty,
            "final_converged": self.converged,
            "final_iterations": self.iterations,
            "final_objective": self.objective_value,
            "convergence_slope": self.convergence_slope,
            "optimizer_failures": self.optimizer_failures,
            "probability_semantics": "reference_p_with_model_exposure",
            "information_access": "observed labels and declared propensity covariates only",
            "adaptations": (
                "clean-room multi-target EM|shared dyadic propensity|"
                "local-certainty stabilization"
            ),
        }


def _history_slope(history: Sequence[Array]) -> float:
    if len(history) < 2:
        return float("inf")
    values = np.stack(history, axis=0)
    time = np.arange(len(values), dtype=np.float64)
    centered = time - time.mean()
    denominator = float(centered @ centered)
    slopes = np.tensordot(centered, values, axes=(0, 0)) / denominator
    return float(np.mean(np.abs(slopes)))


def fit_sar_em(
    X: Any,
    observed: Any,
    measured: Any,
    propensity_design: Any | None = None,
    *,
    l2_classifier: float = 1e-2,
    l2_propensity: float = 1e-2,
    local_certainty: float = 0.9,
    maxiter: int = 100,
    tolerance: float = 1e-4,
    convergence_window: int = 10,
    inner_maxiter: int = 200,
    seed: int = 0,
) -> SAREMFit:
    """Fit the legacy clean-room SAR-EM adaptation.

    This function is not the authors' implementation and is no longer used by
    the measurement benchmark.  It remains available for backward-compatible
    diagnostics only.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    # The convex logistic subproblems and initialization are deterministic.  We
    # still validate and record a tuning seed so the public API is uniform.
    x = _array2d(X, "X")
    s, w = _labels(observed, measured, len(x))
    if np.any(np.sum(w, axis=0) == 0):
        raise ValueError("SAR-EM requires at least one measured row for every target")
    phi_raw = _propensity_array(propensity_design, s.shape)
    phi = _augment_propensity(phi_raw)
    xa = _augment_intercept(x)
    l2_classifier = _positive_float(l2_classifier, "l2_classifier")
    l2_propensity = _positive_float(l2_propensity, "l2_propensity")
    if not np.isfinite(local_certainty) or not 0 < float(local_certainty) <= 1:
        raise ValueError("local_certainty must lie in (0, 1]")
    local_certainty = float(local_certainty)
    maxiter = _positive_int(maxiter, "maxiter")
    inner_maxiter = _positive_int(inner_maxiter, "inner_maxiter")
    convergence_window = _positive_int(convergence_window, "convergence_window")
    if convergence_window < 2:
        raise ValueError("convergence_window must be at least two")
    tolerance = _positive_float(tolerance, "tolerance")
    n_targets = s.shape[1]
    classifier = np.zeros((xa.shape[1], n_targets), dtype=np.float64)
    optimizer_failures = 0

    # Algorithm 2: initialize the classifier through the SCAR approximation.
    expected_y = np.zeros_like(s)
    for target in range(n_targets):
        rows = w[:, target]
        beta, success, _, _, _ = _fit_soft_logistic(
            xa[rows],
            s[rows, target],
            np.ones(int(np.sum(rows))),
            l2=l2_classifier,
            initial=None,
            maxiter=inner_maxiter,
            tolerance=max(tolerance, 1e-8),
        )
        optimizer_failures += int(not success)
        q = expit(xa[rows] @ beta)
        c = float(np.clip(np.max(q), 1e-3, 1 - 1e-3))
        posterior = np.divide(
            q * (1 - c),
            np.maximum(c * (1 - q), np.finfo(float).tiny),
        )
        posterior = np.clip(posterior, 0.0, 1.0)
        expected_y[rows, target] = np.where(s[rows, target] == 1, 1.0, posterior)
        beta, success, _, _, _ = _fit_soft_logistic(
            xa[rows],
            expected_y[rows, target],
            np.ones(int(np.sum(rows))),
            l2=l2_classifier,
            initial=beta,
            maxiter=inner_maxiter,
            tolerance=max(tolerance, 1e-8),
        )
        optimizer_failures += int(not success)
        classifier[:, target] = beta

    flat = w.ravel()
    # Initialize the propensity model to the empirical expected label frequency,
    # then let the declared design explain systematic variation during EM.
    initial_c = float(np.clip(np.sum(s[w]) / max(np.sum(expected_y[w]), _EPS), 1e-3, 1 - 1e-3))
    propensity = np.zeros(phi.shape[2], dtype=np.float64)
    propensity[0] = np.log(initial_c) - np.log1p(-initial_c)
    history: list[Array] = []
    convergence_slope = float("inf")
    converged = False

    for iteration in range(1, maxiter + 1):
        p = expit(xa @ classifier)
        e = expit(np.einsum("ntr,r->nt", phi, propensity))
        decayed_e = local_certainty * e
        posterior = np.divide(
            p * (1 - decayed_e),
            np.maximum(1 - p * decayed_e, np.finfo(float).tiny),
        )
        expected_y = np.where(s == 1, 1.0, posterior)
        expected_y[~w] = 0.0

        for target in range(n_targets):
            rows = w[:, target]
            classifier[:, target], success, _, _, _ = _fit_soft_logistic(
                xa[rows],
                expected_y[rows, target],
                np.ones(int(np.sum(rows))),
                l2=l2_classifier,
                initial=classifier[:, target],
                maxiter=inner_maxiter,
                tolerance=max(tolerance, 1e-8),
            )
            optimizer_failures += int(not success)

        propensity, success, _, _, _ = _fit_soft_logistic(
            phi.reshape(-1, phi.shape[2])[flat],
            s.ravel()[flat],
            expected_y.ravel()[flat],
            l2=l2_propensity,
            initial=propensity,
            maxiter=inner_maxiter,
            tolerance=max(tolerance, 1e-8),
        )
        optimizer_failures += int(not success)
        e = expit(np.einsum("ntr,r->nt", phi, propensity))
        history.append(e[w].copy())
        history = history[-convergence_window:]
        if len(history) == convergence_window:
            convergence_slope = _history_slope(history)
            if convergence_slope < tolerance:
                converged = True
                break

    p = expit(xa @ classifier)
    e = expit(np.einsum("ntr,r->nt", phi, propensity))
    objective = _observed_log_loss(s, p * e, w)
    return SAREMFit(
        classifier_coefficients=classifier,
        propensity_coefficients=propensity,
        n_features=x.shape[1],
        n_targets=n_targets,
        n_propensity_features=phi_raw.shape[2],
        default_target_exposure=propensity_design is None,
        l2_classifier=l2_classifier,
        l2_propensity=l2_propensity,
        local_certainty=local_certainty,
        converged=converged,
        iterations=iteration,
        objective_value=objective,
        convergence_slope=convergence_slope,
        optimizer_failures=optimizer_failures,
    )


def _unique_candidates(candidates: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if not candidates:
        raise ValueError("at least one candidate configuration is required")
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        value = dict(candidate)
        unique.setdefault(canonical_json(value), value)
    result = tuple(unique.values())
    if len(result) > _MAX_COMPARATOR_CANDIDATES:
        raise ValueError(
            f"at most {_MAX_COMPARATOR_CANDIDATES} unique comparator candidates are allowed"
        )
    return result


def _tuning_result(
    model: str,
    fits: Sequence[Any],
    configs: Sequence[Mapping[str, Any]],
    validation_losses: Sequence[float],
    seeds: Sequence[int],
    *,
    require_converged: bool = False,
) -> ComparatorTuningResult:
    trials = tuple(
        ComparatorTrial(
            model=model,
            index=index,
            config=dict(config),
            validation_observed_log_loss=float(loss),
            converged=bool(fitted.converged),
            iterations=int(fitted.iterations),
            objective_value=float(fitted.objective_value),
            seed=int(trial_seed),
        )
        for index, (fitted, config, loss, trial_seed) in enumerate(
            zip(fits, configs, validation_losses, seeds)
        )
    )
    eligible = [
        index for index, trial in enumerate(trials)
        if np.isfinite(trial.validation_observed_log_loss)
        and (trial.converged or not require_converged)
    ]
    if not eligible:
        qualifier = "converged finite" if require_converged else "finite"
        raise ValueError(f"{model} has no {qualifier} candidate")
    winner = min(
        eligible,
        key=lambda index: (
            not trials[index].converged,
            trials[index].validation_observed_log_loss,
            int(configs[index].get("rank", 0)),
            -float(configs[index].get("l2", configs[index].get("l2_classifier", 0.0))),
            index,
        ),
    )
    return ComparatorTuningResult(model, fits[winner], dict(configs[winner]), trials)


def tune_geneml_adapted(
    train_X: Any,
    train_observed: Any,
    train_measured: Any,
    validation_X: Any,
    validation_observed: Any,
    validation_measured: Any,
    *,
    train_propensity_design: Any | None = None,
    validation_propensity_design: Any | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    maxiter: int = 100,
    tolerance: float = 1e-5,
    seed: int = 0,
) -> ComparatorTuningResult:
    x_train = _array2d(train_X, "train_X")
    x_validation = _array2d(validation_X, "validation_X")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("training and validation feature counts differ")
    s_validation, w_validation = _labels(
        validation_observed, validation_measured, len(x_validation)
    )
    maximum = min(x_train.shape[1] + 1, np.asarray(train_observed).shape[1])
    if candidates is None:
        ranks = tuple(sorted({1, min(2, maximum), maximum}))
        candidates = tuple(
            {"rank": rank, "l2_u": penalty, "l2_v": penalty,
             "l2_map": penalty, "l2_exposure": penalty}
            for rank in ranks for penalty in (1e-2, 1e-1)
        )
    configs = _unique_candidates(candidates)
    fits, losses, seeds = [], [], []
    for config in configs:
        trial_seed = stable_seed(seed, "geneml-adapted", canonical_json(config))
        fitted = fit_geneml_adapted(
            x_train,
            train_observed,
            train_measured,
            train_propensity_design,
            **config,
            maxiter=maxiter,
            tolerance=tolerance,
            seed=trial_seed,
        )
        q = fitted.predict_observed(x_validation, validation_propensity_design)
        fits.append(fitted)
        losses.append(_observed_log_loss(s_validation, q, w_validation))
        seeds.append(trial_seed)
    return _tuning_result("GenEML-adapted", fits, configs, losses, seeds)


def tune_shift_imc_adapted(
    train_X: Any,
    train_observed: Any,
    train_measured: Any,
    train_exposure: Any,
    validation_X: Any,
    validation_observed: Any,
    validation_measured: Any,
    validation_exposure: Any,
    *,
    train_target_design: Any | None = None,
    validation_target_design: Any | None = None,
    target_ids: Sequence[Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    maxiter: int = 500,
    tolerance: float = 1e-8,
    seed: int = 0,
) -> ComparatorTuningResult:
    x_train = _array2d(train_X, "train_X")
    x_validation = _array2d(validation_X, "validation_X")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("training and validation feature counts differ")
    s_validation, w_validation = _labels(
        validation_observed, validation_measured, len(x_validation)
    )
    e_validation = _exposure(validation_exposure, s_validation.shape, w_validation)
    target, _, augment_target_intercept, _ = _target_design(
        train_target_design, np.asarray(train_observed).shape[1], target_ids
    )
    if validation_target_design is not None:
        validation_target = _array2d(validation_target_design, "validation_target_design")
        if validation_target.shape != target.shape:
            raise ValueError("training and validation target designs differ")
    else:
        validation_target = None
    target_feature_count = target.shape[1] + int(augment_target_intercept)
    maximum = min(x_train.shape[1] + 1, target_feature_count, len(target))
    if candidates is None:
        ranks = tuple(sorted({1, min(2, maximum), maximum}))
        candidates = tuple(
            {"rank": rank, "l2": penalty}
            for rank in ranks for penalty in (1e-2, 1e-1)
        )
    configs = _unique_candidates(candidates)
    fits, losses, seeds = [], [], []
    for config in configs:
        trial_seed = stable_seed(seed, "shiftimc-adapted", canonical_json(config))
        fitted = fit_shift_imc_adapted(
            x_train,
            train_observed,
            train_measured,
            train_exposure,
            train_target_design,
            target_ids=target_ids,
            **config,
            maxiter=maxiter,
            tolerance=tolerance,
            seed=trial_seed,
        )
        p = fitted.predict_reference(
            x_validation, validation_target, target_ids=target_ids
        )
        fits.append(fitted)
        losses.append(_observed_log_loss(s_validation, p * e_validation, w_validation))
        seeds.append(trial_seed)
    return _tuning_result(
        "ShiftIMC-adapted", fits, configs, losses, seeds,
        require_converged=True,
    )


def tune_sar_em(
    train_X: Any,
    train_observed: Any,
    train_measured: Any,
    validation_X: Any,
    validation_observed: Any,
    validation_measured: Any,
    *,
    train_propensity_design: Any | None = None,
    validation_propensity_design: Any | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    local_certainty: float = 0.9,
    maxiter: int = 100,
    tolerance: float = 1e-4,
    convergence_window: int = 10,
    inner_maxiter: int = 200,
    seed: int = 0,
) -> ComparatorTuningResult:
    x_train = _array2d(train_X, "train_X")
    x_validation = _array2d(validation_X, "validation_X")
    if x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("training and validation feature counts differ")
    s_validation, w_validation = _labels(
        validation_observed, validation_measured, len(x_validation)
    )
    if candidates is None:
        candidates = tuple(
            {"l2_classifier": classifier, "l2_propensity": propensity}
            for classifier in (1e-2, 1e-1)
            for propensity in (1e-2, 1e-1)
        )
    configs = _unique_candidates(candidates)
    fits, losses, seeds = [], [], []
    for config in configs:
        trial_seed = stable_seed(seed, "sar-em", canonical_json(config))
        fitted = fit_sar_em(
            x_train,
            train_observed,
            train_measured,
            train_propensity_design,
            **config,
            local_certainty=local_certainty,
            maxiter=maxiter,
            tolerance=tolerance,
            convergence_window=convergence_window,
            inner_maxiter=inner_maxiter,
            seed=trial_seed,
        )
        q = fitted.predict_observed(x_validation, validation_propensity_design)
        fits.append(fitted)
        losses.append(_observed_log_loss(s_validation, q, w_validation))
        seeds.append(trial_seed)
    return _tuning_result("SAR-EM", fits, configs, losses, seeds)


__all__ = [
    "AssayTargetPropensityEncoder",
    "PerTargetAssayPropensityEncoder",
    "ComparatorTrial",
    "ComparatorTuningResult",
    "GenEMLFit",
    "SAREMFit",
    "ShiftIMCFit",
    "fit_geneml_adapted",
    "fit_sar_em",
    "fit_shift_imc_adapted",
    "shift_imc_unbiased_entry_loss",
    "tune_geneml_adapted",
    "tune_sar_em",
    "tune_shift_imc_adapted",
]
