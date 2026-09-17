"""Verified authors-kernel wrapper around the SAR-PU implementation.

The vendored statistical routine is called once per target after excluding
rows for which that target was not measured.  Gene2Wire supplies only a thin
measurement-mask wrapper, a current-scikit-learn compatibility estimator, and
strict validation/diagnostics.  Reference labels and true propensities are not
accepted by this API.
"""

from __future__ import annotations

import _imp
import hashlib
import importlib
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .._vendor import sarpu as _VENDOR_PACKAGE
from ..seeds import stable_seed


Array = np.ndarray
_UPSTREAM_REPOSITORY = "https://github.com/ML-KULeuven/SAR-PU"
_UPSTREAM_COMMIT = "6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7"
_UPSTREAM_SHA256 = {
    "pu_learning.py": "6d929a4db0555b2d2e1ef05686618d819decc0bdfa3bbdcab59fec8ba4261b43",
    "PUmodels.py": "28b41fd8adb50231c500b937de2ef924c9bc6fb0e597d574b1a6b1c91035403a",
    "LICENSE": "3ffe98aa8f5153033951f5cbc2322c8c8943704827dcfefd188eea72b5dec52e",
}
_VENDORED_SHA256 = {
    "pu_learning.py": _UPSTREAM_SHA256["pu_learning.py"],
    # Upstream omits the final LF; the vendored Python tokens are identical.
    "PUmodels.py": "bbef08af6c9b7dae2536dc07602cfdf355757ca6874cc1e66a55b5d890547cb4",
    "LICENSE": _UPSTREAM_SHA256["LICENSE"],
}
_COMPATIBILITY_DESCRIPTION = (
    "Per-target exclusion of W=0 rows; target-specific propensity columns are "
    "reduced outcome-blindly to an intercept-plus-full-rank centered-orthonormal "
    "assay/QC design, appended to cell X, and selected only by the upstream "
    "propensity model; a "
    "LogisticRegressionPU subclass removes obsolete scikit-learn constructor "
    "arguments while retaining the upstream liblinear L2 defaults; deterministic "
    "random_state values, validation, and fail-closed diagnostics are added."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


SARPU_AUTHORS_PROVENANCE: Mapping[str, Any] = {
    "repository": _UPSTREAM_REPOSITORY,
    "commit": _UPSTREAM_COMMIT,
    "source_repository": _UPSTREAM_REPOSITORY,
    "source_commit": _UPSTREAM_COMMIT,
    "source_file_hashes": dict(_VENDORED_SHA256),
    "license": "MIT",
    # The vendored EM function itself is the pinned authors' source, but this
    # module does not execute their unmodified end-to-end program: it supplies
    # a current-sklearn estimator and a per-target known-W wrapper.
    "authors_code": False,
    "authors_source": True,
    "executes_unmodified_authors_code": False,
    "uses_unmodified_authors_em_kernel": True,
    "adapted": True,
    "implementation_variant": "authors_sar_em_per_target_measurement_wrapper",
    "upstream_source_sha256": dict(_UPSTREAM_SHA256),
    "vendored_source_sha256": dict(_VENDORED_SHA256),
    "compatibility_patch": _COMPATIBILITY_DESCRIPTION,
    "compatibility_patch_sha256": hashlib.sha256(
        _COMPATIBILITY_DESCRIPTION.encode("utf-8")
    ).hexdigest(),
    "wrapper_source_sha256": _sha256_file(Path(__file__)),
}


class SARPUAuthorsError(RuntimeError):
    """Base class for a fail-closed authors-code comparator failure."""


class SARPUAuthorsSourceError(SARPUAuthorsError):
    """Raised when the pinned authors' source snapshot does not match."""


class SARPUAuthorsConvergenceError(SARPUAuthorsError):
    """Raised when an upstream target fit does not demonstrably converge."""


class SARPUAuthorsSupportError(ValueError):
    """Raised when a target lacks the labeled/unlabeled measured support."""


def verify_sarpu_vendor() -> dict[str, str]:
    """Re-hash the pinned authors' files and fail closed on any drift."""
    root = Path(_VENDOR_PACKAGE.__file__).resolve().parent
    try:
        actual = {name: _sha256_file(root / name) for name in _VENDORED_SHA256}
    except OSError as error:
        raise SARPUAuthorsSourceError(
            f"Pinned SAR-PU source verification failed: {error}"
        ) from error
    mismatches = {
        name: {"expected": _VENDORED_SHA256[name], "actual": value}
        for name, value in actual.items()
        if value != _VENDORED_SHA256[name]
    }
    if mismatches:
        raise SARPUAuthorsSourceError(
            "Pinned SAR-PU source verification failed: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


_VERIFIED_SOURCE_HASHES = verify_sarpu_vendor()
_PU_MODELS = importlib.import_module("gene2wire._vendor.sarpu.PUmodels")


def _load_upstream_module():
    """Load pristine upstream code while containing its Python-2-style import."""

    module_name = "gene2wire._vendor.sarpu.pu_learning"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    _imp.acquire_lock()
    old_package = sys.modules.get("sarpu")
    old_models = sys.modules.get("sarpu.PUmodels")
    try:
        sys.modules["sarpu"] = _VENDOR_PACKAGE
        sys.modules["sarpu.PUmodels"] = _PU_MODELS
        return importlib.import_module(module_name)
    finally:
        if old_package is None:
            sys.modules.pop("sarpu", None)
        else:
            sys.modules["sarpu"] = old_package
        if old_models is None:
            sys.modules.pop("sarpu.PUmodels", None)
        else:
            sys.modules["sarpu.PUmodels"] = old_models
        _imp.release_lock()


_UPSTREAM = _load_upstream_module()


class CompatibleLogisticRegressionPU(LogisticRegression, _PU_MODELS.BasePU):
    """Upstream LogisticRegressionPU behavior on supported sklearn releases.

    Upstream passes the removed ``multi_class`` argument and the now-ignored
    ``n_jobs`` argument. Omitting them retains its explicitly requested
    L2/liblinear estimator. The PU weighting transformation is copied without
    alteration from upstream ``BasePU`` and only the modern keyword call is
    different.
    """

    def __init__(
        self, *, penalty: str = "l2", random_state: int | None = None
    ) -> None:
        if penalty != "l2":
            raise ValueError("The pinned authors' estimator requires penalty='l2'")
        super().__init__(
            penalty=penalty,
            C=1.0,
            dual=False,
            tol=1e-4,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=random_state,
            solver="liblinear",
            max_iter=100,
            verbose=0,
            warm_start=False,
        )

    def fit(
        self,
        x: Any,
        s: Any,
        e: Any | None = None,
        sample_weight: Any | None = None,
    ) -> "CompatibleLogisticRegressionPU":
        # sklearn 1.8 deprecates the ``penalty`` keyword even though the
        # pinned upstream class explicitly sets ``penalty='l2'``. Contain that
        # compatibility-only warning while preserving the authors' estimator.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"'penalty' was deprecated.*",
                category=FutureWarning,
            )
            if e is None:
                LogisticRegression.fit(self, x, s, sample_weight=sample_weight)
            else:
                xp, yp, wp = self._make_propensity_weighted_data(
                    np.asarray(x), np.asarray(s), np.asarray(e), sample_weight
                )
                LogisticRegression.fit(self, xp, yp, sample_weight=wp)
        return self


def _array2d(value: Any, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _fit_array2d(value: Any, name: str) -> Array:
    """Validate shape while deferring finiteness to target-measured rows."""

    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    return array


def _binary_matrix(value: Any, name: str, n_rows: int) -> Array:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != n_rows or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_samples, n_targets)")
    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be binary") from error
    if not np.all(np.isfinite(numeric)) or not np.all((numeric == 0) | (numeric == 1)):
        raise ValueError(f"{name} must contain only finite binary values")
    return numeric.astype(np.int8, copy=False)


def _observed_matrix(value: Any, measured: Array) -> Array:
    """Validate observed labels on W and erase arbitrary payload outside W."""

    raw = np.asarray(value)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.shape != measured.shape:
        raise ValueError("observed and measured must have the same shape")

    # A literal positive outside W is a contract violation.  Other outside-W
    # payload (including NaN/Inf or sentinel values) is semantically absent and
    # is never passed to the authors' routine.
    for item in raw[~measured].ravel():
        try:
            numeric_item = float(item)
        except (TypeError, ValueError):
            continue
        if numeric_item == 1.0:
            raise ValueError("observed positives cannot occur outside the measurement mask")

    try:
        included = np.asarray(raw[measured], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("observed labels on measured entries must be binary") from error
    if (
        not np.all(np.isfinite(included))
        or not np.all((included == 0) | (included == 1))
    ):
        raise ValueError("observed labels on measured entries must be finite and binary")
    result = np.zeros(measured.shape, dtype=np.int8)
    result[measured] = included.astype(np.int8, copy=False)
    return result


def _propensity_array(value: Any | None, shape: tuple[int, int]) -> Array:
    if value is None:
        return np.empty((*shape, 0), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[:2] != shape:
        raise ValueError(
            "propensity_design must have shape (n_samples, n_targets, n_features)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("propensity_design must contain only finite values")
    return array


def _fit_propensity_array(
    value: Any | None,
    measured: Array,
) -> Array:
    """Validate/copy only W-included target-specific propensity rows."""

    if value is None:
        return np.empty((*measured.shape, 0), dtype=np.float64)
    raw = np.asarray(value)
    if raw.ndim != 3 or raw.shape[:2] != measured.shape:
        raise ValueError(
            "propensity_design must have shape (n_samples, n_targets, n_features)"
        )
    result = np.zeros(raw.shape, dtype=np.float64)
    for target in range(measured.shape[1]):
        rows = measured[:, target]
        try:
            included = np.asarray(raw[rows, target, :], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "propensity_design on measured entries must be numeric"
            ) from error
        if not np.all(np.isfinite(included)):
            raise ValueError(
                "propensity_design on measured entries must contain only finite values"
            )
        result[rows, target, :] = included
    return result


def _feature_indices(
    value: Sequence[int] | Array | None,
    n_features: int,
) -> Array:
    if value is None:
        return np.arange(n_features, dtype=np.int64)
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError("classification_attributes must be one-dimensional")
    if raw.dtype == np.bool_:
        if len(raw) != n_features:
            raise ValueError(
                "boolean classification_attributes must match the X feature count"
            )
        indices = np.flatnonzero(raw)
    else:
        if raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("classification_attributes must be nonempty integer indices")
        indices = raw.astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError("classification_attributes must select at least one feature")
    if np.any(indices < 0) or np.any(indices >= n_features):
        raise ValueError("classification_attributes contains an out-of-range index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("classification_attributes must not contain duplicates")
    return np.sort(indices).astype(np.int64, copy=False)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _validated_probability(value: Any, shape: tuple[int, ...], name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise FloatingPointError(f"SAR-PU produced invalid {name} values")
    if np.any(array < 0) or np.any(array > 1):
        raise FloatingPointError(f"SAR-PU produced out-of-range {name} values")
    return array


def _independent_propensity_columns(value: Array) -> np.ndarray:
    """Select a deterministic full-rank subset after an implicit intercept.

    The selection depends only on the declared propensity design. It prevents
    padding, constant QC variables, or accidentally duplicated columns from
    reaching the authors' logistic optimizer.
    """

    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("target propensity design must be two-dimensional")
    working = np.ones((len(matrix), 1), dtype=np.float64)
    rank = 1
    selected: list[int] = []
    for column in range(matrix.shape[1]):
        proposed = np.column_stack((working, matrix[:, column]))
        proposed_rank = int(np.linalg.matrix_rank(proposed))
        if proposed_rank == rank + 1:
            selected.append(column)
            working = proposed
            rank = proposed_rank
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class SARPUAuthorsFit:
    """Compact, checkpoint-safe fitted state for per-target authors' SAR-EM."""

    classifier_coefficients: Array
    classifier_intercepts: Array
    propensity_coefficients: Array
    propensity_intercepts: Array
    propensity_constants: Array
    propensity_active_mask: Array
    target_propensity_design_ranks: Array
    classification_attributes: Array
    classifier_seeds: Array
    propensity_seeds: Array
    target_measured_counts: Array
    target_positive_counts: Array
    target_unlabeled_counts: Array
    target_iterations: Array
    target_fit_attempts: Array
    target_retried: Array
    target_max_its: Array
    target_objective_values: Array
    final_upstream_loglikelihoods: Array
    final_propensity_slopes: Array
    final_ll_improvements: Array
    n_features: int
    n_targets: int
    n_propensity_features: int
    target_ids: tuple[str, ...]
    seed: int
    max_its: int
    retry_max_its: int | None
    slope_eps: float
    ll_eps: float
    convergence_window: int
    refit_classifier: bool
    converged: bool
    iterations: int
    objective_value: float
    warning_messages: tuple[str, ...]
    sklearn_version: str
    upstream_commit: str
    vendored_pu_learning_sha256: str
    vendored_pumodels_sha256: str
    wrapper_source_sha256: str

    @property
    def model_name(self) -> str:
        return "SAR-PU"

    def _prediction_x(self, X: Any) -> Array:
        x = _array2d(X, "X")
        if x.shape[1] != self.n_features:
            raise ValueError("prediction X does not match the fitted feature count")
        return x

    def predict_reference(self, X: Any) -> Array:
        x = self._prediction_x(X)
        selected = x[:, np.asarray(self.classification_attributes, dtype=np.int64)]
        result = expit(selected @ self.classifier_coefficients + self.classifier_intercepts)
        return _validated_probability(result, (len(x), self.n_targets), "p")

    def predict_exposure(self, X: Any, propensity_design: Any | None = None) -> Array:
        x = self._prediction_x(X)
        design = _propensity_array(propensity_design, (len(x), self.n_targets))
        if design.shape[2] != self.n_propensity_features:
            raise ValueError("prediction propensity_design does not match the fitted schema")
        if self.n_propensity_features == 0:
            result = np.broadcast_to(
                self.propensity_constants, (len(x), self.n_targets)
            ).copy()
        else:
            result = expit(
                np.einsum("ntr,rt->nt", design, self.propensity_coefficients)
                + self.propensity_intercepts
            )
            constant = ~np.any(self.propensity_active_mask, axis=0)
            if np.any(constant):
                result[:, constant] = self.propensity_constants[constant]
        return _validated_probability(result, (len(x), self.n_targets), "e")

    def predict_observed(self, X: Any, propensity_design: Any | None = None) -> Array:
        p = self.predict_reference(X)
        e = self.predict_exposure(X, propensity_design)
        return _validated_probability(p * e, p.shape, "q")

    def predict_p(self, X: Any) -> Array:
        return self.predict_reference(X)

    def predict_e(self, X: Any, propensity_design: Any | None = None) -> Array:
        return self.predict_exposure(X, propensity_design)

    def predict_q(self, X: Any, propensity_design: Any | None = None) -> Array:
        return self.predict_observed(X, propensity_design)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "selected_structure": "authors_sar_em_per_target_measurement_wrapper",
            "rank": 0,
            "final_converged": bool(self.converged),
            "final_iterations": int(self.iterations),
            "final_objective": float(self.objective_value),
            "target_iterations": [int(value) for value in self.target_iterations],
            "target_fit_attempts": [
                int(value) for value in self.target_fit_attempts
            ],
            "target_retried": [bool(value) for value in self.target_retried],
            "target_max_its": [int(value) for value in self.target_max_its],
            "target_measured_counts": [int(value) for value in self.target_measured_counts],
            "target_positive_counts": [int(value) for value in self.target_positive_counts],
            "target_unlabeled_counts": [int(value) for value in self.target_unlabeled_counts],
            "target_propensity_design_ranks": [
                int(value) for value in self.target_propensity_design_ranks
            ],
            "target_ids": list(self.target_ids),
            "final_propensity_slopes": [float(value) for value in self.final_propensity_slopes],
            "final_ll_improvements": [float(value) for value in self.final_ll_improvements],
            "probability_semantics": "p=reference,e=selection_given_positive,q=p*e",
            "information_access": (
                "observed labels, cell X, known W-row inclusion, and declared "
                "propensity design only; no reference labels or true propensities"
            ),
            "authors_code": False,
            "authors_source": True,
            "executes_unmodified_authors_code": False,
            "uses_unmodified_authors_em_kernel": True,
            "source_faithful": False,
            "authors_repository": _UPSTREAM_REPOSITORY,
            "authors_commit": self.upstream_commit,
            "compatibility_patch_sha256": SARPU_AUTHORS_PROVENANCE[
                "compatibility_patch_sha256"
            ],
            "wrapper_source_sha256": self.wrapper_source_sha256,
            "sklearn_version": self.sklearn_version,
            "max_its": int(self.max_its),
            "retry_max_its": (
                None if self.retry_max_its is None else int(self.retry_max_its)
            ),
            "slope_eps": float(self.slope_eps),
            "ll_eps": float(self.ll_eps),
            "convergence_window": int(self.convergence_window),
            "refit_classifier": bool(self.refit_classifier),
            "warning_messages": list(self.warning_messages),
        }


def fit_sarpu_authors(
    X: Any,
    observed: Any,
    measured: Any,
    propensity_design: Any | None = None,
    *,
    target_ids: Sequence[Any] | None = None,
    classification_attributes: Sequence[int] | Array | None = None,
    max_its: int = 500,
    retry_max_its: int | None = None,
    slope_eps: float = 1e-4,
    ll_eps: float = 1e-4,
    convergence_window: int = 10,
    refit_classifier: bool = True,
    seed: int = 0,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> SARPUAuthorsFit:
    """Fit the authors' binary SAR-EM independently for each measured target.

    For target ``j``, only rows where ``measured[:, j] == 1`` are passed to
    upstream.  Target-specific propensity columns are appended to that target's
    feature matrix and selected only by the propensity model.  Any unsupported,
    non-converged, optimizer/numeric-warning-emitting, or non-finite target
    aborts the entire fit; this function never substitutes another estimator.
    When ``retry_max_its`` is supplied, only a target that reaches the first
    iteration ceiling without satisfying both upstream stopping criteria is
    restarted with that higher ceiling. Successfully fitted targets are kept.
    """

    source_hashes = verify_sarpu_vendor()
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    x = _fit_array2d(X, "X")
    w = _binary_matrix(measured, "measured", len(x)).astype(bool)
    s = _observed_matrix(observed, w)
    design = _fit_propensity_array(propensity_design, w)
    if target_ids is None:
        fitted_target_ids = tuple(str(index) for index in range(s.shape[1]))
        seed_coordinates: tuple[Any, ...] = tuple(range(s.shape[1]))
    else:
        fitted_target_ids = tuple(map(str, target_ids))
        if (
            len(fitted_target_ids) != s.shape[1]
            or len(set(fitted_target_ids)) != len(fitted_target_ids)
        ):
            raise ValueError("target_ids must be unique and align with observed columns")
        seed_coordinates = fitted_target_ids
    selected = _feature_indices(classification_attributes, x.shape[1])
    max_its = _positive_int(max_its, "max_its")
    convergence_window = _positive_int(convergence_window, "convergence_window")
    if convergence_window < 2:
        raise ValueError("convergence_window must be at least two")
    if max_its < convergence_window + 2:
        raise ValueError("max_its must be at least convergence_window + 2")
    if retry_max_its is not None:
        retry_max_its = _positive_int(retry_max_its, "retry_max_its")
        if retry_max_its <= max_its:
            raise ValueError("retry_max_its must exceed max_its")
    slope_eps = _positive_float(slope_eps, "slope_eps")
    ll_eps = _positive_float(ll_eps, "ll_eps")
    if not isinstance(refit_classifier, (bool, np.bool_)):
        raise ValueError("refit_classifier must be boolean")
    refit_classifier = bool(refit_classifier)

    measured_counts = np.sum(w, axis=0).astype(np.int64)
    positive_counts = np.sum((s == 1) & w, axis=0).astype(np.int64)
    unlabeled_counts = measured_counts - positive_counts
    unsupported = np.flatnonzero(
        (measured_counts == 0) | (positive_counts == 0) | (unlabeled_counts == 0)
    )
    if unsupported.size:
        details = {
            int(target): {
                "measured": int(measured_counts[target]),
                "positive": int(positive_counts[target]),
                "unlabeled": int(unlabeled_counts[target]),
            }
            for target in unsupported
        }
        raise SARPUAuthorsSupportError(
            "Authors' SAR-EM requires measured positive and unlabeled support for "
            f"every target; unsupported targets: {json.dumps(details, sort_keys=True)}"
        )

    n_targets = s.shape[1]
    n_classification = len(selected)
    n_propensity = design.shape[2]
    classifier_coefficients = np.empty((n_classification, n_targets), dtype=np.float64)
    classifier_intercepts = np.empty(n_targets, dtype=np.float64)
    propensity_coefficients = np.zeros((n_propensity, n_targets), dtype=np.float64)
    propensity_intercepts = np.zeros(n_targets, dtype=np.float64)
    propensity_constants = np.zeros(n_targets, dtype=np.float64)
    propensity_active_mask = np.zeros(
        (n_propensity, n_targets), dtype=bool
    )
    target_propensity_design_ranks = np.empty(n_targets, dtype=np.int64)
    classifier_seeds = np.empty(n_targets, dtype=np.int64)
    propensity_seeds = np.empty(n_targets, dtype=np.int64)
    target_iterations = np.empty(n_targets, dtype=np.int64)
    target_fit_attempts = np.ones(n_targets, dtype=np.int64)
    target_retried = np.zeros(n_targets, dtype=bool)
    target_max_its = np.full(n_targets, max_its, dtype=np.int64)
    target_objectives = np.empty(n_targets, dtype=np.float64)
    final_loglikelihoods = np.empty(n_targets, dtype=np.float64)
    final_slopes = np.empty(n_targets, dtype=np.float64)
    final_improvements = np.empty(n_targets, dtype=np.float64)
    warning_messages: list[str] = []

    for target in range(n_targets):
        target_started = time.monotonic()
        if on_progress is not None:
            on_progress({
                "event": "target_start",
                "target_index": target + 1,
                "target_total": n_targets,
                "target_id": fitted_target_ids[target],
                "elapsed_seconds": 0.0,
            })
        rows = w[:, target]
        try:
            selected_x = np.asarray(x[rows][:, selected], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"X selected features must be numeric on measured rows for target {target}"
            ) from error
        if not np.all(np.isfinite(selected_x)):
            raise ValueError(
                f"X selected features must be finite on measured rows for target {target}"
            )
        active_propensity = _independent_propensity_columns(
            design[rows, target]
        )
        propensity_active_mask[active_propensity, target] = True
        target_propensity_design_ranks[target] = len(active_propensity) + 1
        target_design = design[rows, target][:, active_propensity]
        target_x = np.concatenate([selected_x, target_design], axis=1)
        target_s = s[rows, target].astype(np.int64, copy=False)
        classifier_attributes = list(range(n_classification))
        propensity_attributes = list(
            range(n_classification, n_classification + len(active_propensity))
        )
        seed_coordinate = seed_coordinates[target]
        classifier_seed = stable_seed(
            seed, "sarpu_authors", seed_coordinate, "classifier"
        )
        propensity_seed = stable_seed(
            seed, "sarpu_authors", seed_coordinate, "propensity"
        )
        classifier_seeds[target] = classifier_seed
        propensity_seeds[target] = propensity_seed
        attempt_limits = (max_its,) + (
            () if retry_max_its is None else (retry_max_its,)
        )
        for attempt_index, attempt_max_its in enumerate(attempt_limits):
            classifier_model = CompatibleLogisticRegressionPU(
                random_state=classifier_seed
            )
            propensity_model = CompatibleLogisticRegressionPU(
                random_state=propensity_seed
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fitted_classifier, fitted_propensity, info = (
                        _UPSTREAM.pu_learn_sar_em(
                            target_x,
                            target_s,
                            propensity_attributes,
                            classification_attributes=classifier_attributes,
                            classification_model=classifier_model,
                            propensity_model=propensity_model,
                            max_its=attempt_max_its,
                            slope_eps=slope_eps,
                            ll_eps=ll_eps,
                            convergence_window=convergence_window,
                            refit_classifier=refit_classifier,
                        )
                    )
                except Exception as error:
                    raise SARPUAuthorsError(
                        f"Authors' SAR-EM failed for target {target}: {error}"
                    ) from error

            target_warning_messages = [
                f"target {target}: {item.category.__name__}: {item.message}"
                for item in caught
            ]
            fatal_warnings = [
                item
                for item in caught
                if issubclass(item.category, (ConvergenceWarning, RuntimeWarning))
            ]
            if fatal_warnings:
                raise SARPUAuthorsConvergenceError(
                    "Authors' SAR-EM emitted a fatal optimizer/numeric warning: "
                    + " | ".join(target_warning_messages)
                )
            warning_messages.extend(target_warning_messages)

            required_info = {
                "nb_iterations",
                "loglikelihoods",
                "propensity_slopes",
                "max_ll_improvements",
            }
            if not isinstance(info, dict) or not required_info.issubset(info):
                raise SARPUAuthorsConvergenceError(
                    "Authors' SAR-EM returned incomplete diagnostics for "
                    f"target {target}"
                )
            iteration_index = int(info["nb_iterations"])
            loglikelihood_history = np.asarray(
                info["loglikelihoods"], dtype=np.float64
            )
            slope_history = np.asarray(
                info["propensity_slopes"], dtype=np.float64
            )
            improvement_history = np.asarray(
                info["max_ll_improvements"], dtype=np.float64
            )
            if (
                loglikelihood_history.ndim != 1
                or slope_history.ndim != 1
                or improvement_history.ndim != 1
                or loglikelihood_history.size == 0
                or slope_history.size == 0
                or improvement_history.size == 0
                or not np.all(np.isfinite(loglikelihood_history))
                or not np.all(np.isfinite(slope_history))
                or not np.all(np.isfinite(improvement_history))
            ):
                raise SARPUAuthorsConvergenceError(
                    "Authors' SAR-EM returned non-finite/incomplete histories "
                    f"for target {target}"
                )
            final_slope = float(slope_history[-1])
            final_improvement = float(improvement_history[-1])
            converged = (
                0 <= iteration_index < attempt_max_its - 1
                and final_slope < slope_eps
                and final_improvement < ll_eps
            )
            if converged:
                target_fit_attempts[target] = attempt_index + 1
                target_retried[target] = attempt_index > 0
                target_max_its[target] = attempt_max_its
                break
            if attempt_index + 1 < len(attempt_limits):
                if on_progress is not None:
                    on_progress({
                        "event": "target_retry",
                        "target_index": target + 1,
                        "target_total": n_targets,
                        "target_id": fitted_target_ids[target],
                        "attempt": attempt_index + 2,
                        "previous_max_its": attempt_max_its,
                        "max_its": attempt_limits[attempt_index + 1],
                        "previous_iteration_index": iteration_index,
                        "previous_slope": final_slope,
                        "previous_ll_improvement": final_improvement,
                        "elapsed_seconds": time.monotonic() - target_started,
                    })
                continue
            raise SARPUAuthorsConvergenceError(
                "Authors' SAR-EM did not satisfy both upstream stopping criteria "
                f"for target {target}: iteration_index={iteration_index}, "
                f"slope={final_slope:.6g}, ll_improvement={final_improvement:.6g}"
            )

        classifier_inner = fitted_classifier.model
        classifier_coef = np.asarray(classifier_inner.coef_, dtype=np.float64)
        classifier_intercept = np.asarray(classifier_inner.intercept_, dtype=np.float64)
        if classifier_coef.shape != (1, n_classification) or classifier_intercept.shape != (1,):
            raise SARPUAuthorsError(
                f"Unexpected authors' classifier shape for target {target}"
            )
        classifier_coefficients[:, target] = classifier_coef[0]
        classifier_intercepts[target] = classifier_intercept[0]

        propensity_inner = fitted_propensity.model
        if len(active_propensity) == 0:
            if not isinstance(propensity_inner, _UPSTREAM.NoFeaturesModel):
                raise SARPUAuthorsError("Expected authors' constant propensity model")
            propensity_constants[target] = float(propensity_inner.prior)
        else:
            propensity_coef = np.asarray(propensity_inner.coef_, dtype=np.float64)
            propensity_intercept = np.asarray(propensity_inner.intercept_, dtype=np.float64)
            if (
                propensity_coef.shape != (1, len(active_propensity))
                or propensity_intercept.shape != (1,)
            ):
                raise SARPUAuthorsError(
                    f"Unexpected authors' propensity shape for target {target}"
                )
            propensity_coefficients[active_propensity, target] = propensity_coef[0]
            propensity_intercepts[target] = propensity_intercept[0]

        p = _validated_probability(
            fitted_classifier.predict_proba(target_x), (len(target_s),), "training p"
        )
        e = _validated_probability(
            fitted_propensity.predict_proba(target_x), (len(target_s),), "training e"
        )
        q = _validated_probability(p * e, (len(target_s),), "training q")
        if np.any((q <= 0) | (q >= 1)):
            raise FloatingPointError(
                f"Authors' SAR-EM produced boundary q probabilities for target {target}"
            )
        objective = -float(_UPSTREAM.loglikelihood_probs(p, e, target_s))
        if not np.isfinite(objective):
            raise FloatingPointError(
                f"Authors' SAR-EM produced a non-finite objective for target {target}"
            )
        target_iterations[target] = iteration_index + 1
        target_objectives[target] = objective
        final_loglikelihoods[target] = float(loglikelihood_history[-1])
        final_slopes[target] = final_slope
        final_improvements[target] = final_improvement
        if on_progress is not None:
            on_progress({
                "event": "target_complete",
                "target_index": target + 1,
                "target_total": n_targets,
                "target_id": fitted_target_ids[target],
                "iterations": int(target_iterations[target]),
                "fit_attempts": int(target_fit_attempts[target]),
                "retried": bool(target_retried[target]),
                "max_its": int(target_max_its[target]),
                "objective_value": float(objective),
                "elapsed_seconds": time.monotonic() - target_started,
                "cache_status": "fitted",
            })

    for name, value in {
        "classifier coefficients": classifier_coefficients,
        "classifier intercepts": classifier_intercepts,
        "propensity coefficients": propensity_coefficients,
        "propensity intercepts": propensity_intercepts,
        "propensity constants": propensity_constants,
        "target objectives": target_objectives,
    }.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"SAR-PU produced non-finite {name}")

    return SARPUAuthorsFit(
        classifier_coefficients=classifier_coefficients,
        classifier_intercepts=classifier_intercepts,
        propensity_coefficients=propensity_coefficients,
        propensity_intercepts=propensity_intercepts,
        propensity_constants=propensity_constants,
        propensity_active_mask=propensity_active_mask,
        target_propensity_design_ranks=target_propensity_design_ranks,
        classification_attributes=selected.copy(),
        classifier_seeds=classifier_seeds,
        propensity_seeds=propensity_seeds,
        target_measured_counts=measured_counts,
        target_positive_counts=positive_counts,
        target_unlabeled_counts=unlabeled_counts,
        target_iterations=target_iterations,
        target_fit_attempts=target_fit_attempts,
        target_retried=target_retried,
        target_max_its=target_max_its,
        target_objective_values=target_objectives,
        final_upstream_loglikelihoods=final_loglikelihoods,
        final_propensity_slopes=final_slopes,
        final_ll_improvements=final_improvements,
        n_features=x.shape[1],
        n_targets=n_targets,
        n_propensity_features=n_propensity,
        target_ids=fitted_target_ids,
        seed=seed,
        max_its=max_its,
        retry_max_its=retry_max_its,
        slope_eps=slope_eps,
        ll_eps=ll_eps,
        convergence_window=convergence_window,
        refit_classifier=refit_classifier,
        converged=True,
        iterations=int(np.max(target_iterations)),
        objective_value=float(np.mean(target_objectives)),
        warning_messages=tuple(warning_messages),
        sklearn_version=str(sklearn.__version__),
        upstream_commit=_UPSTREAM_COMMIT,
        vendored_pu_learning_sha256=source_hashes["pu_learning.py"],
        vendored_pumodels_sha256=source_hashes["PUmodels.py"],
        wrapper_source_sha256=str(SARPU_AUTHORS_PROVENANCE["wrapper_source_sha256"]),
    )


__all__ = [
    "CompatibleLogisticRegressionPU",
    "SARPU_AUTHORS_PROVENANCE",
    "SARPUAuthorsConvergenceError",
    "SARPUAuthorsError",
    "SARPUAuthorsFit",
    "SARPUAuthorsSourceError",
    "SARPUAuthorsSupportError",
    "fit_sarpu_authors",
    "verify_sarpu_vendor",
]
