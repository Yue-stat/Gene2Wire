"""Python 3 measurement-mask port of the pinned authors' GenEML source.

The statistical updates follow src/model.py from nirbhayjm/GenEML at
GENEML_UPSTREAM_COMMIT. The upstream program is Python 2.7 and has no
measurement-mask input. This port makes two scoped changes: Python 3/current
NumPy compatibility, and exclusion of entries whose measurement mask is
false. Exposure remains one scalar mu[j] per target; it is not replaced by a
contextual propensity model.

The pristine source snapshot and exact provenance are stored in
gene2wire._vendor.geneml. A fit fails closed if that snapshot changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


Array = np.ndarray
EPS = 1e-8
GENEML_UPSTREAM_REPOSITORY = "https://github.com/nirbhayjm/GenEML"
GENEML_UPSTREAM_COMMIT = "f6c08c8f3c69a009b565955231509633eda611d1"
GENEML_PORT_VERSION = "1"

# SHA-256 hashes are over the packaged bytes. ops.py2 differs from upstream
# only by a final LF, as recorded in _vendor/geneml/UPSTREAM.md.
_VENDOR_SHA256 = {
    "LICENSE": "3ed527acda1b355b24dfc0a2e1fcdb5809cd37ac37f979e4c988b5194930b799",
    "README.upstream.md": "e24a880e1900bd5e764806d8458b26a5cad45807f0001b1fed2a5ea46b0c76ca",
    "inputs.py2": "3b780499cfa5ccdf45118c8de67d9c505775de6359743a08cb499dd771852f69",
    "main.py2": "d4d6920af47d3490190080a717faeaeb57b54e4a6dfe0d9b5809557544bb12cd",
    "model.py2": "608d6fda9879ddee9a86ee32c51bfb47cf1c35b2b2059374feee30417ff72dbb",
    "ops.py2": "3725993d19ccefaf10fa10010ee7a8e95ac8ae82740aa0bed52c9c0ea9fc6183",
}
GENEML_PORT_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GENEML_AUTHORS_PROVENANCE = {
    "model_id": "GenEML-authors-mask",
    # ``authors_code`` is reserved for an unmodified upstream runtime.  This
    # implementation is derived from the authors' equations/source, but the
    # Python-2 program itself is not executed.
    "authors_code": False,
    "authors_source": True,
    "executes_unmodified_authors_code": False,
    "adapted": True,
    "source_repository": GENEML_UPSTREAM_REPOSITORY,
    "source_commit": GENEML_UPSTREAM_COMMIT,
    "license": "MIT",
    "implementation_variant": "authors-source-python3-measurement-mask-port",
    "port_version": GENEML_PORT_VERSION,
    "port_source_sha256": GENEML_PORT_SOURCE_SHA256,
    "vendor_sha256": dict(_VENDOR_SHA256),
    "adaptations": [
        "Python 3/current NumPy compatibility",
        "known measurement-mask exclusion",
        "dense-array input adapter",
        "full-batch execution",
    ],
    "training_inputs": (
        "cell features X + observed labels S + known measurement mask W"
    ),
}


def verify_geneml_vendor() -> dict[str, str]:
    """Return freshly verified hashes for the pinned GenEML source snapshot."""

    root = Path(__file__).resolve().parents[1] / "_vendor" / "geneml"
    actual_hashes: dict[str, str] = {}
    for name, expected in _VENDOR_SHA256.items():
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"pinned GenEML source is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_hashes[name] = actual
        if actual != expected:
            raise RuntimeError(
                f"pinned GenEML source hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
    return actual_hashes


def _dense_2d(value: Any, name: str) -> Array:
    array = value.toarray() if sparse.issparse(value) else np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _sigmoid(value: Array) -> Array:
    # Preserve the authors' [-20, 20] clipping.
    clipped = np.clip(value, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize_rows(value: Array) -> Array:
    # Equivalent to the authors' default sklearn row-L2 normalization for a
    # finite dense matrix. Zero rows are unchanged.
    norms = np.sqrt(np.sum(value * value, axis=1, keepdims=True))
    return np.divide(value, norms, out=np.zeros_like(value), where=norms != 0)


def _posterior_exposure(
    latent: Array,
    target_factors: Array,
    target_exposure: Array,
    observed: Array,
    measured: Array,
) -> tuple[Array, Array]:
    psi = latent @ target_factors.T
    expected_omega = 0.5 * np.tanh(0.5 * psi) / (EPS + psi)
    negative_probability = _sigmoid(-psi)
    numerator = target_exposure * negative_probability
    posterior = numerator / (EPS + numerator + (1.0 - target_exposure))
    posterior[observed == 1.0] = 1.0
    posterior[~measured] = 0.0
    expected_omega[~measured] = 0.0
    return posterior, expected_omega


def _observed_log_loss(
    observed: Array, probability: Array, measured: Array
) -> float:
    clipped = np.clip(probability[measured], 1e-12, 1.0 - 1e-12)
    labels = observed[measured]
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
    )


@dataclass(frozen=True)
class GenEMLAuthorsMaskFit:
    """Serializable fitted state for the authors-source GenEML mask port."""

    feature_map: Array
    target_factors: Array
    target_exposure: Array
    rank: int
    n_features: int
    n_targets: int
    normalize_features: bool
    lam_u: float
    lam_v: float
    lam_w: float
    batch_size: int
    num_epochs: int
    converged: bool
    iterations: int
    objective_value: float
    implementation_provenance: str
    upstream_commit: str
    port_version: str
    port_source_sha256: str

    @property
    def model_name(self) -> str:
        # Keep distinct until a Python-2 golden parity fixture is available.
        return "GenEML-authors-mask"

    def _features(self, X: Any) -> Array:
        x = _dense_2d(X, "X")
        if x.shape[1] != self.n_features:
            raise ValueError("prediction X does not match the fitted feature count")
        return _normalize_rows(x) if self.normalize_features else x

    def predict_reference(self, X: Any) -> Array:
        x = self._features(X)
        latent = x @ self.feature_map.T
        return _sigmoid(latent @ self.target_factors.T)

    def predict_exposure(self, X: Any | None = None) -> Array:
        if X is None:
            return np.asarray(self.target_exposure, dtype=np.float64).copy()
        x = self._features(X)
        return np.broadcast_to(
            self.target_exposure, (len(x), self.n_targets)
        ).copy()

    def predict_observed(self, X: Any) -> Array:
        return self.predict_reference(X) * self.predict_exposure(X)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "selected_structure": "authors_geneml_inductive_lowrank",
            "rank": self.rank,
            "lam_u": self.lam_u,
            "lam_v": self.lam_v,
            "lam_w": self.lam_w,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "final_converged": self.converged,
            "final_iterations": self.iterations,
            "final_objective": self.objective_value,
            "termination_reason": "completed_fixed_epoch_schedule",
            "convergence_definition": "all scheduled author updates completed finitely",
            "probability_semantics": "reference_p_with_model_exposure",
            "exposure_model": "one_global_probability_per_target",
            "implementation_provenance": (
                self.implementation_provenance
            ),
            "authors_code": False,
            "authors_source": True,
            "executes_unmodified_authors_code": False,
            "adapted": True,
            "source_repository": GENEML_UPSTREAM_REPOSITORY,
            "source_commit": self.upstream_commit,
            "license": "MIT",
            "implementation_variant": self.implementation_provenance,
            "upstream_repository": GENEML_UPSTREAM_REPOSITORY,
            "upstream_commit": self.upstream_commit,
            "port_version": self.port_version,
            "port_source_sha256": self.port_source_sha256,
            "vendor_sha256": dict(_VENDOR_SHA256),
            "adaptations": (
                "Python3 compatibility|known-W exclusion|dense-array adapter|"
                "full-batch execution"
            ),
            "training_inputs": (
                "cell features X + observed labels S + known measurement mask W"
            ),
        }


def _fit_geneml_port(
    X: Any,
    observed: Any,
    measured: Any | None,
    *,
    rank: int,
    batch_size: int | None,
    num_epochs: int,
    lam_u: float,
    lam_v: float,
    lam_w: float,
    pg_iters: int,
    lr_alpha: float,
    lr_tau: float,
    init_mu_a: float,
    init_mu_b: float,
    init_std: float,
    init_w: float,
    normalize_features: bool,
    seed: int,
) -> GenEMLAuthorsMaskFit:
    verify_geneml_vendor()
    x = _dense_2d(X, "X")
    labels_raw = (
        observed.toarray() if sparse.issparse(observed) else np.asarray(observed)
    )
    if labels_raw.ndim != 2 or labels_raw.shape[0] != len(x):
        raise ValueError("observed must be a two-dimensional row-aligned array")
    if measured is None:
        mask = np.ones(labels_raw.shape, dtype=bool)
    else:
        mask_raw = (
            measured.toarray()
            if sparse.issparse(measured)
            else np.asarray(measured)
        )
        if np.asarray(mask_raw).shape != labels_raw.shape:
            raise ValueError("measured must have the same shape as observed")
        try:
            numeric_mask = np.asarray(mask_raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("measured must be finite and binary") from exc
        if (not np.all(np.isfinite(numeric_mask))
                or not np.all((numeric_mask == 0.0) | (numeric_mask == 1.0))):
            raise ValueError("measured must be finite and binary")
        mask = numeric_mask.astype(bool)
    measured_labels = np.asarray(labels_raw[mask], dtype=np.float64)
    if not np.all(np.isfinite(measured_labels)) or not np.all(
        (measured_labels == 0.0) | (measured_labels == 1.0)
    ):
        raise ValueError(
            "observed labels on measured entries must be binary and finite"
        )
    # Payload outside W is ignored by construction.
    labels = np.zeros(mask.shape, dtype=np.float64)
    labels[mask] = measured_labels
    if labels.shape[1] == 0:
        raise ValueError("observed must contain at least one target")
    measured_per_target = np.sum(mask, axis=0)
    if np.any(measured_per_target == 0):
        targets = np.flatnonzero(measured_per_target == 0).tolist()
        raise ValueError(
            f"GenEML cannot identify targets with no measured entries: {targets}"
        )

    rank = _positive_int(rank, "rank")
    num_epochs = _positive_int(num_epochs, "num_epochs")
    pg_iters = _positive_int(pg_iters, "pg_iters")
    lam_u = _positive_float(lam_u, "lam_u")
    lam_v = _positive_float(lam_v, "lam_v")
    lam_w = _positive_float(lam_w, "lam_w")
    lr_alpha = _positive_float(lr_alpha, "lr_alpha")
    lr_tau = _positive_float(lr_tau, "lr_tau")
    init_mu_a = _positive_float(init_mu_a, "init_mu_a")
    init_mu_b = _positive_float(init_mu_b, "init_mu_b")
    init_std = _positive_float(init_std, "init_std")
    init_w = _positive_float(init_w, "init_w")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)

    n_rows, n_targets = labels.shape
    if batch_size is None:
        # This preserves the authors' complete-batch loop without silently
        # dropping a final partial batch.
        batch_size = n_rows
    batch_size = _positive_int(batch_size, "batch_size")
    if batch_size > n_rows or n_rows % batch_size:
        raise ValueError(
            "batch_size must divide the training row count; the authors' loop "
            "silently drops a partial final batch, which this port refuses to do"
        )
    if not isinstance(normalize_features, (bool, np.bool_)):
        raise ValueError("normalize_features must be boolean")
    normalize_features = bool(normalize_features)
    if normalize_features:
        x = _normalize_rows(x)

    n_features = x.shape[1]
    rng = np.random.RandomState(seed)
    # Preserve the upstream draw order and float32 initialization.
    target_factors = init_std * rng.randn(n_targets, rank).astype(np.float32)
    feature_map = init_w * rng.randn(rank, n_features).astype(np.float32)
    target_exposure = rng.beta(init_mu_a, init_mu_b, size=n_targets)
    sigma_v = [
        lam_v * np.eye(rank, dtype=np.float64) for _ in range(n_targets)
    ]
    x_v = np.zeros((n_targets, rank), dtype=np.float64)
    sigma_map = lam_w * np.eye(n_features, dtype=np.float64)
    x_map = np.zeros((n_features, rank), dtype=np.float64)
    identity_rank = np.eye(rank, dtype=np.float64)
    identity_features = np.eye(n_features, dtype=np.float64)

    minibatches = n_rows // batch_size
    learning_rates = lr_alpha * (
        1.0 + np.arange(minibatches * num_epochs, dtype=np.float64)
    ) ** (-lr_tau)
    iteration = 0
    # The upstream program allocates one fixed U_batch buffer and reuses it
    # across minibatches and epochs; its contents initialize the next E step.
    latent_batch = np.zeros((batch_size, rank), dtype=np.float32)
    try:
        for _epoch in range(num_epochs):
            for batch_index in range(minibatches):
                lo = batch_index * batch_size
                hi = lo + batch_size
                xb = x[lo:hi]
                sb = labels[lo:hi]
                wb = mask[lo:hi]
                gamma = learning_rates[iteration]
                latent = latent_batch

                # update_U from authors' model.py, with W-mask factors.
                for _ in range(pg_iters):
                    posterior, omega = _posterior_exposure(
                        latent, target_factors, target_exposure, sb, wb
                    )
                    kappa = sb - 0.5
                    posterior_omega = posterior * omega
                    posterior_kappa = posterior * kappa * wb
                    for row in range(batch_size):
                        row_weights = posterior_omega[row][:, None]
                        precision = (
                            target_factors.T
                            @ (row_weights * target_factors)
                            + lam_u * identity_rank
                        )
                        rhs = target_factors.T @ posterior_kappa[row]
                        rhs += (lam_u * feature_map) @ xb[row]
                        latent[row] = np.linalg.solve(precision, rhs)

                # update_V from the authors' stochastic sufficient statistics.
                posterior, omega = _posterior_exposure(
                    latent, target_factors, target_exposure, sb, wb
                )
                kappa = sb - 0.5
                for target in range(n_targets):
                    weights = posterior[:, target] * omega[:, target]
                    weighted_kappa = (
                        posterior[:, target]
                        * kappa[:, target]
                        * wb[:, target]
                    )
                    sigma = latent.T @ (weights[:, None] * latent)
                    rhs = latent.T @ weighted_kappa
                    sigma_v[target] = (
                        (1.0 - gamma) * sigma_v[target] + gamma * sigma
                    )
                    x_v[target] = (
                        (1.0 - gamma) * x_v[target] + gamma * rhs
                    )
                    target_factors[target] = np.linalg.solve(
                        sigma_v[target], x_v[target]
                    )

                # update_observance. The sole mask adaptation is using the
                # target-specific measured count and posterior sum.
                posterior, _ = _posterior_exposure(
                    latent, target_factors, target_exposure, sb, wb
                )
                batch_counts = np.sum(wb, axis=0)
                estimable = batch_counts > 0
                candidate_mu = target_exposure.copy()
                candidate_mu[estimable] = (
                    init_mu_a
                    + np.sum(posterior, axis=0)[estimable]
                    - 1.0
                ) / (
                    init_mu_a
                    + init_mu_b
                    + batch_counts[estimable]
                    - 2.0
                )
                target_exposure[estimable] = (
                    (1.0 - gamma) * target_exposure[estimable]
                    + gamma * candidate_mu[estimable]
                )

                # update_W. A wholly unmeasured row cannot enter this statistic.
                active_rows = np.any(wb, axis=1)
                if np.any(active_rows):
                    active_x = xb[active_rows]
                    active_latent = latent[active_rows]
                    sigma = (
                        active_x.T @ active_x
                        + lam_w * identity_features
                    )
                    rhs = active_x.T @ active_latent
                    sigma_map = (
                        (1.0 - gamma) * sigma_map + gamma * sigma
                    )
                    x_map = (1.0 - gamma) * x_map + gamma * rhs
                    # Preserve the authors' explicit inverse in this path.
                    feature_map = np.asarray(
                        np.linalg.inv(sigma_map) @ x_map
                    ).T

                arrays = (
                    latent,
                    target_factors,
                    feature_map,
                    target_exposure,
                )
                if not all(
                    np.all(np.isfinite(value)) for value in arrays
                ):
                    raise FloatingPointError(
                        "GenEML update produced a non-finite value"
                    )
                iteration += 1
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise RuntimeError(
            f"GenEML authors-source port failed at update {iteration + 1}"
        ) from exc

    fitted = GenEMLAuthorsMaskFit(
        feature_map=np.asarray(feature_map, dtype=np.float64),
        target_factors=np.asarray(target_factors, dtype=np.float64),
        target_exposure=np.asarray(target_exposure, dtype=np.float64),
        rank=rank,
        n_features=n_features,
        n_targets=n_targets,
        normalize_features=normalize_features,
        lam_u=lam_u,
        lam_v=lam_v,
        lam_w=lam_w,
        batch_size=batch_size,
        num_epochs=num_epochs,
        converged=True,
        iterations=iteration,
        objective_value=float("nan"),
        implementation_provenance=(
            "authors-source-python3-measurement-mask-port"
        ),
        upstream_commit=GENEML_UPSTREAM_COMMIT,
        port_version=GENEML_PORT_VERSION,
        port_source_sha256=GENEML_PORT_SOURCE_SHA256,
    )
    objective = _observed_log_loss(
        labels, fitted.predict_observed(X), mask
    )
    return replace(fitted, objective_value=objective)


def fit_geneml_authors_mask(
    X: Any,
    observed: Any,
    measured: Any,
    *,
    rank: int = 2,
    batch_size: int | None = None,
    num_epochs: int = 10,
    lam_u: float = 1e-3,
    lam_v: float = 1e-3,
    lam_w: float = 1e-4,
    pg_iters: int = 1,
    lr_alpha: float = 0.6,
    lr_tau: float = 0.55,
    init_mu_a: float = 1.0,
    init_mu_b: float = 1.0,
    init_std: float = 1e-2,
    init_w: float = 1e-2,
    normalize_features: bool = True,
    seed: int = 0,
) -> GenEMLAuthorsMaskFit:
    """Fit the authors-source port while excluding unmeasured entries.

    observed outside measured is deliberately ignored. No reference labels or
    generator propensities are accepted by this API.
    """

    return _fit_geneml_port(
        X,
        observed,
        measured,
        rank=rank,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lam_u=lam_u,
        lam_v=lam_v,
        lam_w=lam_w,
        pg_iters=pg_iters,
        lr_alpha=lr_alpha,
        lr_tau=lr_tau,
        init_mu_a=init_mu_a,
        init_mu_b=init_mu_b,
        init_std=init_std,
        init_w=init_w,
        normalize_features=normalize_features,
        seed=seed,
    )


def fit_geneml_authors_port(
    X: Any,
    observed: Any,
    **kwargs: Any,
) -> GenEMLAuthorsMaskFit:
    """Fit the Python 3 source port with upstream all-measured semantics."""

    labels = (
        observed.toarray() if sparse.issparse(observed) else np.asarray(observed)
    )
    if labels.ndim != 2:
        raise ValueError("observed must be a two-dimensional array")
    defaults = {
        "rank": 2,
        "batch_size": None,
        "num_epochs": 10,
        "lam_u": 1e-3,
        "lam_v": 1e-3,
        "lam_w": 1e-4,
        "pg_iters": 1,
        "lr_alpha": 0.6,
        "lr_tau": 0.55,
        "init_mu_a": 1.0,
        "init_mu_b": 1.0,
        "init_std": 1e-2,
        "init_w": 1e-2,
        "normalize_features": True,
        "seed": 0,
    }
    unknown = sorted(set(kwargs).difference(defaults))
    if unknown:
        raise TypeError(
            "unexpected GenEML port keyword argument(s): "
            + ", ".join(unknown)
        )
    defaults.update(kwargs)
    return _fit_geneml_port(
        X, observed, None, **defaults
    )


__all__ = [
    "GENEML_UPSTREAM_COMMIT",
    "GENEML_UPSTREAM_REPOSITORY",
    "GENEML_AUTHORS_PROVENANCE",
    "GENEML_PORT_VERSION",
    "GENEML_PORT_SOURCE_SHA256",
    "GenEMLAuthorsMaskFit",
    "fit_geneml_authors_mask",
    "fit_geneml_authors_port",
    "verify_geneml_vendor",
]
