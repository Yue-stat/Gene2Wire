"""Canonical direct, low-rank, and shared-plus-residual PU models.

All variants use the same masked Bernoulli likelihood, float64 L-BFGS
optimizer, initialization path, and regularization convention.  This removes
optimizer and penalty-scaling confounds from model comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from .config import FitConfig, ModelConfig
from .data import DatasetBundle


Array = NDArray[np.float64]


def _exposure_matrix(exposure: Any, shape: tuple[int, int]) -> Array:
    e = np.asarray(exposure, dtype=np.float64)
    if e.ndim == 0:
        e = np.full(shape, float(e), dtype=np.float64)
    elif e.ndim == 1:
        matches_cells = e.shape == (shape[0],)
        matches_targets = e.shape == (shape[1],)
        if matches_cells and matches_targets:
            raise ValueError(
                "ambiguous 1-D exposure because n_cells == n_targets; "
                "pass shape (n_cells, 1), (1, n_targets), or (n_cells, n_targets)"
            )
        if matches_cells:
            e = np.broadcast_to(e[:, None], shape).copy()
        elif matches_targets:
            e = np.broadcast_to(e[None, :], shape).copy()
        else:
            raise ValueError(f"1-D exposure length cannot align to {shape}")
    elif e.ndim <= 2:
        try:
            e = np.broadcast_to(e, shape).astype(np.float64, copy=True)
        except ValueError as exc:
            raise ValueError(f"exposure cannot broadcast to {shape}") from exc
    else:
        raise ValueError("exposure must be scalar, 1-D, or 2-D")
    if not np.all(np.isfinite(e)) or np.any(e < 0) or np.any(e > 1):
        raise ValueError("exposure probabilities must be finite and in [0, 1]")
    return e


def _logit(values: Array) -> Array:
    x = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(x) - np.log1p(-x)


@dataclass(frozen=True)
class FittedModel:
    """Fitted parameters and optimization diagnostics."""

    config: ModelConfig
    cell_shared: Array | None
    target_shared: Array | None
    residual: Array | None
    target_coeff: Array | None
    target_features: Array | None
    intercept: Array
    objective: float
    converged: bool
    iterations: int
    message: str

    def latent_logit(self, x_cell: Any) -> Array:
        x = np.asarray(x_cell, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x_cell must be 2-D")
        eta = np.broadcast_to(self.intercept, (x.shape[0], self.intercept.size)).copy()
        if self.cell_shared is not None and self.target_shared is not None:
            if x.shape[1] != self.cell_shared.shape[0]:
                raise ValueError("x_cell feature count differs from fitted model")
            eta += (x @ self.cell_shared) @ self.target_shared.T
        if self.residual is not None:
            if x.shape[1] != self.residual.shape[0]:
                raise ValueError("x_cell feature count differs from fitted model")
            eta += x @ self.residual
        if self.target_coeff is not None and self.target_features is not None:
            if x.shape[1] != self.target_coeff.shape[0]:
                raise ValueError("x_cell feature count differs from fitted target term")
            eta += (x @ self.target_coeff) @ self.target_features.T
        return eta

    def predict_proba(self, x_cell: Any) -> Array:
        """Latent biological probability p(y=1|x)."""

        return expit(self.latent_logit(x_cell))

    def predict_observed(self, x_cell: Any, exposure: Any = 1.0) -> Array:
        """Observed-label probability q=e*p (or q=p for non-PU config)."""

        p = self.predict_proba(x_cell)
        if not self.config.pu:
            return p
        return _exposure_matrix(exposure, p.shape) * p

    def state_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly state mapping."""

        return {
            "config": asdict(self.config),
            "cell_shared": self.cell_shared,
            "target_shared": self.target_shared,
            "residual": self.residual,
            "target_coeff": self.target_coeff,
            "target_features": self.target_features,
            "intercept": self.intercept,
            "objective": self.objective,
            "converged": self.converged,
            "iterations": self.iterations,
            "message": self.message,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FittedModel":
        """Reconstruct a fitted model from a trusted, validated checkpoint."""

        required = {
            "config",
            "cell_shared",
            "target_shared",
            "residual",
            "target_coeff",
            "target_features",
            "intercept",
            "objective",
            "converged",
            "iterations",
            "message",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"fitted state is missing fields: {sorted(missing)}")

        def optional_array(name: str) -> Array | None:
            value = state[name]
            return None if value is None else np.asarray(value, dtype=np.float64)

        return cls(
            config=ModelConfig(**dict(state["config"])),
            cell_shared=optional_array("cell_shared"),
            target_shared=optional_array("target_shared"),
            residual=optional_array("residual"),
            target_coeff=optional_array("target_coeff"),
            target_features=optional_array("target_features"),
            intercept=np.asarray(state["intercept"], dtype=np.float64),
            objective=float(state["objective"]),
            converged=bool(state["converged"]),
            iterations=int(state["iterations"]),
            message=str(state["message"]),
        )


class UnifiedPUModel:
    """Fit one canonical projection model with a shared likelihood.

    The mean data term is

    ``-mean_W[S log(q) + (1-S) log(1-q)]``, where ``q=e*sigmoid(eta)``.

    Penalties use one convention everywhere: ``0.5 * lambda * ||theta||^2``.
    Biases are not penalized.  ``kind`` controls only the predictor structure:

    * direct: ``eta = X C + b``
    * lowrank: ``eta = (X B) A.T + b``
    * joint: ``eta = (X B) A.T + X C + b``

    With ``use_target_features=True``, every structure also adds the fixed-target
    term ``(X D) Y_target.T`` with penalty ``0.5*target_l2*||D||^2``.
    """

    def __init__(self, config: ModelConfig, fit_config: FitConfig | None = None):
        self.config = config
        self.fit_config = fit_config or FitConfig()
        self.fitted_: FittedModel | None = None

    def fit(self, data: DatasetBundle, exposure: Any = 1.0, seed: int = 0) -> FittedModel:
        x = np.asarray(data.X_cell, dtype=np.float64)
        s = np.asarray(data.S_observed, dtype=np.float64)
        w = np.asarray(data.W_measured, dtype=bool)
        e = _exposure_matrix(exposure, s.shape) if self.config.pu else np.ones_like(s)
        if np.any((s == 1) & w & (e <= 0)):
            raise ValueError("a measured positive cannot have zero exposure")

        n_features = x.shape[1]
        n_targets = s.shape[1]
        target_features = None if data.Y_target is None else np.asarray(data.Y_target, dtype=np.float64)
        if self.config.use_target_features and target_features is None:
            raise ValueError("use_target_features=True requires DatasetBundle.Y_target")
        n_target_features = 0 if target_features is None else target_features.shape[1]
        if self.config.rank > min(n_features, n_targets):
            raise ValueError(
                f"rank {self.config.rank} exceeds min(n_features, n_targets)="
                f"{min(n_features, n_targets)}"
            )

        theta0 = self._initialize(x, s, w, e, target_features, seed)

        def fun(theta: Array) -> tuple[float, Array]:
            return self._objective_gradient(theta, x, s, w, e, target_features)

        result = minimize(
            fun,
            theta0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.fit_config.maxiter,
                "ftol": self.fit_config.tolerance,
                "gtol": self.fit_config.tolerance,
                "maxls": 30,
            },
        )
        if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
            raise FloatingPointError("optimizer produced non-finite parameters")
        b_shared, a_shared, residual, target_coeff, intercept = self._unpack(
            result.x, n_features, n_targets, n_target_features
        )
        fitted = FittedModel(
            config=self.config,
            cell_shared=b_shared,
            target_shared=a_shared,
            residual=residual,
            target_coeff=target_coeff,
            target_features=target_features if self.config.use_target_features else None,
            intercept=intercept,
            objective=float(result.fun),
            converged=bool(result.success),
            iterations=int(result.nit),
            message=str(result.message),
        )
        self.fitted_ = fitted
        return fitted

    def _initialize(
        self,
        x: Array,
        s: Array,
        w: NDArray[np.bool_],
        e: Array,
        target_features: Array | None,
        seed: int,
    ) -> Array:
        n_features = x.shape[1]
        n_targets = s.shape[1]
        rng = np.random.default_rng(seed)

        effective = np.sum(e * w, axis=0)
        prevalence = np.divide(
            np.sum(s * w, axis=0),
            effective,
            out=np.full(n_targets, 0.1, dtype=np.float64),
            where=effective > 0,
        )
        intercept = _logit(np.clip(prevalence, 0.01, 0.99))

        if self.config.kind == "direct":
            residual = np.zeros((n_features, n_targets), dtype=np.float64)
            target_coeff = (
                np.zeros((n_features, target_features.shape[1]), dtype=np.float64)
                if self.config.use_target_features and target_features is not None
                else None
            )
            return self._pack(None, None, residual, target_coeff, intercept)

        rank = self.config.rank
        if self.fit_config.initialization == "random":
            cell_shared = rng.normal(0.0, 0.02, size=(n_features, rank))
            target_shared = rng.normal(0.0, 0.02, size=(n_targets, rank))
            residual = (
                np.zeros((n_features, n_targets), dtype=np.float64)
                if self.config.kind == "joint"
                else None
            )
            target_coeff = (
                np.zeros((n_features, target_features.shape[1]), dtype=np.float64)
                if self.config.use_target_features and target_features is not None
                else None
            )
            return self._pack(cell_shared, target_shared, residual, target_coeff, intercept)

        direct_l2 = self.config.residual_l2 if self.config.kind == "joint" else self.config.shared_l2
        direct_config = ModelConfig(
            name=f"{self.config.name}__initializer",
            kind="direct",
            residual_l2=direct_l2,
            use_target_features=self.config.use_target_features,
            target_l2=self.config.target_l2,
            pu=self.config.pu,
        )
        direct_fit = UnifiedPUModel(
            direct_config,
            FitConfig(
                maxiter=min(self.fit_config.maxiter, self.fit_config.init_direct_maxiter),
                tolerance=max(self.fit_config.tolerance, 1e-7),
                initialization="random",
                init_direct_maxiter=self.fit_config.init_direct_maxiter,
            ),
        ).fit(
            DatasetBundle(x, s, w, Y_target=target_features),
            exposure=e,
            seed=seed,
        )
        direct_coef = direct_fit.residual
        assert direct_coef is not None
        u, singular, vt = np.linalg.svd(direct_coef, full_matrices=False)
        root = np.sqrt(np.maximum(singular[:rank], 0.0))
        cell_shared = u[:, :rank] * root[None, :]
        target_shared = vt[:rank, :].T * root[None, :]
        residual = None
        if self.config.kind == "joint":
            residual = direct_coef - cell_shared @ target_shared.T
        return self._pack(
            cell_shared,
            target_shared,
            residual,
            direct_fit.target_coeff,
            direct_fit.intercept,
        )

    def _pack(
        self,
        cell_shared: Array | None,
        target_shared: Array | None,
        residual: Array | None,
        target_coeff: Array | None,
        intercept: Array,
    ) -> Array:
        parts: list[Array] = []
        if self.config.kind in {"lowrank", "joint"}:
            assert cell_shared is not None and target_shared is not None
            parts.extend([cell_shared.ravel(), target_shared.ravel()])
        if self.config.kind in {"direct", "joint"}:
            assert residual is not None
            parts.append(residual.ravel())
        if self.config.use_target_features:
            assert target_coeff is not None
            parts.append(target_coeff.ravel())
        parts.append(intercept.ravel())
        return np.concatenate(parts).astype(np.float64, copy=False)

    def _unpack(
        self,
        theta: Array,
        n_features: int,
        n_targets: int,
        n_target_features: int,
    ) -> tuple[Array | None, Array | None, Array | None, Array | None, Array]:
        cursor = 0
        cell_shared = None
        target_shared = None
        residual = None
        target_coeff = None
        if self.config.kind in {"lowrank", "joint"}:
            count = n_features * self.config.rank
            cell_shared = theta[cursor : cursor + count].reshape(n_features, self.config.rank)
            cursor += count
            count = n_targets * self.config.rank
            target_shared = theta[cursor : cursor + count].reshape(n_targets, self.config.rank)
            cursor += count
        if self.config.kind in {"direct", "joint"}:
            count = n_features * n_targets
            residual = theta[cursor : cursor + count].reshape(n_features, n_targets)
            cursor += count
        if self.config.use_target_features:
            if n_target_features < 1:
                raise ValueError("target feature count must be positive")
            count = n_features * n_target_features
            target_coeff = theta[cursor : cursor + count].reshape(n_features, n_target_features)
            cursor += count
        intercept = theta[cursor : cursor + n_targets]
        cursor += n_targets
        if cursor != theta.size:
            raise RuntimeError("internal parameter layout mismatch")
        return cell_shared, target_shared, residual, target_coeff, intercept

    def _objective_gradient(
        self,
        theta: Array,
        x: Array,
        s: Array,
        w: NDArray[np.bool_],
        e: Array,
        target_features: Array | None,
    ) -> tuple[float, Array]:
        n_features = x.shape[1]
        n_targets = s.shape[1]
        n_target_features = 0 if target_features is None else target_features.shape[1]
        cell_shared, target_shared, residual, target_coeff, intercept = self._unpack(
            theta, n_features, n_targets, n_target_features
        )
        eta = np.broadcast_to(intercept, s.shape).copy()
        hidden = None
        if cell_shared is not None and target_shared is not None:
            hidden = x @ cell_shared
            eta += hidden @ target_shared.T
        if residual is not None:
            eta += x @ residual
        if target_coeff is not None and target_features is not None:
            eta += (x @ target_coeff) @ target_features.T

        p = expit(eta)
        n_observed = int(np.sum(w))

        # Exact, numerically stable BCE for q=e*sigmoid(eta), with no clipping.
        # For S=1: -log(e) + softplus(-eta).
        # For S=0: softplus(eta) - softplus(eta + log(1-e)).
        # This keeps the reported objective and analytic gradient consistent even
        # for extreme logits, which matter for rare targets.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_exposure = np.log(e)
            log_one_minus_exposure = np.log1p(-e)
        positive_loss = -log_exposure + np.logaddexp(0.0, -eta)
        negative_loss = np.logaddexp(0.0, eta) - np.logaddexp(
            0.0, eta + log_one_minus_exposure
        )
        entry_loss = np.where(s == 1, positive_loss, negative_loss)
        loss = float(np.sum(entry_loss[w])) / n_observed

        # d/deta of the two exact expressions above.
        negative_grad = p - expit(eta + log_one_minus_exposure)
        grad_eta = np.where(s == 1, p - 1, negative_grad)
        grad_eta *= w / n_observed

        gradients: list[Array] = []
        if cell_shared is not None and target_shared is not None:
            loss += 0.5 * self.config.shared_l2 * (
                float(np.sum(cell_shared**2)) + float(np.sum(target_shared**2))
            )
            grad_cell = x.T @ (grad_eta @ target_shared) + self.config.shared_l2 * cell_shared
            assert hidden is not None
            grad_target = grad_eta.T @ hidden + self.config.shared_l2 * target_shared
            gradients.extend([grad_cell.ravel(), grad_target.ravel()])
        if residual is not None:
            loss += 0.5 * self.config.residual_l2 * float(np.sum(residual**2))
            grad_residual = x.T @ grad_eta + self.config.residual_l2 * residual
            gradients.append(grad_residual.ravel())
        if target_coeff is not None and target_features is not None:
            loss += 0.5 * self.config.target_l2 * float(np.sum(target_coeff**2))
            grad_target_coeff = (
                x.T @ (grad_eta @ target_features) + self.config.target_l2 * target_coeff
            )
            gradients.append(grad_target_coeff.ravel())
        gradients.append(np.sum(grad_eta, axis=0).ravel())
        return loss, np.concatenate(gradients)
