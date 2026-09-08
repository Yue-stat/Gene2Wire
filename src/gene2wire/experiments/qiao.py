"""Two-end-covariate bilinear comparators migrated from the simulation notebook.

``squared_error`` retains the notebook's linear bilinear response model;
``logit`` is its Bernoulli adaptation, with free, unpenalized target intercepts.
Both require explicit cell and target covariates. Neither is a PU learner.

All empirical losses are MEANS over measured entries. The legacy squared-error
notebook used SUM squared error: its penalty ``lambda_old`` corresponds to
``lambda_old / n_measured`` here. Identical numeric penalty grids across old and
new implementations therefore are not equivalent. These adapters reproduce the
notebook model families, not every preprocessing/optimization detail of the
published Qiao experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..seeds import stable_seed


_EPS = 1e-7
_OBJECTIVES = ("logit", "squared_error")


def _array2d(value: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return a


def _outcomes(D: np.ndarray, W: np.ndarray | None, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(D, dtype=float)
    if d.shape != shape:
        raise ValueError("Outcomes must align with cells and target descriptors")
    if W is None:
        w = np.ones(shape, dtype=bool)
    else:
        raw_w = np.asarray(W)
        if raw_w.shape != shape or not np.all(np.isin(raw_w, [0, 1])):
            raise ValueError("Measured mask must be aligned and binary")
        w = raw_w.astype(bool)
    if not w.any() or not np.all(np.isin(d[w], [0, 1])):
        raise ValueError("At least one measured entry and binary measured outcomes are required")
    return np.where(w, d, 0.0), w


def _augment(x: np.ndarray) -> np.ndarray:
    return np.column_stack((x, np.ones(len(x))))


def _unpack(theta: np.ndarray, p: int, q: int, rank: int,
            n_targets: int, objective: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_end, b_end = p * rank, (p + q) * rank
    a = theta[:a_end].reshape(p, rank)
    b = theta[a_end:b_end].reshape(q, rank)
    intercept = theta[b_end:] if objective == "logit" else np.zeros(n_targets)
    return a, b, intercept


def _objective_gradient(theta: np.ndarray, xb: np.ndarray, yb: np.ndarray,
                        d: np.ndarray, w: np.ndarray, rank: int,
                        l2: float, objective: str) -> tuple[float, np.ndarray]:
    a, b, intercept = _unpack(theta, xb.shape[1], yb.shape[1], rank, len(yb), objective)
    u, v = xb @ a, yb @ b
    eta = u @ v.T + intercept
    count = int(np.count_nonzero(w))
    if objective == "logit":
        loss = float(np.sum(np.where(w, np.logaddexp(0, eta) - d * eta, 0)) / count)
        residual = np.where(w, expit(eta) - d, 0.0) / count
    else:
        residual = np.where(w, eta - d, 0.0)
        loss = float(np.sum(residual**2) / count)
        residual *= 2.0 / count
    loss += 0.5 * l2 * float(np.sum(a**2) + np.sum(b**2))
    ga = xb.T @ residual @ v + l2 * a
    gb = yb.T @ residual.T @ u + l2 * b
    parts = [ga.ravel(), gb.ravel()]
    if objective == "logit":
        parts.append(residual.sum(axis=0))
    return loss, np.concatenate(parts)


def _direct_warm_start(xb: np.ndarray, d: np.ndarray, w: np.ndarray,
                       maxiter: int) -> np.ndarray:
    p, t = xb.shape[1], d.shape[1]
    count = int(w.sum())
    l2 = 3e-3

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta.reshape(p, t)
        eta = xb @ beta
        loss = float(np.sum(np.where(w, np.logaddexp(0, eta) - d * eta, 0)) / count)
        loss += 0.5 * l2 * float(np.sum(beta[:-1]**2))
        gradient = xb.T @ np.where(w, expit(eta) - d, 0.0) / count
        gradient[:-1] += l2 * beta[:-1]
        return loss, gradient.ravel()

    fitted = minimize(objective, np.zeros(p * t), method="L-BFGS-B", jac=True,
                      options={"maxiter": maxiter, "ftol": 1e-10})
    if not np.all(np.isfinite(fitted.x)):
        raise FloatingPointError("Nonfinite Qiao direct warm start")
    return fitted.x.reshape(p, t)


@dataclass(frozen=True)
class QiaoFit:
    cell_loading: np.ndarray
    target_loading: np.ndarray
    target_intercept: np.ndarray
    target_features: np.ndarray
    objective: str
    rank: int
    l2: float
    converged: bool
    iterations: int
    objective_value: float
    optimizer_message: str
    n_measured: int

    @property
    def selected_structure(self) -> str:
        return "target_covariate_bilinear"

    @property
    def probability_semantics(self) -> str:
        return "observed_detection_probability" if self.objective == "logit" else "clipped_bilinear_score"

    def predict_score(self, X: np.ndarray, Y_target: np.ndarray | None = None) -> np.ndarray:
        x = _array2d(X, "X")
        if x.shape[1] + 1 != self.cell_loading.shape[0]:
            raise ValueError("Prediction X does not match the fitted cell feature dimension")
        y = self.target_features if Y_target is None else _array2d(Y_target, "Y_target")
        if y.shape[1] + 1 != self.target_loading.shape[0]:
            raise ValueError("Prediction target features do not match fitted dimensions")
        if self.objective == "logit" and (y.shape != self.target_features.shape or
                                            not np.array_equal(y, self.target_features)):
            raise ValueError("Logit adaptation has free intercepts for the fitted target panel only")
        score = (_augment(x) @ self.cell_loading) @ (_augment(y) @ self.target_loading).T
        if self.objective == "logit":
            score = score + self.target_intercept
        return score

    def predict_proba(self, X: np.ndarray, Y_target: np.ndarray | None = None) -> np.ndarray:
        score = self.predict_score(X, Y_target)
        if self.objective == "logit":
            return expit(score)
        return np.clip(score, _EPS, 1 - _EPS)


def fit_qiao(
    X: np.ndarray,
    Y_target: np.ndarray,
    D: np.ndarray,
    W: np.ndarray | None = None,
    *,
    rank: int = 2,
    l2: float = 3e-3,
    objective: str = "logit",
    seed: int = 0,
    maxiter: int = 1000,
    tolerance: float = 1e-8,
    init_direct_maxiter: int = 120,
    initial_fit: QiaoFit | None = None,
) -> QiaoFit:
    """Fit a masked bilinear observed-label model using explicit target features."""
    x, y = _array2d(X, "X"), _array2d(Y_target, "Y_target")
    if x.shape[1] < 1 or y.shape[1] < 1:
        raise ValueError("Qiao requires at least one cell feature and one target descriptor")
    d, w = _outcomes(D, W, (len(x), len(y)))
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}")
    if isinstance(rank, bool) or int(rank) != rank or not 1 <= rank <= min(x.shape[1] + 1, y.shape[1] + 1):
        raise ValueError("rank exceeds augmented cell or target feature dimensions")
    if not np.isfinite(l2) or l2 < 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("l2 must be nonnegative and tolerance positive")
    if maxiter <= 0 or init_direct_maxiter <= 0:
        raise ValueError("Optimizer iteration budgets must be positive")
    rank = int(rank)
    xb, yb = _augment(x), _augment(y)
    if initial_fit is not None:
        if (initial_fit.objective != objective or initial_fit.rank != rank or
                initial_fit.l2 != l2 or
                initial_fit.cell_loading.shape != (xb.shape[1], rank) or
                not np.array_equal(initial_fit.target_features, y)):
            raise ValueError("A continued Qiao fit must use the same features and configuration")
        parts = [initial_fit.cell_loading.ravel(), initial_fit.target_loading.ravel()]
        if objective == "logit":
            parts.append(initial_fit.target_intercept)
        theta0 = np.concatenate(parts)
    elif objective == "logit":
        warm = _direct_warm_start(xb, d, w, init_direct_maxiter)
        coefficient = warm @ np.linalg.pinv(yb.T)
        u, singular, vt = np.linalg.svd(coefficient, full_matrices=False)
        root = np.sqrt(np.maximum(singular[:rank], 0.0))
        a0, b0 = u[:, :rank] * root, vt[:rank, :].T * root
        theta0 = np.concatenate((a0.ravel(), b0.ravel(), np.zeros(len(y))))
    else:
        rng = np.random.default_rng(seed)
        a0 = rng.normal(scale=0.05, size=(xb.shape[1], rank))
        b0 = rng.normal(scale=0.05, size=(yb.shape[1], rank))
        theta0 = np.concatenate((a0.ravel(), b0.ravel()))
    result = minimize(_objective_gradient, theta0, method="L-BFGS-B", jac=True,
                      args=(xb, yb, d, w, rank, l2, objective),
                      options={"maxiter": int(maxiter), "ftol": float(tolerance),
                               "gtol": float(tolerance), "maxls": 30})
    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise FloatingPointError("Qiao optimization produced nonfinite parameters")
    a, b, intercept = _unpack(result.x, xb.shape[1], yb.shape[1], rank, len(y), objective)
    return QiaoFit(a.copy(), b.copy(), intercept.copy(), y.copy(), objective, rank,
                   float(l2), bool(result.success), int(result.nit), float(result.fun),
                   str(result.message), int(w.sum()))


@dataclass(frozen=True)
class QiaoTrial:
    rank: int
    l2: float
    validation_loss: float
    converged: bool
    iterations: int
    objective_value: float
    selection_metric: str


@dataclass(frozen=True)
class QiaoTuningResult:
    selected_fit: QiaoFit
    selected_config: Mapping[str, Any]
    trials: tuple[QiaoTrial, ...]
    # Selection is performed on validation D; fit remains training-only.
    # A caller must refit selected_config on authorized development inputs.


def tune_qiao(
    train_X: np.ndarray,
    Y_target: np.ndarray,
    train_D: np.ndarray,
    validation_X: np.ndarray,
    validation_D: np.ndarray,
    *,
    train_mask: np.ndarray | None = None,
    validation_mask: np.ndarray | None = None,
    candidates: Iterable[Mapping[str, Any]] | None = None,
    objective: str = "logit",
    selection_metric: str = "observed_log_loss",
    seed: int = 0,
    maxiter: int = 1000,
    tolerance: float = 1e-8,
    init_direct_maxiter: int = 120,
) -> QiaoTuningResult:
    """Select by common observed validation log loss; never accepts reference labels.

    Squared-error validation is available only as an explicitly named legacy
    sensitivity analysis. Candidate checkpointing/refit belongs to the shared
    experiment runner; these functions do not create a second cache protocol.
    """
    if selection_metric not in ("observed_log_loss", "squared_error"):
        raise ValueError("Unknown Qiao validation metric")
    val_x, y = _array2d(validation_X, "validation_X"), _array2d(Y_target, "Y_target")
    val_d, val_w = _outcomes(validation_D, validation_mask, (len(val_x), len(y)))
    if candidates is None:
        limit = min(np.asarray(train_X).shape[1] + 1, y.shape[1] + 1)
        ranks = sorted({r for r in (1, 2, 4, limit) if r <= limit})
        candidates = ({"rank": r, "l2": l2} for r in ranks for l2 in (3e-4, 3e-3, 3e-2))
    trials: list[QiaoTrial] = []
    best: tuple[tuple[Any, ...], QiaoFit, dict[str, Any]] | None = None
    seen: set[tuple[int, float]] = set()
    for candidate in candidates:
        raw_rank, raw_l2 = candidate["rank"], candidate["l2"]
        if isinstance(raw_rank, bool) or int(raw_rank) != raw_rank:
            raise ValueError("Candidate rank must be an integer")
        rank, l2 = int(raw_rank), float(raw_l2)
        if (rank, l2) in seen:
            continue
        seen.add((rank, l2))
        config = {"rank": rank, "l2": l2, "objective": objective}
        fitted = fit_qiao(train_X, y, train_D, train_mask, **config,
                          seed=stable_seed(seed, "qiao", objective, rank, l2),
                          maxiter=maxiter, tolerance=tolerance,
                          init_direct_maxiter=init_direct_maxiter)
        if selection_metric == "observed_log_loss":
            prediction = np.clip(fitted.predict_proba(val_x)[val_w], _EPS, 1 - _EPS)
            d = val_d[val_w]
            loss = float(np.mean(-d * np.log(prediction) - (1 - d) * np.log1p(-prediction)))
        else:
            prediction = fitted.predict_score(val_x) if objective == "squared_error" else fitted.predict_proba(val_x)
            loss = float(np.mean((prediction[val_w] - val_d[val_w])**2))
        trials.append(QiaoTrial(rank, l2, loss, fitted.converged, fitted.iterations,
                                fitted.objective_value, selection_metric))
        key = (not fitted.converged, loss, rank, -l2)
        if best is None or key < best[0]:
            best = (key, fitted, config)
    if best is None:
        raise ValueError("At least one Qiao candidate is required")
    return QiaoTuningResult(best[1], best[2], tuple(trials))
