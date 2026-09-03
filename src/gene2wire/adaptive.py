"""Generic NumPy/Adam models for matched direct and structured comparisons.

This module intentionally knows only arrays, masks, exposure probabilities, and
the three coefficient structures supported by the core.  Dataset acquisition,
splitting, calibration, reference truth, and scientific metrics belong to the
experiment layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

from .data import DatasetBundle
from .models import _exposure_matrix


@dataclass(frozen=True)
class AdaptiveFitConfig:
    """Optimization settings for the float32 Adam backend."""

    max_epochs: int = 70
    batch_size: int = 8192
    patience: int = 8
    gradient_clip_norm: float = 5.0
    improvement_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least one")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if self.patience < 1:
            raise ValueError("patience must be at least one")
        if not np.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite")
        if not np.isfinite(self.improvement_tolerance) or self.improvement_tolerance < 0:
            raise ValueError("improvement_tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class AdaptiveModelConfig:
    """One optimizer candidate independent of any dataset-specific vocabulary."""

    name: str
    kind: str
    pu: bool
    rank: int = 0
    epsilon: float = 0.0
    shared_l2: float = 0.0
    residual_l1: float = 0.0
    residual_l2: float = 0.0
    learning_rate: float = 0.01
    initialization: str = "random"
    frozen_direct: bool = False
    nested_role: str = "candidate"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be nonempty")
        if self.kind not in {"direct", "lowrank", "joint"}:
            raise ValueError("kind must be direct, lowrank, or joint")
        if self.rank < 0:
            raise ValueError("rank must be nonnegative")
        if self.kind == "direct" and self.rank != 0:
            raise ValueError("direct candidates must use rank zero")
        if self.kind != "direct" and self.rank < 1 and not self.frozen_direct:
            raise ValueError("trainable structured candidates require positive rank")
        for field_name in ("shared_l2", "residual_l1", "residual_l2"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if not 0 <= self.epsilon < 1:
            raise ValueError("epsilon must be in [0, 1)")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")


def _binary_log_loss(labels: Any, probabilities: Any) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _masked_macro_metrics(
    labels: Any,
    scores: Any,
    measured: Any,
    target_mask: Any,
) -> tuple[float, float]:
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(scores, dtype=np.float64)
    w = np.asarray(measured, dtype=bool)
    eligible = np.asarray(target_mask, dtype=bool)
    losses: list[float] = []
    average_precisions: list[float] = []
    for target in range(y.shape[1]):
        rows = w[:, target]
        if not eligible[target] or not np.any(rows):
            continue
        target_y = y[rows, target]
        target_p = p[rows, target]
        losses.append(_binary_log_loss(target_y, target_p))
        if np.unique(target_y).size == 2:
            average_precisions.append(float(average_precision_score(target_y, target_p)))
    loss = float(np.mean(losses)) if losses else float("nan")
    auprc = float(np.mean(average_precisions)) if average_precisions else float("nan")
    return loss, auprc


def per_target_validation_metrics(
    fitted: "AdaptiveFittedModel",
    validation: DatasetBundle,
    exposure: Any,
    target_mask: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired per-target observed-label loss and average precision."""

    detection = fitted.predict_observed(validation.X_cell, exposure)
    eligible = np.asarray(target_mask, dtype=bool)
    losses = np.full(validation.n_targets, np.nan, dtype=np.float64)
    average_precisions = np.full(validation.n_targets, np.nan, dtype=np.float64)
    for target in range(validation.n_targets):
        rows = validation.W_measured[:, target]
        if not eligible[target] or not np.any(rows):
            continue
        target_y = validation.S_observed[rows, target]
        target_p = detection[rows, target]
        losses[target] = _binary_log_loss(target_y, target_p)
        if np.unique(target_y).size == 2:
            average_precisions[target] = average_precision_score(target_y, target_p)
    return losses, average_precisions


@dataclass(frozen=True)
class AdaptiveFittedModel:
    """Fitted Adam parameters and validation-checkpoint diagnostics."""

    config: AdaptiveModelConfig
    cell_shared: np.ndarray | None
    target_shared: np.ndarray | None
    residual: np.ndarray | None
    intercept: np.ndarray
    best_validation_loss: float
    best_validation_auprc: float | None
    epochs_trained: int
    best_epoch: int
    initialization_error: float | None
    residual_zero_fraction: float | None

    @property
    def uses_shared(self) -> bool:
        return self.config.kind in {"lowrank", "joint"}

    @property
    def uses_residual(self) -> bool:
        return self.config.kind in {"direct", "joint"}

    def latent_logit(self, x_cell: Any) -> np.ndarray:
        x = np.asarray(x_cell, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("x_cell must be 2-D")
        eta = np.broadcast_to(self.intercept, (x.shape[0], self.intercept.size)).copy()
        if self.uses_shared:
            if self.cell_shared is None or self.target_shared is None:
                raise ValueError("fitted shared parameters are missing")
            if x.shape[1] != self.cell_shared.shape[0]:
                raise ValueError("x_cell feature count differs from fitted model")
            eta += (x @ self.cell_shared) @ self.target_shared.T
        if self.uses_residual:
            if self.residual is None:
                raise ValueError("fitted residual parameters are missing")
            if x.shape[1] != self.residual.shape[0]:
                raise ValueError("x_cell feature count differs from fitted model")
            eta += x @ self.residual
        return eta

    def predict_proba(self, x_cell: Any) -> np.ndarray:
        return expit(self.latent_logit(x_cell)).astype(np.float32)

    def predict_observed(self, x_cell: Any, exposure: Any = 1.0) -> np.ndarray:
        latent = self.predict_proba(x_cell)
        if not self.config.pu:
            return latent
        sensitivity = _exposure_matrix(exposure, latent.shape).astype(np.float32)
        sensitivity = np.clip(sensitivity, self.config.epsilon + 1e-4, 1.0)
        return (
            self.config.epsilon
            + (sensitivity - self.config.epsilon) * latent
        ).astype(np.float32)

    def effective_beta(self) -> np.ndarray:
        if self.config.kind == "direct":
            assert self.residual is not None
            return self.residual.copy()
        assert self.cell_shared is not None and self.target_shared is not None
        shared = self.cell_shared @ self.target_shared.T
        if self.config.kind == "lowrank":
            return shared
        assert self.residual is not None
        return shared + self.residual

    def diagnostics(self) -> dict[str, float | None]:
        if self.config.kind != "joint":
            return {
                "residual_zero_fraction": None,
                "shared_beta_frobenius_norm": None,
                "residual_beta_frobenius_norm": None,
                "residual_norm_fraction": None,
            }
        assert self.cell_shared is not None
        assert self.target_shared is not None
        assert self.residual is not None
        shared_norm = float(np.linalg.norm(self.cell_shared @ self.target_shared.T))
        residual_norm = float(np.linalg.norm(self.residual))
        return {
            "residual_zero_fraction": self.residual_zero_fraction,
            "shared_beta_frobenius_norm": shared_norm,
            "residual_beta_frobenius_norm": residual_norm,
            "residual_norm_fraction": residual_norm / max(shared_norm + residual_norm, 1e-12),
        }

    def checkpoint_parts(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        metadata: dict[str, Any] = {
            "config": asdict(self.config),
            "best_validation_loss": self.best_validation_loss,
            "best_validation_auprc": self.best_validation_auprc,
            "epochs_trained": self.epochs_trained,
            "best_epoch": self.best_epoch,
            "initialization_error": self.initialization_error,
            "residual_zero_fraction": self.residual_zero_fraction,
            "array_fields": ["intercept"],
        }
        arrays = {"intercept": np.asarray(self.intercept)}
        for name in ("cell_shared", "target_shared", "residual"):
            value = getattr(self, name)
            if value is not None:
                metadata["array_fields"].append(name)
                arrays[name] = np.asarray(value)
        return metadata, arrays

    @classmethod
    def from_checkpoint_parts(
        cls,
        metadata: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        *,
        prefix: str = "",
    ) -> "AdaptiveFittedModel":
        fields = metadata.get("array_fields")
        if not isinstance(fields, list) or "intercept" not in fields:
            raise ValueError("adaptive model checkpoint has an invalid array index")
        values: dict[str, np.ndarray | None] = {}
        for name in ("cell_shared", "target_shared", "residual", "intercept"):
            key = f"{prefix}{name}"
            values[name] = np.asarray(arrays[key], dtype=np.float32) if name in fields else None
            if name in fields and key not in arrays:
                raise ValueError(f"adaptive model checkpoint is missing {key}")
        assert values["intercept"] is not None
        return cls(
            config=AdaptiveModelConfig(**dict(metadata["config"])),
            cell_shared=values["cell_shared"],
            target_shared=values["target_shared"],
            residual=values["residual"],
            intercept=values["intercept"],
            best_validation_loss=float(metadata["best_validation_loss"]),
            best_validation_auprc=(
                None
                if metadata.get("best_validation_auprc") is None
                else float(metadata["best_validation_auprc"])
            ),
            epochs_trained=int(metadata["epochs_trained"]),
            best_epoch=int(metadata["best_epoch"]),
            initialization_error=(
                None
                if metadata.get("initialization_error") is None
                else float(metadata["initialization_error"])
            ),
            residual_zero_fraction=(
                None
                if metadata.get("residual_zero_fraction") is None
                else float(metadata["residual_zero_fraction"])
            ),
        )


class AdaptivePUModel:
    """Fit a direct, low-rank, or shared-plus-residual model with Adam."""

    def __init__(self, config: AdaptiveModelConfig, fit_config: AdaptiveFitConfig | None = None):
        if config.frozen_direct:
            raise ValueError("a frozen direct alias is not trainable")
        self.config = config
        self.fit_config = fit_config or AdaptiveFitConfig()

    def _sensitivity(self, exposure: Any, shape: tuple[int, int]) -> np.ndarray:
        if not self.config.pu:
            return np.ones(shape, dtype=np.float32)
        values = _exposure_matrix(exposure, shape).astype(np.float32)
        return np.clip(values, self.config.epsilon + 1e-4, 1.0)

    def fit(
        self,
        train: DatasetBundle,
        validation: DatasetBundle,
        *,
        train_exposure: Any = 1.0,
        validation_exposure: Any = 1.0,
        validation_target_mask: Any | None = None,
        seed: int = 0,
        initial_beta: Any | None = None,
        initial_bias: Any | None = None,
        preserve_initial_checkpoint: bool = False,
    ) -> AdaptiveFittedModel:
        if train.n_features != validation.n_features or train.target_ids != validation.target_ids:
            raise ValueError("train and validation dimensions/target order must match")
        x_train = np.asarray(train.X_cell, dtype=np.float32)
        x_validation = np.asarray(validation.X_cell, dtype=np.float32)
        z_train = np.asarray(train.S_observed, dtype=np.float32)
        z_validation = np.asarray(validation.S_observed, dtype=np.float32)
        w_train = np.asarray(train.W_measured, dtype=bool)
        w_validation = np.asarray(validation.W_measured, dtype=bool)
        sensitivity_train = self._sensitivity(train_exposure, z_train.shape)
        sensitivity_validation = self._sensitivity(validation_exposure, z_validation.shape)
        if np.any((z_train == 1) & w_train & (sensitivity_train <= self.config.epsilon)):
            raise ValueError("a measured positive cannot have zero effective exposure")
        eligible = (
            np.ones(train.n_targets, dtype=bool)
            if validation_target_mask is None
            else np.asarray(validation_target_mask, dtype=bool)
        )
        if eligible.shape != (train.n_targets,) or not np.any(eligible):
            raise ValueError("validation_target_mask must select at least one aligned target")

        rng = np.random.default_rng(seed)
        n_rows, n_features = x_train.shape
        n_targets = train.n_targets
        observed_prevalence = np.sum(z_train * w_train, axis=0) / np.maximum(w_train.sum(axis=0), 1)
        mean_sensitivity = np.sum(sensitivity_train * w_train, axis=0) / np.maximum(w_train.sum(axis=0), 1)
        prevalence_for_bias = np.minimum(
            np.maximum(observed_prevalence, self.config.epsilon + 1e-4),
            np.maximum(mean_sensitivity - 1e-4, self.config.epsilon + 2e-4),
        )
        latent_prevalence = np.clip(
            (prevalence_for_bias - self.config.epsilon)
            / np.maximum(mean_sensitivity - self.config.epsilon, 1e-4),
            1e-4,
            1 - 1e-4,
        )
        intercept = np.log(latent_prevalence / (1 - latent_prevalence)).astype(np.float32)

        beta = None if initial_beta is None else np.asarray(initial_beta, dtype=np.float32)
        if beta is not None and beta.shape != (n_features, n_targets):
            raise ValueError("initial_beta has the wrong shape")
        if initial_bias is not None:
            intercept = np.asarray(initial_bias, dtype=np.float32).copy()
            if intercept.shape != (n_targets,):
                raise ValueError("initial_bias has the wrong shape")

        cell_shared: np.ndarray | None = None
        target_shared: np.ndarray | None = None
        residual: np.ndarray | None = None
        parameters: list[np.ndarray] = []
        parameter_names: list[str] = []
        initialization_error: float | None = None
        if self.config.kind in {"lowrank", "joint"}:
            if self.config.rank > min(n_features, n_targets):
                raise ValueError("rank exceeds the available feature/target dimension")
            if beta is not None:
                left, singular_values, right_t = np.linalg.svd(beta, full_matrices=False)
                used_rank = min(self.config.rank, len(singular_values))
                cell_shared = np.zeros((n_features, self.config.rank), dtype=np.float32)
                target_shared = np.zeros((n_targets, self.config.rank), dtype=np.float32)
                root = np.sqrt(np.maximum(singular_values[:used_rank], 0.0)).astype(np.float32)
                cell_shared[:, :used_rank] = left[:, :used_rank] * root[None, :]
                target_shared[:, :used_rank] = right_t[:used_rank].T * root[None, :]
            else:
                cell_shared = rng.normal(0, 0.05, (n_features, self.config.rank)).astype(np.float32)
                target_shared = rng.normal(0, 0.05, (n_targets, self.config.rank)).astype(np.float32)
            parameters.extend([cell_shared, target_shared])
            parameter_names.extend(["cell_shared", "target_shared"])
        if self.config.kind in {"direct", "joint"}:
            if beta is not None and self.config.kind == "joint":
                assert cell_shared is not None and target_shared is not None
                residual = (beta - cell_shared @ target_shared.T).astype(np.float32)
            else:
                scale = 0.0 if self.config.kind == "joint" else 0.005
                residual = rng.normal(0, scale, (n_features, n_targets)).astype(np.float32)
            parameters.append(residual)
            parameter_names.append("residual")
        if beta is not None:
            if self.config.kind == "lowrank":
                assert cell_shared is not None and target_shared is not None
                effective = cell_shared @ target_shared.T
            elif self.config.kind == "joint":
                assert cell_shared is not None and target_shared is not None and residual is not None
                effective = cell_shared @ target_shared.T + residual
            else:
                assert residual is not None
                effective = residual
            initialization_error = float(np.max(np.abs(effective - beta)))
            if self.config.kind == "joint" and initialization_error >= 2e-6:
                raise RuntimeError("shared-plus-residual initialization failed exact reconstruction")
        parameters.append(intercept)
        parameter_names.append("intercept")

        def forward(x: np.ndarray, sensitivity: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
            theta = None
            if self.config.kind in {"lowrank", "joint"}:
                assert cell_shared is not None and target_shared is not None
                theta = x @ cell_shared
                eta = theta @ target_shared.T
                if self.config.kind == "joint":
                    assert residual is not None
                    eta = eta + x @ residual
            else:
                assert residual is not None
                eta = x @ residual
            latent = expit(eta + intercept).astype(np.float32)
            detection = (
                self.config.epsilon
                + (sensitivity - self.config.epsilon) * latent
            ).astype(np.float32)
            return theta, latent, detection

        def penalty() -> float:
            value = 0.0
            if cell_shared is not None and target_shared is not None:
                value += self.config.shared_l2 * (
                    np.sum(cell_shared**2) + np.sum(target_shared**2)
                )
            if residual is not None:
                value += self.config.residual_l2 * np.sum(residual**2)
                value += self.config.residual_l1 * np.sum(np.abs(residual))
            return float(value)

        def objective(
            x: np.ndarray,
            z: np.ndarray,
            w: np.ndarray,
            sensitivity: np.ndarray,
        ) -> float:
            _, _, probability = forward(x, sensitivity)
            probability = np.clip(probability, 1e-6, 1 - 1e-6)
            safe_z = np.where(w, z, 0.0)
            weights = w.astype(np.float32)
            bce = -np.sum(
                weights * (safe_z * np.log(probability) + (1 - safe_z) * np.log1p(-probability))
            ) / max(float(weights.sum()), 1e-12)
            return float(bce + penalty())

        first_moment = [np.zeros_like(parameter) for parameter in parameters]
        second_moment = [np.zeros_like(parameter) for parameter in parameters]
        adam_step = 0
        best_loss = float("inf")
        best_auprc = float("-inf")
        best_parameters: list[np.ndarray] | None = None
        best_epoch: int | None = None
        stale_epochs = 0

        if preserve_initial_checkpoint:
            _, _, initial_detection = forward(x_validation, sensitivity_validation)
            initial_loss, initial_auprc = _masked_macro_metrics(
                z_validation, initial_detection, w_validation, eligible
            )
            if not np.isfinite(initial_loss):
                raise RuntimeError("the warm-start validation loss is not finite")
            best_loss = initial_loss
            best_auprc = initial_auprc
            best_parameters = [parameter.copy() for parameter in parameters]
            best_epoch = 0

        for epoch in range(self.fit_config.max_epochs):
            shuffled = rng.permutation(n_rows)
            batches = max(1, math.ceil(n_rows / self.fit_config.batch_size))
            for indices in np.array_split(shuffled, batches):
                x_batch = x_train[indices]
                z_batch = z_train[indices]
                w_batch = w_train[indices]
                sensitivity_batch = sensitivity_train[indices]
                theta, latent, detection = forward(x_batch, sensitivity_batch)
                detection = np.clip(detection, 1e-6, 1 - 1e-6)
                normalization = max(float(w_train.sum()) / batches, 1e-12)
                derivative = w_batch.astype(np.float32) * (
                    (detection - z_batch) / (detection * (1 - detection))
                )
                derivative *= (
                    (sensitivity_batch - self.config.epsilon)
                    * latent
                    * (1 - latent)
                    / normalization
                )
                gradients: list[np.ndarray] = []
                if cell_shared is not None and target_shared is not None:
                    assert theta is not None
                    gradients.extend(
                        [
                            x_batch.T @ (derivative @ target_shared)
                            + 2 * self.config.shared_l2 * cell_shared,
                            derivative.T @ theta + 2 * self.config.shared_l2 * target_shared,
                        ]
                    )
                if residual is not None:
                    gradients.append(
                        x_batch.T @ derivative + 2 * self.config.residual_l2 * residual
                    )
                gradients.append(derivative.sum(axis=0))
                total_norm = math.sqrt(sum(float(np.sum(gradient**2)) for gradient in gradients))
                if total_norm > self.fit_config.gradient_clip_norm:
                    scale = self.fit_config.gradient_clip_norm / max(total_norm, 1e-12)
                    gradients = [gradient * scale for gradient in gradients]
                adam_step += 1
                for index, (name, parameter, gradient) in enumerate(
                    zip(parameter_names, parameters, gradients)
                ):
                    first_moment[index] = 0.9 * first_moment[index] + 0.1 * gradient
                    second_moment[index] = 0.999 * second_moment[index] + 0.001 * gradient**2
                    m_hat = first_moment[index] / (1 - 0.9**adam_step)
                    v_hat = second_moment[index] / (1 - 0.999**adam_step)
                    step_size = self.config.learning_rate / (np.sqrt(v_hat) + 1e-8)
                    parameter -= step_size * m_hat
                    if name == "residual" and self.config.residual_l1 > 0:
                        parameter[...] = np.sign(parameter) * np.maximum(
                            np.abs(parameter) - step_size * self.config.residual_l1,
                            0.0,
                        )

            _, _, validation_detection = forward(x_validation, sensitivity_validation)
            validation_loss, validation_auprc = _masked_macro_metrics(
                z_validation, validation_detection, w_validation, eligible
            )
            if not np.isfinite(validation_loss):
                raise RuntimeError("validation loss became non-finite")
            improved = validation_loss < best_loss - self.fit_config.improvement_tolerance
            if abs(validation_loss - best_loss) <= self.fit_config.improvement_tolerance:
                improved = improved or (
                    np.isfinite(validation_auprc)
                    and (not np.isfinite(best_auprc) or validation_auprc > best_auprc)
                )
            if improved:
                best_loss = validation_loss
                best_auprc = validation_auprc
                best_parameters = [parameter.copy() for parameter in parameters]
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.fit_config.patience:
                    break

        if best_parameters is None or best_epoch is None:
            raise RuntimeError("no finite validation checkpoint was recorded")
        position = 0
        if cell_shared is not None and target_shared is not None:
            cell_shared = best_parameters[position]
            target_shared = best_parameters[position + 1]
            position += 2
        if residual is not None:
            residual = best_parameters[position]
            position += 1
        intercept = best_parameters[position]
        residual_zero_fraction = (
            float(np.mean(np.isclose(residual, 0.0, atol=1e-8)))
            if self.config.kind == "joint" and residual is not None
            else None
        )
        return AdaptiveFittedModel(
            config=self.config,
            cell_shared=None if cell_shared is None else cell_shared.copy(),
            target_shared=None if target_shared is None else target_shared.copy(),
            residual=None if residual is None else residual.copy(),
            intercept=intercept.copy(),
            best_validation_loss=float(best_loss),
            best_validation_auprc=(float(best_auprc) if np.isfinite(best_auprc) else None),
            epochs_trained=epoch + 1,
            best_epoch=int(best_epoch),
            initialization_error=initialization_error,
            residual_zero_fraction=residual_zero_fraction,
        )


__all__ = [
    "AdaptiveFitConfig",
    "AdaptiveFittedModel",
    "AdaptiveModelConfig",
    "AdaptivePUModel",
    "per_target_validation_metrics",
]
