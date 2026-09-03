"""Data-agnostic orchestration for tuning, refitting, prediction, and resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .checkpoint import (
    AtomicArrayCheckpointStore,
    AtomicCheckpointStore,
    canonical_json,
    experiment_fingerprint,
    sha256_array,
    unit_key,
)
from .config import FitConfig, ModelConfig, TuningConfig
from .data import DatasetBundle
from .models import FittedModel, UnifiedPUModel, _exposure_matrix
from .seeds import stable_seed
from .tuning import TuningResult, tune_model


CORE_API_VERSION = "0.2.0"


@dataclass(frozen=True)
class ModelRunResult:
    """One selected/refitted model and its test-set predictions."""

    model_name: str
    tuning: TuningResult
    fitted: FittedModel
    latent_probability: np.ndarray
    observed_probability: np.ndarray
    resumed: bool

    def summary(self) -> dict[str, Any]:
        config = self.tuning.best_config
        return {
            "model": self.model_name,
            "kind": config.kind,
            "pu": config.pu,
            "rank": config.rank,
            "shared_l2": config.shared_l2,
            "residual_l2": config.residual_l2,
            "use_target_features": config.use_target_features,
            "target_l2": config.target_l2,
            "validation_observed_log_loss": self.tuning.best_validation_loss,
            "tuning_trials": len(self.tuning.trials),
            "final_converged": self.fitted.converged,
            "final_iterations": self.fitted.iterations,
            "final_objective": self.fitted.objective,
            "resumed": self.resumed,
        }


@dataclass(frozen=True)
class GridRunResult:
    """Results for a complete, fingerprinted model grid run."""

    fingerprint: str
    code_version: str
    test_cell_ids: tuple[str, ...]
    models: Mapping[str, ModelRunResult]

    def summary_rows(self) -> list[dict[str, Any]]:
        return [result.summary() for result in self.models.values()]


def _require_reference_free(name: str, bundle: DatasetBundle) -> None:
    if bundle.Z_reference is not None or bundle.reference_mask is not None:
        raise ValueError(
            f"{name} contains evaluation-only reference truth; call without_reference() "
            "before passing it to run_model_grid"
        )


def _assert_aligned(left: DatasetBundle, right: DatasetBundle, names: str) -> None:
    if left.n_features != right.n_features:
        raise ValueError(f"{names} feature counts differ")
    if left.target_ids != right.target_ids:
        raise ValueError(f"{names} target_ids differ or are permuted")
    if left.feature_blocks != right.feature_blocks:
        raise ValueError(f"{names} feature_blocks differ")
    if (left.Y_target is None) != (right.Y_target is None):
        raise ValueError(f"{names} Y_target availability differs")
    if left.Y_target is not None and not np.array_equal(left.Y_target, right.Y_target):
        raise ValueError(f"{names} Y_target matrices differ or are permuted")
    if set(left.groups) != set(right.groups):
        raise ValueError(f"{names} group columns differ")


def _concatenate_for_refit(train: DatasetBundle, validation: DatasetBundle) -> DatasetBundle:
    _assert_aligned(train, validation, "train/validation")
    overlap = set(train.cell_ids).intersection(validation.cell_ids)
    if overlap:
        raise ValueError("train and validation cell_ids overlap")
    return DatasetBundle(
        X_cell=np.concatenate([train.X_cell, validation.X_cell], axis=0),
        S_observed=np.concatenate([train.S_observed, validation.S_observed], axis=0),
        W_measured=np.concatenate([train.W_measured, validation.W_measured], axis=0),
        Y_target=train.Y_target,
        cell_ids=(*train.cell_ids, *validation.cell_ids),
        target_ids=train.target_ids,
        groups={
            name: np.concatenate([train.groups[name], validation.groups[name]])
            for name in train.groups
        },
        feature_blocks=train.feature_blocks,
        semantics=train.semantics,
        metadata=train.metadata,
    )


def _bundle_hash(bundle: DatasetBundle) -> str:
    parts: dict[str, Any] = {
        "X_cell": sha256_array(bundle.X_cell),
        "S_observed": sha256_array(bundle.S_observed.astype(np.uint8)),
        "W_measured": sha256_array(bundle.W_measured.astype(np.uint8)),
        "cell_ids": sha256_array(np.asarray(bundle.cell_ids, dtype=str)),
        "target_ids": sha256_array(np.asarray(bundle.target_ids, dtype=str)),
        "feature_blocks": dict(bundle.feature_blocks),
        "groups": {
            name: sha256_array(np.asarray(values, dtype=str))
            for name, values in sorted(bundle.groups.items())
        },
    }
    if bundle.Y_target is not None:
        parts["Y_target"] = sha256_array(bundle.Y_target)
    return hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()


def _resolve_tuning(
    value: TuningConfig | Mapping[str, TuningConfig], model_name: str
) -> TuningConfig:
    if isinstance(value, TuningConfig):
        return value
    if model_name not in value or not isinstance(value[model_name], TuningConfig):
        raise ValueError(f"missing TuningConfig for model {model_name!r}")
    return value[model_name]


def _resolve_fit(value: FitConfig | Mapping[str, FitConfig] | None, model_name: str) -> FitConfig:
    if value is None:
        return FitConfig()
    if isinstance(value, FitConfig):
        return value
    if model_name not in value or not isinstance(value[model_name], FitConfig):
        raise ValueError(f"missing FitConfig for model {model_name!r}")
    return value[model_name]


def _state_to_checkpoint(fitted: FittedModel) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = fitted.state_dict()
    arrays: dict[str, np.ndarray] = {}
    array_fields = (
        "cell_shared",
        "target_shared",
        "residual",
        "target_coeff",
        "target_features",
        "intercept",
    )
    present: list[str] = []
    for name in array_fields:
        value = state.pop(name)
        if value is not None:
            arrays[f"fitted__{name}"] = np.asarray(value)
            present.append(name)
    state["array_fields"] = present
    return state, arrays


def _state_from_checkpoint(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> FittedModel:
    present = metadata.get("array_fields")
    if not isinstance(present, list) or "intercept" not in present:
        raise ValueError("completed model checkpoint has an invalid fitted array index")
    state = dict(metadata)
    state.pop("array_fields", None)
    for name in (
        "cell_shared",
        "target_shared",
        "residual",
        "target_coeff",
        "target_features",
        "intercept",
    ):
        key = f"fitted__{name}"
        state[name] = arrays[key] if name in present else None
        if name in present and key not in arrays:
            raise ValueError(f"completed model checkpoint is missing {key}")
    return FittedModel.from_state_dict(state)


def run_model_grid(
    *,
    train: DatasetBundle,
    validation: DatasetBundle,
    test_X: Any,
    models: Sequence[ModelConfig],
    tuning: TuningConfig | Mapping[str, TuningConfig],
    train_exposure: Any = 1.0,
    validation_exposure: Any = 1.0,
    test_exposure: Any = 1.0,
    fit: FitConfig | Mapping[str, FitConfig] | None = None,
    refit: DatasetBundle | None = None,
    refit_exposure: Any | None = None,
    test_cell_ids: Sequence[Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    unit_context: Mapping[str, Any] | None = None,
    seed: int = 0,
    code_version: str = CORE_API_VERSION,
    run_fingerprint: str | None = None,
    on_model: Callable[[str, str], None] | None = None,
) -> GridRunResult:
    """Tune and refit multiple models without access to evaluation truth.

    ``train`` and ``validation`` must contain only observed PU labels.  The
    outer-test interface deliberately accepts ``test_X`` rather than a bundle,
    so hidden masks and pre-hide reference labels cannot enter model selection.
    Candidate trials and completed fitted models/predictions are checkpointed
    independently when ``checkpoint_dir`` is supplied.
    """

    _require_reference_free("train", train)
    _require_reference_free("validation", validation)
    _assert_aligned(train, validation, "train/validation")
    if set(train.cell_ids).intersection(validation.cell_ids):
        raise ValueError("train and validation cell_ids overlap")
    base_models = tuple(models)
    if not base_models:
        raise ValueError("models must be nonempty")
    if len({model.name for model in base_models}) != len(base_models):
        raise ValueError("model names must be unique")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(code_version, str) or not code_version:
        raise ValueError("code_version must be nonempty")

    test_array = np.asarray(test_X, dtype=np.float64)
    if test_array.ndim != 2 or test_array.shape[1] != train.n_features:
        raise ValueError("test_X must be 2-D with the fitted feature count")
    if not np.all(np.isfinite(test_array)):
        raise ValueError("test_X contains non-finite values")
    test_ids = (
        tuple(f"test_{index}" for index in range(test_array.shape[0]))
        if test_cell_ids is None
        else tuple(str(value) for value in test_cell_ids)
    )
    if len(test_ids) != test_array.shape[0] or len(set(test_ids)) != len(test_ids):
        raise ValueError("test_cell_ids must be unique and align with test_X")
    if set(test_ids).intersection((*train.cell_ids, *validation.cell_ids)):
        raise ValueError("outer-test cell IDs overlap train or validation")

    train_e = _exposure_matrix(train_exposure, train.S_observed.shape)
    validation_e = _exposure_matrix(validation_exposure, validation.S_observed.shape)
    test_e = _exposure_matrix(test_exposure, (test_array.shape[0], train.n_targets))
    if refit is None:
        development = _concatenate_for_refit(train, validation)
        development_e = np.concatenate([train_e, validation_e], axis=0)
    else:
        _require_reference_free("refit", refit)
        _assert_aligned(train, refit, "train/refit")
        if set(refit.cell_ids) != set((*train.cell_ids, *validation.cell_ids)):
            raise ValueError("refit cell IDs must equal the train/validation union")
        if refit_exposure is None:
            raise ValueError("refit_exposure is required when refit is supplied")
        development = refit
        development_e = _exposure_matrix(refit_exposure, refit.S_observed.shape)

    context = dict(unit_context or {})
    reserved = {"task", "model", "runner_model"}
    if reserved.intersection(context):
        raise ValueError(f"unit_context uses reserved keys: {sorted(reserved.intersection(context))}")
    if any(not isinstance(key, str) or not key for key in context):
        raise ValueError("unit_context keys must be nonempty strings")

    semantic_config = {
        "models": [asdict(model) for model in base_models],
        "tuning": {
            model.name: asdict(_resolve_tuning(tuning, model.name)) for model in base_models
        },
        "fit": {model.name: asdict(_resolve_fit(fit, model.name)) for model in base_models},
        "unit_context": context,
        "refit_supplied": refit is not None,
    }
    input_hashes = {
        "train": _bundle_hash(train),
        "validation": _bundle_hash(validation),
        "refit": _bundle_hash(development),
        "test_X": sha256_array(test_array),
        "test_cell_ids": sha256_array(np.asarray(test_ids, dtype=str)),
        "train_exposure": sha256_array(train_e),
        "validation_exposure": sha256_array(validation_e),
        "refit_exposure": sha256_array(development_e),
        "test_exposure": sha256_array(test_e),
    }
    fingerprint = run_fingerprint or experiment_fingerprint(
        semantic_config,
        input_hashes=input_hashes,
        code_version=code_version,
        seeds={"base_seed": seed},
    )
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("run_fingerprint must be None or a nonempty string")

    trial_store = None
    model_store = None
    if checkpoint_dir is not None:
        checkpoint_root = Path(checkpoint_dir)
        trial_store = AtomicCheckpointStore(checkpoint_root / "trials")
        model_store = AtomicArrayCheckpointStore(checkpoint_root / "models")

    results: dict[str, ModelRunResult] = {}
    for base_model in base_models:
        model_key = unit_key(task="completed_model", **context, model=base_model.name)
        cached = None if model_store is None else model_store.load(model_key, fingerprint)
        if cached is not None:
            payload = cached["payload"]
            arrays = cached["arrays"]
            if payload.get("model_name") != base_model.name:
                raise ValueError("completed checkpoint model name does not match")
            if payload.get("base_config") != asdict(base_model):
                raise ValueError("completed checkpoint base model does not match")
            if tuple(payload.get("test_cell_ids", [])) != test_ids:
                raise ValueError("completed checkpoint test cell order does not match")
            fitted_model = _state_from_checkpoint(payload["fitted"], arrays)
            tuned = TuningResult.from_dict(payload["tuning"])
            latent = np.asarray(arrays["latent_probability"], dtype=np.float64)
            observed = np.asarray(arrays["observed_probability"], dtype=np.float64)
            expected_shape = (test_array.shape[0], train.n_targets)
            if latent.shape != expected_shape or observed.shape != expected_shape:
                raise ValueError("completed checkpoint prediction shape does not match")
            result = ModelRunResult(
                model_name=base_model.name,
                tuning=tuned,
                fitted=fitted_model,
                latent_probability=latent,
                observed_probability=observed,
                resumed=True,
            )
            results[base_model.name] = result
            if on_model is not None:
                on_model(base_model.name, "resumed")
            continue

        if on_model is not None:
            on_model(base_model.name, "started")
        model_tuning = _resolve_tuning(tuning, base_model.name)
        model_fit = _resolve_fit(fit, base_model.name)
        tuning_seed = stable_seed(
            seed,
            "tune",
            base_model.kind,
            base_model.use_target_features,
        )
        tuned = tune_model(
            train=train,
            validation=validation,
            train_exposure=train_e,
            validation_exposure=validation_e,
            base_model=base_model,
            tuning=model_tuning,
            fit=model_fit,
            seed=tuning_seed,
            checkpoint_store=trial_store,
            checkpoint_fingerprint=fingerprint if trial_store is not None else None,
            checkpoint_context={**context, "runner_model": base_model.name},
        )
        final_seed = stable_seed(
            seed,
            "refit",
            tuned.best_config.kind,
            tuned.best_config.rank,
            tuned.best_config.use_target_features,
        )
        fitted_model = UnifiedPUModel(tuned.best_config, model_fit).fit(
            development,
            exposure=development_e,
            seed=final_seed,
        )
        latent = fitted_model.predict_proba(test_array)
        observed = fitted_model.predict_observed(test_array, exposure=test_e)
        result = ModelRunResult(
            model_name=base_model.name,
            tuning=tuned,
            fitted=fitted_model,
            latent_probability=latent,
            observed_probability=observed,
            resumed=False,
        )
        if model_store is not None:
            fitted_metadata, fitted_arrays = _state_to_checkpoint(fitted_model)
            model_store.save_complete(
                model_key,
                fingerprint,
                {
                    "model_name": base_model.name,
                    "base_config": asdict(base_model),
                    "tuning": tuned.to_dict(),
                    "fitted": fitted_metadata,
                    "test_cell_ids": list(test_ids),
                },
                {
                    **fitted_arrays,
                    "latent_probability": latent,
                    "observed_probability": observed,
                },
            )
        results[base_model.name] = result
        if on_model is not None:
            on_model(base_model.name, "completed")

    return GridRunResult(
        fingerprint=fingerprint,
        code_version=code_version,
        test_cell_ids=test_ids,
        models=results,
    )
