"""Data-agnostic orchestration for tuning, refitting, prediction, and resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .checkpoint import (
    AtomicArrayCheckpointStore,
    AtomicCheckpointStore,
    canonical_json,
    experiment_fingerprint,
    sha256_array,
    sha256_source_tree,
    unit_key,
)
from .config import FitConfig, ModelConfig, TuningConfig
from .data import DatasetBundle
from .models import (
    DirectWarmStartCache,
    FittedModel,
    UnifiedPUModel,
    _exposure_matrix,
    model_identity,
)
from .seeds import stable_seed
from .tuning import TuningResult, tune_model


# Checkpoint/model-identity schema epoch; independent of the package release.
CORE_API_VERSION = "0.6.0"


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
            "selected_structure": config.kind,
            "probability_semantics": "reference_p" if config.pu else "observed_q",
            "pu": config.pu,
            "rank": config.rank,
            "shared_l2": config.shared_l2,
            "residual_l2": config.residual_l2,
            "use_target_features": config.use_target_features,
            "target_l2": config.target_l2,
            "nuisance_l2": config.nuisance_l2,
            "n_nuisance": self.fitted.nuisance_coeff.shape[0]
            if self.fitted.nuisance_coeff is not None
            else 0,
            "factor_parameterization": (
                "separate_A"
                if config.lowrank_feature_groups
                else ("shared_A" if config.kind in {"lowrank", "joint"} else "none")
            ),
            "lowrank_feature_groups": "|".join(config.lowrank_feature_groups),
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
    if left.nuisance_names != right.nuisance_names:
        raise ValueError(f"{names} nuisance schemas differ")


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
        X_nuisance=(
            None
            if train.X_nuisance is None
            else np.concatenate(
                [train.X_nuisance, validation.X_nuisance], axis=0
            )
        ),
        nuisance_names=train.nuisance_names,
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
        "nuisance_names": list(bundle.nuisance_names),
    }
    if bundle.X_nuisance is not None:
        parts["X_nuisance"] = sha256_array(bundle.X_nuisance)
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


def _compatible_joint_endpoint(candidate: ModelConfig, joint: ModelConfig) -> bool:
    """Return whether a standalone model is the same structural endpoint.

    Nuisance penalties define a different fitted objective, and a grouped
    low-rank model (the gene-overlap Separate-A ablation) is not the low-rank
    boundary of the ordinary Joint family.
    """
    return (
        joint.kind == "joint"
        and candidate.kind in {"direct", "lowrank"}
        and candidate.pu == joint.pu
        and candidate.use_target_features == joint.use_target_features
        and candidate.nuisance_l2 == joint.nuisance_l2
        and not candidate.lowrank_feature_groups
    )


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
        "nuisance_coeff",
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
        "nuisance_coeff",
        "intercept",
    ):
        key = f"fitted__{name}"
        state[name] = arrays[key] if name in present else None
        if name in present and key not in arrays:
            raise ValueError(f"completed model checkpoint is missing {key}")
    return FittedModel.from_state_dict(state)


def _prediction_nuisance(
    value: Any | None,
    *,
    n_rows: int,
    names: tuple[str, ...],
    label: str,
) -> np.ndarray | None:
    """Validate a truth-free nuisance matrix for prediction."""

    if not names:
        if value is not None:
            raise ValueError(f"{label} was supplied but the fitted schema has no nuisance terms")
        return None
    if value is None:
        raise ValueError(f"{label} is required by the fitted nuisance schema")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape != (n_rows, len(names)):
        raise ValueError(
            f"{label} must have shape ({n_rows}, {len(names)})"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains non-finite values")
    return result


def run_model_grid(
    *,
    train: DatasetBundle,
    validation: DatasetBundle,
    test_X: Any,
    test_nuisance: Any | None = None,
    models: Sequence[ModelConfig],
    tuning: TuningConfig | Mapping[str, TuningConfig],
    train_exposure: Any = 1.0,
    validation_exposure: Any = 1.0,
    test_exposure: Any = 1.0,
    fit: FitConfig | Mapping[str, FitConfig] | None = None,
    refit: DatasetBundle | None = None,
    refit_exposure: Any | None = None,
    refit_test_X: Any | None = None,
    refit_test_nuisance: Any | None = None,
    test_cell_ids: Sequence[Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    unit_context: Mapping[str, Any] | None = None,
    seed: int = 0,
    code_version: str = CORE_API_VERSION,
    run_fingerprint: str | None = None,
    on_model: Callable[[str, str], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> GridRunResult:
    """Tune and refit multiple models without access to evaluation truth.

    ``train`` and ``validation`` must contain authorized training labels only.
    Clean-reference supervision is compiled upstream as labels with exposure 1;
    validation always retains its original detections and calibrated exposure.
    ``refit_test_X`` explicitly supplies the same test cells transformed by a
    development-refitted preprocessor, permitting a new feature schema. The
    outer-test interface deliberately accepts ``test_X`` rather than a bundle,
    so hidden masks and pre-hide reference labels cannot enter model selection.
    Candidate trials, reusable direct warm starts, and completed fitted
    models/predictions are checkpointed independently when ``checkpoint_dir``
    is supplied. ``on_progress`` reports timed model, candidate, and final-refit
    events, including compatible checkpoint and in-memory reuse. It supplements
    the existing ``on_model`` callback without changing its behavior.
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

    # Joint is a nested model family.  When exact endpoints are requested,
    # direct and low-rank models with the same PU/target-feature semantics are
    # tuned first, even if the caller supplied a different model order.  This
    # keeps the public API order-independent while making the inherited winner
    # available to Joint tuning and to the shared candidate/refit caches.
    def endpoint_sources(base_model: ModelConfig) -> tuple[ModelConfig, ...]:
        model_tuning = _resolve_tuning(tuning, base_model.name)
        if base_model.kind != "joint" or not model_tuning.include_endpoints:
            return ()
        matches = [candidate for candidate in base_models
                   if _compatible_joint_endpoint(candidate, base_model)]
        by_kind = {}
        for candidate in matches:
            by_kind.setdefault(candidate.kind, []).append(candidate)
        selected = []
        for kind in ("direct", "lowrank"):
            values = by_kind.get(kind, [])
            if len(values) > 1:
                raise ValueError(
                    f"Joint model {base_model.name!r} has multiple compatible {kind} "
                    "models; exact endpoint inheritance is ambiguous"
                )
            if values:
                source = values[0]
                if asdict(_resolve_fit(fit, source.name)) != asdict(_resolve_fit(fit, base_model.name)):
                    raise ValueError(
                        f"Joint model {base_model.name!r} and its {kind} endpoint "
                        "must use the same FitConfig"
                    )
                selected.append(source)
        return tuple(selected)

    endpoint_source_map = {
        model.name: endpoint_sources(model)
        for model in base_models if model.kind == "joint"
    }
    # A source model is always executed before its dependent Joint alias.  The
    # returned mapping below is restored to the caller's order.
    kind_order = {"direct": 0, "lowrank": 1, "joint": 2}
    execution_models = tuple(sorted(
        base_models,
        key=lambda model: (int(model.pu), kind_order[model.kind], base_models.index(model)),
    ))

    test_array = np.asarray(test_X, dtype=np.float64)
    if test_array.ndim != 2 or test_array.shape[1] != train.n_features:
        raise ValueError("test_X must be 2-D with the fitted feature count")
    if not np.all(np.isfinite(test_array)):
        raise ValueError("test_X contains non-finite values")
    test_nuisance_array = _prediction_nuisance(
        test_nuisance,
        n_rows=test_array.shape[0],
        names=train.nuisance_names,
        label="test_nuisance",
    )
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
        if refit_test_X is None:
            _assert_aligned(train, refit, "train/refit")
        elif train.target_ids != refit.target_ids:
            raise ValueError("train/refit target_ids differ or are permuted")
        if set(refit.cell_ids) != set((*train.cell_ids, *validation.cell_ids)):
            raise ValueError("refit cell IDs must equal the train/validation union")
        if refit_exposure is None:
            raise ValueError("refit_exposure is required when refit is supplied")
        development = refit
        development_e = _exposure_matrix(refit_exposure, refit.S_observed.shape)

    final_test_array = test_array if refit_test_X is None else np.asarray(refit_test_X, dtype=np.float64)
    if refit_test_X is not None and refit is None:
        raise ValueError("refit_test_X requires an explicit refit bundle")
    if (
        final_test_array.ndim != 2
        or final_test_array.shape != (len(test_ids), development.n_features)
        or not np.all(np.isfinite(final_test_array))
    ):
        raise ValueError("refit_test_X must align with test cells and the refit feature schema")
    if refit_test_nuisance is not None and refit is None:
        raise ValueError("refit_test_nuisance requires an explicit refit bundle")
    final_test_nuisance = _prediction_nuisance(
        test_nuisance_array
        if refit_test_nuisance is None
        else refit_test_nuisance,
        n_rows=len(test_ids),
        names=development.nuisance_names,
        label="refit_test_nuisance",
    )

    context = dict(unit_context or {})
    reserved = {"task", "model", "runner_model"}
    if reserved.intersection(context):
        raise ValueError(f"unit_context uses reserved keys: {sorted(reserved.intersection(context))}")
    if any(not isinstance(key, str) or not key for key in context):
        raise ValueError("unit_context keys must be nonempty strings")

    if run_fingerprint is not None and (not isinstance(run_fingerprint, str) or not run_fingerprint):
        raise ValueError("run_fingerprint must be None or a nonempty string")
    semantic_config = {
        "unit_context": context,
        "refit_supplied": refit is not None,
        "refit_preprocessing_supplied": refit_test_X is not None,
        "caller_fingerprint": run_fingerprint,
    }
    input_hashes = {
        "train": _bundle_hash(train),
        "validation": _bundle_hash(validation),
        "refit": _bundle_hash(development),
        "test_X": sha256_array(test_array),
        "refit_test_X": sha256_array(final_test_array),
        "test_nuisance": (
            sha256_array(np.empty((len(test_ids), 0), dtype=np.float64))
            if test_nuisance_array is None
            else sha256_array(test_nuisance_array)
        ),
        "refit_test_nuisance": (
            sha256_array(np.empty((len(test_ids), 0), dtype=np.float64))
            if final_test_nuisance is None
            else sha256_array(final_test_nuisance)
        ),
        "test_cell_ids": sha256_array(np.asarray(test_ids, dtype=str)),
        "train_exposure": sha256_array(train_e),
        "validation_exposure": sha256_array(validation_e),
        "refit_exposure": sha256_array(development_e),
        "test_exposure": sha256_array(test_e),
    }
    source_hash = sha256_source_tree(Path(__file__).resolve().parent)
    # The caller fingerprint supplements, never overrides, data/config/source
    # identity. This shared identity deliberately excludes the model list.
    data_fingerprint = experiment_fingerprint(
        semantic_config, input_hashes=input_hashes, code_version=code_version,
        seeds={"base_seed": seed}, source_hash=source_hash,
    )
    model_fingerprints = {
        model.name: experiment_fingerprint(
            {"data_fingerprint": data_fingerprint, "model": asdict(model),
             "tuning": asdict(_resolve_tuning(tuning, model.name)),
             "fit": asdict(_resolve_fit(fit, model.name)),
             "inherited_endpoint_sources": [
                 {"model": model_identity(source),
                  "tuning": asdict(_resolve_tuning(tuning, source.name)),
                  "fit": asdict(_resolve_fit(fit, source.name))}
                 for source in endpoint_source_map.get(model.name, ())]},
            input_hashes=input_hashes, code_version=code_version,
            seeds={"base_seed": seed}, source_hash=source_hash,
        ) for model in base_models
    }
    fingerprint = hashlib.sha256(canonical_json(model_fingerprints).encode("utf-8")).hexdigest()

    trial_store = None
    model_store = None
    warm_start_store = None
    refit_store = None
    if checkpoint_dir is not None:
        checkpoint_root = Path(checkpoint_dir)
        trial_store = AtomicCheckpointStore(checkpoint_root / "trials")
        model_store = AtomicArrayCheckpointStore(checkpoint_root / "models")
        refit_store = AtomicArrayCheckpointStore(checkpoint_root / "refits")
        warm_start_store = AtomicArrayCheckpointStore(
            checkpoint_root / "warm_starts"
        )

    # These caches are deliberately scoped to this run's fixed input arrays.
    # Tuning and refit use separate caches because their training rows differ.
    # A shared cache across model families removes repeated deterministic direct
    # fits while leaving every candidate's objective and SVD initialization
    # numerically unchanged.
    tuning_warm_starts = DirectWarmStartCache(
        checkpoint_store=warm_start_store,
        fingerprint=data_fingerprint if warm_start_store is not None else None,
        context={"phase": "tuning", "runner_context": context},
    )
    refit_warm_starts = DirectWarmStartCache(
        checkpoint_store=warm_start_store,
        fingerprint=data_fingerprint if warm_start_store is not None else None,
        context={"phase": "refit", "runner_context": context},
    )

    candidate_cache = {}
    final_fit_cache = {}

    def final_coordinates(config: ModelConfig, fit_config: FitConfig) -> tuple[str, int]:
        identity = model_identity(config)
        final_seed = stable_seed(
            seed, "refit", config.kind, config.rank, config.use_target_features,
        )
        return unit_key(task="canonical_refit", context=context, candidate=identity,
                        fit=asdict(fit_config), seed=final_seed), final_seed

    results: dict[str, ModelRunResult] = {}

    def emit(event: str, model: str, **details: Any) -> None:
        if on_progress is not None:
            on_progress({"event": event, "model": model, **details})

    for model_index, base_model in enumerate(execution_models, start=1):
        model_started = perf_counter()
        model_key = unit_key(task="completed_model", **context, model=base_model.name)
        model_fingerprint = model_fingerprints[base_model.name]
        cached = None if model_store is None else model_store.load(model_key, model_fingerprint)
        emit("model_start", base_model.name, index=model_index, total=len(base_models),
             cache_status="checkpoint" if cached is not None else "pending")
        if cached is not None:
            payload = cached["payload"]
            arrays = cached["arrays"]
            if payload.get("model_name") != base_model.name:
                raise ValueError("completed checkpoint model name does not match")
            if canonical_json(payload.get("base_config")) != canonical_json(
                asdict(base_model)
            ):
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
            refit_key, _ = final_coordinates(tuned.best_config, _resolve_fit(fit, base_model.name))
            final_fit_cache[refit_key] = fitted_model
            if on_model is not None:
                on_model(base_model.name, "resumed")
            emit("model_complete", base_model.name, index=model_index, total=len(base_models),
                 cache_status="checkpoint", elapsed_seconds=perf_counter() - model_started,
                 resumed=True, summary=result.summary())
            continue

        if on_model is not None:
            on_model(base_model.name, "started")
        model_tuning = _resolve_tuning(tuning, base_model.name)
        model_fit = _resolve_fit(fit, base_model.name)
        tuning_seed = stable_seed(seed, "tune")
        required = tuple(
            results[source.name].tuning.best_config.with_updates(name=base_model.name)
            for source in endpoint_source_map.get(base_model.name, ())
            if source.name in results
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
            checkpoint_fingerprint=data_fingerprint if trial_store is not None else None,
            checkpoint_context=context,
            warm_start_cache=tuning_warm_starts,
            candidate_cache=candidate_cache,
            on_progress=on_progress,
            required_endpoints=required or None,
        )
        refit_started = perf_counter()
        refit_key, final_seed = final_coordinates(tuned.best_config, model_fit)
        fitted_model = final_fit_cache.get(refit_key)
        refit_cache_status = "memory" if fitted_model is not None else "pending"
        if fitted_model is None and refit_store is not None:
            cached_refit = refit_store.load(refit_key, data_fingerprint)
            if cached_refit is not None:
                fitted_model = _state_from_checkpoint(cached_refit["payload"]["fitted"], cached_refit["arrays"])
                if model_identity(fitted_model.config) != model_identity(tuned.best_config):
                    raise ValueError("canonical refit checkpoint configuration does not match")
                refit_cache_status = "checkpoint"
        emit("refit_start", base_model.name, cache_status=refit_cache_status,
             config=asdict(tuned.best_config), seed=final_seed)
        if fitted_model is None:
            fitted_model = UnifiedPUModel(tuned.best_config, model_fit, warm_start_cache=refit_warm_starts).fit(
                development, exposure=development_e, seed=final_seed,
            )
            if refit_store is not None:
                metadata, arrays = _state_to_checkpoint(fitted_model)
                refit_store.save_complete(refit_key, data_fingerprint, {"fitted": metadata}, arrays)
            refit_cache_status = "fitted"
        final_fit_cache[refit_key] = fitted_model
        emit("refit_complete", base_model.name, cache_status=refit_cache_status,
             elapsed_seconds=perf_counter() - refit_started,
             config=asdict(tuned.best_config), seed=final_seed,
             converged=fitted_model.converged, iterations=fitted_model.iterations,
             objective=fitted_model.objective)
        # Preserve the reporting alias while reusing the exact canonical fit.
        fitted_model = replace(fitted_model, config=tuned.best_config)
        latent = fitted_model.predict_proba(
            final_test_array, x_nuisance=final_test_nuisance
        )
        observed = fitted_model.predict_observed(
            final_test_array,
            exposure=test_e,
            x_nuisance=final_test_nuisance,
        )
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
                model_fingerprint,
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
        emit("model_complete", base_model.name, index=model_index, total=len(base_models),
             cache_status="fitted", elapsed_seconds=perf_counter() - model_started,
             resumed=False, summary=result.summary())

    return GridRunResult(
        fingerprint=fingerprint,
        code_version=code_version,
        test_cell_ids=test_ids,
        models={model.name: results[model.name] for model in base_models},
    )
