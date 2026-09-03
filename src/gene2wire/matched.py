"""Matched-budget selection for direct and structured Adam models.

The orchestration in this module is deliberately data agnostic.  It receives
already processed train/validation arrays plus test features, never outer-test
labels or evaluation reference truth.  Each tuning candidate and each complete
model unit is atomically checkpointed for runtime-resume safety.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .adaptive import (
    AdaptiveFitConfig,
    AdaptiveFittedModel,
    AdaptiveModelConfig,
    AdaptivePUModel,
    per_target_validation_metrics,
)
from .checkpoint import (
    AtomicArrayCheckpointStore,
    canonical_json,
    experiment_fingerprint,
    sha256_array,
    unit_key,
)
from .data import DatasetBundle
from .models import _exposure_matrix


MATCHED_API_VERSION = "0.3.0"


@dataclass(frozen=True)
class MatchedModelConfig:
    """Named model family supplied to :func:`run_matched_model_grid`."""

    name: str
    kind: str
    pu: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be nonempty")
        if self.kind not in {"direct", "lowrank", "joint"}:
            raise ValueError("kind must be direct, lowrank, or joint")


@dataclass(frozen=True)
class MatchedTuningConfig:
    """Fair direct/rank/refinement grids for matched model comparisons."""

    direct_l2: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    learning_rates: tuple[float, ...] = (0.005, 0.01)
    ranks: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    shared_l2: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3)
    residual_l2: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)
    epsilon: float = 0.0
    loss_se_multiplier: float = 1.0
    auprc_se_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.direct_l2 or not self.learning_rates or not self.ranks:
            raise ValueError("direct_l2, learning_rates, and ranks must be nonempty")
        if not self.shared_l2 or not self.residual_l2:
            raise ValueError("both structured refinement grids must be nonempty")
        if len(set(self.ranks)) != len(self.ranks) or any(rank < 1 for rank in self.ranks):
            raise ValueError("ranks must be unique positive integers")
        for name in ("direct_l2", "learning_rates", "shared_l2", "residual_l2"):
            values = tuple(float(value) for value in getattr(self, name))
            if any(not np.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{name} must contain finite nonnegative values")
        if any(value <= 0 for value in self.learning_rates):
            raise ValueError("learning_rates must be positive")
        if not 0 <= self.epsilon < 1:
            raise ValueError("epsilon must be in [0, 1)")
        if self.loss_se_multiplier < 0 or self.auprc_se_multiplier < 0:
            raise ValueError("gate SE multipliers must be nonnegative")

    @property
    def direct_budget(self) -> int:
        return len(self.direct_l2) * len(self.learning_rates)

    def structured_budget(self, kind: str) -> int:
        refinement = self.shared_l2 if kind == "lowrank" else self.residual_l2
        return 1 + len(self.ranks) + len(refinement)


@dataclass(frozen=True)
class AdaptiveTrialResult:
    candidate_id: int
    stage: str
    config: Mapping[str, Any]
    validation_loss: float
    validation_auprc: float | None
    selection_epochs: int
    selected_checkpoint_epoch: int
    initialization_error: float | None
    resumed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "config": dict(self.config),
            "validation_loss": self.validation_loss,
            "validation_auprc": self.validation_auprc,
            "selection_epochs": self.selection_epochs,
            "selected_checkpoint_epoch": self.selected_checkpoint_epoch,
            "initialization_error": self.initialization_error,
            "resumed": self.resumed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdaptiveTrialResult":
        return cls(
            candidate_id=int(value["candidate_id"]),
            stage=str(value["stage"]),
            config=dict(value["config"]),
            validation_loss=float(value["validation_loss"]),
            validation_auprc=(
                None if value.get("validation_auprc") is None else float(value["validation_auprc"])
            ),
            selection_epochs=int(value["selection_epochs"]),
            selected_checkpoint_epoch=int(value["selected_checkpoint_epoch"]),
            initialization_error=(
                None if value.get("initialization_error") is None else float(value["initialization_error"])
            ),
            resumed=bool(value.get("resumed", False)),
        )


@dataclass(frozen=True)
class MatchedModelRunResult:
    model: MatchedModelConfig
    selected_config: Mapping[str, Any]
    selection: Mapping[str, Any]
    trials: tuple[AdaptiveTrialResult, ...]
    fitted: AdaptiveFittedModel
    latent_probability: np.ndarray
    observed_probability: np.ndarray
    resumed: bool

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model.name,
            "kind": self.model.kind,
            "pu": self.model.pu,
            **dict(self.selected_config),
            **dict(self.selection),
            "tuning_trials": len(self.trials),
            "resumed": self.resumed,
        }

    def trial_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "model": self.model.name,
                **trial.to_dict(),
                **dict(trial.config),
            }
            for trial in self.trials
        ]


@dataclass(frozen=True)
class MatchedGridRunResult:
    fingerprint: str
    code_version: str
    test_cell_ids: tuple[str, ...]
    models: Mapping[str, MatchedModelRunResult]

    def summary_rows(self) -> list[dict[str, Any]]:
        return [result.summary() for result in self.models.values()]

    def trial_rows(self) -> list[dict[str, Any]]:
        return [row for result in self.models.values() for row in result.trial_rows()]


@dataclass(frozen=True)
class _CandidateRecord:
    trial: AdaptiveTrialResult
    fitted: AdaptiveFittedModel


def _require_reference_free(name: str, bundle: DatasetBundle) -> None:
    if bundle.Z_reference is not None or bundle.reference_mask is not None:
        raise ValueError(
            f"{name} contains evaluation-only reference truth; call without_reference() "
            "before passing it to run_matched_model_grid"
        )


def _assert_aligned(train: DatasetBundle, validation: DatasetBundle) -> None:
    if train.n_features != validation.n_features:
        raise ValueError("train/validation feature counts differ")
    if train.target_ids != validation.target_ids:
        raise ValueError("train/validation target_ids differ or are permuted")
    if train.feature_blocks != validation.feature_blocks:
        raise ValueError("train/validation feature_blocks differ")
    if (train.Y_target is None) != (validation.Y_target is None):
        raise ValueError("train/validation Y_target availability differs")
    if train.Y_target is not None and not np.array_equal(train.Y_target, validation.Y_target):
        raise ValueError("train/validation Y_target matrices differ or are permuted")


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


def _mean_and_se(values: Any) -> tuple[float | None, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None, None
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None
    return mean, se


def _selection_key(record: _CandidateRecord) -> tuple[float, float]:
    ap = record.trial.validation_auprc
    return (record.trial.validation_loss, -(ap if ap is not None else float("-inf")))


def _direct_candidates(model: MatchedModelConfig, tuning: MatchedTuningConfig) -> list[AdaptiveModelConfig]:
    result: list[AdaptiveModelConfig] = []
    for residual_l2 in tuning.direct_l2:
        for learning_rate in tuning.learning_rates:
            result.append(
                AdaptiveModelConfig(
                    name=model.name,
                    kind="direct",
                    pu=model.pu,
                    rank=0,
                    epsilon=tuning.epsilon,
                    residual_l2=float(residual_l2),
                    learning_rate=float(learning_rate),
                    initialization="random",
                    nested_role="direct grid",
                )
            )
    return result


def _rank_candidates(
    model: MatchedModelConfig,
    parent: AdaptiveModelConfig,
    tuning: MatchedTuningConfig,
) -> list[AdaptiveModelConfig]:
    result = [
        AdaptiveModelConfig(
            name=model.name,
            kind=model.kind,
            pu=model.pu,
            rank=0,
            epsilon=tuning.epsilon,
            residual_l2=parent.residual_l2,
            learning_rate=parent.learning_rate,
            initialization="frozen_matched_direct",
            frozen_direct=True,
            nested_role="rank-0 frozen matched direct",
        )
    ]
    initialization = "direct_svd" if model.kind == "lowrank" else "svd_shared_exact_remainder"
    for rank in tuning.ranks:
        result.append(
            AdaptiveModelConfig(
                name=model.name,
                kind=model.kind,
                pu=model.pu,
                rank=int(rank),
                epsilon=tuning.epsilon,
                learning_rate=parent.learning_rate,
                initialization=initialization,
                nested_role="fair rank screen",
            )
        )
    return result


def _refinement_candidates(
    model: MatchedModelConfig,
    parent: AdaptiveModelConfig,
    rank: int,
    tuning: MatchedTuningConfig,
) -> list[AdaptiveModelConfig]:
    if model.kind == "lowrank":
        return [
            AdaptiveModelConfig(
                name=model.name,
                kind=model.kind,
                pu=model.pu,
                rank=rank,
                epsilon=tuning.epsilon,
                shared_l2=float(value),
                learning_rate=parent.learning_rate,
                initialization="direct_svd",
                nested_role="shared-ridge refinement",
            )
            for value in tuning.shared_l2
        ]
    return [
        AdaptiveModelConfig(
            name=model.name,
            kind=model.kind,
            pu=model.pu,
            rank=rank,
            epsilon=tuning.epsilon,
            residual_l2=float(value),
            learning_rate=parent.learning_rate,
            initialization="svd_shared_exact_remainder",
            nested_role="residual-ridge refinement",
        )
        for value in tuning.residual_l2
    ]


def _pack_fitted(
    fitted: AdaptiveFittedModel,
    *,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata, arrays = fitted.checkpoint_parts()
    return metadata, {f"{prefix}{name}": value for name, value in arrays.items()}


def _evaluate_candidate(
    *,
    model: MatchedModelConfig,
    candidate: AdaptiveModelConfig,
    candidate_id: int,
    stage: str,
    train: DatasetBundle,
    validation: DatasetBundle,
    train_exposure: np.ndarray,
    validation_exposure: np.ndarray,
    validation_target_mask: np.ndarray,
    fit: AdaptiveFitConfig,
    seed: int,
    direct_reference: AdaptiveFittedModel | None,
    store: AtomicArrayCheckpointStore | None,
    fingerprint: str,
    context: Mapping[str, Any],
    on_candidate: Callable[[str, int, str], None] | None,
) -> _CandidateRecord:
    key = unit_key(
        task="adaptive_candidate",
        **context,
        model=model.name,
        stage=stage,
        candidate_id=candidate_id,
    )
    cached = None if store is None else store.load(key, fingerprint)
    if cached is not None:
        payload = cached["payload"]
        if payload.get("candidate") != asdict(candidate):
            raise ValueError("candidate checkpoint configuration does not match")
        fitted = AdaptiveFittedModel.from_checkpoint_parts(
            payload["fitted"], cached["arrays"], prefix="fitted__"
        )
        trial = AdaptiveTrialResult.from_dict(payload["trial"])
        trial = AdaptiveTrialResult(**{**trial.__dict__, "resumed": True})
        if on_candidate is not None:
            on_candidate(model.name, candidate_id, "resumed")
        return _CandidateRecord(trial=trial, fitted=fitted)

    if on_candidate is not None:
        on_candidate(model.name, candidate_id, "started")
    if candidate.frozen_direct:
        if direct_reference is None:
            raise ValueError("frozen candidates require their fitted direct parent")
        fitted = direct_reference
        validation_loss = fitted.best_validation_loss
        validation_auprc = fitted.best_validation_auprc
        initialization_error = 0.0
    else:
        initial_beta = None
        initial_bias = None
        preserve_initial = False
        if candidate.kind in {"lowrank", "joint"}:
            if direct_reference is None:
                raise ValueError("structured candidates require their fitted direct parent")
            initial_beta = direct_reference.effective_beta()
            initial_bias = direct_reference.intercept
            preserve_initial = True
        fitted = AdaptivePUModel(candidate, fit).fit(
            train,
            validation,
            train_exposure=train_exposure,
            validation_exposure=validation_exposure,
            validation_target_mask=validation_target_mask,
            seed=seed,
            initial_beta=initial_beta,
            initial_bias=initial_bias,
            preserve_initial_checkpoint=preserve_initial,
        )
        validation_loss = fitted.best_validation_loss
        validation_auprc = fitted.best_validation_auprc
        initialization_error = fitted.initialization_error
    trial = AdaptiveTrialResult(
        candidate_id=candidate_id,
        stage=stage,
        config=asdict(candidate),
        validation_loss=validation_loss,
        validation_auprc=validation_auprc,
        selection_epochs=fitted.epochs_trained,
        selected_checkpoint_epoch=fitted.best_epoch,
        initialization_error=initialization_error,
        resumed=False,
    )
    if store is not None:
        fitted_metadata, fitted_arrays = _pack_fitted(fitted, prefix="fitted__")
        store.save_complete(
            key,
            fingerprint,
            {
                "model": asdict(model),
                "candidate": asdict(candidate),
                "trial": trial.to_dict(),
                "fitted": fitted_metadata,
            },
            fitted_arrays,
        )
    if on_candidate is not None:
        on_candidate(model.name, candidate_id, "completed")
    return _CandidateRecord(trial=trial, fitted=fitted)


def _gate(
    direct: AdaptiveFittedModel,
    structured: AdaptiveFittedModel,
    validation: DatasetBundle,
    validation_exposure: np.ndarray,
    validation_target_mask: np.ndarray,
    tuning: MatchedTuningConfig,
) -> dict[str, Any]:
    direct_loss, direct_ap = per_target_validation_metrics(
        direct, validation, validation_exposure, validation_target_mask
    )
    structured_loss, structured_ap = per_target_validation_metrics(
        structured, validation, validation_exposure, validation_target_mask
    )
    loss_delta = direct_loss - structured_loss
    ap_delta = structured_ap - direct_ap
    mean_loss_delta, loss_se = _mean_and_se(loss_delta)
    mean_ap_delta, ap_se = _mean_and_se(ap_delta)
    usable_loss = int(np.isfinite(loss_delta).sum())
    usable_ap = int(np.isfinite(ap_delta).sum())
    loss_gate = bool(
        mean_loss_delta is not None
        and loss_se is not None
        and mean_loss_delta > tuning.loss_se_multiplier * loss_se
    )
    if usable_ap == 0:
        ap_gate = True
    elif usable_ap == 1:
        ap_gate = bool(mean_ap_delta is not None and mean_ap_delta >= 0)
    else:
        ap_gate = bool(
            mean_ap_delta is not None
            and ap_se is not None
            and mean_ap_delta >= -tuning.auprc_se_multiplier * ap_se
        )
    return {
        "gate_accepted_structured": bool(loss_gate and ap_gate),
        "gate_se_unit": "eligible validation targets",
        "gate_usable_loss_targets": usable_loss,
        "gate_usable_auprc_targets": usable_ap,
        "gate_loss_improvement": mean_loss_delta,
        "gate_loss_improvement_se": loss_se,
        "gate_macro_auprc_difference": mean_ap_delta,
        "gate_macro_auprc_difference_se": ap_se,
    }


def _save_completed(
    *,
    store: AtomicArrayCheckpointStore | None,
    key: str,
    fingerprint: str,
    result: MatchedModelRunResult,
    test_cell_ids: tuple[str, ...],
    support_model: AdaptiveFittedModel | None,
) -> None:
    if store is None:
        return
    fitted_metadata, fitted_arrays = _pack_fitted(result.fitted, prefix="fitted__")
    arrays = {
        **fitted_arrays,
        "latent_probability": result.latent_probability,
        "observed_probability": result.observed_probability,
    }
    support_metadata = None
    if support_model is not None:
        support_metadata, support_arrays = _pack_fitted(support_model, prefix="support__")
        arrays.update(support_arrays)
    store.save_complete(
        key,
        fingerprint,
        {
            "model": asdict(result.model),
            "selected_config": dict(result.selected_config),
            "selection": dict(result.selection),
            "trials": [trial.to_dict() for trial in result.trials],
            "fitted": fitted_metadata,
            "support_model": support_metadata,
            "test_cell_ids": list(test_cell_ids),
        },
        arrays,
    )


def _load_completed(
    *,
    store: AtomicArrayCheckpointStore | None,
    key: str,
    fingerprint: str,
    model: MatchedModelConfig,
    expected_shape: tuple[int, int],
    test_cell_ids: tuple[str, ...],
) -> tuple[MatchedModelRunResult, AdaptiveFittedModel | None] | None:
    cached = None if store is None else store.load(key, fingerprint)
    if cached is None:
        return None
    payload = cached["payload"]
    arrays = cached["arrays"]
    if payload.get("model") != asdict(model):
        raise ValueError("completed model checkpoint family does not match")
    if tuple(payload.get("test_cell_ids", [])) != test_cell_ids:
        raise ValueError("completed model checkpoint test cell order does not match")
    fitted = AdaptiveFittedModel.from_checkpoint_parts(
        payload["fitted"], arrays, prefix="fitted__"
    )
    support = None
    if payload.get("support_model") is not None:
        support = AdaptiveFittedModel.from_checkpoint_parts(
            payload["support_model"], arrays, prefix="support__"
        )
    latent = np.asarray(arrays["latent_probability"], dtype=np.float32)
    observed = np.asarray(arrays["observed_probability"], dtype=np.float32)
    if latent.shape != expected_shape or observed.shape != expected_shape:
        raise ValueError("completed model checkpoint prediction shape does not match")
    result = MatchedModelRunResult(
        model=model,
        selected_config=dict(payload["selected_config"]),
        selection=dict(payload["selection"]),
        trials=tuple(AdaptiveTrialResult.from_dict(row) for row in payload["trials"]),
        fitted=fitted,
        latent_probability=latent,
        observed_probability=observed,
        resumed=True,
    )
    return result, support


def run_matched_model_grid(
    *,
    train: DatasetBundle,
    validation: DatasetBundle,
    test_X: Any,
    models: Sequence[MatchedModelConfig],
    tuning: MatchedTuningConfig | None = None,
    fit: AdaptiveFitConfig | None = None,
    train_exposure: Any = 1.0,
    validation_exposure: Any = 1.0,
    test_exposure: Any = 1.0,
    validation_target_mask: Any | None = None,
    test_cell_ids: Sequence[Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    unit_context: Mapping[str, Any] | None = None,
    seed: int = 0,
    code_version: str = MATCHED_API_VERSION,
    run_fingerprint: str | None = None,
    on_model: Callable[[str, str], None] | None = None,
    on_candidate: Callable[[str, int, str], None] | None = None,
) -> MatchedGridRunResult:
    """Tune matched direct/structured families and predict an unlabeled test set.

    The final fit deliberately remains on ``train`` with ``validation`` used
    only for early stopping; it does not concatenate the validation rows.  This
    preserves grouped single-validation designs while keeping outer-test truth
    physically outside the core interface.
    """

    _require_reference_free("train", train)
    _require_reference_free("validation", validation)
    _assert_aligned(train, validation)
    if set(train.cell_ids).intersection(validation.cell_ids):
        raise ValueError("train and validation cell_ids overlap")
    specs = tuple(models)
    if not specs or len({model.name for model in specs}) != len(specs):
        raise ValueError("models must be nonempty with unique names")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not code_version:
        raise ValueError("code_version must be nonempty")
    tuning = tuning or MatchedTuningConfig()
    fit = fit or AdaptiveFitConfig()
    for kind in {model.kind for model in specs if model.kind != "direct"}:
        if tuning.structured_budget(kind) != tuning.direct_budget:
            raise ValueError(
                f"{kind} budget {tuning.structured_budget(kind)} does not match "
                f"direct budget {tuning.direct_budget}"
            )
    parent_by_pu: dict[bool, MatchedModelConfig] = {}
    for model in specs:
        if model.kind == "direct":
            if model.pu in parent_by_pu:
                raise ValueError("each PU setting may have only one direct parent")
            parent_by_pu[model.pu] = model
    for model in specs:
        if model.kind != "direct" and model.pu not in parent_by_pu:
            raise ValueError(f"structured model {model.name!r} has no matched direct parent")

    test_array = np.asarray(test_X, dtype=np.float32)
    if test_array.ndim != 2 or test_array.shape[1] != train.n_features:
        raise ValueError("test_X must be 2-D with the fitted feature count")
    if not np.all(np.isfinite(test_array)):
        raise ValueError("test_X contains non-finite values")
    test_ids = (
        tuple(f"test_{index}" for index in range(len(test_array)))
        if test_cell_ids is None
        else tuple(str(value) for value in test_cell_ids)
    )
    if len(test_ids) != len(test_array) or len(set(test_ids)) != len(test_ids):
        raise ValueError("test_cell_ids must be unique and align with test_X")
    if set(test_ids).intersection((*train.cell_ids, *validation.cell_ids)):
        raise ValueError("outer-test cell IDs overlap train or validation")

    train_e = _exposure_matrix(train_exposure, train.S_observed.shape)
    validation_e = _exposure_matrix(validation_exposure, validation.S_observed.shape)
    test_e = _exposure_matrix(test_exposure, (len(test_array), train.n_targets))
    eligible = (
        np.ones(train.n_targets, dtype=bool)
        if validation_target_mask is None
        else np.asarray(validation_target_mask, dtype=bool)
    )
    if eligible.shape != (train.n_targets,) or not np.any(eligible):
        raise ValueError("validation_target_mask must select at least one aligned target")

    context = dict(unit_context or {})
    reserved = {"task", "model", "stage", "candidate_id"}
    if reserved.intersection(context):
        raise ValueError(f"unit_context uses reserved keys: {sorted(reserved.intersection(context))}")
    if any(not isinstance(key, str) or not key for key in context):
        raise ValueError("unit_context keys must be nonempty strings")
    semantic_config = {
        "models": [asdict(model) for model in specs],
        "tuning": asdict(tuning),
        "fit": asdict(fit),
        "unit_context": context,
        "final_training_rows": "train_only",
    }
    input_hashes = {
        "train": _bundle_hash(train),
        "validation": _bundle_hash(validation),
        "test_X": sha256_array(test_array),
        "test_cell_ids": sha256_array(np.asarray(test_ids, dtype=str)),
        "train_exposure": sha256_array(train_e),
        "validation_exposure": sha256_array(validation_e),
        "test_exposure": sha256_array(test_e),
        "validation_target_mask": sha256_array(eligible.astype(np.uint8)),
    }
    fingerprint = run_fingerprint or experiment_fingerprint(
        semantic_config,
        input_hashes=input_hashes,
        code_version=code_version,
        seeds={"base_seed": seed},
    )
    if not fingerprint:
        raise ValueError("run_fingerprint must be None or nonempty")

    candidate_store = None
    model_store = None
    if checkpoint_dir is not None:
        root = Path(checkpoint_dir)
        candidate_store = AtomicArrayCheckpointStore(root / "candidates")
        model_store = AtomicArrayCheckpointStore(root / "models")

    execution_order = sorted(
        specs,
        key=lambda model: (model.pu, {"direct": 0, "lowrank": 1, "joint": 2}[model.kind]),
    )
    results: dict[str, MatchedModelRunResult] = {}
    validation_direct: dict[bool, AdaptiveFittedModel] = {}
    final_direct: dict[bool, AdaptiveFittedModel] = {}
    direct_config: dict[bool, AdaptiveModelConfig] = {}
    expected_shape = (len(test_array), train.n_targets)

    for model in execution_order:
        model_key = unit_key(task="completed_adaptive_model", **context, model=model.name)
        cached = _load_completed(
            store=model_store,
            key=model_key,
            fingerprint=fingerprint,
            model=model,
            expected_shape=expected_shape,
            test_cell_ids=test_ids,
        )
        if cached is not None:
            result, support = cached
            results[model.name] = result
            if model.kind == "direct":
                if support is None:
                    raise ValueError("cached direct model is missing validation support model")
                validation_direct[model.pu] = support
                final_direct[model.pu] = result.fitted
                direct_config[model.pu] = AdaptiveModelConfig(**dict(result.selected_config))
            if on_model is not None:
                on_model(model.name, "resumed")
            continue

        if on_model is not None:
            on_model(model.name, "started")
        family_seed = seed + (100_000 if model.pu else 0)
        model_train_e = train_e if model.pu else np.ones_like(train_e)
        model_validation_e = validation_e if model.pu else np.ones_like(validation_e)
        model_test_e = test_e if model.pu else np.ones_like(test_e)

        if model.kind == "direct":
            records: list[_CandidateRecord] = []
            for candidate_id, candidate in enumerate(_direct_candidates(model, tuning)):
                records.append(
                    _evaluate_candidate(
                        model=model,
                        candidate=candidate,
                        candidate_id=candidate_id,
                        stage="direct grid",
                        train=train,
                        validation=validation,
                        train_exposure=model_train_e,
                        validation_exposure=model_validation_e,
                        validation_target_mask=eligible,
                        fit=fit,
                        seed=family_seed + 10_000 * (candidate_id + 1) + 1,
                        direct_reference=None,
                        store=candidate_store,
                        fingerprint=fingerprint,
                        context=context,
                        on_candidate=on_candidate,
                    )
                )
            selected_record = min(records, key=_selection_key)
            selected_candidate = AdaptiveModelConfig(**dict(selected_record.trial.config))
            final_model = AdaptivePUModel(selected_candidate, fit).fit(
                train,
                validation,
                train_exposure=model_train_e,
                validation_exposure=model_validation_e,
                validation_target_mask=eligible,
                seed=family_seed + 800_000,
            )
            selection: dict[str, Any] = {
                "validation_observed_log_loss": selected_record.trial.validation_loss,
                "validation_observed_macro_auprc": selected_record.trial.validation_auprc,
                "final_refit_epochs": final_model.epochs_trained,
                "gate_accepted_structured": False,
                "frozen_direct_fallback_used": False,
                "max_frozen_prediction_difference": None,
                **final_model.diagnostics(),
            }
            validation_direct[model.pu] = selected_record.fitted
            final_direct[model.pu] = final_model
            direct_config[model.pu] = selected_candidate
            support_model = selected_record.fitted
        else:
            parent_validation = validation_direct[model.pu]
            parent_final = final_direct[model.pu]
            parent_config = direct_config[model.pu]
            records = []
            rank_candidates = _rank_candidates(model, parent_config, tuning)
            for candidate_id, candidate in enumerate(rank_candidates):
                records.append(
                    _evaluate_candidate(
                        model=model,
                        candidate=candidate,
                        candidate_id=candidate_id,
                        stage="rank screen",
                        train=train,
                        validation=validation,
                        train_exposure=model_train_e,
                        validation_exposure=model_validation_e,
                        validation_target_mask=eligible,
                        fit=fit,
                        seed=family_seed + 10_000 * (candidate_id + 1) + 1,
                        direct_reference=parent_validation,
                        store=candidate_store,
                        fingerprint=fingerprint,
                        context=context,
                        on_candidate=on_candidate,
                    )
                )
            nonzero = [record for record in records if int(record.trial.config["rank"]) > 0]
            best_rank = int(min(nonzero, key=_selection_key).trial.config["rank"])
            refinements = _refinement_candidates(model, parent_config, best_rank, tuning)
            for candidate_id, candidate in enumerate(refinements, start=len(records)):
                records.append(
                    _evaluate_candidate(
                        model=model,
                        candidate=candidate,
                        candidate_id=candidate_id,
                        stage="ridge refinement",
                        train=train,
                        validation=validation,
                        train_exposure=model_train_e,
                        validation_exposure=model_validation_e,
                        validation_target_mask=eligible,
                        fit=fit,
                        seed=family_seed + 10_000 * (candidate_id + 1) + 1,
                        direct_reference=parent_validation,
                        store=candidate_store,
                        fingerprint=fingerprint,
                        context=context,
                        on_candidate=on_candidate,
                    )
                )
            trainable = [record for record in records if not record.trial.config["frozen_direct"]]
            best_structured = min(trainable, key=_selection_key)
            structured_config = AdaptiveModelConfig(**dict(best_structured.trial.config))
            structured_refit = AdaptivePUModel(structured_config, fit).fit(
                train,
                validation,
                train_exposure=model_train_e,
                validation_exposure=model_validation_e,
                validation_target_mask=eligible,
                seed=family_seed + 900_000,
                initial_beta=parent_final.effective_beta(),
                initial_bias=parent_final.intercept,
                preserve_initial_checkpoint=True,
            )
            gate = _gate(
                parent_final,
                structured_refit,
                validation,
                model_validation_e,
                eligible,
                tuning,
            )
            if gate["gate_accepted_structured"]:
                selected_record = best_structured
                final_model = structured_refit
                frozen_difference = None
            else:
                selected_record = records[0]
                final_model = parent_final
                parent_latent = parent_final.predict_proba(train.X_cell)
                fallback_latent = final_model.predict_proba(train.X_cell)
                parent_observed = parent_final.predict_observed(train.X_cell, model_train_e)
                fallback_observed = final_model.predict_observed(train.X_cell, model_train_e)
                frozen_difference = float(
                    max(
                        np.max(np.abs(parent_latent - fallback_latent)),
                        np.max(np.abs(parent_observed - fallback_observed)),
                    )
                )
                if frozen_difference != 0.0:
                    raise RuntimeError("rejected structured model did not reuse direct parent exactly")
            selected_candidate = AdaptiveModelConfig(**dict(selected_record.trial.config))
            selection = {
                "validation_observed_log_loss": selected_record.trial.validation_loss,
                "validation_observed_macro_auprc": selected_record.trial.validation_auprc,
                "final_refit_epochs": final_model.epochs_trained,
                **gate,
                "frozen_direct_fallback_used": not gate["gate_accepted_structured"],
                "max_frozen_prediction_difference": frozen_difference,
                "initialization_reconstruction_error": final_model.initialization_error,
                **final_model.diagnostics(),
            }
            support_model = None

        latent = final_model.predict_proba(test_array)
        observed = final_model.predict_observed(test_array, model_test_e)
        result = MatchedModelRunResult(
            model=model,
            selected_config=asdict(selected_candidate),
            selection=selection,
            trials=tuple(record.trial for record in records),
            fitted=final_model,
            latent_probability=latent,
            observed_probability=observed,
            resumed=False,
        )
        _save_completed(
            store=model_store,
            key=model_key,
            fingerprint=fingerprint,
            result=result,
            test_cell_ids=test_ids,
            support_model=support_model,
        )
        results[model.name] = result
        if on_model is not None:
            on_model(model.name, "completed")

    ordered_results = {model.name: results[model.name] for model in specs}
    return MatchedGridRunResult(
        fingerprint=fingerprint,
        code_version=code_version,
        test_cell_ids=test_ids,
        models=ordered_results,
    )


__all__ = [
    "AdaptiveTrialResult",
    "MATCHED_API_VERSION",
    "MatchedGridRunResult",
    "MatchedModelConfig",
    "MatchedModelRunResult",
    "MatchedTuningConfig",
    "run_matched_model_grid",
]
