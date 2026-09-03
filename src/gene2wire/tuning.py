"""Leakage-resistant deterministic hyperparameter tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .checkpoint import AtomicCheckpointStore, unit_key
from .config import FitConfig, ModelConfig, TuningConfig
from .data import DatasetBundle
from .metrics import masked_log_loss
from .models import DirectWarmStartCache, UnifiedPUModel
from .seeds import stable_seed


@dataclass(frozen=True)
class TrialResult:
    stage: str
    index: int
    config: ModelConfig
    validation_loss: float
    converged: bool
    iterations: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialResult":
        required = {
            "stage",
            "index",
            "config",
            "validation_loss",
            "converged",
            "iterations",
            "seed",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"checkpointed trial is missing fields: {sorted(missing)}")
        return cls(
            stage=str(value["stage"]),
            index=int(value["index"]),
            config=ModelConfig(**dict(value["config"])),
            validation_loss=float(value["validation_loss"]),
            converged=bool(value["converged"]),
            iterations=int(value["iterations"]),
            seed=int(value["seed"]),
        )


@dataclass(frozen=True)
class TuningResult:
    best_config: ModelConfig
    best_validation_loss: float
    trials: tuple[TrialResult, ...]
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_config": asdict(self.best_config),
            "best_validation_loss": self.best_validation_loss,
            "strategy": self.strategy,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TuningResult":
        required = {"best_config", "best_validation_loss", "strategy", "trials"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"checkpointed tuning result is missing fields: {sorted(missing)}")
        trials = value["trials"]
        if not isinstance(trials, (list, tuple)):
            raise ValueError("checkpointed tuning trials must be a sequence")
        return cls(
            best_config=ModelConfig(**dict(value["best_config"])),
            best_validation_loss=float(value["best_validation_loss"]),
            trials=tuple(TrialResult.from_dict(item) for item in trials),
            strategy=str(value["strategy"]),
        )


def full_joint_candidates(base: ModelConfig, tuning: TuningConfig) -> tuple[ModelConfig, ...]:
    """Enumerate the complete relevant Cartesian grid, without duplicates."""

    candidates: list[ModelConfig] = []
    target_grid = tuning.target_l2 if base.use_target_features else (base.target_l2,)
    if base.kind == "direct":
        candidates = [
            base.with_updates(
                rank=0,
                shared_l2=0.0,
                residual_l2=residual,
                target_l2=target,
            )
            for residual, target in product(tuning.residual_l2, target_grid)
        ]
    elif base.kind == "lowrank":
        candidates = [
            base.with_updates(
                rank=rank,
                shared_l2=shared,
                residual_l2=0.0,
                target_l2=target,
            )
            for rank, shared, target in product(tuning.ranks, tuning.shared_l2, target_grid)
            if rank >= 1
        ]
    else:
        for rank in tuning.ranks:
            if rank == 0:
                candidates.extend(
                    base.with_updates(
                        rank=0,
                        shared_l2=0.0,
                        residual_l2=residual,
                        target_l2=target,
                    )
                    for residual, target in product(tuning.residual_l2, target_grid)
                )
            else:
                candidates.extend(
                    base.with_updates(
                        rank=rank,
                        shared_l2=shared,
                        residual_l2=residual,
                        target_l2=target,
                    )
                    for shared, residual, target in product(
                        tuning.shared_l2, tuning.residual_l2, target_grid
                    )
                )

    unique: dict[tuple[Any, ...], ModelConfig] = {}
    for config in candidates:
        key = (
            config.kind,
            config.rank,
            config.shared_l2,
            config.residual_l2,
            config.use_target_features,
            config.target_l2,
            config.pu,
        )
        unique[key] = config
    if not unique:
        raise ValueError(f"grid has no valid candidates for model kind {base.kind!r}")
    return tuple(unique.values())


def _stage_one_candidates(base: ModelConfig, tuning: TuningConfig) -> tuple[ModelConfig, ...]:
    if base.kind == "direct":
        return full_joint_candidates(base, tuning)
    candidates = []
    for rank in tuning.ranks:
        if base.kind == "lowrank" and rank == 0:
            continue
        candidates.append(
            base.with_updates(
                rank=rank,
                shared_l2=0.0 if rank == 0 else tuning.anchor_shared_l2,
                residual_l2=0.0 if base.kind == "lowrank" else tuning.anchor_residual_l2,
                target_l2=tuning.anchor_target_l2 if base.use_target_features else base.target_l2,
            )
        )
    if not candidates:
        raise ValueError("staged rank grid has no valid candidates")
    return tuple(candidates)


def _stage_two_candidates(
    base: ModelConfig, tuning: TuningConfig, selected_rank: int
) -> tuple[ModelConfig, ...]:
    narrowed = TuningConfig(
        strategy=tuning.strategy,
        ranks=(selected_rank,),
        shared_l2=tuning.shared_l2,
        residual_l2=tuning.residual_l2,
        target_l2=tuning.target_l2,
        anchor_shared_l2=tuning.anchor_shared_l2,
        anchor_residual_l2=tuning.anchor_residual_l2,
        anchor_target_l2=tuning.anchor_target_l2,
        metric=tuning.metric,
    )
    return full_joint_candidates(base, narrowed)


def _winner(trials: Iterable[TrialResult]) -> TrialResult:
    values = tuple(trials)
    if not values:
        raise ValueError("cannot select from zero trials")

    def tie_key(trial: TrialResult) -> tuple[Any, ...]:
        config = trial.config
        # Prefer simpler rank, then stronger regularization, only after loss.
        return (
            not trial.converged,
            trial.validation_loss,
            config.rank,
            -config.shared_l2,
            -config.residual_l2,
            -config.target_l2,
            trial.index,
        )

    return min(values, key=tie_key)


def _validate_schema_alignment(
    train: DatasetBundle,
    validation: DatasetBundle,
    base_model: ModelConfig,
) -> None:
    if train.n_features != validation.n_features:
        raise ValueError("train and validation feature counts differ")
    if train.n_targets != validation.n_targets:
        raise ValueError("train and validation target counts differ")
    if train.target_ids != validation.target_ids:
        raise ValueError("train and validation target_ids differ or are permuted")
    if train.feature_blocks != validation.feature_blocks:
        raise ValueError("train and validation feature_blocks differ")
    if set(train.cell_ids).intersection(validation.cell_ids):
        raise ValueError("train and validation cell_ids overlap")
    if (train.Y_target is None) != (validation.Y_target is None):
        raise ValueError("train and validation Y_target availability differs")
    if train.Y_target is not None and not np.array_equal(train.Y_target, validation.Y_target):
        raise ValueError("train and validation Y_target matrices differ or are permuted")
    if base_model.use_target_features and train.Y_target is None:
        raise ValueError("model requires aligned Y_target features")


def tune_model(
    train: DatasetBundle,
    validation: DatasetBundle,
    train_exposure: Any,
    validation_exposure: Any,
    base_model: ModelConfig,
    tuning: TuningConfig,
    fit: FitConfig | None = None,
    seed: int = 0,
    on_trial: Callable[[TrialResult], None] | None = None,
    checkpoint_store: AtomicCheckpointStore | None = None,
    checkpoint_fingerprint: str | None = None,
    checkpoint_context: Mapping[str, Any] | None = None,
    warm_start_cache: DirectWarmStartCache | None = None,
) -> TuningResult:
    """Tune only on explicitly supplied train and validation bundles.

    Reference truth is stripped before every fit.  The selection score is the
    observed-label log loss for ``q=e*p`` on measured validation entries.
    """

    _validate_schema_alignment(train, validation, base_model)
    if (checkpoint_store is None) != (checkpoint_fingerprint is None):
        raise ValueError("checkpoint_store and checkpoint_fingerprint must be supplied together")
    if checkpoint_fingerprint is not None and not checkpoint_fingerprint:
        raise ValueError("checkpoint_fingerprint must be nonempty")
    context = dict(checkpoint_context or {})
    if any(not isinstance(key, str) or not key for key in context):
        raise ValueError("checkpoint_context keys must be nonempty strings")
    reserved_context_keys = {"tuning_model", "tuning_stage", "candidate", "trial_seed"}
    if reserved_context_keys.intersection(context):
        raise ValueError(
            "checkpoint_context uses reserved keys: "
            f"{sorted(reserved_context_keys.intersection(context))}"
        )

    fit_config = fit or FitConfig()
    train_safe = train.without_reference()
    validation_safe = validation.without_reference()
    trials: list[TrialResult] = []

    def evaluate(candidates: Iterable[ModelConfig], stage: str) -> tuple[TrialResult, ...]:
        stage_trials: list[TrialResult] = []
        for config in candidates:
            trial_index = len(trials)
            trial_seed = stable_seed(
                seed,
                "model_initialization",
                config.kind,
                config.rank,
                config.use_target_features,
            )
            coordinates = dict(context)
            coordinates.update(
                {
                    "tuning_model": base_model.name,
                    "tuning_stage": stage,
                    "candidate": asdict(config),
                    "trial_seed": trial_seed,
                }
            )
            checkpoint_key = unit_key(**coordinates)
            if checkpoint_store is not None:
                cached = checkpoint_store.load(
                    checkpoint_key, fingerprint=checkpoint_fingerprint
                )
                if cached is not None:
                    payload = cached.get("payload")
                    if not isinstance(payload, Mapping) or not isinstance(
                        payload.get("trial"), Mapping
                    ):
                        raise ValueError("checkpoint payload does not contain a trial")
                    trial = TrialResult.from_dict(payload["trial"])
                    if (
                        trial.stage != stage
                        or trial.index != trial_index
                        or trial.config != config
                        or trial.seed != trial_seed
                        or not np.isfinite(trial.validation_loss)
                    ):
                        raise ValueError("checkpointed trial does not match requested candidate")
                    trials.append(trial)
                    stage_trials.append(trial)
                    if on_trial is not None:
                        on_trial(trial)
                    continue
            fitted = UnifiedPUModel(
                config,
                fit_config,
                warm_start_cache=warm_start_cache,
            ).fit(
                train_safe, exposure=train_exposure, seed=trial_seed
            )
            q_validation = fitted.predict_observed(
                validation_safe.X_cell, exposure=validation_exposure
            )
            score = masked_log_loss(
                validation_safe.S_observed,
                q_validation,
                validation_safe.W_measured,
            )
            trial = TrialResult(
                stage=stage,
                index=trial_index,
                config=config,
                validation_loss=score,
                converged=fitted.converged,
                iterations=fitted.iterations,
                seed=trial_seed,
            )
            trials.append(trial)
            stage_trials.append(trial)
            if checkpoint_store is not None:
                checkpoint_store.save_complete(
                    checkpoint_key,
                    checkpoint_fingerprint,
                    {"trial": trial.to_dict()},
                )
            if on_trial is not None:
                on_trial(trial)
        return tuple(stage_trials)

    if tuning.strategy == "full_joint" or base_model.kind == "direct":
        selectable = evaluate(full_joint_candidates(base_model, tuning), "joint_grid")
    else:
        rank_trials = evaluate(_stage_one_candidates(base_model, tuning), "rank")
        selected_rank = _winner(rank_trials).config.rank
        selectable = evaluate(
            _stage_two_candidates(base_model, tuning, selected_rank),
            "penalty",
        )

    if not any(trial.converged for trial in selectable):
        raise RuntimeError(
            f"all selectable {base_model.name!r} tuning candidates failed to converge"
        )
    best = _winner(selectable)
    return TuningResult(
        best_config=best.config,
        best_validation_loss=best.validation_loss,
        trials=tuple(trials),
        strategy=tuning.strategy,
    )
