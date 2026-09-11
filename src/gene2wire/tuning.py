"""Leakage-resistant deterministic hyperparameter tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .checkpoint import (
    AtomicCheckpointStore, canonical_json, experiment_fingerprint,
    sha256_array, sha256_source_tree, unit_key,
)
from .config import FitConfig, ModelConfig, TuningConfig
from .candidate_design import select_balanced_candidates
from .data import DatasetBundle
from .metrics import masked_log_loss
from .models import DirectWarmStartCache, UnifiedPUModel, canonical_model_config, model_identity
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
    """Enumerate the relevant grid, with optional exact endpoints and budget cap.

    A finite budget selects a deterministic, balanced subset of each
    structural family.  It is a bounded Cartesian search, not a staged fit.
    """

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

    if base.kind == "joint" and tuning.include_endpoints:
        # Exact endpoints are independent optimization structures, not zero or
        # very large residual-penalty approximations.
        direct = base.with_updates(kind="direct", rank=0, shared_l2=0.0)
        candidates.extend(full_joint_candidates(direct, replace(tuning, candidate_budget=None)))
        if any(rank > 0 for rank in tuning.ranks):
            lowrank = base.with_updates(kind="lowrank", rank=max(1, base.rank), residual_l2=0.0)
            candidates.extend(full_joint_candidates(lowrank, replace(tuning, candidate_budget=None)))
        candidates = [canonical_model_config(config) for config in candidates]

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
    return _budget_candidates(tuple(unique.values()), tuning.candidate_budget)


def _budget_candidates(
    candidates: tuple[ModelConfig, ...], budget: int | None
) -> tuple[ModelConfig, ...]:
    """Allocate an equal initial share to each structure, then use spare slots.

    The order is based only on the declared grid. No labels or scores are used.
    """

    if budget is None or len(candidates) <= budget:
        return candidates
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate.kind, []).append(candidate)
    if budget < len(groups):
        raise ValueError("candidate_budget is too small to retain every requested structure")
    allocation = {kind: 0 for kind in groups}
    for _ in range(budget):
        available = [kind for kind in groups if allocation[kind] < len(groups[kind])]
        chosen = min(available, key=lambda kind: (allocation[kind], list(groups).index(kind)))
        allocation[chosen] += 1
    chosen = []
    for kind, values in groups.items():
        chosen.extend(select_balanced_candidates(values, allocation[kind]))
    return tuple(chosen)


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
    if base.kind == "joint" and tuning.include_endpoints:
        candidates = [canonical_model_config(config) for config in candidates]
        candidates.append(base.with_updates(
            kind="direct", rank=0, shared_l2=0.0,
            residual_l2=tuning.anchor_residual_l2,
            target_l2=tuning.anchor_target_l2 if base.use_target_features else 0.0,
        ))
        candidates.extend(base.with_updates(
            kind="lowrank", rank=rank, shared_l2=tuning.anchor_shared_l2,
            residual_l2=0.0,
            target_l2=tuning.anchor_target_l2 if base.use_target_features else 0.0,
        ) for rank in tuning.ranks if rank > 0)
    if not candidates:
        raise ValueError("staged rank grid has no valid candidates")
    unique = {canonical_json(model_identity(candidate)): candidate for candidate in candidates}
    return tuple(unique.values())


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
        candidate_budget=None,
        include_endpoints=False,
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
    candidate_cache: MutableMapping[str, TrialResult] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    required_endpoints: Sequence[ModelConfig] | None = None,
) -> TuningResult:
    """Tune only on explicitly supplied train and validation bundles.

    Reference truth is stripped before every fit.  The selection score is the
    observed-label log loss for ``q=e*p`` on measured validation entries.
    ``on_progress`` receives stage-specific cache inventories and timed candidate
    events. Its diagnostics do not participate in fitting or cache identity.

    When ``required_endpoints`` is supplied for a Joint model, the bounded
    ``candidate_budget`` applies only to genuine ``kind='joint'`` candidates.
    The supplied direct and low-rank configurations are evaluated as mandatory
    boundary candidates in addition to that budget. Their statistical fits are
    eligible for the shared candidate/refit caches, so carrying a tuned
    standalone winner into Joint does not duplicate optimization work.
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

    required = tuple(required_endpoints or ())
    if required:
        if base_model.kind != "joint":
            raise ValueError("required_endpoints are supported only for Joint tuning")
        normalized_required = []
        seen_required = set()
        for endpoint in required:
            if not isinstance(endpoint, ModelConfig):
                raise TypeError("required_endpoints must contain ModelConfig values")
            config = canonical_model_config(endpoint).with_updates(name=base_model.name)
            if config.kind not in {"direct", "lowrank"}:
                raise ValueError("required Joint endpoints must be direct or lowrank")
            if (config.pu != base_model.pu
                    or config.use_target_features != base_model.use_target_features):
                raise ValueError("required Joint endpoints must match PU and target-feature settings")
            identity = canonical_json(model_identity(config))
            if identity not in seen_required:
                normalized_required.append(config)
                seen_required.add(identity)
        required = tuple(normalized_required)
    # Also protect standalone tune_model callers: a supplied cache namespace is
    # never permission to reuse scores for different labels or exposures.
    hashes = {}
    for phase, bundle, exposure in (
        ("train", train_safe, train_exposure),
        ("validation", validation_safe, validation_exposure),
    ):
        for key in ("X_cell", "S_observed", "W_measured"):
            hashes[f"{phase}_{key}"] = sha256_array(getattr(bundle, key))
        hashes[f"{phase}_exposure"] = sha256_array(np.asarray(exposure, dtype=float))
        hashes[f"{phase}_cell_ids"] = sha256_array(np.asarray(bundle.cell_ids, dtype=str))
        hashes[f"{phase}_target_ids"] = sha256_array(np.asarray(bundle.target_ids, dtype=str))
        if bundle.Y_target is not None:
            hashes[f"{phase}_Y_target"] = sha256_array(bundle.Y_target)
    cache_fingerprint = experiment_fingerprint(
        {"caller_fingerprint": checkpoint_fingerprint, "context": context},
        input_hashes=hashes, code_version="canonical-tuning-0908",
        seeds={"seed": int(seed)},
        source_hash=sha256_source_tree(Path(__file__).resolve().parent),
    )

    memory = {} if candidate_cache is None else candidate_cache

    def emit(event: str, **details: Any) -> None:
        if on_progress is not None:
            on_progress({"event": event, "model": base_model.name, **details})

    def evaluate(candidates: Iterable[ModelConfig], stage: str) -> tuple[TrialResult, ...]:
        stage_trials: list[TrialResult] = []
        # Inspect each candidate using the exact lookup used for evaluation, so
        # an incompatible file never appears in the resume count. Retain these
        # small TrialResult records to avoid reading checkpoint files twice.
        prepared = []
        for requested_config in candidates:
            config = canonical_model_config(requested_config)
            identity = model_identity(config)
            # Matched PU/non-PU fits use the same starting randomness.  The
            # likelihood, not an unrelated seed, is their experimental contrast.
            trial_seed = stable_seed(
                seed, "model_initialization", config.kind, config.rank,
                config.use_target_features,
            )
            coordinates = {
                **context,
                "candidate": identity,
                "fit": asdict(fit_config),
                "trial_seed": trial_seed,
            }
            checkpoint_key = unit_key(**coordinates)
            # Memory is scoped to fixed data by the runner. Including the
            # supplied fingerprint also protects explicitly shared caller caches.
            cache_key = canonical_json([cache_fingerprint, coordinates])
            previous = memory.get(cache_key)
            cache_status = "memory" if previous is not None else "pending"
            if previous is None and checkpoint_store is not None:
                cached = checkpoint_store.load(checkpoint_key, fingerprint=cache_fingerprint)
                if cached is not None:
                    payload = cached.get("payload")
                    if not isinstance(payload, Mapping) or not isinstance(payload.get("trial"), Mapping):
                        raise ValueError("checkpoint payload does not contain a trial")
                    previous = TrialResult.from_dict(payload["trial"])
                    cache_status = "checkpoint"
            if previous is not None:
                if (
                    model_identity(previous.config) != identity
                    or previous.seed != trial_seed
                    or not np.isfinite(previous.validation_loss)
                ):
                    raise ValueError("checkpointed trial does not match requested candidate")
            prepared.append((config, trial_seed, checkpoint_key, cache_key, previous, cache_status))

        memory_cached = sum(item[-1] == "memory" for item in prepared)
        checkpoint_cached = sum(item[-1] == "checkpoint" for item in prepared)
        cached_count = memory_cached + checkpoint_cached
        emit("candidate_inventory", stage=stage, strategy=tuning.strategy,
             total=len(prepared), cached=cached_count, memory_cached=memory_cached,
             checkpoint_cached=checkpoint_cached, pending=len(prepared) - cached_count)
        stage_started = perf_counter()
        for stage_index, (config, trial_seed, checkpoint_key, cache_key, previous,
                          cache_status) in enumerate(prepared, start=1):
            trial_index = len(trials)
            candidate_started = perf_counter()
            details = {"stage": stage, "index": stage_index, "total": len(prepared),
                       "trial_index": trial_index, "config": asdict(config)}
            emit("candidate_start", **details, cache_status=cache_status)
            if previous is not None:
                trial = replace(previous, stage=stage, index=trial_index, config=config)
            else:
                fitted = UnifiedPUModel(config, fit_config, warm_start_cache=warm_start_cache).fit(
                    train_safe, exposure=train_exposure, seed=trial_seed
                )
                q_validation = fitted.predict_observed(validation_safe.X_cell, exposure=validation_exposure)
                score = masked_log_loss(validation_safe.S_observed, q_validation, validation_safe.W_measured)
                trial = TrialResult(
                    stage=stage, index=trial_index, config=config,
                    validation_loss=score, converged=fitted.converged,
                    iterations=fitted.iterations, seed=trial_seed,
                )
                if checkpoint_store is not None:
                    checkpoint_store.save_complete(checkpoint_key, cache_fingerprint, {"trial": trial.to_dict()})
                cache_status = "fitted"
            memory[cache_key] = trial
            trials.append(trial)
            stage_trials.append(trial)
            if on_trial is not None:
                on_trial(trial)
            emit("candidate_complete", **details, cache_status=cache_status,
                 elapsed_seconds=perf_counter() - candidate_started,
                 validation_loss=trial.validation_loss, converged=trial.converged,
                 iterations=trial.iterations, seed=trial.seed)
        emit("stage_complete", stage=stage, total=len(stage_trials),
             elapsed_seconds=perf_counter() - stage_started)
        return tuple(stage_trials)

    if required:
        # The incumbent endpoint policy is deliberately independent of the
        # enumeration used by the legacy include_endpoints grid.  Positive
        # ranks are genuine Joint configurations; rank zero is a direct alias
        # and is therefore not counted against the native Joint budget.
        positive_ranks = tuple(rank for rank in tuning.ranks if rank > 0)
        native_trials: tuple[TrialResult, ...]
        if not positive_ranks:
            native_trials = ()
        else:
            # Keep the declared search strategy.  In particular, a staged
            # Joint search gets a rank stage followed by a penalty stage; the
            # only difference from the ordinary path is that its budget is
            # reserved for genuine Joint candidates and exact standalone
            # winners are appended afterwards.
            native_tuning = replace(
                tuning, ranks=positive_ranks, include_endpoints=False
            )
            if tuning.strategy == "staged_rank_l2":
                first = _stage_one_candidates(base_model, native_tuning)
                if tuning.candidate_budget is not None:
                    first_budget = min(
                        tuning.candidate_budget,
                        max(1, tuning.candidate_budget // 2),
                    )
                    first = _budget_candidates(first, first_budget)
                rank_trials = evaluate(first, "rank")
                selected = _winner(rank_trials).config
                second = _stage_two_candidates(
                    selected, native_tuning, selected.rank
                )
                done = {
                    canonical_json(model_identity(trial.config))
                    for trial in rank_trials
                }
                second = tuple(
                    candidate
                    for candidate in second
                    if canonical_json(model_identity(candidate)) not in done
                )
                if tuning.candidate_budget is not None:
                    remaining = tuning.candidate_budget - len(rank_trials)
                    second = (
                        ()
                        if remaining <= 0
                        else _budget_candidates(second, remaining)
                    )
                native_trials = (*rank_trials, *evaluate(second, "penalty"))
            else:
                native = full_joint_candidates(base_model, native_tuning)
                native_trials = evaluate(native, "joint_grid")
        endpoint_trials = evaluate(required, "inherited_endpoint")
        selectable = (*native_trials, *endpoint_trials)
    elif tuning.strategy == "full_joint" or base_model.kind == "direct":
        selectable = evaluate(full_joint_candidates(base_model, tuning), "joint_grid")
    else:
        first = _stage_one_candidates(base_model, tuning)
        if tuning.candidate_budget is not None:
            kinds = len({candidate.kind for candidate in first})
            first_budget = min(tuning.candidate_budget, max(kinds, tuning.candidate_budget // 2))
            first = _budget_candidates(first, first_budget)
        rank_trials = evaluate(first, "rank")
        selected = _winner(rank_trials).config
        second = _stage_two_candidates(selected, tuning, selected.rank)
        done = {canonical_json(model_identity(trial.config)) for trial in rank_trials}
        second = tuple(candidate for candidate in second if canonical_json(model_identity(candidate)) not in done)
        if tuning.candidate_budget is not None:
            remaining = tuning.candidate_budget - len(rank_trials)
            second = () if remaining <= 0 else _budget_candidates(second, remaining)
        # Rank-stage anchor fits are real candidates and already used selection
        # information. Retaining them avoids losing an exact endpoint in stage 2.
        selectable = (*rank_trials, *evaluate(second, "penalty"))

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
