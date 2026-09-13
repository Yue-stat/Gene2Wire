"""Leakage-resistant deterministic hyperparameter tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations, product
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
from .models import (
    DirectWarmStartCache,
    FittedModel,
    JointEndpointStarts,
    UnifiedPUModel,
    canonical_model_config,
    model_identity,
)
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
    total_shrinkage: float | None = None
    residual_shared_ratio: float | None = None
    optimizer_message: str = ""

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
        config = ModelConfig(**dict(value["config"]))
        total, ratio = _joint_penalty_summary(config)
        for name, expected in (
            ("total_shrinkage", total),
            ("residual_shared_ratio", ratio),
        ):
            if name not in value:  # Backward-compatible compact checkpoint.
                continue
            supplied = value.get(name)
            if supplied is None and expected is None:
                continue
            if supplied is None or expected is None or not np.isclose(
                float(supplied), expected, rtol=1e-12, atol=0.0
            ):
                raise ValueError(
                    f"checkpointed trial {name} does not match its model config"
                )
        return cls(
            stage=str(value["stage"]),
            index=int(value["index"]),
            config=config,
            validation_loss=float(value["validation_loss"]),
            converged=bool(value["converged"]),
            iterations=int(value["iterations"]),
            seed=int(value["seed"]),
            total_shrinkage=total,
            residual_shared_ratio=ratio,
            optimizer_message=str(value.get("optimizer_message", "")),
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


def _joint_penalty_summary(
    config: ModelConfig,
) -> tuple[float | None, float | None]:
    """Auditable derived coordinates for a genuine Joint configuration."""

    if config.kind != "joint" or config.rank < 1:
        return None, None
    total = float(np.sqrt(config.shared_l2 * config.residual_l2))
    ratio = (
        float(config.residual_l2 / config.shared_l2)
        if config.shared_l2 > 0
        else None
    )
    return total, ratio


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

    unique: dict[str, ModelConfig] = {}
    for config in candidates:
        # Use the complete statistical identity so future fixed model
        # coordinates cannot be accidentally omitted from candidate deduping.
        key = canonical_json(model_identity(config))
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


def _trial_tie_key(trial: TrialResult) -> tuple[Any, ...]:
    """Stable selection order shared by final and rank-stage selection."""

    config = trial.config
    # Prefer a converged fit, then the declared validation objective.  Model
    # simplicity is only a deterministic tie break; it never overrides loss.
    return (
        not trial.converged,
        trial.validation_loss,
        config.rank,
        -config.shared_l2,
        -config.residual_l2,
        -config.target_l2,
        trial.index,
    )


def _top_rank_trials(
    trials: Iterable[TrialResult], count: int = 2
) -> tuple[TrialResult, ...]:
    """Return the best distinct, converged ranks from a rank screen."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("rank retention count must be a positive integer")
    ordered = sorted((trial for trial in trials if trial.converged), key=_trial_tie_key)
    expected = min(count, len({trial.config.rank for trial in trials}))
    selected: list[TrialResult] = []
    seen: set[int] = set()
    for trial in ordered:
        if trial.config.rank in seen:
            continue
        selected.append(trial)
        seen.add(trial.config.rank)
        if len(selected) == count:
            break
    if len(selected) != expected:
        raise RuntimeError(
            "Joint rank screen did not yield the required number of converged "
            f"ranks: expected {expected}, observed {len(selected)}"
        )
    return tuple(selected)


def _penalty_log(value: float, floor: float) -> float:
    """Log coordinate with a separate finite location for an exact zero."""

    return float(np.log10(value)) if value > 0 else floor


def _joint_shrinkage_coordinates(
    candidates: Sequence[ModelConfig],
) -> np.ndarray:
    """Map Joint penalties to rank, total-shrinkage, and ratio coordinates.

    For positive penalties the two search coordinates are

    ``log_total = (log10(shared_l2) + log10(residual_l2)) / 2`` and
    ``log_ratio = log10(residual_l2 / shared_l2)``.

    Thus the change of variables is one-to-one and does not silently alter the
    user's penalty grid.  An exact zero is placed one log decade below the
    smallest positive declared penalty so grids containing zero remain finite.
    """

    values = tuple(candidates)
    if not values:
        raise ValueError("cannot map an empty Joint candidate grid")
    penalties = [
        penalty
        for candidate in values
        for penalty in (candidate.shared_l2, candidate.residual_l2)
        if penalty > 0
    ]
    floor = (float(np.log10(min(penalties))) - 1.0) if penalties else -1.0
    rows = []
    for candidate in values:
        shared = _penalty_log(candidate.shared_l2, floor)
        residual = _penalty_log(candidate.residual_l2, floor)
        target = _penalty_log(candidate.target_l2, floor)
        rows.append(
            (
                float(candidate.rank),
                0.5 * (shared + residual),
                residual - shared,
                target,
            )
        )
    return np.asarray(rows, dtype=np.float64)


def _select_joint_shrinkage_candidates(
    candidates: Sequence[ModelConfig], count: int
) -> tuple[ModelConfig, ...]:
    """Outcome-independent bounded design in total/ratio coordinates.

    The greedy design first balances marginal levels of retained rank, total
    shrinkage, residual/shared ratio, and (when active) target penalty.  Ties
    maximize separation in normalized coordinate space.  Numeric statistical
    identity is the final tie break, so enumeration order, display name, PU
    status, labels, and random seed cannot change the selected grid.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    values = tuple(candidates)
    if not values:
        raise ValueError("cannot select from an empty Joint candidate grid")
    if any(candidate.kind != "joint" or candidate.rank < 1 for candidate in values):
        raise ValueError("shrinkage design requires positive-rank Joint candidates")
    unique = {canonical_json(model_identity(candidate)): candidate for candidate in values}
    if len(unique) != len(values):
        raise ValueError("Joint shrinkage candidate grid contains duplicates")
    ordered = tuple(
        sorted(
            values,
            key=lambda candidate: (
                candidate.rank,
                candidate.shared_l2,
                candidate.residual_l2,
                candidate.target_l2,
            ),
        )
    )
    if count >= len(ordered):
        return ordered

    raw = _joint_shrinkage_coordinates(ordered)
    level_counts: list[np.ndarray] = []
    level_codes: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    active_columns: list[int] = []
    for column in range(raw.shape[1]):
        levels, inverse = np.unique(raw[:, column], return_inverse=True)
        if len(levels) == 1:
            continue
        active_columns.append(column)
        level_counts.append(np.zeros(len(levels), dtype=np.int64))
        level_codes.append(inverse)
        normalized.append((raw[:, column] - levels.min()) / np.ptp(levels))

    # Distinct statistical candidates imply at least one active coordinate.
    points = np.column_stack(normalized)
    pair_counts: list[np.ndarray] = []
    pair_codes: list[np.ndarray] = []
    for left, right in combinations(range(len(level_codes)), 2):
        code = level_codes[left] * len(level_counts[right]) + level_codes[right]
        pair_counts.append(
            np.zeros(
                len(level_counts[left]) * len(level_counts[right]),
                dtype=np.int64,
            )
        )
        pair_codes.append(code)
    total_ratio_count = None
    total_ratio_code = None
    if 1 in active_columns and 2 in active_columns:
        total_index = active_columns.index(1)
        ratio_index = active_columns.index(2)
        total_ratio_code = (
            level_codes[total_index] * len(level_counts[ratio_index])
            + level_codes[ratio_index]
        )
        total_ratio_count = np.zeros(
            len(level_counts[total_index]) * len(level_counts[ratio_index]),
            dtype=np.int64,
        )
    available = np.ones(len(ordered), dtype=bool)
    nearest = np.full(len(ordered), np.inf)
    selected: list[int] = []
    for _ in range(count):
        imbalance = sum(
            2 * frequency[code] + 1
            for frequency, code in zip(level_counts, level_codes)
        )
        eligible = np.flatnonzero(available)
        if total_ratio_count is not None and total_ratio_code is not None:
            shrinkage_imbalance = (
                2 * total_ratio_count[total_ratio_code] + 1
            )
            eligible = eligible[
                shrinkage_imbalance[eligible]
                == shrinkage_imbalance[eligible].min()
            ]
        eligible = eligible[imbalance[eligible] == imbalance[eligible].min()]
        if pair_counts:
            pair_imbalance = sum(
                2 * frequency[code] + 1
                for frequency, code in zip(pair_counts, pair_codes)
            )
            eligible = eligible[
                pair_imbalance[eligible] == pair_imbalance[eligible].min()
            ]
        separation = (
            nearest[eligible]
            if selected
            else -np.sum((points[eligible] - 0.5) ** 2, axis=1)
        )
        best = separation.max()
        chosen = int(
            eligible[
                np.flatnonzero(
                    np.isclose(separation, best, rtol=0.0, atol=1e-12)
                )[0]
            ]
        )
        selected.append(chosen)
        available[chosen] = False
        for frequency, code in zip(level_counts, level_codes):
            frequency[code[chosen]] += 1
        for frequency, code in zip(pair_counts, pair_codes):
            frequency[code[chosen]] += 1
        if total_ratio_count is not None and total_ratio_code is not None:
            total_ratio_count[total_ratio_code[chosen]] += 1
        nearest = np.minimum(
            nearest, np.sum((points - points[chosen]) ** 2, axis=1)
        )
    return tuple(ordered[index] for index in selected)


def _joint_rank_screen_count(rank_count: int, budget: int | None) -> int:
    """Number of declared ranks inspected before adaptive refinement."""

    if rank_count < 1:
        return 0
    if budget is None:
        return rank_count
    minimum = rank_count + min(2, rank_count)
    if budget < minimum:
        raise ValueError(
            "rank_top2_total_ratio candidate_budget is too small: screening all "
            f"{rank_count} positive ranks and reserving refinement for the retained "
            f"ranks requires at least {minimum} native candidates"
        )
    return rank_count


def candidate_search_plan(
    base: ModelConfig, tuning: TuningConfig
) -> dict[str, Any]:
    """Describe the static upper bound of a model's candidate search.

    Endpoint-aware Joint refinement depends on validation-ranked stage-one
    results, so ``len(full_joint_candidates(...))`` is not its execution plan.
    This helper exposes the declared budget and stage bounds without looking at
    outcomes.  ``inherited_endpoint_candidates`` is a maximum: the runner can
    supply fewer endpoints only when the corresponding standalone model is not
    part of the requested model set.
    """

    if (
        base.kind == "joint"
        and tuning.strategy == "rank_top2_total_ratio"
        and not tuning.include_endpoints
    ):
        raise ValueError(
            "rank_top2_total_ratio Joint search requires include_endpoints=True"
        )
    if (
        base.kind != "joint"
        or not tuning.include_endpoints
        or tuning.strategy != "rank_top2_total_ratio"
    ):
        candidates = full_joint_candidates(base, tuning)
        return {
            "adaptive": False,
            "declared_native_budget": tuning.candidate_budget,
            "rank_screen_candidates": 0,
            "retained_ranks": 0,
            "refinement_candidates": len(candidates),
            "maximum_native_candidates": len(candidates),
            "inherited_endpoint_candidates": 0,
            "maximum_total_trials": len(candidates),
            "optimizer_starts_per_native_candidate": 1,
            "maximum_scored_candidate_start_paths": len(candidates),
        }

    positive_ranks = tuple(rank for rank in tuning.ranks if rank > 0)
    rank_count = _joint_rank_screen_count(
        len(positive_ranks), tuning.candidate_budget
    )
    retained = min(2, rank_count)
    target_count = len(tuning.target_l2) if base.use_target_features else 1
    positive_shared = sum(value > 0 for value in tuning.shared_l2)
    positive_residual = sum(value > 0 for value in tuning.residual_l2)
    # One anchor per retained rank can be duplicated by the positive refinement
    # pool.  Subtract it only when both anchor penalties are themselves valid.
    duplicate_anchors = retained * int(
        tuning.anchor_shared_l2 > 0
        and tuning.anchor_residual_l2 > 0
        and (
            not base.use_target_features
            or tuning.anchor_target_l2 in tuning.target_l2
        )
    )
    refinement_pool = max(
        0,
        retained * positive_shared * positive_residual * target_count
        - duplicate_anchors,
    )
    if tuning.candidate_budget is None:
        refinement_count = refinement_pool
    else:
        refinement_count = min(
            refinement_pool, max(0, tuning.candidate_budget - rank_count)
        )
    native = rank_count + refinement_count
    inherited = 2
    return {
        "adaptive": True,
        "declared_native_budget": tuning.candidate_budget,
        "rank_screen_candidates": rank_count,
        "retained_ranks": retained,
        "refinement_candidates": refinement_count,
        "maximum_native_candidates": native,
        "inherited_endpoint_candidates": inherited,
        "maximum_total_trials": native + inherited,
        "optimizer_starts_per_native_candidate": inherited,
        # Counts independently initialized paths for scored candidates. A
        # deterministic continuation after non-convergence is a retry of the
        # same path, and internal endpoint construction is reported separately.
        "maximum_scored_candidate_start_paths": native * inherited + inherited,
    }


def _winner(trials: Iterable[TrialResult]) -> TrialResult:
    values = tuple(trials)
    if not values:
        raise ValueError("cannot select from zero trials")

    return min(values, key=_trial_tie_key)


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
    if train.nuisance_names != validation.nuisance_names:
        raise ValueError("train and validation nuisance schemas differ")


def _fitted_state_identity(fitted: FittedModel) -> dict[str, Any]:
    """Content identity for an optimizer start used by a Joint candidate."""

    arrays = {
        "cell_shared": fitted.cell_shared,
        "target_shared": fitted.target_shared,
        "residual": fitted.residual,
        "target_coeff": fitted.target_coeff,
        "target_features": fitted.target_features,
        "nuisance_coeff": fitted.nuisance_coeff,
        "intercept": fitted.intercept,
    }
    return {
        "config": model_identity(fitted.config),
        "arrays": {
            name: None if value is None else sha256_array(np.asarray(value))
            for name, value in arrays.items()
        },
        "lowrank_feature_indices": fitted.lowrank_feature_indices,
    }


def _joint_start_identity(starts: JointEndpointStarts) -> dict[str, Any]:
    """Checkpoint coordinates for the deterministic two-endpoint recipe."""

    return {
        "recipe": "exact_inner_train_direct_lowrank_v1",
        "direct": _fitted_state_identity(starts.direct),
        "lowrank": _fitted_state_identity(starts.lowrank),
    }


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
    eligible for the shared trial cache. Joint candidates are optimized from
    both converged endpoint solutions fitted on *inner training only*.  Only
    those two endpoint states are retained locally. A checkpoint-only resume
    deterministically rebuilds them; a development-set refit is never used for
    tuning.
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
        if (
            tuning.strategy == "rank_top2_total_ratio"
            and not tuning.include_endpoints
        ):
            raise ValueError(
                "rank_top2_total_ratio Joint tuning requires include_endpoints=True"
            )
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
            if config.nuisance_l2 != base_model.nuisance_l2:
                raise ValueError("required Joint endpoints must match nuisance_l2")
            if config.lowrank_feature_groups:
                raise ValueError(
                    "a grouped low-rank model is not an endpoint of the ordinary Joint family"
                )
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
        if bundle.X_nuisance is not None:
            hashes[f"{phase}_X_nuisance"] = sha256_array(bundle.X_nuisance)
        hashes[f"{phase}_nuisance_names"] = sha256_array(
            np.asarray(bundle.nuisance_names, dtype=str)
        )
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
    fitted_by_identity: dict[str, FittedModel] = {}

    def emit(event: str, **details: Any) -> None:
        if on_progress is not None:
            on_progress({"event": event, "model": base_model.name, **details})

    def evaluate(
        candidates: Iterable[ModelConfig],
        stage: str,
        *,
        joint_endpoint_starts: JointEndpointStarts | None = None,
        retain_fits: bool = False,
    ) -> tuple[TrialResult, ...]:
        stage_trials: list[TrialResult] = []
        start_identity = (
            None
            if joint_endpoint_starts is None
            else _joint_start_identity(joint_endpoint_starts)
        )
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
            if start_identity is not None:
                if config.kind != "joint":
                    raise ValueError(
                        "endpoint multi-start can be used only for Joint candidates"
                    )
                # A local optimum can depend on the exact endpoint solutions.
                # Their content identity therefore belongs in both memory and
                # disk checkpoint coordinates, not merely in diagnostics.
                coordinates["joint_endpoint_starts"] = start_identity
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
            prepared.append(
                (
                    config,
                    trial_seed,
                    checkpoint_key,
                    cache_key,
                    previous,
                    cache_status,
                )
            )

        memory_cached = sum(item[-1] == "memory" for item in prepared)
        checkpoint_cached = sum(item[-1] == "checkpoint" for item in prepared)
        cached_count = memory_cached + checkpoint_cached
        emit("candidate_inventory", stage=stage, strategy=tuning.strategy,
             total=len(prepared), cached=cached_count, memory_cached=memory_cached,
             checkpoint_cached=checkpoint_cached, pending=len(prepared) - cached_count)
        stage_started = perf_counter()
        for stage_index, (
            config,
            trial_seed,
            checkpoint_key,
            cache_key,
            previous,
            cache_status,
        ) in enumerate(prepared, start=1):
            trial_index = len(trials)
            candidate_started = perf_counter()
            details = {"stage": stage, "index": stage_index, "total": len(prepared),
                       "trial_index": trial_index, "config": asdict(config)}
            emit("candidate_start", **details, cache_status=cache_status)
            fitted = None
            if previous is not None:
                trial = replace(previous, stage=stage, index=trial_index, config=config)
                if retain_fits:
                    # Compact trial checkpoints intentionally omit parameter
                    # arrays.  Rebuild the endpoint on the identical inner
                    # training split rather than borrowing the later
                    # train+validation refit, which would leak validation.
                    fitted = UnifiedPUModel(
                        config,
                        fit_config,
                        warm_start_cache=warm_start_cache,
                    ).fit(train_safe, exposure=train_exposure, seed=trial_seed)
                    rebuilt_q = fitted.predict_observed(
                        validation_safe.X_cell,
                        exposure=validation_exposure,
                        x_nuisance=validation_safe.X_nuisance,
                    )
                    rebuilt_loss = masked_log_loss(
                        validation_safe.S_observed,
                        rebuilt_q,
                        validation_safe.W_measured,
                    )
                    if (
                        fitted.converged != trial.converged
                        or not np.isclose(
                            rebuilt_loss,
                            trial.validation_loss,
                            rtol=1e-8,
                            atol=1e-10,
                        )
                    ):
                        raise RuntimeError(
                            "rebuilt inner-training endpoint does not match its "
                            "cached tuning trial"
                        )
                    cache_status = f"{cache_status}+inner_train_refit"
            else:
                fitted = UnifiedPUModel(
                    config,
                    fit_config,
                    warm_start_cache=warm_start_cache,
                    joint_endpoint_starts=joint_endpoint_starts,
                ).fit(train_safe, exposure=train_exposure, seed=trial_seed)
                q_validation = fitted.predict_observed(
                    validation_safe.X_cell,
                    exposure=validation_exposure,
                    x_nuisance=validation_safe.X_nuisance,
                )
                score = masked_log_loss(validation_safe.S_observed, q_validation, validation_safe.W_measured)
                total_shrinkage, residual_shared_ratio = _joint_penalty_summary(
                    config
                )
                trial = TrialResult(
                    stage=stage, index=trial_index, config=config,
                    validation_loss=score, converged=fitted.converged,
                    iterations=fitted.iterations, seed=trial_seed,
                    total_shrinkage=total_shrinkage,
                    residual_shared_ratio=residual_shared_ratio,
                    optimizer_message=fitted.message,
                )
                if checkpoint_store is not None:
                    checkpoint_store.save_complete(checkpoint_key, cache_fingerprint, {"trial": trial.to_dict()})
                cache_status = "fitted"
            if fitted is not None and retain_fits:
                fitted_by_identity[canonical_json(model_identity(config))] = fitted
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

    def evaluate_rank_top2_total_ratio(
        starts: JointEndpointStarts | None,
    ) -> tuple[TrialResult, ...]:
        """Run the genuine-Joint portion of the adaptive bounded search."""

        if base_model.kind != "joint":
            raise ValueError("rank_top2_total_ratio is Joint-specific")
        positive_ranks = tuple(rank for rank in tuning.ranks if rank > 0)
        if not positive_ranks:
            return ()
        adaptive_tuning = replace(
            tuning,
            ranks=positive_ranks,
            include_endpoints=False,
            candidate_budget=None,
        )
        rank_candidates = _stage_one_candidates(base_model, adaptive_tuning)
        if tuning.candidate_budget is not None:
            screen_count = _joint_rank_screen_count(
                len(rank_candidates), tuning.candidate_budget
            )
            if screen_count < len(rank_candidates):
                rank_candidates = select_balanced_candidates(
                    rank_candidates, screen_count
                )
        rank_trials = evaluate(
            rank_candidates,
            "rank",
            joint_endpoint_starts=starts,
        )
        retained = _top_rank_trials(rank_trials, count=2)
        refinement_pool: list[ModelConfig] = []
        for rank_trial in retained:
            refinement_pool.extend(
                _stage_two_candidates(
                    base_model, adaptive_tuning, rank_trial.config.rank
                )
            )
        done = {
            canonical_json(model_identity(trial.config))
            for trial in rank_trials
        }
        refinement = tuple(
            candidate
            for candidate in refinement_pool
            if candidate.shared_l2 > 0
            and candidate.residual_l2 > 0
            and canonical_json(model_identity(candidate)) not in done
        )
        if tuning.candidate_budget is not None:
            remaining = tuning.candidate_budget - len(rank_trials)
            refinement = (
                ()
                if remaining <= 0 or not refinement
                else _select_joint_shrinkage_candidates(
                    refinement, min(remaining, len(refinement))
                )
            )
        elif refinement:
            refinement = _select_joint_shrinkage_candidates(
                refinement, len(refinement)
            )
        penalty_trials = (
            evaluate(
                refinement,
                "penalty",
                joint_endpoint_starts=starts,
            )
            if refinement
            else ()
        )
        result = (*rank_trials, *penalty_trials)
        if (
            tuning.candidate_budget is not None
            and len(result) > tuning.candidate_budget
        ):
            raise RuntimeError(
                "internal error: Joint native candidate budget exceeded"
            )
        return result

    if required:
        # Evaluate and retain exact endpoint fits first.  These fits use only
        # inner-training rows; validation is used solely for their score.  The
        # endpoint trials remain selectable boundary models but never consume
        # the native Joint candidate budget.
        positive_ranks = tuple(rank for rank in tuning.ranks if rank > 0)
        endpoint_trials = evaluate(
            required,
            "inherited_endpoint",
            retain_fits=bool(positive_ranks),
        )
        endpoint_fits: dict[str, FittedModel] = {}
        if positive_ranks:
            for endpoint in required:
                key = canonical_json(model_identity(endpoint))
                fitted_endpoint = fitted_by_identity.get(key)
                if fitted_endpoint is None:
                    raise RuntimeError("failed to retain an inner-training endpoint fit")
                if endpoint.kind in endpoint_fits:
                    raise ValueError(
                        f"multiple tuned {endpoint.kind} endpoints make Joint initialization ambiguous"
                    )
                endpoint_fits[endpoint.kind] = fitted_endpoint
            if set(endpoint_fits) != {"direct", "lowrank"}:
                raise ValueError(
                    "endpoint-aware Joint tuning requires exactly one tuned direct "
                    "and one tuned low-rank endpoint"
                )
            joint_starts = JointEndpointStarts(
                direct=endpoint_fits["direct"],
                lowrank=endpoint_fits["lowrank"],
            )
        else:
            joint_starts = None

        # Positive ranks are genuine Joint structures; rank zero is the exact
        # direct endpoint and is already represented above.  The explicit
        # rank_top2_total_ratio strategy screens ranks before penalties so a
        # 32-candidate budget is not spent on arbitrary Cartesian triples.
        # Legacy strategies retain their advertised enumeration semantics.
        native_trials: tuple[TrialResult, ...]
        if not positive_ranks:
            native_trials = ()
        else:
            native_tuning = replace(
                tuning, ranks=positive_ranks, include_endpoints=False,
            )
            if tuning.strategy == "rank_top2_total_ratio":
                native_trials = evaluate_rank_top2_total_ratio(joint_starts)
            elif tuning.strategy == "staged_rank_l2":
                first = _stage_one_candidates(base_model, native_tuning)
                if tuning.candidate_budget is not None:
                    first_budget = min(
                        tuning.candidate_budget,
                        max(1, tuning.candidate_budget // 2),
                    )
                    first = _budget_candidates(first, first_budget)
                rank_trials = evaluate(
                    first, "rank", joint_endpoint_starts=joint_starts
                )
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
                native_trials = (
                    *rank_trials,
                    *evaluate(
                        second,
                        "penalty",
                        joint_endpoint_starts=joint_starts,
                    ),
                )
            else:
                native = full_joint_candidates(base_model, native_tuning)
                native_trials = evaluate(
                    native,
                    "joint_grid",
                    joint_endpoint_starts=joint_starts,
                )
        if (
            tuning.candidate_budget is not None
            and len(native_trials) > tuning.candidate_budget
        ):
            raise RuntimeError("internal error: Joint native candidate budget exceeded")
        selectable = (*native_trials, *endpoint_trials)
    elif tuning.strategy == "rank_top2_total_ratio":
        if base_model.kind == "joint":
            raise ValueError(
                "rank_top2_total_ratio Joint tuning requires include_endpoints=True "
                "and the tuned direct/low-rank configs in required_endpoints"
            )
        else:
            # The adaptive reparameterization is Joint-specific. Direct and
            # low-rank endpoints retain their bounded Cartesian searches.
            selectable = evaluate(
                full_joint_candidates(base_model, tuning), "joint_grid"
            )
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
