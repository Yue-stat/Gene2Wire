"""Regression gates for the endpoint-aware bounded Joint search."""

from collections import Counter
from itertools import product

import numpy as np
import pytest

from gene2wire import DatasetBundle, FitConfig, ModelConfig, TuningConfig
from gene2wire.tuning import (
    TrialResult,
    _joint_shrinkage_coordinates,
    _select_joint_shrinkage_candidates,
    _top_rank_trials,
    _trial_tie_key,
    candidate_search_plan,
    tune_model,
)


PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _tuning(**changes) -> TuningConfig:
    values = dict(
        strategy="rank_top2_total_ratio",
        ranks=(0, 1, 2, 4, 8, 16),
        shared_l2=PENALTIES,
        residual_l2=PENALTIES,
        target_l2=PENALTIES,
        anchor_shared_l2=1e-3,
        anchor_residual_l2=1e-3,
        anchor_target_l2=1e-3,
        candidate_budget=32,
        include_endpoints=True,
    )
    values.update(changes)
    return TuningConfig(**values)


def test_canonical_joint_plan_reserves_rank_screen_and_top_two_refinement():
    plan = candidate_search_plan(
        ModelConfig(name="PU-Joint", kind="joint", rank=1), _tuning()
    )
    assert plan == {
        "adaptive": True,
        "declared_native_budget": 32,
        "rank_screen_candidates": 5,
        "retained_ranks": 2,
        "refinement_candidates": 27,
        "maximum_native_candidates": 32,
        "inherited_endpoint_candidates": 2,
        "maximum_total_trials": 34,
        "optimizer_starts_per_native_candidate": 2,
        "maximum_scored_candidate_start_paths": 66,
    }


def test_rank_screen_has_no_hidden_cap_and_rejects_an_inadequate_budget():
    ranks = (0, *range(1, 11))
    plan = candidate_search_plan(
        ModelConfig(name="PU-Joint", kind="joint", rank=1),
        _tuning(ranks=ranks, candidate_budget=32),
    )
    assert plan["rank_screen_candidates"] == 10
    assert plan["refinement_candidates"] == 22
    with pytest.raises(ValueError, match="screening all 10 positive ranks"):
        candidate_search_plan(
            ModelConfig(name="PU-Joint", kind="joint", rank=1),
            _tuning(ranks=ranks, candidate_budget=11),
        )


def test_adaptive_joint_requires_both_declared_and_fitted_endpoints():
    base = ModelConfig(name="PU-Joint", kind="joint", rank=1)
    without_endpoints = _tuning(include_endpoints=False)
    with pytest.raises(ValueError, match="requires include_endpoints=True"):
        candidate_search_plan(base, without_endpoints)

    train, validation = _partitions()
    with pytest.raises(ValueError, match="required_endpoints"):
        tune_model(
            train,
            validation,
            0.7,
            0.7,
            base,
            _tuning(ranks=(0, 1), candidate_budget=3),
            fit=FitConfig(maxiter=20, retry_maxiter=30, tolerance=1e-5),
        )


def test_top_two_rank_gate_does_not_silently_accept_one_failed_rank():
    configs = (
        ModelConfig(
            name="PU-Joint",
            kind="joint",
            rank=1,
            shared_l2=0.1,
            residual_l2=0.1,
        ),
        ModelConfig(
            name="PU-Joint",
            kind="joint",
            rank=2,
            shared_l2=0.1,
            residual_l2=0.1,
        ),
    )
    trials = tuple(
        TrialResult(
            stage="rank",
            index=index,
            config=config,
            validation_loss=0.2 + index,
            converged=index == 0,
            iterations=4,
            seed=8,
        )
        for index, config in enumerate(configs)
    )
    with pytest.raises(RuntimeError, match="expected 2, observed 1"):
        _top_rank_trials(trials)


def test_total_ratio_design_is_balanced_and_enumeration_independent():
    grid = tuple(
        ModelConfig(
            name="PU-Joint",
            kind="joint",
            rank=rank,
            shared_l2=shared,
            residual_l2=residual,
        )
        for rank, shared, residual in product((2, 8), PENALTIES, PENALTIES)
    )
    selected = _select_joint_shrinkage_candidates(grid, 27)
    renamed_reversed = tuple(
        candidate.with_updates(name="Joint", pu=False)
        for candidate in reversed(grid)
    )
    alternate = _select_joint_shrinkage_candidates(renamed_reversed, 27)

    identity = lambda value: (
        value.rank,
        value.shared_l2,
        value.residual_l2,
        value.target_l2,
    )
    assert tuple(map(identity, selected)) == tuple(map(identity, alternate))
    rank_counts = Counter(candidate.rank for candidate in selected)
    assert set(rank_counts) == {2, 8}
    assert max(rank_counts.values()) - min(rank_counts.values()) <= 1

    coordinates = _joint_shrinkage_coordinates(selected)
    # The change of variables is invertible: selected configurations within a
    # rank cannot collapse to duplicate total/ratio points.
    for rank in (2, 8):
        rows = coordinates[coordinates[:, 0] == rank, 1:3]
        assert len(np.unique(rows, axis=0)) == len(rows)
    # Pair-frequency balancing prevents the same 13--14 total/ratio locations
    # from simply being repeated at both retained ranks.
    pairs = [tuple(row) for row in coordinates[:, 1:3]]
    assert max(Counter(pairs).values()) <= 1


def _partitions() -> tuple[DatasetBundle, DatasetBundle]:
    rng = np.random.default_rng(815)
    x = rng.normal(size=(100, 4))
    coefficient = rng.normal(scale=0.4, size=(4, 3))
    probability = 1.0 / (1.0 + np.exp(-(x @ coefficient - 0.2)))
    reference = rng.random(probability.shape) < probability
    observed = reference & (rng.random(reference.shape) < 0.7)

    def bundle(rows: range) -> DatasetBundle:
        return DatasetBundle(
            x[rows],
            observed[rows],
            np.ones_like(observed[rows]),
            cell_ids=tuple(f"cell_{row}" for row in rows),
            target_ids=("a", "b", "c"),
        )

    return bundle(range(70)), bundle(range(70, 100))


def test_joint_tuner_uses_exact_budget_and_refines_only_top_two_ranks():
    train, validation = _partitions()
    tuning = _tuning(
        ranks=(0, 1, 2, 3),
        shared_l2=(0.03, 0.1),
        residual_l2=(0.03, 0.1),
        target_l2=(0.03,),
        anchor_shared_l2=0.1,
        anchor_residual_l2=0.1,
        anchor_target_l2=0.03,
        candidate_budget=8,
    )
    endpoints = (
        ModelConfig(name="PU", kind="direct", residual_l2=0.03),
        ModelConfig(name="PU-MIRT", kind="lowrank", rank=2, shared_l2=0.03),
    )
    result = tune_model(
        train,
        validation,
        0.7,
        0.7,
        ModelConfig(name="PU-Joint", kind="joint", rank=1),
        tuning,
        fit=FitConfig(
            maxiter=300,
            retry_maxiter=400,
            init_direct_maxiter=100,
            tolerance=1e-6,
        ),
        seed=71,
        required_endpoints=endpoints,
    )

    endpoints_trials = [
        trial for trial in result.trials if trial.stage == "inherited_endpoint"
    ]
    rank_trials = [trial for trial in result.trials if trial.stage == "rank"]
    penalty_trials = [trial for trial in result.trials if trial.stage == "penalty"]
    assert len(endpoints_trials) == 2
    assert len(rank_trials) == 3
    assert len(rank_trials) + len(penalty_trials) == 8
    assert len(result.trials) == 10

    retained = {
        trial.config.rank
        for trial in sorted(
            (trial for trial in rank_trials if trial.converged),
            key=_trial_tie_key,
        )[:2]
    }
    assert {trial.config.rank for trial in penalty_trials} == retained
    assert all(trial.config.kind == "joint" for trial in (*rank_trials, *penalty_trials))


def test_new_strategy_leaves_lowrank_endpoint_on_bounded_full_grid():
    train, validation = _partitions()
    tuning = _tuning(
        ranks=(1, 2, 3),
        shared_l2=(0.03, 0.1),
        residual_l2=(0.03, 0.1),
        target_l2=(0.03,),
        anchor_shared_l2=0.1,
        anchor_residual_l2=0.1,
        anchor_target_l2=0.03,
        candidate_budget=4,
        include_endpoints=False,
    )
    result = tune_model(
        train,
        validation,
        0.7,
        0.7,
        ModelConfig(name="PU-MIRT", kind="lowrank", rank=1),
        tuning,
        fit=FitConfig(maxiter=300, retry_maxiter=400, tolerance=1e-6),
    )
    assert len(result.trials) == 4
    assert {trial.stage for trial in result.trials} == {"joint_grid"}
    assert all(trial.config.kind == "lowrank" for trial in result.trials)


def test_single_penalty_anchor_can_exhaust_refinement_without_error():
    train, validation = _partitions()
    endpoints = (
        ModelConfig(name="PU", kind="direct", residual_l2=0.01),
        ModelConfig(name="PU-MIRT", kind="lowrank", rank=1, shared_l2=0.01),
    )
    result = tune_model(
        train,
        validation,
        0.7,
        0.7,
        ModelConfig(name="PU-Joint", kind="joint", rank=1),
        TuningConfig(
            strategy="rank_top2_total_ratio",
            ranks=(0, 1),
            shared_l2=(0.01,),
            residual_l2=(0.01,),
            target_l2=(0.01,),
            anchor_shared_l2=0.01,
            anchor_residual_l2=0.01,
            anchor_target_l2=0.01,
            candidate_budget=2,
            include_endpoints=True,
        ),
        fit=FitConfig(maxiter=300, retry_maxiter=400, tolerance=1e-6),
        required_endpoints=endpoints,
    )
    assert Counter(trial.stage for trial in result.trials) == {
        "inherited_endpoint": 2,
        "rank": 1,
    }


def test_trial_checkpoint_derives_and_validates_total_ratio_coordinates():
    config = ModelConfig(
        name="PU-Joint",
        kind="joint",
        rank=2,
        shared_l2=0.01,
        residual_l2=0.1,
    )
    trial = TrialResult(
        stage="penalty",
        index=0,
        config=config,
        validation_loss=0.2,
        converged=True,
        iterations=4,
        seed=8,
        total_shrinkage=float(np.sqrt(0.001)),
        residual_shared_ratio=10.0,
        optimizer_message="initializer=direct_endpoint",
    )
    payload = trial.to_dict()
    restored = TrialResult.from_dict(payload)
    assert restored.total_shrinkage == pytest.approx(np.sqrt(0.001))
    assert restored.residual_shared_ratio == pytest.approx(10.0)
    assert restored.optimizer_message == "initializer=direct_endpoint"

    legacy = dict(payload)
    for name in (
        "total_shrinkage",
        "residual_shared_ratio",
        "optimizer_message",
    ):
        legacy.pop(name)
    restored_legacy = TrialResult.from_dict(legacy)
    assert restored_legacy.total_shrinkage == pytest.approx(np.sqrt(0.001))
    assert restored_legacy.residual_shared_ratio == pytest.approx(10.0)
    assert restored_legacy.optimizer_message == ""

    corrupted = dict(payload, residual_shared_ratio=1.0)
    with pytest.raises(ValueError, match="does not match its model config"):
        TrialResult.from_dict(corrupted)
