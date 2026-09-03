from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gene2wire import (
    AdaptiveFitConfig,
    DatasetBundle,
    MatchedModelConfig,
    MatchedTuningConfig,
    run_matched_model_grid,
)


def partitions(seed: int = 21):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 5)).astype(np.float32)
    coefficient = rng.normal(scale=0.45, size=(5, 3))
    probability = 1 / (1 + np.exp(-(x @ coefficient - 0.15)))
    reference = rng.random(probability.shape) < probability
    exposure = np.broadcast_to(np.array([0.75, 0.65, 0.85]), reference.shape).copy()
    observed = reference & (rng.random(reference.shape) < exposure)
    measured = np.ones_like(observed, dtype=bool)

    def bundle(rows):
        return DatasetBundle(
            X_cell=x[rows],
            S_observed=observed[rows],
            W_measured=measured[rows],
            cell_ids=[f"cell_{index}" for index in rows],
            target_ids=["a", "b", "c"],
        )

    return bundle(range(0, 54)), bundle(range(54, 75)), x[75:], exposure


def tiny_settings(loss_se_multiplier: float = 1.0):
    return (
        MatchedTuningConfig(
            direct_l2=(0.0, 1e-3),
            learning_rates=(0.005, 0.01),
            ranks=(1, 2),
            shared_l2=(1e-3,),
            residual_l2=(1e-3,),
            loss_se_multiplier=loss_se_multiplier,
        ),
        AdaptiveFitConfig(max_epochs=2, batch_size=32, patience=1),
    )


def six_models():
    return (
        MatchedModelConfig("Logistic", "direct", False),
        MatchedModelConfig("PU logistic", "direct", True),
        MatchedModelConfig("MIRT", "lowrank", False),
        MatchedModelConfig("PU-MIRT", "lowrank", True),
        MatchedModelConfig("Hybrid", "joint", False),
        MatchedModelConfig("PU-Hybrid", "joint", True),
    )


def test_matched_grid_exact_fallback_and_completed_resume(tmp_path: Path):
    train, validation, test_x, exposure = partitions()
    tuning, fit = tiny_settings(loss_se_multiplier=1e20)
    kwargs = dict(
        train=train,
        validation=validation,
        test_X=test_x,
        models=six_models(),
        tuning=tuning,
        fit=fit,
        train_exposure=exposure[:54],
        validation_exposure=exposure[54:75],
        test_exposure=exposure[75:],
        test_cell_ids=[f"cell_{index}" for index in range(75, 96)],
        checkpoint_dir=tmp_path / "checkpoints",
        unit_context={"outer_fold": 0},
        seed=42,
        code_version="unit-test",
    )
    first = run_matched_model_grid(**kwargs)
    assert all(not result.resumed for result in first.models.values())
    for structured, direct in (
        ("MIRT", "Logistic"),
        ("Hybrid", "Logistic"),
        ("PU-MIRT", "PU logistic"),
        ("PU-Hybrid", "PU logistic"),
    ):
        assert first.models[structured].selection["frozen_direct_fallback_used"]
        np.testing.assert_array_equal(
            first.models[structured].latent_probability,
            first.models[direct].latent_probability,
        )
        np.testing.assert_array_equal(
            first.models[structured].observed_probability,
            first.models[direct].observed_probability,
        )

    with patch(
        "gene2wire.matched.AdaptivePUModel.fit",
        side_effect=AssertionError("completed units were refitted"),
    ):
        second = run_matched_model_grid(**kwargs)
    assert all(result.resumed for result in second.models.values())
    assert first.fingerprint == second.fingerprint


def test_candidate_resume_after_interruption(tmp_path: Path):
    train, validation, test_x, exposure = partitions()
    tuning, fit = tiny_settings()
    statuses = []

    def interrupt(model, candidate_id, status):
        statuses.append((model, candidate_id, status))
        if candidate_id == 1 and status == "completed":
            raise RuntimeError("simulated disconnect")

    kwargs = dict(
        train=train,
        validation=validation,
        test_X=test_x,
        models=(MatchedModelConfig("PU direct", "direct", True),),
        tuning=tuning,
        fit=fit,
        train_exposure=exposure[:54],
        validation_exposure=exposure[54:75],
        test_exposure=exposure[75:],
        test_cell_ids=[f"cell_{index}" for index in range(75, 96)],
        checkpoint_dir=tmp_path / "checkpoints",
        seed=9,
        code_version="unit-test",
    )
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        run_matched_model_grid(**kwargs, on_candidate=interrupt)

    resumed_statuses = []
    result = run_matched_model_grid(
        **kwargs,
        on_candidate=lambda model, candidate_id, status: resumed_statuses.append(
            (model, candidate_id, status)
        ),
    )
    assert ("PU direct", 0, "resumed") in resumed_statuses
    assert ("PU direct", 1, "resumed") in resumed_statuses
    assert not result.models["PU direct"].resumed


def test_matched_runner_rejects_reference_truth(tmp_path: Path):
    train, validation, test_x, _ = partitions()
    unsafe = DatasetBundle(
        train.X_cell,
        train.S_observed,
        train.W_measured,
        Z_reference=train.S_observed,
        cell_ids=train.cell_ids,
        target_ids=train.target_ids,
    )
    tuning, fit = tiny_settings()
    with pytest.raises(ValueError, match="reference truth"):
        run_matched_model_grid(
            train=unsafe,
            validation=validation,
            test_X=test_x,
            models=(MatchedModelConfig("direct", "direct"),),
            tuning=tuning,
            fit=fit,
            checkpoint_dir=tmp_path,
        )
