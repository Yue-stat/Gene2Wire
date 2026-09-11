import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gene2wire import DatasetBundle, FitConfig, ModelConfig, TuningConfig, run_model_grid
from gene2wire.checkpoint import AtomicArrayCheckpointStore, unit_key
from gene2wire.models import model_identity
from gene2wire.runner import _compatible_joint_endpoint


def partitions(seed: int = 11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(72, 4))
    coefficient = rng.normal(scale=0.5, size=(4, 3))
    probability = 1 / (1 + np.exp(-(x @ coefficient - 0.2)))
    truth = rng.random(probability.shape) < probability
    exposure = np.broadcast_to(np.array([0.8, 0.7, 0.9]), truth.shape)
    observed = truth & (rng.random(truth.shape) < exposure)
    measured = np.ones_like(observed, dtype=bool)

    def bundle(rows):
        return DatasetBundle(
            x[rows],
            observed[rows],
            measured[rows],
            cell_ids=[f"cell_{index}" for index in rows],
            target_ids=["a", "b", "c"],
        )

    return bundle(range(0, 36)), bundle(range(36, 54)), x[54:], exposure


def tiny_tuning():
    return TuningConfig(
        ranks=(1,),
        shared_l2=(0.1,),
        residual_l2=(0.03, 0.1),
        target_l2=(0.1,),
        anchor_shared_l2=0.1,
        anchor_residual_l2=0.1,
        anchor_target_l2=0.1,
    )


def test_completed_model_resume_skips_tuning_and_fit(tmp_path: Path):
    train, validation, test_x, exposure = partitions()
    kwargs = dict(
        train=train,
        validation=validation,
        test_X=test_x,
        train_exposure=exposure[:36],
        validation_exposure=exposure[36:54],
        test_exposure=exposure[54:],
        test_cell_ids=[f"cell_{index}" for index in range(54, 72)],
        models=(ModelConfig(name="PU logistic", kind="direct", pu=True),),
        tuning=tiny_tuning(),
        fit=FitConfig(maxiter=300, tolerance=1e-7, initialization="random"),
        checkpoint_dir=tmp_path / "checkpoints",
        unit_context={"fold": 0, "condition": "example"},
        seed=17,
        code_version="test-commit",
    )
    first = run_model_grid(**kwargs)
    assert not first.models["PU logistic"].resumed
    with patch("gene2wire.runner.tune_model", side_effect=AssertionError("retuned")), patch(
        "gene2wire.runner.UnifiedPUModel.fit", side_effect=AssertionError("refitted")
    ):
        second = run_model_grid(**kwargs)
    assert second.models["PU logistic"].resumed
    np.testing.assert_allclose(
        first.models["PU logistic"].latent_probability,
        second.models["PU logistic"].latent_probability,
    )
    assert first.fingerprint == second.fingerprint


def test_reference_truth_is_rejected_before_core(tmp_path: Path):
    train, validation, test_x, exposure = partitions()
    unsafe = DatasetBundle(
        train.X_cell,
        train.S_observed,
        train.W_measured,
        Z_reference=train.S_observed,
        cell_ids=train.cell_ids,
        target_ids=train.target_ids,
    )
    with pytest.raises(ValueError, match="reference truth"):
        run_model_grid(
            train=unsafe,
            validation=validation,
            test_X=test_x,
            models=(ModelConfig(name="d", kind="direct"),),
            tuning=tiny_tuning(),
            checkpoint_dir=tmp_path,
        )


def test_array_checkpoint_detects_tampering(tmp_path: Path):
    store = AtomicArrayCheckpointStore(tmp_path)
    key = unit_key(task="model", fold=0)
    manifest = store.save_complete(key, "fingerprint", {"score": 1.0}, {"x": np.arange(4)})
    record = json.loads(manifest.read_text(encoding="utf-8"))
    arrays_path = tmp_path / record["arrays_file"]
    arrays_path.write_bytes(arrays_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="array checksum mismatch"):
        store.load(key, "fingerprint")


def test_runner_shares_warm_starts_across_model_families(tmp_path: Path):
    train, validation, test_x, exposure = partitions()
    tuning = TuningConfig(
        ranks=(1, 2),
        shared_l2=(0.1,),
        residual_l2=(0.1,),
        target_l2=(0.1,),
        anchor_shared_l2=0.1,
        anchor_residual_l2=0.1,
        anchor_target_l2=0.1,
    )
    run_model_grid(
        train=train,
        validation=validation,
        test_X=test_x,
        train_exposure=exposure[:36],
        validation_exposure=exposure[36:54],
        test_exposure=exposure[54:],
        test_cell_ids=[f"cell_{index}" for index in range(54, 72)],
        models=(
            ModelConfig(name="MIRT", kind="lowrank", rank=1, pu=False),
            ModelConfig(name="Joint", kind="joint", rank=1, pu=False),
        ),
        tuning=tuning,
        fit=FitConfig(
            maxiter=300,
            tolerance=1e-7,
            initialization="svd",
            init_direct_maxiter=100,
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        unit_context={"fold": 0, "condition": "cache-test"},
        seed=23,
        code_version="warm-start-cache-test",
    )

    # The two compatible model families and both ranks need only one direct
    # initializer for tuning and one for the distinct development/refit data.
    manifests = list((tmp_path / "checkpoints" / "warm_starts").glob("*.json"))
    assert len(manifests) == 2


def test_runner_carries_standalone_winners_into_joint_independent_of_input_order():
    train, validation, test_x, exposure = partitions()
    tuning = TuningConfig(
        ranks=(1,), shared_l2=(.1,), residual_l2=(.1,), target_l2=(.1,),
        candidate_budget=1, include_endpoints=True,
    )
    models = (
        ModelConfig(name="Joint", kind="joint", rank=1, pu=False),
        ModelConfig(name="MIRT", kind="lowrank", rank=1, pu=False),
        ModelConfig(name="Logistic", kind="direct", pu=False),
    )
    result = run_model_grid(
        train=train, validation=validation, test_X=test_x,
        train_exposure=exposure[:36], validation_exposure=exposure[36:54],
        test_exposure=exposure[54:],
        test_cell_ids=[f"cell_{index}" for index in range(54, 72)],
        models=models, tuning=tuning,
        fit=FitConfig(maxiter=30, retry_maxiter=40, tolerance=1e-5),
        seed=93, code_version="joint-endpoint-test",
    )
    assert tuple(result.models) == tuple(model.name for model in models)
    joint_trials = result.models["Joint"].tuning.trials
    assert len(joint_trials) == 3
    assert sum(trial.config.kind == "joint" for trial in joint_trials) == 1
    assert sum(trial.stage == "inherited_endpoint" for trial in joint_trials) == 2
    expected = {
        str(model_identity(result.models["Logistic"].tuning.best_config)),
        str(model_identity(result.models["MIRT"].tuning.best_config)),
    }
    assert expected.issubset({str(model_identity(trial.config)) for trial in joint_trials})


def test_joint_endpoint_compatibility_excludes_nuisance_and_separate_a_models():
    joint = ModelConfig(name="Joint", kind="joint", rank=1, nuisance_l2=.1)
    assert _compatible_joint_endpoint(
        ModelConfig(name="Logistic", kind="direct", nuisance_l2=.1), joint
    )
    assert not _compatible_joint_endpoint(
        ModelConfig(name="Logistic-other", kind="direct", nuisance_l2=.2), joint
    )
    assert not _compatible_joint_endpoint(
        ModelConfig(
            name="Separate-A", kind="lowrank", rank=1, nuisance_l2=.1,
            lowrank_feature_groups=("panel_A_gene", "panel_B_gene"),
        ),
        joint,
    )
