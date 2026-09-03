import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gene2wire import DatasetBundle, FitConfig, ModelConfig, TuningConfig, run_model_grid
from gene2wire.checkpoint import AtomicArrayCheckpointStore, unit_key


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
