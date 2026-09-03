from pathlib import Path

import numpy as np
from scipy.optimize import check_grad

from gene2wire import DatasetBundle, FitConfig, ModelConfig, TuningConfig
from gene2wire.checkpoint import sha256_array
from gene2wire.models import UnifiedPUModel, _exposure_matrix
from gene2wire.tuning import full_joint_candidates


def small_bundle() -> DatasetBundle:
    return DatasetBundle(
        X_cell=np.array([[0.2, -0.1], [1.0, 0.3], [-0.4, 0.8], [0.6, -0.5]]),
        S_observed=np.array([[1, 0], [0, 1], [0, 0], [1, 0]]),
        W_measured=np.array([[1, 1], [1, 1], [1, 1], [1, 1]]),
        Y_target=np.array([[1.0], [-1.0]]),
    )


def test_core_source_contains_no_dataset_specific_name():
    root = Path(__file__).resolve().parents[1] / "src" / "gene2wire"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "SPIDER" not in text
    assert "Projection-TAGs" not in text
    assert "GSE277718" not in text


def test_gradient_for_all_structures():
    data = small_bundle()
    exposure = _exposure_matrix(np.array([[0.7], [0.8], [0.9], [1.0]]), data.S_observed.shape)
    for kind, rank in (("direct", 0), ("lowrank", 1), ("joint", 1)):
        config = ModelConfig(
            name=kind,
            kind=kind,
            rank=rank,
            shared_l2=0.03 if kind != "direct" else 0.0,
            residual_l2=0.04 if kind != "lowrank" else 0.0,
            use_target_features=True,
            target_l2=0.02,
        )
        model = UnifiedPUModel(config, FitConfig(initialization="random"))
        theta = model._initialize(
            data.X_cell,
            data.S_observed.astype(float),
            data.W_measured,
            exposure,
            data.Y_target,
            7,
        )
        objective = lambda value: model._objective_gradient(
            value,
            data.X_cell,
            data.S_observed.astype(float),
            data.W_measured,
            exposure,
            data.Y_target,
        )[0]
        gradient = lambda value: model._objective_gradient(
            value,
            data.X_cell,
            data.S_observed.astype(float),
            data.W_measured,
            exposure,
            data.Y_target,
        )[1]
        assert check_grad(objective, gradient, theta) < 2e-5


def test_complete_joint_grid_and_array_hash():
    tuning = TuningConfig(
        ranks=(0, 1, 2),
        shared_l2=(0.1, 0.2),
        residual_l2=(0.3, 0.4),
        target_l2=(0.5,),
    )
    assert len(full_joint_candidates(ModelConfig(name="joint", kind="joint"), tuning)) == 10
    array = np.arange(8, dtype=np.float64).reshape(4, 2)
    assert sha256_array(array) == sha256_array(array.copy())
    assert sha256_array(array) != sha256_array(array.astype(np.float32))


def test_bundle_rejects_positive_outside_measurement_mask():
    try:
        DatasetBundle([[0.0]], [[1]], [[0]])
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("invalid bundle was accepted")
