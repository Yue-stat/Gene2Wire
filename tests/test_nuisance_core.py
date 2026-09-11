from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import check_grad

from gene2wire.adapters import load_npz_bundle, save_npz_bundle
from gene2wire.config import FitConfig, ModelConfig, TuningConfig
from gene2wire.data import DatasetBundle
from gene2wire.models import DirectWarmStartCache, FittedModel, UnifiedPUModel
from gene2wire.runner import run_model_grid


def nuisance_bundle(n: int = 24) -> DatasetBundle:
    rng = np.random.default_rng(91)
    x = rng.normal(size=(n, 2))
    nuisance = (np.arange(n) % 2).astype(float)[:, None]
    logits = np.column_stack(
        (
            0.6 * x[:, 0] + 0.8 * nuisance[:, 0] - 0.4,
            -0.5 * x[:, 1] - 0.7 * nuisance[:, 0] + 0.2,
        )
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    labels = rng.binomial(1, probability)
    # Guarantee both outcomes in both panel/target strata for stable tiny tests.
    labels[:4] = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    return DatasetBundle(
        X_cell=x,
        S_observed=labels,
        W_measured=np.ones_like(labels),
        cell_ids=tuple(f"cell_{index}" for index in range(n)),
        target_ids=("left", "right"),
        X_nuisance=nuisance,
        nuisance_names=("overlap_panel[B]",),
    )


@pytest.mark.parametrize("kind,rank", [("direct", 0), ("lowrank", 1), ("joint", 1)])
def test_nuisance_gradient_for_every_structure(kind, rank):
    data = nuisance_bundle(12)
    config = ModelConfig(
        name=kind,
        kind=kind,
        rank=rank,
        shared_l2=0.03 if kind != "direct" else 0.0,
        residual_l2=0.04 if kind != "lowrank" else 0.0,
        nuisance_l2=0.02,
    )
    model = UnifiedPUModel(config, FitConfig(initialization="random"))
    exposure = np.full(data.S_observed.shape, 0.8)
    theta = model._initialize(
        data.X_cell,
        data.S_observed.astype(float),
        data.W_measured,
        exposure,
        None,
        7,
        nuisance=data.X_nuisance,
    )
    args = (
        data.X_cell,
        data.S_observed.astype(float),
        data.W_measured,
        exposure,
        None,
        data.X_nuisance,
    )
    error = check_grad(
        lambda value: model._objective_gradient(value, *args)[0],
        lambda value: model._objective_gradient(value, *args)[1],
        theta,
    )
    assert error < 3e-5


def test_fit_prediction_and_state_roundtrip_require_nuisance():
    data = nuisance_bundle()
    fitted = UnifiedPUModel(
        ModelConfig(name="PU", kind="direct", residual_l2=0.03, nuisance_l2=1e-4),
        FitConfig(maxiter=300, tolerance=1e-7),
    ).fit(data, exposure=0.8, seed=3)
    assert fitted.nuisance_coeff is not None
    assert fitted.nuisance_coeff.shape == (1, data.n_targets)
    assert fitted.nuisance_names == data.nuisance_names
    prediction = fitted.predict_proba(data.X_cell, x_nuisance=data.X_nuisance)
    restored = FittedModel.from_state_dict(fitted.state_dict())
    np.testing.assert_allclose(
        prediction,
        restored.predict_proba(data.X_cell, x_nuisance=data.X_nuisance),
    )
    with pytest.raises(ValueError, match="x_nuisance is required"):
        fitted.predict_proba(data.X_cell)
    with pytest.raises(ValueError, match="prediction rows"):
        fitted.predict_proba(data.X_cell, x_nuisance=data.X_nuisance[:-1])


def test_nuisance_design_and_absent_schema_are_validated():
    data = nuisance_bundle()
    with pytest.raises(ValueError, match="nuisance_names requires"):
        replace(data, X_nuisance=None, nuisance_names=("bad",))
    constant = replace(data, X_nuisance=np.ones((data.n_cells, 1)))
    with pytest.raises(ValueError, match="not identifiable"):
        UnifiedPUModel(ModelConfig(name="PU", kind="direct", nuisance_l2=1e-4)).fit(
            constant
        )
    plain = DatasetBundle(data.X_cell, data.S_observed, data.W_measured)
    assert plain.n_nuisance == 0 and plain.nuisance_names == ()
    with pytest.raises(ValueError, match="nuisance_l2 must be zero"):
        UnifiedPUModel(ModelConfig(name="PU", kind="direct", nuisance_l2=1e-4)).fit(
            plain
        )


def test_nuisance_survives_bundle_operations_and_npz(tmp_path):
    data = nuisance_bundle()
    subset = data.subset_rows(np.arange(8))
    safe = subset.without_reference()
    np.testing.assert_array_equal(safe.X_nuisance, data.X_nuisance[:8])
    assert safe.nuisance_names == data.nuisance_names
    destination = save_npz_bundle(data, tmp_path / "bundle.npz")
    restored = load_npz_bundle(destination)
    np.testing.assert_array_equal(restored.X_nuisance, data.X_nuisance)
    assert restored.nuisance_names == data.nuisance_names


def test_direct_warm_start_reuses_nuisance_fit_across_structures():
    data = nuisance_bundle()
    cache = DirectWarmStartCache()
    fit = FitConfig(maxiter=100, init_direct_maxiter=80, tolerance=1e-7)
    lowrank = ModelConfig(
        name="PU-MIRT", kind="lowrank", rank=1, shared_l2=0.1, nuisance_l2=1e-4
    )
    joint = ModelConfig(
        name="PU-Joint",
        kind="joint",
        rank=1,
        shared_l2=0.1,
        residual_l2=0.1,
        nuisance_l2=1e-4,
    )
    UnifiedPUModel(lowrank, fit, warm_start_cache=cache).fit(data, exposure=0.8, seed=5)
    UnifiedPUModel(joint, fit, warm_start_cache=cache).fit(data, exposure=0.8, seed=5)
    assert cache.misses == 1
    assert cache.hits == 1


def _runner_inputs(data: DatasetBundle):
    train = data.subset_rows(np.arange(12))
    validation = data.subset_rows(np.arange(12, 18))
    test = np.arange(18, 24)
    tuning = TuningConfig(
        ranks=(0,),
        shared_l2=(0.1,),
        residual_l2=(0.1,),
        target_l2=(0.1,),
    )
    return train, validation, test, tuning


def test_runner_requires_and_hashes_prediction_nuisance():
    data = nuisance_bundle()
    train, validation, test, tuning = _runner_inputs(data)
    kwargs = dict(
        train=train,
        validation=validation,
        test_X=data.X_cell[test],
        models=(
            ModelConfig(
                name="PU", kind="direct", residual_l2=0.1, nuisance_l2=1e-4
            ),
        ),
        tuning=tuning,
        train_exposure=0.8,
        validation_exposure=0.8,
        test_exposure=0.8,
        test_cell_ids=np.asarray(data.cell_ids)[test],
        fit=FitConfig(maxiter=200, tolerance=1e-7),
        seed=11,
    )
    with pytest.raises(ValueError, match="test_nuisance is required"):
        run_model_grid(**kwargs)
    first = run_model_grid(**kwargs, test_nuisance=data.X_nuisance[test])
    changed = data.X_nuisance[test].copy()
    changed[:, 0] = 1.0 - changed[:, 0]
    second = run_model_grid(**kwargs, test_nuisance=changed)
    assert first.fingerprint != second.fingerprint
    assert first.models["PU"].latent_probability.shape == (len(test), data.n_targets)
