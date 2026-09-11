from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import check_grad

from gene2wire import (
    DatasetBundle,
    FitConfig,
    ModelConfig,
    TuningConfig,
    run_model_grid,
)
from gene2wire.models import FittedModel, UnifiedPUModel, model_identity
from gene2wire.tuning import full_joint_candidates


def _grouped_bundle(rows=None) -> DatasetBundle:
    rng = np.random.default_rng(20260911)
    n_cells = 24
    panels = np.arange(n_cells) % 2
    raw = rng.normal(size=(n_cells, 2))
    x = np.zeros((n_cells, 4), dtype=float)
    x[panels == 0, :2] = raw[panels == 0]
    x[panels == 1, 2:] = raw[panels == 1]
    coefficient_a = np.array([[0.8, -0.3, 0.5], [-0.4, 0.7, 0.2]])
    coefficient_b = np.array([[0.5, 0.1, -0.6], [0.3, -0.8, 0.4]])
    eta = np.full((n_cells, 3), -0.2)
    eta[panels == 0] += raw[panels == 0] @ coefficient_a
    eta[panels == 1] += raw[panels == 1] @ coefficient_b + 0.15
    probability = 1 / (1 + np.exp(-eta))
    truth = rng.random(probability.shape) < probability
    exposure = np.array([0.75, 0.8, 0.7])[None, :]
    observed = truth & (rng.random(truth.shape) < exposure)
    selected = np.arange(n_cells) if rows is None else np.asarray(rows, dtype=int)
    return DatasetBundle(
        X_cell=x[selected],
        S_observed=observed[selected],
        W_measured=np.ones_like(observed[selected], dtype=bool),
        cell_ids=tuple(f"cell_{index}" for index in selected),
        target_ids=("t0", "t1", "t2"),
        feature_blocks={"panel_A_gene": (0, 1), "panel_B_gene": (2, 3)},
        X_nuisance=panels[selected, None].astype(float),
        nuisance_names=("panel_B",),
    )


def _grouped_config(name="Separate-A") -> ModelConfig:
    return ModelConfig(
        name=name,
        kind="lowrank",
        rank=1,
        shared_l2=0.1,
        nuisance_l2=1e-6,
        lowrank_feature_groups=("panel_A_gene", "panel_B_gene"),
    )


def test_grouped_lowrank_config_and_feature_partition_validation():
    with pytest.raises(ValueError, match="only for lowrank"):
        ModelConfig(
            name="bad",
            kind="direct",
            lowrank_feature_groups=("a", "b"),
        )
    with pytest.raises(ValueError, match="at least two"):
        ModelConfig(
            name="bad",
            kind="lowrank",
            rank=1,
            lowrank_feature_groups=("a",),
        )
    with pytest.raises(ValueError, match="unique"):
        ModelConfig(
            name="bad",
            kind="lowrank",
            rank=1,
            lowrank_feature_groups=("a", "a"),
        )

    bundle = _grouped_bundle()
    missing = _grouped_config().with_updates(
        lowrank_feature_groups=("panel_A_gene", "missing")
    )
    with pytest.raises(ValueError, match="missing feature blocks"):
        UnifiedPUModel(missing).fit(bundle)

    incomplete = DatasetBundle(
        X_cell=bundle.X_cell,
        S_observed=bundle.S_observed,
        W_measured=bundle.W_measured,
        feature_blocks={"panel_A_gene": (0,), "panel_B_gene": (2, 3)},
        X_nuisance=bundle.X_nuisance,
        nuisance_names=bundle.nuisance_names,
    )
    with pytest.raises(ValueError, match="partition every X_cell feature"):
        UnifiedPUModel(_grouped_config()).fit(incomplete)

    with pytest.raises(ValueError, match="feature/target cap 2"):
        UnifiedPUModel(_grouped_config().with_updates(rank=3)).fit(bundle)


def test_grouped_lowrank_pu_gradient_with_shared_nuisance():
    bundle = _grouped_bundle()
    exposure = np.broadcast_to(
        np.array([0.75, 0.8, 0.7]), bundle.S_observed.shape
    ).copy()
    groups = ((0, 1), (2, 3))
    model = UnifiedPUModel(
        _grouped_config(), FitConfig(initialization="random")
    )
    theta = model._initialize(
        bundle.X_cell,
        bundle.S_observed.astype(float),
        bundle.W_measured,
        exposure,
        None,
        7,
        nuisance=bundle.X_nuisance,
        lowrank_feature_indices=groups,
    )

    def value(parameters):
        return model._objective_gradient(
            parameters,
            bundle.X_cell,
            bundle.S_observed.astype(float),
            bundle.W_measured,
            exposure,
            None,
            bundle.X_nuisance,
            groups,
        )

    assert check_grad(lambda z: value(z)[0], lambda z: value(z)[1], theta) < 3e-5


def test_grouped_fitted_logits_are_exact_sum_of_panel_terms():
    config = _grouped_config()
    cell_factor = np.array([[0.2], [0.5], [-0.4], [0.3]])
    target_factor = np.array(
        [[[0.7], [-0.2], [0.4]], [[-0.1], [0.6], [0.8]]]
    )
    nuisance_coeff = np.array([[0.1, -0.3, 0.2]])
    intercept = np.array([-0.5, 0.2, 0.1])
    fitted = FittedModel(
        config=config,
        cell_shared=cell_factor,
        target_shared=target_factor,
        residual=None,
        target_coeff=None,
        target_features=None,
        nuisance_coeff=nuisance_coeff,
        nuisance_names=("panel_B",),
        intercept=intercept,
        objective=0.0,
        converged=True,
        iterations=1,
        message="test",
        lowrank_feature_indices=((0, 1), (2, 3)),
    )
    x = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, -1.0, 3.0]])
    nuisance = np.array([[0.0], [1.0]])
    expected = np.broadcast_to(intercept, (2, 3)).copy()
    expected += nuisance @ nuisance_coeff
    expected += (x[:, :2] @ cell_factor[:2]) @ target_factor[0].T
    expected += (x[:, 2:] @ cell_factor[2:]) @ target_factor[1].T
    np.testing.assert_allclose(fitted.latent_logit(x, nuisance), expected)


def test_grouped_fit_and_checkpoint_resume_preserve_separate_target_factors(
    tmp_path: Path,
):
    train = _grouped_bundle(np.arange(0, 12))
    validation = _grouped_bundle(np.arange(12, 18))
    test = _grouped_bundle(np.arange(18, 24))
    config = _grouped_config()
    tuning = TuningConfig(
        ranks=(1,),
        shared_l2=(0.1,),
        residual_l2=(0.1,),
        target_l2=(0.1,),
        anchor_shared_l2=0.1,
        anchor_residual_l2=0.1,
        anchor_target_l2=0.1,
    )
    kwargs = dict(
        train=train,
        validation=validation,
        test_X=test.X_cell,
        test_nuisance=test.X_nuisance,
        test_cell_ids=test.cell_ids,
        models=(config,),
        tuning=tuning,
        train_exposure=0.75,
        validation_exposure=0.75,
        test_exposure=0.75,
        fit=FitConfig(
            maxiter=80,
            tolerance=1e-7,
            initialization="svd",
            init_direct_maxiter=40,
        ),
        checkpoint_dir=tmp_path / "checkpoints",
        code_version="grouped-lowrank-test",
        seed=9,
    )
    first = run_model_grid(**kwargs)
    second = run_model_grid(**kwargs)
    first_model = first.models[config.name]
    second_model = second.models[config.name]
    assert first_model.fitted.target_shared.shape == (2, 3, 1)
    assert first_model.fitted.lowrank_feature_indices == ((0, 1), (2, 3))
    assert second_model.resumed
    np.testing.assert_allclose(
        first_model.latent_probability, second_model.latent_probability
    )
    assert second_model.summary()["factor_parameterization"] == "separate_A"


def test_standard_and_grouped_lowrank_have_distinct_complete_identities():
    shared = _grouped_config("Shared-A").with_updates(lowrank_feature_groups=())
    separate = _grouped_config("Separate-A")
    assert model_identity(shared) != model_identity(separate)
    tuning = TuningConfig(
        ranks=(1, 2),
        shared_l2=(0.1,),
        residual_l2=(0.1,),
        target_l2=(0.1,),
    )
    candidates = full_joint_candidates(separate, tuning)
    assert len(candidates) == 2
    assert all(
        candidate.lowrank_feature_groups
        == ("panel_A_gene", "panel_B_gene")
        for candidate in candidates
    )
