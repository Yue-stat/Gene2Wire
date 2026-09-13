"""Deterministic, leakage-safe endpoint initialization for Joint models."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gene2wire import DatasetBundle, FitConfig, JointEndpointStarts, ModelConfig
from gene2wire.models import UnifiedPUModel


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(412)
    x = rng.normal(size=(120, 4))
    coefficient = rng.normal(scale=0.45, size=(4, 3))
    probability = 1.0 / (1.0 + np.exp(-(x @ coefficient - 0.25)))
    reference = rng.random(probability.shape) < probability
    exposure = 0.7
    observed = reference & (rng.random(reference.shape) < exposure)
    return DatasetBundle(x, observed, np.ones_like(observed))


def _endpoints(data: DatasetBundle) -> JointEndpointStarts:
    fit = FitConfig(
        maxiter=400,
        retry_maxiter=500,
        init_direct_maxiter=150,
        tolerance=1e-6,
    )
    direct = UnifiedPUModel(
        ModelConfig(name="PU", kind="direct", residual_l2=0.03), fit
    ).fit(data, exposure=0.7, seed=9)
    lowrank = UnifiedPUModel(
        ModelConfig(name="PU-MIRT", kind="lowrank", rank=2, shared_l2=0.03),
        fit,
    ).fit(data, exposure=0.7, seed=9)
    assert direct.converged and lowrank.converged
    return JointEndpointStarts(direct=direct, lowrank=lowrank)


@pytest.mark.parametrize("joint_rank", [1, 2, 3])
def test_joint_endpoint_starts_reproduce_both_inner_train_endpoint_logits(joint_rank):
    data = _bundle()
    endpoints = _endpoints(data)
    model = UnifiedPUModel(
        ModelConfig(
            name="PU-Joint",
            kind="joint",
            rank=joint_rank,
            shared_l2=0.03,
            residual_l2=0.03,
        ),
        FitConfig(tolerance=1e-6),
        joint_endpoint_starts=endpoints,
    )

    starts = model._joint_endpoint_initializers(
        data.X_cell,
        data.n_targets,
        data.Y_target,
        data.X_nuisance,
        data.nuisance_names,
        (),
    )
    assert [label for label, _ in starts] == [
        "direct_endpoint",
        "lowrank_endpoint",
    ]
    expected = {
        "direct_endpoint": endpoints.direct.latent_logit(data.X_cell),
        "lowrank_endpoint": endpoints.lowrank.latent_logit(data.X_cell),
    }
    for label, theta in starts:
        shared, target, residual, _, _, intercept = model._unpack(
            theta,
            data.n_features,
            data.n_targets,
            0,
        )
        actual = (data.X_cell @ shared) @ target.T
        actual += data.X_cell @ residual
        actual += intercept
        np.testing.assert_allclose(actual, expected[label], rtol=1e-11, atol=1e-11)


def test_joint_fit_runs_both_endpoint_starts_and_reports_the_winner():
    data = _bundle()
    endpoints = _endpoints(data)
    model = UnifiedPUModel(
        ModelConfig(
            name="PU-Joint",
            kind="joint",
            rank=2,
            shared_l2=0.03,
            residual_l2=0.03,
        ),
        FitConfig(tolerance=1e-6),
        joint_endpoint_starts=endpoints,
    )
    starts = dict(
        model._joint_endpoint_initializers(
            data.X_cell,
            data.n_targets,
            data.Y_target,
            data.X_nuisance,
            data.nuisance_names,
            (),
        )
    )

    def fake_optimizer(_fun, theta):
        is_lowrank = np.array_equal(theta, starts["lowrank_endpoint"])
        return (
            SimpleNamespace(
                x=np.array(theta, copy=True),
                fun=0.2 if is_lowrank else 0.3,
                success=True,
                message="mock convergence",
            ),
            4,
            False,
        )

    with patch.object(model, "_minimize_from_start", side_effect=fake_optimizer) as run:
        fitted = model.fit(data, exposure=0.7, seed=9)

    assert run.call_count == 2
    assert fitted.objective == pytest.approx(0.2)
    assert fitted.iterations == 8
    assert "initializer=lowrank_endpoint" in fitted.message
    assert "direct_endpoint=ok" in fitted.message
    assert "lowrank_endpoint=ok" in fitted.message


def test_nonconverged_endpoint_is_never_used_as_an_initializer():
    data = _bundle()
    endpoints = _endpoints(data)
    with pytest.raises(RuntimeError, match="direct endpoint initializer did not converge"):
        JointEndpointStarts(
            direct=replace(endpoints.direct, converged=False),
            lowrank=endpoints.lowrank,
        )


def test_internal_direct_initializer_retries_then_fails_explicitly():
    data = _bundle()
    model = UnifiedPUModel(
        ModelConfig(name="PU-MIRT", kind="lowrank", rank=1, shared_l2=0.03),
        FitConfig(
            maxiter=2,
            init_direct_maxiter=1,
            retry_maxiter=3,
            tolerance=1e-6,
        ),
    )

    def never_converges(fun, theta, **_kwargs):
        objective, _ = fun(np.asarray(theta, dtype=float))
        return SimpleNamespace(
            x=np.asarray(theta, dtype=float),
            fun=float(objective),
            success=False,
            nit=1,
            message="iteration limit",
        )

    with patch("gene2wire.models.minimize", side_effect=never_converges) as optimize:
        with pytest.raises(RuntimeError, match="direct initializer did not converge"):
            model.fit(data, exposure=0.7, seed=4)

    # One short initializer fit and exactly one declared continuation.
    assert optimize.call_count == 2


def test_optimizer_continuation_uses_the_declared_retry_solution():
    model = UnifiedPUModel(
        ModelConfig(name="PU", kind="direct"),
        FitConfig(maxiter=1, retry_maxiter=3),
    )
    theta = np.array([0.2, -0.4])
    first = SimpleNamespace(
        x=theta + 0.1,
        fun=0.7,
        success=False,
        nit=1,
        message="iteration limit",
    )
    second = SimpleNamespace(
        x=theta + 0.2,
        fun=0.6,
        success=True,
        nit=2,
        message="converged",
    )
    with patch("gene2wire.models.minimize", side_effect=(first, second)) as optimize:
        result, iterations, retried = model._minimize_from_start(
            lambda value: (float(value @ value), 2.0 * value), theta
        )

    assert optimize.call_count == 2
    np.testing.assert_array_equal(optimize.call_args_list[1].args[1], first.x)
    assert result is second
    assert iterations == 3
    assert retried


def test_joint_endpoint_schema_rejects_pu_mismatch_before_optimization():
    data = _bundle()
    endpoints = _endpoints(data)
    incompatible = JointEndpointStarts(
        direct=replace(
            endpoints.direct,
            config=endpoints.direct.config.with_updates(pu=False),
        ),
        lowrank=endpoints.lowrank,
    )
    model = UnifiedPUModel(
        ModelConfig(name="Joint", kind="joint", rank=2, pu=True),
        FitConfig(tolerance=1e-6),
        joint_endpoint_starts=incompatible,
    )
    with pytest.raises(ValueError, match="PU setting differs"):
        model.fit(data, exposure=0.7, seed=4)
