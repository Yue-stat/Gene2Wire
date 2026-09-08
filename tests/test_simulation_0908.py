"""Scientific invariants for the new simulation adapter and bilinear baselines."""
import pickle

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.special import expit

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.qiao import _objective_gradient, fit_qiao, tune_qiao


def _dataset(rho=0.5, repetition=0, **overrides):
    settings = dict(n_cells=65, n_targets=8, n_gene_features=6,
                    n_location_features=3, n_slices=9, n_target_features=3)
    settings.update(overrides)
    return generate_simulation(repetition, rho, seed=42, **settings)


@pytest.mark.parametrize("use_location", [False, True])
@pytest.mark.parametrize("use_target_features", [False, True])
def test_actual_feature_dimensions_and_train_only_scaling(use_location, use_target_features):
    data = _dataset(truth_uses_location=use_location)
    fold = data.split_builder(3, 11)[0]
    features = data.feature_builder(fold.train_rows, use_location, use_target_features)
    expected_d = 9 if use_location else 6
    assert features.X.shape == (65, expected_d)
    assert data.metadata["true_coefficient"].shape == (expected_d, 8)
    assert data.metadata["truth_uses_location"] is use_location
    assert len(features.feature_names) == expected_d
    assert (features.Y_target is not None) is use_target_features
    np.testing.assert_allclose(features.X[fold.train_rows].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(features.X[fold.train_rows].std(0), 1, atol=1e-12)
    # Test-set means generally differ: this was not standardized using all cells.
    assert np.max(np.abs(features.X[fold.test_rows].mean(0))) > 0.1


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_truth_coefficient_reconstruction_and_declared_sharing(rho):
    data = _dataset(rho)
    x = data.feature_builder.genes
    m = data.metadata
    np.testing.assert_allclose(expit(x @ m["true_coefficient"] + m["true_intercept"]),
                               m["true_probability"], atol=1e-12)
    np.testing.assert_allclose(m["true_probability"].mean(0), m["requested_prevalence"], atol=1e-12)
    shared, specific = x @ m["shared_coefficient"], x @ m["specific_coefficient"]
    np.testing.assert_allclose(shared.var(0), 1, atol=1e-12)
    np.testing.assert_allclose(specific.var(0), 1, atol=1e-12)
    np.testing.assert_allclose((shared * specific).mean(0), 0, atol=1e-12)
    mixture = rho**0.5 * shared + (1-rho)**0.5 * specific
    np.testing.assert_allclose((rho**0.5 * shared).var(0) / mixture.var(0), rho, atol=1e-12)
    if rho == 1:
        assert np.linalg.matrix_rank(m["true_coefficient"]) == 2
    elif rho == 0:
        assert np.linalg.matrix_rank(m["true_coefficient"]) > 2


def test_complete_spatial_folds_are_within_one_independent_repetition():
    data = _dataset(repetition=3)
    folds = data.split_builder(3, 11)
    visits = np.zeros(65, dtype=int)
    for fold in folds:
        fold.validate(65)
        visits[fold.test_rows] += 1
        slices = fold.metadata["test_slices"]
        assert slices == list(range(min(slices), max(slices) + 1))
        assert fold.metadata["independent_unit"] == "generated_dataset"
        assert fold.metadata["repetition"] == 3
    np.testing.assert_array_equal(visits, 1)
    assert data.metadata["fold_aggregation"] == "aggregate_within_repetition_before_uncertainty"
    assert not np.array_equal(data.reference, _dataset(repetition=4).reference)


def test_pairing_across_rho_and_pickleable_builders():
    a, b = _dataset(0), _dataset(1)
    np.testing.assert_array_equal(a.feature_builder.genes, b.feature_builder.genes)
    np.testing.assert_array_equal(a.metadata["latent_uniform"], b.metadata["latent_uniform"])
    np.testing.assert_array_equal(a.technical_score, b.technical_score)
    restored = pickle.loads(pickle.dumps(a))
    f = restored.split_builder(3, 0)[0]
    assert restored.feature_builder(f.train_rows, False, False).X.shape == (65, 6)


def test_switches_fail_explicitly_when_requested_covariates_do_not_exist():
    data = _dataset(n_target_features=0, n_location_features=0)
    rows = data.split_builder(3, 0)[0].train_rows
    with pytest.raises(ValueError, match="location"):
        data.feature_builder(rows, True, False)
    with pytest.raises(ValueError, match="target descriptors"):
        data.feature_builder(rows, False, True)
    with pytest.raises(ValueError, match="sharing_strength"):
        _dataset(1.1)


@pytest.mark.parametrize("objective", ["logit", "squared_error"])
def test_qiao_masked_analytic_gradient(objective):
    rng = np.random.default_rng(51)
    xb, yb = rng.normal(size=(7, 4)), rng.normal(size=(5, 3))
    d = rng.integers(0, 2, size=(7, 5)).astype(float)
    w = rng.uniform(size=d.shape) > 0.3
    size = (4 + 3) * 2 + (5 if objective == "logit" else 0)
    theta = rng.normal(scale=0.2, size=size)
    args = (xb, yb, d, w, 2, 0.07, objective)
    error = check_grad(lambda z: _objective_gradient(z, *args)[0],
                       lambda z: _objective_gradient(z, *args)[1], theta)
    assert error < 2e-6


@pytest.mark.parametrize("objective", ["logit", "squared_error"])
def test_qiao_ignores_unassayed_labels_and_adapts_feature_dimensions(objective):
    rng = np.random.default_rng(52)
    x, y = rng.normal(size=(28, 3)), rng.normal(size=(5, 2))
    d = rng.integers(0, 2, size=(28, 5)).astype(float)
    w = rng.uniform(size=d.shape) > 0.2
    missing_nan, missing_extreme = d.copy(), d.copy()
    missing_nan[~w], missing_extreme[~w] = np.nan, 100.0
    kwargs = dict(rank=2, objective=objective, seed=11, maxiter=100)
    first = fit_qiao(x, y, missing_nan, w, **kwargs)
    second = fit_qiao(x, y, missing_extreme, w, **kwargs)
    np.testing.assert_allclose(first.predict_proba(x), second.predict_proba(x), atol=0, rtol=0)
    assert first.predict_proba(x[:4]).shape == (4, 5)
    assert first.n_measured == w.sum()
    assert np.all(np.isfinite(first.predict_proba(x)))
    assert first.probability_semantics != "latent_projection_probability"
    with pytest.raises(ValueError, match="cell feature dimension"):
        first.predict_proba(np.zeros((4, 6)))
    with pytest.raises(ValueError, match="target descriptor"):
        fit_qiao(x, np.empty((5, 0)), d, w, **kwargs)


def test_qiao_selection_uses_observed_validation_and_unique_candidates():
    rng = np.random.default_rng(53)
    x, y = rng.normal(size=(30, 3)), rng.normal(size=(5, 2))
    d = rng.integers(0, 2, size=(30, 5))
    candidates = [{"rank": 1, "l2": 0.03}, {"rank": 1, "l2": 0.03}, {"rank": 2, "l2": 0.03}]
    result = tune_qiao(x[:20], y, d[:20], x[20:], d[20:], candidates=candidates, maxiter=100)
    assert len(result.trials) == 2
    assert result.selected_fit.n_measured == 20 * 5
    assert all(t.selection_metric == "observed_log_loss" for t in result.trials)
    assert result.selected_config["objective"] == "logit"
