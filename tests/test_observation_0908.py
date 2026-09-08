"""Statistical boundaries of the shared 0908 observation/supervision protocol."""

import numpy as np
import pytest
from scipy.special import expit

from gene2wire import DatasetBundle, FitConfig, ModelConfig
from gene2wire.models import UnifiedPUModel
from gene2wire.experiments.observation import make_observation_design, thin_reference
from gene2wire.experiments.calibration import (
    CalibrationNotEstimable, detection_diagnostics, fit_detection_calibrator,
    sample_paired_rows,
)
from gene2wire.experiments.supervision import compile_training_bundle


@pytest.mark.parametrize("mechanism", ["scar", "target_sar", "technical_sar"])
def test_fold_local_normalization_nested_masks_and_unassayed_entries(mechanism):
    rng = np.random.default_rng(23)
    z = rng.random((200, 4)) < 0.4
    w = rng.random(z.shape) < 0.8
    train = np.arange(90)
    design = make_observation_design(*z.shape, seed=6)
    previous = w & z
    for loss_rate in (0, .2, .4, .6, .8):
        result = thin_reference(z, w, train, loss_rate, mechanism, design)
        assert not np.any(result.observed & ~previous)
        assert not np.any(result.observed & ~w)
        assert not np.any(result.observed & ~z)
        np.testing.assert_allclose(result.expected_train_retention, 1 - loss_rate, atol=1e-11)
        # Test outcomes cannot affect gamma, training D, or any propensity.
        altered = z.copy()
        altered[90:] = ~altered[90:]
        alternate = thin_reference(altered, w, train, loss_rate, mechanism, design)
        np.testing.assert_array_equal(result.sensitivity, alternate.sensitivity)
        np.testing.assert_array_equal(result.observed[train], alternate.observed[train])
        previous = result.observed
    # Independent folds really use their own positive covariate distributions.
    if mechanism == "technical_sar":
        other = thin_reference(z, w, np.arange(90, 200), .8, mechanism, design)
        assert abs(result.gamma - other.gamma) > 1e-6


def test_old_mechanisms_not_silently_redefined():
    design = make_observation_design(3, 1, 7)
    with pytest.raises(ValueError, match="archived designs"):
        thin_reference(np.ones((3, 1)), np.ones((3, 1)), [0, 1], .8, "technical_sar_old", design)
    with pytest.raises(ValueError, match="no reference positives"):
        thin_reference(np.zeros((3, 1)), np.ones((3, 1)), [0, 1], .8, "scar", design)


def test_paired_sampling_exact_budget_nested_and_order_independent():
    eligible = np.arange(100, 200)
    groups = np.repeat(["a", "b", "c"], [40, 30, 30])
    selected = [sample_paired_rows(eligible, f, 7, groups) for f in (.1, .2, .4)]
    assert [len(s) for s in selected] == [10, 20, 40]
    assert set(selected[0]) <= set(selected[1]) <= set(selected[2])
    reversed_rows = sample_paired_rows(eligible[::-1], .2, 7, groups[::-1])
    np.testing.assert_array_equal(selected[1], reversed_rows)
    assert len(set(groups[np.isin(eligible, selected[0])])) == 3


def test_detector_learns_from_paired_only_and_technical_omission_is_real():
    rng = np.random.default_rng(11)
    n, t = 2500, 3
    u = rng.normal(size=n)
    truth = expit(-1 + np.array([-.7, 0, .7])[None, :] + 1.2 * u[:, None])
    reference = np.ones((n, t))
    measured = np.ones((n, t), dtype=bool)
    observed = rng.random((n, t)) < truth
    paired = np.arange(1600)
    model = fit_detection_calibrator(observed, reference, measured, paired, technical_score=u)
    prediction = model.predict(n, technical_score=u)
    assert np.mean(np.abs(prediction[1600:] - truth[1600:])) < .04
    hidden_reference = reference.copy()
    hidden_reference[1600:] = np.nan
    hidden_d = observed.astype(float)
    hidden_d[1600:] = np.nan
    altered = fit_detection_calibrator(hidden_d, hidden_reference, measured, paired, technical_score=u)
    np.testing.assert_array_equal(model.coefficients, altered.coefficients)
    omitted = fit_detection_calibrator(observed, reference, measured, paired, technical_score=u, use_technical=False)
    omitted_pred = omitted.predict(n)
    np.testing.assert_array_equal(omitted_pred[0], omitted_pred[-1])
    good = detection_diagnostics(observed, reference, measured, prediction, rows=np.arange(1600, n), true_sensitivity=truth)
    bad = detection_diagnostics(observed, reference, measured, omitted_pred, rows=np.arange(1600, n), true_sensitivity=truth)
    assert good["detection_log_loss"] < bad["detection_log_loss"] - .04
    assert good["sensitivity_rmse"] < bad["sensitivity_rmse"]


def test_calibration_sparse_targets_platform_fallback_and_not_estimable():
    reference = np.ones((80, 3))
    reference[:, 2] = 0
    measured = np.ones_like(reference, dtype=bool)
    observed = np.zeros_like(reference)
    observed[40:, :2] = 1
    platform = np.repeat(["low", "high"], 40)
    fit = fit_detection_calibrator(observed, reference, measured, np.arange(80), platform=platform)
    prediction = fit.predict(3, platform=["low", "high", "unseen"])
    assert np.all(np.isfinite(prediction)) and np.all((prediction > 0) & (prediction < 1))
    assert prediction[0, 0] < prediction[1, 0]
    assert prediction[0, 0] < prediction[2, 0] < prediction[1, 0]
    assert fit.diagnostics["targets_without_paired_positives"] == 1
    with pytest.raises(CalibrationNotEstimable):
        fit_detection_calibrator(observed * 0, reference * 0, measured, np.arange(80))
    with pytest.raises(ValueError, match="requires the declared technical_score"):
        fit_detection_calibrator(observed, reference, measured, np.arange(80), use_technical=True)


@pytest.mark.parametrize("detected", [0, 1])
def test_detector_regularization_keeps_single_class_paired_samples_finite(detected):
    reference = np.ones((12, 2))
    observed = np.full_like(reference, detected)
    fit = fit_detection_calibrator(observed, reference, np.ones_like(reference), np.arange(12))
    probabilities = fit.predict(3)
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert np.all(probabilities < .5) if detected == 0 else np.all(probabilities > .5)


def supervision_example():
    x = np.array([[.2, -.1], [1., .3], [-.4, .8], [.6, -.5], [.4, .9]])
    z = np.array([[1, 1], [0, 1], [1, 0], [0, 0], [1, 1]])
    d = np.array([[1, 0], [0, 1], [0, 0], [0, 0], [1, 0]])
    w = np.ones_like(z, dtype=bool)
    w[1, 0] = False
    return DatasetBundle(x, d, w, Z_reference=z), z


def test_mixed_compiler_matches_disjoint_core_likelihood():
    bundle, reference = supervision_example()
    e = np.full(reference.shape, .4)
    paired = np.array([0, 2])
    train = np.arange(4)
    compiled, exposure = compile_training_bundle(bundle, train, paired, reference, e, "reference_plus_pu")
    assert compiled.Z_reference is None and compiled.reference_mask is None
    assert not compiled.W_measured[4].any()
    np.testing.assert_array_equal(exposure[paired], 1)
    assert compiled.S_observed[0, 1]  # A hidden paired positive becomes supervised.
    config = ModelConfig("mixed", "direct", pu=True)
    model = UnifiedPUModel(config, FitConfig(initialization="random"))
    coefficients = np.array([[.1, -.2], [.3, .15]])
    intercept = np.array([-.7, -1.1])
    theta = np.r_[coefficients.ravel(), intercept]
    core_loss = model._objective_gradient(theta, compiled.X_cell, compiled.S_observed.astype(float), compiled.W_measured, exposure, None)[0]
    p = expit(bundle.X_cell @ coefficients + intercept)
    terms = []
    for row in train:
        for target in range(2):
            if bundle.W_measured[row, target]:
                y = float(reference[row, target] if row in paired else bundle.S_observed[row, target])
                probability = p[row, target] if row in paired else e[row, target] * p[row, target]
                terms.append(-y * np.log(probability) - (1 - y) * np.log1p(-probability))
    np.testing.assert_allclose(core_loss, np.mean(terms), atol=1e-12)


def test_reference_only_never_falls_back_to_evaluation_truth():
    bundle, reference = supervision_example()
    with pytest.raises(ValueError, match="explicit paired_reference"):
        compile_training_bundle(bundle, np.arange(4), [0, 2], mode="reference_only")
    with pytest.raises(ValueError, match="authorized train_rows"):
        compile_training_bundle(bundle, np.arange(4), [4], reference, mode="reference_only")
    blinded = np.full(reference.shape, np.nan)
    blinded[[0, 2]] = reference[[0, 2]]
    first, e = compile_training_bundle(bundle, np.arange(4), [0, 2], blinded, mode="reference_only")
    expected = np.zeros_like(reference, dtype=bool)
    expected[[0, 2]] = bundle.W_measured[[0, 2]]
    np.testing.assert_array_equal(first.W_measured, expected)
    np.testing.assert_array_equal(e, 1)
    assert first.Z_reference is None
    # Compact authorized references give exactly the same training matrix.
    second, _ = compile_training_bundle(bundle, np.arange(4), [0, 2], reference[[0, 2]], mode="reference_only")
    np.testing.assert_array_equal(first.S_observed, second.S_observed)


def test_calibrated_pu_does_not_open_reference_labels():
    bundle, reference = supervision_example()
    poisoned = np.full(reference.shape, np.nan)
    compiled, e = compile_training_bundle(bundle, np.arange(4), [0, 2], poisoned, np.full(reference.shape, .5), "calibrated_pu")
    np.testing.assert_array_equal(compiled.S_observed[:4], bundle.S_observed[:4])
    np.testing.assert_array_equal(e[:4][compiled.W_measured[:4]], .5)
