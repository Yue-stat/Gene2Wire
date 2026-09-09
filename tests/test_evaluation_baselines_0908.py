"""Scientific invariants for the shared paper evaluation and baseline adapters."""
from pathlib import Path

import numpy as np
import pytest

from gene2wire.experiments.baselines import fit_baseline
from gene2wire.experiments.evaluation import (
    evaluate_detection_calibration,
    evaluate_predictions,
    expected_top_k_recall,
)


def test_tied_cutoff_recall_is_fractional_and_permutation_invariant():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.8, 0.5, 0.5, 0.2])
    assert expected_top_k_recall(y, score, 2) == pytest.approx(0.75)
    permutation = [2, 3, 1, 0]
    assert expected_top_k_recall(y[permutation], score[permutation], 2) == pytest.approx(0.75)
    assert np.isnan(expected_top_k_recall(np.zeros(4), score, 2))


def test_constant_target_sensitivity_preserves_within_target_hidden_ranking():
    z = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 1]])
    d = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]])
    p = np.array([[0.8, 0.2], [0.2, 0.8], [0.6, 0.6], [0.4, 0.4], [0.9, 0.9]])
    result = evaluate_predictions(z, d, np.ones_like(z), p, np.array([0.2, 0.7]))
    for row in result["per_target"]:
        assert row["hidden_recall_at_h_p_ranking"] == row["hidden_recall_at_h_h_ranking"]
        assert row["hidden_auprc_p_ranking"] == row["hidden_auprc_h_ranking"]
    np.testing.assert_allclose(result["scores"]["q"], p * [0.2, 0.7])


def test_feature_dependent_detection_changes_fixed_predictor_recovery_scoring():
    z = np.array([[1], [0], [1]])
    d = np.array([[0], [0], [1]])
    p = np.array([[0.6], [0.4], [0.8]])
    e = np.array([[0.9], [0.1], [0.5]])
    summary = evaluate_predictions(z, d, np.ones_like(z), p, e)["summary"]
    assert summary["hidden_recall_at_h_p_ranking"] == 1.0
    assert summary["hidden_recall_at_h_h_ranking"] == 0.0
    assert summary["hidden_recall_at_h"] == 0.0
    observed = evaluate_predictions(z, d, np.ones_like(z), p, e,
                                    probability_semantics="observed")
    assert observed["summary"]["hidden_recall_at_h"] == 1.0
    assert np.isnan(observed["summary"]["hidden_recall_at_h_h_ranking"])
    assert observed["scores"]["p"] is None
    np.testing.assert_array_equal(observed["scores"]["q"], p)


def test_mixed_label_score_is_not_rescaled_or_given_bayes_hidden_ranking():
    z, d = np.array([[1], [0], [1]]), np.array([[0], [0], [1]])
    score = np.array([[.6], [.4], [.8]])
    result = evaluate_predictions(z, d, np.ones_like(z), score, np.array([[.9], [.1], [.5]]),
                                  probability_semantics="mixed")
    assert result["summary"]["hidden_recall_at_h"] == 1.
    assert np.isnan(result["summary"]["hidden_recall_at_h_h_ranking"])
    assert result["summary"]["macro_log_loss"] == pytest.approx(-np.log([.6, .6, .8]).mean())
    assert all(result["scores"][key] is None for key in ("p", "q", "h"))
    np.testing.assert_array_equal(result["scores"]["mixed_label_score"], score)


def test_mixed_rf_uses_raw_score_on_observed_validation_and_resumes(tmp_path):
    x = np.arange(24, dtype=float).reshape(12, 2)
    labels = (np.arange(12) % 3 == 0).astype(int)[:, None]
    d = np.array([[0], [1], [0]])
    args = (x, labels, np.ones_like(labels), x[:3], d, np.ones_like(d), .2, x[3:6], .3)
    config = [{"n_estimators": 8, "min_samples_leaf": 2, "max_features": 1.}]
    mixed = fit_baseline(*args, probability_semantics="mixed", candidate_configs=config,
                         checkpoint_dir=tmp_path)
    # The two semantics have the same raw-score selection rule and RF fitting.
    observed = fit_baseline(*args, probability_semantics="observed", candidate_configs=config,
                            checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(mixed.prediction, observed.prediction)
    assert mixed.candidate_records[0]["validation_observed_log_loss"] == observed.candidate_records[0]["validation_observed_log_loss"]
    assert mixed.observed_prediction is None
    assert mixed.diagnostics["validation_score_semantics"] == "raw_mixed_label_score"
    assert observed.diagnostics["final_fit_resumed"] is True


def test_zero_hidden_and_single_class_are_not_reported_as_zero_discrimination():
    z = np.array([[1, 0], [1, 0], [1, 0]])
    result = evaluate_predictions(z, z, np.ones_like(z), np.full(z.shape, 0.5), 1)
    assert np.isnan(result["summary"]["macro_auprc"])
    assert np.isnan(result["summary"]["macro_auroc"])
    assert np.isnan(result["summary"]["hidden_recall_at_h"])
    assert result["summary"]["n_targets_hidden_recall_at_h"] == 0
    assert result["summary"]["n_targets_auprc"] == 0
    assert result["summary"]["macro_brier"] == 0.25


def test_off_panel_values_are_ignored_and_measured_false_positive_is_rejected():
    z = np.array([[1, 99], [0, np.nan]])
    d = np.array([[0, 100], [0, np.nan]])
    w = np.array([[1, 0], [1, 0]])
    p = np.array([[0.8, np.nan], [0.2, np.nan]])
    summary = evaluate_predictions(z, d, w, p, 0.5)["summary"]
    assert summary["n_measured"] == 2
    assert summary["macro_auprc"] == 1.0
    d[1, 0] = 1
    with pytest.raises(ValueError, match="every detection"):
        evaluate_predictions(z, d, w, p, 0.5)


def test_brier_skill_uses_provided_training_prevalence_only():
    z = np.array([[1], [0]])
    p = np.array([[0.8], [0.2]])
    supplied = evaluate_predictions(z, z, np.ones_like(z), p, 1,
                                    train_reference_prevalence=np.array([0.2]))
    expected_baseline = ((1 - 0.2) ** 2 + (0 - 0.2) ** 2) / 2
    assert supplied["summary"]["micro_baseline_brier"] == pytest.approx(expected_baseline)
    assert supplied["summary"]["macro_brier_skill"] == pytest.approx(1 - 0.04 / expected_baseline)
    missing = evaluate_predictions(z, z, np.ones_like(z), p, 1)
    assert np.isnan(missing["summary"]["macro_brier_skill"])
    assert missing["summary"]["n_entries_brier_skill"] == 0


def test_detection_errors_use_reference_positives_and_no_oracle_on_natural_data():
    z, d = np.array([[1], [0], [1]]), np.array([[1], [0], [0]])
    e = np.array([[0.8], [0.1], [0.4]])
    true = np.array([[0.9], [0.9], [0.2]])
    result = evaluate_detection_calibration(z, d, np.ones_like(z), e, true_sensitivity=true)
    assert result["summary"]["sensitivity_mae"] == pytest.approx(0.15)
    assert result["summary"]["sensitivity_rmse"] == pytest.approx(np.sqrt(0.025))
    assert sum(row["n"] for row in result["reliability"] if row["target"] == "0") == 2
    natural = evaluate_detection_calibration(z, d, np.ones_like(z), e)
    assert np.isnan(natural["summary"]["sensitivity_mae"])


def test_unbounded_bilinear_ranking_is_not_destroyed_by_probability_clipping():
    z = np.array([[1], [0]])
    d = np.zeros_like(z)
    raw_score = np.array([[3.0], [2.0]])
    probability = np.clip(raw_score, 0, 1)
    without = evaluate_predictions(z, d, np.ones_like(z), probability, .5,
                                   probability_semantics="observed")
    with_raw = evaluate_predictions(z, d, np.ones_like(z), probability, .5,
                                    probability_semantics="observed", ranking_score=raw_score)
    assert without["summary"]["macro_auprc"] == .5
    assert with_raw["summary"]["macro_auprc"] == 1.
    assert with_raw["summary"]["hidden_recall_at_h"] == 1.
    assert with_raw["summary"]["macro_log_loss"] == without["summary"]["macro_log_loss"]
    np.testing.assert_array_equal(with_raw["scores"]["ranking_score"], raw_score)
    with pytest.raises(ValueError, match="only for observed"):
        evaluate_predictions(z, d, np.ones_like(z), probability, .5,
                             probability_semantics="reference", ranking_score=raw_score)


def _baseline_inputs():
    x = np.arange(24, dtype=float).reshape(12, 2)
    y = np.column_stack([np.zeros(12), np.ones(12), np.zeros(12), np.arange(12) % 2])
    w = np.ones_like(y, dtype=bool)
    w[:, 2] = False
    return x, y, w


def test_rf_masked_training_single_class_and_empty_target_fallback(tmp_path):
    x, y, w = _baseline_inputs()
    y[:, 2] = 999  # Forbidden label contents must have no influence.
    dv = np.tile([0, 1, 0, 1], (3, 1))
    settings = [{"n_estimators": 8, "min_samples_leaf": 1, "max_features": 1.0}]
    first = fit_baseline(x, y, w, x[:3], dv, np.ones_like(dv), 0.5, x[3:6], 0.5,
                        candidate_configs=settings, checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(first.prediction[:, 0], 0)
    np.testing.assert_array_equal(first.prediction[:, 1], 1)
    np.testing.assert_array_equal(first.prediction[:, 2], 0.5)
    assert first.diagnostics["empty_training_targets"] == [2]
    y[:, 2] = -999
    second = fit_baseline(x, y, w, x[:3], dv, np.ones_like(dv), 0.5, x[3:6], 0.5,
                         candidate_configs=settings, checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(first.prediction, second.prediction)
    assert second.candidate_records[0]["resumed"] is True
    assert second.diagnostics["final_fit_resumed"] is True


def test_reference_baseline_selects_with_observed_validation_and_refits_authorized_view():
    x = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([[0], [0], [1], [1], [1], [1]])
    w = np.ones_like(labels)
    dv = np.array([[0], [1]])
    result = fit_baseline(x, labels, w, x[:2], dv, np.ones_like(dv), 0.3,
                          x[2:4], 0.4, kind="prevalence", probability_semantics="reference",
                          refit_X=x, refit_labels=np.ones_like(labels), refit_measured=w)
    # Selection uses q=e*p=0.3*(2/3)=0.2, without validation references.
    assert result.candidate_records[0]["validation_observed_log_loss"] == pytest.approx(-0.5 * np.log(0.2 * 0.8))
    np.testing.assert_array_equal(result.prediction, 1)
    np.testing.assert_array_equal(result.observed_prediction, 0.4)
    assert result.diagnostics["n_candidates_evaluated"] == 1


def test_baseline_refuses_partial_refit_and_nested_parallel_override():
    x, y, w = _baseline_inputs()
    dv = np.tile([0, 1, 0, 1], (3, 1))
    with pytest.raises(ValueError, match="together"):
        fit_baseline(x, y, w, x[:3], dv, np.ones_like(dv), 1, x[:3], 1,
                     kind="prevalence", refit_X=x)
    with pytest.raises(ValueError, match="cannot override"):
        fit_baseline(x, y, w, x[:3], dv, np.ones_like(dv), 1, x[:3], 1,
                     candidate_configs=[{"n_jobs": 4}])


def test_corrupt_baseline_prediction_checkpoint_is_recomputed(tmp_path):
    x, y, w = _baseline_inputs()
    dv = np.tile([0, 1, 0, 1], (3, 1))
    arguments = (x, y, w, x[:3], dv, np.ones_like(dv), 1, x[3:6], 1)
    result = fit_baseline(*arguments, kind="prevalence", checkpoint_dir=tmp_path)
    for path in Path(tmp_path).glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            metadata, prediction = data["metadata"], data["prediction"]
        np.savez_compressed(path, metadata=metadata, prediction=prediction + 0.1)
    repeated = fit_baseline(*arguments, kind="prevalence", checkpoint_dir=tmp_path)
    np.testing.assert_array_equal(repeated.prediction, result.prediction)
    assert repeated.diagnostics["final_fit_resumed"] is False
