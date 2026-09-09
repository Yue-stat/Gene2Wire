"""Reuse fitted predictions without reusing a different scenario's selection."""
from concurrent.futures import ThreadPoolExecutor
import os
from threading import Barrier
from unittest.mock import patch

import numpy as np
import pytest

from gene2wire.experiments import baselines


def test_shared_predictions_still_reselect_on_current_validation_labels(tmp_path, monkeypatch):
    def fitted(x, y, mask, x_predict, *, kind, config, seed):
        probability = .2 if config["min_samples_leaf"] == 1 else .8
        return np.full((len(x_predict), y.shape[1]), probability), {}

    monkeypatch.setattr(baselines, "_fit_predict", fitted)
    x = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([[0], [1], [0], [1], [0], [1]])
    mask = np.ones_like(labels, dtype=bool)
    configs = [{"n_estimators": 8, "min_samples_leaf": leaf} for leaf in (1, 3)]
    arguments = dict(X_train=x, labels=labels, measured=mask, X_validation=x[:2],
        validation_measured=mask[:2], validation_sensitivity=1., X_test=x[:2],
        test_sensitivity=1., probability_semantics="reference",
        candidate_configs=configs, checkpoint_dir=tmp_path)
    first = baselines.fit_baseline(validation_observed=np.zeros((2, 1)), **arguments)
    with patch.object(baselines, "_fit_predict", side_effect=AssertionError("identical RF refitted")):
        second = baselines.fit_baseline(validation_observed=np.ones((2, 1)), **arguments)
    assert first.selected_config["min_samples_leaf"] == 1
    assert second.selected_config["min_samples_leaf"] == 3
    assert all(row["resumed"] for row in second.candidate_records)
    np.testing.assert_array_equal(first.prediction, .2)
    np.testing.assert_array_equal(second.prediction, .8)


@pytest.mark.skipif(os.name != "posix", reason="OnDemand deduplication uses POSIX advisory locks")
def test_concurrent_identical_rf_fits_compute_once(tmp_path, monkeypatch):
    calls = []
    gate = Barrier(2)

    def fitted(x, y, mask, x_predict, **kwargs):
        calls.append(1)
        return np.full((len(x_predict), y.shape[1]), .4), {}

    monkeypatch.setattr(baselines, "_fit_predict", fitted)
    x = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([[0], [1], [0], [1], [0], [1]])

    def run():
        gate.wait(timeout=10)
        return baselines._cached_fit_predict(x, labels, np.ones_like(labels, dtype=bool), x[:2],
            kind="random_forest", config={"n_estimators": 8}, seed=1, checkpoint_dir=tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]
    assert len(calls) == 1
    assert sorted(result[2] for result in results) == [False, True]
    np.testing.assert_array_equal(results[0][0], results[1][0])
