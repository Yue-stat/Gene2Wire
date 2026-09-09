"""Progress must describe actual reusable fits, without changing their results."""
import json

import numpy as np
import pytest

from gene2wire.experiments import baselines


def _inputs():
    rng = np.random.default_rng(901)
    x = rng.normal(size=(18, 3))
    y = np.column_stack((x[:, 0] > 0, x[:, 1] > 0)).astype(float)
    mask = np.ones_like(y, dtype=bool)
    return (x[:10], y[:10], mask[:10], x[10:14], y[10:14], mask[10:14],
            0.7, x[14:], 0.7)


def _configs():
    return [{"n_estimators": 4, "min_samples_leaf": leaf, "max_features": 1.0}
            for leaf in (1, 2, 3)]


def test_baseline_progress_reports_fits_without_checkpoint_storage():
    arguments = _inputs()
    events = []
    plain = baselines.fit_baseline(*arguments, candidate_configs=_configs())
    reported = baselines.fit_baseline(*arguments, candidate_configs=_configs(),
                                      on_progress=events.append)
    np.testing.assert_array_equal(reported.prediction, plain.prediction)
    assert reported.candidate_records == plain.candidate_records
    assert events[0] == {"event": "candidate_inventory", "stage": "tuning", "total": 3,
                         "cached": 0, "memory_cached": 0, "checkpoint_cached": 0, "pending": 3}
    assert [event["event"] for event in events] == [
        "candidate_inventory", "candidate_start", "candidate_complete",
        "candidate_start", "candidate_complete", "candidate_start", "candidate_complete",
        "refit_start", "refit_complete"]
    starts = [event for event in events if event["event"] == "candidate_start"]
    assert [event["index"] for event in starts] == [1, 2, 3]
    assert [event["config"] for event in starts] == _configs()
    assert all(event["cache_status"] == "pending" for event in starts)
    completions = [event for event in events if event["event"].endswith("_complete")]
    assert all(event["cache_status"] == "fitted" for event in completions)
    assert all(event["elapsed_seconds"] >= 0 for event in completions)


def test_baseline_progress_reuses_existing_callback_free_checkpoints(tmp_path, monkeypatch):
    arguments = _inputs()
    first = baselines.fit_baseline(*arguments, candidate_configs=_configs(), checkpoint_dir=tmp_path)
    events = []

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Complete verified caches must not trigger fitting")

    monkeypatch.setattr(baselines, "_fit_predict", forbidden_fit)
    repeated = baselines.fit_baseline(*arguments, candidate_configs=_configs(), checkpoint_dir=tmp_path,
                                      on_progress=events.append)
    np.testing.assert_array_equal(repeated.prediction, first.prediction)
    assert events[0]["cached"] == events[0]["checkpoint_cached"] == 3
    assert events[0]["pending"] == 0
    assert all(event["cache_status"] == "checkpoint" for event in events[1:])


@pytest.mark.parametrize("corruption", ["checksum", "truncated_zip"])
def test_baseline_inventory_excludes_corrupt_candidate_caches(tmp_path, corruption):
    arguments = _inputs()
    first = baselines.fit_baseline(*arguments, candidate_configs=_configs(), checkpoint_dir=tmp_path)
    x, y, mask, xv = arguments[:4]
    identity = baselines._fit_identity(x, y, mask, xv, kind="random_forest", config=_configs()[1], seed=0)
    path = tmp_path / f"baseline_{identity}.npz"
    assert path.exists()
    if corruption == "checksum":
        with np.load(path, allow_pickle=False) as saved:
            prediction, metadata = saved["prediction"], saved["metadata"]
        assert json.loads(str(metadata.item()))["identity"] == identity
        np.savez_compressed(path, prediction=1 - prediction, metadata=metadata)
    else:
        path.write_bytes(b"PK\x03\x04not a complete zip archive")
    # Unrelated files must not inflate the count either.
    (tmp_path / "baseline_unrelated.npz").write_bytes(b"unrelated")
    events = []
    repeated = baselines.fit_baseline(*arguments, candidate_configs=_configs(), checkpoint_dir=tmp_path,
                                      on_progress=events.append)
    assert events[0]["cached"] == events[0]["checkpoint_cached"] == 2
    assert events[0]["pending"] == 1
    starts = [event for event in events if event["event"] == "candidate_start"]
    assert [event["cache_status"] for event in starts] == ["checkpoint", "pending", "checkpoint"]
    assert [row["resumed"] for row in repeated.candidate_records] == [True, False, True]
    np.testing.assert_array_equal(repeated.prediction, first.prediction)
