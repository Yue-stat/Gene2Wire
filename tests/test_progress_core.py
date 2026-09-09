"""Progress inventories describe actual reusable fits without altering results."""

from unittest.mock import patch

import numpy as np
import pytest

from gene2wire import DatasetBundle, FitConfig, ModelConfig, TuningConfig, run_model_grid
from gene2wire.checkpoint import AtomicCheckpointStore
from gene2wire.tuning import tune_model


def _inputs():
    rng = np.random.default_rng(127)
    x = rng.normal(size=(64, 3))
    y = rng.random((64, 2)) < 0.4

    def bundle(start, stop):
        return DatasetBundle(x[start:stop], y[start:stop], np.ones_like(y[start:stop]),
                             cell_ids=[f"cell_{i}" for i in range(start, stop)],
                             target_ids=["a", "b"])

    return bundle(0, 36), bundle(36, 52), x[52:]


def _tuning():
    return TuningConfig(ranks=(0, 1), shared_l2=(0.1,), residual_l2=(0.1, 1.0),
                        target_l2=(0.1,))


def _fit():
    return FitConfig(maxiter=300, tolerance=1e-7, initialization="random")


def test_partial_candidate_inventory_uses_compatible_checkpoints(tmp_path):
    train, validation, _ = _inputs()
    events, old_callback = [], []
    kwargs = dict(train=train, validation=validation, train_exposure=1.0,
                  validation_exposure=1.0, base_model=ModelConfig(name="PU", kind="direct", pu=True),
                  tuning=_tuning(), fit=_fit(), seed=17,
                  checkpoint_store=AtomicCheckpointStore(tmp_path),
                  checkpoint_fingerprint="caller")

    def stop_after_candidate(event):
        events.append(event)
        if event["event"] == "candidate_complete":
            raise InterruptedError("simulate notebook interruption")

    with pytest.raises(InterruptedError):
        tune_model(**kwargs, on_progress=stop_after_candidate)
    events.clear()
    resumed = tune_model(**kwargs, on_progress=events.append, on_trial=old_callback.append)
    inventory = next(e for e in events if e["event"] == "candidate_inventory")
    assert (inventory["total"], inventory["checkpoint_cached"], inventory["pending"]) == (2, 1, 1)
    complete = [e for e in events if e["event"] == "candidate_complete"]
    assert [e["cache_status"] for e in complete] == ["checkpoint", "fitted"]
    assert all(e["elapsed_seconds"] >= 0 for e in complete)
    assert old_callback == list(resumed.trials)

    # Changing exposure invalidates the old trials even though files remain.
    events.clear()
    tune_model(**{**kwargs, "train_exposure": 0.8}, on_progress=events.append)
    inventory = next(e for e in events if e["event"] == "candidate_inventory")
    assert inventory["cached"] == 0 and inventory["pending"] == 2


def test_refit_memory_and_completed_model_events_are_exact(tmp_path):
    train, validation, test_x = _inputs()
    models = (ModelConfig(name="PU", kind="direct", pu=True),
              ModelConfig(name="PU alias", kind="direct", pu=True))
    kwargs = dict(train=train, validation=validation, test_X=test_x, models=models,
                  tuning=_tuning(), fit=_fit(), checkpoint_dir=tmp_path, seed=17)

    def stop_after_refit(event):
        if event["event"] == "refit_complete":
            raise InterruptedError("refit complete but model export interrupted")

    with pytest.raises(InterruptedError):
        run_model_grid(**kwargs, on_progress=stop_after_refit)
    events, old_callback = [], []
    with patch("gene2wire.models.UnifiedPUModel.fit", side_effect=AssertionError("unexpected refit")):
        result = run_model_grid(**kwargs, on_progress=events.append,
                                on_model=lambda name, status: old_callback.append((name, status)))
    inventories = [e for e in events if e["event"] == "candidate_inventory"]
    assert [(e["checkpoint_cached"], e["memory_cached"], e["pending"])
            for e in inventories] == [(2, 0, 0), (0, 2, 0)]
    assert [e["cache_status"] for e in events if e["event"] == "refit_complete"] == ["checkpoint", "memory"]
    assert old_callback == [("PU", "started"), ("PU", "completed"),
                            ("PU alias", "started"), ("PU alias", "completed")]
    assert all(e["summary"]["rank"] == 0 for e in events if e["event"] == "model_complete")

    # A fully completed model skips candidate and refit work altogether.
    events.clear()
    with patch("gene2wire.runner.tune_model", side_effect=AssertionError("unexpected tuning")):
        resumed = run_model_grid(**kwargs, on_progress=events.append)
    assert [e["event"] for e in events] == ["model_start", "model_complete"] * 2
    assert all(e["cache_status"] == "checkpoint" for e in events)
    assert all(e["resumed"] for e in events if e["event"] == "model_complete")

    # Diagnostics cannot alter selected configurations or predictions.
    plain = run_model_grid(**{**kwargs, "checkpoint_dir": None})
    for name in result.models:
        assert result.models[name].tuning == plain.models[name].tuning
        np.testing.assert_array_equal(result.models[name].latent_probability,
                                      plain.models[name].latent_probability)
        np.testing.assert_array_equal(resumed.models[name].latent_probability,
                                      plain.models[name].latent_probability)


def test_staged_progress_reports_separate_known_stage_inventories():
    train, validation, _ = _inputs()
    events = []
    tuned = tune_model(train, validation, 1.0, 1.0,
                       ModelConfig(name="MIRT", kind="lowrank", rank=1, pu=True),
                       TuningConfig(strategy="staged_rank_l2", ranks=(1, 2),
                                    shared_l2=(0.1, 1.0), residual_l2=(0.1,),
                                    anchor_shared_l2=0.1, anchor_residual_l2=0.1),
                       fit=_fit(), on_progress=events.append)
    inventories = [e for e in events if e["event"] == "candidate_inventory"]
    assert [e["stage"] for e in inventories] == ["rank", "penalty"]
    assert sum(e["total"] for e in inventories) == len(tuned.trials)
    assert all(e["pending"] == e["total"] for e in inventories)
