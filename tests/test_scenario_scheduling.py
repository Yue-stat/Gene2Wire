"""Scheduling changes must preserve predictions, label draws and fine-grained resume."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.pipeline import run_experiment
from gene2wire.experiments.protocol import Settings
from gene2wire.models import UnifiedPUModel


def _settings(**changes):
    return replace(Settings(n_jobs=1, n_repetitions=1, penalties=(.01,),
        candidate_budget=3, maxiter=50, retry_maxiter=100, tolerance=1e-5,
        init_direct_maxiter=20, loss_rates=(0., .8), run_information_controls=False,
        run_random_forest=False, run_qiao=False, run_mechanism_controls=False,
        run_calibration_controls=False), **changes)


def _data():
    return generate_simulation(0, .5, seed=9208, n_cells=120, n_targets=6,
                               n_gene_features=5, n_slices=12)


def _ordered(frame):
    keys = [c for c in ("outer_fold", "loss_rate", "model", "stage", "index") if c in frame]
    return frame.sort_values(keys).reset_index(drop=True).sort_index(axis=1)


def _assert_export_arrays_equal(first, second):
    left = {p.relative_to(first.export_dir): p for p in first.export_dir.glob("units/*/*.npz")}
    right = {p.relative_to(second.export_dir): p for p in second.export_dir.glob("units/*/*.npz")}
    assert left and set(left) == set(right)
    for relative in left:
        with np.load(left[relative], allow_pickle=False) as a, np.load(right[relative], allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for name in a.files:
                if a[name].dtype.kind in "fc":
                    np.testing.assert_allclose(a[name], b[name], atol=1e-12, rtol=1e-12, equal_nan=True)
                else:
                    np.testing.assert_array_equal(a[name], b[name])


def test_runtime_schedule_settings_are_not_scientific_inputs():
    settings = Settings()
    assert settings.parallel_unit == "scenario"
    assert settings.scientific_dict() == replace(settings, n_jobs=2, parallel_unit="fold").scientific_dict()
    with pytest.raises(ValueError, match="parallel_unit"):
        replace(settings, parallel_unit="candidate")


def test_fold_serial_matches_scenarios_with_more_workers_than_folds_and_resumes(tmp_path):
    data = _data()
    first = run_experiment(data, _settings(parallel_unit="fold"),
        checkpoint_dir=tmp_path / "fold_checkpoints", export_dir=tmp_path / "fold_exports", progress=False)
    kwargs = dict(checkpoint_dir=tmp_path / "scenario_checkpoints",
                  export_dir=tmp_path / "scenario_exports", progress=False)
    second = run_experiment(data, _settings(n_jobs=5), **kwargs)
    assert first.manifest["run_id"] == second.manifest["run_id"]
    assert first.manifest["scheduled_tasks"] == 3
    assert first.manifest["effective_n_jobs"] == 1
    assert second.manifest["scheduled_tasks"] == 6
    assert second.manifest["effective_n_jobs"] == 5  # greater than folds * repetitions = 3
    assert second.manifest["planned_scenario_units"] == 6
    assert second.manifest["pending_scenario_units"] == 6
    for name in ("metrics", "tuning", "thinning_audit", "detection"):
        pd.testing.assert_frame_equal(_ordered(first.tables[name]), _ordered(second.tables[name]),
                                      atol=1e-12, rtol=1e-12)
    _assert_export_arrays_equal(first, second)
    events = second.tables["progress_events"]
    assert (events["event"] == "unit_complete").sum() == 6
    assert (events["event"] == "model_complete").sum() == 36
    assert second.manifest["completed_model_evaluations"] == 36
    # Changing the runtime schedule reuses identical scenario summaries.
    with patch.object(UnifiedPUModel, "fit", side_effect=AssertionError("Cached fit reran")):
        resumed = run_experiment(data, _settings(n_jobs=1, parallel_unit="fold"), **kwargs)
        assert resumed.tables["checkpoint_inventory"]["fully_cached"].all()
        assert resumed.manifest["effective_n_jobs"] == 0
        assert resumed.manifest["scheduled_tasks"] == 0
        prediction = next(second.export_dir.glob("units/*/PU-Joint_predictions.npz"))
        with np.load(prediction) as archive:
            expected = archive["prediction"].copy()
        prediction.unlink()
        rebuilt = run_experiment(data, _settings(n_jobs=1), **kwargs)
        assert rebuilt.tables["checkpoint_inventory"]["fully_cached"].sum() == 5
        assert rebuilt.manifest["pending_scenario_units"] == 1
        assert rebuilt.manifest["effective_n_jobs"] == 1
        assert rebuilt.manifest["cached_model_evaluations"] == 30
        assert rebuilt.manifest["completed_model_evaluations"] == 36
        with np.load(prediction) as archive:
            np.testing.assert_array_equal(archive["prediction"], expected)


def test_failed_scenarios_do_not_invalidate_successful_scenario_summaries(tmp_path):
    from gene2wire.experiments import pipeline
    from gene2wire.experiments.calibration import CalibrationNotEstimable

    original = pipeline._detector
    seen = []

    def detector(prepared, observed, paired, scenario, technical_score):
        seen.append(scenario["loss_rate"])
        if scenario["loss_rate"] == .8:
            raise CalibrationNotEstimable("Synthetic fixture with no estimable calibration")
        return original(prepared, observed, paired, scenario, technical_score)

    kwargs = dict(checkpoint_dir=tmp_path / "checkpoints", export_dir=tmp_path / "exports", progress=False)
    with patch.object(pipeline, "_detector", detector):
        first = run_experiment(_data(), _settings(), **kwargs)
        assert first.manifest["status"] == "complete_with_failures"
        assert first.manifest["completed_model_evaluations"] == 18
        assert len(first.tables["failures"]) == 3
        seen.clear()
        with patch.object(UnifiedPUModel, "fit", side_effect=AssertionError("Successful scenario refitted")):
            second = run_experiment(_data(), _settings(parallel_unit="fold"), **kwargs)
        assert seen == [.8, .8, .8]
        assert second.tables["checkpoint_inventory"]["fully_cached"].sum() == 3
        assert second.manifest["cached_model_evaluations"] == 18
        assert second.manifest["completed_model_evaluations"] == 18
