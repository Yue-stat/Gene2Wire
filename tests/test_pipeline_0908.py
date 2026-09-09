"""Fast integration gates for the shared experiment pipeline, without downloads."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.io import cached_download
from gene2wire.experiments.pipeline import run_experiment, run_simulation_experiments
from gene2wire.experiments.protocol import MODEL_ORDER, Settings
from gene2wire.models import UnifiedPUModel


OPTIONS = dict(n_cells=120, n_targets=6, n_gene_features=5, n_location_features=2,
               n_target_features=3, n_slices=12, true_rank=2)


def settings(**changes):
    base = Settings(n_outer_folds=3, n_jobs=1, n_repetitions=1,
                    candidate_budget=3, penalties=(.01,),
                    maxiter=25, retry_maxiter=30, tolerance=1e-5,
                    init_direct_maxiter=20, loss_rates=(0.,),
                    run_random_forest=False, run_information_controls=False,
                    run_mechanism_controls=False, run_calibration_controls=False, run_qiao=False)
    return replace(base, **changes)


def dataset(**changes):
    return generate_simulation(0, .5, seed=9208, **{**OPTIONS, **changes})


def comparable(frame):
    sort = [key for key in ("dataset", "repetition", "outer_fold", "model", "loss_rate") if key in frame]
    return frame.sort_values(sort).reset_index(drop=True).sort_index(axis=1)


def prediction_files(artifacts):
    return {path.relative_to(artifacts.export_dir): path
            for path in artifacts.export_dir.rglob("*_predictions.npz")}


def assert_predictions_equal(first, second):
    left, right = prediction_files(first), prediction_files(second)
    assert left and set(left) == set(right)
    for key in left:
        with np.load(left[key], allow_pickle=False) as a, np.load(right[key], allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for name in a.files:
                if a[name].dtype.kind in "fc":
                    np.testing.assert_allclose(a[name], b[name], rtol=1e-12, atol=1e-12, equal_nan=True)
                else:
                    np.testing.assert_array_equal(a[name], b[name])


def test_serial_parallel_predictions_and_metrics_match(tmp_path):
    data = dataset()
    serial = run_experiment(data, settings(), checkpoint_dir=tmp_path / "serial_checkpoints",
                            export_dir=tmp_path / "serial_exports")
    parallel = run_experiment(data, settings(n_jobs=2), checkpoint_dir=tmp_path / "parallel_checkpoints",
                              export_dir=tmp_path / "parallel_exports")
    pd.testing.assert_frame_equal(comparable(serial.tables["metrics"]),
                                  comparable(parallel.tables["metrics"]), atol=1e-12, rtol=1e-12)
    assert_predictions_equal(serial, parallel)
    assert serial.manifest["run_id"] == parallel.manifest["run_id"]


def test_resume_and_changed_worker_count_reuse_core_fits(tmp_path):
    data = dataset()
    kwargs = dict(checkpoint_dir=tmp_path / "checkpoints", export_dir=tmp_path / "exports")
    first = run_experiment(data, settings(), **kwargs)
    with patch.object(UnifiedPUModel, "fit", side_effect=AssertionError("completed model refitted")):
        resumed = run_experiment(data, settings(), **kwargs)
    core = resumed.tables["selected"].query("model in @MODEL_ORDER")
    assert len(core) == 3 * len(MODEL_ORDER)
    assert core["resumed"].eq(True).all()
    changed = run_experiment(data, settings(n_jobs=2), **kwargs)
    assert changed.tables["selected"].query("model in @MODEL_ORDER")["resumed"].eq(True).all()
    assert first.manifest["run_id"] == resumed.manifest["run_id"] == changed.manifest["run_id"]
    pd.testing.assert_frame_equal(comparable(first.tables["metrics"]), comparable(changed.tables["metrics"]))


@pytest.mark.parametrize("use_location,use_target_features", [(False, False), (True, False), (False, True), (True, True)])
def test_feature_switches_reach_end_to_end_fits(tmp_path, use_location, use_target_features):
    data = dataset(truth_uses_location=use_location)
    config = settings(use_location=use_location, use_target_features=use_target_features)
    original = UnifiedPUModel.fit
    seen = []

    def recording(self, bundle, *args, **kwargs):
        seen.append((bundle.n_features, None if bundle.Y_target is None else bundle.Y_target.shape,
                     self.config.use_target_features))
        return original(self, bundle, *args, **kwargs)

    with patch.object(UnifiedPUModel, "fit", recording):
        artifacts = run_experiment(data, config, checkpoint_dir=tmp_path / "checkpoints",
                                   export_dir=tmp_path / "exports")
    assert seen
    assert {item[0] for item in seen} == {7 if use_location else 5}
    assert {item[1] for item in seen} == {(6, 3) if use_target_features else None}
    assert {item[2] for item in seen} == {use_target_features}
    assert artifacts.manifest["status"] == "complete"
    assert artifacts.tables["metrics"]["model"].isin(MODEL_ORDER).sum() == 18


def test_simulation_intervals_count_generated_datasets_not_folds(tmp_path):
    config = settings(n_repetitions=2)
    artifacts = run_simulation_experiments(
        config, raw_cache_dir=tmp_path / "raw", checkpoint_dir=tmp_path / "checkpoints",
        export_dir=tmp_path / "exports", sharing_strengths=(.5,), simulation_options=OPTIONS)
    interval = artifacts.tables["simulation_intervals"].query("metric == 'macro_log_loss'")
    assert interval["n"].eq(2).all()
    assert len(artifacts.tables["per_repetition"]) == 2 * 6  # six core methods; optional controls disabled
    assert artifacts.manifest["uncertainty_unit"] == "generated_dataset"
    raw_files = list((tmp_path / "raw").glob("*.npz"))
    assert len(raw_files) == 2
    for path in raw_files:
        with np.load(path, allow_pickle=False) as archive:
            assert archive["genes"].shape == (120, 5)
            assert archive["reference"].shape == (120, 6)


def test_validated_raw_cache_is_read_offline(tmp_path):
    # Existing publisher-verified bytes need no HEAD request or download.
    import hashlib
    raw = tmp_path / "raw_data" / "published_fixture.bin"
    raw.parent.mkdir()
    raw.write_bytes(b"small reproducible raw data fixture\n")
    expected = hashlib.sha256(raw.read_bytes()).hexdigest()
    with patch("gene2wire.experiments.io.urlopen", side_effect=AssertionError("network used")):
        first = cached_download("https://example.invalid/raw.bin", raw, sha256=expected)
        second = cached_download("https://example.invalid/raw.bin", raw, sha256=expected)
    assert first == second == raw
    assert raw.with_name(raw.name + ".download.json").exists()


def test_mixed_rf_real_fit_exports_score_semantics_and_resumes(tmp_path, monkeypatch):
    from gene2wire.experiments import baselines
    monkeypatch.setattr(baselines, "baseline_candidates", lambda kind, budget: [
        {"n_estimators": 8, "min_samples_leaf": leaf, "max_features": 1.}
        for leaf in (1, 3)])
    config = settings(loss_rates=(.8,), run_random_forest=True, run_information_controls=True)
    data = dataset()
    kwargs = dict(checkpoint_dir=tmp_path / "checkpoints", export_dir=tmp_path / "exports", progress=False)
    first = run_experiment(data, config, **kwargs)
    assert first.manifest["status"] == "complete"
    metrics = first.tables["metrics"]
    assert len(metrics) == 30  # Three folds; six core + Reference+PU + three RF.
    mixed = metrics.loc[metrics["model"].eq("RF-mixed")]
    assert len(mixed) == 3 and mixed["probability_semantics"].eq("mixed").all()
    assert mixed["uses_paired_reference"].all()
    assert mixed["uses_pu_likelihood"].eq(False).all()
    assert mixed["macro_log_loss"].notna().all()
    selected = first.tables["selected"].query("model == 'RF-mixed'")
    assert selected["validation_score_semantics"].eq("raw_mixed_label_score").all()
    files = list(first.export_dir.rglob("RF-mixed_predictions.npz"))
    assert len(files) == 3
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            assert not {"p", "q", "h"} & set(archive.files)
            np.testing.assert_allclose(archive["mixed_label_score"], archive["prediction"], equal_nan=True)
    with patch.object(baselines, "fit_baseline", side_effect=AssertionError("completed RF refitted")):
        second = run_experiment(data, config, **kwargs)
    assert second.tables["checkpoint_inventory"]["fully_cached"].all()
