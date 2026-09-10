import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.block_design import make_block_folds
from gene2wire.experiments.contracts import ExperimentDataset, FeatureSet
from gene2wire.experiments.io import atomic_json, atomic_npz
from gene2wire.experiments.native_panel import (native_panel_tables,
    run_native_panel_experiment, summarize_native_predictions)
from gene2wire.experiments.pipeline import Artifacts
from gene2wire.experiments.protocol import Settings
from gene2wire.experiments.reporting import load_export_artifacts
from gene2wire.models import UnifiedPUModel


def _source(n=90):
    rng = np.random.default_rng(9210)
    x = rng.normal(size=(n, 4))
    animals = np.repeat(['A', 'B', 'C'], n//3)
    panels = np.array([[1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0], [1, 0, 1, 0, 1, 1]], bool)
    w = panels[np.repeat(np.arange(3), n//3)]
    z = (rng.random(w.shape) < .35) & w
    def features(rows, use_location, use_target):
        assert not use_location and not use_target
        return FeatureSet((x-x[rows].mean(axis=0))/(x[rows].std(axis=0)+1e-8), {'gene': range(4)})
    return ExperimentDataset('native_toy', z, w, tuple(f'c{i}' for i in range(n)),
        tuple(f't{i}' for i in range(6)), features,
        lambda folds, seed: make_block_folds(animals, folds, seed), groups={'animal': animals})


def test_real_native_mask_repeated_splits_full_oof_forecasts_and_resume(tmp_path):
    data = _source()
    settings = Settings(supervision_profile='assay_only', paired_fraction=0., n_jobs=1,
        n_repetitions=2, n_outer_folds=3, penalties=(.01,), candidate_budget=3,
        maxiter=50, retry_maxiter=100, tolerance=1e-5,
        run_random_forest=False, run_qiao=False)
    kwargs = dict(checkpoint_dir=tmp_path/'checkpoints', export_dir=tmp_path/'exports', progress=False)
    result = run_native_panel_experiment(data, settings, **kwargs)
    assert result.manifest['completed_model_evaluations'] == 2*3*3
    assert set(result.tables['metrics'].model) == {'Logistic', 'MIRT', 'Joint'}
    assert result.tables['detection'].empty
    assert len(result.tables['per_animal_aggregate']) == 3*3
    assert not any('hidden' in col or 'detection_' in col for col in result.tables['per_animal_metrics'])
    assert result.tables['unassayed_summary'].evaluation_eligible.eq(False).all()
    assert result.tables['unassayed_summary'].reference.isna().all()
    assert result.tables['unassayed_summary'].min_repeats.eq(2).all()
    splits = result.tables['native_splits']
    test = splits.query("split == 'test'")
    assert test.groupby(['repetition', 'cell_id']).size().eq(1).all()
    assert not test.query('repetition == 0').set_index('cell_id').outer_fold.equals(
        test.query('repetition == 1').set_index('cell_id').outer_fold)
    with np.load(result.export_dir/'native_unassayed/Joint_oof_forecasts.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['repeat_count'], (~data.measured)*2)
        assert np.isfinite(saved['mean_assay_probability'][~data.measured]).all()
        assert np.isnan(saved['mean_assay_probability'][data.measured]).all()
        assert saved['cell_ids'].dtype.kind == 'U'
        assert np.isnan(saved['reference']).all()
    csv = pd.read_csv(result.export_dir/'native_unassayed/Joint_cell_forecasts.csv.gz')
    assert len(csv) == (~data.measured).sum()
    assert csv.reference.isna().all() and not csv.evaluation_eligible.any()
    reloaded = load_export_artifacts(result.export_dir)
    for name in ('per_animal_metrics', 'per_animal_aggregate', 'per_animal_target',
                 'per_animal_target_aggregate', 'unassayed_summary'):
        assert len(reloaded.tables[name]) == len(result.tables[name])
    assert reloaded.manifest['native_predictions_completed']
    with patch.object(UnifiedPUModel, 'fit', side_effect=AssertionError('Unexpected refit')):
        resumed = run_native_panel_experiment(data, settings, **kwargs)
    assert resumed.manifest['run_id'] == result.manifest['run_id']
    assert resumed.manifest['cached_model_evaluations'] == 18


def test_native_panel_availability_does_not_follow_positive_incidence():
    data = _source()
    data = replace(data, reference=np.zeros_like(data.reference))
    panel = native_panel_tables(data)['native_panel']
    assert panel.measured.sum() == 11
    assert panel.loc[panel.measured, 'n_positive'].eq(0).all()
    assert panel.loc[~panel.measured, 'n_positive'].isna().all()


def _saved_fixture(tmp_path):
    n = 12
    ids = np.array([f'c{i}' for i in range(n)])
    animals = np.repeat(['A', 'B'], 6)
    w = np.tile([True, True, False], (n, 1))
    w[6:] = [False, True, True]
    z = ((np.arange(n)[:, None]+np.arange(3)) % 3 == 0) & w
    cells = pd.DataFrame({'cell_id': ids, 'animal': animals})
    panel = pd.DataFrame([{'animal': a, 'target': f't{j}', 'measured': bool(w[i, j])}
                         for a, i in [('A', 0), ('B', 6)] for j in range(3)])
    for rep in range(2):
        for fold in range(2):
            rows = np.arange(n)[np.arange(n) % 2 == fold]
            p = np.where(w[rows], .2+.6*z[rows], .3+.1*rep)
            unit = tmp_path/'units'/f'r{rep}f{fold}'
            atomic_json({'dataset': 'toy', 'repetition': rep, 'outer_fold': fold,
                         'mechanism': 'assay_only'}, unit/'audit.json')
            atomic_npz(unit/'Joint_predictions.npz', cell_ids=ids[rows], target_ids=np.array(['t0','t1','t2']),
                measured=w[rows], reference=z[rows], prediction=p)
    return Artifacts({'native_cells': cells, 'native_panel': panel}, tmp_path,
                     {'experiment': 'native_panels', 'protocol': {'n_repetitions': 2}}), w


def test_unmeasured_forecasts_never_enter_measured_metrics(tmp_path):
    artifacts, w = _saved_fixture(tmp_path)
    first = summarize_native_predictions(artifacts)
    expected_metrics = first.tables['per_animal_metrics'].copy()
    with np.load(tmp_path/'native_unassayed/Joint_oof_forecasts.npz', allow_pickle=False) as f:
        np.testing.assert_allclose(f['mean_assay_probability'][~w], .35)
        np.testing.assert_allclose(f['split_standard_deviation'][~w], np.sqrt(.005))
    for path in (tmp_path/'units').glob('*/Joint_predictions.npz'):
        with np.load(path, allow_pickle=False) as f:
            values = {k:f[k].copy() for k in f.files}
        values['prediction'][~values['measured']] = .99
        # Reference values outside W are deliberately changed; they cannot be scored.
        values['reference'][~values['measured']] = True
        atomic_npz(path, **values)
    second = summarize_native_predictions(artifacts)
    pd.testing.assert_frame_equal(expected_metrics, second.tables['per_animal_metrics'])
    np.testing.assert_allclose(second.tables['unassayed_summary'].mean_prediction, .99)


def test_repeated_test_cell_is_rejected(tmp_path):
    artifacts, _ = _saved_fixture(tmp_path)
    path = tmp_path/'units/r0f0/Joint_predictions.npz'
    with np.load(path, allow_pickle=False) as f:
        arrays = {k:f[k].copy() for k in f.files}
    atomic_json({'dataset': 'toy', 'repetition': 0, 'outer_fold': 2, 'mechanism': 'assay_only'},
                tmp_path/'units/duplicate/audit.json')
    atomic_npz(tmp_path/'units/duplicate/Joint_predictions.npz', **arrays)
    with pytest.raises(ValueError, match='multiple test folds'):
        summarize_native_predictions(artifacts)
