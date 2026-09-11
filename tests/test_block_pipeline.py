from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments import baselines, pipeline
from gene2wire.experiments.block_design import BlockMaskConfig
from gene2wire.experiments.block_evaluation import evaluate_block_predictions
from gene2wire.experiments.block_experiment import (
    _block_views, preview_block_experiment, run_block_experiment,
    run_block_simulation_experiments)
from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.observation import make_observation_design, thin_reference
from gene2wire.experiments.protocol import Settings
from gene2wire.models import UnifiedPUModel


def _source():
    source = generate_simulation(0, .5, seed=9208, n_cells=120, n_targets=6,
                                 n_gene_features=5, n_slices=12)
    return replace(source, name="toy_animal_data",
                   groups={"animal": np.repeat(["animal1", "animal2"], 60)},
                   metadata={})


def _settings(**options):
    return replace(Settings(n_jobs=1, n_repetitions=1, penalties=(.01,), candidate_budget=3,
                    maxiter=50, retry_maxiter=100, tolerance=1e-5, loss_rates=(0.,),
                    run_random_forest=False, run_information_controls=False,
                    run_calibration_controls=False, run_mechanism_controls=False,
                    run_qiao=False), **options)


def test_hidden_outcomes_cannot_change_any_fit_calibration_or_validation_input(tmp_path, monkeypatch):
    source, settings = _source(), _settings(loss_rates=(.6,), run_information_controls=True,
                                           run_random_forest=True)
    config = BlockMaskConfig(fractions=(.8,))
    preview = preview_block_experiment(source, settings, config)
    hidden = preview['design'].masks[.8]
    altered = source.reference.copy()
    altered[hidden] = 1-altered[hidden]
    second = replace(source, reference=altered)
    captures = []
    detectors = []
    forests = []

    def detector(observed, reference, measured, paired_rows, **kwargs):
        detectors.append((observed.copy(), reference.copy(), measured.copy(), np.array(paired_rows)))
        assert not np.any(measured[hidden])
        assert not np.any(observed[hidden])
        assert not np.any(reference[np.isfinite(reference) & hidden])
        return SimpleNamespace(predict=lambda n_cells, **kw: np.full(observed.shape, .4),
                               to_dict=lambda: {'test_detector': .4})

    def core(**kwargs):
        captures.append(kwargs)
        result = {}
        for model in kwargs['models']:
            result[model.name] = SimpleNamespace(
                fitted=SimpleNamespace(config=model),
                latent_probability=np.full((len(kwargs['test_X']), kwargs['train'].n_targets), .3),
                summary=lambda name=model.name: {'model': name}, tuning=SimpleNamespace(trials=[]))
        return SimpleNamespace(models=result)

    def forest(*args, **kwargs):
        forests.append(args)
        return SimpleNamespace(prediction=np.full((len(args[7]), args[1].shape[1]), .3),
                               selected_config={}, candidate_records=[], diagnostics={})

    monkeypatch.setattr(pipeline, 'fit_detection_calibrator', detector)
    monkeypatch.setattr(pipeline, 'run_model_grid', core)
    monkeypatch.setattr(baselines, 'fit_baseline', forest)
    all_metrics = []
    for run, data in enumerate((source, second)):
        views, _ = _block_views(data, settings, config, 0)
        view = views[0]
        fold = view.split_builder(3, settings.seed)[0]
        prepared = pipeline._prepare(view, fold, settings)
        table = pipeline._run_fold(prepared, 0, settings, tmp_path/'cache', tmp_path/str(run), 'test')
        all_metrics.append(table['metrics'])
        for role, rows in [('train', fold.train_rows), ('validation', fold.validation_rows),
                            ('refit', np.sort(np.r_[fold.train_rows, fold.validation_rows]))]:
            for call in captures[run*3:(run+1)*3]:
                bundle = call[role]
                assert bundle.Z_reference is None and bundle.reference_mask is None
                assert not np.any(bundle.W_measured[hidden[rows]])
                assert not np.any(bundle.S_observed[hidden[rows]])
    assert len(captures) == 6 and len(detectors) == 4 and len(forests) == 6
    for a, b in zip(captures[:3], captures[3:]):
        for role in ['train', 'validation', 'refit']:
            for field in ['X_cell', 'S_observed', 'W_measured']:
                np.testing.assert_array_equal(getattr(a[role], field), getattr(b[role], field))
        for field in ['train_exposure', 'validation_exposure', 'refit_exposure', 'test_exposure']:
            np.testing.assert_array_equal(a[field], b[field])
        assert a['seed'] == b['seed']
    for a, b in zip(detectors[:2], detectors[2:]):
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
    for a, b in zip(forests[:3], forests[3:]):
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
    assert all_metrics[0][0]['macro_log_loss'] != all_metrics[1][0]['macro_log_loss']


def test_scar_keeps_identical_observations_on_shared_visible_entries():
    source, settings = _source(), _settings(loss_rates=(0., .2, .4, .6, .8))
    views, _ = _block_views(source, settings, BlockMaskConfig(fractions=(.4, .8)), 0)
    full = views[-1]
    train = full.split_builder(3, settings.seed)[0].train_rows
    design = make_observation_design(*source.reference.shape, seed=137)
    for rate in settings.loss_rates:
        reference = thin_reference(full.reference, full.measured, train, rate, 'scar', design)
        for view in views[:-1]:
            result = thin_reference(view.reference, view.measured, train, rate, 'scar', design)
            np.testing.assert_array_equal(result.observed[view.measured], reference.observed[view.measured])
            np.testing.assert_allclose(result.sensitivity, 1-rate, atol=1e-12)


def test_natural_labels_and_reference_budget_respect_same_structural_mask():
    source = _source()
    source = replace(source, natural_observed=source.reference &
                     (np.random.default_rng(92).random(source.reference.shape) < .6))
    views, _ = _block_views(source, _settings(), BlockMaskConfig(fractions=(.8,)), 0)
    masked, full = views
    assert not masked.natural_observed[~masked.measured].any()
    assert not masked.reference[~masked.measured].any()
    np.testing.assert_array_equal(masked.natural_observed[masked.measured],
                                   full.natural_observed[masked.measured])
    prepared = pipeline._prepare(masked, masked.split_builder(3, 1)[0], _settings())
    assert pipeline._scenarios(prepared, _settings())[0]['mechanism'] == 'natural'


def test_block_evaluation_scores_original_reference_without_hidden_posterior():
    truth = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    mask = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], bool)
    prediction = np.array([[.8, .4], [.2, .5], [.6, .1], [.2, .9]])
    result = evaluate_block_predictions(truth, mask, prediction,
                                         groups=['a', 'a', 'b', 'b'])
    assert result['summary']['macro_auprc'] == 1
    assert result['summary']['macro_brier'] == pytest.approx(.025)
    assert result['summary']['macro_log_loss'] == pytest.approx((-np.log(.8)-np.log(.9))/2)
    assert result['summary']['n_evaluated'] == 4
    assert len(result['per_group']) == 2
    assert not any('hidden' in key or 'observed' in key for key in result['summary'])
    # Changes outside evaluation support cannot change any score.
    altered = truth.copy()
    altered[~mask] = 1-altered[~mask]
    other = evaluate_block_predictions(altered, mask, prediction)
    np.testing.assert_equal(result['summary'], other['summary'])


def test_full_control_fits_once_per_fold_and_evaluates_all_block_masks_with_resume(tmp_path):
    source, settings = _source(), _settings()
    config = BlockMaskConfig(fractions=(.4, .8))
    kwargs = dict(checkpoint_dir=tmp_path/'cache', export_dir=tmp_path/'export', progress=False)
    artifacts = run_block_experiment(source, settings, config, **kwargs)
    assert artifacts.manifest['completed_model_evaluations'] == 3*3*6
    plan = artifacts.tables['model_evaluation_plan']
    assert len(plan.query("training_panel == 'full'")) == 3*6
    full = artifacts.tables['metrics'].query("training_panel == 'full' and model == 'PU'")
    assert len(full) == 3*2
    assert set(full.block_fraction) == {.4, .8}
    metrics = artifacts.tables['metrics']
    count = metrics.query("model == 'PU'").pivot(index=['outer_fold', 'block_fraction'],
                                                columns='training_panel', values='n_evaluated')
    np.testing.assert_array_equal(count['masked'], count['full'])
    assert set(artifacts.tables) >= {'block_assignment', 'block_summary', 'block_groups',
                                     'block_splits', 'per_group'}
    assert len(artifacts.tables['aggregate'].query("model == 'PU'")) == 4
    assert not any('hidden' in column for column in metrics)
    prediction_path = next(artifacts.export_dir.rglob('PU_predictions.npz'))
    with np.load(prediction_path) as archive:
        assert set(archive.files) >= {'reference', 'evaluation_masks', 'block_groups',
                                      'source_measured', 'training_measured', 'prediction'}
        assert 'h' not in archive.files
    with patch.object(UnifiedPUModel, 'fit', side_effect=AssertionError('Unexpected refit')):
        resumed = run_block_experiment(source, settings, config, **kwargs)
    assert artifacts.manifest['run_id'] == resumed.manifest['run_id']
    assert resumed.tables['checkpoint_inventory'].fully_cached.all()
    assert resumed.manifest['cached_model_evaluations'] == 54


def test_simulation_wrapper_uses_requested_location_truth_and_independent_replication(tmp_path, monkeypatch):
    captured = {}
    def execute(datasets, settings, *args, **kwargs):
        captured.update(datasets=datasets, kwargs=kwargs)
        return 'ok'
    monkeypatch.setattr('gene2wire.experiments.block_experiment._execute', execute)
    settings = _settings(n_repetitions=2, use_location=True)
    config = BlockMaskConfig(fractions=(.4,), group_mode='artificial')
    result = run_block_simulation_experiments(settings, config, raw_cache_dir=tmp_path/'raw',
        checkpoint_dir=tmp_path/'cache', export_dir=tmp_path/'export', sharing_strengths=(0., 1.),
        simulation_options=dict(n_cells=120, n_targets=6, n_gene_features=5, n_slices=12))
    assert result == 'ok'
    assert len(captured['datasets']) == 2*2*2
    assert all(d.metadata['truth_uses_location'] for d in captured['datasets'])
    assert {d.metadata['experiment_repetition'] for d in captured['datasets']} == {0, 1}
    assert len(list((tmp_path/'raw').glob('*.npz'))) == 4
    with pytest.raises(ValueError, match='must match'):
        run_block_simulation_experiments(settings, config, raw_cache_dir=tmp_path/'raw',
            checkpoint_dir=tmp_path/'cache', export_dir=tmp_path/'export',
            simulation_options={'truth_uses_location': False})


def test_simulation_raw_cache_restores_without_generation_and_rejects_changed_bytes(tmp_path, monkeypatch):
    captured = []
    def execute(datasets, *args, **kwargs):
        captured.append(datasets)
        return 'ok'
    monkeypatch.setattr('gene2wire.experiments.block_experiment._execute', execute)
    settings = _settings(use_location=True, use_target_features=True)
    config = BlockMaskConfig(fractions=(.4,), group_mode='artificial')
    kwargs = dict(raw_cache_dir=tmp_path/'raw', checkpoint_dir=tmp_path/'checkpoints',
                  export_dir=tmp_path/'exports', sharing_strengths=(.5,),
                  simulation_options=dict(n_cells=120, n_targets=6, n_gene_features=5, n_slices=12))
    run_block_simulation_experiments(settings, config, **kwargs)
    with patch('gene2wire.experiments.datasets.simulation.generate_simulation',
               side_effect=AssertionError('Raw cache must be restored without generation')):
        run_block_simulation_experiments(settings, config, **kwargs)
        first, second = captured[0][0], captured[1][0]
        for key in ['reference', 'measured', 'technical_score']:
            np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
        assert first.cell_ids == second.cell_ids and first.target_ids == second.target_ids
        for key in first.metadata:
            np.testing.assert_equal(first.metadata[key], second.metadata[key])
        a = first.feature_builder(np.arange(60), True, True)
        b = second.feature_builder(np.arange(60), True, True)
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.Y_target, b.Y_target)
        for key in first.groups:
            np.testing.assert_array_equal(first.groups[key], second.groups[key])
        raw_path = next((tmp_path/'raw').glob('*.npz'))
        before = raw_path.read_bytes() + b'changed'
        raw_path.write_bytes(before)
        with pytest.raises(ValueError, match='checksum mismatch'):
            run_block_simulation_experiments(settings, config, **kwargs)
        assert raw_path.read_bytes() == before
