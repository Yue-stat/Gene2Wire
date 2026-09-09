from unittest.mock import patch

import numpy as np

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.pipeline import run_experiment
from gene2wire.experiments.protocol import Settings
from gene2wire.experiments.progress import ProgressRelay
from gene2wire.models import UnifiedPUModel


def test_whole_unit_inventory_and_missing_export_recovery(tmp_path, capsys):
    data = generate_simulation(0, .5, seed=9208, n_cells=120, n_targets=6,
                               n_gene_features=5, n_slices=12)
    settings = Settings(n_jobs=1, n_repetitions=1, penalties=(.01,), candidate_budget=3,
                        maxiter=50, retry_maxiter=100, tolerance=1e-5,
                        loss_rates=(0.,), run_random_forest=False,
                        run_information_controls=False, run_calibration_controls=False,
                        run_mechanism_controls=False, run_qiao=False)
    kwargs = dict(checkpoint_dir=tmp_path / 'checkpoints', export_dir=tmp_path / 'exports')
    first = run_experiment(data, settings, **kwargs)
    output = capsys.readouterr().out
    assert '0/24 model evaluations reusable' in output
    assert 'finished units 24/24' in output
    assert '[start]' not in output and '[done]' not in output and '[trial]' not in output
    assert not first.tables['checkpoint_inventory']['fully_cached'].any()
    events = first.tables['progress_events']
    assert (events['event'] == 'unit_complete').sum() == 3
    assert set(events.loc[events['event'] == 'model_complete', 'model']) >= {'PU-Joint', 'PU'}
    with patch.object(UnifiedPUModel, 'fit', side_effect=AssertionError('Unexpected fit')):
        second = run_experiment(data, settings, **kwargs)
        assert second.tables['checkpoint_inventory']['fully_cached'].all()
        assert second.tables['selected']['resumed'].all()
        assert '24/24 model evaluations reusable' in capsys.readouterr().out
        prediction = next(first.export_dir.rglob('PU-Joint_predictions.npz'))
        with np.load(prediction) as archive:
            expected = archive['prediction'].copy()
        prediction.unlink()
        rebuilt = run_experiment(data, settings, progress=False, **kwargs)
        assert rebuilt.tables['checkpoint_inventory']['fully_cached'].sum() == 2
        with np.load(prediction) as archive:
            np.testing.assert_array_equal(archive['prediction'], expected)
    assert first.manifest['run_id'] == second.manifest['run_id'] == rebuilt.manifest['run_id']


def test_parent_relay_keeps_partial_lines_and_shows_context(tmp_path, capsys):
    relay = ProgressRelay(tmp_path, enabled=True, level="model")
    writer = relay.writer('unit1', dataset='BARseq M1', repetition=2, outer_fold=1)
    writer({'event': 'model_start', 'model': 'PU-Joint'})
    path = tmp_path / 'unit2.jsonl'
    path.write_bytes(b'{"event": "unit_complete"')
    relay.drain()
    assert len(relay.rows) == 1
    with path.open('ab') as handle:
        handle.write(b', "work_id": "unit2"}\n')
    relay.drain()
    relay.drain()
    assert len(relay.rows) == 2
    assert 'BARseq M1 | rep=2 | fold=1 | PU-Joint' in capsys.readouterr().out


def test_summary_heartbeat_deduplicates_models_and_uses_la_time(tmp_path, capsys):
    from datetime import datetime
    relay = ProgressRelay(tmp_path, total_units=5, cached_units=1)
    relay.started = relay.last_heartbeat = 100.
    writer = relay.writer('fold', work_id='fold', dataset='BARseq A1', repetition=0, outer_fold=0)
    first = dict(model='PU', analysis='primary', mechanism='technical_sar', loss_rate=.8,
                 calibration_fraction=.2, calibration_spec='correct')
    writer({**first, 'event': 'model_start'})
    writer({**first, 'event': 'candidate_complete'})
    writer({**first, 'event': 'model_complete'})
    writer({**first, 'event': 'model_complete'})
    writer({**first, 'event': 'model_complete', 'calibration_fraction': .4})
    writer({'event': 'unit_complete'})
    writer({**first, 'event': 'calibration_failed', 'calibration_spec': 'omit_technical'})
    relay.drain()
    assert relay.done_units == 3
    assert capsys.readouterr().out == ''
    summer = datetime.fromisoformat('2026-09-09T17:42:00+00:00').timestamp()
    with patch('gene2wire.experiments.progress.time.time', return_value=summer):
        assert not relay.heartbeat(159.)
        assert relay.heartbeat(160.)
        assert not relay.heartbeat(161.)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert '[running 1min] finished units 3/5, current time 2026-09-09 10:42:00 PDT' in lines[0]
    assert 'failed calibration scenarios 1' in lines[0]
    winter = datetime.fromisoformat('2026-12-09T17:42:00+00:00').timestamp()
    with patch('gene2wire.experiments.progress.time.time', return_value=winter):
        assert relay.heartbeat(220.)
    assert '2026-12-09 09:42:00 PST' in capsys.readouterr().out


def test_summary_final_and_disabled_output(tmp_path, capsys):
    with ProgressRelay(tmp_path / 'complete', total_units=2, cached_units=2):
        pass
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert '[cache] 2/2' in lines[0] and '[finished 0min] finished units 2/2' in lines[1]
    with ProgressRelay(tmp_path / 'quiet', enabled=False, total_units=2) as relay:
        relay.writer('x', work_id='x')({'event': 'model_complete', 'model': 'PU'})
    assert capsys.readouterr().out == ''
    assert relay.done_units == 1 and len(relay.rows) == 1


def test_default_real_plan_includes_all_fifty_models_per_fold_repetition():
    from types import SimpleNamespace
    from gene2wire.experiments.pipeline import _planned_models
    settings = Settings()
    planned = _planned_models(SimpleNamespace(metadata={}, natural_observed=None), settings)
    assert len(planned) == 50
    assert len(planned) * settings.n_outer_folds * settings.n_repetitions == 750
    assert {row['model'] for row in planned} >= {'Qiao-ID-squared', 'Qiao-ID-logit',
        'Reference-only', 'Reference+PU', 'RF-reference', 'Prevalence-observed'}
