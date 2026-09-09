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
                        run_mechanism_controls=False)
    kwargs = dict(checkpoint_dir=tmp_path / 'checkpoints', export_dir=tmp_path / 'exports')
    first = run_experiment(data, settings, **kwargs)
    output = capsys.readouterr().out
    assert '0/3 complete fold/repetition units' in output
    assert '[start]' in output and '[done]' in output and 'candidates reusable' in output
    assert not first.tables['checkpoint_inventory']['fully_cached'].any()
    events = first.tables['progress_events']
    assert (events['event'] == 'unit_complete').sum() == 3
    assert set(events.loc[events['event'] == 'model_complete', 'model']) >= {'PU-Joint', 'PU'}
    with patch.object(UnifiedPUModel, 'fit', side_effect=AssertionError('Unexpected fit')):
        second = run_experiment(data, settings, **kwargs)
        assert second.tables['checkpoint_inventory']['fully_cached'].all()
        assert second.tables['selected']['resumed'].all()
        assert '3/3 complete fold/repetition units' in capsys.readouterr().out
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
    relay = ProgressRelay(tmp_path, enabled=True)
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
