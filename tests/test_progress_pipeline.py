from unittest.mock import patch

import numpy as np

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.pipeline import run_experiment
from gene2wire.experiments.protocol import Settings
from gene2wire.experiments.progress import PhaseProgress, ProgressRelay
from gene2wire.models import UnifiedPUModel


def test_whole_unit_inventory_and_missing_export_recovery(tmp_path, capsys):
    data = generate_simulation(0, .5, seed=9208, n_cells=120, n_targets=6,
                               n_gene_features=5, n_slices=12)
    settings = Settings(n_jobs=1, n_repetitions=1, penalties=(.01,), candidate_budget=3,
                        maxiter=50, retry_maxiter=100, tolerance=1e-5,
                        loss_rates=(0.,), run_random_forest=False,
                        run_information_controls=False, run_calibration_controls=False,
                        run_mechanism_controls=False, run_qiao=False)
    worker_snapshots = []
    kwargs = dict(checkpoint_dir=tmp_path / 'checkpoints', export_dir=tmp_path / 'exports',
                  worker_status=worker_snapshots.append)
    first = run_experiment(data, settings, **kwargs)
    output = capsys.readouterr().out
    assert "[phase] feature preparation and run identity: started" in output
    assert "[phase] completed-result cache inventory: started" in output
    assert 'verified result summaries 0/18' in output
    assert 'unchecked does not mean uncached' in output
    assert 'finished units 18/18' in output
    assert '[start]' not in output and '[done]' not in output and '[trial]' not in output
    assert "[phase] result consolidation: started" in output
    assert "[phase] summary and diagnostic tables: started" in output
    assert "[phase] CSV export: started" in output
    assert "[phase] core export finalization: completed" in output
    assert output.rfind("[phase] core export finalization: completed") < output.rfind(
        "All results exported to:")
    assert not first.tables['checkpoint_inventory']['fully_cached'].any()
    events = first.tables['progress_events']
    assert (events['event'] == 'unit_complete').sum() == 3
    assert (events['event'] == 'task_group_start').sum() == 3
    assert (events['event'] == 'task_group_complete').sum() == 3
    assert worker_snapshots[-1]['phase'] == 'finished'
    assert worker_snapshots[-1]['occupied_workers'] == 0
    assert worker_snapshots[-1]['available_workers'] == 1
    assert set(events.loc[events['event'] == 'model_complete', 'model']) >= {'PU-Joint', 'PU'}
    with patch.object(UnifiedPUModel, 'fit', side_effect=AssertionError('Unexpected fit')):
        second = run_experiment(data, settings, **kwargs)
        assert second.tables['checkpoint_inventory']['fully_cached'].all()
        assert second.tables['selected']['resumed'].all()
        assert worker_snapshots[-1]['worker_slots'] == 0
        assert worker_snapshots[-1]['occupied_workers'] == 0
        assert 'verified result summaries 18/18' in capsys.readouterr().out
        prediction = next(first.export_dir.rglob('PU-Joint_predictions.npz'))
        with np.load(prediction) as archive:
            expected = archive['prediction'].copy()
        prediction.unlink()
        rebuilt = run_experiment(data, settings, progress=False, **kwargs)
        assert rebuilt.tables['checkpoint_inventory']['fully_cached'].sum() == 2
        assert len(rebuilt.tables['model_evaluation_plan']) == 18
        assert rebuilt.manifest['cached_model_evaluations'] == 12
        assert rebuilt.manifest['reused_fit_model_evaluations'] == 6
        assert rebuilt.manifest['new_or_mixed_model_evaluations'] == 0
        assert rebuilt.manifest['unknown_fit_model_evaluations'] == 0
        assert rebuilt.tables['model_cache_accounting']['accounting'].value_counts().to_dict() == {
            'restored_results': 12, 'reused_fits': 6}
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
    assert '[cache] verified result summaries 2/2' in lines[0] and '[finished 0min] finished units 2/2' in lines[1]
    with ProgressRelay(tmp_path / 'quiet', enabled=False, total_units=2) as relay:
        relay.writer('x', work_id='x')({'event': 'model_complete', 'model': 'PU'})
    assert capsys.readouterr().out == ''
    assert relay.done_units == 1 and len(relay.rows) == 1


def test_phase_progress_has_counts_timing_heartbeat_and_quiet_mode(capsys):
    with patch("gene2wire.experiments.progress.time.monotonic", return_value=100.):
        with PhaseProgress("cache inventory", total=2, unit="scenarios") as phase:
            phase.advance(detail="fold 0")
            phase.advance(detail="fold 1")
            phase.set_summary("2 reusable summaries")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[phase] cache inventory: started (2 scenarios)"
    assert "completed 2/2 scenarios; elapsed 0.0s; 2 reusable summaries" in lines[-1]

    heartbeat = PhaseProgress("feature hashes", interval=60., total=3, unit="folds")
    heartbeat.started = heartbeat.last_heartbeat = 100.
    heartbeat.advance(detail="SPIDER-Seq fold 1")
    assert not heartbeat.heartbeat(159.)
    assert heartbeat.heartbeat(160.)
    output = capsys.readouterr().out
    assert "running 1/3 folds; elapsed 60.0s" in output
    assert "current SPIDER-Seq fold 1" in output

    with PhaseProgress("quiet", enabled=False, total=1) as quiet:
        quiet.advance()
    assert capsys.readouterr().out == ""

    try:
        with PhaseProgress("failing", total=1) as failing:
            failing.set_detail("bad fold")
            raise RuntimeError("expected")
    except RuntimeError as error:
        assert str(error) == "expected"
    assert failing.thread is not None and not failing.thread.is_alive()
    assert "[phase] failing: interrupted 0/1 items" in capsys.readouterr().out


def test_summary_heartbeat_names_fit_or_cache_reconstruction(tmp_path, capsys):
    relay = ProgressRelay(tmp_path, total_units=2, interval=60.)
    relay.started = relay.last_heartbeat = 100.
    first = relay.writer(
        "first", work_id="first", dataset="Projection-TAGs", repetition=1,
        outer_fold=2, analysis="primary", mechanism="assay_target_sar",
        gene_requested_coverage=.7, target_requested_coverage=.7,
        positive_retention=.4,
    )
    second = relay.writer(
        "second", work_id="second", dataset="Projection-TAGs", repetition=0,
        outer_fold=1, analysis="primary", mechanism="assay_target_sar",
    )
    first({"event": "candidate_start", "model": "GenEML-adapted",
           "index": 7, "total": 32, "stage": "tuning",
           "cache_status": "pending"})
    second({"event": "candidate_start", "model": "PU-Joint",
            "index": 12, "total": 32, "stage": "penalty_refinement",
            "cache_status": "checkpoint"})
    relay.drain()
    assert relay.heartbeat(160.)
    output = capsys.readouterr().out
    assert "active tasks 2" in output
    assert "candidate optimizer fit/retry 7/32 (tuning)" in output
    assert "candidate cache validation/reconstruction 12/32" in output
    assert "Projection-TAGs | rep=1 | fold=2 | primary | assay_target_sar" in output
    assert "gene=0.7 | target=0.7 | retain=0.4 | GenEML-adapted" in output
    assert "candidate fit/cache work" in ProgressRelay._active_description(
        {"event": "candidate_start", "cache_status": "unknown"}, "Qiao-ID-logit")
    target_activity = ProgressRelay._active_description(
        {
            "event": "target_start",
            "target_index": 4,
            "target_total": 7,
            "target_id": "SC",
            "stage": "refit",
        },
        "Projection-TAGs | SAR-PU",
    )
    assert "SAR-PU target optimizer fit 4/7 (SC) [refit]" in target_activity


def test_default_real_plan_runs_identical_fifteen_models_at_every_primary_rate():
    from types import SimpleNamespace
    from gene2wire.experiments.pipeline import _planned_models
    settings = Settings()
    planned = _planned_models(SimpleNamespace(metadata={}, natural_observed=None), settings)
    assert len(planned) == 75
    assert len(planned) * settings.n_outer_folds * settings.n_repetitions == 1125
    models = {row['model'] for row in planned}
    assert len(models) == 15
    assert models >= {'Qiao-ID-squared', 'Qiao-ID-logit', 'Reference-only', 'Reference+PU',
                      'Reference+PU-MIRT', 'Reference+PU-Joint',
                      'RF-observed', 'RF-reference', 'RF-mixed'}
    assert not models & {'Prevalence-observed', 'Prevalence-reference'}
    for rate in settings.loss_rates:
        assert {row['model'] for row in planned if row['loss_rate'] == rate} == models
    natural = _planned_models(SimpleNamespace(metadata={}, natural_observed=True), settings)
    assert len(natural) == 15 and {row['loss_rate'] for row in natural} == {None}
