"""A model wrapper or final-refit hit must not masquerade as its entire fit history."""
from gene2wire.experiments.progress import ProgressRelay


def test_model_counts_partition_restored_reused_new_and_unknown(tmp_path, capsys):
    relay = ProgressRelay(tmp_path, total_units=6, cached_units=1)
    writer = relay.writer('scenario', work_id='scenario', dataset='example', loss_rate=.8)

    def event(model, event, status=None):
        writer({'model': model, 'event': event, 'cache_status': status})

    event('whole model cache', 'model_complete', 'checkpoint')
    # Core labels the rebuilt wrapper "fitted", but no new fit happened.
    event('rebuilt wrapper', 'candidate_complete', 'checkpoint')
    event('rebuilt wrapper', 'refit_complete', 'memory')
    event('rebuilt wrapper', 'model_complete', 'fitted')
    # RF can reuse the final refit while having fitted new candidates.
    event('mixed RF', 'candidate_complete', 'fitted')
    event('mixed RF', 'refit_complete', 'checkpoint')
    event('mixed RF', 'model_complete', 'checkpoint')
    # Qiao can reuse all candidates but require a fresh final fit.
    event('mixed Qiao', 'candidate_complete', 'checkpoint')
    event('mixed Qiao', 'refit_complete', 'fitted')
    event('mixed Qiao', 'model_complete', 'fitted')
    # Legacy incomplete telemetry is kept explicitly unknown.
    event('unknown', 'model_complete', 'fitted')
    event('rebuilt wrapper', 'model_complete', 'fitted')  # Duplicate has no effect.
    relay.drain()
    assert relay.done_units == 6
    assert relay.cached_units == 1
    assert relay.reused_fit_units == 2
    assert relay.new_or_mixed_units == 2
    assert relay.unknown_units == 1
    assert len(relay.model_accounting) == 5
    assert relay.done_units == (relay.cached_units + relay.reused_fit_units
                                + relay.new_or_mixed_units + relay.unknown_units)
    relay.heartbeat(relay.last_heartbeat + 61)
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert 'finished units 6/6' in output
    assert 'restored results 1, reused fits 2, new/mixed 2, fit status unknown 1' in output


def test_all_fits_reused_does_not_mean_previously_completed_results(tmp_path, capsys):
    with ProgressRelay(tmp_path, total_units=1, cached_units=0) as relay:
        writer = relay.writer('x', work_id='x')
        writer({'event': 'candidate_complete', 'model': 'PU', 'cache_status': 'checkpoint'})
        writer({'event': 'refit_complete', 'model': 'PU', 'cache_status': 'checkpoint'})
        writer({'event': 'model_complete', 'model': 'PU', 'cache_status': 'fitted'})
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 2
    assert 'verified result summaries 0/1' in output
    assert 'restored results 0, reused fits 1, new/mixed 0' in output
