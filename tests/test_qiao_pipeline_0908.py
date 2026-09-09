"""Qiao experiment integration: scientific identity and interruption-safe resume."""
from dataclasses import replace

import numpy as np
import pytest

from gene2wire.experiments import pipeline, qiao
from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.protocol import Settings


def _inputs():
    settings = Settings(n_jobs=1, n_repetitions=1, use_target_features=True,
                        run_qiao=True, candidate_budget=2, penalties=(.003, .03),
                        maxiter=30, retry_maxiter=60, init_direct_maxiter=30,
                        tolerance=1e-6)
    data = generate_simulation(0, .5, seed=72, n_cells=54, n_targets=6,
                               n_gene_features=4, n_target_features=2, n_slices=9)
    fold = data.split_builder(3, 0)[0]
    prepared = pipeline._prepare(data, fold, settings)
    context = {"dataset": "simulation", "repetition": 0, "outer_fold": 0,
               "sharing_strength": .5, "analysis": "primary", "mechanism": "technical_sar",
               "loss_rate": .8, "calibration_fraction": .2, "calibration_spec": "correct"}
    return prepared, settings, context


def _run(prepared, settings, context, checkpoint, observed=None):
    records, tables = {}, {"selected": [], "tuning": []}

    def record(name, prediction, semantics, extra=None, *, ranking_score=None):
        records[name] = {"prediction": prediction.copy(), "semantics": semantics,
                         "ranking_score": ranking_score.copy(), "extra": extra}

    d = prepared.reference.copy() if observed is None else observed
    pipeline._run_qiao_controls(prepared, d, None, None, settings, context,
                               record, tables, checkpoint)
    return records, tables


def test_second_run_resumes_candidates_and_final_despite_runtime_and_other_model_flags(tmp_path, monkeypatch):
    prepared, settings, context = _inputs()
    monkeypatch.setattr(pipeline, "source_hash", lambda: "fixed-test-source")
    original = qiao.fit_qiao
    calls = []

    def spy(*args, **kwargs):
        calls.append((len(args[0]), kwargs["objective"], kwargs["initial_fit"] is not None))
        return original(*args, **kwargs)

    monkeypatch.setattr(qiao, "fit_qiao", spy)
    first, first_tables = _run(prepared, settings, context, tmp_path)
    count = len(calls)
    assert count >= 6  # Two candidates and one final refit per response objective.
    changed = replace(settings, n_jobs=32, run_random_forest=False,
                      run_information_controls=False, n_repetitions=5)
    # Test labels do not determine fits or their identities.
    d = prepared.reference.copy()
    d[prepared.fold.test_rows] = 1 - d[prepared.fold.test_rows]
    second, second_tables = _run(prepared, changed, context, tmp_path, d)
    assert len(calls) == count
    assert all(row["resumed"] for row in second_tables["tuning"] + second_tables["selected"])
    assert len(second_tables["tuning"]) == 4
    assert all(row["tuning_trials"] == 2 for row in second_tables["selected"])
    for name in first:
        np.testing.assert_array_equal(first[name]["prediction"], second[name]["prediction"])
        np.testing.assert_array_equal(first[name]["ranking_score"], second[name]["ranking_score"])
        assert first[name]["semantics"] == "observed"
    # Raw squared scores, not clipped probabilities, are delivered to ranking.
    assert first["Qiao-squared"]["extra"]["probability_transform"] == "clip_linear_score"
    assert all(row["selection_metric"] == "observed_log_loss" for row in first_tables["tuning"])


def test_interruption_preserves_the_first_completed_candidate_attempt(tmp_path, monkeypatch):
    prepared, settings, context = _inputs()
    monkeypatch.setattr(pipeline, "source_hash", lambda: "fixed-test-source")
    original = qiao.fit_qiao
    completed, interrupt_once = [], [True]

    def interrupted(*args, **kwargs):
        signature = (kwargs["objective"], kwargs["rank"], kwargs["l2"],
                     kwargs["initial_fit"] is not None, len(args[0]))
        if completed and interrupt_once[0]:
            interrupt_once[0] = False
            raise RuntimeError("simulated interruption")
        fitted = original(*args, **kwargs)
        completed.append(signature)
        return fitted

    monkeypatch.setattr(qiao, "fit_qiao", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run(prepared, settings, context, tmp_path)
    assert len(completed) == 1
    assert list((tmp_path / "qiao").glob("*.json"))
    first_attempt = completed[0]
    _, tables = _run(prepared, settings, context, tmp_path)
    assert completed.count(first_attempt) == 1
    assert len(tables["selected"]) == 2


def test_validation_outcomes_rescore_without_retraining_candidates_and_source_change_invalidates(tmp_path, monkeypatch):
    prepared, settings, context = _inputs()
    version = ["source-v1"]
    monkeypatch.setattr(pipeline, "source_hash", lambda: version[0])
    original = qiao.fit_qiao
    counts = {"candidate": 0, "final": 0}

    def counted(*args, **kwargs):
        phase = "candidate" if len(args[0]) == len(prepared.fold.train_rows) else "final"
        counts[phase] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(qiao, "fit_qiao", counted)
    _, initial = _run(prepared, settings, context, tmp_path)
    before = counts["candidate"]
    d = prepared.reference.copy()
    d[prepared.fold.validation_rows] = 1 - d[prepared.fold.validation_rows]
    _, changed = _run(prepared, settings, context, tmp_path, d)
    assert counts["candidate"] == before
    assert not np.allclose([r["validation_loss"] for r in initial["tuning"]],
                           [r["validation_loss"] for r in changed["tuning"]])
    version[0] = "source-v2"
    _run(prepared, settings, context, tmp_path, d)
    assert counts["candidate"] > before


def test_target_id_design_ignores_outcomes_and_grid_is_bounded_and_balanced():
    y, yr, kind = qiao.qiao_target_inputs(None, None, 7, use_target_features=False)
    np.testing.assert_array_equal(y, np.eye(7))
    np.testing.assert_array_equal(yr, y)
    assert kind == 'known_target_identity'
    with pytest.raises(ValueError, match='Disabled target features'):
        qiao.qiao_target_inputs(y, yr, 7, use_target_features=False)
    with pytest.raises(ValueError, match='requires aligned'):
        qiao.qiao_target_inputs(None, None, 7, use_target_features=True)
    penalties = (1e-4, 1e-3, 1e-2, .1, 1., 10.)
    grid = qiao.qiao_candidate_grid(23, 23, y, yr, penalties=penalties, candidate_budget=12)
    assert len(grid) == 12
    assert {row['l2'] for row in grid} == set(penalties)
    assert max(row['rank'] for row in grid) <= 7
    assert grid == qiao.qiao_candidate_grid(23, 23, y, yr,
                                           penalties=penalties, candidate_budget=12)


def test_real_data_path_runs_id_comparators_and_resumes(tmp_path, monkeypatch):
    prepared, settings, context = _inputs()
    settings = replace(settings, use_target_features=False, candidate_budget=3, loss_rates=(0.,),
        run_information_controls=False, run_random_forest=False,
        run_mechanism_controls=False, run_calibration_controls=False)
    prepared = replace(prepared, name='empirical-fixture', metadata={},
        train_features=replace(prepared.train_features, Y_target=None),
        refit_features=replace(prepared.refit_features, Y_target=None))
    # Exercise the actual real-data branch, not merely the comparator helper.
    events = []
    output = pipeline._run_fold(prepared, 0, settings, tmp_path / 'cache', tmp_path / 'export',
                                'source-fixture', on_progress=events.append)
    names = {'Qiao-ID-squared', 'Qiao-ID-logit'}
    assert {row['model'] for row in output['metrics']} >= names
    chosen = [row for row in output['selected'] if row['model'] in names]
    assert len(chosen) == 2
    assert all(row['target_input_kind'] == 'known_target_identity' for row in chosen)
    assert all(row['final_converged'] == row['converged'] for row in chosen)
    assert {row['model'] for row in events if row['event'] == 'model_complete'} >= names
    context = {**context, 'dataset': 'empirical-fixture'}
    # With only the feature switch changed, helper fits must use the ID mode and
    # reuse complete candidate/refit checkpoints on their second invocation.
    first, _ = _run(prepared, settings, context, tmp_path / 'helper')
    def forbidden(*args, **kwargs):
        raise AssertionError('Qiao resumed fit unexpectedly retrained')
    monkeypatch.setattr(qiao, 'fit_qiao', forbidden)
    second, tables = _run(prepared, settings, context, tmp_path / 'helper')
    assert all(row['resumed'] for row in tables['selected'])
    for name in names:
        np.testing.assert_array_equal(first[name]['prediction'], second[name]['prediction'])
