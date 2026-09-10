"""Diagnostics must retain missing evidence and reload old exports without fitting."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from gene2wire.experiments.reporting import (
    compact_summaries, diagnostic_summaries, display_diagnostics, load_existing_exports,
    load_export_artifacts,
)


def write_export(path, *, name="BARseq A1", completed=True):
    path.mkdir()
    manifest = {"completed": completed, "source_hash": "older-core",
                "protocol": {"n_repetitions": 4}, "datasets": [{"name": name}],
                "table_files": ["metrics.csv", "selected.csv", "failures.csv"]}
    (path / "manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame({"dataset": [name], "model": ["PU-Joint"], "macro_auprc": [.31]}).to_csv(path / "metrics.csv", index=False)
    pd.DataFrame({"model": ["PU-Joint"], "rank": [0], "kind": ["direct"],
                  "final_converged": [True]}).to_csv(path / "selected.csv", index=False)
    (path / "failures.csv").write_text("\n")
    return path


def test_results_reload_preserves_manifest_and_empty_tables_without_fitting(tmp_path, monkeypatch):
    import gene2wire.experiments.pipeline as pipeline
    def forbidden(*args, **kwargs):
        raise AssertionError("Results reload attempted fitting")
    monkeypatch.setattr(pipeline, "run_experiment", forbidden)
    path = write_export(tmp_path / "old-run")
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    result = load_export_artifacts(path)
    assert result.manifest["source_hash"] == "older-core"
    assert result.manifest["protocol"]["n_repetitions"] == 4
    assert result.tables["metrics"].loc[0, "macro_auprc"] == .31
    assert result.tables["failures"].empty
    assert before == {p.name: p.read_bytes() for p in path.iterdir()}


def test_results_only_requires_both_correct_barseq_panels(tmp_path):
    a1 = write_export(tmp_path / "a1")
    m1 = write_export(tmp_path / "m1", name="BARseq M1")
    with pytest.raises(ValueError, match="exactly these keys"):
        load_existing_exports({"A1": a1}, expected_labels=("A1", "M1"))
    with pytest.raises(ValueError, match="explicit completed"):
        load_existing_exports({"A1": a1, "M1": None}, expected_labels=("A1", "M1"))
    with pytest.raises(ValueError, match="contains datasets"):
        load_existing_exports({"A1": m1, "M1": a1}, expected_labels=("A1", "M1"))
    assert tuple(load_existing_exports({"A1": a1, "M1": m1}, expected_labels=("A1", "M1"))) == ("A1", "M1")


def test_missing_and_incomplete_results_are_not_silently_dropped(tmp_path):
    path = write_export(tmp_path / "incomplete", completed=False)
    with pytest.raises(ValueError, match="incomplete run"):
        load_export_artifacts(path)
    path = write_export(tmp_path / "complete")
    (path / "selected.csv").unlink()
    with pytest.raises(FileNotFoundError, match="declared result table is missing"):
        load_export_artifacts(path)


def test_manifest_cannot_read_tables_outside_export_directory(tmp_path):
    path = write_export(tmp_path / "unsafe")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["table_files"] = ["../metrics.csv"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid result table filename"):
        load_export_artifacts(path)


def test_convergence_unknown_is_neither_success_nor_failure():
    selected = pd.DataFrame({"model": ["PU-Joint"] * 3 + ["RF"],
                             "kind": ["direct", "lowrank", "joint", None],
                             "rank": [0, 2, 2, None],
                             "final_converged": [True, False, None, None]})
    summaries = diagnostic_summaries({"selected": selected})
    actual = summaries["final_fit_convergence"]
    counts = actual.loc[actual.model.eq("PU-Joint")].set_index("convergence_status")["fit_count"].to_dict()
    assert counts == {"converged": 1, "not_converged": 1, "not_recorded": 1}
    assert actual.loc[actual.model.eq("RF"), "convergence_status"].tolist() == ["not_recorded"]
    assert summaries["candidate_fit_convergence"].empty
    assert summaries["selected_configuration_frequency"]["selected_count"].sum() == 4


def test_all_original_tables_displayed_untruncated_and_empty_explicit(tmp_path, capsys):
    result = load_export_artifacts(write_export(tmp_path / "run"))
    result.tables["extra_table"] = pd.DataFrame({"long": ["x" * 500] * 100})
    shown = []
    # Retain test-process pandas settings despite deliberately global notebook configuration.
    with pd.option_context("display.max_rows", 2, "display.max_columns", 2,
                           "display.max_colwidth", 10, "display.max_seq_items", 2,
                           "display.width", 80, "display.expand_frame_repr", True):
        display_diagnostics(result, display_fn=shown.append, full=True)
        for option in ("display.max_rows", "display.max_columns", "display.max_colwidth", "display.max_seq_items"):
            assert pd.get_option(option) is None
        for original in result.tables.values():
            assert any(value is original for value in shown)
    output = capsys.readouterr().out
    assert "older-core" in output
    assert "failures: 0 rows" in output
    assert "This exported table has zero rows" in output
    assert "extra_table: 100 rows" in output


def test_compact_default_keeps_useful_endpoints_without_dumping_raw_tables(tmp_path, capsys):
    result = load_export_artifacts(write_export(tmp_path / "compact"))
    result.tables["aggregate"] = pd.DataFrame({
        "dataset": ["BARseq A1"] * 5, "analysis": ["primary"] * 4 + ["calibration_size"],
        "model": ["PU", "PU", "Reference-only", "Reference+PU", "PU"],
        "loss_rate": [0., .8, .8, .8, .8], "macro_auprc": [.2, .3, .4, .5, .9],
        "macro_log_loss": [.5, .4, .3, .2, .1], "macro_hidden_recall_at_h": [None, .2, .3, .4, .9]})
    result.tables["tuning"] = pd.DataFrame({"model": ["PU"] * 1000, "rank": [0] * 1000,
                                            "residual_l2": [.001] * 1000, "converged": [True] * 1000})
    result.tables["extra_table"] = pd.DataFrame({"long": ["x" * 500] * 100})
    before = {name: frame.copy(deep=True) for name, frame in result.tables.items()}
    shown = []
    display_diagnostics(result, display_fn=shown.append)
    summary = shown[0]
    assert summary.model.tolist() == ["PU", "Reference-only", "Reference+PU"]
    assert summary.macro_auprc.tolist() == [.3, .4, .5]
    assert "macro_hidden_recall_at_h" in summary
    assert all(value is not result.tables["tuning"] and value is not result.tables["extra_table"] for value in shown)
    assert not any(len(value) > 10 for value in shown)
    for name in before:
        pd.testing.assert_frame_equal(before[name], result.tables[name])
    output = capsys.readouterr().out
    assert "complete diagnostics" not in output and "older-core" not in output
    assert "Recorded failed units: 0" in output
    assert "SHOW_FULL_DIAGNOSTICS=True" in output


def test_compact_natural_endpoint_and_failure_status_are_preserved():
    frames = {"aggregate": pd.DataFrame({"dataset": ["Projection-TAGs"] * 2,
        "model": ["PU", "Reference-only"], "loss_rate": [None, None],
        "macro_auprc": [.2, .3]}),
        "selected": pd.DataFrame({"model": ["PU"] * 3, "rank": [0] * 3,
            "residual_l2": [.001, .001, .1], "final_converged": [True, False, None]}),
        "failures": pd.DataFrame({"outer_fold": [1], "reason": ["calibration failed"]})}
    summaries = compact_summaries(frames)
    assert len(summaries["primary_endpoint_metrics"]) == 2
    row = summaries["selection_and_convergence_all_primary_rates"].iloc[0]
    assert (row["converged"], row["not_converged"], row["not_recorded"]) == (1, 1, 1)
    assert row["unique_selected_configs"] == 2
    assert row["mode_count"] == 2
    assert len(summaries["nonconverged_final_fits_first_10_see_selected_csv"]) == 1
    assert summaries["failed_units_first_10_see_failures_csv"].reason.tolist() == ["calibration failed"]


def test_compact_baseline_parameters_and_nonprimary_convergence():
    selected = pd.DataFrame([
        dict(model='RF-reference', analysis='primary', n_estimators=300,
             min_samples_leaf=1, max_features='sqrt', max_depth=None),
        dict(model='RF-reference', analysis='primary', n_estimators=300,
             min_samples_leaf=5, max_features='sqrt', max_depth=12),
        dict(model='Qiao-ID-logit', analysis='primary', rank=2, l2=.001, objective='logit', converged=True),
        dict(model='Qiao-ID-logit', analysis='primary', rank=2, l2=.1, objective='logit', converged=True),
        dict(model='PU', analysis='calibration_misspecification', rank=0, final_converged=False)])
    result = compact_summaries({'selected': selected, 'tuning': selected})
    modes = result['selection_and_convergence_all_primary_rates'].set_index('model')
    assert modes.loc['RF-reference', 'unique_selected_configs'] == 2
    assert 'depth=unlimited' in modes.loc['RF-reference', 'most_selected_config']
    assert modes.loc['Qiao-ID-logit', 'unique_selected_configs'] == 2
    assert modes.loc['Qiao-ID-logit', 'converged'] == 2
    status = result['convergence_all_scenarios'].set_index('records')
    assert status.loc['final fits', 'not_converged'] == 1
    assert result['nonconverged_final_fits_first_10_see_selected_csv'].model.tolist() == ['PU']
    assert result['candidate_coverage_all_primary_rates'].set_index('model').loc[
        'Qiao-ID-logit', 'unique_bilinear_penalties'] == 2
    only_control = compact_summaries({'selected': selected.iloc[[-1]]})
    assert only_control['convergence_all_scenarios'].iloc[0]['not_converged'] == 1


def test_block_summaries_keep_panel_coordinates_without_hidden_recall():
    rows, selections = [], []
    base = {"dataset": "BARseq A1", "analysis": "primary", "model": "PU-Joint",
            "experiment": "animal_target_blocks", "group_mode": "animal"}
    for panel, training_fraction in (("masked", .2), ("masked", .8), ("full", .0)):
        unit = {**base, "training_panel": panel, "training_block_fraction": training_fraction}
        selections.append({**unit, "rank": 2, "kind": "joint", "final_converged": True,
                           "converged": True})
        for evaluation_fraction in (.2, .8):
            for scope in ("blocked", "observed"):
                for rate in (.0, .8):
                    rows.append({**unit, "block_fraction": evaluation_fraction,
                                 "evaluation_scope": scope, "loss_rate": rate,
                                 "macro_auprc": .3, "macro_log_loss": .4, "macro_brier": .1})
    summaries = compact_summaries({"aggregate": pd.DataFrame(rows),
                                   "selected": pd.DataFrame(selections),
                                   "tuning": pd.DataFrame(selections)})
    endpoints = summaries["primary_endpoint_metrics"]
    assert len(endpoints) == 12
    assert (endpoints.loss_rate == .8).all()
    for key in ("experiment", "group_mode", "training_panel", "training_block_fraction",
                "block_fraction", "evaluation_scope"):
        assert key in endpoints
    assert "macro_brier" in endpoints
    assert not any("hidden" in key for key in endpoints)
    for name in ("selection_and_convergence_all_primary_rates", "candidate_coverage_all_primary_rates"):
        frame = summaries[name]
        assert len(frame) == 3
        assert {"training_panel", "training_block_fraction", "experiment", "group_mode"} <= set(frame)
