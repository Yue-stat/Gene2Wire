"""Diagnostics must retain missing evidence and reload old exports without fitting."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from gene2wire.experiments.reporting import (
    diagnostic_summaries, display_diagnostics, load_existing_exports,
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
        display_diagnostics(result, display_fn=shown.append)
        for option in ("display.max_rows", "display.max_columns", "display.max_colwidth", "display.max_seq_items"):
            assert pd.get_option(option) is None
        for original in result.tables.values():
            assert any(value is original for value in shown)
    output = capsys.readouterr().out
    assert "older-core" in output
    assert "failures: 0 rows" in output
    assert "This exported table has zero rows" in output
    assert "extra_table: 100 rows" in output
