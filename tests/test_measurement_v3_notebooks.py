"""Contracts for the results-only measurement-degradation V3 notebooks."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "notebooks" / "build_measurement_degradation_v3.py"
V2_DIR = ROOT / "notebooks" / "measurement_degradation"
V3_DIR = ROOT / "notebooks" / "measurement_degradation_v3"

V2_HASHES = {
    "barseq_a1.ipynb": "6eb4301ea2511e3dcc37b888d09236c8e10221857cd16aaf5612c2c14fff75bc",
    "barseq_m1.ipynb": "0d246730aeca1ed5ff56475823900644d53c8336f0593d41c79a747b160e28c0",
    "merge_seq.ipynb": "9f61c7423d5e184887a0572cb43ee6e11152b1120c10c4e96263a772c92e0f27",
    "projection_tags.ipynb": "427f4d0afeb433e22038e83f6024330b7fbe817bd545ec5e3b369a0252022280",
    "simulation.ipynb": "1a3b0229945f20bfd9f05ce8876c4acca1ef336d4ee5e28cab1155d96badc628",
    "spider_seq.ipynb": "b7b9e398cd9d2f132e3fb0c829c850bf42b46b1a09d11f7baba0853767f262a7",
    "spider_spatial.ipynb": "cc7d7f1e93c8f839a87575bf8dae6233a398a8eeeec469ca296d75c157e982f4",
}


def builder():
    spec = importlib.util.spec_from_file_location("measurement_v3_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(book, kind=None):
    return "\n".join(
        cell["source"] for cell in book["cells"]
        if kind is None or cell["cell_type"] == kind
    )


def _config(book):
    return next(
        cell["source"] for cell in book["cells"]
        if cell["cell_type"] == "code"
        and "PROTOCOL_VERSION =" in cell["source"]
        and "RESULTS_ONLY" in cell["source"]
    )


def _v2_hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in V2_DIR.glob("*.ipynb")
    }


def test_v3_builder_writes_only_new_directory_and_preserves_v2_bytes(tmp_path):
    before = _v2_hashes()
    assert before == V2_HASHES
    paths = builder().build("a" * 40, "b" * 64, tmp_path / "v3", "0914")
    assert len(paths) == 7
    assert all(path.parent == tmp_path / "v3" for path in paths)
    assert _v2_hashes() == before


@pytest.mark.parametrize("name", (
    "simulation", "BARseq_A1", "BARseq_M1", "MERGE_seq",
    "Projection_TAGs", "SPIDER", "SPIDER_Seq",
))
def test_v3_notebooks_are_clean_results_only_and_never_run_models(name):
    book = builder().notebook(name, "a" * 40, "b" * 64, "0914")
    nbformat.validate(nbformat.from_dict(book))
    provenance = book["metadata"]["gene2wire"]
    assert provenance["view"] == "measurement_degradation_v3_results_only"
    assert provenance["source_result_core_commit"] == (
        builder().SOURCE_RESULTS_CORE_COMMIT
        if builder().EXACT_EXPORT_DIRS[name] is not None else None
    )
    assert provenance["source_result_hash"] == (
        builder().SOURCE_RESULTS_SOURCE_HASH
        if builder().EXACT_EXPORT_DIRS[name] is not None else None
    )
    namespace = {}
    if builder().EXACT_EXPORT_DIRS[name] is None:
        with pytest.raises(RuntimeError, match="No verified completed source export"):
            exec(_config(book), namespace)
    else:
        exec(_config(book), namespace)
    assert namespace["RESULTS_ONLY"] is True
    assert namespace["EXISTING_EXPORT_DIRS"] == {
        builder().LABELS[name]: builder().EXACT_EXPORT_DIRS[name]
    }
    assert "measurement_degradation_v3" in str(namespace["FIGURE_DIR"])
    code = _source(book, "code")
    if builder().EXACT_EXPORT_DIRS[name] is not None:
        for contract in (
            "manifest.get('completed') is not True",
            "manifest.get('run_id')",
            "manifest.get('source_hash')",
            "expected_measurement =",
            "required_tables =",
            "missing_tables =",
        ):
            assert contract in code
    forbidden = (
        "load_barseq", "load_merge_seq_overlap", "load_projection_tags",
        "load_spider", "load_spider_seq_measurement", "generate_simulation",
        "run_measurement_experiment", "run_measurement_simulation_experiments",
        "preview_measurement_experiment",
    )
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse(cell["source"])
        assert not any(token in cell["source"] for token in forbidden)
    assert "for directory in (RAW_DATA_DIR, CHECKPOINT_DIR, EXPORT_DIR, FIGURE_DIR)" not in code
    assert "FIGURE_DIR.mkdir(parents=True, exist_ok=True)" in code


def test_five_exact_exports_are_prefilled_and_missing_spider_runs_fail_closed():
    module = builder()
    expected_suffixes = {
        "simulation": "simulation_measurement_degradation/a447d9aaa72f2f847fb9",
        "BARseq_A1": "BARseq_A1_measurement_degradation/20561ae1ec481dbeab96",
        "BARseq_M1": "BARseq_M1_measurement_degradation/ad4abecbc92743f8d04f",
        "MERGE_seq": "MERGE-seq_measurement_degradation/d1b46506c668c646e54f",
        "Projection_TAGs": "Projection-TAGs_measurement_degradation/58f93f29e60290169415",
    }
    for name, suffix in expected_suffixes.items():
        assert module.EXACT_EXPORT_DIRS[name].endswith(suffix)
        assert len(Path(module.EXACT_EXPORT_DIRS[name]).name) == 20
    assert module.EXACT_EXPORT_DIRS["SPIDER"] is None
    assert module.EXACT_EXPORT_DIRS["SPIDER_Seq"] is None
    for name in ("SPIDER", "SPIDER_Seq"):
        code = _config(module.notebook(name, "a" * 40, "b" * 64, "0914"))
        assert "Keep RESULTS_ONLY=True" in code
        assert "source hash, and core commit" in code


def test_v3_plots_pair_auprc_with_log_loss_and_create_two_funky_scorecards():
    book = builder().notebook("BARseq_A1", "a" * 40, "b" * 64, "0914")
    code = _source(book, "code")
    # One definition plus calls from the retention and shared coverage cells.
    assert code.count("plot_degradation_pair(") == 3
    assert "x_col='positive_retention'" in code
    assert "x_col=coverage_col" in code
    assert "for mode in ('key', 'full')" in code
    assert "axis.invert_xaxis()" in code
    assert "plot_measurement_funky_summary(" in code
    assert "f'funkyheatmap_{mode}_v3'" in code
    assert "scorecard.to_csv(scorecard_path" in code
    assert "panel_stability_figure_paths = {}" not in code
    assert "plot_accuracy_panel_sensitivity(" not in code
    assert "artifacts.export_dir" not in next(
        cell["source"] for cell in book["cells"]
        if "funky_figure_paths = {}" in cell.get("source", "")
    )


def test_projection_v3_uses_conditional_h_and_false_positive_aware_metrics():
    book = builder().notebook("Projection_TAGs", "a" * 40, "b" * 64, "0914")
    code = _source(book, "code")
    markdown = _source(book, "markdown")
    assert "derive_projection_natural_metrics(artifacts)" in code
    assert "plot_projection_natural_comparison(" in code
    assert "plot_projection_budget_metrics(" in code
    assert "amplification_reference_false_positive_count" in code
    assert "micro_amplification_reference_fdr" in code
    assert "micro_amplification_reference_fpr" in code
    assert "(1.0 - e) * p" in code
    assert "not proven biological false" in markdown


def test_checked_in_v3_notebooks_match_the_builder_byte_for_byte(tmp_path):
    generated = builder().build(
        "25e7439d1c15c493714edb1149aee07768bbc3d6",
        "41aa4a33fead2d3f436d8a88ae62bf9d7777b4a169efbb3a206e58f6966c35d9",
        tmp_path,
        "0914",
    )
    for path in generated:
        assert path.read_bytes() == (V3_DIR / path.name).read_bytes()
