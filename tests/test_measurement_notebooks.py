"""Contracts for the combined measurement-degradation notebook family."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "notebooks" / "build_measurement_degradation.py"
NAMES = (
    "simulation",
    "BARseq_A1",
    "BARseq_M1",
    "MERGE_seq",
    "Projection_TAGs",
    "SPIDER",
    "SPIDER_Seq",
)
NOTEBOOK_FILENAMES = {
    "simulation": "simulation.ipynb",
    "BARseq_A1": "barseq_a1.ipynb",
    "BARseq_M1": "barseq_m1.ipynb",
    "MERGE_seq": "merge_seq.ipynb",
    "Projection_TAGs": "projection_tags.ipynb",
    "SPIDER": "spider_spatial.ipynb",
    "SPIDER_Seq": "spider_seq.ipynb",
}


def builder():
    spec = importlib.util.spec_from_file_location(
        "measurement_notebook_generator", BUILDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(book, kind=None):
    return "\n".join(
        cell["source"] for cell in book["cells"]
        if kind is None or cell["cell_type"] == kind
    )


def test_builder_generates_exactly_seven_stable_output_paths(tmp_path):
    module = builder()
    assert module.NAMES == NAMES
    assert module.NOTEBOOK_FILENAMES == NOTEBOOK_FILENAMES
    paths = module.build("a" * 40, "b" * 64, tmp_path, "0912")
    assert [path.name for path in paths] == [
        NOTEBOOK_FILENAMES[name] for name in NAMES
    ]
    assert {path.name for path in tmp_path.glob("*.ipynb")} == set(
        NOTEBOOK_FILENAMES.values()
    )
    assert len(paths) == len(set(paths)) == 7


@pytest.mark.parametrize("name", NAMES)
def test_generated_notebooks_are_valid_clean_navigable_python(name):
    book = builder().notebook(name, "a" * 40, "b" * 64, "0912")
    nbformat.validate(nbformat.from_dict(book))
    assert book["nbformat"] == 4
    ids = [cell["id"] for cell in book["cells"]]
    assert len(ids) == len(set(ids))
    for index, cell in enumerate(book["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        assert index > 0
        assert book["cells"][index - 1]["cell_type"] == "markdown"
        assert book["cells"][index - 1]["source"].startswith("## ")
        ast.parse(cell["source"])


@pytest.mark.parametrize("name", NAMES)
def test_notebooks_are_thin_source_pinned_ondemand_entrypoints(name):
    book = builder().notebook(name, "a" * 40, "b" * 64, "0912")
    code = _source(book, "code")
    provenance = book["metadata"]["gene2wire"]
    assert provenance == {
        "protocol": "0912-v5-measurement-degradation",
        "core_commit": "a" * 40,
        "source_hash": "b" * 64,
        "dataset": name,
        "notebook_date": "0912",
        "environment": "OnDemand",
        "experiment": "measurement_degradation",
    }
    assert "CORE_COMMIT = '" + "a" * 40 + "'" in code
    assert "EXPECTED_SOURCE_HASH = '" + "b" * 64 + "'" in code
    assert "https://github.com/Yue-stat/Gene2Wire.git" in code
    assert "GENE2WIRE_PROJECT_DIR" in code
    assert "paper_figure_exports" in code
    assert "pip install" not in code
    assert "google.colab" not in code
    assert ".png" not in code
    assert "class " not in code
    assert "make_observation_plan" not in code
    assert "draw_cyclic_panel_family" not in code


def test_config_uses_revised_coverage_not_legacy_fixed_k_overlap():
    book = builder().notebook("BARseq_A1", "a" * 40, "b" * 64, "0912")
    namespace = {}
    exec(book["cells"][2]["source"], namespace)
    expected = {
        "N_OUTER_FOLDS": 3,
        "N_JOBS": 32,
        "N_REPETITIONS": 5,
        "N_PANEL_SEEDS": 5,
        "USE_LOCATION": False,
        "USE_TARGET_FEATURES": False,
        "STRATEGY": "full_joint",
        "CANDIDATE_BUDGET": 32,
        "PAIRED_FRACTION": .20,
        "GENE_COVERAGES": (1.0, 5 / 6, 2 / 3, .5),
        "TARGET_COVERAGES": (1.0, 2 / 3, .5),
        "POSITIVE_RETENTIONS": (1.0, .75, .5, .25, .10),
        "ANCHOR_GENE_COVERAGE": 2 / 3,
        "ANCHOR_TARGET_COVERAGE": 2 / 3,
        "ANCHOR_POSITIVE_RETENTION": .5,
        "VIRTUAL_ASSAYS": ("A", "B", "C"),
        "INCLUDE_MATCHED_UNIFORM_CONTROL": True,
        "INCLUDE_NATURAL_RECOVERY": True,
    }
    for key, value in expected.items():
        assert namespace[key] == value
    assert "PANEL_SIZE" not in namespace
    markdown = _source(book, "markdown")
    assert "coverage 1.0 measures all 23/23 genes" in markdown
    assert "legacy same-eight-gene" in markdown
    code = _source(book, "code")
    assert "int(full.iloc[0]['panel_size']) != pool_size" in code


@pytest.mark.parametrize("name", NAMES)
def test_every_notebook_enables_all_shared_controls_and_pu_comparators(name):
    code = _source(
        builder().notebook(name, "a" * 40, "b" * 64, "0912"), "code"
    )
    for switch in (
        "RUN_INFORMATION_CONTROLS",
        "RUN_RANDOM_FOREST",
        "RUN_MECHANISM_CONTROLS",
        "RUN_CALIBRATION_CONTROLS",
        "RUN_QIAO",
        "RUN_PU_COMPARATORS",
    ):
        assert f"{switch} = True" in code
    assert "run_information_controls=RUN_INFORMATION_CONTROLS" in code
    assert "run_random_forest=RUN_RANDOM_FOREST" in code
    assert "run_mechanism_controls=RUN_MECHANISM_CONTROLS" in code
    assert "run_calibration_controls=RUN_CALIBRATION_CONTROLS" in code
    assert "run_qiao=RUN_QIAO" in code
    assert "run_pu_comparators=RUN_PU_COMPARATORS" in code
    assert "n_panel_seeds=N_PANEL_SEEDS" in code
    assert "supervision_profile='paired_reference'" in code
    for model in (
        "PU-Joint", "GenEML-adapted", "Inductive-PU-MC", "SAR-PU"
    ):
        assert model in code


@pytest.mark.parametrize("name", NAMES)
def test_results_only_guards_raw_loading_preview_and_fitting(name):
    book = builder().notebook(name, "a" * 40, "b" * 64, "0912")
    guarded = []
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        if any(token in cell["source"] for token in (
            "load_barseq(", "load_merge_seq_overlap(", "load_projection_tags(",
            "load_spider(", "load_spider_seq_measurement(", "generate_simulation(",
            "run_measurement_experiment(", "run_measurement_simulation_experiments(",
        )):
            assert cell["source"].startswith("if not RESULTS_ONLY:")
            guarded.append(cell["source"])
    assert len(guarded) >= 2
    for source in guarded:
        exec(compile(source, "results-only", "exec"), {"RESULTS_ONLY": True})


@pytest.mark.parametrize(
    ("name", "loader", "arguments"),
    (
        ("BARseq_A1", "load_barseq", "['A1']"),
        ("BARseq_M1", "load_barseq", "['M1']"),
        ("MERGE_seq", "load_merge_seq_overlap", "source_gene_count=MERGE_SOURCE_GENE_COUNT"),
        ("Projection_TAGs", "load_projection_tags", "location_features_csv=LOCATION_FEATURES_CSV"),
        ("SPIDER", "load_spider", "target_features_csv=TARGET_FEATURES_CSV"),
        ("SPIDER_Seq", "load_spider_seq_measurement", "gene_pool_size=SPIDER_SEQ_GENE_POOL_SIZE"),
    ),
)
def test_real_notebooks_use_the_required_shared_loader(name, loader, arguments):
    code = _source(
        builder().notebook(name, "a" * 40, "b" * 64, "0912"), "code"
    )
    assert f"import {loader}" in code
    assert f"{loader}(" in code
    assert arguments in code
    assert "preview_measurement_experiment(" in code
    assert "run_measurement_experiment(" in code


def test_simulation_uses_the_public_measurement_runner_and_current_truth_grid():
    code = _source(
        builder().notebook("simulation", "a" * 40, "b" * 64, "0912"),
        "code",
    )
    assert "run_measurement_simulation_experiments(" in code
    assert "SHARING_STRENGTHS = (0.0, 0.5, 1.0)" in code
    assert "'n_cells': 400" in code
    assert "'n_targets': 36" in code
    assert "'n_gene_features': 16" in code
    assert "'truth_uses_location': USE_LOCATION" in code


@pytest.mark.parametrize("name", NAMES)
def test_requested_plots_use_fixed_scopes_and_export_pdf(name):
    code = _source(
        builder().notebook(name, "a" * 40, "b" * 64, "0912"), "code"
    )
    assert "plot_retention_auprc(" in code
    assert "retention_col='positive_retention'" in code
    assert "for mode in ('key', 'full')" in code
    assert "plot_gene_target_brier_heatmap(" in code
    assert "heatmap_baseline_selection" in code
    assert "comparison_model='PU-MIRT'" in code
    assert "evaluation_scope'].eq('native_reference')" in code
    assert "figure.savefig(" in code
    assert ".pdf'" in code
    if name == "Projection_TAGs":
        assert "plot_projection_tags_budget_recall(" in code
        assert "projection_budget_recall" in code
        assert "amplification_confirmed_recall" in code
        assert "plot_accuracy_panel_sensitivity(" not in code
    else:
        assert "plot_accuracy_panel_sensitivity(" in code
        assert "panel_stability" in code
        assert "panel_sensitivity" in code
        assert "np.isclose(stability['target_coverage'], 1.0)" in code
        assert "is_reference_probability" in code
        assert "plot_projection_tags_budget_recall(" not in code


@pytest.mark.parametrize("name", NAMES)
def test_visible_audits_cover_panels_parameters_detection_and_failures(name):
    code = _source(
        builder().notebook(name, "a" * 40, "b" * 64, "0912"), "code"
    )
    for token in (
        "measurement_gene_panel_conditions",
        "measurement_gene_panel_membership",
        "measurement_target_panel_conditions",
        "measurement_target_panel_membership",
        "measurement_assignment",
        "measurement_scenarios",
        "measurement_support",
        "measurement_support_assays",
        "measurement_support_pairs",
        "measurement_gene_target_support",
        "measurement_unsupported_gene_target_pairs",
        "metrics",
        "selected",
        "tuning",
        "detection",
        "observation_diagnostics",
        "projection_budget_recall_per_target",
        "convergence",
        "failures",
        "display_diagnostics(artifacts",
    ):
        assert token in code


def test_public_measurement_notebook_api_is_importable():
    from gene2wire.experiments import measurement_experiment, measurement_plotting

    for name in (
        "MeasurementConfig",
        "preview_measurement_experiment",
        "run_measurement_experiment",
        "run_measurement_simulation_experiments",
    ):
        assert callable(getattr(measurement_experiment, name))
    for name in (
        "plot_retention_auprc",
        "plot_gene_target_brier_heatmap",
        "plot_accuracy_panel_sensitivity",
        "plot_projection_tags_budget_recall",
    ):
        assert callable(getattr(measurement_plotting, name))
