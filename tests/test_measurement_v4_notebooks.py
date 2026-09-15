"""Contracts for runnable six-model measurement-degradation V4 notebooks."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT / "scripts" / "notebooks" / "build_measurement_degradation_v4.py"
)
V4_DIR = ROOT / "notebooks" / "measurement_degradation_v4"
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
        "measurement_v4_notebook_generator", BUILDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(book, kind=None):
    return "\n".join(
        cell["source"]
        for cell in book["cells"]
        if kind is None or cell["cell_type"] == kind
    )


def _cell(book, marker):
    matches = [
        cell["source"]
        for cell in book["cells"]
        if cell["cell_type"] == "code" and marker in cell["source"]
    ]
    assert len(matches) == 1, marker
    return matches[0]


def test_v4_builder_generates_exactly_seven_stable_output_paths(tmp_path):
    module = builder()
    assert module.NAMES == NAMES
    assert module.NOTEBOOK_FILENAMES == NOTEBOOK_FILENAMES
    paths = module.build("a" * 40, "b" * 64, tmp_path, "0915")
    assert [path.name for path in paths] == [
        NOTEBOOK_FILENAMES[name] for name in NAMES
    ]
    assert {path.name for path in tmp_path.glob("*.ipynb")} == set(
        NOTEBOOK_FILENAMES.values()
    )
    assert len(paths) == len(set(paths)) == 7


@pytest.mark.parametrize("name", NAMES)
def test_v4_notebooks_are_clean_valid_source_pinned_python(name):
    book = builder().notebook(name, "a" * 40, "b" * 64, "0915")
    nbformat.validate(nbformat.from_dict(book))
    provenance = book["metadata"]["gene2wire"]
    assert provenance == {
        "protocol": builder().PROTOCOL_VERSION,
        "core_commit": "a" * 40,
        "source_hash": "b" * 64,
        "dataset": name,
        "notebook_date": "0915",
        "environment": "OnDemand",
        "experiment": "measurement_degradation_v4",
        "view": "six_model_multimetric_funkyheatmap",
    }
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

    code = _source(book, "code")
    assert "CORE_COMMIT = '" + "a" * 40 + "'" in code
    assert "EXPECTED_SOURCE_HASH = '" + "b" * 64 + "'" in code
    assert "https://github.com/Yue-stat/Gene2Wire.git" in code
    assert "measurement_degradation_v4_compact" in code
    assert "figures' / 'measurement_degradation_v4'" in code
    assert "'funkyheatmappy'" in code
    assert "pip install" not in code
    assert "google.colab" not in code
    markdown = _source(book, "markdown")
    assert "pinned authors' SAR-EM kernel" in markdown
    assert "not an execution of the authors' unmodified end-to-end program" in markdown
    assert "pinned authors-source Python 3" in markdown
    assert "global per-target exposure model" in markdown
    assert "labelled paper-based" in markdown


def test_v4_configuration_declares_new_grid_and_only_six_models():
    book = builder().notebook("BARseq_A1", "a" * 40, "b" * 64, "0915")
    configuration = _cell(book, "POSITIVE_RETENTIONS =")
    namespace = {}
    exec(configuration, namespace)
    expected = {
        "N_REPETITIONS": 2,
        "N_PANEL_SEEDS": 2,
        "GENE_COVERAGES": (1.0, 0.7, 0.4),
        "TARGET_COVERAGES": (1.0, 0.7, 0.4),
        "POSITIVE_RETENTIONS": (1.0, 0.8, 0.6, 0.4, 0.2),
        "ANCHOR_GENE_COVERAGE": 0.7,
        "ANCHOR_TARGET_COVERAGE": 0.7,
        "ANCHOR_POSITIVE_RETENTION": 0.4,
        "CORE_MODEL_ALLOWLIST": ("PU", "PU-MIRT", "PU-Joint"),
        "PU_COMPARATOR_MODELS": (
            "GenEML-authors-mask",
            "Inductive-PU-MC",
            "SAR-PU",
        ),
        "RUN_INFORMATION_CONTROLS": False,
        "RUN_RANDOM_FOREST": False,
        "RUN_MECHANISM_CONTROLS": False,
        "RUN_CALIBRATION_CONTROLS": False,
        "RUN_QIAO": False,
        "RUN_PU_COMPARATORS": True,
        "INCLUDE_MATCHED_UNIFORM_CONTROL": False,
        "INCLUDE_NATURAL_RECOVERY": False,
    }
    for key, value in expected.items():
        assert namespace[key] == value

    settings = _cell(book, "measurement_config = MeasurementConfig(")
    assert "schedule='v4'" in settings
    assert "model_allowlist=CORE_MODEL_ALLOWLIST" in settings
    assert "loss_rates=tuple(1.0 - value for value in POSITIVE_RETENTIONS)" in settings
    for switch in (
        "run_information_controls=RUN_INFORMATION_CONTROLS",
        "run_random_forest=RUN_RANDOM_FOREST",
        "run_mechanism_controls=RUN_MECHANISM_CONTROLS",
        "run_calibration_controls=RUN_CALIBRATION_CONTROLS",
        "run_qiao=RUN_QIAO",
        "run_pu_comparators=RUN_PU_COMPARATORS",
    ):
        assert switch in settings


@pytest.mark.parametrize("name", NAMES)
def test_v4_results_only_guards_raw_loading_preflight_and_fitting(name):
    book = builder().notebook(name, "a" * 40, "b" * 64, "0915")
    guarded = []
    tokens = (
        "load_barseq(",
        "load_merge_seq_overlap(",
        "load_projection_tags(",
        "load_spider(",
        "load_spider_seq_measurement(",
        "generate_simulation(",
        "run_measurement_experiment(",
        "run_measurement_simulation_experiments(",
    )
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        if any(token in cell["source"] for token in tokens):
            assert cell["source"].startswith("if not RESULTS_ONLY:")
            guarded.append(cell["source"])
    assert len(guarded) >= 2
    for source in guarded:
        exec(compile(source, "results-only", "exec"), {"RESULTS_ONLY": True})

    code = _source(book, "code")
    assert "RESULTS_ONLY=True requires the exact completed export directory" in code
    assert "saved.get('schedule') != 'v4'" in code
    assert "saved.get('model_allowlist')" in code
    assert "available.difference(EXPECTED_INTERNAL_MODELS)" in code
    assert "unavailable declared models will be shown as N/A" in code
    assert "No fallback model or partial-model funky normalization" in code


def test_v4_results_only_rejects_missing_declared_physical_condition():
    book = builder().notebook("BARseq_A1", "a" * 40, "b" * 64, "0915")
    validation = _cell(book, "def _validate_v4_export_factorial(")
    rows = []
    for repetition in (0, 1):
        for gene in (1.0, 0.7, 0.4):
            for target in (1.0, 0.7, 0.4):
                for retention in (1.0, 0.8, 0.6, 0.4, 0.2):
                    if (repetition, gene, target, retention) == (1, 1.0, 1.0, 0.8):
                        continue
                    rows.append(
                        {
                            "repetition": repetition,
                            "model": "PU",
                            "evaluation_scope": "native_reference",
                            "mechanism": "assay_target_sar",
                            "gene_requested_coverage": gene,
                            "target_requested_coverage": target,
                            "positive_retention": retention,
                            "sharing_strength": np.nan,
                        }
                    )
    manifest = {
        "experiment": "measurement_degradation",
        "completed": True,
        "source_hash": "b" * 64,
        "protocol": {
            "n_repetitions": 2,
            "protocol_version": builder().PROTOCOL_VERSION,
        },
        "measurement_protocol": {
            "gene_coverages": (1.0, 0.7, 0.4),
            "target_coverages": (1.0, 0.7, 0.4),
            "retentions": (1.0, 0.8, 0.6, 0.4, 0.2),
            "schedule": "v4",
            "model_allowlist": ("PU", "PU-MIRT", "PU-Joint"),
            "n_panel_seeds": 2,
        },
    }
    namespace = {
        "RESULTS_ONLY": True,
        "all_artifacts": {
            "BARseq A1": SimpleNamespace(
                manifest=manifest,
                tables={"per_repetition": pd.DataFrame(rows)},
            )
        },
        "GENE_COVERAGES": (1.0, 0.7, 0.4),
        "TARGET_COVERAGES": (1.0, 0.7, 0.4),
        "POSITIVE_RETENTIONS": (1.0, 0.8, 0.6, 0.4, 0.2),
        "CORE_MODEL_ALLOWLIST": ("PU", "PU-MIRT", "PU-Joint"),
        "EXPECTED_INTERNAL_MODELS": (
            "PU",
            "PU-MIRT",
            "PU-Joint",
            "GenEML-authors-mask",
            "Inductive-PU-MC",
            "SAR-PU",
        ),
        "N_REPETITIONS": 2,
        "N_PANEL_SEEDS": 2,
        "EXPECTED_SOURCE_HASH": "b" * 64,
        "PROTOCOL_VERSION": builder().PROTOCOL_VERSION,
        "np": np,
        "pd": pd,
    }
    with pytest.raises(
        RuntimeError, match="missing declared V4 physical conditions"
    ):
        exec(validation, namespace)


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    (
        ({"source_hash": "c" * 64}, "export source_hash"),
        (
            {"protocol": {"protocol_version": "wrong"}},
            "export protocol_version",
        ),
    ),
)
def test_v4_results_only_rejects_wrong_source_or_protocol(
    manifest_update, message
):
    book = builder().notebook("BARseq_A1", "a" * 40, "b" * 64, "0915")
    validation = _cell(book, "def _validate_v4_export_factorial(")
    manifest = {
        "experiment": "measurement_degradation",
        "completed": True,
        "source_hash": "b" * 64,
        "protocol": {"protocol_version": builder().PROTOCOL_VERSION},
    }
    manifest.update(manifest_update)
    namespace = {
        "RESULTS_ONLY": True,
        "all_artifacts": {
            "BARseq A1": SimpleNamespace(manifest=manifest, tables={})
        },
        "EXPECTED_SOURCE_HASH": "b" * 64,
        "PROTOCOL_VERSION": builder().PROTOCOL_VERSION,
    }
    with pytest.raises(RuntimeError, match=message):
        exec(validation, namespace)


@pytest.mark.parametrize("name", NAMES)
def test_v4_has_only_requested_configurable_curve_and_funky_plots(name):
    book = builder().notebook(name, "a" * 40, "b" * 64, "0915")
    metrics = "('auprc', 'auroc', 'log_loss', 'brier', 'hidden_recall')"

    family_curves = _cell(book, "gene2wire_curve_paths = {}")
    assert family_curves.lstrip().startswith(f"PLOT_METRICS = {metrics}")
    assert "models=GENE2WIRE_FAMILY_MODELS" in family_curves
    assert "plot_v4_metric_grid(" in family_curves

    external_curves = _cell(book, "external_curve_paths = {}")
    assert external_curves.lstrip().startswith(f"PLOT_METRICS = {metrics}")
    assert "models=EXTERNAL_CURVE_MODELS" in external_curves
    assert "plot_v4_metric_grid(" in external_curves

    funky = _cell(book, "funky_figure_paths = {}")
    assert funky.lstrip().startswith(f"FUNKY_METRICS = {metrics}")
    assert "models=FUNKY_MODELS" in funky
    assert "'gene_target_coverage'" not in funky
    assert "'combined_masks'" not in funky
    assert "prepare_v4_funky_table(" in funky
    assert "normalize_v4_funky_table(scorecard)" in funky
    assert "plot_v4_funky_heatmap(" in funky
    assert "funkyheatmap_scorecard_values_v4" in funky

    for source in (family_curves, external_curves):
        for family in (
            "positive_label_loss",
            "gene_coverage",
            "target_coverage",
            "combined_masks",
        ):
            assert repr(family) in source

    code = _source(book, "code")
    for forbidden in (
        "plot_gene_target_brier_heatmap(",
        "plot_retention_auprc(",
        "plot_coverage_auprc(",
        "heatmap_figure_paths = {}",
        "heatmap_baseline_selection",
        "heatmap_contrasts",
        "for mode in ('key', 'full')",
        "for mode in (\"key\", \"full\")",
    ):
        assert forbidden not in code


def test_v4_embeds_native_funkyheatmappy_and_public_plot_aliases():
    book = builder().notebook("simulation", "a" * 40, "b" * 64, "0915")
    helper = _cell(book, "def prepare_v4_curve_table(")
    assert "def plot_v4_metric_grid(" in helper
    assert "def prepare_v4_funky_table(" in helper
    assert "def plot_v4_funky_heatmap(" in helper
    assert "import funkyheatmappy" in helper
    assert '"geom": "funkyrect"' in helper
    assert '"metric_blue"' in helper
    assert '"PU-Joint": "Gene2Wire"' in helper
    assert '"Inductive-PU-MC": "PU matrix completion"' in helper
    assert "Path(__file__).read_text" not in helper


@pytest.mark.parametrize(
    ("name", "loader"),
    (
        ("BARseq_A1", "load_barseq"),
        ("BARseq_M1", "load_barseq"),
        ("MERGE_seq", "load_merge_seq_overlap"),
        ("Projection_TAGs", "load_projection_tags"),
        ("SPIDER", "load_spider"),
        ("SPIDER_Seq", "load_spider_seq_measurement"),
    ),
)
def test_v4_real_notebooks_keep_the_shared_dataset_adapters(name, loader):
    code = _source(
        builder().notebook(name, "a" * 40, "b" * 64, "0915"), "code"
    )
    assert f"from gene2wire.experiments.datasets" in code
    assert f"{loader}(" in code
    assert "preflight_measurement(dataset)" in code
    assert "run_measurement_experiment(" in code


def test_v4_projection_does_not_claim_the_disabled_natural_recovery_condition():
    book = builder().notebook(
        "Projection_TAGs", "a" * 40, "b" * 64, "0915"
    )
    markdown = _source(book, "markdown")
    assert "no separate natural amplification" in markdown
    assert "separate natural condition ranks" not in markdown


def test_v4_simulation_keeps_the_shared_truth_grid_and_public_runner():
    code = _source(
        builder().notebook("simulation", "a" * 40, "b" * 64, "0915"),
        "code",
    )
    assert "SHARING_STRENGTHS = (0.0, 0.5, 1.0)" in code
    assert "'n_cells': 400" in code
    assert "'n_targets': 36" in code
    assert "'n_gene_features': 16" in code
    assert "run_measurement_simulation_experiments(" in code


def test_checked_in_v4_notebooks_match_their_builder_metadata_byte_for_byte():
    module = builder()
    assert {path.name for path in V4_DIR.glob("*.ipynb")} == set(
        NOTEBOOK_FILENAMES.values()
    )
    for name, filename in NOTEBOOK_FILENAMES.items():
        destination = V4_DIR / filename
        checked_in = json.loads(destination.read_text(encoding="utf-8"))
        provenance = checked_in["metadata"]["gene2wire"]
        expected = module.notebook(
            name,
            provenance["core_commit"],
            provenance["source_hash"],
            provenance["notebook_date"],
        )
        assert checked_in == expected
