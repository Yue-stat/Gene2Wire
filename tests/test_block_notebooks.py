"""Executable entry-point contracts for independent target-panel experiments."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("BARseq_A1", "BARseq_M1", "Projection_TAGs", "simulation", "SPIDER")


def builder():
    spec = importlib.util.spec_from_file_location(
        "block_notebook_generator", ROOT / "scripts" / "build_block_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_clean_executable_cells_have_navigation_and_replot_skips_fits(name):
    nb = builder().notebook(name, "a" * 40, "b" * 64, "0909")
    code = []
    guarded = []
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert nb["cells"][index - 1]["cell_type"] == "markdown"
        assert nb["cells"][index - 1]["source"].startswith("## ")
        assert cell["execution_count"] is None and cell["outputs"] == []
        ast.parse(cell["source"])
        code.append(cell["source"])
        if cell["source"].startswith("if not RESULTS_ONLY:"):
            guarded.append(cell["source"])
            # Loading and fitting must not require ANY other variable when replotting.
            exec(compile(cell["source"], "results-only", "exec"), {"RESULTS_ONLY": True})
    assert len(guarded) >= 2
    joined = "\n".join(code)
    assert "plot_block_results(" in joined
    assert "plot_results(" not in joined
    assert "__main__" not in joined
    assert ".png" not in joined
    assert "paper_figure_exports" in joined
    assert "preview_block_experiment(" in joined


@pytest.mark.parametrize("name", NAMES)
def test_shared_settings_and_correct_group_source(name):
    nb = builder().notebook(name, "a" * 40, "b" * 64, "0909")
    config = nb["cells"][2]["source"]
    namespace = {}
    exec(config, namespace)
    for key, expected in {
        "N_OUTER_FOLDS": 3, "N_JOBS": 32, "N_REPETITIONS": 5,
        "USE_LOCATION": False, "USE_TARGET_FEATURES": False,
        "STRATEGY": "full_joint", "PAIRED_FRACTION": .2,
        "BLOCK_FRACTIONS": (.2, .4, .6, .8), "POSITIVE_LOSS_RATES": (0.,),
        "INCLUDE_FULL_PANEL_CONTROL": True, "SHOW_FULL_DIAGNOSTICS": False,
        "RUN_INFORMATION_CONTROLS": True, "RUN_RANDOM_FOREST": True, "RUN_QIAO": True,
    }.items():
        assert namespace[key] == expected
    assert namespace["GROUP_MODE"] == (
        "artificial" if name in {"BARseq_M1", "simulation", "SPIDER"} else "animal")


def test_location_switch_reaches_simulation_truth_and_settings():
    nb = builder().notebook("simulation", "a" * 40, "b" * 64, "0909")
    code = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "'truth_uses_location': USE_LOCATION" in code
    assert "use_location=USE_LOCATION" in code
    assert "SHARING_STRENGTHS = (0.0, 0.5, 1.0)" in code


def test_block_notebook_uses_public_execution_api():
    # Catch builder/import drift before publishing source-pinned entry points.
    from gene2wire.experiments import block_experiment, block_plotting
    for name in ("run_block_experiment", "run_block_simulation_experiments", "preview_block_experiment"):
        assert callable(getattr(block_experiment, name))
    assert callable(block_plotting.plot_block_results)


def test_spider_reuses_shared_loader_and_never_infers_animal_ids():
    nb = builder().notebook("SPIDER", "a" * 40, "b" * 64, "0909")
    code = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    prose = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "from gene2wire.experiments.datasets.spider import load_spider" in code
    assert "target_features_csv=TARGET_FEATURES_CSV" in code
    assert "group_mode=GROUP_MODE" in code
    assert "no verified animal IDs" in prose
    assert "not relabelled as animals" in prose


@pytest.mark.parametrize("name", NAMES)
def test_block_report_calls_shared_bounded_report_instead_of_dumping_raw_tables(name):
    nb = builder().notebook(name, "a" * 40, "b" * 64, "0909")
    final = nb["cells"][-1]["source"]
    assert "display_diagnostics(artifacts, label=label, full=SHOW_FULL_DIAGNOSTICS)" in final
    assert "for name in names" not in final
