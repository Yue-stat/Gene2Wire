import ast
import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("barseq", "projection_tags", "spider", "simulation")


def _builder():
    path = ROOT / "scripts" / "build_gene_overlap_notebooks.py"
    spec = importlib.util.spec_from_file_location("_overlap_notebook_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlap_notebooks_are_clean_pinned_and_editable():
    commits = set()
    source_hashes = set()
    for name in NAMES:
        path = ROOT / f"gene2wire_{name}_overlap_grid.ipynb"
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        ids = [cell["id"] for cell in notebook["cells"]]
        assert len(ids) == len(set(ids))
        source = "\n".join(cell["source"] for cell in notebook["cells"])
        assert "OVERLAP_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]" in source
        assert "N_REPETITIONS = 10" in source
        assert "RUN_RANDOM_FOREST = False" in source
        assert "RF is disabled and is never scheduled" in source
        commit = re.search(r"^CORE_COMMIT = '([0-9a-f]{40})'$", source, re.MULTILINE)
        source_hash = re.search(
            r"^EXPECTED_SOURCE_HASH = '([0-9a-f]{64})'$", source, re.MULTILINE)
        assert commit is not None and source_hash is not None
        commits.add(commit.group(1))
        source_hashes.add(source_hash.group(1))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                ast.parse(cell["source"])
    assert len(commits) == 1
    assert len(source_hashes) == 1


def test_v2_builder_exposes_scientific_switches_and_explains_same_k_endpoint():
    builder = _builder()
    for name in NAMES:
        notebook = builder.notebook(name, "a" * 40, "b" * 64)
        source = "\n".join(cell["source"] for cell in notebook["cells"])
        markdown = "\n".join(cell["source"] for cell in notebook["cells"]
                             if cell["cell_type"] == "markdown")
        assert "INCLUDE_SHARED_A_ABLATION = True" in source
        assert "RUN_INDEPENDENT_PANEL_SENSITIVITY" not in source
        assert "include_independent_panel_sensitivity" not in source
        assert "NUISANCE_L2 = 1e-4" in source
        assert "nuisance_l2=NUISANCE_L2" in source
        assert "require_release_v2=True" in source
        assert "expected_overlap_grid=OVERLAP_GRID" in source
        assert "include_shared_a_ablation=INCLUDE_SHARED_A_ABLATION" in source
        assert "available_genes_per_cell" in source
        assert "same_K_100pct_overlap" in source
        assert "all_P_gene_oracle" in source
        assert "At 100% overlap" in markdown
        assert "same K genes" in markdown
        assert "all-`P`-gene" in markdown
        assert "fully independent panel calibrations/fits are deliberately" in markdown
        assert "if not paths:" in source
        assert "No gene-overlap figures were produced" in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])

        if name == "simulation":
            assert "RUN_TECH_SAR_80_SENSITIVITY = False" in source
            assert "TECH_SAR_80_EXPORT_DIR = None" in source
            assert "prepare_tech_sar80_sensitivity" in source
            assert "tech_sar80_plan.settings" in source
            assert "load_export_artifacts(TECH_SAR_80_EXPORT_DIR)" in source
        else:
            assert "RUN_TECH_SAR_80_SENSITIVITY" not in source
            assert "prepare_tech_sar80_sensitivity" not in source


def test_projection_builder_balances_platform_across_the_paired_panels():
    notebook = _builder().notebook("projection_tags", "a" * 40, "b" * 64)
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "mapping = {1: 'A', 4: 'B', 2: 'B', 6: 'A', 3: 'A', 5: 'B'}" in source
    assert "Animal-level platform × assigned gene-panel audit" in source
    assert "panel assignment is confounded with platform" in source
