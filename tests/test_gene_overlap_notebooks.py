import ast
import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "gene_panel_overlap"
NAMES = ("barseq", "projection_tags", "spider", "merge_seq", "simulation")
CLEAN_NOTEBOOK_NAMES = NAMES
NOTEBOOK_FILENAMES = {
    "barseq": "barseq.ipynb",
    "projection_tags": "projection_tags.ipynb",
    "spider": "spider_spatial.ipynb",
    "merge_seq": "merge_seq.ipynb",
    "simulation": "simulation.ipynb",
}


def _builder():
    path = ROOT / "scripts" / "notebooks" / "build_gene_panel_overlap.py"
    spec = importlib.util.spec_from_file_location("_overlap_notebook_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overlap_notebooks_are_clean_pinned_and_editable():
    assert _builder().NOTEBOOK_FILENAMES == NOTEBOOK_FILENAMES
    commits = set()
    source_hashes = set()
    for name in CLEAN_NOTEBOOK_NAMES:
        path = NOTEBOOK_DIR / NOTEBOOK_FILENAMES[name]
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


def test_merge_overlap_builder_uses_leak_isolated_crossed_design_without_sample_nuisance():
    notebook = _builder().notebook("merge_seq", "a" * 40, "b" * 64)
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    merge_source = "\n".join(
        cell["source"] for cell in notebook["cells"]
        if "load_merge_seq_overlap" in cell["source"]
    )
    markdown = "\n".join(cell["source"] for cell in notebook["cells"]
                         if cell["cell_type"] == "markdown")
    assert "from gene2wire.experiments.datasets.merge_seq import load_merge_seq_overlap" in merge_source
    assert "source_gene_count=MERGE_SOURCE_GENE_COUNT" in merge_source
    assert "panel_design_fraction=MERGE_PANEL_DESIGN_FRACTION" in merge_source
    assert "strata=('sample', 'assay_status')" in merge_source
    assert "nuisance_groups=()" in merge_source
    assert "common_target_dataset" not in merge_source
    assert "measured.shape[1] != 5" in merge_source
    assert "positive_support" in merge_source
    assert "MERGE_seq_gene_overlap_0910" in source
    assert "all fixed 128 candidate genes" in markdown
    assert "reserved `W=0` design cells" in markdown
    assert "whole transcriptome" in markdown
    assert "excludes every `barcode*`" in markdown
    assert "post-normalization availability experiment" in markdown
    assert "only five common projection targets" in markdown
    assert "whole-sample outer folds" in markdown
    assert "reserved_design_cells_removed" in merge_source
    assert "overlap_expression_normalization" in merge_source


def test_projection_builder_balances_platform_across_the_paired_panels():
    notebook = _builder().notebook("projection_tags", "a" * 40, "b" * 64)
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "mapping = {1: 'A', 4: 'B', 2: 'B', 6: 'A', 3: 'A', 5: 'B'}" in source
    assert "Animal-level platform × assigned gene-panel audit" in source
    assert "panel assignment is confounded with platform" in source
