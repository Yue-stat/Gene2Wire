import importlib.util
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "notebooks" / "build_native_target_panels.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "native_target_panels" / "spider_seq.ipynb"


def _builder():
    spec = importlib.util.spec_from_file_location("seq_builder", BUILDER_PATH)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def test_native_notebook_is_a_shared_core_entrypoint_with_truthful_defaults():
    builder = _builder()
    book = nbformat.from_dict(builder.notebook('a'*40, 'b'*64, '0911'))
    nbformat.validate(book)
    code = '\n'.join(c.source for c in book.cells if c.cell_type == 'code')
    assert 'N_OUTER_FOLDS = 3' in code and 'N_REPETITIONS = 5' in code
    assert 'N_JOBS = 32' in code and 'SHOW_FULL_DIAGNOSTICS = False' in code
    assert "supervision_profile='assay_only', paired_fraction=0." in code
    assert 'run_native_panel_experiment(' in code and 'load_spider_seq(' in code
    assert 'make_block_design(' not in code and 'run_block_experiment(' not in code
    assert 'RUN_INFORMATION_CONTROLS = True' not in code
    assert 'N_HVG = 2000' in code and 'N_GENE_COMPONENTS = 50' in code
    assert 'SPIDER-Seq' in code and '/home/yueyue/gene2wire' in code
    assert 'GENE2WIRE_PROJECT_DIR' in code
    for index, cell in enumerate(book.cells):
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook>', 'exec')
            assert not cell.outputs
            assert index and book.cells[index-1].cell_type == 'markdown'
            assert '## ' in book.cells[index-1].source
    assert "__main__" not in code


def test_native_builder_uses_stable_checked_in_path(tmp_path):
    builder = _builder()
    generated = builder.build("a" * 40, "b" * 64, tmp_path, "0911")
    assert generated.name == "spider_seq.ipynb"
    assert NOTEBOOK_PATH.is_file()
    released = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(released)
    assert all(not cell.get("outputs") for cell in released.cells
               if cell.cell_type == "code")
