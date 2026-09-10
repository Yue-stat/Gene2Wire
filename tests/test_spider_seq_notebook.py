import importlib.util
from pathlib import Path

import nbformat


def test_native_notebook_is_a_shared_core_entrypoint_with_truthful_defaults():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('seq_builder', root/'scripts/build_spider_seq_notebook.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    book = nbformat.from_dict(builder.notebook('a'*40, 'b'*64, '0910'))
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
    for index, cell in enumerate(book.cells):
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook>', 'exec')
            assert not cell.outputs
            assert index and book.cells[index-1].cell_type == 'markdown'
            assert '## ' in book.cells[index-1].source
    assert "__main__" not in code
