import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("barseq", "projection_tags", "spider", "simulation")


def test_overlap_notebooks_are_clean_pinned_and_editable():
    for name in NAMES:
        path = ROOT / f"gene2wire_{name}_overlap_grid.ipynb"
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        ids = [cell["id"] for cell in notebook["cells"]]
        assert len(ids) == len(set(ids))
        source = "\n".join(cell["source"] for cell in notebook["cells"])
        assert "OVERLAP_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]" in source
        assert "RUN_RANDOM_FOREST = False" in source
        assert "RF is disabled and is never scheduled" in source
        assert "CORE_COMMIT = '5d1d5fefa73ac45b91594c8634045404c5e30d33'" in source
        assert "EXPECTED_SOURCE_HASH = '1301b7c19b61d5860ba997f2b7b0701d8096fb6067cc19902afa9a8d0b6c18d5'" in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                ast.parse(cell["source"])
