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
        assert "N_REPETITIONS = 10" in source
        assert "RUN_RANDOM_FOREST = False" in source
        assert "RF is disabled and is never scheduled" in source
        assert "CORE_COMMIT = 'e0ffa16876c158523bf08e4f4bbe6542c4cfc07f'" in source
        assert "EXPECTED_SOURCE_HASH = '95b5a806452e966f5464850e9db83652fc3097b8061022a473137e2cb910b0e3'" in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                ast.parse(cell["source"])
