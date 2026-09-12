"""Repository-wide contracts for canonical notebook organization and provenance."""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import tarfile

import pytest

from gene2wire.experiments.protocol import PROTOCOL_VERSION, source_hash


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
EXPECTED_NOTEBOOKS = {
    "positive_label_hiding/barseq.ipynb",
    "positive_label_hiding/merge_seq.ipynb",
    "positive_label_hiding/projection_tags.ipynb",
    "positive_label_hiding/spider_spatial.ipynb",
    "positive_label_hiding/simulation.ipynb",
    "target_block_masking/barseq_a1.ipynb",
    "target_block_masking/barseq_m1.ipynb",
    "target_block_masking/merge_seq.ipynb",
    "target_block_masking/projection_tags.ipynb",
    "target_block_masking/spider_spatial.ipynb",
    "target_block_masking/spider_seq.ipynb",
    "target_block_masking/simulation.ipynb",
    "gene_panel_overlap/barseq.ipynb",
    "gene_panel_overlap/merge_seq.ipynb",
    "gene_panel_overlap/projection_tags.ipynb",
    "gene_panel_overlap/spider_spatial.ipynb",
    "gene_panel_overlap/simulation.ipynb",
    "native_target_panels/spider_seq.ipynb",
    "measurement_degradation/simulation.ipynb",
    "measurement_degradation/barseq_a1.ipynb",
    "measurement_degradation/barseq_m1.ipynb",
    "measurement_degradation/merge_seq.ipynb",
    "measurement_degradation/projection_tags.ipynb",
    "measurement_degradation/spider_spatial.ipynb",
    "measurement_degradation/spider_seq.ipynb",
}


def _canonical_paths():
    return sorted(NOTEBOOK_ROOT.glob("*/*.ipynb"))


def test_only_expected_canonical_notebooks_use_the_stable_layout():
    relative = {path.relative_to(NOTEBOOK_ROOT).as_posix()
                for path in _canonical_paths()}
    assert relative == EXPECTED_NOTEBOOKS
    assert not list(ROOT.glob("*.ipynb"))


def test_canonical_notebooks_are_clean_and_share_current_core_provenance():
    commits = set()
    hashes = set()
    for path in _canonical_paths():
        notebook = json.loads(path.read_text())
        provenance = notebook["metadata"]["gene2wire"]
        assert re.fullmatch(r"[0-9a-f]{40}", provenance["core_commit"])
        assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_hash"])
        assert re.fullmatch(r"[0-9]{4}", provenance["notebook_date"])
        assert provenance["protocol"] == PROTOCOL_VERSION

        code = "\n".join(
            cell["source"] for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert f"CORE_COMMIT = '{provenance['core_commit']}'" in code
        assert f"EXPECTED_SOURCE_HASH = '{provenance['source_hash']}'" in code
        assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []

        commits.add(provenance["core_commit"])
        hashes.add(provenance["source_hash"])

    assert len(commits) == 1
    assert hashes == {source_hash()}


def test_pinned_commit_contains_the_declared_source_tree_when_available():
    notebook = json.loads(_canonical_paths()[0].read_text())
    provenance = notebook["metadata"]["gene2wire"]
    commit = provenance["core_commit"]
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if available.returncode:
        pytest.skip("pinned ancestor is unavailable in this shallow checkout")

    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", commit, "src/gene2wire"],
        cwd=ROOT,
    )
    digest = hashlib.sha256()
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as tree:
        sources = sorted(
            [member for member in tree.getmembers()
             if member.isfile() and member.name.endswith(".py")],
            key=lambda member: member.name,
        )
        for member in sources:
            relative = Path(member.name).relative_to("src/gene2wire").as_posix()
            digest.update(relative.encode())
            digest.update(tree.extractfile(member).read())
    assert digest.hexdigest() == provenance["source_hash"]
