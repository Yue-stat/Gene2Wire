"""Public release and checkpoint-schema version contracts."""
from importlib.metadata import version
import re

import gene2wire


def test_distribution_and_imported_package_versions_match():
    assert gene2wire.__version__ == version("gene2wire")


def test_core_api_version_is_an_independent_semantic_version():
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", gene2wire.CORE_API_VERSION)
