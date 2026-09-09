"""Contract checks for the five executable, cached OnDemand entry points."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("simulation", "SPIDER", "MERGE_seq", "Projection_TAGs", "BARseq")


def load(name):
    return json.loads((ROOT / f"{name}_0908.ipynb").read_text())


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


@pytest.mark.parametrize("name", NAMES)
def test_clean_valid_python_and_shared_defaults(name):
    nb = load(name)
    assert nb["nbformat"] == 4
    assert len({cell["id"] for cell in nb["cells"]}) == len(nb["cells"])
    assignments = {}
    all_code = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = text(cell)
        parsed = ast.parse(source)
        all_code.append(source)
        for node in parsed.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value.value
    for key, expected in {
        "N_OUTER_FOLDS": 3, "USE_LOCATION": False, "USE_TARGET_FEATURES": False,
        "N_JOBS": 32, "N_REPETITIONS": 5, "STRATEGY": "full_joint",
        "RUN_INFORMATION_CONTROLS": True, "RUN_RANDOM_FOREST": True,
        "RUN_MECHANISM_CONTROLS": True, "RUN_CALIBRATION_CONTROLS": True,
        "RUN_QIAO": True,
    }.items():
        assert assignments[key] == expected
    joined = "\n".join(all_code)
    assert "__main__" not in joined
    assert "google.colab" not in joined
    assert "pip install" not in joined
    assert "paper_figure_exports" in joined
    assert "raw_data" in joined and "checkpoints" in joined
    assert "plot_results(artifacts, output_dir=FIGURE_DIR)" in joined


def test_location_switch_controls_simulation_truth_and_fitted_features():
    code = "\n".join(text(cell) for cell in load("simulation")["cells"] if cell["cell_type"] == "code")
    parsed = ast.parse(code)
    options = next(node.value for node in ast.walk(parsed) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "SIMULATION_OPTIONS" for t in node.targets))
    mapping = dict(zip((key.value for key in options.keys), options.values))
    assert isinstance(mapping["truth_uses_location"], ast.Name)
    assert mapping["truth_uses_location"].id == "USE_LOCATION"
    assert "use_location=USE_LOCATION" in code
    assert "SHARING_STRENGTHS = (0.0, 0.5, 1.0)" in code


def builder():
    spec = importlib.util.spec_from_file_location("notebook_diagnostics_builder", ROOT / "scripts" / "build_notebooks_0908.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_generated_results_only_skips_all_raw_and_fit_cells(name):
    module = builder()
    nb = module.notebook(name, "a" * 40, "b" * 64)
    guarded = [text(cell) for cell in nb["cells"]
               if cell["cell_type"] == "code" and text(cell).startswith("if not RESULTS_ONLY:")]
    assert len(guarded) >= 2
    namespace = {"RESULTS_ONLY": True}
    for source in guarded:
        # No loader, Settings, paths or fit names exist: execution must be a no-op.
        exec(compile(source, "results-only-cell", "exec"), namespace)
    assert "all_artifacts" not in namespace
    assert "dataset" not in namespace and "datasets" not in namespace
    all_code = "\n".join(text(cell) for cell in nb["cells"] if cell["cell_type"] == "code")
    assert "RESULTS_ONLY = False" in all_code
    assert "load_existing_exports(" in all_code
    assert "configure_full_display()" in all_code
    assert "display_diagnostics(artifacts, label=label, full=SHOW_FULL_DIAGNOSTICS)" in all_code
    assert "SHOW_FULL_DIAGNOSTICS = False" in all_code
    assert "plot_information_budget_results(" in all_code
    assert "RUN_QIAO = True" in all_code
    assert "enabled in simulation only" not in all_code
    assert "plot_results(artifacts, output_dir=FIGURE_DIR)" in all_code
    assert "plot_benchmark_results(artifacts, output_dir=FIGURE_DIR, show=True)" in all_code
    assert "SHOW_PROGRESS = True" in all_code
    assert "PROGRESS_LEVEL = 'summary'" in all_code
    assert "PROGRESS_INTERVAL_SECONDS = 60.0" in all_code
    run_calls = [node for node in ast.walk(ast.parse(all_code))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id in {"run_experiment", "run_simulation_experiments"}]
    assert run_calls
    for call in run_calls:
        values = {keyword.arg: keyword.value.id for keyword in call.keywords
                  if isinstance(keyword.value, ast.Name)}
        assert values["progress"] == "SHOW_PROGRESS"
        assert values["progress_level"] == "PROGRESS_LEVEL"
        assert values["progress_interval"] == "PROGRESS_INTERVAL_SECONDS"
    if name == "BARseq":
        assert "EXISTING_EXPORT_DIRS = {'A1': None, 'M1': None}" in all_code


def test_generator_reproduces_checked_in_notebooks(tmp_path):
    path = ROOT / "scripts" / "build_notebooks_0908.py"
    spec = importlib.util.spec_from_file_location("notebook_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = load("simulation")["metadata"]["gene2wire"]
    generated = module.build(manifest["core_commit"], manifest["source_hash"], tmp_path)
    for output in generated:
        assert output.read_bytes() == (ROOT / output.name).read_bytes()


def test_bootstrap_reuses_verified_local_sources_without_network(tmp_path):
    cells = load("simulation")["cells"]
    config = next(text(cell) for cell in cells if cell["cell_type"] == "code" and "N_OUTER_FOLDS =" in text(cell))
    bootstrap = next(text(cell) for cell in cells if cell["cell_type"] == "code" and "def verify_checkout" in text(cell))
    setup = f'''
BASE_DIR = Path({str(tmp_path)!r})
RAW_DATA_DIR = BASE_DIR / 'raw_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / '0908'
EXPORT_DIR = BASE_DIR / 'paper_figure_exports'
FIGURE_DIR = BASE_DIR / 'figures' / '0908'
CODE_CACHE_DIR = BASE_DIR / 'code'
CORE_COMMIT = '0' * 40
REQUIRED_MODULES = ['numpy', 'scipy', 'yaml']  # Bootstrap-only test; no display cells run.
import hashlib
package_root = Path.cwd() / 'src' / 'gene2wire'
digest = hashlib.sha256()
for source_path in sorted(package_root.rglob('*.py')):
    digest.update(source_path.relative_to(package_root).as_posix().encode())
    digest.update(source_path.read_bytes())
EXPECTED_SOURCE_HASH = digest.hexdigest()
import subprocess
def forbidden_network(*args, **kwargs):
    raise AssertionError('Verified local bootstrap tried to invoke git/network')
subprocess.run = forbidden_network
subprocess.check_output = forbidden_network
'''
    result = subprocess.run([sys.executable, "-c", config + "\n" + setup + "\n" + bootstrap],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert str(ROOT / "src" / "gene2wire") in result.stdout
    assert (tmp_path / "paper_figure_exports").is_dir()
    assert not (tmp_path / "code").exists()
