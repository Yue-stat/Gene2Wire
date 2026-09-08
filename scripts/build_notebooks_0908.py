"""Rebuild clean OnDemand entry points for the shared 0908 experiment protocol.

Usage: python scripts/build_notebooks_0908.py --commit <SHA> --source-hash <SHA256>
The generator itself has no model or dataset implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_NAMES = ("simulation", "SPIDER", "MERGE_seq", "Projection_TAGs", "BARseq")


def cell(kind, source, index):
    result = {"cell_type": kind, "id": f"g2w0908-{index:03d}", "metadata": {},
              "source": textwrap.dedent(source).strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


CONFIG = '''
from pathlib import Path
import os

# All notebooks share these defaults. Change switches here before Run All.
N_OUTER_FOLDS = 3
USE_LOCATION = False
USE_TARGET_FEATURES = False
N_JOBS = 32
N_REPETITIONS = 5
STRATEGY = 'full_joint'
SEED = 20260908

RUN_INFORMATION_CONTROLS = True
RUN_RANDOM_FOREST = True
RUN_MECHANISM_CONTROLS = True
RUN_CALIBRATION_CONTROLS = True
RUN_QIAO = False

BASE_DIR = Path('/home/yueyue/gene2wire').expanduser()
RAW_DATA_DIR = BASE_DIR / 'raw_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / '0908'
EXPORT_DIR = BASE_DIR / 'paper_figure_exports'
FIGURE_DIR = BASE_DIR / 'figures' / '0908'
CODE_CACHE_DIR = BASE_DIR / 'code'

CORE_COMMIT = '__CORE_COMMIT_0908__'
EXPECTED_SOURCE_HASH = '__SOURCE_HASH_0908__'
REPO_URL = 'https://github.com/Yue-stat/Gene2Wire.git'

# Set before importing NumPy/SciPy. Parallelism is at fold/repetition level.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
'''


BOOTSTRAP = '''
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile

if sys.version_info < (3, 10):
    raise RuntimeError('Select an OnDemand Python 3.10 or newer kernel.')
if not re.fullmatch(r'[0-9a-f]{40}', CORE_COMMIT):
    raise RuntimeError('This notebook needs its released 40-character CORE_COMMIT pin.')
if not re.fullmatch(r'[0-9a-f]{64}', EXPECTED_SOURCE_HASH):
    raise RuntimeError('This notebook needs its released source checksum.')

def notebook_source_hash(package_root):
    # Same byte-level convention as experiments.protocol.source_hash().
    digest = hashlib.sha256()
    for source_path in sorted(package_root.rglob('*.py')):
        digest.update(source_path.relative_to(package_root).as_posix().encode())
        digest.update(source_path.read_bytes())
    return digest.hexdigest()

def verify_checkout(checkout, require_git_pin=True):
    package_root = checkout / 'src' / 'gene2wire'
    if not package_root.is_dir():
        raise RuntimeError(f'Missing Gene2Wire sources in {checkout}')
    if require_git_pin:
        actual_commit = subprocess.check_output(
            ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if actual_commit != CORE_COMMIT:
            raise RuntimeError(f'Cached code has commit {actual_commit}, expected {CORE_COMMIT}.')
    if notebook_source_hash(package_root) != EXPECTED_SOURCE_HASH:
        raise RuntimeError(f'Source checksum mismatch in {checkout}; use the released code.')
    return checkout

# Running from the exact local repository is supported without any network call.
CORE_CHECKOUT = None
for candidate in (Path.cwd(), *Path.cwd().parents):
    package_root = candidate / 'src' / 'gene2wire'
    if package_root.is_dir() and notebook_source_hash(package_root) == EXPECTED_SOURCE_HASH:
        CORE_CHECKOUT = verify_checkout(candidate, require_git_pin=False)
        break

if CORE_CHECKOUT is None:
    CODE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_checkout = CODE_CACHE_DIR / CORE_COMMIT
    if not cached_checkout.exists():
        stage = Path(tempfile.mkdtemp(prefix='.gene2wire-download-', dir=CODE_CACHE_DIR))
        try:
            for arguments in (
                ['git', 'init', '--quiet', str(stage)],
                ['git', '-C', str(stage), 'remote', 'add', 'origin', REPO_URL],
                ['git', '-C', str(stage), 'fetch', '--quiet', '--depth', '1', 'origin', CORE_COMMIT],
                ['git', '-C', str(stage), 'checkout', '--quiet', '--detach', 'FETCH_HEAD'],
            ):
                subprocess.run(arguments, check=True)
            verify_checkout(stage)
            try:
                stage.rename(cached_checkout)
            except OSError:
                # Another notebook may have completed this same immutable cache.
                if not cached_checkout.exists():
                    raise
                verify_checkout(cached_checkout)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    CORE_CHECKOUT = verify_checkout(cached_checkout)

existing = sys.modules.get('gene2wire')
if existing is not None:
    same_path = Path(existing.__file__).resolve().parent == (CORE_CHECKOUT / 'src' / 'gene2wire').resolve()
    same_source = getattr(existing, '_notebook_source_hash_0908', None) == EXPECTED_SOURCE_HASH
    if not (same_path and same_source):
        raise RuntimeError('A different or unverified gene2wire is already imported. Restart the kernel, then Run All.')

missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError('Use an OnDemand Python kernel containing these dependencies: '
                       + ', '.join(missing) + '. See the repository environment instructions.')
sys.path.insert(0, str(CORE_CHECKOUT / 'src'))
import gene2wire
from gene2wire.experiments.protocol import Settings, source_hash
if source_hash() != EXPECTED_SOURCE_HASH:
    raise RuntimeError('Imported code does not match the released source checksum.')
gene2wire._notebook_source_hash_0908 = EXPECTED_SOURCE_HASH
for directory in (RAW_DATA_DIR, CHECKPOINT_DIR, EXPORT_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print({'core_commit': CORE_COMMIT, 'source_hash': source_hash(),
       'imported_from': gene2wire.__file__, 'raw_cache': str(RAW_DATA_DIR),
       'checkpoints': str(CHECKPOINT_DIR), 'exports': str(EXPORT_DIR)})
'''


SETTINGS = '''
from dataclasses import asdict
import numpy as np
import pandas as pd
from IPython.display import display
from gene2wire.experiments.pipeline import run_experiment, run_simulation_experiments
from gene2wire.experiments.plotting import plot_results
from gene2wire.tuning import full_joint_candidates

settings = Settings(
    n_outer_folds=N_OUTER_FOLDS, use_location=USE_LOCATION,
    use_target_features=USE_TARGET_FEATURES, n_jobs=N_JOBS,
    n_repetitions=N_REPETITIONS, strategy=STRATEGY, seed=SEED,
    run_information_controls=RUN_INFORMATION_CONTROLS,
    run_random_forest=RUN_RANDOM_FOREST,
    run_mechanism_controls=RUN_MECHANISM_CONTROLS,
    run_calibration_controls=RUN_CALIBRATION_CONTROLS,
    run_qiao=RUN_QIAO,
)
display(pd.DataFrame([asdict(settings)]).T.rename(columns={0: 'setting'}))
'''


PREFLIGHT = '''
def preflight(dataset):
    dataset.validate()
    folds = tuple(dataset.split_builder(settings.n_outer_folds, settings.seed))
    for fold in folds:
        fold.validate(len(dataset.cell_ids))
    features = dataset.feature_builder(
        folds[0].train_rows, settings.use_location, settings.use_target_features)
    roles = []
    for fold in folds:
        roles.append({'dataset': dataset.name, 'outer_fold': fold.outer_fold,
                      'inner_train_cells': len(fold.train_rows),
                      'validation_cells': len(fold.validation_rows),
                      'test_cells': len(fold.test_rows),
                      'split_design': dict(fold.metadata)})
    scalar_metadata = {key: value for key, value in dataset.metadata.items()
                       if value is None or isinstance(value, (str, bool, int, float))}
    display(pd.DataFrame([{'dataset': dataset.name, 'cells': len(dataset.cell_ids),
                           'targets': len(dataset.target_ids),
                           'measured_pairs': int(dataset.measured.sum()),
                           'reference_positives': int(dataset.reference.sum()),
                           'feature_columns': features.X.shape[1],
                           'feature_blocks': dict(features.feature_blocks),
                           'target_feature_columns': 0 if features.Y_target is None else features.Y_target.shape[1],
                           'natural_paired_assay': dataset.natural_observed is not None}]))
    display(pd.DataFrame(roles))
    display(pd.DataFrame([scalar_metadata]).T.rename(columns={0: 'dataset metadata'}))
    tuning = settings.tuning_config(features.X.shape[1], len(dataset.target_ids))
    display(pd.DataFrame([
        {'model': model.name, 'strategy': settings.strategy,
         'maximum_trials': tuning.candidate_budget,
         'bounded_grid_candidates': len(full_joint_candidates(model, tuning)),
         'eligible_structures': sorted({c.kind for c in full_joint_candidates(model, tuning)})}
        for model in settings.models()
    ]))
    print('Features above are fitted on inner-training cells only. Final refit uses the development cells.')
    print('Repeated masks share a biological dataset; they are not additional independent animals.')
    return folds
'''


FINAL = '''
# Full tables and predictions are already exported by the shared pipeline.
# The notebook presents one wide metric summary; every detailed table remains on disk.
for label, artifacts in all_artifacts.items():
    print(label)
    candidate_names = ('aggregate', 'summary', 'summary_metrics', 'metrics_summary', 'aggregate_metrics', 'metrics')
    chosen = next((artifacts.tables[name] for name in candidate_names
                   if name in artifacts.tables), None)
    if chosen is None:
        chosen = next((value for value in artifacts.tables.values()
                       if isinstance(value, pd.DataFrame)), None)
    if chosen is not None:
        display(chosen)
    print('Full result export:', artifacts.export_dir)
    print('Saved table names:', list(artifacts.tables))
print('Figure PDFs:', FIGURE_DIR)
print('Reusable raw cache:', RAW_DATA_DIR)
print('Resumable checkpoints:', CHECKPOINT_DIR)
'''


def dataset_cells(name):
    if name == "simulation":
        return [
            ("markdown", """**Simulation configuration.** The three sharing strengths are run together.
            `USE_LOCATION=False` excludes location from both the fitted predictor and the generated
            projection signal. Setting it to `True` includes the declared location basis in both.
            Generated raw arrays and truths are saved for reproducibility; truth remains outside fitting.
            Each repetition generates an independent dataset; folds are aggregated within repetitions."""),
            ("code", '''
            from gene2wire.experiments.datasets.simulation import generate_simulation
            SHARING_STRENGTHS = (0.0, 0.5, 1.0)
            SIMULATION_OPTIONS = {
                'n_cells': 400, 'n_targets': 36, 'n_gene_features': 16,
                'n_location_features': 4, 'n_slices': 20, 'true_rank': 2,
                'n_target_features': 4, 'signal_sd': 1.35,
                'target_feature_noise': 0.85,
                'target_prevalence_range': (0.10, 0.30),
                'truth_uses_location': USE_LOCATION,
            }
            representative = generate_simulation(
                repetition=0, sharing_strength=SHARING_STRENGTHS[0], seed=SEED,
                **SIMULATION_OPTIONS)
            preflight_folds = preflight(representative)
            '''),
            ("markdown", """**Run the frozen experiment matrix.** This cell is the expensive step.
            Existing compatible checkpoints resume automatically. Mechanism and calibration controls
            follow the shared protocol; they do not multiply every dataset/rate/model combination."""),
            ("code", '''
            artifacts = run_simulation_experiments(
                settings=settings, raw_cache_dir=RAW_DATA_DIR / 'simulation',
                checkpoint_dir=CHECKPOINT_DIR, export_dir=EXPORT_DIR,
                sharing_strengths=SHARING_STRENGTHS,
                simulation_options=SIMULATION_OPTIONS,
            )
            all_artifacts = {'simulation': artifacts}
            '''),
        ]
    if name == "BARseq":
        return [
            ("markdown", """**BARseq A1 and M1.** Both panels are fitted and reported separately.
            Location uses the native measured spatial covariates. Optional target features are
            anatomy descriptors derived from target names, not postsynaptic gene expression.
            Qiao comparisons are reserved for the simulation's declared target descriptors."""),
            ("code", '''
            from gene2wire.experiments.datasets.barseq import load_barseq
            if RUN_QIAO:
                raise ValueError('The Qiao comparison is enabled in simulation only; use RUN_QIAO=False here.')
            datasets = load_barseq(RAW_DATA_DIR / 'BARseq')
            PANELS = ('A1', 'M1')
            for panel in PANELS:
                preflight(datasets[panel])
            '''),
            ("code", '''
            all_artifacts = {}
            for panel in PANELS:
                all_artifacts[panel] = run_experiment(
                    dataset=datasets[panel], settings=settings,
                    checkpoint_dir=CHECKPOINT_DIR, export_dir=EXPORT_DIR)
            '''),
        ]
    if name == "SPIDER":
        load = """
        from gene2wire.experiments.datasets.spider import load_spider
        dataset = load_spider(RAW_DATA_DIR / 'SPIDER', target_features_csv=TARGET_FEATURES_CSV)
        """
        options = "TARGET_FEATURES_CSV = None  # Optional target-ID-indexed numeric CSV."
        description = """Native location is available. If `USE_TARGET_FEATURES=True`, supply
        a CSV indexed by the exact target IDs; columns must be independently measured numeric
        descriptors. Missing descriptors trigger a clear error before fitting."""
    elif name == "MERGE_seq":
        load = """
        from gene2wire.experiments.datasets.merge_seq import load_merge_seq
        dataset = load_merge_seq(
            RAW_DATA_DIR / 'MERGE_seq', n_gene_features=N_GENE_FEATURES,
            location_features_csv=LOCATION_FEATURES_CSV,
            target_features_csv=TARGET_FEATURES_CSV)
        """
        options = """
        N_GENE_FEATURES = 128
        LOCATION_FEATURES_CSV = None  # Optional cell-ID-indexed numeric CSV.
        TARGET_FEATURES_CSV = None  # Optional target-ID-indexed numeric CSV.
        """
        description = """Gene selection is fitted on training cells. Native locations and
        postsynaptic target descriptors are absent from this adapter's raw input. Supply aligned,
        outcome-independent CSV features before enabling the corresponding switch.
        Thinning scales are estimated from inner-training reference positives only."""
    else:
        load = """
        from gene2wire.experiments.datasets.projection_tags import load_projection_tags
        dataset = load_projection_tags(
            RAW_DATA_DIR / 'Projection_TAGs',
            location_features_csv=LOCATION_FEATURES_CSV,
            target_features_csv=TARGET_FEATURES_CSV)
        """
        options = """
        LOCATION_FEATURES_CSV = None  # Optional cell-ID-indexed numeric CSV; overrides native source-origin covariate.
        TARGET_FEATURES_CSV = None  # Optional target-ID-indexed numeric CSV.
        """
        description = """The standard assay and standard-or-amplified union form the natural
        paired evaluation. No artificial 0–80% thinning curve is imposed on these labels.
        The retained cohort/filter audit is exported. Optional native source-origin location is
        a coarse covariate; an external location CSV may be supplied instead.
        Target features require an aligned numeric CSV independent of the projection outcomes."""
    return [
        ("markdown", f"**Dataset inputs.** {description}"),
        ("code", textwrap.dedent(options).strip()),
        ("code", "if RUN_QIAO:\n    raise ValueError('The Qiao comparison is enabled in simulation only; use RUN_QIAO=False here.')\n"
         + textwrap.dedent(load).strip() + "\npreflight_folds = preflight(dataset)"),
        ("markdown", """**Run all configured repetitions.** Validated raw files are reused
        without downloading again. The shared pipeline checkpoints completed work and exports
        full fold/target/repetition metrics, selected settings, calibration diagnostics, and predictions."""),
        ("code", '''
        artifacts = run_experiment(
            dataset=dataset, settings=settings,
            checkpoint_dir=CHECKPOINT_DIR, export_dir=EXPORT_DIR)
        all_artifacts = {dataset.name: artifacts}
        '''),
    ]


def notebook(name, commit, source_hash):
    modules = ["numpy", "scipy", "pandas", "sklearn", "joblib", "matplotlib", "yaml", "IPython"]
    if name in {"SPIDER", "Projection_TAGs"}:
        modules.append("rdata")
    if name == "Projection_TAGs":
        modules.append("openpyxl")
    parts = [
        ("markdown", f"""# Gene2Wire {name} — OnDemand 0908

        This notebook runs the shared, versioned paper experiment code. It contains no
        dataset-specific model patches. Use a Python 3.10+ OnDemand kernel with the repository's
        experiment dependencies available; the notebook does not install packages into the kernel.

        Run cells from top to bottom. The first uncached use downloads the pinned code and raw
        data; subsequent runs reuse verified local files. Checkpoints persist across kernel
        disconnects. All result tables and predictions go beneath
        `/home/yueyue/gene2wire/paper_figure_exports`; plots display here and save as PDF only.

        Outputs are deliberately cleared. Earlier paper numbers remain provisional until
        the harmonized reruns and their diagnostics have been reviewed."""),
        ("code", CONFIG.replace("__CORE_COMMIT_0908__", commit).replace("__SOURCE_HASH_0908__", source_hash)
         + f"\nREQUIRED_MODULES = {modules!r}\n"),
        ("markdown", """**Load the pinned code.** A verified local checkout works offline.
        A stale package already imported in this kernel requires a kernel restart;
        the notebook never reloads or rewrites installed model source."""),
        ("code", BOOTSTRAP),
        ("markdown", """**Shared scientific settings.** `full_joint` evaluates simultaneous
        rank/penalty candidates from a deterministic bounded Cartesian grid. Exact direct and
        residual-off endpoints are included in Joint's budget. Inner validation selects models;
        final preprocessing, calibration, and fitting use the designated development data.
        Mechanism/calibration stress tests are simulation controls. Every model shares each
        scenario's observation mask and paired-reference subset."""),
        ("code", SETTINGS),
        ("code", PREFLIGHT),
    ]
    parts.extend(dataset_cells(name))
    parts.extend([
        ("markdown", """**Loss-rate curves and PDF figures.** Curves retain the complete configured
        loss-rate grid. Simulation includes all three sharing strengths; BARseq includes both panels.
        The natural Projection-TAGs analysis uses paired-audit and model-comparison panels.
        No PNG files are written."""),
        ("code", '''
        for label, artifacts in all_artifacts.items():
            plot_results(artifacts, output_dir=FIGURE_DIR)
        '''),
        ("markdown", "**Result tables and reusable exports.**"),
        ("code", FINAL),
    ])
    return {"cells": [cell(kind, source, i) for i, (kind, source) in enumerate(parts)],
            "metadata": {"kernelspec": {"display_name": "Python 3 (OnDemand)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10"},
                         "gene2wire": {"protocol": "0908-v1", "core_commit": commit,
                                       "source_hash": source_hash, "dataset": name,
                                       "environment": "OnDemand"}},
            "nbformat": 4, "nbformat_minor": 5}


def build(commit="__CORE_COMMIT_0908__", source_hash="__SOURCE_HASH_0908__", output_dir=ROOT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in NOTEBOOK_NAMES:
        destination = output_dir / f"{name}_0908.ipynb"
        destination.write_text(json.dumps(notebook(name, commit, source_hash), indent=1, ensure_ascii=False) + "\n")
        paths.append(destination)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="__CORE_COMMIT_0908__")
    parser.add_argument("--source-hash", default="__SOURCE_HASH_0908__")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    for path in build(args.commit, args.source_hash, args.output_dir):
        print(path)
