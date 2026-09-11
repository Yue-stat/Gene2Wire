"""Build pinned OnDemand notebooks for partial input-gene panel overlap."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_gene2wire_notebook_common", ROOT / "scripts" / "build_notebooks_0908.py")
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

NAMES = ("barseq", "projection_tags", "spider", "simulation")


CONFIG = '''
from pathlib import Path
import os

# Change this list before Run All. Realized overlap is also reported because
# odd panel sizes (for example BARseq K=11) require integer gene counts.
OVERLAP_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
PANEL_SIZE = None  # None uses floor(number_of_genes / 2).
INCLUDE_CONTROLS = True

N_OUTER_FOLDS = 3
N_REPETITIONS = 10
N_JOBS = 32
PARALLEL_UNIT = 'scenario'
STRATEGY = 'full_joint'
CANDIDATE_BUDGET = 32
SEED = 20260910
PAIRED_FRACTION = 0.20
POSITIVE_LOSS_RATES = (0.0,)
USE_LOCATION = False
USE_TARGET_FEATURES = False

# RF is intentionally excluded from this experiment for now.
RUN_RANDOM_FOREST = False
RUN_INFORMATION_CONTROLS = False
RUN_QIAO = False

SHOW_PROGRESS = True
PROGRESS_LEVEL = 'summary'
PROGRESS_INTERVAL_SECONDS = 60.0
SHOW_FULL_DIAGNOSTICS = False

BASE_DIR = Path(os.environ.get('GENE2WIRE_PROJECT_DIR', '/home/yueyue/gene2wire')).expanduser()
RAW_DATA_DIR = BASE_DIR / 'raw_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'gene_overlap_v1'
EXPORT_DIR = BASE_DIR / 'paper_figure_exports'
FIGURE_DIR = BASE_DIR / 'figures' / 'gene_overlap_0910'
CODE_CACHE_DIR = BASE_DIR / 'code'

RESULTS_ONLY = False
EXISTING_EXPORT_DIRS = __EXISTING__  # Exact completed run directories or None.
EXPECTED_EXPORT_LABELS = __LABELS__
CORE_COMMIT = '__COMMIT__'
EXPECTED_SOURCE_HASH = '__HASH__'
REPO_URL = 'https://github.com/Yue-stat/Gene2Wire.git'

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
'''


SETTINGS = '''
from dataclasses import asdict
import numpy as np
import pandas as pd
from IPython.display import display
from gene2wire.experiments.protocol import Settings
from gene2wire.experiments.gene_overlap import (
    build_overlap_views, common_target_dataset, paired_group_folds,
    run_gene_overlap_experiment,
)
from gene2wire.experiments.gene_overlap_plotting import plot_overlap_results
from gene2wire.experiments.reporting import (
    configure_full_display, configure_compact_display,
    load_existing_exports, display_diagnostics,
)

if SHOW_FULL_DIAGNOSTICS:
    configure_full_display()
else:
    configure_compact_display()

settings = Settings(
    n_outer_folds=N_OUTER_FOLDS, use_location=USE_LOCATION,
    use_target_features=USE_TARGET_FEATURES, n_jobs=N_JOBS,
    parallel_unit=PARALLEL_UNIT, n_repetitions=N_REPETITIONS,
    strategy=STRATEGY, candidate_budget=CANDIDATE_BUDGET, seed=SEED,
    paired_fraction=PAIRED_FRACTION, calibration_fractions=(PAIRED_FRACTION,),
    loss_rates=POSITIVE_LOSS_RATES,
    run_information_controls=RUN_INFORMATION_CONTROLS,
    run_random_forest=RUN_RANDOM_FOREST, run_qiao=RUN_QIAO,
    run_mechanism_controls=False, run_calibration_controls=False,
)
assert not RUN_RANDOM_FOREST and not settings.run_random_forest
assert not USE_LOCATION and not USE_TARGET_FEATURES
if SHOW_FULL_DIAGNOSTICS:
    display(pd.DataFrame([asdict(settings)]).T.rename(columns={0: 'settings'}))
else:
    print({'overlap_grid': OVERLAP_GRID, 'panel_size': PANEL_SIZE,
           'folds': N_OUTER_FOLDS, 'repetitions': N_REPETITIONS,
           'workers': N_JOBS, 'models': ['PU', 'PU-MIRT', 'PU-Joint'],
           'random_forest': RUN_RANDOM_FOREST, 'positive_loss_rates': POSITIVE_LOSS_RATES})
if RESULTS_ONLY:
    all_artifacts = load_existing_exports(
        EXISTING_EXPORT_DIRS, expected_labels=EXPECTED_EXPORT_LABELS)
    print('RESULTS_ONLY: loading completed CSV/NPZ exports without raw-data loading or fitting.')
'''


PREFLIGHT = '''
def preview_overlap_views(label, views):
    first_rep = min(int(view.metadata['experiment_repetition']) for view in views)
    selected = [view for view in views
                if int(view.metadata['experiment_repetition']) == first_rep]
    rows = []
    for view in selected:
        context = view.metadata['experiment_context']
        rows.append({
            'label': label, 'dataset': view.name,
            'design': context['panel_design'], 'arm': context['arm'],
            'requested_overlap': context['requested_overlap'],
            'actual_overlap': context['actual_overlap'],
            'panel_size': context['panel_size'],
            'cells': len(view.cell_ids), 'targets': len(view.target_ids),
            'genes_total': view.gene_matrix.shape[1],
            'panel_A_cells': int(np.sum(view.groups['overlap_panel'] == 'A')),
            'panel_B_cells': int(np.sum(view.groups['overlap_panel'] == 'B')),
            'models': ', '.join(view.metadata['model_allowlist']),
        })
    frame = pd.DataFrame(rows).drop_duplicates()
    display(frame if SHOW_FULL_DIAGNOSTICS else frame.head(30))
    union = next(view for view in selected
                 if view.metadata['experiment_context']['arm'] == 'union')
    fold = union.split_builder(N_OUTER_FOLDS, SEED)[0]
    features = union.feature_builder(fold.train_rows, False, False)
    print({'first_fold_feature_shape': features.X.shape,
           'feature_blocks': dict(features.feature_blocks),
           'targets_unchanged': True,
           'hidden_expression_used_by_scaler': False})
'''


RUN_HELPER = '''
def execute_views(label, views, export_name):
    preview_overlap_views(label, views)
    return run_gene_overlap_experiment(
        views, settings, checkpoint_dir=CHECKPOINT_DIR, export_dir=EXPORT_DIR,
        export_name=export_name, progress=SHOW_PROGRESS,
        progress_interval=PROGRESS_INTERVAL_SECONDS,
        progress_level=PROGRESS_LEVEL, worker_status=worker_status)
'''


DIAGNOSTICS = '''
figure_paths = {}
for label, artifacts in all_artifacts.items():
    figure_paths[label] = plot_overlap_results(
        artifacts, FIGURE_DIR / label.replace(' ', '_'),
        prefix=label.replace(' ', '_'), display=True)
    display_diagnostics(artifacts, label=label, full=SHOW_FULL_DIAGNOSTICS)
display(figure_paths)
print('CSV/NPZ exports:', {label: str(value.export_dir) for label, value in all_artifacts.items()})
print('Persistent checkpoints:', CHECKPOINT_DIR)
'''


def guarded(source):
    return "if not RESULTS_ONLY:\n" + textwrap.indent(textwrap.dedent(source).strip(), "    ")


def notebook(name: str, commit: str, source_hash: str):
    if name not in NAMES:
        raise ValueError(name)
    labels = {
        "barseq": ("BARseq A1 crossed", "BARseq A1 animal-aligned", "BARseq M1 crossed"),
        "projection_tags": ("Projection-TAGs",), "spider": ("SPIDER",),
        "simulation": ("simulation",),
    }[name]
    modules = ["numpy", "scipy", "pandas", "sklearn", "joblib", "threadpoolctl",
               "matplotlib", "yaml", "IPython"]
    if name == "projection_tags":
        modules += ["rdata", "openpyxl"]
    elif name == "spider":
        modules += ["rdata"]
    config = (CONFIG.replace("__COMMIT__", commit).replace("__HASH__", source_hash)
              .replace("__LABELS__", repr(labels))
              .replace("__EXISTING__", repr({label: None for label in labels})))
    config += f"\nREQUIRED_MODULES = {modules!r}\n"
    title = {
        "barseq": "BARseq gene-panel overlap",
        "projection_tags": "Projection-TAGs gene-panel overlap",
        "spider": "SPIDER gene-panel overlap",
        "simulation": "Simulation gene-panel overlap",
    }[name]
    description = {
        "barseq": "A1 includes a crossed within-animal design and a two-real-animal panel-aligned stress test; M1 uses crossed pseudo-panels within its single animal.",
        "projection_tags": "Six animals are arranged as three A/B pairs. Every outer fold uses one complete pair for train, validation and test. Targets are filtered by the native W mask before fitting so every retained target is assayed in every animal.",
        "spider": "A/B pseudo-panels are balanced within spatial slices while the original contiguous AP outer folds remain unchanged.",
        "simulation": "Within each repetition and sharing strength, truth, outcomes and spatial folds stay fixed while only the outcome-independent gene-panel overlap changes.",
    }[name]
    parts = [
        ("markdown", f"""# {title}

        {description}

        Each cell sees exactly `PANEL_SIZE` genes from panel A or B. Only input
        expression is hidden; projection targets, target labels, measurement masks,
        calibration rows and outer test cells are unchanged. Fold-specific scaling
        reads only expression values visible in training. Missing values become zero
        after centering, meaning the visible-training mean rather than biological zero.

        Primary comparisons are mask-aware union PU, PU-MIRT and PU-Joint. Controls
        include intersection-only PU, a disjoint-feature panel-separated PU
        parameterization, the same-K 100% endpoint, and an all-gene PU oracle. The
        panel-separated arm has distinct A/B slopes and intercepts but shares one
        tuning and detection-calibration run; it is not two independently calibrated
        fits. RF is disabled and is never scheduled.

        Scope boundary: this primary v1 does not include the separate-A MIRT ablation,
        an unpenalized animal-by-target nuisance model, or the optional Tech-SAR 80%
        sensitivity analysis. Do not use these notebooks alone for those claims.

        Run top to bottom in the OnDemand environment. Change `OVERLAP_GRID` in the
        first code cell if needed. Compact diagnostics remain on by default; complete
        tables are always written to CSV."""),
        ("markdown", "## Configuration"), ("code", config),
        ("markdown", "## Load and verify the pinned Gene2Wire core"),
        ("code", _common.BOOTSTRAP),
        ("markdown", "## Shared model and overlap settings"), ("code", SETTINGS),
        ("markdown", "## CPU allocation and active experiment workers"),
        ("code", _common.WORKER_STATUS),
        ("markdown", "## Outcome-independent design preview"), ("code", PREFLIGHT),
        ("code", RUN_HELPER),
    ]
    if name == "barseq":
        parts += [
            ("markdown", "## Load BARseq A1/M1 and construct gene panels"),
            ("code", guarded('''
                from gene2wire.experiments.datasets.barseq import load_barseq
                datasets = load_barseq(RAW_DATA_DIR / 'BARseq')
                a1, m1 = datasets['A1'], datasets['M1']
                a1_crossed = build_overlap_views(
                    a1, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='crossed',
                    strata=('animal', 'slice'), include_controls=INCLUDE_CONTROLS, seed=SEED)
                animals = sorted(np.unique(a1.groups['animal']), key=str)
                if len(animals) != 2:
                    raise ValueError('BARseq A1 animal-aligned design requires exactly two animals')
                a1_mapping = {animals[0]: 'A', animals[1]: 'B'}
                a1_aligned = build_overlap_views(
                    a1, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='animal_aligned',
                    aligned_group='animal', aligned_mapping=a1_mapping,
                    include_controls=INCLUDE_CONTROLS, seed=SEED)
                m1_crossed = build_overlap_views(
                    m1, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='crossed',
                    strata=('slice',), include_controls=INCLUDE_CONTROLS, seed=SEED)
            ''')),
            ("markdown", "## Run A1 crossed, A1 aligned stress test, and M1 crossed"),
            ("code", guarded('''
                all_artifacts = {
                    'BARseq A1 crossed': execute_views('BARseq A1 crossed', a1_crossed,
                                                       'BARseq_A1_crossed_gene_overlap_0910'),
                    'BARseq A1 animal-aligned': execute_views('BARseq A1 animal-aligned', a1_aligned,
                                                              'BARseq_A1_animal_aligned_gene_overlap_0910'),
                    'BARseq M1 crossed': execute_views('BARseq M1 crossed', m1_crossed,
                                                       'BARseq_M1_crossed_gene_overlap_0910'),
                }
            ''')),
        ]
    elif name == "projection_tags":
        parts += [
            ("markdown", "## Load Projection-TAGs and audit common target coverage"),
            ("code", "LOCATION_FEATURES_CSV = None\nTARGET_FEATURES_CSV = None"),
            ("code", guarded('''
                from gene2wire.experiments.datasets.projection_tags import load_projection_tags
                dataset = load_projection_tags(
                    RAW_DATA_DIR / 'Projection_TAGs',
                    location_features_csv=LOCATION_FEATURES_CSV,
                    target_features_csv=TARGET_FEATURES_CSV)
                dataset = common_target_dataset(dataset, 'animal')
                pairs = ((1, 4), (2, 6), (3, 5))
                dataset = paired_group_folds(dataset, 'animal', pairs)
                mapping = {1: 'A', 4: 'B', 2: 'A', 6: 'B', 3: 'A', 5: 'B'}
                views = build_overlap_views(
                    dataset, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='animal_aligned',
                    aligned_group='animal', aligned_mapping=mapping,
                    include_controls=INCLUDE_CONTROLS, seed=SEED)
                display(pd.DataFrame([dataset.metadata['common_target_audit']]))
            ''')),
            ("markdown", "## Run paired-animal gene-overlap experiment"),
            ("code", guarded('''
                all_artifacts = {'Projection-TAGs': execute_views(
                    'Projection-TAGs', views, 'Projection_TAGs_gene_overlap_0910')}
            ''')),
        ]
    elif name == "spider":
        parts += [
            ("markdown", "## Load SPIDER and retain spatial outer folds"),
            ("code", guarded('''
                from gene2wire.experiments.datasets.spider import load_spider
                dataset = load_spider(RAW_DATA_DIR / 'SPIDER')
                views = build_overlap_views(
                    dataset, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='crossed',
                    strata=('slice',), include_controls=INCLUDE_CONTROLS, seed=SEED)
            ''')),
            ("markdown", "## Run SPIDER gene-overlap experiment"),
            ("code", guarded('''
                all_artifacts = {'SPIDER': execute_views(
                    'SPIDER', views, 'SPIDER_gene_overlap_0910')}
            ''')),
        ]
    else:
        parts += [
            ("markdown", "## Generate fixed simulation truths and overlap views"),
            ("code", guarded('''
                from gene2wire.experiments.datasets.simulation import generate_simulation
                SHARING_STRENGTHS = (0.0, 0.5, 1.0)
                SIMULATION_OPTIONS = {
                    'n_cells': 400, 'n_targets': 36, 'n_gene_features': 16,
                    'n_location_features': 4, 'n_slices': 20, 'true_rank': 2,
                    'n_target_features': 4, 'signal_sd': 1.35,
                    'target_feature_noise': 0.85,
                    'target_prevalence_range': (0.10, 0.30),
                    'truth_uses_location': False,
                }
                views = []
                for rho in SHARING_STRENGTHS:
                    for repetition in range(N_REPETITIONS):
                        dataset = generate_simulation(
                            repetition, rho, seed=SEED, **SIMULATION_OPTIONS)
                        views.extend(build_overlap_views(
                            dataset, OVERLAP_GRID, panel_size=PANEL_SIZE,
                            n_repetitions=1, panel_design='crossed', strata=('slice',),
                            include_controls=INCLUDE_CONTROLS, seed=SEED))
                views = tuple(views)
            ''')),
            ("markdown", "## Run simulation gene-overlap experiment"),
            ("code", guarded('''
                all_artifacts = {'simulation': execute_views(
                    'simulation', views, 'simulation_gene_overlap_0910')}
            ''')),
        ]
    parts += [
        ("markdown", "## Figures, essential metrics, selected hyperparameters and diagnostics"),
        ("code", DIAGNOSTICS),
    ]
    return {
        "cells": [_common.cell(kind, source, index, "0910")
                  for index, (kind, source) in enumerate(parts)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-hash", required=True)
    args = parser.parse_args()
    for name in NAMES:
        path = ROOT / f"gene2wire_{name}_overlap_grid.ipynb"
        path.write_text(json.dumps(notebook(name, args.commit, args.source_hash),
                                   indent=1, ensure_ascii=False) + "\n")
        print(path.name)


if __name__ == "__main__":
    main()
