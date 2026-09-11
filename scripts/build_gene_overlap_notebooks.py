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
INCLUDE_SHARED_A_ABLATION = True

# Fixed, model-identical stabilization for the separate panel/animal-by-target
# nuisance coefficients. It is not part of the hyperparameter search.
NUISANCE_L2 = 1e-4

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
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'gene_overlap_v2'
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
    run_gene_overlap_experiment, validate_overlap_artifact,
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
    nuisance_l2=NUISANCE_L2,
)
assert not RUN_RANDOM_FOREST and not settings.run_random_forest
assert not USE_LOCATION and not USE_TARGET_FEATURES
if SHOW_FULL_DIAGNOSTICS:
    display(pd.DataFrame([asdict(settings)]).T.rename(columns={0: 'settings'}))
else:
    print({'overlap_grid': OVERLAP_GRID, 'panel_size': PANEL_SIZE,
           'folds': N_OUTER_FOLDS, 'repetitions': N_REPETITIONS,
           'workers': N_JOBS, 'models': ['PU', 'PU-MIRT', 'PU-Joint'],
           'shared_A_ablation': INCLUDE_SHARED_A_ABLATION,
           'nuisance_l2': NUISANCE_L2,
           'random_forest': RUN_RANDOM_FOREST, 'positive_loss_rates': POSITIVE_LOSS_RATES})
if RESULTS_ONLY:
    all_artifacts = load_existing_exports(
        EXISTING_EXPORT_DIRS, expected_labels=EXPECTED_EXPORT_LABELS)
    for saved_label, saved_artifacts in all_artifacts.items():
        validate_overlap_artifact(
            saved_artifacts,
            expected_panel_design=EXPECTED_PANEL_DESIGNS[saved_label],
            expected_overlap_grid=OVERLAP_GRID,
            require_release_v2=True)
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
        arm = context['arm']
        panel_a = tuple(view.metadata['gene_overlap']['panel_A'])
        panel_b = tuple(view.metadata['gene_overlap']['panel_B'])
        source_gene_count = int(view.gene_matrix.shape[1])
        genes_per_cell = (source_gene_count if arm == 'all_gene_oracle'
                          else int(context['panel_size']))
        if arm == 'intersection':
            model_gene_columns = len(set(panel_a).intersection(panel_b))
        elif arm in {'disjoint_coefficient', 'shared_a', 'separate_a'}:
            model_gene_columns = 2 * int(context['panel_size'])
        elif arm == 'independent_panel':
            model_gene_columns = int(context['panel_size'])
        elif arm == 'all_gene_oracle':
            model_gene_columns = source_gene_count
        else:
            model_gene_columns = len(set(panel_a).union(panel_b))
        rows.append({
            'label': label, 'dataset': view.name,
            'design': context['panel_design'], 'arm': arm,
            'requested_overlap': context['requested_overlap'],
            'actual_overlap': context['actual_overlap'],
            'panel_size': context['panel_size'],
            'cells': len(view.cell_ids), 'targets': len(view.target_ids),
            'source_genes_P': source_gene_count,
            'available_genes_per_cell': genes_per_cell,
            'common_gene_count': len(set(panel_a).intersection(panel_b)),
            'union_gene_count': len(set(panel_a).union(panel_b)),
            'model_gene_columns': model_gene_columns,
            'panel_A_cells': int(np.sum(view.groups['overlap_panel'] == 'A')),
            'panel_B_cells': int(np.sum(view.groups['overlap_panel'] == 'B')),
            'models': ', '.join(view.metadata['model_allowlist']),
        })
    frame = pd.DataFrame(rows).drop_duplicates()
    display(frame)
    union = next(view for view in selected
                 if view.metadata['experiment_context']['arm'] == 'union')
    fold = union.split_builder(N_OUTER_FOLDS, SEED)[0]
    features = union.feature_builder(fold.train_rows, False, False)
    resolved_k = int(union.metadata['experiment_context']['panel_size'])
    source_p = int(union.gene_matrix.shape[1])
    print({
        'same_K_100pct_overlap': (
            f'both panels measure the same K={resolved_k} genes; this is not all {source_p} genes'),
        'all_P_gene_oracle': f'all P={source_p} genes are visible; plotted as a separate upper-bound point',
    })
    print({'first_fold_feature_shape': features.X.shape,
           'feature_blocks': dict(features.feature_blocks),
           'nuisance_shape': (None if features.X_nuisance is None
                              else features.X_nuisance.shape),
           'nuisance_names': tuple(features.nuisance_names),
           'targets_unchanged': True,
           'scaler_policy': 'visible outer-training expression only'})
    for checked_fold in union.split_builder(N_OUTER_FOLDS, SEED):
        for role, role_rows in (
                ('train', checked_fold.train_rows),
                ('validation', checked_fold.validation_rows),
                ('test', checked_fold.test_rows)):
            present = set(np.asarray(union.groups['overlap_panel'])[role_rows].astype(str))
            if present != {'A', 'B'}:
                raise ValueError(
                    f'Fold {checked_fold.outer_fold} {role} does not contain both gene panels')
    print('Fold audit: every train/validation/test role contains panels A and B.')
    if union.platform is not None and 'animal' in union.groups:
        animal_platform_panel = pd.DataFrame({
            'animal': np.asarray(union.groups['animal']).astype(str),
            'platform': np.asarray(union.platform).astype(str),
            'panel': np.asarray(union.groups['overlap_panel']).astype(str),
        }).drop_duplicates()
        if animal_platform_panel['animal'].duplicated().any():
            raise ValueError('Each Projection-TAGs animal must have one platform and one panel')
        platform_sets = animal_platform_panel.groupby('panel')['platform'].agg(lambda x: set(x))
        if len(platform_sets) != 2 or platform_sets.iloc[0] != platform_sets.iloc[1]:
            raise ValueError('Projection-TAGs panel assignment is confounded with platform')
        print('Animal-level platform × assigned gene-panel audit:')
        display(pd.crosstab(animal_platform_panel['panel'], animal_platform_panel['platform']))
'''


RUN_HELPER = '''
def execute_views(label, views, export_name, run_settings=None):
    preview_overlap_views(label, views)
    return run_gene_overlap_experiment(
        views, settings if run_settings is None else run_settings,
        checkpoint_dir=CHECKPOINT_DIR, export_dir=EXPORT_DIR,
        export_name=export_name, progress=SHOW_PROGRESS,
        progress_interval=PROGRESS_INTERVAL_SECONDS,
        progress_level=PROGRESS_LEVEL, worker_status=worker_status)
'''


DIAGNOSTICS = '''
figure_paths = {}
for label, artifacts in all_artifacts.items():
    paths = plot_overlap_results(
        artifacts, FIGURE_DIR / label.replace(' ', '_'),
        prefix=label.replace(' ', '_'), display=True)
    if not paths:
        raise RuntimeError(f'No gene-overlap figures were produced for {label!r}')
    figure_paths[label] = paths
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
    expected_designs = {
        "barseq": {"BARseq A1 crossed": "crossed",
                   "BARseq A1 animal-aligned": "animal_aligned",
                   "BARseq M1 crossed": "crossed"},
        "projection_tags": {"Projection-TAGs": "animal_aligned"},
        "spider": {"SPIDER": "crossed"},
        "simulation": {"simulation": "crossed"},
    }[name]
    config += f"EXPECTED_PANEL_DESIGNS = {expected_designs!r}\n"
    if name == "simulation":
        config += """

# Optional secondary analysis. It is simulation-only, union-arm-only, fixed at
# 80% Technical-SAR positive-label loss, and writes a separate export.
RUN_TECH_SAR_80_SENSITIVITY = False
TECH_SAR_80_EXPORT_DIR = None  # Exact completed sensitivity run for RESULTS_ONLY.
"""
    title = {
        "barseq": "BARseq gene-panel overlap",
        "projection_tags": "Projection-TAGs gene-panel overlap",
        "spider": "SPIDER gene-panel overlap",
        "simulation": "Simulation gene-panel overlap",
    }[name]
    description = {
        "barseq": "A1 includes a crossed within-animal design and a two-real-animal panel-aligned stress test; M1 uses crossed pseudo-panels within its single animal.",
        "projection_tags": "Six animals are arranged as three A/B pairs, with pair orientation chosen to balance sequencing platforms across panels. Every outer fold uses one complete pair for train, validation and test. Targets are filtered by the native W mask before fitting so every retained target is assayed in every animal.",
        "spider": "A/B pseudo-panels are balanced within spatial slices while the original contiguous AP outer folds remain unchanged.",
        "simulation": "Within each repetition and sharing strength, truth, outcomes and spatial folds stay fixed while only the outcome-independent gene-panel overlap changes.",
    }[name]
    parts = [
        ("markdown", f"""# {title}

        {description}

        In every same-budget arm, each cell sees exactly the resolved `K` genes from
        panel A or B. At 100% overlap, both groups see the *same K genes*; they do
        not suddenly see every source gene. The separately labeled all-`P`-gene
        oracle is the only arm that exposes every original gene. Only input
        expression is hidden; projection targets, target labels, measurement masks,
        calibration rows and outer test cells are unchanged. Fold-specific scaling
        reads only expression values visible in training. Missing values become zero
        after centering, meaning the visible-training mean rather than biological zero.

        Primary comparisons are mask-aware union PU, PU-MIRT and PU-Joint. Controls
        include intersection-only PU, a one-fit disjoint-coefficient PU control,
        the same-K 100% endpoint, and an all-gene PU oracle. Shared-A versus
        Separate-A MIRT is enabled by default as a matched pooled-fit ablation:
        its two arms differ only in whether the duplicated A/B gene blocks share
        one target-loading matrix. Panel/eligible-animal target offsets live in
        a separate nuisance design and use the same fixed stabilization in every
        primary model. Two fully independent panel calibrations/fits are deliberately
        not exposed here: they change both pooling and calibration and therefore do
        not isolate target-loading sharing. RF is disabled and is never scheduled.

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
                    strata=('animal', 'slice'), include_controls=INCLUDE_CONTROLS,
                    include_shared_a_ablation=INCLUDE_SHARED_A_ABLATION,
                    nuisance_groups=('animal',),
                    seed=SEED)
                animals = sorted(np.unique(a1.groups['animal']), key=str)
                if len(animals) != 2:
                    raise ValueError('BARseq A1 animal-aligned design requires exactly two animals')
                a1_mapping = {animals[0]: 'A', animals[1]: 'B'}
                a1_aligned = build_overlap_views(
                    a1, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='animal_aligned',
                    aligned_group='animal', aligned_mapping=a1_mapping,
                    include_controls=INCLUDE_CONTROLS,
                    include_shared_a_ablation=INCLUDE_SHARED_A_ABLATION,
                    seed=SEED)
                m1_crossed = build_overlap_views(
                    m1, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='crossed',
                    strata=('slice',), include_controls=INCLUDE_CONTROLS,
                    include_shared_a_ablation=INCLUDE_SHARED_A_ABLATION,
                    seed=SEED)
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
                if not np.asarray(dataset.measured, dtype=bool).all():
                    raise ValueError('Common-target Projection-TAGs W audit did not retain full coverage')
                pairs = ((1, 4), (2, 6), (3, 5))
                dataset = paired_group_folds(dataset, 'animal', pairs)
                # Preserve one A/B animal in each pair while balancing the two
                # sequencing platforms across panels (two RNA + one multiome each).
                mapping = {1: 'A', 4: 'B', 2: 'B', 6: 'A', 3: 'A', 5: 'B'}
                views = build_overlap_views(
                    dataset, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='animal_aligned',
                    aligned_group='animal', aligned_mapping=mapping,
                    include_controls=INCLUDE_CONTROLS,
                    include_shared_a_ablation=INCLUDE_SHARED_A_ABLATION,
                    seed=SEED)
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
                if not np.asarray(dataset.measured, dtype=bool).all():
                    raise ValueError('SPIDER target assays are not common across all retained cells')
                print({'common_target_coverage': True,
                       'retained_targets': len(dataset.target_ids)})
                views = build_overlap_views(
                    dataset, OVERLAP_GRID, panel_size=PANEL_SIZE,
                    n_repetitions=N_REPETITIONS, panel_design='crossed',
                    strata=('slice',), include_controls=INCLUDE_CONTROLS,
                    include_shared_a_ablation=INCLUDE_SHARED_A_ABLATION,
                    seed=SEED)
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
                from gene2wire.experiments.gene_overlap_sensitivity import (
                    prepare_tech_sar80_sensitivity,
                )
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
                            include_controls=INCLUDE_CONTROLS,
                            include_shared_a_ablation=INCLUDE_SHARED_A_ABLATION,
                            seed=SEED))
                views = tuple(views)
                tech_sar80_plan = prepare_tech_sar80_sensitivity(
                    views, settings, enabled=RUN_TECH_SAR_80_SENSITIVITY)
            ''')),
            ("markdown", "## Run simulation gene-overlap experiment"),
            ("code", guarded('''
                all_artifacts = {'simulation': execute_views(
                    'simulation', views, 'simulation_gene_overlap_0910')}
                if tech_sar80_plan is not None:
                    all_artifacts[tech_sar80_plan.label] = execute_views(
                        tech_sar80_plan.label, tech_sar80_plan.views,
                        tech_sar80_plan.export_name, run_settings=tech_sar80_plan.settings)
            ''')),
            ("code", '''
if RESULTS_ONLY and RUN_TECH_SAR_80_SENSITIVITY:
    if TECH_SAR_80_EXPORT_DIR is None or not str(TECH_SAR_80_EXPORT_DIR).strip():
        raise ValueError('Set TECH_SAR_80_EXPORT_DIR to the exact completed sensitivity run')
    from gene2wire.experiments.gene_overlap_sensitivity import TECH_SAR_80_LABEL
    from gene2wire.experiments.reporting import load_export_artifacts
    sensitivity_artifacts = load_export_artifacts(TECH_SAR_80_EXPORT_DIR)
    validate_overlap_artifact(
        sensitivity_artifacts, expected_panel_design='crossed',
        expected_overlap_grid=OVERLAP_GRID,
        require_release_v2=True, tech_sar80=True)
    all_artifacts[TECH_SAR_80_LABEL] = sensitivity_artifacts
'''),
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
