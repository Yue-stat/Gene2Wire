"""Build the canonical combined measurement-degradation notebooks.

Usage: python scripts/notebooks/build_measurement_degradation.py \
    --commit <SHA> --source-hash <SHA256> [--date MMDD]

The notebooks are deliberately thin entry points.  Panel construction,
positive censoring, fitting, evaluation, reporting, and scientific contrasts
all live in the version-pinned shared package.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "notebooks" / "measurement_degradation"
_spec = importlib.util.spec_from_file_location(
    "_gene2wire_measurement_notebook_common",
    Path(__file__).with_name("build_positive_label_hiding.py"),
)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

PROTOCOL_VERSION = "0912-v5-measurement-degradation"
NAMES = (
    "simulation",
    "BARseq_A1",
    "BARseq_M1",
    "MERGE_seq",
    "Projection_TAGs",
    "SPIDER",
    "SPIDER_Seq",
)
NOTEBOOK_FILENAMES = {
    "simulation": "simulation.ipynb",
    "BARseq_A1": "barseq_a1.ipynb",
    "BARseq_M1": "barseq_m1.ipynb",
    "MERGE_seq": "merge_seq.ipynb",
    "Projection_TAGs": "projection_tags.ipynb",
    "SPIDER": "spider_spatial.ipynb",
    "SPIDER_Seq": "spider_seq.ipynb",
}


CONFIG = '''
from pathlib import Path
import os

# Shared scientific defaults. Change switches here before Run All.
N_OUTER_FOLDS = 3
USE_LOCATION = False
USE_TARGET_FEATURES = False
N_JOBS = 32
PARALLEL_UNIT = 'scenario'
N_REPETITIONS = 5  # Independent generated datasets per simulation sharing strength.
N_PANEL_SEEDS = 5  # Panel-mask repeats within each real or generated dataset.
STRATEGY = 'full_joint'
CANDIDATE_BUDGET = 32
SEED = 20260912
PAIRED_FRACTION = 0.20
CALIBRATION_FRACTIONS = (PAIRED_FRACTION,)

# Per-assay coverage: 1.0 means every assay measures every available item.
# These are coverage levels, not the legacy fixed-K pairwise-overlap levels.
GENE_COVERAGES = (1.0, 5.0 / 6.0, 2.0 / 3.0, 0.50)
TARGET_COVERAGES = (1.0, 2.0 / 3.0, 0.50)
POSITIVE_RETENTIONS = (1.0, 0.75, 0.50, 0.25, 0.10)
ANCHOR_GENE_COVERAGE = 2.0 / 3.0
ANCHOR_TARGET_COVERAGE = 2.0 / 3.0
ANCHOR_POSITIVE_RETENTION = 0.50
VIRTUAL_ASSAYS = ('A', 'B', 'C')
HETEROGENEITY_DELTA = 1.0
INCLUDE_MATCHED_UNIFORM_CONTROL = True
INCLUDE_NATURAL_RECOVERY = True

# Retain the complete shared benchmark/control suite and add the three PU comparators.
RUN_INFORMATION_CONTROLS = True
RUN_RANDOM_FOREST = True
RUN_MECHANISM_CONTROLS = True
RUN_CALIBRATION_CONTROLS = True
RUN_QIAO = True
RUN_PU_COMPARATORS = True

# Display switches never change the scientific cache identity.
SHOW_PROGRESS = True
PROGRESS_LEVEL = 'summary'
PROGRESS_INTERVAL_SECONDS = 60.0
SHOW_FULL_DIAGNOSTICS = False

BASE_DIR = Path(os.environ.get(
    'GENE2WIRE_PROJECT_DIR', '/home/yueyue/gene2wire')).expanduser()
RAW_DATA_DIR = BASE_DIR / 'raw_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'measurement_degradation_v1'
EXPORT_DIR = BASE_DIR / 'paper_figure_exports'
FIGURE_DIR = BASE_DIR / 'figures' / 'measurement_degradation' / '__DATE__'
CODE_CACHE_DIR = BASE_DIR / 'code'

# Read-only redraw mode. Values are exact completed directories containing manifest.json.
RESULTS_ONLY = False
EXISTING_EXPORT_DIRS = __EXPORT_DIRS__

CORE_COMMIT = '__COMMIT__'
EXPECTED_SOURCE_HASH = '__HASH__'
REPO_URL = 'https://github.com/Yue-stat/Gene2Wire.git'

# The outer pool is the sole source of parallelism.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
'''


SETTINGS = '''
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from gene2wire.experiments.measurement_experiment import (
    MeasurementConfig,
    preview_measurement_experiment,
    run_measurement_experiment,
    run_measurement_simulation_experiments,
)
from gene2wire.experiments.measurement_plotting import (
    plot_accuracy_panel_sensitivity,
    plot_gene_target_brier_heatmap,
    plot_projection_tags_budget_recall,
    plot_retention_auprc,
)
from gene2wire.experiments.reporting import (
    configure_compact_display,
    configure_full_display,
    display_diagnostics,
    load_existing_exports,
)
from gene2wire.tuning import full_joint_candidates

if SHOW_FULL_DIAGNOSTICS:
    configure_full_display()
else:
    configure_compact_display()

settings = Settings(
    n_outer_folds=N_OUTER_FOLDS,
    use_location=USE_LOCATION,
    use_target_features=USE_TARGET_FEATURES,
    n_jobs=N_JOBS,
    parallel_unit=PARALLEL_UNIT,
    n_repetitions=N_REPETITIONS,
    strategy=STRATEGY,
    seed=SEED,
    paired_fraction=PAIRED_FRACTION,
    calibration_fractions=CALIBRATION_FRACTIONS,
    loss_rates=tuple(1.0 - value for value in POSITIVE_RETENTIONS),
    candidate_budget=CANDIDATE_BUDGET,
    run_information_controls=RUN_INFORMATION_CONTROLS,
    run_random_forest=RUN_RANDOM_FOREST,
    run_mechanism_controls=RUN_MECHANISM_CONTROLS,
    run_calibration_controls=RUN_CALIBRATION_CONTROLS,
    run_qiao=RUN_QIAO,
    run_pu_comparators=RUN_PU_COMPARATORS,
    supervision_profile='paired_reference',
)
measurement_config = MeasurementConfig(
    gene_coverages=GENE_COVERAGES,
    target_coverages=TARGET_COVERAGES,
    retentions=POSITIVE_RETENTIONS,
    anchor_gene_coverage=ANCHOR_GENE_COVERAGE,
    anchor_target_coverage=ANCHOR_TARGET_COVERAGE,
    anchor_retention=ANCHOR_POSITIVE_RETENTION,
    assay_ids=VIRTUAL_ASSAYS,
    heterogeneity_delta=HETEROGENEITY_DELTA,
    include_matched_uniform=INCLUDE_MATCHED_UNIFORM_CONTROL,
    include_natural_recovery=INCLUDE_NATURAL_RECOVERY,
    n_panel_seeds=N_PANEL_SEEDS,
)
KEY_MEASUREMENT_MODELS = (
    'PU', 'PU-MIRT', 'PU-Joint',
    'GenEML-adapted', 'Inductive-PU-MC', 'SAR-PU',
)

if SHOW_FULL_DIAGNOSTICS:
    display(pd.DataFrame([asdict(settings)]).T.rename(columns={0: 'shared setting'}))
    display(pd.DataFrame([asdict(measurement_config)]).T.rename(columns={0: 'measurement design'}))
else:
    print({
        'folds': N_OUTER_FOLDS, 'repetitions': N_REPETITIONS,
        'panel_seeds_per_dataset': N_PANEL_SEEDS,
        'n_jobs': N_JOBS, 'parallel_unit': PARALLEL_UNIT,
        'use_location': USE_LOCATION, 'use_target_features': USE_TARGET_FEATURES,
        'strategy': STRATEGY, 'candidate_budget': CANDIDATE_BUDGET,
        'paired_fraction': PAIRED_FRACTION,
        'gene_coverages': GENE_COVERAGES,
        'target_coverages': TARGET_COVERAGES,
        'positive_retentions': POSITIVE_RETENTIONS,
        'virtual_assays': VIRTUAL_ASSAYS,
        'all_controls': {
            'information': RUN_INFORMATION_CONTROLS,
            'random_forest': RUN_RANDOM_FOREST,
            'mechanism': RUN_MECHANISM_CONTROLS,
            'calibration': RUN_CALIBRATION_CONTROLS,
            'qiao': RUN_QIAO,
            'pu_comparators': RUN_PU_COMPARATORS,
        },
    })

if RESULTS_ONLY:
    all_artifacts = load_existing_exports(
        EXISTING_EXPORT_DIRS, expected_labels=EXPECTED_EXPORT_LABELS)
    print('RESULTS_ONLY: raw loading, design preview and fitting are skipped.')
    print('Plots and diagnostics use each export manifest as the authoritative settings record.')
'''


PREFLIGHT = '''
def _panel_membership_summary(membership):
    measured = membership.loc[membership['measured'].astype(bool)].copy()
    return (measured.groupby(
        ['requested_coverage', 'actual_coverage', 'assay'],
        dropna=False, observed=True,
    )['item'].agg(
        measured_items='size',
        first_items=lambda values: '; '.join(map(str, values.iloc[:12])),
    ).reset_index())


def preflight_measurement(dataset):
    dataset.validate()
    preview = preview_measurement_experiment(
        dataset, settings, measurement_config, repetition=0)
    views, tables = preview['views'], preview['tables']
    if not views:
        raise RuntimeError('Measurement preview produced no experiment views.')

    folds = tuple(dataset.split_builder(settings.n_outer_folds, settings.seed))
    for fold in folds:
        fold.validate(len(dataset.cell_ids))
    features = views[0].feature_builder(
        folds[0].train_rows, settings.use_location, settings.use_target_features)
    display(pd.DataFrame([{
        'dataset': dataset.name,
        'cells': len(dataset.cell_ids),
        'targets': len(dataset.target_ids),
        'source_gene_pool_P': len(dataset.gene_names),
        'native_measured_pairs': int(np.asarray(dataset.measured, dtype=bool).sum()),
        'reference_positives': int(np.asarray(dataset.reference, dtype=bool).sum()),
        'preview_views': len(views),
        'model_input_columns_first_view': features.X.shape[1],
        'feature_blocks_first_view': dict(features.feature_blocks),
        'target_feature_columns': 0 if features.Y_target is None else features.Y_target.shape[1],
        'natural_standard_assay': dataset.natural_observed is not None,
    }]))

    for dimension, pool_size in (
        ('gene', len(dataset.gene_names)),
        ('target', len(dataset.target_ids)),
    ):
        conditions = tables[f'measurement_{dimension}_panel_conditions']
        full = conditions.loc[np.isclose(conditions['requested_coverage'], 1.0)]
        if len(full) != 1 or int(full.iloc[0]['panel_size']) != pool_size:
            raise RuntimeError(
                f'{dimension} coverage 1.0 must measure all {pool_size} available items in every assay.')
        print(f'{dimension} coverage 1.0 audit: every panel measures {pool_size}/{pool_size} items.')
        display(conditions)
        display(tables[f'measurement_{dimension}_panel_assays'])
        display(tables[f'measurement_{dimension}_panel_pairs'])
        display(tables[f'measurement_{dimension}_panel_multiplicity'])
        membership = tables[f'measurement_{dimension}_panel_membership']
        display(_panel_membership_summary(membership))
        if SHOW_FULL_DIAGNOSTICS:
            display(membership)

    assignment = tables['measurement_assignment']
    display(assignment.groupby(
        ['stratum_kind', 'virtual_assay'], dropna=False, observed=True
    ).size().rename('cells').reset_index())
    display(tables['measurement_scenarios'])
    support = tables['measurement_support']
    support_columns = [column for column in (
        'gene_coverage', 'target_coverage', 'outer_fold', 'split', 'n_rows',
        'effective_union_coverage', 'graph_connected', 'n_components',
        'unsupported_target_count', 'single_assay_target_count',
        'unsupported_targets',
    ) if column in support]
    display(support.loc[:, support_columns])
    joint_support = tables['measurement_gene_target_support']
    joint_columns = [column for column in (
        'gene_coverage', 'target_coverage', 'outer_fold', 'split',
        'gene_target_pair_count', 'supported_gene_target_pair_count',
        'unsupported_gene_target_pair_count',
        'unsupported_gene_target_pair_fraction',
        'all_gene_target_pairs_supported', 'minimum_co_measured_cells',
        'median_co_measured_cells', 'maximum_co_measured_cells',
    ) if column in joint_support]
    display(joint_support.loc[:, joint_columns])
    unsupported_pairs = tables['measurement_unsupported_gene_target_pairs']
    print(f'Fold-local unsupported training gene-target pairs: {len(unsupported_pairs)} rows.')
    if not unsupported_pairs.empty:
        if not SHOW_FULL_DIAGNOSTICS and len(unsupported_pairs) > 60:
            print('Showing 60 rows; the complete unsupported-pairs table is exported.')
        display(unsupported_pairs if SHOW_FULL_DIAGNOSTICS else unsupported_pairs.head(60))
    if SHOW_FULL_DIAGNOSTICS:
        display(tables['measurement_support_counts'])

    tuning = settings.tuning_config(features.X.shape[1], len(dataset.target_ids))
    base_candidates = pd.DataFrame([{
        'model': model.name,
        'maximum_trials': tuning.candidate_budget,
        'bounded_grid_candidates': len(full_joint_candidates(model, tuning)),
        'eligible_structures': '; '.join(sorted({
            candidate.kind for candidate in full_joint_candidates(model, tuning)
        })),
    } for model in settings.models()])
    display(base_candidates)
    print('GenEML-adapted, Inductive-PU-MC and SAR-PU use the same candidate-budget ceiling; '
          'their realized trials and selected parameters are recorded in tuning.csv and selected.csv.')
    print('Masked target entries are excluded from calibration, fitting and validation loss; they are never negatives.')
    print('Gene masking precedes train-only scaling; value, observation-mask and assay-ID blocks distinguish missing from zero.')
    return preview
'''


RESULT_AUDIT = '''
def _bounded_table(frame, limit=60):
    if SHOW_FULL_DIAGNOSTICS or len(frame) <= limit:
        return frame
    print(f'Showing {limit}/{len(frame)} rows; the complete table is in the export directory.')
    return frame.head(limit)


def _balanced_model_table(frame, limit=60):
    if SHOW_FULL_DIAGNOSTICS or 'model' not in frame or len(frame) <= limit:
        return frame
    models = tuple(dict.fromkeys(frame['model'].astype(str)))
    per_model = max(1, limit // len(models))
    result = pd.concat([
        frame.loc[frame['model'].astype(str).eq(model)].head(per_model)
        for model in models
    ], ignore_index=True)
    print(f'Showing a model-balanced {len(result)}/{len(frame)} rows; '
          'the complete table is in the export directory.')
    return result


def display_measurement_audits(artifacts, label):
    print(f'\\n{label} — measurement-design and comparator audit tables')
    for table_name in (
        'measurement_gene_panel_conditions',
        'measurement_gene_panel_assays',
        'measurement_gene_panel_pairs',
        'measurement_gene_panel_multiplicity',
        'measurement_gene_panel_membership',
        'measurement_target_panel_conditions',
        'measurement_target_panel_assays',
        'measurement_target_panel_pairs',
        'measurement_target_panel_multiplicity',
        'measurement_target_panel_membership',
        'measurement_support',
        'measurement_support_assays',
        'measurement_support_pairs',
        'measurement_gene_target_support',
        'measurement_unsupported_gene_target_pairs',
        'measurement_scenarios',
        'observation_diagnostics',
        'heatmap_baseline_selection',
        'heatmap_contrasts',
        'panel_stability',
        'projection_budget_recall',
        'projection_budget_recall_per_target',
    ):
        frame = artifacts.tables.get(table_name, pd.DataFrame())
        if not frame.empty:
            print(f'{table_name}: {len(frame)} rows')
            display(_bounded_table(frame))

    metrics = artifacts.tables.get('metrics', pd.DataFrame())
    if not metrics.empty and 'model' in metrics:
        important = metrics.loc[metrics['model'].isin(KEY_MEASUREMENT_MODELS)].copy()
        columns = [column for column in (
            'dataset', 'sharing_strength', 'condition_roles', 'mechanism',
            'gene_coverage', 'target_coverage', 'positive_retention',
            'evaluation_scope', 'model', 'macro_auprc', 'macro_log_loss',
            'macro_brier', 'macro_predicted_prevalence', 'macro_reference_prevalence',
        ) if column in important]
        print('Important-model metrics (complete fold/repetition rows remain in metrics.csv):')
        display(_balanced_model_table(important.loc[:, columns]))

    selected = artifacts.tables.get('selected', pd.DataFrame())
    if not selected.empty and 'model' in selected:
        important = selected.loc[selected['model'].isin(KEY_MEASUREMENT_MODELS)].copy()
        parameter_hints = (
            'rank', 'l2', 'lambda', 'penalty', 'propensity', 'exposure',
            'factor', 'iteration', 'converged', 'retry', 'objective',
            'structure', 'validation', 'trial', 'budget', 'maxiter',
            'tolerance', 'message', 'status',
        )
        identity = (
            'dataset', 'sharing_strength', 'condition_roles', 'mechanism',
            'gene_coverage', 'target_coverage', 'positive_retention',
            'repetition', 'outer_fold', 'model',
        )
        columns = [column for column in important if (
            column in identity or any(hint in column.lower() for hint in parameter_hints)
        )]
        print('Selected parameters for PU / structured PU / new comparators:')
        display(_balanced_model_table(important.loc[:, columns]))

    tuning = artifacts.tables.get('tuning', pd.DataFrame())
    if not tuning.empty and 'model' in tuning:
        important = tuning.loc[tuning['model'].isin(KEY_MEASUREMENT_MODELS)].copy()
        groups = [column for column in ('model', 'condition_roles', 'mechanism') if column in important]
        if groups:
            important['validation_loss'] = pd.to_numeric(
                important.get('validation_loss'), errors='coerce')
            summary = important.groupby(groups, dropna=False, observed=True).agg(
                recorded_candidate_trials=('model', 'size'),
                best_recorded_validation_loss=('validation_loss', 'min'),
            ).reset_index()
            print('Recorded tuning trials for important models:')
            display(_balanced_model_table(summary))

    detection = artifacts.tables.get('detection', pd.DataFrame())
    if not detection.empty:
        values = [column for column in (
            'detection_log_loss', 'detection_brier', 'detection_mean',
            'detection_observed_rate', 'sensitivity_mae', 'sensitivity_rmse',
        ) if column in detection]
        groups = [column for column in (
            'mechanism', 'positive_retention', 'gene_coverage', 'target_coverage',
        ) if column in detection]
        if values:
            summary = detection.groupby(groups, dropna=False, observed=True)[values].mean().reset_index()
            print('Detection / propensity diagnostics:')
            display(_bounded_table(summary))

    convergence_rows = []
    for table_name, flag in (('selected', 'final_converged'), ('tuning', 'converged')):
        frame = artifacts.tables.get(table_name, pd.DataFrame())
        if not frame.empty and 'model' in frame:
            values = frame.get(flag, pd.Series(index=frame.index, dtype=object))
            if table_name == 'selected' and 'converged' in frame:
                values = values.combine_first(frame['converged'])
            values = values.fillna('not_recorded').astype(str).str.lower()
            for (model, status), count in values.groupby([frame['model'], values]).size().items():
                convergence_rows.append({
                    'table': table_name, 'model': model,
                    'convergence_status': status, 'records': int(count),
                })
    if convergence_rows:
        print('Final-fit and candidate convergence counts:')
        display(_balanced_model_table(pd.DataFrame(convergence_rows)))

    failures = artifacts.tables.get('failures')
    if failures is None:
        print('Failure status was not recorded.')
    elif failures.empty:
        print('Recorded failed units: 0.')
    else:
        print(f'Recorded failed units: {len(failures)}; no failed condition is dropped.')
        display(_bounded_table(failures, limit=20))
'''


PLOT_HELPERS = '''
def _role_rows(frame, role):
    required = {'condition_roles', 'evaluation_scope', 'mechanism'}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f'metrics.csv lacks measurement coordinates: {sorted(missing)}')
    return frame.loc[
        frame['condition_roles'].astype(str).str.contains(role, regex=False)
        & frame['evaluation_scope'].eq('native_reference')
        & frame['mechanism'].eq('assay_target_sar')
    ].copy()


def _plot_groups(frame):
    if 'sharing_strength' not in frame or frame['sharing_strength'].isna().all():
        return [('all', frame)]
    return [(f'sharing_{value:g}', group) for value, group in frame.groupby(
        'sharing_strength', dropna=False, observed=True)]


def _figure_path(label, suffix):
    safe = ''.join(character if character.isalnum() or character in '-_' else '_'
                   for character in str(label)).strip('_')
    path = FIGURE_DIR / f'{safe}_{suffix}.pdf'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_show(figure, label, suffix):
    path = _figure_path(label, suffix)
    figure.savefig(path, bbox_inches='tight')
    display(path)
    plt.show()
    plt.close(figure)
    return path
'''


RETENTION_PLOTS = '''
retention_figure_paths = {}
for label, artifacts in all_artifacts.items():
    per_repetition = artifacts.tables.get('per_repetition', pd.DataFrame())
    retention = _role_rows(per_repetition, 'retention_curve')
    retention['positive_retention'] = pd.to_numeric(
        retention['positive_retention'], errors='coerce')
    retention['macro_auprc'] = pd.to_numeric(retention['macro_auprc'], errors='coerce')
    retention = retention.loc[
        np.isfinite(retention['positive_retention'])
        & np.isfinite(retention['macro_auprc'])
    ]
    if retention.empty:
        raise RuntimeError(f'{label}: no native-reference retention-curve metrics were recorded.')
    for group_label, group in _plot_groups(retention):
        for mode in ('key', 'full'):
            options = ({'key_models': KEY_MEASUREMENT_MODELS} if mode == 'key' else {})
            figure, axis = plot_retention_auprc(
                group, mode=mode, retention_col='positive_retention',
                title=f'{label} ({group_label}) — {mode} models', **options)
            suffix = f'{group_label}_retention_macro_auprc_{mode}'
            retention_figure_paths[(label, group_label, mode)] = _save_show(
                figure, label, suffix)
display(retention_figure_paths)
'''


HEATMAP_PLOTS = '''
heatmap_figure_paths = {}
for label, artifacts in all_artifacts.items():
    selection = artifacts.tables.get('heatmap_baseline_selection', pd.DataFrame())
    if selection.empty or not {'model', 'selected'}.issubset(selection.columns):
        raise RuntimeError(f'{label}: fixed development-selected heatmap baseline was not recorded.')
    selected_flag = selection['selected'].astype(str).str.lower().isin(('true', '1'))
    selected_baseline = selection.loc[selected_flag, 'model']
    if len(selected_baseline) != 1:
        raise RuntimeError(f'{label}: expected exactly one fixed heatmap baseline.')
    baseline_model = str(selected_baseline.iloc[0])
    heatmap = _role_rows(
        artifacts.tables.get('per_repetition', pd.DataFrame()), 'coverage_heatmap')
    for column in ('gene_coverage', 'target_coverage', 'macro_brier'):
        heatmap[column] = pd.to_numeric(heatmap[column], errors='coerce')
    heatmap = heatmap.loc[np.isfinite(heatmap['macro_brier'])]
    if heatmap.empty:
        raise RuntimeError(f'{label}: no native-reference coverage-heatmap metrics were recorded.')
    for group_label, group in _plot_groups(heatmap):
        figure, axis = plot_gene_target_brier_heatmap(
            group, baseline_model=baseline_model, comparison_model='PU-MIRT',
            title=f'{label} ({group_label}) — fixed {baseline_model} contrast')
        suffix = f'{group_label}_gene_target_brier_heatmap'
        heatmap_figure_paths[(label, group_label)] = _save_show(figure, label, suffix)
display(heatmap_figure_paths)
'''


PANEL_STABILITY_PLOT = '''
panel_stability_figure_paths = {}
for label, artifacts in all_artifacts.items():
    stability = artifacts.tables.get('panel_stability', pd.DataFrame())
    required = {
        'model', 'macro_auprc', 'panel_sensitivity', 'gene_coverage',
        'target_coverage', 'positive_retention', 'mechanism',
        'is_reference_probability',
    }
    if stability.empty or not required.issubset(stability.columns):
        raise RuntimeError(f'{label}: panel_stability.csv is missing or incomplete.')
    gene_conditions = artifacts.tables.get(
        'measurement_gene_panel_conditions', pd.DataFrame())
    condition_columns = {'requested_coverage', 'actual_coverage'}
    if gene_conditions.empty or not condition_columns.issubset(gene_conditions.columns):
        raise RuntimeError(f'{label}: gene-panel condition audit is missing or incomplete.')
    saved_protocol = getattr(artifacts, 'manifest', {}).get('measurement_protocol', {})
    saved_anchor_gene_coverage = float(saved_protocol.get(
        'anchor_gene_coverage', ANCHOR_GENE_COVERAGE))
    saved_anchor_retention = float(saved_protocol.get(
        'anchor_retention', ANCHOR_POSITIVE_RETENTION))
    requested = pd.to_numeric(gene_conditions['requested_coverage'], errors='coerce')
    anchor_actual = pd.to_numeric(gene_conditions.loc[
        np.isclose(requested, saved_anchor_gene_coverage), 'actual_coverage'
    ], errors='coerce').dropna().unique()
    if len(anchor_actual) != 1:
        raise RuntimeError(f'{label}: expected one realized anchor gene coverage.')
    stability['gene_coverage'] = pd.to_numeric(stability['gene_coverage'], errors='coerce')
    stability['target_coverage'] = pd.to_numeric(
        stability['target_coverage'], errors='coerce')
    stability['positive_retention'] = pd.to_numeric(
        stability['positive_retention'], errors='coerce')
    stability['macro_auprc'] = pd.to_numeric(stability['macro_auprc'], errors='coerce')
    stability['panel_sensitivity'] = pd.to_numeric(
        stability['panel_sensitivity'], errors='coerce')
    reference_probability = stability['is_reference_probability'].astype(
        str).str.lower().isin(('true', '1'))
    stability = stability.loc[
        np.isclose(stability['gene_coverage'], anchor_actual[0])
        & np.isclose(stability['target_coverage'], 1.0)
        & np.isclose(stability['positive_retention'], saved_anchor_retention)
        & stability['mechanism'].eq('assay_target_sar')
        & reference_probability
        & np.isfinite(stability['macro_auprc'])
        & np.isfinite(stability['panel_sensitivity'])
    ]
    if stability.empty:
        raise RuntimeError(f'{label}: no finite panel-stability rows at the anchor coverage.')
    for group_label, group in _plot_groups(stability):
        figure, axis = plot_accuracy_panel_sensitivity(
            group, models=KEY_MEASUREMENT_MODELS,
            title=f'{label} ({group_label}) — accuracy versus panel sensitivity')
        axis.set_xlabel('Mean within-pair prediction SD across panel seeds ↓')
        suffix = f'{group_label}_accuracy_panel_sensitivity'
        panel_stability_figure_paths[(label, group_label)] = _save_show(
            figure, label, suffix)
display(panel_stability_figure_paths)
'''


PROJECTION_BUDGET_PLOT = '''
projection_budget_figure_paths = {}
for label, artifacts in all_artifacts.items():
    budget = artifacts.tables.get('projection_budget_recall', pd.DataFrame())
    required = {'model', 'budget_fraction', 'amplification_confirmed_recall'}
    if budget.empty or not required.issubset(budget.columns):
        raise RuntimeError(f'{label}: projection_budget_recall.csv is missing or incomplete.')
    budget['budget_fraction'] = pd.to_numeric(budget['budget_fraction'], errors='coerce')
    budget['amplification_confirmed_recall'] = pd.to_numeric(
        budget['amplification_confirmed_recall'], errors='coerce')
    budget = budget.loc[
        np.isfinite(budget['budget_fraction'])
        & np.isfinite(budget['amplification_confirmed_recall'])
    ]
    figure, axis = plot_projection_tags_budget_recall(
        budget, models=KEY_MEASUREMENT_MODELS,
        title=f'{label} — amplification-confirmed recovery')
    projection_budget_figure_paths[label] = _save_show(
        figure, label, 'amplification_confirmed_budget_recall')
display(projection_budget_figure_paths)
'''


DIAGNOSTICS = '''
for label, artifacts in all_artifacts.items():
    display_measurement_audits(artifacts, label)
    display_diagnostics(artifacts, label=label, full=SHOW_FULL_DIAGNOSTICS)
print('Figure PDFs:', FIGURE_DIR)
print('Reusable raw cache:', RAW_DATA_DIR)
print('Resumable checkpoints:', CHECKPOINT_DIR)
'''


def guarded(source: str) -> str:
    return "if not RESULTS_ONLY:\n" + textwrap.indent(
        textwrap.dedent(source).strip(), "    "
    )


def _dataset_parts(name: str):
    if name == "simulation":
        return [
            ("markdown", """## Simulation truth, preview and fixed input dimensions

            Sharing strengths 0, 0.5 and 1 are generated independently for each repetition.
            The gene-pool coverage grid acts on all 16 declared source genes; consequently,
            gene coverage 1.0 is 16/16 in A, B and C. Latent truth is evaluator-only."""),
            ("code", guarded('''
                from gene2wire.experiments.datasets.simulation import generate_simulation
                SHARING_STRENGTHS = (0.0, 0.5, 1.0)
                SIMULATION_OPTIONS = {
                    'n_cells': 400,
                    'n_targets': 36,
                    'n_gene_features': 16,
                    'n_location_features': 4,
                    'n_slices': 20,
                    'true_rank': 2,
                    'n_target_features': 4,
                    'signal_sd': 1.35,
                    'target_feature_noise': 0.85,
                    'target_prevalence_range': (0.10, 0.30),
                    'truth_uses_location': USE_LOCATION,
                }
                representative = generate_simulation(
                    repetition=0, sharing_strength=SHARING_STRENGTHS[0],
                    seed=SEED, **SIMULATION_OPTIONS)
                measurement_preview = preflight_measurement(representative)
            ''')),
            ("markdown", """## Run all simulation measurement conditions

            This is the expensive cell. The runner reuses compatible fold/model checkpoints,
            preserves every failed condition, and exports complete per-unit predictions and tables."""),
            ("code", guarded('''
                artifacts = run_measurement_simulation_experiments(
                    settings=settings,
                    config=measurement_config,
                    raw_cache_dir=RAW_DATA_DIR / 'simulation_measurement_degradation',
                    checkpoint_dir=CHECKPOINT_DIR,
                    export_dir=EXPORT_DIR,
                    sharing_strengths=SHARING_STRENGTHS,
                    simulation_options=SIMULATION_OPTIONS,
                    progress=SHOW_PROGRESS,
                    progress_level=PROGRESS_LEVEL,
                    progress_interval=PROGRESS_INTERVAL_SECONDS,
                    worker_status=worker_status,
                )
                all_artifacts = {'simulation': artifacts}
            ''')),
        ]

    if name.startswith("BARseq_"):
        panel = name.removeprefix("BARseq_")
        return [
            ("markdown", f"""## Load BARseq {panel} and preview the exact panels

            The source pool is the adapter's declared 23-gene BARseq panel. In particular,
            coverage 1.0 measures all 23/23 genes in virtual assays A, B and C; it is not
            the legacy same-eight-gene endpoint. Biological animal IDs are retained for
            stratified virtual-assay assignment when available."""),
            ("code", guarded(f'''
                from gene2wire.experiments.datasets.barseq import load_barseq
                dataset = load_barseq(RAW_DATA_DIR / 'BARseq')['{panel}']
                measurement_preview = preflight_measurement(dataset)
            ''')),
        ]
    if name == "MERGE_seq":
        return [
            ("markdown", """## MERGE-seq leak-isolated source-gene pool

            `load_merge_seq_overlap` freezes 128 outcome-independent candidate genes using
            reserved structurally unassayed design cells, then removes those cells before
            splitting. Coverage 1.0 means 128/128 genes in every virtual assay. The five
            native projection targets and recorded sample strata remain unchanged."""),
            ("code", "MERGE_SOURCE_GENE_COUNT = 128\nMERGE_PANEL_DESIGN_FRACTION = 0.20"),
            ("markdown", "## Load MERGE-seq and preview the exact panels"),
            ("code", guarded('''
                from gene2wire.experiments.datasets.merge_seq import load_merge_seq_overlap
                dataset = load_merge_seq_overlap(
                    RAW_DATA_DIR / 'MERGE_seq',
                    source_gene_count=MERGE_SOURCE_GENE_COUNT,
                    panel_design_fraction=MERGE_PANEL_DESIGN_FRACTION,
                    seed=SEED,
                )
                measurement_preview = preflight_measurement(dataset)
            ''')),
        ]
    if name == "Projection_TAGs":
        return [
            ("markdown", """## Projection-TAGs feature inputs and natural recovery

            Artificial degradation uses the same three-assay design as other datasets. The
            separate natural condition ranks standard-assay non-detections and evaluates only
            amplified-confirmed positives. The union is an assay reference, not anatomical truth."""),
            ("code", "LOCATION_FEATURES_CSV = None\nTARGET_FEATURES_CSV = None"),
            ("markdown", "## Load Projection-TAGs and preview the exact panels"),
            ("code", guarded('''
                from gene2wire.experiments.datasets.projection_tags import load_projection_tags
                dataset = load_projection_tags(
                    RAW_DATA_DIR / 'Projection_TAGs',
                    location_features_csv=LOCATION_FEATURES_CSV,
                    target_features_csv=TARGET_FEATURES_CSV,
                )
                measurement_preview = preflight_measurement(dataset)
            ''')),
        ]
    if name == "SPIDER":
        return [
            ("markdown", """## SPIDER spatial inputs

            The processed spatial adapter exposes native slices but no verified animal IDs.
            Virtual assays are balanced within the available slice strata without relabelling
            slices as animals. Optional target descriptors must be outcome-independent."""),
            ("code", "TARGET_FEATURES_CSV = None"),
            ("markdown", "## Load SPIDER and preview the exact panels"),
            ("code", guarded('''
                from gene2wire.experiments.datasets.spider import load_spider
                dataset = load_spider(
                    RAW_DATA_DIR / 'SPIDER',
                    target_features_csv=TARGET_FEATURES_CSV,
                )
                measurement_preview = preflight_measurement(dataset)
            ''')),
        ]
    if name == "SPIDER_Seq":
        return [
            ("markdown", """## SPIDER-Seq fixed measurement gene pool

            Adult1/2/3 retain their verified native target panels and animal IDs. The dedicated
            measurement adapter freezes a 2,000-gene post-normalization pool by gene identity,
            without reading projection outcomes. Artificial target panels intersect native
            availability; an absent injection is never converted to a negative."""),
            ("code", "SPIDER_SEQ_GENE_POOL_SIZE = 2000\nN_GENE_COMPONENTS = 50\n"
                     "LOCATION_FEATURES_CSV = None\nTARGET_FEATURES_CSV = None"),
            ("markdown", "## Load SPIDER-Seq and preview the exact panels"),
            ("code", guarded('''
                from gene2wire.experiments.datasets.spider_seq import load_spider_seq_measurement
                dataset = load_spider_seq_measurement(
                    RAW_DATA_DIR / 'SPIDER_Seq',
                    gene_pool_size=SPIDER_SEQ_GENE_POOL_SIZE,
                    n_gene_components=N_GENE_COMPONENTS,
                    location_features_csv=LOCATION_FEATURES_CSV,
                    target_features_csv=TARGET_FEATURES_CSV,
                )
                measurement_preview = preflight_measurement(dataset)
            ''')),
        ]
    raise ValueError(f"Unknown measurement notebook: {name}")


def notebook(name: str, commit: str, source_hash: str, date_suffix=None):
    if name not in NAMES:
        raise ValueError(f"Unknown measurement notebook: {name}")
    date_suffix = _common.release_date(date_suffix)
    label = {
        "simulation": "simulation",
        "BARseq_A1": "BARseq A1",
        "BARseq_M1": "BARseq M1",
        "MERGE_seq": "MERGE-seq",
        "Projection_TAGs": "Projection-TAGs",
        "SPIDER": "SPIDER",
        "SPIDER_Seq": "SPIDER-Seq",
    }[name]
    modules = [
        "numpy", "scipy", "pandas", "sklearn", "joblib", "threadpoolctl",
        "matplotlib", "yaml", "IPython",
    ]
    if name in {"Projection_TAGs", "SPIDER", "SPIDER_Seq"}:
        modules.append("rdata")
    if name == "Projection_TAGs":
        modules.append("openpyxl")
    config = (
        CONFIG.replace("__DATE__", date_suffix)
        .replace("__COMMIT__", commit)
        .replace("__HASH__", source_hash)
        .replace("__EXPORT_DIRS__", repr({label: None}))
        + f"\nREQUIRED_MODULES = {modules!r}\n"
        + f"EXPECTED_EXPORT_LABELS = {(label,)!r}\n"
    )
    third_plot = (
        PROJECTION_BUDGET_PLOT if name == "Projection_TAGs" else PANEL_STABILITY_PLOT
    )
    parts = [
        ("markdown", f"""# Gene2Wire {label}: combined measurement degradation — OnDemand {date_suffix}

        This notebook combines three separately audited mechanisms: assay-specific gene
        coverage, assay-specific target coverage, and assay×target positive censoring.
        `GENE_COVERAGES` and `TARGET_COVERAGES` are **per-assay coverage**, not pairwise
        overlap. At coverage 1.0, A, B and C each measure every available item. Thus BARseq
        uses all 23 genes per assay at 1.0; the earlier fixed-eight-gene 100%-overlap endpoint
        is not reused.

        Gene and target panels share the same nested, union-preserving cyclic generator, but
        their downstream meanings differ. Missing genes are masked before train-only scaling
        and enter the learner as value + observation-mask + assay-ID features. Missing targets
        form `W_fit = W_native & panel`; those entries are omitted from every training loss and
        are never labelled negative. Within measured target entries, heterogeneous censoring
        hides positives only. The native reference test scope remains fixed across conditions.

        Run top to bottom in a Python 3.10+ OnDemand kernel with the repository environment.
        The notebook installs nothing, contains no local model/mask implementation, and uses
        an exact source pin. Raw caches, checkpoints, complete CSV/NPZ exports, and PDF figures
        persist below `GENE2WIRE_PROJECT_DIR`. `RESULTS_ONLY=True` redraws an explicit completed
        export without loading raw data or launching fits. Canonical outputs are cleared."""),
        ("markdown", "## Run configuration and measurement grid"),
        ("code", config),
        ("markdown", """## Load and verify the frozen shared core

        A matching local checkout is accepted without network access. Otherwise the exact
        commit is cached and its byte-level source hash is verified before import."""),
        ("code", _common.BOOTSTRAP),
        ("markdown", """## Shared models, comparators and controls

        PU-Joint and the full retained benchmark suite use the same split, observed masks,
        paired-reference budget and candidate ceiling. This family additionally enables
        GenEML-adapted, Inductive-PU-MC and SAR-PU. The `-adapted` label is retained until the
        Python 3 port is validated as implementation-equivalent to the historical GenEML code."""),
        ("code", SETTINGS),
        ("markdown", """## CPU allocation and live workers

        The worker display distinguishes requested/effective experiment workers from detected
        machine capacity. Numerical libraries remain single-threaded inside each outer worker."""),
        ("code", _common.WORKER_STATUS),
        ("markdown", "## Define outcome-blind panel and split audits"),
        ("code", PREFLIGHT),
    ]
    parts.extend(_dataset_parts(name))
    if name != "simulation":
        parts.extend([
            ("markdown", """## Run every predeclared measurement condition

            This is the expensive cell. Compatible fold/model checkpoints resume automatically.
            Duplicate rounded coverage conditions are fitted once and remain declared in the
            panel audit; failures stay in the exported grid rather than triggering a new mask."""),
            ("code", guarded(f'''
                artifacts = run_measurement_experiment(
                    dataset=dataset,
                    settings=settings,
                    config=measurement_config,
                    checkpoint_dir=CHECKPOINT_DIR,
                    export_dir=EXPORT_DIR,
                    progress=SHOW_PROGRESS,
                    progress_level=PROGRESS_LEVEL,
                    progress_interval=PROGRESS_INTERVAL_SECONDS,
                    worker_status=worker_status,
                )
                all_artifacts = {{{label!r}: artifacts}}
            ''')),
        ])
    parts.extend([
        ("markdown", "## Figure helpers and fixed evaluation scope"),
        ("code", PLOT_HELPERS),
        ("markdown", """## Positive-retention degradation curves

        Both panels use native-reference test entries. The key view names the three established
        PU models and the three new comparators; the full view retains every fitted benchmark.
        All five retention levels are shown and simulation is separated by sharing strength."""),
        ("code", RETENTION_PLOTS),
        ("markdown", """## Joint gene-coverage × target-coverage heatmap

        One baseline is selected using development validation loss over the whole grid, then
        frozen. Positive color values mean `Brier(fixed baseline) - Brier(PU-MIRT) > 0` and
        therefore favor PU-MIRT. No baseline is selected cell-by-cell or from test metrics."""),
        ("code", HEATMAP_PLOTS),
        ("markdown", (
            "## Amplification-confirmed recovery at fixed budgets\n\n"
            "Only the natural Projection-TAGs condition enters this plot. Recall is evaluated "
            "among standard-negative, amplification-confirmed positives at top 1%, 5%, 10% "
            "and 20% ranked-cell budgets."
            if name == "Projection_TAGs" else
            "## Accuracy versus cross-panel prediction sensitivity\n\n"
            "Macro-AUPRC is paired with the same-cell prediction change across independently "
            "drawn gene panels at full target coverage and 50% positive retention. The plot "
            "selects the realized anchor gene coverage before comparing models. Lower sensitivity "
            "means more stable predictions. This table uses fixed native-reference pairs and "
            "never treats mask repetitions as animals."
        )),
        ("code", third_plot),
        ("markdown", """## Essential metrics, selected parameters and diagnostics

        The visible report includes exact panel/support audits, important-model metrics,
        PU-MIRT/PU-Joint/new-comparator selected settings, candidate counts, detector/propensity
        checks, convergence/retry/boundary evidence and every failure count. Complete tuning,
        per-target, per-group, reliability, checkpoint and prediction records remain exported;
        no failed grid cell is silently omitted."""),
        ("code", RESULT_AUDIT),
        ("markdown", "## Display bounded shared and measurement-specific reports"),
        ("code", DIAGNOSTICS),
    ])
    return {
        "cells": [
            _common.cell(kind, source, index, date_suffix)
            for index, (kind, source) in enumerate(parts)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (OnDemand)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "gene2wire": {
                "protocol": PROTOCOL_VERSION,
                "core_commit": commit,
                "source_hash": source_hash,
                "dataset": name,
                "notebook_date": date_suffix,
                "environment": "OnDemand",
                "experiment": "measurement_degradation",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(commit: str, source_hash: str, output_dir=OUTPUT_DIR, date=None):
    date_suffix = _common.release_date(date)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in NAMES:
        destination = output_dir / NOTEBOOK_FILENAMES[name]
        destination.write_text(
            json.dumps(
                notebook(name, commit, source_hash, date_suffix),
                indent=1,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        paths.append(destination)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--date", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    for path in build(args.commit, args.source_hash, args.output_dir, args.date):
        print(path)


if __name__ == "__main__":
    main()
