"""Build runnable measurement-degradation V4 notebooks.

V4 keeps the established data adapters, masks, split discipline, checkpoint
resume, and source pinning from the canonical measurement-degradation family.
It narrows fitting to the six methods used in the requested comparisons and
replaces the legacy full-model/coverage-heatmap figures with configurable
multi-metric curves and a native ``funkyheatmappy`` scorecard.

Usage: python scripts/notebooks/build_measurement_degradation_v4.py \
    --commit <SHA> --source-hash <SHA256> [--date MMDD]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "notebooks" / "measurement_degradation_v4"

_base_spec = importlib.util.spec_from_file_location(
    "_gene2wire_measurement_v4_base",
    Path(__file__).with_name("build_measurement_degradation.py"),
)
_base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base)

NAMES = _base.NAMES
NOTEBOOK_FILENAMES = _base.NOTEBOOK_FILENAMES
PROTOCOL_VERSION = _base.PROTOCOL_VERSION

LABELS = {
    "simulation": "simulation",
    "BARseq_A1": "BARseq A1",
    "BARseq_M1": "BARseq M1",
    "MERGE_seq": "MERGE-seq",
    "Projection_TAGs": "Projection-TAGs",
    "SPIDER": "SPIDER",
    "SPIDER_Seq": "SPIDER-Seq",
}


def _analysis_source() -> str:
    """Embed the reviewed V4 plotting implementation in every notebook."""

    path = Path(__file__).with_name("measurement_v4_analysis.py")
    source = path.read_text(encoding="utf-8")
    source = source.replace("from __future__ import annotations\n\n", "", 1)
    marker = "\ndef notebook_source() -> str:\n"
    if marker not in source:
        raise RuntimeError(f"Could not locate notebook-source boundary in {path}")
    return source.split(marker, 1)[0].rstrip() + "\n"


CONFIG = '''
from pathlib import Path
import os

# Shared scientific defaults. Change switches here before Run All.
PROTOCOL_VERSION = '__PROTOCOL__'
N_OUTER_FOLDS = 3
USE_LOCATION = False
USE_TARGET_FEATURES = False
N_JOBS = 32
PARALLEL_UNIT = 'scenario'
N_REPETITIONS = 2
N_PANEL_SEEDS = 2
STRATEGY = 'rank_top2_total_ratio'
CANDIDATE_BUDGET = 32
SEED = 20260912
PAIRED_FRACTION = 0.20
CALIBRATION_FRACTIONS = (PAIRED_FRACTION,)

# Requested/theoretical measurement levels. Realized integer panel coverage is
# preserved in audit tables, but publication labels use these declared values.
GENE_COVERAGES = (1.0, 0.70, 0.40)
TARGET_COVERAGES = (1.0, 0.70, 0.40)
POSITIVE_RETENTIONS = (1.0, 0.80, 0.60, 0.40, 0.20)
ANCHOR_GENE_COVERAGE = 0.70
ANCHOR_TARGET_COVERAGE = 0.70
ANCHOR_POSITIVE_RETENTION = 0.40
VIRTUAL_ASSAYS = ('A', 'B', 'C')
HETEROGENEITY_DELTA = 1.0

# V4 fits only the three Gene2Wire-family estimators plus the three declared
# literature comparators. The allowlist is carried into every view and cache ID.
CORE_MODEL_ALLOWLIST = ('PU', 'PU-MIRT', 'PU-Joint')
PU_COMPARATOR_MODELS = ('GenEML-authors-mask', 'Inductive-PU-MC', 'SAR-PU')
RUN_INFORMATION_CONTROLS = False
RUN_RANDOM_FOREST = False
RUN_MECHANISM_CONTROLS = False
RUN_CALIBRATION_CONTROLS = False
RUN_QIAO = False
RUN_PU_COMPARATORS = True
INCLUDE_MATCHED_UNIFORM_CONTROL = False
INCLUDE_NATURAL_RECOVERY = False

SHOW_PROGRESS = True
PROGRESS_LEVEL = 'summary'
PROGRESS_INTERVAL_SECONDS = 60.0
SHOW_FULL_DIAGNOSTICS = False

BASE_DIR = Path(os.environ.get(
    'GENE2WIRE_PROJECT_DIR', '/home/yueyue/gene2wire')).expanduser()
RAW_DATA_DIR = BASE_DIR / 'raw_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'measurement_degradation_v4_compact'
EXPORT_DIR = BASE_DIR / 'paper_figure_exports'
FIGURE_DIR = BASE_DIR / 'figures' / 'measurement_degradation_v4' / '__DATE__'
CODE_CACHE_DIR = BASE_DIR / 'code'

# Read-only redraw mode. Replace None with the exact completed run directory.
RESULTS_ONLY = False
EXISTING_EXPORT_DIRS = __EXPORT_DIRS__

CORE_COMMIT = '__COMMIT__'
EXPECTED_SOURCE_HASH = '__HASH__'
REPO_URL = 'https://github.com/Yue-stat/Gene2Wire.git'

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
from gene2wire import candidate_search_plan
from gene2wire.experiments.measurement_experiment import (
    MeasurementConfig,
    preview_measurement_experiment,
    run_measurement_experiment,
    run_measurement_simulation_experiments,
)
from gene2wire.experiments.reporting import (
    configure_compact_display,
    configure_full_display,
    display_diagnostics,
    load_existing_exports,
)

if SHOW_FULL_DIAGNOSTICS:
    configure_full_display()
else:
    configure_compact_display()

settings = Settings(
    protocol_version=PROTOCOL_VERSION,
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
    schedule='v4',
    model_allowlist=CORE_MODEL_ALLOWLIST,
)
EXPECTED_INTERNAL_MODELS = (*CORE_MODEL_ALLOWLIST, *PU_COMPARATOR_MODELS)

if SHOW_FULL_DIAGNOSTICS:
    display(pd.DataFrame([asdict(settings)]).T.rename(columns={0: 'shared setting'}))
    display(pd.DataFrame([asdict(measurement_config)]).T.rename(
        columns={0: 'measurement design'}))
else:
    print({
        'folds': N_OUTER_FOLDS,
        'repetitions': N_REPETITIONS,
        'panel_seeds_per_dataset': N_PANEL_SEEDS,
        'models': EXPECTED_INTERNAL_MODELS,
        'gene_coverages': GENE_COVERAGES,
        'target_coverages': TARGET_COVERAGES,
        'positive_retentions': POSITIVE_RETENTIONS,
        'measurement_schedule': measurement_config.schedule,
    })

if RESULTS_ONLY:
    if any(path is None for path in EXISTING_EXPORT_DIRS.values()):
        raise RuntimeError(
            'RESULTS_ONLY=True requires the exact completed export directory '
            'for every requested dataset label.')
    all_artifacts = load_existing_exports(
        EXISTING_EXPORT_DIRS, expected_labels=EXPECTED_EXPORT_LABELS)
    print('RESULTS_ONLY: raw loading, preflight and fitting are skipped.')
'''


RESULTS_VALIDATION = '''
def _validate_v4_export_factorial(label, manifest, saved, per_repetition):
    required = {
        'repetition', 'evaluation_scope', 'mechanism',
        'gene_requested_coverage', 'target_requested_coverage',
        'positive_retention',
    }
    missing_columns = sorted(required.difference(per_repetition.columns))
    if missing_columns:
        raise RuntimeError(
            f'{label}: per_repetition.csv lacks V4 schedule columns '
            f'{missing_columns}.')
    relevant = per_repetition.loc[
        per_repetition['evaluation_scope'].astype(str).eq('native_reference')
        & per_repetition['mechanism'].astype(str).eq('assay_target_sar')
    ].copy()
    if relevant.empty:
        raise RuntimeError(
            f'{label}: no native-reference assay_target_sar results exist.')
    coordinates = (
        'gene_requested_coverage', 'target_requested_coverage',
        'positive_retention',
    )
    for column in ('repetition', *coordinates):
        relevant[column] = pd.to_numeric(relevant[column], errors='coerce')
    if relevant[['repetition', *coordinates]].isna().any().any():
        raise RuntimeError(
            f'{label}: V4 result coordinates contain missing/non-numeric values.')

    def key(values):
        return tuple(round(float(value), 12) for value in values)

    expected_conditions = {
        key((gene, target, retention))
        for gene in saved['gene_coverages']
        for target in saved['target_coverages']
        for retention in saved['retentions']
    }
    protocol = manifest.get('protocol') or {}
    expected_repetitions = int(protocol.get('n_repetitions', N_REPETITIONS))
    if expected_repetitions != N_REPETITIONS:
        raise RuntimeError(
            f'{label}: export has {expected_repetitions} repetitions, '
            f'expected {N_REPETITIONS}.')
    if int(saved.get('n_panel_seeds', -1)) != N_PANEL_SEEDS:
        raise RuntimeError(
            f"{label}: export n_panel_seeds={saved.get('n_panel_seeds')!r}, "
            f'expected {N_PANEL_SEEDS}.')

    sharing_strengths = manifest.get('simulation_sharing_strengths')
    if sharing_strengths is None:
        expected_units = [('all', None, repetition)
                          for repetition in range(N_PANEL_SEEDS)]
        finite_sharing = (
            pd.to_numeric(relevant['sharing_strength'], errors='coerce')
            if 'sharing_strength' in relevant else pd.Series(dtype=float)
        )
        if finite_sharing.notna().any():
            raise RuntimeError(
                f'{label}: real-data export unexpectedly mixes sharing facets.')
    else:
        if 'sharing_strength' not in relevant:
            raise RuntimeError(
                f'{label}: simulation export lacks sharing_strength.')
        relevant['sharing_strength'] = pd.to_numeric(
            relevant['sharing_strength'], errors='coerce')
        if relevant['sharing_strength'].isna().any():
            raise RuntimeError(
                f'{label}: simulation sharing_strength contains missing values.')
        expected_units = [
            (f'sharing={float(sharing):g}', float(sharing), repetition)
            for sharing in sharing_strengths
            for repetition in range(expected_repetitions)
        ]

    problems = []
    for unit_label, sharing, repetition in expected_units:
        selected = relevant.loc[np.isclose(relevant['repetition'], repetition)]
        if sharing is not None:
            selected = selected.loc[
                np.isclose(selected['sharing_strength'], sharing)]
        actual_conditions = {
            key(record)
            for record in selected[list(coordinates)].drop_duplicates().itertuples(
                index=False, name=None)
        }
        missing = sorted(expected_conditions.difference(actual_conditions))
        unexpected = sorted(actual_conditions.difference(expected_conditions))
        if missing or unexpected:
            problems.append(
                f'{unit_label}, repetition {repetition}: '
                f'missing {missing[:5]}'
                + (f' ({len(missing)} total)' if len(missing) > 5 else '')
                + f'; unexpected {unexpected[:5]}'
                + (f' ({len(unexpected)} total)' if len(unexpected) > 5 else '')
            )
    if problems:
        raise RuntimeError(
            f'{label}: missing declared V4 physical conditions or contains '
            'undeclared conditions: ' + ' | '.join(problems))


if RESULTS_ONLY:
    for label, artifacts in all_artifacts.items():
        manifest = getattr(artifacts, 'manifest', {}) or {}
        if manifest.get('experiment') != 'measurement_degradation':
            raise RuntimeError(f'{label}: not a measurement-degradation export.')
        if manifest.get('completed') is not True:
            raise RuntimeError(f'{label}: export is not marked completed.')
        if manifest.get('source_hash') != EXPECTED_SOURCE_HASH:
            raise RuntimeError(
                f'{label}: export source_hash={manifest.get("source_hash")!r}, '
                f'expected the pinned V4 source {EXPECTED_SOURCE_HASH!r}.')
        saved_protocol_version = (manifest.get('protocol') or {}).get(
            'protocol_version')
        if saved_protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f'{label}: export protocol_version={saved_protocol_version!r}, '
                f'expected {PROTOCOL_VERSION!r}.')
        saved = manifest.get('measurement_protocol') or {}
        expected_coordinates = {
            'gene_coverages': GENE_COVERAGES,
            'target_coverages': TARGET_COVERAGES,
            'retentions': POSITIVE_RETENTIONS,
        }
        for coordinate, expected in expected_coordinates.items():
            actual = saved.get(coordinate)
            if (actual is None or len(actual) != len(expected)
                    or not np.allclose(np.asarray(actual, dtype=float), expected)):
                raise RuntimeError(
                    f'{label}: export {coordinate}={actual!r}, expected {expected!r}.')
        if saved.get('schedule') != 'v4':
            raise RuntimeError(f'{label}: export does not use the V4 schedule.')
        if tuple(saved.get('model_allowlist') or ()) != CORE_MODEL_ALLOWLIST:
            raise RuntimeError(f'{label}: export does not use the V4 model allowlist.')
        per_repetition = artifacts.tables.get('per_repetition', pd.DataFrame())
        if per_repetition.empty:
            raise RuntimeError(f'{label}: completed export lacks per_repetition.csv.')
        _validate_v4_export_factorial(
            label, manifest, saved, per_repetition)
        available = set(per_repetition.get('model', pd.Series(dtype=str)).astype(str))
        unexpected = sorted(available.difference(EXPECTED_INTERNAL_MODELS))
        if unexpected:
            raise RuntimeError(f'{label}: export contains undeclared models {unexpected}.')
        missing = sorted(set(EXPECTED_INTERNAL_MODELS).difference(available))
        if missing:
            print(
                f'{label}: unavailable declared models will be shown as N/A: {missing}. '
                'No fallback model or partial-model funky normalization is used.'
            )
'''


PREFLIGHT = '''
def _panel_membership_summary(membership):
    measured = membership.loc[membership['measured'].astype(bool)].copy()
    return measured.groupby(
        ['requested_coverage', 'actual_coverage', 'assay'],
        dropna=False,
        observed=True,
    )['item'].agg(measured_items='size').reset_index()


def preflight_measurement(dataset):
    dataset.validate()
    preview = preview_measurement_experiment(
        dataset, settings, measurement_config, repetition=0)
    views, tables = preview['views'], preview['tables']
    if not views:
        raise RuntimeError('Measurement preview produced no experiment views.')
    for view in views:
        if tuple(view.metadata.get('model_allowlist') or ()) != CORE_MODEL_ALLOWLIST:
            raise RuntimeError('The V4 core-model allowlist was not propagated.')

    folds = tuple(dataset.split_builder(settings.n_outer_folds, settings.seed))
    features = views[0].feature_builder(
        folds[0].train_rows, settings.use_location, settings.use_target_features)
    display(pd.DataFrame([{
        'dataset': dataset.name,
        'cells': len(dataset.cell_ids),
        'targets': len(dataset.target_ids),
        'source_gene_pool': len(dataset.gene_names),
        'preview_views': len(views),
        'input_columns': features.X.shape[1],
        'models': ', '.join(EXPECTED_INTERNAL_MODELS),
    }]))

    for dimension, pool_size in (
        ('gene', len(dataset.gene_names)),
        ('target', len(dataset.target_ids)),
    ):
        conditions = tables[f'measurement_{dimension}_panel_conditions']
        full = conditions.loc[np.isclose(conditions['requested_coverage'], 1.0)]
        if len(full) != 1 or int(full.iloc[0]['panel_size']) != pool_size:
            raise RuntimeError(
                f'{dimension} coverage 1.0 must measure all {pool_size} items.')
        display(conditions)
        display(_panel_membership_summary(
            tables[f'measurement_{dimension}_panel_membership']))

    scenarios = tables['measurement_scenarios']
    expected_roles = (
        'positive_label_loss_only',
        'gene_coverage_only',
        'target_coverage_only',
        'combined_mask_curve',
    )
    for role in expected_roles:
        if not scenarios['condition_roles'].astype(str).str.contains(
                role, regex=False).any():
            raise RuntimeError(f'V4 preview lacks the {role!r} condition family.')
    if scenarios['condition_roles'].astype(str).str.contains(
            'coverage_heatmap|retention_curve', regex=True).any():
        raise RuntimeError('Legacy measurement roles leaked into the V4 schedule.')
    display(scenarios)
    display(tables['measurement_support'])
    display(tables['measurement_gene_target_support'])

    allowed = [
        model for model in settings.models()
        if model.name in CORE_MODEL_ALLOWLIST
    ]
    plans = []
    for model in allowed:
        plan = candidate_search_plan(
            model, settings.tuning_config(features.X.shape[1], len(dataset.target_ids)))
        plans.append({
            'model': model.name,
            'declared_native_budget': plan['declared_native_budget'],
            'maximum_native_candidates': plan['maximum_native_candidates'],
            'optimizer_starts_per_native_candidate': (
                plan['optimizer_starts_per_native_candidate']),
        })
    display(pd.DataFrame(plans))
    print('Only the six declared model IDs are scheduled.')
    print('Requested coverage labels remain theoretical; realized coverage stays in audits.')
    return preview
'''


FIGURE_HELPERS = '''
def _figure_path(label, suffix):
    safe = ''.join(
        character if character.isalnum() or character in '-_' else '_'
        for character in str(label)
    ).strip('_')
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


GENE2WIRE_CURVES = '''
PLOT_METRICS = ('auprc', 'auroc', 'log_loss', 'brier', 'hidden_recall')
PLOT_FAMILIES = (
    'positive_label_loss', 'gene_coverage',
    'target_coverage', 'combined_masks',
)

gene2wire_curve_paths = {}
for label, artifacts in all_artifacts.items():
    per_repetition = artifacts.tables.get('per_repetition', pd.DataFrame())
    for family in PLOT_FAMILIES:
        figure, axes = plot_v4_metric_grid(
            per_repetition,
            family=family,
            models=GENE2WIRE_FAMILY_MODELS,
            metrics=PLOT_METRICS,
            title=f'{label} — Gene2Wire family — {family}',
        )
        suffix = f'{family}_gene2wire_family_v4'
        gene2wire_curve_paths[(label, family)] = _save_show(
            figure, label, suffix)
display(gene2wire_curve_paths)
'''


EXTERNAL_CURVES = '''
PLOT_METRICS = ('auprc', 'auroc', 'log_loss', 'brier', 'hidden_recall')
PLOT_FAMILIES = (
    'positive_label_loss', 'gene_coverage',
    'target_coverage', 'combined_masks',
)

external_curve_paths = {}
for label, artifacts in all_artifacts.items():
    per_repetition = artifacts.tables.get('per_repetition', pd.DataFrame())
    for family in PLOT_FAMILIES:
        figure, axes = plot_v4_metric_grid(
            per_repetition,
            family=family,
            models=EXTERNAL_CURVE_MODELS,
            metrics=PLOT_METRICS,
            title=f'{label} — external PU comparisons — {family}',
        )
        suffix = f'{family}_external_comparators_v4'
        external_curve_paths[(label, family)] = _save_show(
            figure, label, suffix)
display(external_curve_paths)
'''


FUNKY_PLOT = '''
FUNKY_METRICS = ('auprc', 'auroc', 'log_loss', 'brier', 'hidden_recall')
FUNKY_FAMILIES = (
    'positive_label_loss', 'gene_coverage',
    'target_coverage',
)

funky_figure_paths = {}
for label, artifacts in all_artifacts.items():
    scorecard = prepare_v4_funky_table(
        artifacts.tables.get('per_repetition', pd.DataFrame()),
        metrics=FUNKY_METRICS,
        models=FUNKY_MODELS,
        families=FUNKY_FAMILIES,
    )
    scorecard_path = _figure_path(
        label, 'funkyheatmap_scorecard_values_v4').with_suffix('.csv')
    normalized_scorecard = normalize_v4_funky_table(scorecard)
    normalized_scorecard.to_csv(scorecard_path, index=False)
    display(scorecard_path)
    figure = plot_v4_funky_heatmap(
        scorecard,
        title=f'{label} — four-model measurement scorecard',
    )
    funky_figure_paths[label] = _save_show(
        figure, label, 'funkyheatmap_v4')
display(funky_figure_paths)
'''


RESULT_AUDIT = '''
def _bounded(frame, limit=80):
    if SHOW_FULL_DIAGNOSTICS or len(frame) <= limit:
        return frame
    print(f'Showing {limit}/{len(frame)} rows; complete data remain exported.')
    return frame.head(limit)


for label, artifacts in all_artifacts.items():
    print(f'\\n{label} — V4 design, implementation and failure audits')
    for table_name in (
        'measurement_scenarios',
        'measurement_support',
        'measurement_gene_target_support',
        'information_access',
        'selected',
        'tuning',
        'failures',
    ):
        table = artifacts.tables.get(table_name, pd.DataFrame())
        if not table.empty:
            if 'model' in table:
                table = table.loc[
                    table['model'].astype(str).isin(EXPECTED_INTERNAL_MODELS)]
            print(f'{table_name}: {len(table)} rows')
            display(_bounded(table))
    display_diagnostics(artifacts, label=label, full=SHOW_FULL_DIAGNOSTICS)

print('Figure PDFs and scorecard CSVs:', FIGURE_DIR)
print('Resumable V4 checkpoints:', CHECKPOINT_DIR)
'''


def guarded(source: str) -> str:
    return "if not RESULTS_ONLY:\n" + textwrap.indent(
        textwrap.dedent(source).strip(), "    "
    )


def _run_part(name: str, label: str) -> tuple[str, str]:
    if name == "simulation":
        # The base simulation part already contains the public simulation run.
        raise ValueError("Simulation run is supplied by the shared dataset parts")
    return (
        "code",
        guarded(
            f'''
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
            '''
        ),
    )


def notebook(name: str, commit: str, source_hash: str, date_suffix=None):
    if name not in NAMES:
        raise ValueError(f"Unknown measurement notebook: {name}")
    date_suffix = _base._common.release_date(date_suffix)
    label = LABELS[name]
    modules = [
        "numpy", "scipy", "pandas", "sklearn", "joblib", "threadpoolctl",
        "matplotlib", "yaml", "IPython", "funkyheatmappy",
    ]
    if name in {"Projection_TAGs", "SPIDER", "SPIDER_Seq"}:
        modules.append("rdata")
    if name == "Projection_TAGs":
        modules.append("openpyxl")
    config = (
        CONFIG.replace("__PROTOCOL__", PROTOCOL_VERSION)
        .replace("__DATE__", date_suffix)
        .replace("__COMMIT__", commit)
        .replace("__HASH__", source_hash)
        .replace("__EXPORT_DIRS__", repr({label: None}))
        + f"\nREQUIRED_MODULES = {modules!r}\n"
        + f"EXPECTED_EXPORT_LABELS = {(label,)!r}\n"
    )

    parts = [
        (
            "markdown",
            f"""# Gene2Wire {label}: measurement degradation V4 — OnDemand {date_suffix}

            V4 fits only PU, PU-MIRT, Gene2Wire, GenEML, PU matrix completion,
            and SAR-PU. Internal cache/export identifiers remain unchanged; plots
            publish PU-Joint as **Gene2Wire**. Four predeclared curve families
            separate positive-label loss, gene coverage, target coverage, and the
            combined retained-measurement harmonic mean. The first three are
            literal one-factor slices of the complete `3 x 3 x 5` factorial; the
            combined line uses every factorial condition. Publication labels use
            requested theoretical percentages; exact realized integer coverages
            remain visible in the audit tables.

            Run top to bottom in the repository environment. The notebook contains
            no model or masking implementation and pins the exact shared source.
            `RESULTS_ONLY=True` redraws only an explicitly supplied compatible V4
            export and never loads raw data or launches a fit.""",
        ),
        ("markdown", "## Run configuration and V4 measurement grid"),
        ("code", config),
        ("markdown", "## Load and verify the source-pinned shared core"),
        ("code", _base._common.BOOTSTRAP),
        (
            "markdown",
            """## Six-model fitting configuration

            SAR-PU calls the pinned authors' SAR-EM kernel through an adapted
            current-scikit-learn estimator and an outer per-target measurement-mask
            wrapper; the wrapper excludes `W=0` rows before each binary fit. This is
            not an execution of the authors' unmodified end-to-end program. Its
            outcome-blind propensity covariates use a separate per-target
            centered-orthonormal assay/QC design.
            GenEML is a pinned authors-source Python 3
            port, not an execution of the unmodified Python-2 authors' program.
            The minimal known-`W` exclusion and full-batch execution are explicit
            adaptations; the port retains the authors' global per-target exposure model. PU matrix
            completion is labelled paper-based until a verified authors' ShiftIMC
            solver is available. Exported selected/tuning rows record repository,
            revision, implementation variant and adaptation status; no backend
            silently falls back to a differently named estimator.

            All six models use the same splits, known measurement mask, observed
            labels and evaluation scope. Method-specific authorized inputs are
            disclosed; paired-calibration exposure is passed only to estimators
            designed to consume it. Hidden Recall@H ranks fitted-panel test entries
            with `D=0`, and its measured-entry, unlabeled-candidate and
            hidden-positive supports are exported with the metric.""",
        ),
        ("code", SETTINGS),
        ("markdown", "## Validate a completed results-only export"),
        ("code", RESULTS_VALIDATION),
        ("markdown", "## CPU allocation and live workers"),
        ("code", _base._common.WORKER_STATUS),
        ("markdown", "## Define outcome-blind V4 preflight audits"),
        ("code", PREFLIGHT),
    ]
    dataset_parts = _base._dataset_parts(name)
    if name == "Projection_TAGs":
        # V4 deliberately disables the legacy natural-recovery condition. Keep
        # the shared loader cell, but do not inherit prose claiming it is run.
        dataset_parts[0] = (
            "markdown",
            """## Projection-TAGs feature inputs

            V4 applies only the common artificial gene-panel, target-panel and
            positive-retention schedule. The source assay mask remains the fixed
            native-reference evaluation scope; no separate natural amplification-
            recovery condition is scheduled in this notebook.""",
        )
    parts.extend(dataset_parts)
    if name != "simulation":
        parts.extend(
            [
                (
                    "markdown",
                    "## Run the four predeclared V4 measurement curve families",
                ),
                _run_part(name, label),
            ]
        )
    parts.extend(
        [
            ("markdown", "## Source-embedded V4 plotting implementation"),
            ("code", _analysis_source()),
            ("markdown", "## Shared figure export helpers"),
            ("code", FIGURE_HELPERS),
            (
                "markdown",
                "## Multi-metric curves: PU, PU-MIRT and Gene2Wire",
            ),
            ("code", GENE2WIRE_CURVES),
            (
                "markdown",
                "## Multi-metric curves: Gene2Wire and external comparators",
            ),
            ("code", EXTERNAL_CURVES),
            (
                "markdown",
                "## Native funkyheatmap: four-model scorecard",
            ),
            ("code", FUNKY_PLOT),
            ("markdown", "## Essential V4 provenance and diagnostics"),
            ("code", RESULT_AUDIT),
        ]
    )
    return {
        "cells": [
            _base._common.cell(kind, source, index, date_suffix)
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
                "experiment": "measurement_degradation_v4",
                "view": "six_model_multimetric_funkyheatmap",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(commit: str, source_hash: str, output_dir=OUTPUT_DIR, date=None):
    date_suffix = _base._common.release_date(date)
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
            )
            + "\n",
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
    arguments = parser.parse_args()
    for path in build(
        arguments.commit,
        arguments.source_hash,
        arguments.output_dir,
        arguments.date,
    ):
        print(path)


if __name__ == "__main__":
    main()
