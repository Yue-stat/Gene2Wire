"""Build results-only measurement-degradation V3 visualization notebooks.

V3 never modifies the canonical V2 notebooks or completed experiment exports.
Five notebooks point at exact completed run directories recovered from the
executed result notebooks supplied for this analysis.  SPIDER and SPIDER-Seq
remain safely blocked until their exact run directories are supplied.

Usage: python scripts/notebooks/build_measurement_degradation_v3.py \
    --commit <SHA> --source-hash <SHA256> [--date MMDD]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "notebooks" / "measurement_degradation_v3"

_base_spec = importlib.util.spec_from_file_location(
    "_gene2wire_measurement_v2_builder",
    Path(__file__).with_name("build_measurement_degradation.py"),
)
_base = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base)

NAMES = _base.NAMES
NOTEBOOK_FILENAMES = _base.NOTEBOOK_FILENAMES
PROTOCOL_VERSION = _base.PROTOCOL_VERSION
SOURCE_RESULTS_CORE_COMMIT = "5fce6cab3c7662256ad1f82d0576188e00389d07"
SOURCE_RESULTS_SOURCE_HASH = (
    "45011b9f2b618c53703a02fe6dcdcac70250b793caf9231299ef5e8735b39eaa"
)

EXACT_EXPORT_DIRS = {
    "simulation": "/home/yueyue/gene2wire/paper_figure_exports/"
    "simulation_measurement_degradation/a447d9aaa72f2f847fb9",
    "BARseq_A1": "/home/yueyue/gene2wire/paper_figure_exports/"
    "BARseq_A1_measurement_degradation/20561ae1ec481dbeab96",
    "BARseq_M1": "/home/yueyue/gene2wire/paper_figure_exports/"
    "BARseq_M1_measurement_degradation/ad4abecbc92743f8d04f",
    "MERGE_seq": "/home/yueyue/gene2wire/paper_figure_exports/"
    "MERGE-seq_measurement_degradation/d1b46506c668c646e54f",
    "Projection_TAGs": "/home/yueyue/gene2wire/paper_figure_exports/"
    "Projection-TAGs_measurement_degradation/58f93f29e60290169415",
    "SPIDER": None,
    "SPIDER_Seq": None,
}

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
    path = Path(__file__).with_name("measurement_v3_analysis.py")
    source = path.read_text(encoding="utf-8")
    source = source.replace("from __future__ import annotations\n\n", "", 1)
    marker = "\ndef notebook_source() -> str:\n"
    if marker not in source:
        raise RuntimeError(f"Could not locate notebook-source boundary in {path}")
    return source.split(marker, 1)[0].rstrip() + "\n"


RESULTS_VALIDATION = '''
for label, artifacts in all_artifacts.items():
    manifest = getattr(artifacts, 'manifest', {}) or {}
    if manifest.get('experiment') != 'measurement_degradation':
        raise RuntimeError(
            f'{label}: expected a measurement_degradation export, got '
            f'{manifest.get("experiment")!r}.')
    if manifest.get('completed') is not True:
        raise RuntimeError(f'{label}: source export is not marked completed.')
    if (EXPECTED_RESULTS_RUN_ID is not None
            and str(manifest.get('run_id')) != EXPECTED_RESULTS_RUN_ID):
        raise RuntimeError(
            f'{label}: export run ID {manifest.get("run_id")!r} does not match '
            f'the pinned V3 source result {EXPECTED_RESULTS_RUN_ID!r}.')
    if (EXPECTED_RESULTS_SOURCE_HASH is not None
            and manifest.get('source_hash') != EXPECTED_RESULTS_SOURCE_HASH):
        raise RuntimeError(
            f'{label}: source export hash {manifest.get("source_hash")!r} does not '
            f'match the pinned result hash {EXPECTED_RESULTS_SOURCE_HASH!r}.')
    saved_version = (manifest.get('protocol') or {}).get('protocol_version')
    if saved_version != PROTOCOL_VERSION:
        raise RuntimeError(
            f'{label}: export protocol {saved_version!r} does not match '
            f'{PROTOCOL_VERSION!r}.')
    saved_measurement = manifest.get('measurement_protocol') or {}
    expected_measurement = {
        'gene_coverages': GENE_COVERAGES,
        'target_coverages': TARGET_COVERAGES,
        'retentions': POSITIVE_RETENTIONS,
    }
    for coordinate, expected_values in expected_measurement.items():
        saved_values = saved_measurement.get(coordinate)
        if (saved_values is None or len(saved_values) != len(expected_values)
                or not np.allclose(
                    np.asarray(saved_values, dtype=float),
                    np.asarray(expected_values, dtype=float))):
            raise RuntimeError(
                f'{label}: export {coordinate}={saved_values!r} does not match '
                f'the V3 result contract {expected_values!r}.')
    expected_anchors = {
        'anchor_gene_coverage': ANCHOR_GENE_COVERAGE,
        'anchor_target_coverage': ANCHOR_TARGET_COVERAGE,
        'anchor_retention': ANCHOR_POSITIVE_RETENTION,
        'n_panel_seeds': N_PANEL_SEEDS,
    }
    for coordinate, expected_value in expected_anchors.items():
        saved_value = saved_measurement.get(coordinate)
        if saved_value is None or not np.isclose(float(saved_value), float(expected_value)):
            raise RuntimeError(
                f'{label}: export {coordinate}={saved_value!r} does not match '
                f'the V3 result contract {expected_value!r}.')
    if str(artifacts.export_dir.name) != str(manifest.get('run_id')):
        raise RuntimeError(
            f'{label}: export directory run ID and manifest run ID disagree.')
    required_tables = {'metrics', 'per_repetition', 'heatmap_baseline_selection'}
    if label == 'Projection-TAGs':
        required_tables.update({'projection_budget_recall',
                                'projection_budget_recall_per_target'})
    missing_tables = sorted(required_tables.difference(artifacts.tables))
    if missing_tables:
        raise RuntimeError(f'{label}: completed export lacks tables {missing_tables}.')
    print({
        'results_only_dataset': label,
        'read_only_export': str(artifacts.export_dir),
        'run_id': manifest.get('run_id'),
        'source_result_core_commit': SOURCE_RESULTS_CORE_COMMIT,
        'export_source_hash': manifest.get('source_hash'),
        'export_protocol_version': (manifest.get('protocol') or {}).get('protocol_version'),
        'v3_writes_only_to': str(FIGURE_DIR),
    })
'''


RETENTION_PLOTS_V3 = '''
retention_figure_paths = {}
for label, artifacts in all_artifacts.items():
    per_repetition = artifacts.tables.get('per_repetition', pd.DataFrame())
    retention = _role_rows(per_repetition, 'retention_curve')
    required = {'positive_retention', 'macro_auprc', 'macro_log_loss', 'model'}
    if retention.empty or not required.issubset(retention.columns):
        raise RuntimeError(f'{label}: retention metrics are missing or incomplete.')
    for column in ('positive_retention', 'macro_auprc', 'macro_log_loss'):
        retention[column] = pd.to_numeric(retention[column], errors='coerce')
    retention = retention.loc[
        np.isfinite(retention['positive_retention'])
        & np.isfinite(retention['macro_auprc'])
    ]
    if retention.empty:
        raise RuntimeError(f'{label}: no finite native-reference retention rows.')
    for group_label, group in _plot_groups(retention):
        for mode in ('key', 'full'):
            options = ({'key_models': KEY_MEASUREMENT_MODELS} if mode == 'key' else {})
            figure, axes = plot_degradation_pair(
                group,
                x_col='positive_retention',
                x_label='Positive retention',
                mode=mode,
                title=f'{label} ({group_label}) — {mode} models',
                **options,
            )
            suffix = f'{group_label}_retention_auprc_log_loss_{mode}_v3'
            retention_figure_paths[(label, group_label, mode)] = _save_show(
                figure, label, suffix)
display(retention_figure_paths)
'''


COVERAGE_PLOTS_V3 = '''
coverage_figure_paths = {}
for label, artifacts in all_artifacts.items():
    coverage = _role_rows(
        artifacts.tables.get('per_repetition', pd.DataFrame()), 'coverage_heatmap')
    required = {
        'gene_requested_coverage', 'gene_coverage',
        'target_requested_coverage', 'target_coverage',
        'positive_retention', 'macro_auprc', 'macro_log_loss', 'model',
    }
    if coverage.empty or not required.issubset(coverage.columns):
        raise RuntimeError(f'{label}: coverage-curve coordinates are missing or incomplete.')
    for column in required.difference({'model'}):
        coverage[column] = pd.to_numeric(coverage[column], errors='coerce')
    saved_protocol = getattr(artifacts, 'manifest', {}).get('measurement_protocol', {})
    saved_anchor_gene_coverage = float(saved_protocol.get(
        'anchor_gene_coverage', ANCHOR_GENE_COVERAGE))
    saved_anchor_target_coverage = float(saved_protocol.get(
        'anchor_target_coverage', ANCHOR_TARGET_COVERAGE))
    saved_anchor_retention = float(saved_protocol.get(
        'anchor_retention', ANCHOR_POSITIVE_RETENTION))
    common = coverage.loc[
        np.isclose(coverage['positive_retention'], saved_anchor_retention)
        & np.isfinite(coverage['macro_auprc'])
    ].copy()
    curve_specs = (
        (
            'gene',
            common.loc[np.isclose(
                common['target_requested_coverage'], saved_anchor_target_coverage)],
            'gene_coverage',
            f'Gene coverage (target anchor={saved_anchor_target_coverage:g})',
        ),
        (
            'target',
            common.loc[np.isclose(
                common['gene_requested_coverage'], saved_anchor_gene_coverage)],
            'target_coverage',
            f'Target coverage (gene anchor={saved_anchor_gene_coverage:g})',
        ),
    )
    for dimension, curve, coverage_col, subtitle in curve_specs:
        curve = curve.loc[np.isfinite(curve[coverage_col])]
        if curve.empty:
            raise RuntimeError(
                f'{label}: no finite {dimension}-coverage rows at the saved anchors.')
        for group_label, group in _plot_groups(curve):
            for mode in ('key', 'full'):
                options = ({'key_models': KEY_MEASUREMENT_MODELS} if mode == 'key' else {})
                figure, axes = plot_degradation_pair(
                    group,
                    x_col=coverage_col,
                    x_label=subtitle,
                    mode=mode,
                    title=(f'{label} ({group_label}) — {dimension} degradation; '
                           f'retention={saved_anchor_retention:g}; {mode} models'),
                    **options,
                )
                suffix = f'{group_label}_{dimension}_coverage_auprc_log_loss_{mode}_v3'
                coverage_figure_paths[(label, group_label, dimension, mode)] = _save_show(
                    figure, label, suffix)
display(coverage_figure_paths)
'''


HEATMAP_PLOTS_V3 = _base.HEATMAP_PLOTS.replace(
    "suffix = f'{group_label}_gene_target_brier_heatmap'",
    "# All degradation coordinates run from 100% at left toward 0% at right.\n"
    "        if axis.get_xlim()[0] < axis.get_xlim()[1]:\n"
    "            axis.invert_xaxis()\n"
    "        suffix = f'{group_label}_gene_target_brier_heatmap_v3'",
)


FUNKY_PLOTS_V3 = '''
funky_figure_paths = {}
projection_v3_tables = {}
for label, artifacts in all_artifacts.items():
    projection_natural = None
    if label == 'Projection-TAGs':
        projection_v3_tables[label] = derive_projection_natural_metrics(artifacts)
        projection_natural = projection_v3_tables[label]['summary']
    saved_protocol = getattr(artifacts, 'manifest', {}).get('measurement_protocol', {})
    if not saved_protocol:
        raise RuntimeError(f'{label}: manifest lacks measurement_protocol.')
    scorecard = prepare_measurement_funky_table(
        artifacts.tables.get('per_repetition', pd.DataFrame()),
        measurement_protocol=saved_protocol,
        projection_natural=projection_natural,
    )
    scorecard_path = _figure_path(label, 'funkyheatmap_scorecard_values_v3').with_suffix('.csv')
    scorecard.to_csv(scorecard_path, index=False)
    display(scorecard_path)
    for mode in ('key', 'full'):
        options = ({'key_models': KEY_MEASUREMENT_MODELS} if mode == 'key' else {})
        figure, axes = plot_measurement_funky_summary(
            scorecard,
            mode=mode,
            title=f'{label} — {mode} models (funkyheatmap-style)',
            **options,
        )
        funky_figure_paths[(label, mode)] = _save_show(
            figure, label, f'funkyheatmap_{mode}_v3')
display(funky_figure_paths)
'''


PROJECTION_PLOTS_V3 = '''
projection_v3_figure_paths = {}
for label, artifacts in all_artifacts.items():
    derived = projection_v3_tables.get(label)
    if derived is None:
        derived = derive_projection_natural_metrics(artifacts)
        projection_v3_tables[label] = derived
    for table_name, table in derived.items():
        path = _figure_path(label, f'projection_natural_{table_name}_v3').with_suffix('.csv')
        table.to_csv(path, index=False)
        display(path)
    print('Natural full-panel panel-seed copies were counted once only after '
          'complete fold/seed and exact-array identity checks:')
    display(derived['audit'])

    figure, axes = plot_projection_natural_comparison(
        derived['summary'],
        models=KEY_MEASUREMENT_MODELS,
        title=(f'{label} — standard-negative candidates; '
               'h=P(reference+ | standard−, X)'),
    )
    projection_v3_figure_paths[(label, 'candidate_metrics_key')] = _save_show(
        figure, label, 'amplification_candidate_metrics_key_v3')

    for mode, models in (
        ('key', KEY_MEASUREMENT_MODELS),
        ('full', FULL_MODEL_ORDER),
    ):
        figure, axes = plot_projection_budget_metrics(
            derived['budget'],
            models=models,
            title=(f'{label} — amplification-confirmed recovery and '
                   f'union-reference FP; {mode} models'),
        )
        projection_v3_figure_paths[(label, f'budget_metrics_{mode}')] = _save_show(
            figure, label, f'amplification_budget_metrics_{mode}_v3')

    ten_percent = derived['budget'].loc[np.isclose(
        pd.to_numeric(derived['budget']['budget_fraction'], errors='coerce'), 0.10
    )].copy()
    columns = [column for column in (
        'model', 'selected_candidates', 'recovered_positive_count',
        'amplification_reference_false_positive_count',
        'amplification_confirmed_recall',
        'amplification_confirmed_precision',
        'amplification_reference_fdr',
        'amplification_reference_fpr',
        'amplification_confirmed_lift',
        'micro_amplification_confirmed_recall',
        'micro_amplification_confirmed_precision',
        'micro_amplification_reference_fdr',
        'micro_amplification_reference_fpr',
        'micro_amplification_confirmed_lift',
    ) if column in ten_percent]
    print('10% budget. FP/FDR use the amplification union as assay reference; '
          'they are not proven biological false positives.')
    display(ten_percent.loc[ten_percent['model'].isin(KEY_MEASUREMENT_MODELS), columns])
display(projection_v3_figure_paths)
'''


def _replace_config(source: str, *, name: str, date_suffix: str) -> str:
    label = LABELS[name]
    old_mapping = repr({label: None})
    export_path = EXACT_EXPORT_DIRS[name]
    new_mapping = repr({label: export_path})
    result = source.replace(
        "FIGURE_DIR = BASE_DIR / 'figures' / 'measurement_degradation' / "
        f"'{date_suffix}'",
        "FIGURE_DIR = BASE_DIR / 'figures' / 'measurement_degradation_v3' / "
        f"'{date_suffix}'",
    )
    result = result.replace("RESULTS_ONLY = False", "RESULTS_ONLY = True")
    result = result.replace(
        f"EXISTING_EXPORT_DIRS = {old_mapping}",
        f"EXISTING_EXPORT_DIRS = {new_mapping}",
    )
    result_hash_line = (
        f"'{SOURCE_RESULTS_SOURCE_HASH}'" if export_path is not None else "None"
    )
    result_run_id_line = f"'{Path(export_path).name}'" if export_path is not None else "None"
    source_result_commit_line = (
        f"'{SOURCE_RESULTS_CORE_COMMIT}'" if export_path is not None else "None"
    )
    insertion = (
        f"SOURCE_RESULTS_CORE_COMMIT = {source_result_commit_line}\n"
        f"EXPECTED_RESULTS_SOURCE_HASH = {result_hash_line}\n"
        f"EXPECTED_RESULTS_RUN_ID = {result_run_id_line}\n"
        "V3_RESULTS_POLICY = 'read-only exact export; never train or mutate prior results'\n"
    )
    result = result.replace("CORE_COMMIT = '", insertion + "\nCORE_COMMIT = '", 1)
    if export_path is None:
        result += (
            "\n# No exact completed export was present in the supplied result notebooks.\n"
            "# Keep RESULTS_ONLY=True. To activate this notebook, provide the exact run-ID\n"
            "# directory plus its verified source core commit and source hash above; do not\n"
            "# switch to fitting merely to make this visualization notebook run.\n"
        )
    result += (
        "\nif RESULTS_ONLY is not True:\n"
        "    raise RuntimeError('V3 is results-only. Do not use it to fit or resume models.')\n"
    )
    if export_path is None:
        result += (
            "if (next(iter(EXISTING_EXPORT_DIRS.values())) is None\n"
            "        or SOURCE_RESULTS_CORE_COMMIT is None\n"
            "        or EXPECTED_RESULTS_SOURCE_HASH is None\n"
            "        or EXPECTED_RESULTS_RUN_ID is None):\n"
            "    raise RuntimeError(\n"
            "        'No verified completed source export was supplied. Pin its exact '\n"
            "        'directory, run ID, source hash, and core commit; V3 will not fit.')\n"
        )
    return result


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def notebook(name: str, commit: str, source_hash: str, date_suffix=None):
    if name not in NAMES:
        raise ValueError(f"Unknown measurement notebook: {name}")
    date_suffix = _base._common.release_date(date_suffix)
    book = _base.notebook(name, commit, source_hash, date_suffix)
    parts: list[tuple[str, str]] = []
    funky_inserted = False
    analysis_source = _analysis_source()
    for cell in book["cells"]:
        kind = cell["cell_type"]
        source = _source(cell)
        if kind == "markdown" and source.startswith("# Gene2Wire"):
            availability = (
                "The exact completed export is prefilled below."
                if EXACT_EXPORT_DIRS[name] is not None
                else "No exact completed export was supplied for this dataset; the notebook "
                     "fails closed until its run-ID directory is filled in."
            )
            source = source.replace(
                f"combined measurement degradation — OnDemand {date_suffix}",
                f"measurement degradation V3 — results-only {date_suffix}",
            )
            source += (
                "\n\nThis V3 notebook is a read-only visualization layer. It never trains, "
                "resumes checkpoints, or writes into the source export; experiment artifacts "
                "are written only below the separate V3 figure directory. " + availability
            )
        elif kind == "markdown" and source.startswith(
            "## Positive-retention degradation curves"
        ):
            source = (
                "## Positive-retention degradation curves\n\n"
                "Key and full-model views pair native-reference Macro-AUPRC with "
                "reference-probability Macro log loss. Both x axes run from 100% "
                "retention at the left toward 0% at the right; observed/mixed "
                "probability models show ranking metrics but are omitted from proper loss."
            )
        elif kind == "markdown" and source.startswith(
            "## Gene-coverage and target-coverage degradation curves"
        ):
            source = (
                "## Gene-coverage and target-coverage degradation curves\n\n"
                "For both dimensions, key and full-model views pair Macro-AUPRC with "
                "reference-probability Macro log loss. The saved manifest supplies the "
                "anchor conditions, realized integer coverage is plotted, and both x axes "
                "run from 100% at the left toward 0% at the right."
            )
        elif kind == "code" and "PROTOCOL_VERSION =" in source and "RESULTS_ONLY" in source:
            source = _replace_config(source, name=name, date_suffix=date_suffix)
        elif kind == "code" and "def notebook_source_hash" in source:
            source = source.replace(
                "for directory in (RAW_DATA_DIR, CHECKPOINT_DIR, EXPORT_DIR, FIGURE_DIR):\n"
                "    directory.mkdir(parents=True, exist_ok=True)",
                "FIGURE_DIR.mkdir(parents=True, exist_ok=True)",
            )
            source = source.replace(
                "'imported_from': gene2wire.__file__, 'raw_cache': str(RAW_DATA_DIR),\n"
                "       'checkpoints': str(CHECKPOINT_DIR), 'exports': str(EXPORT_DIR)})",
                "'imported_from': gene2wire.__file__,\n"
                "       'v3_figure_dir': str(FIGURE_DIR), 'results_only': True})",
            )
        elif kind == "code" and "from gene2wire import candidate_search_plan" in source:
            source = source.replace("from gene2wire import candidate_search_plan\n", "")
            source = source.replace(
                "from gene2wire.experiments.measurement_experiment import (\n"
                "    MeasurementConfig,\n"
                "    preview_measurement_experiment,\n"
                "    run_measurement_experiment,\n"
                "    run_measurement_simulation_experiments,\n"
                ")",
                "from gene2wire.experiments.measurement_experiment import MeasurementConfig",
            )
            source = source.replace(
                "from gene2wire.experiments.measurement_plotting import (\n"
                "    plot_accuracy_panel_sensitivity,\n"
                "    plot_coverage_auprc,\n"
                "    plot_gene_target_brier_heatmap,\n"
                "    plot_projection_tags_budget_recall,\n"
                "    plot_retention_auprc,\n"
                ")",
                "from gene2wire.experiments.measurement_plotting import (\n"
                "    plot_gene_target_brier_heatmap,\n"
                ")",
            )
            if "if RESULTS_ONLY:" in source and "load_existing_exports" in source:
                source = source.rstrip() + "\n\n" + RESULTS_VALIDATION.strip() + "\n"
        elif kind == "code" and "if RESULTS_ONLY:" in source and "load_existing_exports" in source:
            source = source.rstrip() + "\n\n" + RESULTS_VALIDATION.strip() + "\n"
        elif kind == "code" and "def _role_rows" in source and "def _save_show" in source:
            source = source.rstrip() + "\n\n" + analysis_source
        elif kind == "code" and "retention_figure_paths = {}" in source:
            source = RETENTION_PLOTS_V3
        elif kind == "code" and "coverage_figure_paths = {}" in source:
            source = COVERAGE_PLOTS_V3
        elif kind == "code" and "heatmap_figure_paths = {}" in source:
            source = HEATMAP_PLOTS_V3
        elif kind == "code" and "projection_budget_figure_paths = {}" in source:
            source = PROJECTION_PLOTS_V3
        elif kind == "markdown" and source.startswith(
            "## Accuracy versus cross-panel prediction sensitivity"
        ):
            # This is a diagnostic scatter, not a degradation coordinate, and
            # therefore cannot satisfy the V3 100%-to-0% + paired-loss grammar.
            # Its full table remains available in the read-only diagnostics.
            continue
        elif kind == "code" and "panel_stability_figure_paths = {}" in source:
            continue
        elif kind == "code" and "Resumable checkpoints:" in source:
            source = source.replace(
                "print('Reusable raw cache:', RAW_DATA_DIR)\n"
                "print('Resumable checkpoints:', CHECKPOINT_DIR)",
                "print('Read-only source exports:', {label: str(artifacts.export_dir) "
                "for label, artifacts in all_artifacts.items()})\n"
                "print('No V3 model fitting or checkpoint writes are permitted.')",
            )
        parts.append((kind, source))
        if kind == "code" and "heatmap_figure_paths = {}" in _source(cell):
            parts.extend([
                (
                    "markdown",
                    "## Funkyheatmap-style model scorecards\n\n"
                    "Following the [funkyheatmap visual grammar]"
                    "(https://funkyheatmap.github.io/funkyheatmap/), exactly two scorecards "
                    "are generated with repository-native Matplotlib: key models and all "
                    "fitted models. This keeps results-only execution independent of a new "
                    "R/Python package installation. "
                    "Rows are models; grouped columns are declared masking mechanism × degree × "
                    "metric. Circles encode AUPRC/AUROC and bars encode Brier/log loss; larger "
                    "and darker always means better. Proper losses are `NA` when a model does "
                    "not output reference probabilities. Projection-TAGs additionally includes "
                    "amplification-confirmed recovery among standard-negative candidates.",
                ),
                ("code", FUNKY_PLOTS_V3),
            ])
            funky_inserted = True
    if not funky_inserted:
        raise RuntimeError("Could not locate heatmap cell for V3 scorecard insertion")

    # Remove the entire preview/data-loading/fitting section. V3 contains no
    # executable model runner, so changing a display switch cannot start work.
    results_only_parts = []
    skip_training_section = False
    for kind, source in parts:
        if kind == "markdown" and source.startswith(
            "## Define outcome-blind panel and split audits"
        ):
            skip_training_section = True
            continue
        if kind == "markdown" and source.startswith(
            "## Figure helpers and fixed evaluation scope"
        ):
            skip_training_section = False
        if not skip_training_section:
            results_only_parts.append((kind, source))
    parts = results_only_parts

    if name == "Projection_TAGs":
        for index, (kind, source) in enumerate(parts):
            if kind == "markdown" and source.startswith(
                "## Amplification-confirmed recovery at fixed budgets"
            ):
                parts[index] = (
                    "markdown",
                    "## Amplification-confirmed recovery with reference false positives\n\n"
                    "The source predictions are reused without fitting. Reference-probability "
                    "models are ranked by `h=P(Z=1 | D=0, X)=(1-e)p/(1-ep)` and receive proper "
                    "candidate Brier/log-loss; observed/mixed-semantics models retain ranking "
                    "metrics but show `NA` for those proper scores. Selected union-negative "
                    "pairs are amplification-reference/unconfirmed false positives, not proven "
                    "biological false projections.\n\n"
                    "Natural recovery is a full-panel condition repeated by the old runner for "
                    "each panel seed. V3 requires every saved fold/seed, verifies the candidate "
                    "IDs, truth, scores, and probabilities are exactly identical, and counts one "
                    "canonical copy only. The audit table is displayed and exported; panel seeds "
                    "are not treated as independent natural-recovery repetitions.\n\n"
                    "**Why PU-Joint trails here:** this natural condition has 100% gene and target "
                    "coverage, so there is no structured panel-completion problem for Joint to "
                    "solve. Projection-TAGs has only seven targets and target features are off, "
                    "which leaves limited evidence for a shared low-rank component. In the supplied "
                    "run the learned detector is also close to a constant predictor, so explicit "
                    "censoring structure adds little; GenEML/SAR-PU can instead benefit from their "
                    "exposure-model parameterization. This is a mechanism-based interpretation, "
                    "not causal proof.\n\n"
                    "**Diagnostic only; not publication-final:** in this saved run, "
                    "GenEML/SAR-PU received artificial A/B/C exposure labels while Gene2Wire's "
                    "detector used the recorded platform. GenEML also had 136/144 candidate "
                    "and 2/6 final non-convergences; SAR-PU is the more credible comparator. "
                    "The real-platform comparator bug must be fixed and the natural condition "
                    "rerun before claiming a final ranking.",
                )

    book["cells"] = [
        _base._common.cell(kind, source, index, date_suffix)
        for index, (kind, source) in enumerate(parts)
    ]
    provenance = book["metadata"]["gene2wire"]
    provenance.update({
        "experiment": "measurement_degradation",
        "view": "measurement_degradation_v3_results_only",
        "source_result_core_commit": (
            SOURCE_RESULTS_CORE_COMMIT if EXACT_EXPORT_DIRS[name] is not None else None
        ),
        "source_result_hash": (
            SOURCE_RESULTS_SOURCE_HASH if EXACT_EXPORT_DIRS[name] is not None else None
        ),
        "source_export_dir": EXACT_EXPORT_DIRS[name],
    })
    return book


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
