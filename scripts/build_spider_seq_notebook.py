"""Build the OnDemand SPIDER-Seq native-panel notebook using the shared core."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_notebook_common", ROOT / "scripts/build_notebooks_0908.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

CONFIG = '''
from pathlib import Path
import os

N_OUTER_FOLDS = 3
N_REPETITIONS = 5
N_JOBS = 32
PARALLEL_UNIT = 'scenario'
USE_LOCATION = False
USE_TARGET_FEATURES = False
STRATEGY = 'full_joint'
CANDIDATE_BUDGET = 32
SEED = 20260910

# Shared expression preprocessing; every model receives exactly the same features.
# Both variable-gene selection and PCA are fitted on training cells only.
N_HVG = 2000  # Train-only variance selection from 26,902 genes; try 1000/5000 as a sensitivity check.
N_GENE_COMPONENTS = 50  # Train-only PCA; None uses N_HVG genes directly and is much slower.
LOCATION_FEATURES_CSV = None  # No native cell location in this scRNA-seq object.
TARGET_FEATURES_CSV = None    # Outcome-independent descriptors, indexed by exact target IDs.
RUN_RANDOM_FOREST = True
RUN_QIAO = True  # Target-ID baseline unless USE_TARGET_FEATURES=True.

# Native panels only: no synthetic target blocks, positive thinning, or paired references.
SHOW_FULL_DIAGNOSTICS = False
SHOW_PROGRESS = True
PROGRESS_INTERVAL_SECONDS = 60.0
EXPORT_CELL_FORECASTS = True  # Compressed CSV means/SDs; per-fold NPZs are always retained.

BASE_DIR = Path('/home/yueyue/gene2wire').expanduser()
RAW_DATA_DIR = BASE_DIR / 'raw_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'native_panels_v2'
EXPORT_DIR = BASE_DIR / 'paper_figure_exports'
FIGURE_DIR = BASE_DIR / 'figures' / '__DATE__' / 'SPIDER_Seq_native_panels'
CODE_CACHE_DIR = BASE_DIR / 'code'
RESULTS_ONLY = False
EXISTING_EXPORT_DIRS = {'SPIDER-Seq': None}  # Exact completed run directory.

CORE_COMMIT = '__COMMIT__'
EXPECTED_SOURCE_HASH = '__HASH__'
REPO_URL = 'https://github.com/Yue-stat/Gene2Wire.git'
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
'''

SETTINGS = '''
import numpy as np
import pandas as pd
from IPython.display import display
from gene2wire.experiments.native_panel import (
    native_panel_tables, run_native_panel_experiment, summarize_native_predictions)
from gene2wire.experiments.native_panel_plotting import plot_native_panel_results
from gene2wire.experiments.reporting import (
    configure_compact_display, configure_full_display, display_diagnostics, load_existing_exports)
from gene2wire.seeds import stable_seed

if SHOW_FULL_DIAGNOSTICS:
    configure_full_display()
else:
    configure_compact_display()
settings = Settings(
    supervision_profile='assay_only', paired_fraction=0., calibration_fractions=(),
    loss_rates=(0.,), n_outer_folds=N_OUTER_FOLDS, n_repetitions=N_REPETITIONS,
    n_jobs=N_JOBS, parallel_unit=PARALLEL_UNIT, seed=SEED,
    use_location=USE_LOCATION, use_target_features=USE_TARGET_FEATURES,
    strategy=STRATEGY, candidate_budget=CANDIDATE_BUDGET,
    run_random_forest=RUN_RANDOM_FOREST, run_qiao=RUN_QIAO,
    run_information_controls=False, run_mechanism_controls=False, run_calibration_controls=False)
print({'models': [model.name for model in settings.models()],
       'RF-observed': RUN_RANDOM_FOREST, 'Qiao': RUN_QIAO,
       'folds': N_OUTER_FOLDS, 'repetitions': N_REPETITIONS, 'N_JOBS': N_JOBS,
       'gene_features': {'training_variable_genes': N_HVG, 'training_PCs': N_GENE_COMPONENTS},
       'paired_references': 'none', 'artificial_masking': 'none'})
if RESULTS_ONLY:
    all_artifacts = load_existing_exports(EXISTING_EXPORT_DIRS, expected_labels=('SPIDER-Seq',))
    artifacts = all_artifacts['SPIDER-Seq']
    if artifacts.manifest.get('experiment') != 'native_panels':
        raise ValueError('Select the SPIDER-Seq native-panel export, not spatial/block results.')
    print('RESULTS_ONLY: saved results loaded; raw loading and fitting are skipped.')
'''

LOAD = '''
if not RESULTS_ONLY:
    from gene2wire.experiments.datasets.spider_seq import load_spider_seq
    dataset = load_spider_seq(
        RAW_DATA_DIR / 'SPIDER_Seq', n_hvg=N_HVG, n_gene_components=N_GENE_COMPONENTS,
        location_features_csv=LOCATION_FEATURES_CSV,
        target_features_csv=TARGET_FEATURES_CSV)
    audit = native_panel_tables(dataset)
    display(audit['native_panel'].groupby('animal', observed=True).agg(
        cells=('n_cells', 'first'), measured_targets=('measured', 'sum'),
        measured_pairs=('n_measured', 'sum'), assay_positive_pairs=('n_positive', 'sum')))
    panel = audit['native_panel'].pivot(index='animal', columns='target', values='measured')
    display(panel.astype(int))
    folds = dataset.split_builder(N_OUTER_FOLDS,
        stable_seed(SEED, 'native_panel_splits', dataset.name, 0))
    display(pd.DataFrame([{'fold': f.outer_fold, 'inner_train': len(f.train_rows),
        'validation': len(f.validation_rows), 'test': len(f.test_rows)} for f in folds]))
    print('RNA counts are used; author integrated embeddings are not predictor inputs.')
    print('W is defined by injection panels. An unmeasured entry is not a negative label.')
    print('All three animals occur in every split role; this is within-animal new-cell CV.')
'''


def notebook(commit, source_hash, date_suffix):
    parts = [
        ('markdown', '''# SPIDER-Seq: naturally overlapping target panels — OnDemand

        Adult1, Adult2 and Adult3 share a molecular input space and have 12, 14 and 16
        assayed targets (24 in the union). This is the scRNA-seq Adult.Ex object,
        distinct from the spatial SPIDER notebook. The injection-panel mask is native;
        no target blocks or positive labels are artificially hidden.

        The primary benchmark evaluates held-out cells on targets actually assayed
        in their animal. Naturally unassayed pairs receive separate, unvalidated
        forecasts. There is no measured full-panel control for those absent labels.

        Sources: [paper](https://doi.org/10.1093/nsr/nwag004), Supplementary Table S2,
        [author FigureS3.R](https://github.com/ZhengTiger/SPIDER-Seq/blob/047eb5aaa15087c241570745bd6df07882b0dd78/Supplementary%20Figures/FigureS3.R),
        and [processed Adult.Ex.rds](https://huggingface.co/spaces/TigerZheng/SPIDER-web/blob/22e37e94bc5a5ecda2185f2af4eae5cf49c6092a/data/Adult.Ex.rds).

        ## Configuration

        Run from a Python 3.10+ OnDemand kernel with the repository dependencies.
        Raw and processed data, model checkpoints and results persist under BASE_DIR.
        Each repetition makes new within-animal cell splits; repetitions are not
        independent animals. Set RESULTS_ONLY for replotting saved results.'''),
        ('code', CONFIG.replace('__DATE__', date_suffix).replace('__COMMIT__', commit).replace('__HASH__', source_hash)),
        ('markdown', '## Load the pinned shared core'),
        ('code', "REQUIRED_MODULES = ('numpy', 'scipy', 'pandas', 'sklearn', 'joblib', 'matplotlib', 'yaml', 'rdata')\n" + common.BOOTSTRAP),
        ('markdown', '''## Model and evaluation settings

        No independent paired reference is available here. Models therefore predict
        the probability of an assay-positive call. Logistic, MIRT and Joint use the
        same core estimators and tuning rules as the other notebooks. Joint uses
        the native candidate budget for genuine shared-plus-specific candidates
        and carries the exact selected Logistic and MIRT configurations as two
        mandatory endpoints when those models are in the run. This prevents the
        bounded Joint grid from silently omitting the best standalone low-rank
        configuration. Endpoint fits are reused from the common caches. This
        does not assume perfect biological detection. RF and Qiao use the same
        observed labels and feature budget.'''),
        ('code', SETTINGS),
        ('markdown', '''## Kernel CPU allowance and experiment workers

        CPU affinity, scheduler allocation and cgroup quota are reported separately
        from running experiment workers. The pool can use at most the number of
        independently scheduled fold/repetition tasks.'''),
        ('code', common.WORKER_STATUS),
        ('markdown', '''## Load cached RNA counts and audit the real animal panels

        The raw RDS is checksum pinned. Its RNA counts and processed barcode calls
        are cached in sparse, pickle-free files after the first read. Adult.Ex's
        excitatory cohort is retained without selecting cells by barcode positivity.
        Positive calls use the author's processed values >0; raw-count thresholds
        are not applied a second time. Unmeasured targets stay outside all losses.

        Variable genes and PCA are selected/fitted within the training split and
        refitted on development cells after model selection. Spatial coordinates
        and target descriptors require independently supplied aligned CSVs when
        their switches are enabled.'''),
        ('code', LOAD),
        ('markdown', '''## Train with native panels and repeated within-animal CV

        Each cell is held out once per repetition. All models receive the same
        cells, features, assay mask and validation observations. Joint searches
        include the exact independently tuned direct and low-rank winners as
        inherited endpoints. `CANDIDATE_BUDGET` counts only genuine Joint
        candidates; the maximum Joint selection set is therefore `B + 2` when
        both standalone endpoints are available. Compatible complete and partial
        checkpoints resume automatically. Feature preparation and model
        fitting report separately in Los Angeles time. With the defaults,
        5 repetitions x 3 folds x 2 fit roles gives 30 feature sets before the
        model-unit progress denominator begins.'''),
        ('code', '''if not RESULTS_ONLY:
    print('Preparing training-only features and running the shared native-panel benchmark...')
    artifacts = run_native_panel_experiment(
        dataset, settings, checkpoint_dir=CHECKPOINT_DIR, export_dir=EXPORT_DIR,
        progress=SHOW_PROGRESS, progress_interval=PROGRESS_INTERVAL_SECONDS,
        worker_status=worker_status, export_cell_forecasts=EXPORT_CELL_FORECASTS)
    all_artifacts = {'SPIDER-Seq': artifacts}'''),
        ('markdown', '''## Audit the Joint candidate budget and inherited endpoints

        This compact audit is useful for interpreting a Joint result. The native
        candidate budget is the number of newly evaluated genuine Joint fits;
        `inherited_endpoint` rows are exact standalone winners carried into the
        same validation comparison and normally loaded from the shared candidate
        cache. Complete trial records remain in `tuning.csv`.'''),
        ('code', '''if not RESULTS_ONLY:
    tuning_audit = artifacts.tables.get('tuning', pd.DataFrame()).copy()
    if not tuning_audit.empty:
        joint_audit = tuning_audit.loc[tuning_audit['model'].eq('Joint')].copy()
        if not joint_audit.empty:
            endpoint_summary = (joint_audit.assign(
                endpoint=joint_audit['stage'].eq('inherited_endpoint'),
                genuine_joint=joint_audit['kind'].eq('joint'))
                .groupby(['repetition', 'outer_fold'], observed=True)
                [['endpoint', 'genuine_joint']].sum().reset_index())
            endpoint_summary['max_selectable'] = endpoint_summary['endpoint'] + endpoint_summary['genuine_joint']
            display(endpoint_summary)
            print({'native_joint_budget': CANDIDATE_BUDGET,
                   'expected_inherited_endpoints': 2,
                   'expected_max_selectable': CANDIDATE_BUDGET + 2})
    else:
        print('No tuning table available in RESULTS_ONLY mode; inspect tuning.csv.')'''),
        ('markdown', '''## Recover observed-panel summaries and unassayed forecasts

        This step can run from saved predictions alone. Measured-test metrics and
        per-animal/target scores use W=1 only. W=0 forecasts carry missing reference
        labels and evaluation_eligible=False; their variation across repeated splits
        is not a prediction interval. Compressed per-cell CSVs summarize repeated
        out-of-fold forecasts; every original fold prediction remains in units/.'''),
        ('code', '''if RESULTS_ONLY:
    artifacts = summarize_native_predictions(artifacts, export_cell_forecasts=EXPORT_CELL_FORECASTS)
print('Run exports:', artifacts.export_dir)
print('Unassayed forecasts:', artifacts.export_dir / 'native_unassayed')'''),
        ('markdown', '''## Display figures and save PDF files

        The measured-panel figures report actual held-out assay outcomes. The
        unassayed heatmaps are prediction-only summaries with no reference validation.
        No artificial loss-rate axis or full-panel control is introduced.'''),
        ('code', '''figure_paths = plot_native_panel_results(artifacts, output_dir=FIGURE_DIR, show=True)
display(figure_paths)'''),
        ('markdown', '''## Essential metrics, selected hyperparameters and convergence

        Compact diagnostics retain aggregate and per-animal metrics, important model
        selections by repetition/fold and validation-selection evidence. Complete
        metrics, per-target values, tuning records and forecasts stay in the export
        directory. SHOW_FULL_DIAGNOSTICS=True opts into much larger output.'''),
        ('code', '''display_diagnostics(artifacts, label='SPIDER-Seq native panels', full=SHOW_FULL_DIAGNOSTICS)
print('PDF figures:', FIGURE_DIR)
print('Raw and processed cache:', RAW_DATA_DIR / 'SPIDER_Seq')
print('Resumable checkpoints:', CHECKPOINT_DIR)'''),
    ]
    return {'cells': [common.cell(kind, source, i, date_suffix) for i, (kind, source) in enumerate(parts)],
        'metadata': {'kernelspec': {'display_name': 'Python 3 (OnDemand)', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10'},
            'gene2wire': {'core_commit': commit, 'source_hash': source_hash,
                'notebook_date': date_suffix, 'dataset': 'SPIDER-Seq', 'experiment': 'native_panels'}},
        'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--source-hash', required=True)
    parser.add_argument('--date', default=None)
    args = parser.parse_args()
    date = common.release_date(args.date)
    path = ROOT / f'SPIDER_Seq_natural_panels_{date}.ipynb'
    path.write_text(json.dumps(notebook(args.commit, args.source_hash, date), indent=1, ensure_ascii=False)+'\n')
    print(path)
