"""Build canonical OnDemand group-by-target block experiment notebooks.

No model, mask, or fitting implementation lives in these entry points.
Usage: python scripts/notebooks/build_target_block_masking.py \
    --commit SHA --source-hash SHA256 [--date MMDD]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "notebooks" / "target_block_masking"
_spec = importlib.util.spec_from_file_location(
    "_gene2wire_notebook_common", Path(__file__).with_name("build_positive_label_hiding.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
NAMES = ("BARseq_A1", "BARseq_M1", "Projection_TAGs", "MERGE_seq",
         "simulation", "SPIDER", "SPIDER_Seq")
NOTEBOOK_FILENAMES = {
    "BARseq_A1": "barseq_a1.ipynb",
    "BARseq_M1": "barseq_m1.ipynb",
    "Projection_TAGs": "projection_tags.ipynb",
    "MERGE_seq": "merge_seq.ipynb",
    "simulation": "simulation.ipynb",
    "SPIDER": "spider_spatial.ipynb",
    "SPIDER_Seq": "spider_seq.ipynb",
}


CONFIG = '''
from pathlib import Path
import os

N_OUTER_FOLDS = 3
USE_LOCATION = False
USE_TARGET_FEATURES = False
N_JOBS = 32
PARALLEL_UNIT = 'scenario'
N_REPETITIONS = 5
STRATEGY = 'full_joint'
CANDIDATE_BUDGET = 32
SEED = 20260909
PAIRED_FRACTION = 0.20
SUPERVISION_PROFILE = '__SUPERVISION_PROFILE__'

# Fraction of eligible TARGETS receiving a partial group panel, not positive loss.
BLOCK_FRACTIONS = (0.20, 0.40, 0.60, 0.80)
GROUP_MODE = '__GROUP_MODE__'
N_ARTIFICIAL_GROUPS = 2
VALIDATION_FRACTION = 0.20  # Fraction of each group's development cells.
INCLUDE_FULL_PANEL_CONTROL = True

# Isolate structural missingness by default. Optional: (0., .2, .4, .6, .8).
# Nonzero rates use SCAR with matched draws across masked/full training panels.
# Projection-TAGs always retains its natural observed assay and ignores this grid.
POSITIVE_LOSS_RATES = (0.0,)
RUN_INFORMATION_CONTROLS = True
RUN_RANDOM_FOREST = True
RUN_QIAO = True
SHOW_PROGRESS = True
PROGRESS_LEVEL = 'summary'
PROGRESS_INTERVAL_SECONDS = 60.0
SHOW_FULL_DIAGNOSTICS = False  # True opts into every raw table; output can be large.

BASE_DIR = Path(os.environ.get(
    'GENE2WIRE_PROJECT_DIR', '/home/yueyue/gene2wire')).expanduser()
RAW_DATA_DIR = BASE_DIR / 'raw_data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'block_masking_v1'
EXPORT_DIR = BASE_DIR / 'paper_figure_exports'
FIGURE_DIR = BASE_DIR / 'figures' / '__DATE__'
CODE_CACHE_DIR = BASE_DIR / 'code'
RESULTS_ONLY = False
EXISTING_EXPORT_DIRS = __EXPORT_DIRS__  # Exact completed run directories.
CORE_COMMIT = '__COMMIT__'
EXPECTED_SOURCE_HASH = '__HASH__'
REPO_URL = 'https://github.com/Yue-stat/Gene2Wire.git'
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
'''


SETTINGS = '''
from dataclasses import asdict, replace
import numpy as np
import pandas as pd
from IPython.display import display
from gene2wire.experiments.block_design import BlockMaskConfig
from gene2wire.experiments.block_experiment import (
    run_block_experiment, run_block_simulation_experiments, preview_block_experiment,
)
from gene2wire.experiments.block_plotting import plot_block_results
from gene2wire.experiments.reporting import (
    configure_full_display, configure_compact_display, load_existing_exports, display_diagnostics,
)
from gene2wire.tuning import full_joint_candidates

if SHOW_FULL_DIAGNOSTICS:
    configure_full_display()
else:
    configure_compact_display()
settings = Settings(
    n_outer_folds=N_OUTER_FOLDS, use_location=USE_LOCATION,
    use_target_features=USE_TARGET_FEATURES, n_jobs=N_JOBS,
    parallel_unit=PARALLEL_UNIT, n_repetitions=N_REPETITIONS,
    strategy=STRATEGY, candidate_budget=CANDIDATE_BUDGET, seed=SEED,
    paired_fraction=0.0 if SUPERVISION_PROFILE == 'assay_only' else PAIRED_FRACTION,
    calibration_fractions=() if SUPERVISION_PROFILE == 'assay_only' else (PAIRED_FRACTION,),
    loss_rates=POSITIVE_LOSS_RATES,
    run_information_controls=(RUN_INFORMATION_CONTROLS and SUPERVISION_PROFILE != 'assay_only'),
    run_random_forest=RUN_RANDOM_FOREST, run_qiao=RUN_QIAO,
    run_mechanism_controls=False, run_calibration_controls=False,
    supervision_profile=SUPERVISION_PROFILE,
)
block_config = BlockMaskConfig(
    fractions=BLOCK_FRACTIONS, group_mode=GROUP_MODE,
    n_artificial_groups=N_ARTIFICIAL_GROUPS,
    validation_fraction=VALIDATION_FRACTION,
    include_full_panel_control=INCLUDE_FULL_PANEL_CONTROL,
)
if SHOW_FULL_DIAGNOSTICS:
    display(pd.DataFrame([asdict(settings)]).T.rename(columns={0: 'shared settings'}))
    display(pd.DataFrame([asdict(block_config)]).T.rename(columns={0: 'block design'}))
else:
    print({'folds': N_OUTER_FOLDS, 'repetitions': N_REPETITIONS, 'n_jobs': N_JOBS,
           'use_location': USE_LOCATION, 'use_target_features': USE_TARGET_FEATURES,
           'paired_fraction': settings.paired_fraction, 'supervision_profile': SUPERVISION_PROFILE,
           'block_fractions': BLOCK_FRACTIONS,
           'positive_loss_rates': POSITIVE_LOSS_RATES, 'group_mode': GROUP_MODE,
           'full_panel_control': INCLUDE_FULL_PANEL_CONTROL})
if RESULTS_ONLY:
    all_artifacts = load_existing_exports(
        EXISTING_EXPORT_DIRS, expected_labels=EXPECTED_EXPORT_LABELS)
    print('RESULTS_ONLY: using saved manifests; raw loading and fitting are skipped.')
'''


PREFLIGHT = '''
def preflight_blocks(dataset):
    dataset.validate()
    preview = preview_block_experiment(dataset, settings, block_config, repetition=0)
    groups, folds, design = preview['groups'], preview['folds'], preview['design']
    features = dataset.feature_builder(
        folds[0].train_rows, settings.use_location, settings.use_target_features)
    display(pd.DataFrame([{
        'dataset': dataset.name, 'cells': len(dataset.cell_ids),
        'targets': len(dataset.target_ids), 'groups': len(np.unique(groups)),
        'feature_columns': features.X.shape[1],
        'feature_blocks': dict(features.feature_blocks),
        'target_feature_columns': 0 if features.Y_target is None else features.Y_target.shape[1],
        'native_measured_pairs': int(dataset.measured.sum()),
        'natural_paired_assay': dataset.natural_observed is not None,
    }]))
    display(pd.Series(groups).value_counts().rename_axis('group').to_frame('cells'))
    print('Repetition 0 design; actual fractions account for rounded target counts and native coverage:')
    summary_columns = ['block_fraction', 'n_eligible_targets', 'n_selected_targets',
                       'realized_eligible_target_fraction', 'n_hidden_pairs',
                       'realized_pair_fraction', 'duplicate_of_fraction']
    display(design.summary if SHOW_FULL_DIAGNOSTICS else design.summary[summary_columns])
    display(pd.DataFrame([
        {'fold': f.outer_fold, 'inner_train': len(f.train_rows),
         'validation': len(f.validation_rows), 'test': len(f.test_rows),
         'split': f.metadata['split_type']} for f in folds]))
    if SHOW_FULL_DIAGNOSTICS:
        display(design.assignment)
    tuning = settings.tuning_config(features.X.shape[1], len(dataset.target_ids))
    candidates = []
    for model in settings.models():
        if model.kind == 'joint' and tuning.include_endpoints:
            positive_ranks = tuple(rank for rank in tuning.ranks if rank > 0)
            native_tuning = replace(tuning, ranks=positive_ranks,
                                    include_endpoints=False) if positive_ranks else None
            native = () if native_tuning is None else full_joint_candidates(model, native_tuning)
            candidates.extend({'model': model.name, 'candidate': index,
                               'candidate_role': 'genuine_joint', **asdict(candidate)}
                              for index, candidate in enumerate(native, 1))
            candidates.extend([{'model': model.name, 'candidate': pd.NA,
                                'candidate_role': 'inherited_endpoint', 'kind': kind,
                                'rank': pd.NA, 'shared_l2': pd.NA, 'residual_l2': pd.NA,
                                'use_target_features': model.use_target_features,
                                'target_l2': pd.NA, 'pu': model.pu}
                               for kind in ('direct', 'lowrank')])
        else:
            candidates.extend({'model': model.name, 'candidate': index,
                               'candidate_role': 'native', **asdict(candidate)}
                              for index, candidate in enumerate(full_joint_candidates(model, tuning), 1))
    candidate_table = pd.DataFrame(candidates)
    if SHOW_FULL_DIAGNOSTICS:
        display(candidate_table)
        display(pd.DataFrame([asdict(settings.fit_config())]))
    else:
        display(candidate_table.groupby('model').size().to_frame('candidates'))
    print('Joint rows marked inherited_endpoint are the two exact standalone winners; '
          'they are selected after the standalone fits and reuse their caches.')
    print('Blocked entries are excluded from calibration, model fitting and validation scoring.')
'''


DIAGNOSTICS = '''
for label, artifacts in all_artifacts.items():
    display_diagnostics(artifacts, label=label, full=SHOW_FULL_DIAGNOSTICS)
print('PDF figures:', FIGURE_DIR)
print('Persistent checkpoints:', CHECKPOINT_DIR)
print('Reusable raw data:', RAW_DATA_DIR)
'''


def guarded(source):
    return "if not RESULTS_ONLY:\n" + textwrap.indent(textwrap.dedent(source).strip(), "    ")


def notebook(name, commit, source_hash, date_suffix=None):
    if name not in NAMES:
        raise ValueError(f"Unknown block notebook: {name}")
    date_suffix = _common.release_date(date_suffix)
    group_mode = ("sample" if name == "MERGE_seq" else
                  "artificial" if name in {"BARseq_M1", "simulation", "SPIDER"}
                  else "animal")
    label = {"BARseq_A1": "A1", "BARseq_M1": "M1",
             "Projection_TAGs": "Projection-TAGs", "MERGE_seq": "MERGE-seq",
             "simulation": "simulation", "SPIDER": "SPIDER",
             "SPIDER_Seq": "SPIDER-Seq"}[name]
    modules = ["numpy", "scipy", "pandas", "sklearn", "joblib", "threadpoolctl",
               "matplotlib", "yaml", "IPython"]
    if name == "Projection_TAGs":
        modules += ["rdata", "openpyxl"]
    elif name == "SPIDER_Seq":
        modules.append("rdata")
    elif name == "SPIDER":
        modules.append("rdata")
    config = (CONFIG.replace("__GROUP_MODE__", group_mode)
              .replace("__SUPERVISION_PROFILE__", "assay_only" if name == "SPIDER_Seq" else "paired_reference")
              .replace("__DATE__", date_suffix).replace("__COMMIT__", commit)
              .replace("__HASH__", source_hash).replace("__EXPORT_DIRS__", repr({label: None})))
    config += f"\nREQUIRED_MODULES = {modules!r}\nEXPECTED_EXPORT_LABELS = {(label,)!r}\n"
    description = {
        "BARseq_A1": "A1 uses its two biological animals as groups. This is a within-animal new-cell evaluation, not leave-one-animal-out validation.",
        "BARseq_M1": "M1 contains one biological animal. Two balanced, seeded artificial groups test the controlled panel-missingness mechanism; they are not independent animals.",
        "Projection_TAGs": "Projection-TAGs uses recorded animal IDs and preserves the native assay mask. Only targets measured in at least two animals are eligible. Standard detections remain natural; the union reference is an imperfect evaluation reference.",
        "MERGE_seq": "MERGE-seq uses its four recorded experimental sample IDs as groups. This is a within-sample new-cell evaluation under partial target panels, not leave-one-sample-out validation.",
        "simulation": "Each repetition generates independent data for each sharing strength. Two balanced artificial groups allow known low-rank structure to be tested under partial target panels.",
        "SPIDER": "The current processed SPIDER adapter exposes slices but no verified animal IDs. Two balanced artificial groups test partial target panels while retaining all input genes. These groups are not animals, and slices are not relabelled as animals.",
        "SPIDER_Seq": "SPIDER-Seq Adult1/2/3 provide verified biological animal IDs and partially overlapping native target panels. Only targets measured in at least two animals are eligible for a held-out block.",
    }[name]
    parts = [
        ("markdown", f"""# Gene2Wire {name}: group × target blocks — OnDemand {date_suffix}

        {description}

        All existing input gene features are retained. A hidden group × target block contains
        both positive and negative assay entries; none becomes a training negative.
        The selected targets retain measured training support in other groups, and unselected
        eligible targets remain shared. The same mask applies to every cell in a group.

        Cells are split **within each group** into training, validation and outer test sets.
        Only hidden entries on held-out test cells are scored. Their original references stay
        outside preprocessing, calibration, fitting and selection. The full-panel control uses
        the same test cells, features, paired cell IDs and tuning rules, but restores native
        training/validation availability; it is fitted once and scored on each masked test subset.

        This is a separate experiment from the primary 0908/0909 runs. It tests missing target
        blocks for new cells within represented groups; it does not test new targets or unseen
        animals. Artificial repetitions are not biological replication. No advantage is assumed.

        Run top to bottom in an OnDemand Python 3.10+ kernel. Raw data and checkpoints persist;
        exports remain under `/home/yueyue/gene2wire/paper_figure_exports`. Figures display here
        and save as PDF only. Every code cell has a table-of-contents section.
        To redraw completed results, set `RESULTS_ONLY=True` and supply exact saved run directories.
        Compact diagnostics are the default: key aggregate metrics, selected configurations,
        convergence and control comparisons stay visible, while raw tables remain exported.
        Set `SHOW_FULL_DIAGNOSTICS=True` only when complete output is needed.
        Progress reports once per minute; the worker cell reports CPU allowance and allocation
        metadata separately from this experiment's active workers."""),
        ("markdown", "## Run configuration and masking strength"), ("code", config),
        ("markdown", "## Load and verify the frozen shared core"), ("code", _common.BOOTSTRAP),
        ("markdown", """## Shared model and block settings

        All datasets use the same model implementations, candidate budgets and exact Joint
        endpoints. Only group source, native availability and feature adapters differ.
        Paired references authorize visible entries of the same selected cells, never blocked
        entries. Nonzero optional positive loss uses simple SCAR, separately from structural masks.
        Qiao uses target identity when target features are disabled; enabling the switch requires
        the adapter's declared outcome-independent target descriptors."""), ("code", SETTINGS),
        ("markdown", """## CPU allocation and live experiment workers

        Reports the kernel CPU allowance and detectable machine, affinity, scheduler and cgroup
        allocation limits. This experiment's active workers are shown separately. Other users'
        idle CPUs on the node cannot be inferred from this notebook."""),
        ("code", _common.WORKER_STATUS),
        ("markdown", "## Preview design and tuning candidates"), ("code", PREFLIGHT),
    ]
    if name.startswith("BARseq"):
        panel = name.removeprefix("BARseq_")
        parts += [
            ("markdown", f"## Load cached BARseq {panel} inputs"),
            ("code", guarded(f"""
                from gene2wire.experiments.datasets.barseq import load_barseq
                dataset = load_barseq(RAW_DATA_DIR / 'BARseq')['{panel}']
                preflight_blocks(dataset)
            """)),
        ]
    elif name == "Projection_TAGs":
        parts += [
            ("markdown", "## Optional Projection-TAGs feature inputs"),
            ("code", "LOCATION_FEATURES_CSV = None\nTARGET_FEATURES_CSV = None"),
            ("markdown", "## Load cached Projection-TAGs inputs and native assay coverage"),
            ("code", guarded("""
                from gene2wire.experiments.datasets.projection_tags import load_projection_tags
                dataset = load_projection_tags(
                    RAW_DATA_DIR / 'Projection_TAGs',
                    location_features_csv=LOCATION_FEATURES_CSV,
                    target_features_csv=TARGET_FEATURES_CSV)
                preflight_blocks(dataset)
            """)),
        ]
    elif name == "SPIDER":
        parts += [
            ("markdown", """## SPIDER target features and artificial groups

            The original fully measured target panel is used. Location uses native spatial
            covariates when enabled. Target features require an aligned, outcome-independent
            numeric CSV. Artificial groups are balanced by cell IDs and seed; their allocation
            never uses expression, projections or cell type. This is a within-group new-cell
            experiment, distinct from the primary SPIDER spatial holdout."""),
            ("code", "TARGET_FEATURES_CSV = None"),
            ("markdown", "## Load cached SPIDER inputs and preview the panel masks"),
            ("code", guarded("""
                from gene2wire.experiments.datasets.spider import load_spider
                dataset = load_spider(
                    RAW_DATA_DIR / 'SPIDER', target_features_csv=TARGET_FEATURES_CSV)
                preflight_blocks(dataset)
            """)),
        ]
    elif name == "MERGE_seq":
        parts += [
            ("markdown", """## MERGE-seq sample groups and optional feature inputs

            The four recorded sample IDs define the natural groups. Native physical cell
            locations and outcome-independent target descriptors are not bundled; aligned
            CSVs are required before enabling either optional feature switch."""),
            ("code", "N_GENE_FEATURES = 128\nLOCATION_FEATURES_CSV = None\nTARGET_FEATURES_CSV = None"),
            ("markdown", "## Load cached MERGE-seq inputs and preview the panel masks"),
            ("code", guarded("""
                from gene2wire.experiments.datasets.merge_seq import load_merge_seq
                dataset = load_merge_seq(
                    RAW_DATA_DIR / 'MERGE_seq', n_gene_features=N_GENE_FEATURES,
                    location_features_csv=LOCATION_FEATURES_CSV,
                    target_features_csv=TARGET_FEATURES_CSV)
                preflight_blocks(dataset)
            """)),
        ]
    elif name == "SPIDER_Seq":
        parts += [
            ("markdown", """## SPIDER-Seq native animal panels

            Adult1, Adult2 and Adult3 use the verified partially overlapping target
            panels from the same SPIDER-Seq study. Only targets measured in at least
            two animals are eligible for block masking. The biological animal IDs are
            retained; no artificial groups are introduced."""),
            ("code", "LOCATION_FEATURES_CSV = None  # SPIDER-Seq has no bundled spatial coordinates.\n"
                     "TARGET_FEATURES_CSV = None"),
            ("markdown", "## Load cached SPIDER-Seq inputs and preview native panel masks"),
            ("code", guarded("""
                from gene2wire.experiments.datasets.spider_seq import load_spider_seq
                dataset = load_spider_seq(
                    RAW_DATA_DIR / 'SPIDER_Seq', n_hvg=2000, n_gene_components=50,
                    location_features_csv=LOCATION_FEATURES_CSV,
                    target_features_csv=TARGET_FEATURES_CSV)
                preflight_blocks(dataset)
            """)),
        ]
    else:
        parts += [
            ("markdown", """## Simulation truth and input settings

            The location switch controls both generated projection signal and fitted features.
            Sharing strengths 0, 0.5 and 1 are all retained. The paired subset defaults to 20%.
            Generated data are persisted and reused; the preview below is only a small design audit."""),
            ("code", guarded("""
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
                preflight_blocks(representative)
            """)),
        ]
    if name == "simulation":
        run = """
            artifacts = run_block_simulation_experiments(
                settings=settings, block_config=block_config,
                raw_cache_dir=RAW_DATA_DIR / 'simulation_block_masking',
                checkpoint_dir=CHECKPOINT_DIR, export_dir=EXPORT_DIR,
                sharing_strengths=SHARING_STRENGTHS,
                simulation_options=SIMULATION_OPTIONS,
                progress=SHOW_PROGRESS, progress_level=PROGRESS_LEVEL,
                progress_interval=PROGRESS_INTERVAL_SECONDS, worker_status=worker_status)
            all_artifacts = {'simulation': artifacts}
        """
    else:
        run = f"""
            artifacts = run_block_experiment(
                dataset=dataset, settings=settings, block_config=block_config,
                checkpoint_dir=CHECKPOINT_DIR, export_dir=EXPORT_DIR,
                progress=SHOW_PROGRESS, progress_level=PROGRESS_LEVEL,
                progress_interval=PROGRESS_INTERVAL_SECONDS, worker_status=worker_status)
            all_artifacts = {{{label!r}: artifacts}}
        """
    parts += [
        ("markdown", """## Run all block settings and the full-panel control

        This is the expensive step. A model unit includes its tuning and final refit.
        Compatible completed units and partial fit caches resume automatically. The full-panel
        control is not refitted for every evaluation fraction. At zero additional thinning,
        ordinary and PU objectives can coincide; overlapping curves are expected."""),
        ("code", guarded(run)),
        ("markdown", """## Curves on structurally hidden test entries

        Panels show AUPRC, log loss and Brier score against the fraction of eligible targets
        assigned a partial panel. The full-panel row uses the same hidden test subsets, so its
        curve can change as evaluation targets change even though its fitted model is reused.
        All scores use raw predictions; a missing assay is not an observed negative and receives
        no non-detection posterior. Thus Hidden Recall@H is not used in this experiment.
        Additional plots show all benchmarks, methods without direct reference supervision,
        and the three same-budget reference-plus-observation methods. A separate
        `extra_sharing_gain_auprc` plot shows masked-panel minus full-panel model advantage,
        with one standard error across repetitions after averaging folds. Real-data error bars
        describe repeated mask/split variation, not biological sampling uncertainty."""),
        ("code", "figure_paths = {}\nfor label, artifacts in all_artifacts.items():\n"
         "    figure_paths[label] = plot_block_results(\n"
         "        artifacts, output_dir=FIGURE_DIR, show=True, include_benchmarks=True)\n"
         "display(figure_paths)"),
        ("markdown", """## Essential results and selected hyperparameters

        The default report bounds output and retains aggregate metrics, important-model
        selections by repetition, convergence, calibration and matched block/control diagnostics.
        It states when rows are omitted. Every raw table remains exported;
        `SHOW_FULL_DIAGNOSTICS=True` explicitly opts into large output."""),
        ("code", DIAGNOSTICS),
    ]
    return {"cells": [_common.cell(kind, source, i, date_suffix)
                      for i, (kind, source) in enumerate(parts)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10"},
                         "gene2wire": {"core_commit": commit,
                                       "source_hash": source_hash,
                                       "notebook_date": date_suffix,
                                       "dataset": name,
                                       "protocol": _common.PROTOCOL_VERSION,
                                       "experiment": "target_block_masking"}},
            "nbformat": 4, "nbformat_minor": 5}


def build(commit, source_hash, output_dir=OUTPUT_DIR, date=None):
    date = _common.release_date(date)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in NAMES:
        path = output_dir / NOTEBOOK_FILENAMES[name]
        path.write_text(json.dumps(notebook(name, commit, source_hash, date),
                                   indent=1, ensure_ascii=False) + "\n")
        paths.append(path)
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
