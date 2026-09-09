# Running the 0908 notebooks on OnDemand

The five root notebooks use one package and one experiment protocol. They contain
no Google Drive mount, package installation, notebook-local model patch, or
standalone plotting-script invocation. Plots display in the notebook and save
as PDF. The default output root is `/home/yueyue/gene2wire`.

## 1. Select or prepare a kernel once

If an existing OnDemand Python 3.10+ kernel already contains the dependencies in
the `experiments` and `notebooks` extras, select it. The bootstrap reports missing
modules before data loading. Do not install into a read-only system environment.

To create a separate user-owned environment, use a terminal on a node permitted
to install packages. Copy the released notebook's exact `CORE_COMMIT` into the
variable below; do not replace it with `main`:

```bash
CORE_COMMIT='paste_the_40_character_commit_from_the_notebook'
python3 -m venv /home/yueyue/gene2wire/venvs/0908
/home/yueyue/gene2wire/venvs/0908/bin/python -m pip install --upgrade pip
/home/yueyue/gene2wire/venvs/0908/bin/python -m pip install \
  "gene2wire[experiments,notebooks] @ git+https://github.com/Yue-stat/Gene2Wire.git@${CORE_COMMIT}"
/home/yueyue/gene2wire/venvs/0908/bin/python -m ipykernel install \
  --user --name gene2wire-0908 --display-name "Gene2Wire 0908"
```

Restart or refresh the OnDemand Jupyter session if its kernel menu has not updated,
then select **Gene2Wire 0908**. `rdata==1.1.0` reads the sparse R objects used by
SPIDER and Projection-TAGs; `openpyxl` reads Projection-TAGs' animal barcode map.
The notebooks do not need `pyreadr` or a Google Colab runtime.

If installing from a local released checkout instead of GitHub, activate the
same user-owned environment and run `python -m pip install ".[experiments,notebooks]"`
there. A matching local source checkout lets bootstrap skip its code download.

## 2. Configure the notebook before Run All

All five notebooks expose:

```python
N_OUTER_FOLDS = 3
USE_LOCATION = False
USE_TARGET_FEATURES = False
N_JOBS = 32
PARALLEL_UNIT = 'scenario'
N_REPETITIONS = 5
STRATEGY = 'full_joint'
PAIRED_FRACTION = 0.20
CALIBRATION_FRACTIONS = (PAIRED_FRACTION,)
```

Choose a compute allocation that supports the requested worker count and memory.
`N_JOBS=32` is a maximum. Default `PARALLEL_UNIT='scenario'` schedules three folds
× five repetitions × five rates = 75 tasks for each thinning benchmark, allowing
all 32 workers. BARseq processes A1 and M1 separately. Simulation additionally
spans sharing strengths and its declared calibration/mechanism settings. Models
and candidates remain sequential within each scenario to share fitting caches.
Natural Projection-TAGs has only one setting, so five repetitions provide 15
tasks (one repetition provides three). `PARALLEL_UNIT='fold'` groups a fold's
scenarios into one task. Neither scheduling switch redraws masks nor invalidates
compatible fitted checkpoints. The manifest records actual workers and pending
tasks; requested workers still need matching allocated CPU/memory. Threads inside
each worker are limited to one; avoid importing NumPy before the configuration
cell applies those thread settings.

The shared profile exposes the declared information-budget, RF, mechanism,
calibration, and Qiao controls. Default mechanism/calibration stress
tests are simulation-only; their switches do not add arbitrary new missingness
axes to every real dataset. The complete matrix and search budgets are documented
in [PROTOCOL_0908.md](PROTOCOL_0908.md).

Every code cell has a Markdown section heading visible in the notebook table of
contents. The first configuration section exposes the paired-reference variables.
By default simulation uses only the 20% paired size, including its detector
misspecification controls. Change `CALIBRATION_FRACTIONS` to `(0.10, 0.20, 0.40)`
to restore the size comparison, or change `PAIRED_FRACTION` for the primary budget.

For simulation, `SHARING_STRENGTHS = (0.0, 0.5, 1.0)` remains the default.
`SIMULATION_OPTIONS['truth_uses_location'] = USE_LOCATION` keeps generated truth
and predictor inputs consistent. To study omitted location deliberately, change
that truth setting explicitly and treat it as a separate scientific configuration.

## 3. Use the feature inputs supported by each dataset

Native location exists for SPIDER and BARseq. Projection-TAGs' native location is
the coarse source tissue origin, not a physical coordinate. Its optional external
CSV replaces that source-origin column. MERGE provides no physical location;
pseudotime is not substituted for it.

Before enabling external features, set the corresponding notebook variable:

```python
LOCATION_FEATURES_CSV = Path('/home/yueyue/gene2wire/features/cell_locations.csv')
TARGET_FEATURES_CSV = Path('/home/yueyue/gene2wire/features/target_descriptors.csv')
```

Only notebooks that accept the corresponding input expose that variable. The
first CSV column contains cell IDs for location or target IDs for target features;
remaining columns contain finite numeric covariates. Row order can differ, because
the loader explicitly aligns IDs. Required IDs must be unique and present. Features
must be measured/designed independently of projection outcomes, not derived from
held-out labels or target-positive rates.

SPIDER, MERGE, and Projection-TAGs require external target descriptors to enable
`USE_TARGET_FEATURES`. BARseq has optional label-derived anatomical descriptors;
these are not postsynaptic gene expression. `RUN_QIAO=True` is the default for
all datasets. With target features off, it runs explicitly labelled
`Qiao-ID-squared` / `Qiao-ID-logit` adapters using known target IDs. With the switch
on, `Qiao-squared` / `Qiao-logit` receive the same declared target descriptors as
the other models.

The preflight cell reports input dimensions, feature blocks, split group IDs, and
candidate counts before the expensive run cell. Inspect it after changing inputs
or folds. An unavailable feature produces an explicit error, not an ignored switch.

## 4. Cache code and raw data persistently

The first uncached run downloads the notebook's pinned code and each dataset's
official sources. Keep these locations between OnDemand sessions:

| Notebook | Raw cache |
|---|---|
| Simulation | `/home/yueyue/gene2wire/raw_data/simulation/` |
| Projection-TAGs | `/home/yueyue/gene2wire/raw_data/Projection_TAGs/` |
| SPIDER | `/home/yueyue/gene2wire/raw_data/SPIDER/` |
| MERGE-seq | `/home/yueyue/gene2wire/raw_data/MERGE_seq/` |
| BARseq | `/home/yueyue/gene2wire/raw_data/BARseq/` |

If compute nodes have no outbound access, run the bootstrap and data-loading cells
once on a permitted node using the same persistent paths, then use the populated
cache from the compute session. The loader verifies complete cached files without
contacting upstream metadata services. Do not copy interrupted `.part` downloads
as complete raw files.

MERGE also accepts existing downloads through `MERGESEQ_METADATA_PATH` and
`MERGESEQ_GEO_MATRIX_DIR`, or the loader's `raw_paths` argument. The metadata file
is checked against the audited Git blob. Projection-TAGs' `raw_paths` argument
accepts `expression`, `standard`, `union`, and `animals`; those official files
remain checksum-verified. The adapters record source hashes and cohort counts.

Processed expression caches avoid reparsing large RDS/Matrix Market objects.
Raw-cache validation detects changed bytes. When a processed cache is damaged,
its explicit error identifies the processed file(s) to remove and rebuild from
the retained raw source. Do not bypass checksum validation or overwrite historical
results to make a changed input appear identical.

## 5. Resume and inspect the outputs

After a disconnect, select the same kernel, restart it, and Run All with the same
scientific configuration and persistent paths. Compatible completed candidates
and models are restored. Source or scientific changes create distinct identities;
old-format checkpoints are not automatically valid for this protocol.

The shared run returns an `artifacts` object and prints its exact export directory:

```text
/home/yueyue/gene2wire/paper_figure_exports/<dataset>_0908/<run_id>/
```

Use the path printed by the notebook; dataset names are normalized for directory
names. Inspect `manifest.json` and `failures.csv` alongside the metrics. A run that
finishes with failed calibration units is not a complete evidence set, and those
units must not disappear from reporting.

Exports include wide and long metric tables, per-target values, tuning trials,
selected configurations, detection diagnostics, reliability bins, and per-unit
audits. Prediction NPZs contain aligned cell/target IDs, assayed masks, observed and
reference labels, estimated sensitivity, and available ranking/probability scores.
Keep these files to recompute later figures without retraining models.

The figure cell displays plots and writes PDF files to
`/home/yueyue/gene2wire/figures/0908/`. It retains the configured 0–80% curves for
artificial thinning; Projection-TAGs instead displays its natural paired audit
and model comparisons. Both BARseq panels and all three simulation sharing
strengths remain represented. No PNG export is requested by these notebooks.

## 6. Common runtime messages

| Message | Action |
|---|---|
| Missing dependency | Select the prepared kernel or install the extras once in the user-owned environment |
| Different/unverified `gene2wire` already imported | Restart the kernel and run configuration/bootstrap before other imports |
| Source checksum or code-pin mismatch | Use the released notebook and its matching checkout; source edits require a new version |
| Raw or processed cache checksum mismatch | Preserve the changed file for diagnosis; rebuild the identified cache from its verified source |
| No native location/target descriptors | Supply the aligned CSV or leave that feature switch off |
| No paired reference positives | Inspect the recorded calibration failure; do not borrow validation/test references or silently resample |
| Fewer workers than `N_JOBS` | Expected when fewer pending scenario tasks are available; natural data have one scenario |

## 7. Distinguish verification from formal results

`python -m pytest -q` exercises small local fixtures, including offline loading,
splits, leakage boundaries, gradients, resume, and notebook structure. It does not
download the full datasets or execute all formal reruns. All revised paper values
must come from completed 0908 runs whose manifests and failure/convergence
diagnostics have been reviewed. Mask repetitions are not new animals; simulation
folds within one generated dataset are not independent Monte Carlo samples.

## 8. September 9 diagnostics and existing results

All five notebooks retain their `_0908` names. Their updated core includes the
balanced candidate-selection correction described in `PROTOCOL_0908.md`. Source
and protocol fingerprints change, so earlier fitted checkpoints are retained
but are not silently counted as compatible with this corrected search.

To display the already completed BARseq runs attached on September 9, set these
variables in the first configuration cell, restart the kernel, and Run All:

```python
RESULTS_ONLY = True
EXISTING_EXPORT_DIRS = {
    'A1': BASE_DIR / 'paper_figure_exports/BARseq_A1_0908/2186ba71a14924d025ce',
    'M1': BASE_DIR / 'paper_figure_exports/BARseq_M1_0908/7c01c270e1dd788c8c0d',
}
```

The explicit run directories must exist on that machine. This mode reads their
saved manifests and all CSVs, displays the existing plots plus new plots of all
available benchmarks, adds the same-information-budget plot, and prints concise diagnostics. It skips raw-data loading,
preflight and training. The loaded run's saved protocol remains authoritative.
Every retained benchmark actually present in a curve setting is plotted; both
Prevalence methods are omitted and reference-only logistic is restored.
Default plotting skips horizontal model dots for natural or single-rate scopes.
Projection-TAGs retains its paired audit; its model results remain in tables.
Information-budget plots show independent, MIRT and Joint comparisons as separate
curve rows, only when at least two compared arms were actually run.
Unexecuted methods are not invented.
Historical endpoint-only baselines use disconnected markers. Curve methods show their
actual solid/dashed line and marker in the legend; endpoint-only methods retain
marker-only legend entries. Missing intermediate fits are never interpolated. Additional
PDFs are saved in `figures/0908/all_benchmarks/`; no PNGs are written.

New primary runs evaluate all 15 retained methods at every configured loss rate
(default 0%, 20%, 40%, 60%, 80%). `RF-mixed` uses reference labels on the same
20% paired subset and observed labels on the other cells, with each entry used
once. It performs ordinary RF fitting, not PU correction. Three RF arms share
the candidate grid, seed schedule and validation-label budget. Dashed curves
use neither PU nor paired references; solid curves use at least one. This
includes solid `Reference+PU`, `RF-reference` and `RF-mixed` curves. Qiao uses
observed labels and remains dashed, with target ID one-hot inputs by default.
Set `USE_TARGET_FEATURES=True` to supply the declared target descriptors.

Reference-only logistic and `Reference+PU-MIRT` / `Reference+PU-Joint` now join
every primary rate. All reference arms use the same paired IDs; the three mixed
models use reference loss on C and PU loss on O without duplicating C. These
new controls require fitting; their outcomes cannot be inferred from old plots.

RF fit predictions are cached by actual training/prediction arrays, seed,
configuration and source, across loss rates. Identical paired-only fits are
computed once on POSIX systems using advisory locks; each scenario still scores
candidates on its own observed validation labels and estimated sensitivities.

For training, use `RESULTS_ONLY = False`. New progress switches are:

```python
SHOW_PROGRESS = True
PROGRESS_LEVEL = 'summary'     # 'model' / 'trial' opt in to verbose event output
PROGRESS_INTERVAL_SECONDS = 60.0
SHOW_FULL_DIAGNOSTICS = False
```

Default progress prints one initial cache summary, one compact line per minute,
and one completion summary. For example:

```text
[running 1min] finished units 82/1125, current time 2026-09-09 10:42:00 PDT
```

One unit is one model evaluation at a specified repetition, fold and scenario:
its candidate search and final refit together. The denominator includes all
planned primary, information-budget, RF and Qiao models, including
simulation control scenarios. Optimizer iterations and individual candidates
are not separate units in this counter. A changed loss rate or calibration
condition is a different scenario. Failed calibration is reported separately
and never counted as a completed model evaluation. Los Angeles local time uses
`America/Los_Angeles`, including daylight-saving transitions.

The initial inventory counts model evaluations contained in fully verified
scenario result checkpoints. Unprocessed scenarios can still reuse candidate
and model checkpoints; their precise hits are recorded as each identity becomes
available, without printing a line for every hit. The startup pending count means
work still to process, not necessarily unfitted models. Completed unit summaries are
accepted only when their exported predictions and audits pass content checks.
Worker events remain under `progress/` and in `progress_events.csv`;
`checkpoint_inventory.csv` retains the initial counts.

The final notebook report shows primary endpoint metrics for every recorded
method, a comparison of the retained logistic and RF supervision controls,
selected-configuration frequencies, convergence and candidate coverage. It shows
only a small sample of failure details with complete failure counts. Full tuning,
per-target, calibration and reliability tables remain in the CSV exports; set
`SHOW_FULL_DIAGNOSTICS=True` to display them all. Existing exports can be replotted
without fitting. New RF-mixed results and previously missing intermediate-rate
fits require training; loading older exports does not fabricate unrun methods.
