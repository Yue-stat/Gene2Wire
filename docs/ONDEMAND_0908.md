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
N_REPETITIONS = 5
STRATEGY = 'full_joint'
```

Choose a compute allocation that supports the requested worker count and memory.
`N_JOBS=32` is a maximum: for a real dataset, three folds times five repetitions
provide at most 15 concurrent fold/repetition units. BARseq processes A1 and M1
separately. The simulation supplies more independent units. Reducing `N_JOBS`
does not redraw masks or invalidate compatible fitted checkpoints. Threads inside
each worker are limited to one; avoid importing NumPy before the configuration
cell applies those thread settings.

The shared profile exposes the declared information-budget, RF, mechanism,
calibration, and optional Qiao controls. Default mechanism/calibration stress
tests are simulation-only; their switches do not add arbitrary new missingness
axes to every real dataset. The complete matrix and search budgets are documented
in [PROTOCOL_0908.md](PROTOCOL_0908.md).

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
these are not postsynaptic gene expression. The optional Qiao comparison runs in
simulation only, with `RUN_QIAO=True` and `USE_TARGET_FEATURES=True` together.

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
| Fewer workers than `N_JOBS` | Expected when fewer fold/repetition units are available |

## 7. Distinguish verification from formal results

`python -m pytest -q` exercises small local fixtures, including offline loading,
splits, leakage boundaries, gradients, resume, and notebook structure. It does not
download the full datasets or execute all formal reruns. All revised paper values
must come from completed 0908 runs whose manifests and failure/convergence
diagnostics have been reviewed. Mask repetitions are not new animals; simulation
folds within one generated dataset are not independent Monte Carlo samples.
