# Gene2Wire

Gene2Wire predicts neuronal projection outcomes from measured cellular features
under incomplete detection. The repository contains one shared implementation
of independent logistic, low-rank MIRT, and shared-plus-specific Joint models,
with matched ordinary and positive-unlabeled (PU) likelihoods. Shared dataset
adapters, experiment controls, checkpointing, evaluation, and OnDemand entry
points live in the same versioned repository.

Canonical, output-cleared notebooks are organized by scientific question under
[`notebooks/`](notebooks/README.md). Historical notebooks retain their original
source pins under `archive/notebooks/`, while selected executed runs are stored
separately under `results/notebook_snapshots/`. Do not combine numerical results
across different immutable core pins without rerunning the corresponding study.

## Paper experiment entry points

The latest [SPIDER-Seq native-panel notebook](notebooks/native_target_panels/spider_seq.ipynb)
uses the scRNA-seq Adult1/2/3 panels with native missing targets and no artificial
masking. It compares shared core assay-outcome predictors and exports unvalidated
forecasts for truly unassayed entries. Its feature stage reports progress and
normalizes the sparse count matrix once before split-specific HVG/PCA fitting.
This is distinct from the spatial SPIDER notebooks below. See the
[data audit and protocol](docs/SPIDER_SEQ_NATIVE_PANELS.md).

Open one of these notebooks in an OnDemand Python kernel and run its cells from
top to bottom. Figures display in the notebook and save as PDF; raw data,
checkpoints, full metric tables, and predictions persist on disk.

| Notebook | Experiment |
|---|---|
| [simulation](notebooks/positive_label_hiding/simulation.ipynb) | Sharing strengths 0, 0.5, 1; 0–80% loss curves and declared mechanism/calibration controls |
| [Projection-TAGs](notebooks/positive_label_hiding/projection_tags.ipynb) | Natural paired standard versus standard-or-amplified reference; no artificial loss curve |
| [SPIDER spatial](notebooks/positive_label_hiding/spider_spatial.ipynb) | Spatial holdout and controlled 0–80% positive thinning |
| [MERGE-seq](notebooks/positive_label_hiding/merge_seq.ipynb) | Whole-sample holdout and controlled 0–80% positive thinning |
| [BARseq](notebooks/positive_label_hiding/barseq.ipynb) | A1 and M1 fitted and reported separately; within-animal depth holdout |

The 0909 notebooks add a benchmark plot excluding direct reference-label training
and default to bounded essential diagnostics. Earlier dated notebooks, including 0908, remain
available with their original code pins. See [the 0909 update](docs/ONDEMAND_0909.md)
for replotting existing exports without training. The scientific protocol and
model selection rules are unchanged by this display update.

Legacy notebooks under `archive/notebooks/legacy/` document previous implementations. They
are not alternative entry points for this protocol and must not supply numbers
to the new result exports.

The independent [group × target block experiment](docs/BLOCK_MASKING_0909.md)
keeps all input gene features and evaluates artificially unassayed target blocks
on held-out cells within represented groups:

| Notebook | Groups |
|---|---|
| [BARseq A1](notebooks/target_block_masking/barseq_a1.ipynb) | A1's two biological animals |
| [BARseq M1](notebooks/target_block_masking/barseq_m1.ipynb) | Two artificial groups within M1's single animal |
| [MERGE-seq](notebooks/target_block_masking/merge_seq.ipynb) | Four recorded experimental samples |
| [Projection-TAGs](notebooks/target_block_masking/projection_tags.ipynb) | Recorded animals, preserving native assay coverage |
| [simulation](notebooks/target_block_masking/simulation.ipynb) | Two artificial groups, sharing strengths 0/0.5/1 |
| [SPIDER spatial](notebooks/target_block_masking/spider_spatial.ipynb) | Two artificial groups; the current adapter has no verified animal IDs |
| [SPIDER-Seq](notebooks/target_block_masking/spider_seq.ipynb) | Adult1/2/3 native animal panels; targets measured in at least two animals |

These notebooks include a matched full-training-panel control and use separate
checkpoints and result families. Default additional positive loss is zero;
Projection-TAGs keeps natural detections. Block fractions refer to selected
target columns, not positive loss. The primary notebooks above remain unchanged.

The positive-label and target-block notebooks start with these shared defaults:

```python
N_OUTER_FOLDS = 3
USE_LOCATION = False
USE_TARGET_FEATURES = False
N_JOBS = 32
N_REPETITIONS = 5
STRATEGY = 'full_joint'
SHOW_FULL_DIAGNOSTICS = False
```

`full_joint` searches simultaneous rank/penalty combinations from a deterministic
bounded Cartesian grid. The common profile gives each model family a native
budget of 32 genuine candidates. Joint additionally evaluates the exact selected
direct and low-rank winners when compatible standalone models are present, so its
selection set can contain up to 34 rows; those endpoint fits are reused from the
standalone candidate/refit caches. It does not imply exhaustive evaluation of an
arbitrarily large grid. Actual candidates, endpoint provenance and selected
configurations are exported.

In simulation, the default `truth_uses_location = USE_LOCATION` controls the
generated projection signal as well as predictor inputs. Gene-only simulation
therefore does not silently generate a location-dependent truth. Five
repetitions generate five independent datasets per sharing strength; each
dataset contributes three outer spatial folds. Aggregate its folds before
calculating uncertainty across independent generated datasets.

Read [the experiment protocol](docs/PROTOCOL_0908.md) for the observation model,
paired-reference access rules, exact endpoints, information-budget baselines,
scoring ablations, calibration controls, and statistical interpretation.

## Environment and frozen code

Use Python 3.10 or newer and a configured kernel with the experiment dependencies.
The notebooks never run `pip`, modify an installed model, or install into a
read-only OnDemand environment. From an activated **user-owned environment** and
the released repository checkout, install once:

```bash
python -m pip install ".[experiments,notebooks]"
python -m ipykernel install --user --name gene2wire-0908 --display-name "Gene2Wire 0908"
```

Select **Gene2Wire 0908** in OnDemand. Complete setup, offline-cache instructions,
and common runtime fixes are in [ONDEMAND_0908.md](docs/ONDEMAND_0908.md).

Released notebooks embed a 40-character `CORE_COMMIT` and a SHA256 of all
`src/gene2wire` Python sources. They load that immutable snapshot, not whatever
`main` contains on the day of a run. A matching local source tree works offline;
otherwise the exact commit is downloaded once to the code cache. Later runs
verify local bytes. A previously imported package with a different path or
checksum requires a kernel restart.

The distribution version (`0.4.0`) and `CORE_API_VERSION` (`0.6.0`) serve
different purposes. The latter is the checkpoint/model-identity schema epoch;
it changes only when persisted execution semantics become incompatible.

Dependency version ranges describe supported installation requirements; they
are not an exact environment lock. Each result manifest records the numerical
library versions used for that run. Preserve the environment alongside the
source pin when reproducing its numerical results.

## Persistent outputs

The default base directory is `/home/yueyue/gene2wire`:

| Directory | Contents |
|---|---|
| `raw_data/` | Downloaded source data, validation manifests, reusable processed expression, generated simulation arrays |
| `code/<CORE_COMMIT>/` | Verified immutable source checkout, when a matching local checkout is unavailable |
| `checkpoints/0908/` | Atomic compatible tuning/model checkpoints |
| `paper_figure_exports/<dataset>_0908/<run_id>/` | Manifest, complete CSV tables, per-unit audits, saved test predictions and masks |
| `figures/<notebook_date>/` | PDF figures for the embedded notebook release date; also displayed in the notebook |

There is no Google Drive dependency. The first uncached run needs internet
access to obtain code and raw sources. Validated files are reused offline on
subsequent runs. Hash mismatches raise an error rather than accepting changed
data or downloading a different file silently.

Completed compatible work resumes after a kernel disconnect. Scientific
settings, actual code contents, data, splits, and seeds participate in run/fit
identity. Changing only `N_JOBS` or `PARALLEL_UNIT` changes scheduling, not the
scientific setting. Default `PARALLEL_UNIT='scenario'` schedules folds ×
repetitions × loss/calibration settings, so three folds × five repetitions ×
five rates provide 75 tasks and can use 32 workers. Natural Projection-TAGs has
one setting, so at most 15 tasks. `PARALLEL_UNIT='fold'` groups scenarios in a
worker. Both modes share scenario checkpoints. Numerical libraries and random
forests use one inner thread to avoid nested parallelism.

RF fit caches use the actual data/configuration identity across scenarios. For
example, identical paired-only RF candidate fits can be reused across loss
rates; candidate scores are always recomputed using the current validation D/e.
POSIX per-fit locks prevent simultaneous workers from computing the same RF fit.

## Feature switches and measurement scope

| Dataset | `USE_LOCATION=True` | `USE_TARGET_FEATURES=True` |
|---|---|---|
| Simulation | Generated location basis; also enters truth under the notebook default | Generated, declared target descriptors |
| Projection-TAGs | Native coarse source origin (MO/SSC), or an aligned external location CSV | Requires an aligned external target-feature CSV |
| SPIDER | Native measured spatial covariates | Requires an aligned external target-feature CSV |
| MERGE-seq | Requires an aligned external physical-location CSV; pseudotime is not substituted | Requires an aligned external target-feature CSV |
| BARseq A1/M1 | Native depth/angle covariates with training-fitted spline bases | Anatomy descriptors derived from target names; not postsynaptic gene expression |

External feature rows must align to explicit cell/target IDs and describe
outcome-independent covariates. A missing required input raises an error before
fitting; enabling a switch never silently invents features. Preprocessing and
variable-gene selection are fitted on the appropriate training stage only.

Projection-TAGs retains strictly matched cells, excludes Parse, and keeps the
five prespecified excitatory classes from the audited cohort. Filter counts are
exported. MERGE's default three folds partition four whole samples into test
groups of two/one/one samples; four-fold leave-one-sample-out is also available.
BARseq spatial folds are within-animal, not animal-held-out tests.

## Model and reference semantics

The model core receives `X_cell`, observed labels `S_observed`, assayed-entry
mask `W_measured`, optional `Y_target`, and estimated detection probabilities.
Structurally unassayed entries never become negative examples. Evaluation
references stay outside model selection. Authorized paired training references
enter only through explicit calibration/reference-supervision views.

Ordinary models, including observed-only random forests, estimate detection
probability `q`. PU and reference-supervised predictors estimate reference-relative
projection probability `p`, with observed probability `q = e_hat * p`. An ordinary
random forest is not a PU random forest. A higher-sensitivity assay reference is
not complete anatomical truth.

`RF-mixed` trains on reference labels in the authorized paired subset and on
observed labels elsewhere, using each entry once. Its raw mixed-label score is
evaluated directly, without sensitivity rescaling or Bayes h-ranking; it is not
exported as an identified p or q. The default paired fraction is 20% and remains
configurable. RF-observed, RF-reference and RF-mixed share the candidate grid,
seed schedule and validation-label budget.

All 15 retained methods run at every primary loss rate (0%, 20%, 40%, 60%, 80%);
Projection-TAGs retains its natural paired evaluation. Both Prevalence methods
are removed. Reference-only logistic is restored. Reference + PU logistic,
Reference + PU-MIRT and Reference + PU-Joint directly use the same clean C
outcomes and PU supervision on the remaining O, enabling comparisons of model
structure under matched supervision. The original six models remain unchanged.
Solid lines indicate PU or paired-reference information;
dashed lines indicate neither. Reference + PU logistic and both RF arms using
paired references therefore have solid curves. Historical unrun rates remain
gaps when loading old exports; new intermediate fits require a training run.
Horizontal model dot panels are omitted by default; Projection-TAGs retains
its paired-label audit and natural-condition horizontal model bars. Information-budget plots
compare Reference + PU logistic, RF (observed + paired references), and
Reference + PU-Joint in three metric panels. All use the same paired cells;
RF learns mixed labels without detection correction. Recovery uses PU posterior
scores for the PU models and RF scores for RF. Synthetic unrun/single-condition
comparisons are skipped; natural paired comparisons do not invent a loss axis.

Qiao bilinear comparisons are enabled by default on every dataset's primary
conditions. With target features disabled, `Qiao-ID-squared` and `Qiao-ID-logit`
use one-hot known target IDs; they are explicitly named adaptations. With target
features enabled, `Qiao-squared` and `Qiao-logit` use the same declared descriptors
as the primary models. Squared-error and Bernoulli objectives remain distinct;
neither comparator is a PU learner. Native real-data target anatomy labels are
not equivalent to the original model's postsynaptic expression inputs.

Default progress is one summary line per minute in `America/Los_Angeles` time.
Its unit is one model at one repetition/fold/scenario, including selection and
refit; optimizer candidates are recorded in detailed events rather than printed.
The startup count covers verified complete result summaries only; other cache
status is initially unchecked. Each heartbeat partitions completed units into
restored results, reused fits (disk or memory), and new/mixed fitting. A final
refit cache hit alone cannot classify the entire model as reused. Incomplete
telemetry is explicitly unknown. `model_evaluation_plan.csv` lists the exact
denominator and `model_cache_accounting.csv` records the completed-unit breakdown.
Duplicate calibration fractions are rejected before scheduling.
A separate notebook cell reports CPU capacity from the machine, process affinity,
cgroup quotas and scheduler allocation metadata, refreshed once per progress
interval. It reports this experiment's active workers separately. Allocation is
not globally idle capacity; the notebook cannot count other jobs' unused CPUs.
Notebook reports default to `SHOW_FULL_DIAGNOSTICS=False`: bounded aggregate
metrics, important-model settings by repetition, convergence, calibration and
matched block/control diagnostics. Omitted rows are explicitly counted. Set the
switch to `True` only to display every raw table; that output can be large.
Complete CSV/NPZ exports remain available in either mode.
New exports include `joint_selection_diagnostics.csv`: per-unit candidate counts
and best converged validation scores for direct, lowrank and joint candidates,
plus the recorded selected family and its validation margin versus direct.
This audits the recorded Joint search, not an inferred full MIRT grid. Full
diagnostics derive the same table from older saved trials without refitting.

The separate partial gene-panel overlap workflow is documented in
[`docs/GENE_OVERLAP_0910.md`](docs/GENE_OVERLAP_0910.md). Its five generated notebooks keep projection
targets fixed, expose an editable `OVERLAP_GRID`, mask only input expression,
and disable random forest by default. The MERGE-seq notebook uses crossed
panels within `(sample, assay_status)`, retains its five targets and whole-sample
cross-validation, and therefore serves as a robustness analysis rather than a
general low-rank mechanism test. Its all-`P` oracle means all fixed 128 candidate
genes chosen from reserved `W=0` design cells—not the whole transcriptome—and
does not add sample nuisance effects. Its overlap-only expression normalization
also excludes `barcode*` assay-derived features from the library-size
denominator. The MERGE entry point is
[`notebooks/gene_panel_overlap/merge_seq.ipynb`](notebooks/gene_panel_overlap/merge_seq.ipynb).

## Tests and development

The repository uses one long-lived branch, `main`. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the directory map, notebook regeneration
contract, versioning rules, and the pre-review checklist.

From a user-owned development environment:

```bash
python -m pip install -e ".[test,experiments,notebooks]"
python -m pytest -q
```

CI runs the suite on Python 3.10–3.13. Tests cover analytic gradients, observation
and reference-access boundaries, exact endpoints, deterministic search/resume,
dataset alignment and offline cache fixtures, feature switches, calibration,
scoring, and clean notebook generation. They use small local fixtures and do not
download full research datasets or execute the formal paper training matrix.

The reusable model API remains in `src/gene2wire/`; shared experiment code is in
`src/gene2wire/experiments/`. Notebook builders live in `scripts/notebooks/` and
write stable filenames into the corresponding `notebooks/<experiment>/`
directory. Release dates and immutable core identifiers belong in notebook
metadata and Git tags rather than canonical filenames. The protocol seed and
checkpoint directory do not follow the notebook release date.
Source changes require new pins and corresponding reruns before their outputs
can be treated as one experiment version. To add plots to old completed runs,
use `RESULTS_ONLY=True`; the current conservative checksum includes plotting
code, so a new core pin does not automatically reuse an older fit cache.
