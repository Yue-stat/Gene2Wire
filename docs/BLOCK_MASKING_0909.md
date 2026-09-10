# Group × target block experiment (0909)

These are independent experiments; the dated primary notebooks are preserved.
All five use the existing core estimators, information-budget baselines, tuning
rules and feature adapters. There are no dataset-specific model patches.

| Notebook | Groups | Interpretation |
|---|---|---|
| `BARseq_A1_block_masking_0909.ipynb` | Two recorded biological animals | Within-animal new-cell prediction under complementary target panels |
| `BARseq_M1_block_masking_0909.ipynb` | Two balanced artificial groups within one animal | Controlled group-panel experiment, not two-animal validation |
| `Projection_TAGs_block_masking_0909.ipynb` | Recorded animals, retaining native assay coverage | Additional structural blocks over the naturally incomplete assay |
| `simulation_block_masking_0909.ipynb` | Two artificial groups for each generated dataset | Controlled test across sharing strengths 0, 0.5 and 1 |
| `SPIDER_block_masking_0909.ipynb` | Two balanced artificial groups | Controlled within-group test; no verified animal IDs in the processed adapter |

## Masking and splits

The input gene panel is retained in full: no extra genes are removed. This does
not change each adapter's existing feature selection or turn a targeted panel
into whole-transcriptome measurements. Group IDs are not added as predictors.

For each repetition, eligible targets are ordered using a fixed random seed.
A target is eligible only when natively assayed in at least two groups. At each
configured fraction, a prefix of eligible targets receives a partial panel:
roughly half of its originally measured groups are hidden, with at least one
group retaining that target and at least one native target retained per group.
Two-group complete panels have balanced complementary assignments. All other
eligible targets remain shared. Construction uses IDs and assay availability,
never labels or expression. The masks are nested across fractions. Sparse
availability uses capacity-constrained assignment; an infeasible design raises
an explicit error.

`BLOCK_FRACTIONS=(0.20,0.40,0.60,0.80)` describes the fraction of **eligible
targets receiving a partial group panel**, not the fraction of positives or
pairs hidden. With two equally sized groups and full native coverage, selecting
80% of targets hides approximately 40% of all cell–target entries. Rounded
target counts and the actual hidden pair/block fractions are exported. Original
off-panel entries are never scored. Targets unique to one group remain visible
but are ineligible for artificial masking.

Increasing this x-axis adds targets to the partial panel. It does not increase
the fraction of groups hidden for an already selected target. With two groups,
that target loses one group at every setting. Each setting can also evaluate a
different mix of targets. A roughly constant model gap, or a decreasing raw log
loss along this axis, is therefore not evidence of a masking error or an
improvement caused by removing labels. The compact report computes matched
masked/full losses and the change in each structured model's advantage over PU
using the same repetition, fold and test entries before aggregation.

Each repetition splits cells within each group into three outer test folds.
Within each fold, 20% of development cells per group form inner validation;
the remaining development cells are inner training. Every group appears in
every role, and each cell is outer test data once. This split is deliberately
different from primary spatial or animal holdouts. It estimates prediction for
new cells within represented groups, not unseen-animal transfer or unseen-target
generalization. A1's two animals permit descriptive per-animal comparisons;
they are not a large animal-level replication study.

## Preventing hidden-label access

Let `H[i,t]` indicate a native measured entry in a designated hidden block.
The fitting availability is `W_fit = W_native & ~H`. Both positive and zero
labels in a hidden block are removed. A sanitized fitting view excludes those
entries from observed labels, authorized references, detector estimation and
validation scoring. Full original references are retained separately for
outer-test evaluation. Preprocessing is fitted on the designated training
cells and never uses hidden projection outcomes.

The same 20% paired cell IDs are used across masked and full training panels.
Only their currently visible entries are authorized: an artificially unassayed
target cannot return through a paired reference. Reference-plus-PU methods
replace the observed-label likelihood by reference supervision on authorized
paired entries; they do not count the same entry twice. Every target retains
assayed support in other groups, but a sparse positive count can still make
calibration uncertain; inspect the exported target diagnostics.

The full-panel control restores native training and validation availability,
shares cell splits, feature transforms, paired cell IDs and tuning rules, and
never fits outer-test outcomes. It is fitted once per repetition/fold/detection
setting and evaluated on each block setting's identical hidden test subset.
Its curve can therefore change with the evaluation targets even though its
fitted parameters are reused. Full and masked panels receive the same tuning
budget, although the full control intentionally has more available labels.

## Observation process and models

`POSITIVE_LOSS_RATES=(0.0,)` isolates structural missingness by default for
BARseq and simulation. Optional nonzero rates use simple SCAR with common
random draws across panel conditions; change the variable to
`(0.,.2,.4,.6,.8)` for that additional experiment. The structural schedule
does not normalize detection using hidden labels. Projection-TAGs always uses
the original standard assay and union reference, without artificial thinning.

All enabled models run at every configured setting. Defaults retain the six
ordinary/PU structures, reference-only logistic, three reference-plus-PU
structures, three random-forest supervision variants, and two declared Qiao
adaptations. Existing exact direct and residual-off Joint endpoints and common
tuning budgets are unchanged. At zero additional thinning, ordinary and PU
objectives may coincide. These runs test sharing under missing panels, not an
assumed benefit of PU correction in the absence of positive loss.

Simulation defaults retain five independent datasets per sharing strength,
three cell folds, and 20% paired references. `USE_LOCATION=False` excludes
location from both generated projection signal and fitted predictors.
`USE_TARGET_FEATURES` retains the same declared feature behavior as the primary
notebooks. All feature switches must be set before running.

## Evaluation and plots

Score only originally assayed entries in hidden blocks on outer-test cells,
using original simulation truth or the designated real-data reference. Report
target-macro AUPRC/AUROC, log loss, Brier score, calibration and prevalence
diagnostics, with available per-target and per-group breakdowns. Undefined
metrics remain missing and coverage/count information is retained.

Use the raw prediction score on these unassayed pairs. Do not apply the posterior
for an observed non-detection: an absent assay is not `D=0`. Hidden Recall@H
is therefore not a metric for this experiment. Ordinary detection predictors
are still ordinary predictors; evaluating them against a higher-sensitivity
reference does not make their probabilities anatomically identified.

The notebooks display and save PDF curves for the primary PU trio, all
benchmarks, methods without direct reference training, and the same-budget
reference-plus-PU/RF-mixed/reference-plus-PU-Joint trio. Main columns show AUPRC,
log loss and Brier; rows compare masked and full training panels on matched
test subsets. Lines are descriptive fold/repetition averages, without invented
animal-level confidence intervals. The original dated primary plots remain
available in their original notebooks.

## Reuse, progress and exports

Defaults: 32 requested workers, three folds, five repetitions, `full_joint`,
32 candidates per method, bounded essential diagnostics, and one progress line
per minute in Los Angeles time. `SHOW_FULL_DIAGNOSTICS=False` retains small,
fully rendered aggregate metric/configuration tables and explicit omission counts;
setting it to `True` opts into every raw table. The worker cell reports CPU
allowance and allocation metadata separately from this experiment's active workers.
The expensive unit is a model with tuning/refit at one repetition, fold and
training-panel/detection setting; the full control is not counted as a new fit
for every evaluation fraction. Actual execution capacity depends on scheduled
tasks and allocated CPUs.

Verified raw data persist in `/home/yueyue/gene2wire/raw_data`. New fit caches
use `checkpoints/block_masking_v1`. Full result tables, predictions, native and
hidden masks, assignment/split audits, selected hyperparameters and manifests
remain beneath `/home/yueyue/gene2wire/paper_figure_exports`, in separate
`*_target_block/<run_id>` families. Scientific configuration, source, original
evaluation references and block designs enter run identity. Changing worker
count does not change the scientific identity.

PDFs save in `figures/0909/block_masking` and display in the notebook. Set
`RESULTS_ONLY=True` and supply exact completed export directories to redraw
without raw downloads or fitting. Each notebook pins an immutable core commit
and source checksum. Later dated notebooks do not overwrite older dates.

These entry points do not contain precomputed research results. Small execution
checks validate implementation; full runs and diagnostics are still required
before drawing conclusions about which structure benefits from masking.
