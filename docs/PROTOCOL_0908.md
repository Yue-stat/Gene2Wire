# Gene2Wire 0908 experiment protocol

This document describes the implemented experiment design, not completed results.
All 0908 results remain provisional until the corresponding formal reruns and
export checks finish. Passing implementation tests does not establish a method's
empirical superiority or robustness to a real assay's unknown detection process.

## Common configuration

The OnDemand notebooks use the same `Settings` profile:

```python
N_OUTER_FOLDS = 3
USE_LOCATION = False
USE_TARGET_FEATURES = False
N_JOBS = 32
N_REPETITIONS = 5
STRATEGY = "full_joint"
```

The paired-reference budget is 20% of development cells. Primary artificial
positive-loss rates are 0%, 20%, 40%, 60%, and 80%. Projection-TAGs uses its natural
paired standard/reference measurements and receives no artificial thinning.

All six primary models use the same available features and observation protocol:
Logistic, MIRT, Joint, PU, PU-MIRT, and PU-Joint. The search support is determined
by feature and target dimensions; the simulation search does not receive the
generated true rank. Target intercepts of the projection models are unpenalized.
The common primary search has a maximum of 32 selectable candidates per method;
actual, reused, and converged candidate counts must accompany results. A direct
model with fewer distinct configurations does not repeat fits to exhaust its cap.

The September 9 correction is `0908-v2-balanced`. Within each structural family,
candidate selection balances the marginal coverage of rank and each penalty,
then pairwise level coverage and geometric separation (log distance for positive
penalties). It replaces evenly spaced indices of a flattened Cartesian product,
which could alias onto a single residual penalty. The declared rank/penalty
support and 32-candidate cap are unchanged. With target features disabled, Joint
still has 6 direct, 13 low-rank and 13 shared-plus-residual candidates. Its 13
low-rank endpoints do not contain every standalone MIRT candidate; exact
endpoints therefore do not guarantee inclusion of MIRT's validation winner or
superior held-out performance. Actual candidate tuples are recorded in tuning exports and are displayed
before fits when `SHOW_FULL_DIAGNOSTICS=True`. Revised search results require fresh runs and
must not be relabeled as outputs of the earlier flattened-grid protocol.

`N_JOBS` controls hardware concurrency and is excluded from scientific identity.
Changing it must not change masks, paired IDs, seeds, or which learned fits can
be resumed. Scientific settings, source contents, inputs, and split identities
do affect reproducibility. No notebook edits installed package source code.

## Dataset and split scope

Raw-data interpretation and biological grouping remain dataset-specific; model
patches and hidden dataset-specific tuning rules do not. Structurally unassayed
entries retain `W_measured=False` and never become negative training examples.

Every outer fold partitions cells into inner training, validation, and test.
Preprocessing is fitted on inner training during selection and on development
after selection. Biological groups are kept intact where the evaluation is
defined as animal- or sample-held-out. Spatial evaluations retain their declared
spatial grouping and must not be described as animal-held-out automatically.

MERGE-seq uses three outer folds in this release because the user explicitly
requested `N_OUTER_FOLDS=3`. Whole samples are assigned to folds; four samples
need not produce equal-sized test folds. The actual sample IDs in each role are
exported. This differs from the older four-fold protocol and requires a rerun.

Simulation uses three spatial outer folds per independently generated dataset,
with each cell tested once. It does not retain the old one-rotating-holdout-per-
dataset interpretation. `N_REPETITIONS=5` generates five independent datasets per
sharing strength; the strengths 0, 0.5, and 1 use paired base randomness.

Simulation distinguishes generating truth from predictor inputs. The notebook's
matched-design default sets `truth_uses_location = USE_LOCATION`. Consequently,
the default gene-only experiment also generates projection truth from genes
alone. Available location covariates do not silently enter the generating model.
An omitted-location experiment must explicitly set location-dependent truth and
gene-only prediction, and be labeled as a separate scientific setting. Target
descriptors are supplied to every applicable comparison when
`USE_TARGET_FEATURES=True`; unavailable target covariates must produce an explicit
unsupported-input error rather than silently substituting outcome-derived data.

## Controlled observation mechanisms

The new shared generator uses

\[
e_{it}(r)=\sigma\{\gamma_f(r)+\delta_t+\beta u_i\},\qquad
D_{it}(r)=Z^{\mathrm{ref}}_{it}\mathbf 1\{U_{it}<e_{it}(r)\}.
\]

| Mechanism | Target offset | Technical covariate |
|---|---|---|
| SCAR | Zero | Omitted |
| Target-SAR | Fixed heterogeneous `delta_t` | Omitted |
| Technical-SAR | Fixed heterogeneous `delta_t` | Fixed `beta * u_i` |

The default target offsets span minus to plus `log(2)` in a seeded target order.
The default technical coefficient is `log(2)`. Real-data artificial thinning
uses a synthetic technical covariate; simulation can supply its explicitly
generated feature-correlated score. Supplied scores are used as declared rather
than reconstructed from realized projection labels.

For each fold and loss rate, a single intercept `gamma_f(r)` is determined from
**inner-training reference positives only**, so that their mean expected retention
is `1-r`. It is not normalized using validation or test reference positives.
Different targets need not achieve the same retention rate. The realized number
of retained positives fluctuates around its expectation and should be exported
separately from nominal loss.

At zero artificial loss, `e=1` exactly and the normalization intercept is recorded
as `None`. This means no additional loss relative to the pre-thinning reference;
it does not establish perfect anatomical sensitivity of the original assay.

The target offsets, technical score, and cell-target uniform draws are held fixed
across rates within a fold/repetition. Only the common intercept changes, so the
new masks are nested. Final refitting never regenerates these labels or opens
validation references to renormalize the generator. Changing test references
must leave the normalization intercept, every propensity, and training detections
unchanged. Old names such as `technical_sar_old` are rejected rather than silently
mapped to this new scientific design.

This mechanism is a controlled stress test with separately interpretable loss and
heterogeneity parameters. It is not a fitted physical model of any real assay.

## Paired calibration and information access

Before examining outcomes, the sampler fixes a nested ordering of development
cells, with optional group stratification. A fraction `f` authorizes exactly
`floor(f * N_development)` cells. The same ordering supports the 10%, 20%, and 40%
calibration-size comparison. Sampling does not seek additional positives or
redraw until a desirable outcome count appears.

During inner selection, only authorized paired cells in inner training are
opened. Validation references remain closed even when those cells have been
preselected for the eventual development budget. After the model configuration
is selected, the predeclared paired validation cells may be opened for the common
development refit. The total authorized reference budget remains unchanged.
Test references are used only by evaluators after fitting. Full training
references used internally to construct an artificial benchmark are not
learner-facing supervision and must be described separately from the paired
information budget.

The detector is fitted to observed detections among authorized, assayed reference
positives. Technical-SAR calibration receives the declared technical covariate,
not the generator's true probabilities or coefficients. Target-SAR omits the
technical covariate; SCAR also omits target offsets. Natural paired assays use
platform and target categorical effects instead of inventing a technical score.

All detector variants use the same fixed regularization convention: summed
Bernoulli negative log likelihood, ridge penalties on non-intercept coefficients,
and symmetric pooled-intercept pseudocounts. This keeps all-zero/all-one detection
samples finite. A target without paired positives falls back to pooled effects;
an unseen platform uses pooled target predictions. If the entire paired subset
has no reference positives, calibration is not estimable and the failed unit is
recorded; no outside reference labels are borrowed and no failed repetition is
silently removed. Detector shrinkage is distinct from the unpenalized projection
target intercepts.

## Same-budget supervision controls

Let `C` denote the assayed entries in the authorized paired cells, and `O` the
remaining assayed training entries. The independent-model controls share cells,
features, splits, and reference budget:

| Arm | Projection training objective |
|---|---|
| Reference-only | Reference Bernoulli loss on C, including reference negatives |
| Calibrated PU | Observed PU likelihood on C and O; C estimates detection |
| Reference + PU | Reference loss on C plus observed PU likelihood on O |

The mixed arm compiles `C` to reference labels with effective sensitivity 1 and
`O` to observed labels with estimated sensitivity. It reuses the exact core
likelihood, averaged over the included entries. It does not additionally count
the marginal observed-label loss on `C` as an independent projection outcome.

Reference+PU versus Reference-only isolates the contribution of the additional
detection-only cells under this protocol. Calibrated PU versus Reference-only
also changes how paired references are used and cannot alone establish that
isolated contribution. Common validation scoring uses the original validation
detections: a reference-probability predictor is mapped to observed probability
with the same training-side detector. Validation references do not choose its
hyperparameters. The reference-only predictor may therefore share calibration
information for common selection while using only references to fit projection
coefficients; this access should be reported explicitly.

## Fixed experiment matrix

Scenario generation is independent of test outcomes. Additional axes are
restricted to the following scopes, rather than crossed with every dataset:

| Analysis | Scope |
|---|---|
| Primary six-model curves | Synthetic Technical-SAR at all five rates; natural paired endpoint for Projection-TAGs |
| Mechanism controls | Simulation only: SCAR and Target-SAR at 80%, all three sharing strengths |
| Calibration-size controls | Simulation only: sharing 0 and 1, Technical-SAR 80%, correctly specified detector, 10/20/40% budget |
| Calibration misspecification | Simulation only: sharing 0 and 1, Technical-SAR 80%, 20% budget; omit technical score or pool targets |
| Independent information-budget controls | Natural endpoint or 80% artificial loss |
| Observed-only/reference-only random forests | Natural endpoint or primary 0/80% endpoints |
| Post-hoc sensitivity rescaling | Reuse observed-model predictions in target-constant SCAR/Target-SAR settings |
| Fixed-predictor p/h ranking | Reuse saved primary predictions, without retraining |
| Qiao comparisons | Enabled by default on every primary rate/natural endpoint; target-ID adaptation when descriptors are disabled |

Default scenario counts are one for natural paired data, five for each real-data
thinning benchmark, seven for intermediate-sharing simulation, and eleven for
zero/full-sharing simulation. The middle calibration fraction reuses the primary
20% condition instead of adding a duplicate scenario. Information and RF controls
are model-level additions at the stated endpoints, not extra missingness axes.

Qiao's squared-error and logit variants use only observed training outcomes and
the common observed-validation log-loss rule. When `USE_TARGET_FEATURES=False`,
both use one-hot known target IDs and are explicitly exported as
`Qiao-ID-squared` and `Qiao-ID-logit`. This does not enable external target
features or supply additional outcome information. When target features are
enabled, both use the same declared target descriptors as the other models and
are named `Qiao-squared` and `Qiao-logit`. These are bilinear model-family
comparisons, not a claim to reproduce the published paper's entire pipeline.
Their rank/penalty candidates use the same outcome-independent balanced design,
with at most 32 candidates per objective; the rank is bounded by available
cell/target dimensions and target count. They remain non-PU comparators.

New gene-panel masking, unseen-target prediction, complex neural multitask
models, and additional shared/specific penalties are outside this freeze matrix.

## Probabilities, ranking, and evaluation

Ordinary observed-label models estimate `q`, the observed detection probability.
PU and reference-supervised projection models estimate reference-relative `p`;
their observed prediction is `q = e_hat * p`. A real-data reference is not complete
anatomical truth. No ordinary random forest is relabeled as a PU forest simply
because its output is scored against a reference.

For a fixed PU/reference predictor, hidden-positive ranking among `D=0` candidates
can use `p` or

\[
h=\frac{(1-\widehat e)p}{1-\widehat e p}.
\]

Both rankings are exported. Within a target/fold with constant sensitivity below
one, their rankings agree; cell-dependent detection may change them. Hidden
recall is undefined when there are no hidden positives, including zero artificial
loss. Fractional inclusion at a tied cutoff avoids cell-order-dependent recall.

Post-hoc `q/e_hat` is restricted to target-constant mechanisms in the default
matrix. A predictor of `q(X)` that omits a technical covariate cannot generally be
divided by `e(X,u)` as if the conditioning information matched. Rescaling exports
its clipping frequency and is kept separate from PU likelihood training.

Primary reporting includes target-macro average precision (AUPRC), reference log
loss, and Hidden Recall@H. Exported secondary metrics include AUROC, Brier, ECE,
prevalence errors, reliability bins, and per-target/entry-micro summaries. Brier
skill uses an authorized training-reference baseline, never a fitted test
prevalence. Metrics that require both classes remain undefined for single-class
test targets, with the number of evaluable targets reported.

Detector evaluation uses held-out reference positives. Known synthetic detection
probabilities permit MAE/RMSE; natural paired assays permit detection proper
scores and reliability but have no biological sensitivity truth for RMSE. These
controls measure the tested misspecifications and do not establish arbitrary
real-assay robustness before results are available.

## Persistence and uncertainty

Raw downloads are cached locally with hashes. A validated cached file is reused
without network access; interrupted downloads do not replace a complete file.
Recorded local hashes establish byte identity, not publisher authentication when
the source provides no checksum. Corrupted caches raise an explicit error.

Checkpoint identities include the scientific setting, input and split identities,
and actual source contents. Small work units are saved atomically for resume.
Changing only worker count must preserve their identity. Results exported under
`/home/yueyue/gene2wire/paper_figure_exports` retain dataset, scenario, fold,
repetition, model, selected structure, hyperparameters, optimization diagnostics,
and reference/probability semantics. Saved predictions and masks support metric
recomputation. Notebook curves remain visible and are also saved as PDFs.

In simulation, outer folds from the same generated dataset are dependent.
Aggregate folds within each generated repetition before treating the five
datasets as independent Monte Carlo units. In real data, mask/calibration
repetitions do not create new animals. Keep repetition variability and biological
or spatial fold differences distinguishable; do not label all fold-by-repetition
points independent biological replicates or infer an animal-population confidence
interval from their count.

Old results and plots remain attached to their old protocol versions. No old
number becomes a validated 0908 result through relabeling, and no model is required
to win every dataset as a condition of releasing reproducible code.
