# Combined measurement-degradation experiments

The canonical notebooks in
[`notebooks/measurement_degradation/`](../notebooks/measurement_degradation/)
combine three distinct measurement failures in one predeclared protocol:

1. a cell's virtual assay measures only part of the input-gene universe;
2. it measures only part of the projection-target universe; and
3. some reference-positive, measured target entries are not detected.

The protocol keeps **not measured** separate from **measured but not detected**.
Panel construction and virtual-assay assignment do not inspect expression,
projection outcomes, or test results. The generated notebooks are
output-cleared entry points for a source-pinned local or OnDemand Python
environment; no Colab copies are generated.

## Coverage replaces the old fixed-panel overlap parameter

`gene_coverage` is the fraction of the complete, declared source-gene pool
measured by **each** of three virtual assays A, B, and C. It is not the overlap
fraction between fixed-size panels. If the source pool has `P` genes, each
assay receives the same nested panel size

```text
K(c) = max(ceil(P / 3), round_half_up(c * P)).
```

A seeded permutation places the genes on a circle. The three assays receive
fixed, approximately equally spaced starting points and the next `K(c)` genes
from those points. Consequently:

- the union of A, B, and C contains all `P` source genes at every supported
  coverage;
- decreasing coverage only removes genes from an assay, so panels are nested;
- at `gene_coverage=1`, A, B, and C each contain all `P` genes; and
- pairwise overlap is a diagnostic consequence of coverage, not the parameter
  being held fixed.

For BARseq the declared source pool has 23 genes. The implemented sizes are:

| Requested gene coverage | Genes in A | Genes in B | Genes in C | A/B/C union |
|---:|---:|---:|---:|---:|
| 1 | 23 | 23 | 23 | 23 |
| 5/6 | 19 | 19 | 19 | 23 |
| 2/3 | 15 | 15 | 15 | 23 |
| 1/2 | 12 | 12 | 12 | 23 |

Thus the new 100% condition is **23/23 genes in every BARseq assay**. The eight
genes used by the older gene-overlap notebook were a fixed panel-size design;
that design is not reused here. With small gene or target universes, rounding
can make two requested coverages produce the same `K`. Such conditions are
identified in the panel audit and fitted once rather than presented as two
different masks.

Target panels use the same equal-size, nested, cyclic, union-preserving
construction over the dataset's declared targets. The artificial target panel
is then intersected with native assay availability; it never manufactures a
measurement that the source dataset did not contain. Planned panel incidence
and effective fold-local support are both exported.

## Fixed design and masking order

Outer train/validation/test folds come from the existing dataset adapter. Each
cell is assigned once to A, B, or C using a stable cell-ID hash, balanced as
closely as possible within an outcome-independent biological stratum such as
animal or sample. Cells are not copied. Real-data panel repetitions keep the
same cell-to-assay assignment while changing seeded panel membership, which
supports same-cell panel-sensitivity diagnostics.

For cell `i`, target `j`, and virtual assay `a(i)`, define:

- `Y_ref[i,j]`: the reference outcome;
- `W_native[i,j]`: whether the source dataset measured that target entry;
- `B[a,j]`: whether virtual assay `a` includes target `j`;
- `W_fit[i,j] = W_native[i,j] & B[a(i),j]`: availability shown to calibration
  and model fitting;
- `D[i,j]`: whether a measured reference positive is detected after censoring;
- `S[i,j] = Y_ref[i,j] & W_fit[i,j] & D[i,j]`: the observed positive label.

These states have different meanings:

| State | Interpretation | Learner treatment |
|---|---|---|
| `W_fit=0` | target was not measured in this assay | excluded from the loss |
| `W_fit=1, S=1` | measured positive was detected | observed positive |
| `W_fit=1, S=0` | measured entry is unlabeled | PU unlabeled entry |

An off-panel entry is therefore never converted to a negative. Hidden positives
are not supplied to the learner as a separate indicator.

The gene panel is applied to the source-scale `gene_matrix` before any
fold-fitted transformation. For every gene, centering and scaling are learned
only from visible training values. A missing standardized value is stored as
zero, accompanied by a binary gene-observation indicator; virtual-assay
indicators are also included. This distinguishes an unmeasured gene from an
observed value equal to its training mean. Dataset adapters must expose their
gene matrix before train-fitted scaling, HVG selection, or PCA. For example,
SPIDER-Seq declares a deterministic, outcome-blind 2,000-gene pool selected by a
versioned hash of gene IDs; its matrix is on the cell-local
`counts / total-RNA-library * 10000 -> log1p` scale, before any cross-cell fit.
Its canonical notebook then fits 50 PCA components on the masked, zero-filled
inner-training matrix and scales those component scores on the same rows; the
2,000 binary gene-observation indicators remain separate inputs. Thus the
configured PCA is downstream of masking rather than a full-data preprocessing
shortcut.

## Fold-local positive censoring

The heterogeneous arm samples one immutable standardized assay-by-target score
`h[a,j]` and one immutable uniform value `U[i,j]`. With heterogeneity
`delta=1`, the sensitivity for a measured reference positive is

```text
e[i,j] = sigmoid(alpha[j] + h[a(i),j]).
D[i,j] = 1[U[i,j] < e[i,j]].
```

For every outer fold and requested retention `c`, `alpha[j]` is solved using
only measured reference positives in that fold's inner-training rows, so the
target-specific expected training retention is `c`. If a target has no
authorized inner-training positive, the explicitly reported pooled
inner-training intercept is used. Validation and test outcomes never set an
intercept. At retention 1, sensitivity and detection are set to one.

The same `h` and `U` values are reused across methods and retention levels. This
makes censoring comparable and nested rather than drawing a new missing-label
pattern for every model.

The 20% paired-reference calibration budget is sampled from development cells
without reading their labels. Sampling is stratified by the cross-product of
the available biological group (animal or sample) and virtual assay, so A, B,
and C are represented within biological strata as closely as the fixed global
budget permits. For real-data panel repeats, the paired rows, cell-to-assay
assignment, censoring surface, uniforms, and model-initialization seed are held
fixed; only the seeded panel membership changes. Simulation repetitions redraw
these objects together with each independently generated dataset.

At the anchor condition, a matched-uniform (SCAR) control uses the same fixed
uniforms. Its threshold is selected only from inner-training positives so that
its **realized number of retained training positives exactly equals** the
heterogeneous arm. The control therefore changes the selection mechanism, not
the training-positive budget. The target panel itself remains unchanged.

## Evaluation scopes

Artificial target masking changes the learner's information, not the headline
test population. Every condition is scored on the same outer-test entries:

```text
headline: native_reference = W_native[test]
```

The notebooks also report:

- `on_panel = W_fit[test]`;
- `off_panel = W_native[test] & ~W_fit[test]`; and
- `hidden_candidate = W_fit[test] & (S[test] == 0)`.

`Y_ref[test]` is held by the evaluator only. Native source restrictions remain
in force, so none of these scopes treats a naturally unassayed entry as known.
SPIDER-Seq's native Adult1/2/3 target mask, groups, and existing within-animal
folds are preserved.

## Predeclared condition schedule

The protocol avoids the full `4 x 3 x 5` Cartesian product. It fits these
deduplicated slices:

| Role | Gene coverage | Target coverage | Positive retention | Mechanism |
|---|---:|---:|---:|---|
| Full control | 1 | 1 | 1 | assay-target heterogeneous design with no realized censoring |
| Retention curve | 2/3 | 2/3 | 1, 0.75, 0.50, 0.25, 0.10 | heterogeneous |
| Coverage heatmap | 1, 5/6, 2/3, 1/2 | 1, 2/3, 1/2 | 0.50 | heterogeneous |
| Mechanism control | 2/3 | 2/3 | 0.50 | exactly count-matched uniform/SCAR |

The anchor is `(2/3, 2/3, 0.50)`. Identical conditions shared by the curve,
heatmap, and control schedule are trained once and carry all applicable role
labels. Projection-TAGs additionally retains its natural standard-versus-
amplified recovery condition; it is kept distinct from artificial censoring.
The canonical configuration uses five outcome-blind panel seeds. Simulation
uses five independently generated datasets at each sharing strength and nests
the five panel seeds within each generated dataset, so panel sensitivity is
computed before uncertainty is summarized across independent simulations.

## Models and information access

All 15 retained fitted methods from the current notebooks remain enabled:

1. Logistic, MIRT, Joint, PU, PU-MIRT, and PU-Joint;
2. Reference-only, Reference+PU, Reference+PU-MIRT, and Reference+PU-Joint;
3. RF-observed, RF-reference, and RF-mixed; and
4. Qiao-ID-squared and Qiao-ID-logit when target features are disabled, or the
   corresponding Qiao-squared and Qiao-logit models when they are enabled.

The measurement-degradation notebooks add three fitted comparators, for 18 in
total. Their report labels disclose the adaptations:

| Notebook label | Implemented adaptation | Information used |
|---|---|---|
| `GenEML-adapted` | Python 3 point-EM equations, known-`W` exclusion, and contextual assay-target exposure | observed labels and outcome-independent assay-target design |
| `Inductive-PU-MC (ShiftIMC-adapted)` | bounded sigmoid prediction, known-`W` exclusion, entry-specific estimated exposure, and a factorized nuclear-norm surrogate rather than exact convex ShiftIMC | observed labels, cell inputs, optional declared target inputs, and paired-calibration exposure estimates |
| `SAR-PU (SAR-EM)` | target-specific logistic outcome classifiers plus one shared dyadic logistic propensity model | observed labels and outcome-independent assay-target propensity design; no clean labels or true propensity |

All methods within a fold/scenario receive the same split, `W_fit`, observed
labels, input panels, authorized paired-calibration rows, and evaluation scope.
The three comparators are tuned on measured validation entries with observed
log loss for `q=e*p`; no test reference or simulation propensity selects a
configuration. Duplicate comparator configurations are removed and each added
comparator's search is capped at 32 genuine candidates. `Logistic-rescaled`,
when defined, is a post-hoc prediction and is not counted as a nineteenth fit.

## Notebook diagnostics, exports, and figures

The notebooks show bounded summaries and write the complete tables and arrays.
The audit surface includes:

- data dimensions, biological groups, fixed folds, virtual-assay counts, and
  reproducibility/source manifests;
- exact gene/target membership, actual coverage after rounding, pairwise
  overlap, item multiplicity, union coverage, and duplicate requested levels;
- per-fold assay-by-target measured counts, effective support, connectivity,
  per-assay effective coverage, assay-pair co-measured target counts,
  unsupported targets, and single-assay targets;
- fold/split-local gene-by-target co-measured cell counts summarized by the
  number and fraction of unsupported relationships, with every unsupported
  training gene-target pair exported explicitly (validation/test retain the
  compact summary without duplicating large pair-level tables);
- requested, expected, and realized retention; hidden-positive counts;
  target-specific intercepts; pooled fallbacks; estimated/true propensity
  diagnostics where truth is legitimately available; and matched-uniform
  retained-count checks;
- aggregate and per-target metrics on all declared evaluation scopes, plus
  per-group and per-repetition variability, calibration/reliability tables,
  selected configurations, every tuning trial, convergence, retries, failures,
  runtimes, and checkpoint/cache status; and
- the fixed heatmap-baseline selection and the resulting contrast table.

The primary figures are:

1. macro-AUPRC versus positive-label retention, both for the key PU/structured
   subset and for all fitted methods;
2. a zero-centered gene-coverage by target-coverage heatmap of
   `Brier(fixed baseline) - Brier(PU-MIRT)`, where one baseline is selected from
   development-validation loss for the whole grid and never chosen per test
   cell; and
3. macro-AUPRC versus same-cell cross-panel prediction sensitivity for datasets
   other than Projection-TAGs, or amplification-confirmed recall at the
   prespecified top-ranked budgets for Projection-TAGs.

Run a notebook from top to bottom in Python 3.10 or newer. The bootstrap uses an
exact 40-character source commit and SHA256 source-tree hash, does not run
`pip`, and does not modify the active environment. Checkpoints and complete
exports remain under the configured `BASE_DIR`; the notebooks themselves stay
clean and contain no executed result claims.
