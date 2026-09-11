# Gene-overlap v2 validation contract

This document fixes the scientific meaning and release tests for the optional
gene-overlap extensions.  The primary estimand remains the effect of input-gene
panel overlap at a constant per-cell assay budget `K`, with projection targets,
outer folds, panel assignments, calibration rows and outcome draws held fixed.

## Analysis roles

| Component | Role | Required pooling |
|---|---|---|
| `union` PU / PU-MIRT / PU-Joint | Primary | One pooled fit across panels |
| `intersection` PU | Primary control | Common genes only |
| `disjoint_coefficient` PU | Primary control | Distinct A/B slopes in one fit; pooled detector and tuning |
| `shared_a` PU-MIRT | Mechanism ablation | Panel-specific gene slopes and one shared target loading matrix `A` |
| `separate_a` PU-MIRT | Mechanism ablation | One matched pooled fit with separate target loading matrices for the A/B feature blocks |
| `independent_panel` PU | Deferred developer specification | Model fitting and detector calibration would both be separated by panel |
| `all_gene_oracle` PU | Upper-bound control | All genes in the declared source panel; never interpreted as the 100% same-`K` endpoint |

The matched Shared-A versus Separate-A comparison uses the same duplicated
panel-specific gene representation, pooled rows, nuisance design, detector,
validation split and tuning budget. `shared_a` fits one target loading matrix;
`separate_a` fits one loading matrix per declared panel-feature block. Otherwise
the contrast would also change common-gene coefficient pooling or calibration
and would not isolate sharing of `A`.

`independent_panel` is a declared future developer sensitivity, not a runnable
release-notebook arm. The notebooks do not expose a switch for it. The internal
spec must fail preflight while the shared pipeline does not have a partition-aware
two-fit executor. This is preferable to silently running the old one-fit disjoint
parameterization under an independent-fit label.

The ordinary cell-gene matrix contains genes only.  Panel effects enter the core
through a separate nuisance design and target-specific nuisance coefficients.
The same nuisance schema and fixed `nuisance_l2=1e-4` are used by PU, PU-MIRT
and PU-Joint. This term is structurally free of the gene low-rank constraint;
the tiny fixed ridge prevents divergence under panel-by-target separation and is
never tuned. Panel nuisance is valid because both panel types occur in development
and test.  Animal fixed effects must not be created for animals absent from the
outer-training data; in particular, Projection-TAGs must not estimate or look up
an outer-test-animal intercept.

## Fixed axes

For every repetition, panel A and panel B each contain exactly `K` available
genes at every overlap.  Only the number of common genes changes.  The candidate
rank grid is also fixed across overlap levels within one dataset/design/repetition;
it must not grow merely because the union has more columns at low overlap.  The
configured rank cap is recorded with every view and result row.

The no-added-label-loss run is the only primary gene-overlap run.  Optional
Technical-SAR at 80% added positive-label loss is a separately named sensitivity
artifact, uses union views only, and is enabled only for generated simulation.
It is never silently added to BARseq, SPIDER or Projection-TAGs.  Projection-TAGs
keeps its native standard-versus-amplified observation mechanism and explicitly
rejects the synthetic Technical-SAR helper.

## MERGE-seq contract

MERGE-seq is the fifth generated gene-overlap notebook. It builds crossed panels
within `('sample', 'assay_status')`, uses the unchanged five projection targets
and whole-sample outer folds, and includes no sample nuisance effects. Because it
has only five targets across four samples, it is a CV robustness analysis rather
than a general low-rank mechanism claim. Its adapter must audit identical `W`
columns and measured plus positive support for every target in every sample,
without invoking `common_target_dataset`.

The MERGE-seq source panel is fixed at 128 candidate genes selected using only
reserved structurally unassayed `W=0` design cells. Accordingly, its all-`P`
oracle is all fixed 128 candidate genes, not the whole transcriptome. The
release notebook runs the editable overlap grid with random forest off and ten
outcome-independent repetitions; its export is `MERGE_seq_gene_overlap_0910`.
The overlap-only normalization removes every `barcode*` assay feature from both
the source-panel candidates and the library-size denominator before applying
the availability masks. This is explicitly a post-normalization feature-
availability experiment, rather than a literal targeted-count assay model.

## Mandatory automated tests

1. **Panel design:** equal `K`, nested common genes, expected odd-`K` rounding,
   fixed panel assignment, and unchanged targets/outcomes/folds across overlap.
2. **Rank grid:** every overlap view has the same declared rank cap and produces
   the same rank candidates despite different union dimensions.
3. **Nuisance boundary:** nuisance is absent from `FeatureSet.X`, gene names and
   gene feature blocks; it has a separate aligned matrix/name schema.  All three
   primary models receive the same nuisance matrix and fixed penalty.
4. **Independent-panel guard:** while no partition-aware executor exists, the
   optional arm must fail before fitting. Row partition, feature extraction and
   prediction-stitching helpers must cover each authorized row exactly once.
   If the executor is later enabled, poisoning panel-B labels/features must not
   change its panel-A fit or predictions.
5. **Shared-A match:** `shared_a` and `separate_a` have identical pooled inputs,
   nuisance, detector exposure and model-selection data; only the target-loading
   sharing constraint differs. The selected rank/penalty is exported directly.
6. **Sensitivity isolation:** the helper rejects non-simulation and
   natural-observed collections, selects only union views from a normal mixed-arm
   primary collection, and fixes added positive-label loss at 80%. Primary and
   sensitivity exports, filenames, titles and summary groups cannot mix.
7. **Resume/accounting:** cache identities include arm, grouped-low-rank structure,
   nuisance design and sensitivity scenario. If a partition-aware executor is
   added later, fit panel and detector policy must also enter its identity.
8. **Reporting:** metrics are recorded once per model/fold/repetition;
   panel components are not treated as independent repetitions. `R_M(omega)` is
   restricted to union PU/MIRT/Joint rows within one isolated observation
   scenario; primary and Technical-SAR sensitivity rows are never pooled.

## Runtime guardrails

At the current six-point penalty grid and candidate budget 32, the v2 design
requires roughly 432--442 optimizer calls per fold/repetition for all five
overlaps. Adding the matched Shared-A and grouped Separate-A MIRT ablation is
still expensive, although materially cheaper than two independently tuned MIRT
pipelines. The fully independent PU pipeline remains optional. Adding
Technical-SAR to every arm would approximately double total work and is
prohibited.

The simulation-only 80% sensitivity must use union views only and a separate
export. Numerical end-to-end tests use tiny data and minimal one-candidate
grids; information-boundary and orchestration tests mock optimization. The
current executor materializes Shared-A and Separate-A feature views separately;
SPIDER can therefore require several additional gigabytes. Reusing identical
prepared feature matrices is a future performance optimization and must not
change scientific fingerprints or results.
