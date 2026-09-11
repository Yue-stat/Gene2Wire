# Partial gene-panel overlap experiment

This experiment varies overlap between two equal-budget input-gene panels while
keeping the projection-target set and target measurement masks fixed. It is not
the group-by-target block-masking experiment.

For panel size `K`, requested overlap `omega` maps to
`round_half_up(K * omega)` shared genes. The notebooks expose
`OVERLAP_GRID` as an editable Python list and report both requested and realized
overlap. Panel draws are nested and outcome-independent.

The primary crossed design balances A/B pseudo-panels within animal/spatial
strata. BARseq A1 also includes an animal-aligned stress test. Projection-TAGs
uses paired A/B animals in every train/validation/test role and first restricts
analysis to targets with identical measurement coverage in all animals. SPIDER
keeps its contiguous spatial folds. MERGE-seq uses crossed A/B panels within
`(sample, assay_status)`, preserves its five targets and whole-sample outer
folds, and has no sample nuisance term. With only five targets and four samples,
it is a robustness analysis rather than a general low-rank mechanism test.
Simulation uses the same generated truth, outcomes and outer folds across
overlap levels within each repetition.

MERGE-seq freezes a 128-gene candidate source panel using reserved, structurally
unassayed `W=0` design cells before the analysis cohort is fitted or evaluated.
The adapter audits that all five target-mask columns are identical for each cell
and that every sample has both measured and positive support for every target;
it does not call the common-target filtering helper. MERGE-seq's all-`P` oracle
therefore means all fixed 128 candidate genes selected from those reserved
`W=0` design cells, not the whole transcriptome.

The overlap-only adapter removes every `barcode*` assay-derived feature from
both source-panel selection and the library-size denominator before rebuilding
the log1p-normalized expression matrix. Availability masking is then applied to
that biological-transcriptome normalization. This is therefore an in-silico
post-normalization feature-availability experiment; it does not claim that a
literal 64-gene targeted assay would have identical count-compositional
normalization.

For a gene to enter fold-specific scaling, it must be visible in an
outer-training cell. The scaler reads only those visible values. Unavailable
expression is zero after centering, so zero represents the visible training
mean rather than biological absence. Panel and, where estimable, animal effects
enter through a separate nuisance design with target-specific coefficients. The
nuisance term is outside the gene coefficient matrix and therefore outside the
direct/low-rank/joint structural constraint. Every primary model uses the same
nuisance columns and the same fixed `nuisance_l2=1e-4` release-notebook setting;
generic non-overlap workflows retain their backward-compatible default of zero.
No per-gene missingness indicators are added.

Primary models are PU, PU-MIRT and PU-Joint. Explicit controls are
intersection-only PU, a one-fit disjoint-coefficient PU parameterization,
same-K/100% overlap and an all-gene direct-PU oracle. A matched Shared-A versus
Separate-A MIRT ablation uses the same pooled rows, duplicated A/B features,
nuisance design, detector, folds and tuning budget; only the target-loading
sharing constraint changes. This directly tests whether the common targets
connect the two gene panels. Two fully independent panel calibrations/fits are
not exposed in the release notebooks because they change calibration and
pooling simultaneously. An internal partition/stitch specification is retained
for future work, but without a complete two-fit executor any direct request
fails before fitting rather than silently using the one-fit control.

Random forest, Qiao and the separate reference-information suite are disabled.
Default added positive-label loss is zero; Projection-TAGs retains its natural
observation mechanism. Simulation alone has a default-off, union-only
Technical-SAR 80% sensitivity that writes to a separate export. The formal
notebooks default to ten outcome-independent panel draws.

BARseq A1 crossed includes both panel and animal nuisance columns because both
animals and panels occur in training. Its animal-aligned stress test uses only
the panel term because panel and animal are perfectly confounded. M1,
Projection-TAGs, SPIDER and simulation likewise use only estimable panel terms;
outer-test-animal intercepts are never learned or looked up.
MERGE-seq likewise has the panel nuisance column only: sample fixed effects are
not estimated or looked up.

All results retain repetition and fold identifiers. Exports include aggregate,
per-target, per-panel, worst-panel, selected-hyperparameter and overlap-contrast
tables plus PDF figures. `R_M(omega)` subtracts each union model's advantage
over PU at 100% overlap from its advantage at the current overlap. The separate
`Q_A(omega)` contrast analogously compares Shared-A with Separate-A. Every
scientific row and saved figure has an explicit dataset/design/sharing/scenario
title.
