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
keeps its contiguous spatial folds. Simulation uses the same generated truth,
outcomes and outer folds across overlap levels within each repetition.

For a gene to enter fold-specific scaling, it must be visible in an
outer-training cell. The scaler reads only those visible values. Unavailable
expression is zero after centering, so zero represents the visible training
mean rather than biological absence. A single panel-B nuisance column supplies
a target-specific panel intercept through the ordinary coefficient matrix; no
per-gene missingness indicators are added.

Primary models are PU, PU-MIRT and PU-Joint. Explicit controls are intersection
only, a disjoint-feature panel-separated direct PU parameterization,
same-K/100% overlap and an all-gene direct-PU oracle. The panel-separated arm
has distinct A/B slopes and intercepts, but it uses one selected penalty and
one detection-calibration run rather than two independently calibrated fits.
Random forest, Qiao and the separate reference-information suite are disabled.
Default positive-label loss is zero; Projection-TAGs retains its natural
observation mechanism. The formal notebooks default to ten outcome-independent
panel draws.

This primary v1 does not implement the separate-A MIRT ablation, a free
unpenalized animal-by-target nuisance model, or the optional Tech-SAR 80%
sensitivity analysis. Consequently, the panel-separated control should not be
described as two independent fits, and these notebooks alone do not identify a
shared-A-specific effect. Those extensions require additional statistical-model
and orchestration work rather than a notebook-only switch.

All results retain repetition and fold identifiers. Exports include aggregate,
per-target, per-panel, worst-panel, selected-hyperparameter and overlap-contrast
tables plus PDF figures. The contrast subtracts each model's advantage over PU
at 100% overlap from its advantage at the current overlap.
