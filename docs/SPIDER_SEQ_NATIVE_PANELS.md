# SPIDER-Seq: naturally incomplete target panels

Run `SPIDER_Seq_natural_panels_0911.ipynb` in the existing OnDemand environment.
This uses the authors' scRNA-seq `Adult.Ex.rds`, not spatial `sp.PFC.rds`.
The new notebook preserves earlier dated notebooks and pins its own shared core.

## Audited inputs

- Paper: [SPIDER-Seq, DOI 10.1093/nsr/nwag004](https://doi.org/10.1093/nsr/nwag004).
- [Author FigureS3.R](https://github.com/ZhengTiger/SPIDER-Seq/blob/047eb5aaa15087c241570745bd6df07882b0dd78/Supplementary%20Figures/FigureS3.R)
  and Supplementary Table S2 agree after explicit spelling normalization.
- [Processed data](https://huggingface.co/spaces/TigerZheng/SPIDER-web/blob/22e37e94bc5a5ecda2185f2af4eae5cf49c6092a/data/Adult.Ex.rds)
  contains 15,791 excitatory cells and an RNA count matrix with 26,902 genes.
- Adult1/2/3 have 12/14/16 measured targets, union 24. Pairwise overlaps are
  6, 6 and 12 targets. `spider_seq_panel_manifest.csv` records all 42 injections;
  `spider_seq_source_audit.json` records source revisions and checksums.

The adapter retains this excitatory cohort without adding barcode-based cell
filters. It reads raw RNA counts, not the author integrated assay or embeddings.
Processed barcode metadata >0 defines assay-positive calls; source raw-count
thresholds are not applied again. The measurement mask comes from injection
design, independently of whether any positive calls appear. Off-panel metadata
must remain unmeasured; it never becomes a negative training label.

The first run verifies and caches the 642 MB RDS, then creates sparse, pickle-free
processed files. Later runs reuse those validated files without downloading or
parsing the RDS again. Gene selection and PCA remain split-local, not cached as
whole-cohort fitted transformations.

## Training and evaluation

Defaults: three folds, five repetitions, 32 requested workers, location and target
features disabled, the shared `full_joint` search with candidate budget 32.
Logistic, MIRT, Joint, RF-observed and both Qiao formulations use the same feature
and label budget. The Qiao comparison uses target IDs unless independent target
descriptors are supplied. Joint receives the native 32-candidate shared-plus-
specific search plus the exact selected Logistic and MIRT endpoints; endpoint
fits are reused through the common caches. No estimator or tuning grid is
customized for SPIDER-Seq. The companion `SPIDER_Seq_block_masking_0911.ipynb`
evaluates the same native animal panels under held-out target blocks; its masked
test entries have known original references and are therefore scored separately.

Cell counts are normalized to 10,000 and log transformed. The default selects
2,000 training-variable genes and fits 50 training-only principal components.
`N_HVG` and `N_GENE_COMPONENTS` are exposed in the notebook; setting components to
`None` uses selected gene features directly. Validation and test data do not fit
these transformations. Final refits fit transformations on development cells.
The cell-local normalization is computed once, while gene selection and PCA stay
split-specific. At the default five repetitions and three folds, the run prepares
30 transforms (tuning and final refit for each fold) and reports this stage
separately before model-unit progress starts.
Location and target switches require aligned, independently measured CSV files;
no anatomical target descriptors are inferred from barcode outcomes.

Each repetition changes within-animal train/validation/test cell splits. All three
animals occur in every split role, and each cell is tested once per repetition.
Targets unassayed in an animal remain absent from its fitting and validation
support. Scores are calculated only on native measured test entries. Per-animal
metrics are reported because the measured target mix differs between animals.
Repetitions describe split/model variability, not additional biological animals.

No artificial target-block or positive-label thinning is used. No independent
paired reference is available. The generic `assay_only` profile therefore fits
assay-positive probabilities with no detection calibrator, no paired-reference
controls and no hidden-positive recovery score. Internally, unit additional
retention does not mean perfect biological detection. PU aliases would have the
same likelihood as their corresponding ordinary structures here and are omitted.

## Outputs and interpretation

Outputs remain under `/home/yueyue/gene2wire/paper_figure_exports`. Model checkpoints
use `checkpoints/native_panels_v2`; raw/processed files use `raw_data/SPIDER_Seq`.
Progress is reported once per minute in Los Angeles time. Worker status separates
kernel CPU allowance from running experiment workers.

The notebook displays and saves PDF figures for real panel availability,
measured-test model comparisons, animal-specific comparisons and unassayed
forecasts. Compact diagnostics default to `SHOW_FULL_DIAGNOSTICS=False` and retain
actual model selections, convergence and validation-selection evidence.

Original per-fold prediction matrices include all targets. `native_unassayed/`
contains out-of-fold mean/SD/count matrices and optional compressed per-cell CSVs
for W=0 entries only. These rows have missing reference labels and
`evaluation_eligible=False`. Their SD across splits is not a confidence or
prediction interval. Observed-panel test performance cannot validate the true
native gaps; those forecasts are hypotheses for further measurement. A full-panel
control cannot be reconstructed without actually measuring the missing labels.

For replotting, set `RESULTS_ONLY=True` and supply the exact completed run directory
in `EXISTING_EXPORT_DIRS={'SPIDER-Seq': ...}`. This restores saved settings and
predictions without loading RNA counts or fitting models.
