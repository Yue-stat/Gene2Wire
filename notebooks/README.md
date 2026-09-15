# Gene2Wire notebooks

This directory contains the canonical, output-cleared experiment entry points.
Their filenames are stable; each notebook records its immutable core commit,
source checksum, and release date internally. Historical releases are preserved
under [`archive/notebooks/`](../archive/notebooks/).

## Positive-label hiding

These experiments retain the assay design and turn selected positive labels into
unlabeled observations.

| Dataset | Notebook |
|---|---|
| BARseq A1 and M1 | [`barseq.ipynb`](positive_label_hiding/barseq.ipynb) |
| MERGE-seq | [`merge_seq.ipynb`](positive_label_hiding/merge_seq.ipynb) |
| Projection-TAGs | [`projection_tags.ipynb`](positive_label_hiding/projection_tags.ipynb) |
| SPIDER spatial | [`spider_spatial.ipynb`](positive_label_hiding/spider_spatial.ipynb) |
| Simulation | [`simulation.ipynb`](positive_label_hiding/simulation.ipynb) |

Builder: [`build_positive_label_hiding.py`](../scripts/notebooks/build_positive_label_hiding.py)

## Target-block masking

These experiments hide group-by-target assay blocks; both positive and negative
outcomes in a hidden block become unmeasured.

| Dataset | Notebook |
|---|---|
| BARseq A1 | [`barseq_a1.ipynb`](target_block_masking/barseq_a1.ipynb) |
| BARseq M1 | [`barseq_m1.ipynb`](target_block_masking/barseq_m1.ipynb) |
| MERGE-seq | [`merge_seq.ipynb`](target_block_masking/merge_seq.ipynb) |
| Projection-TAGs | [`projection_tags.ipynb`](target_block_masking/projection_tags.ipynb) |
| SPIDER spatial | [`spider_spatial.ipynb`](target_block_masking/spider_spatial.ipynb) |
| SPIDER-Seq | [`spider_seq.ipynb`](target_block_masking/spider_seq.ipynb) |
| Simulation | [`simulation.ipynb`](target_block_masking/simulation.ipynb) |

Builder: [`build_target_block_masking.py`](../scripts/notebooks/build_target_block_masking.py)

## Combined measurement degradation

These experiments jointly vary union-preserving three-assay gene coverage,
target coverage, and fold-local assay-by-target positive retention. Coverage is
the fraction measured by each assay, not overlap between fixed-size panels:
BARseq at 100% gene coverage measures all 23 genes in each of A/B/C. Artificial
off-panel target entries remain unmeasured through `W_fit`, while every
condition's headline metric uses the fixed native outer-test reference scope.
See the [complete protocol](../docs/MEASUREMENT_DEGRADATION.md).

| Dataset | Notebook |
|---|---|
| Simulation | [`simulation.ipynb`](measurement_degradation/simulation.ipynb) |
| BARseq A1 | [`barseq_a1.ipynb`](measurement_degradation/barseq_a1.ipynb) |
| BARseq M1 | [`barseq_m1.ipynb`](measurement_degradation/barseq_m1.ipynb) |
| MERGE-seq | [`merge_seq.ipynb`](measurement_degradation/merge_seq.ipynb) |
| Projection-TAGs | [`projection_tags.ipynb`](measurement_degradation/projection_tags.ipynb) |
| SPIDER spatial | [`spider_spatial.ipynb`](measurement_degradation/spider_spatial.ipynb) |
| SPIDER-Seq | [`spider_seq.ipynb`](measurement_degradation/spider_seq.ipynb) |

Builder:
[`build_measurement_degradation.py`](../scripts/notebooks/build_measurement_degradation.py)

The notebooks retain the existing 15 fitted methods and add three explicitly
labelled comparators: `GenEML-authors-mask`, paper-based
`Inductive-PU-MC`, and the pinned authors-source SAR-EM kernel executed through
an adapted current-scikit-learn estimator and outer per-target `SAR-PU`
wrapper with a centered-orthonormal assay/QC propensity design—not the authors'
unmodified end-to-end program. They display panel,
support, censoring, selected-parameter, metric, convergence/failure, runtime,
and checkpoint diagnostics plus the retention curve, coverage heatmap, and the
dataset-appropriate recovery or panel-sensitivity figure. They run in the
source-pinned local/OnDemand Python environment and have no Colab copies.

### Results-only V3 figures

[`measurement_degradation_v3/`](measurement_degradation_v3/) contains a
separate, read-only visualization family. It leaves the notebooks above and the
completed result exports unchanged. Each degradation curve runs from 100% at
the left toward 0% at the right and pairs Macro-AUPRC with Macro log loss. When
a verified export is present, each notebook also emits exactly two
funkyheatmap-style scorecards (key and full models) over the declared mechanism
× degree × metric grid, following the
[funkyheatmap visual grammar](https://funkyheatmap.github.io/funkyheatmap/)
without adding a runtime package dependency.

The five supplied completed runs—simulation, BARseq A1/M1, MERGE-seq, and
Projection-TAGs—are pinned by exact export run-ID directory. SPIDER spatial and
SPIDER-Seq intentionally retain a missing explicit path and fail closed until a
verified completed run directory is entered; they never fall back to fitting.
Projection-TAGs additionally recomputes conditional candidate probabilities and
amplification-reference precision, FDR, FPR, F1, lift, AUPRC, AUROC, Brier, and
log loss from saved OOF prediction arrays without retraining.

Builder:
[`build_measurement_degradation_v3.py`](../scripts/notebooks/build_measurement_degradation_v3.py)

### Six-model V4 experiments

[`measurement_degradation_v4/`](measurement_degradation_v4/) contains seven
new runnable experiments. They fit only PU, PU-MIRT, Gene2Wire (internal ID
`PU-Joint`), GenEML, PU matrix completion, and SAR-PU. The grid uses gene and
target coverages `(1.0, 0.7, 0.4)` and positive retentions
`(1.0, 0.8, 0.6, 0.4, 0.2)`. Its complete `3 x 3 x 5` factorial supplies
literal one-factor gene, target, and positive-label-loss slices plus the
all-condition harmonic-mean curve. It produces four configurable multi-metric curve
families and one native `funkyheatmappy` scorecard; it does not create the
legacy full-model curves or gene-by-target contrast heatmap.

The line-plot cells expose their metric tuple first and compare exactly the two
requested model sets. The funkyheatmap cell separately exposes its metric
tuple and compares exactly Gene2Wire, GenEML, PU matrix completion, and SAR-PU.
Every metric is converted to higher-is-better utility (losses are negated),
then min-max scaled within sharing strength, physical condition, and metric.
Blue `funkyrect` glyphs therefore run from a small circle at the minimum to a
square-like maximum, with no number printed inside. Requested/theoretical
coverage labels are used; realized integer coverage remains in audit tables.
The scorecard contains the nine one-factor physical conditions once; the full
factorial is summarized by the combined line rather than duplicated as columns.
All six models share the same splits, known measurement mask, observed labels,
and evaluation scope. Method-specific authorized inputs are disclosed, and a
paired-calibration exposure estimate is supplied only to estimators designed
to consume it. Hidden Recall@H is restricted to fitted-panel test entries with
`D=0`; its measured-entry, unlabeled-candidate, and hidden-positive supports are
exported with the metric.

Builder:
[`build_measurement_degradation_v4.py`](../scripts/notebooks/build_measurement_degradation_v4.py)

## Gene-panel overlap

These experiments change which input-gene columns are visible while holding the
projection targets, target labels, assay mask, and outer folds fixed.

| Dataset | Notebook |
|---|---|
| BARseq | [`barseq.ipynb`](gene_panel_overlap/barseq.ipynb) |
| MERGE-seq | [`merge_seq.ipynb`](gene_panel_overlap/merge_seq.ipynb) |
| Projection-TAGs | [`projection_tags.ipynb`](gene_panel_overlap/projection_tags.ipynb) |
| SPIDER spatial | [`spider_spatial.ipynb`](gene_panel_overlap/spider_spatial.ipynb) |
| Simulation | [`simulation.ipynb`](gene_panel_overlap/simulation.ipynb) |

Builder: [`build_gene_panel_overlap.py`](../scripts/notebooks/build_gene_panel_overlap.py)

## Native target panels

The native-panel workflow uses the naturally overlapping SPIDER-Seq Adult1/2/3
assay panels without synthetic target masking or positive-label thinning.

- [`spider_seq.ipynb`](native_target_panels/spider_seq.ipynb)

Builder: [`build_native_target_panels.py`](../scripts/notebooks/build_native_target_panels.py)

Run canonical notebooks from top to bottom. Generated figures, checkpoints, raw
data, predictions, and full metric exports live under the configurable
`BASE_DIR` and are ignored by Git. Selected executed notebooks may be preserved
under [`results/notebook_snapshots/`](../results/notebook_snapshots/); they are
not interchangeable with these clean entry points.
