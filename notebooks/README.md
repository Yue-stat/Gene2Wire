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
