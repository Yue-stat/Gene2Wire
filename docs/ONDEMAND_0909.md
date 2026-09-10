# OnDemand notebook update — 0909

Use the five primary notebooks or five separate block-masking notebooks ending
in `_0909.ipynb`. The `_0908.ipynb` files
remain as released, with their original code pins. All new notebooks use the
same core source checksum, scientific settings and model implementations.
Dataset adapters continue to define their documented inputs and split policies.
The detailed environment and experiment instructions are in
[ONDEMAND_0908.md](ONDEMAND_0908.md); the differences for this release follow.

## Additional comparison figure

The new **Benchmarks without reference labels as training samples** section
calls `plot_detection_only_benchmark_results`. It includes recorded Logistic,
MIRT, Joint, PU logistic, PU-MIRT, PU-Joint, RF (observed labels), Qiao variants
and sensitivity rescaling. PU and rescaling can still estimate detection from
the paired subset; exclusion concerns direct predictor training on reference
outcomes, not access to calibration information.

Reference-only logistic, all Reference + PU variants, RF (paired references),
and RF (observed + paired references) are excluded from this additional plot.
Existing primary, all-benchmark and three-method paired-budget plots remain.
Solid lines use PU or sensitivity calibration; observed-label baselines are
dashed. Each scientific setting stays separate. Missing fits are not
interpolated. Natural Projection-TAGs model comparisons now use horizontal bars,
alongside its paired-label audit. Removing single-condition dots previously
caused those comparisons to return empty dictionaries; this is now fixed.
Synthetic single-condition dot plots remain disabled by default.

Plots display in the notebook and save only as PDF, under
`/home/yueyue/gene2wire/figures/0909/`. The additional plot uses its
`detection_only_benchmarks/` subdirectory. Existing plot filenames may retain
the scientific protocol suffix `0908`; the containing directory identifies
the notebook release.

## Bounded essential diagnostics and existing results

`SHOW_FULL_DIAGNOSTICS=False` is the default in all current notebooks. The report
retains aggregate metrics, important-model configurations by repetition,
structure/convergence evidence, calibration and matched block/control diagnostics.
It explicitly bounds rows/columns and reports omissions; the retained small
tables render fully instead of allowing pandas to remove middle columns.
Set the switch to `True` only to print every raw table; that output can be large.
Full result exports are preserved in both modes.

The live worker cell leads with the kernel's detectable CPU allowance and shows
machine, affinity, cgroup and scheduler metadata separately. It also shows active
workers for this experiment. It does not subtract running tasks from pool slots
and call the remainder globally available CPUs. Globally idle node capacity is
unknown to the notebook. The one-minute progress interval is unchanged.

For completed runs, set `RESULTS_ONLY=True` and fill `EXISTING_EXPORT_DIRS` with
the exact run directories containing `manifest.json` and `metrics.csv`. BARseq
expects both `A1` and `M1` keys; the other notebooks expect their displayed
dataset key. Run the notebook from the configuration cell after restarting
the kernel. Raw loading and fitting are skipped; figures and diagnostic tables
use the saved manifest and metrics.

This release changes display code, not model objectives, tuning candidates or
mask generation. Nevertheless, the current conservative source checksum hashes
all package Python files, including plotting. Starting a new training run with
the new pin creates a new code identity. Use results-only mode to add these
figures to existing results without repeating training.

Raw files stay under `raw_data/`, exports under `paper_figure_exports/`, and
checkpoint files under `checkpoints/0908/`. The latter is the scientific
protocol cache, not the notebook release date; existing files are preserved.

The separate block notebooks use `checkpoints/block_masking_v1` and the
`*_target_block` export families. SPIDER now has
`SPIDER_block_masking_0909.ipynb`, with two balanced artificial groups because the
current processed adapter does not supply verified animal IDs. It retains all
existing input gene features and calls the same model and masking pipeline as
the other block notebooks. See [the block protocol](BLOCK_MASKING_0909.md).

## Dated notebook generation

`scripts/build_notebooks_0908.py` defaults to today's UTC `MMDD` filename suffix.
It accepts `--date MMDD` for reproducible release builds. A build replaces files
for that date only and never deletes earlier dated notebooks. Notebook dates
do not change the random seed or other scientific settings.
