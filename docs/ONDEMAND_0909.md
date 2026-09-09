# OnDemand notebook update — 0909

Use the five root notebooks ending in `_0909.ipynb`. The `_0908.ipynb` files
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
interpolated. Natural Projection-TAGs and other single-condition model dot
plots remain disabled; their tables and paired-label audit remain available.

Plots display in the notebook and save only as PDF, under
`/home/yueyue/gene2wire/figures/0909/`. The additional plot uses its
`detection_only_benchmarks/` subdirectory. Existing plot filenames may retain
the scientific protocol suffix `0908`; the containing directory identifies
the notebook release.

## Full diagnostics and existing results

`SHOW_FULL_DIAGNOSTICS=True` is now the default. Set it to `False` for compact
summaries. The one-minute progress interval and single live worker-status
output are unchanged.

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

## Dated notebook generation

`scripts/build_notebooks_0908.py` defaults to today's UTC `MMDD` filename suffix.
It accepts `--date MMDD` for reproducible release builds. A build replaces files
for that date only and never deletes earlier dated notebooks. Notebook dates
do not change the random seed or other scientific settings.
