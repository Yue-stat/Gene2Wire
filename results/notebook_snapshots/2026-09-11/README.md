# Executed notebook snapshots — 2026-09-11

These files are the user-supplied executed result notebooks from the September
11 runs. Filenames were normalized during repository consolidation; notebook
cells and outputs were otherwise preserved byte-for-byte.

- [`target_block_masking/`](target_block_masking/README.md): six group-by-target block-missingness runs. Their
  source matches the corresponding pre-consolidation 0911 branch notebooks
  except for machine-specific `N_JOBS` values. They pin core commit `040a303`.
- [`gene_panel_overlap/`](gene_panel_overlap/README.md): four completed overlap-grid runs corresponding to the
  pre-consolidation overlap branch. They use `N_REPETITIONS=4` rather than that
  branch's default of 10 and pin core commit `bd77d41`.

The consolidated canonical notebooks use a newer shared core pin. These saved
outputs therefore remain evidence for their recorded commits and must not be
reported as results from the consolidated pin.

The unexecuted MERGE-seq overlap notebook is not duplicated here. Its exact
source is preserved in Git history and its repaired, output-cleared version is
maintained under `notebooks/gene_panel_overlap/`.
