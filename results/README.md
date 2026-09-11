# Result artifacts

`notebook_snapshots/` preserves selected executed notebooks exactly as they
were produced. They are evidence for a specific run, not the maintained
entry points. Use the output-cleared notebooks under `notebooks/` for new
runs.

Large raw downloads, checkpoints, complete CSV/NPZ exports, and generated
figures belong under the configured `BASE_DIR` and are intentionally ignored by
Git, even when that directory is the repository checkout.

Each dated snapshot directory should state its immutable Gene2Wire core commit
inside the notebook. Machine-specific scheduling values such as `N_JOBS` do
not define a new scientific protocol.
