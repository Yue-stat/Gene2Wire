# Contributing to Gene2Wire

Gene2Wire keeps `main` as its only long-lived branch. Use a short-lived branch
or fork for review, merge it only after the checks below pass, and remove the
remote topic branch after merging.

## Set up a development environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,experiments,notebooks]"
python -m pytest -q
```

The test suite uses small offline fixtures. It does not download the research
datasets or run the full experiment matrix.

## Know which files to change

| Path | Purpose |
|---|---|
| `src/gene2wire/` | Installable package, estimators, tuning, and public API |
| `src/gene2wire/experiments/` | Shared dataset adapters and experiment orchestration |
| `scripts/notebooks/` | Generators for canonical notebooks |
| `notebooks/` | Generated, output-cleared experiment entry points |
| `tests/` | Unit, regression, adapter, and notebook-contract tests |
| `docs/` | Protocol and execution documentation |
| `archive/notebooks/` | Byte-preserved historical releases; do not regenerate |
| `results/notebook_snapshots/` | Selected executed evidence, separate from entry points |

Do not add implementation patches inside a notebook. Put reusable behavior in
`src/gene2wire/`, test it, and regenerate the affected notebook family from its
builder.

## Regenerate canonical notebooks

Canonical notebook filenames are stable. Each notebook embeds a committed core
SHA, a SHA256 over `src/gene2wire/**/*.py`, and a release date. Commit core
changes before generating a new pin, then run:

```bash
CORE_COMMIT=$(git rev-parse HEAD)
SOURCE_HASH=$(python -c "from gene2wire.experiments.protocol import source_hash; print(source_hash())")
RELEASE_DATE=$(date -u +%m%d)

python scripts/notebooks/build_positive_label_hiding.py \
  --commit "$CORE_COMMIT" --source-hash "$SOURCE_HASH" --date "$RELEASE_DATE"
python scripts/notebooks/build_target_block_masking.py \
  --commit "$CORE_COMMIT" --source-hash "$SOURCE_HASH" --date "$RELEASE_DATE"
python scripts/notebooks/build_gene_panel_overlap.py \
  --commit "$CORE_COMMIT" --source-hash "$SOURCE_HASH" --date "$RELEASE_DATE"
python scripts/notebooks/build_native_target_panels.py \
  --commit "$CORE_COMMIT" --source-hash "$SOURCE_HASH" --date "$RELEASE_DATE"
```

Run notebooks from top to bottom. By default they use
`/home/yueyue/gene2wire`; set `GENE2WIRE_PROJECT_DIR` to use another data,
checkpoint, export, and figure root.

The package version in `pyproject.toml` and `src/gene2wire/__init__.py` must stay
in sync. `CORE_API_VERSION` in `src/gene2wire/runner.py` is separate: bump it
when checkpoint structure, model identity, or execution semantics become
incompatible, not for documentation or notebook-layout changes.

When a notebook embeds a new `CORE_COMMIT`, merge with a fast-forward or merge
commit. Do not squash or rebase away that pinned commit; it must remain reachable
after the temporary branch is removed.

## Before requesting review

```bash
python -m pip check
python -m pytest -q
git diff --check
```

Also confirm that canonical notebooks contain no outputs or execution counts,
their embedded pins agree, and no `.ipynb` files have been added at repository
root. Never replace an archived notebook in place or present an executed result
from one core pin as output from another.
