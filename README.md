# Gene2Wire core

Gene2Wire is a data-agnostic implementation of matched logistic, PU-logistic,
MIRT, PU-MIRT, joint, and PU-joint projection models.

This repository deliberately contains only the reusable model layer. Raw-data
download, dataset-specific filtering, feature construction, hiding mechanisms,
splitting, calibration, evaluation, and plotting belong in dataset notebooks.

## Dataset notebooks

Dataset-specific processing and evaluation are intentionally kept outside this
repository. Companion notebooks pin an immutable core commit and may save
resumable checkpoints/results to persistent storage.

## Core data contract

| Value | Shape | Meaning |
|---|---:|---|
| `X_cell` | cells × features | Numeric cell predictors |
| `S_observed` | cells × targets | Observed positives; zero is PU-unlabeled |
| `W_measured` | cells × targets | Entries allowed to enter the likelihood |
| `Y_target` | targets × covariates | Optional target features |
| `exposure` | broadcastable to cells × targets | Detection probability for PU models, with `q = exposure × p` |

Pre-hide truth and hidden-positive masks are evaluation objects. They must stay
outside the core. `run_model_grid` intentionally accepts outer-test `X` rather
than an outer-test label bundle.

## Install

For a reproducible run, install a commit SHA rather than `main`:

```bash
python -m pip install \
  "git+https://github.com/Yue-stat/Gene2Wire-core.git@<CORE_COMMIT_SHA>"
```

## Generic API

```python
from gene2wire import (
    DatasetBundle,
    FitConfig,
    ModelConfig,
    TuningConfig,
    run_model_grid,
)

result = run_model_grid(
    train=train_bundle,                 # X, post-hiding observed labels, W
    validation=validation_bundle,
    test_X=X_test,                      # no test truth enters the core
    train_exposure=e_train,
    validation_exposure=e_validation,
    test_exposure=e_test,
    test_cell_ids=test_ids,
    models=(
        ModelConfig(name="Logistic", kind="direct", pu=False),
        ModelConfig(name="PU logistic", kind="direct", pu=True),
        ModelConfig(name="MIRT", kind="lowrank", rank=4, pu=False),
        ModelConfig(name="PU-MIRT", kind="lowrank", rank=4, pu=True),
        ModelConfig(name="Joint", kind="joint", rank=4, pu=False),
        ModelConfig(name="PU-Joint", kind="joint", rank=4, pu=True),
    ),
    tuning=TuningConfig(
        strategy="full_joint",
        ranks=(2, 4, 8, 12),
        shared_l2=(1e-5, 1e-3, 1e-2),
        residual_l2=(1e-5, 1e-3, 1e-2),
        target_l2=(1e-3,),
    ),
    fit=FitConfig(maxiter=500),
    checkpoint_dir="/persistent/path/checkpoints",
    unit_context={"outer_fold": 0, "condition": "condition_a"},
    seed=20260903,
    code_version="<CORE_COMMIT_SHA>",
)

latent_p = result.models["PU-MIRT"].latent_probability
observed_q = result.models["PU-MIRT"].observed_probability
selected_hyperparameters = result.summary_rows()
```

`tuning` and `fit` may each be either one shared configuration or a mapping
from model name to configuration, allowing different datasets/models to use
different grids.

## Resume guarantees

When `checkpoint_dir` is supplied:

1. Every completed hyperparameter candidate is saved atomically.
2. Every completed selected/refitted model saves checksummed fitted parameters
   and test predictions.
3. The fingerprint covers arrays, IDs/order, exposure, hyperparameters, seed
   policy, semantic unit context, and pinned code version.

After a runtime disconnect, completed candidates are skipped and completed
models are restored without refitting. Changed data or settings produce a new
fingerprint rather than silently reusing stale results.

## Validation

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Tests cover gradients, grid construction, reference-truth isolation, atomic
array integrity, and completed-model resume.
