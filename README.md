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

## Matched adaptive API

For grouped experiments that require float32 Adam, equal candidate budgets,
warm-started rank screening, and an exact direct-model fallback, use the second
generic runner:

```python
from gene2wire import (
    AdaptiveFitConfig,
    DatasetBundle,
    MatchedModelConfig,
    MatchedTuningConfig,
    run_matched_model_grid,
)

result = run_matched_model_grid(
    train=train_bundle,
    validation=validation_bundle,
    test_X=X_test,                       # still no outer-test labels
    train_exposure=e_train,
    validation_exposure=e_validation,
    test_exposure=e_test,
    validation_target_mask=eligible_targets,
    models=(
        MatchedModelConfig("Direct", "direct", pu=False),
        MatchedModelConfig("PU direct", "direct", pu=True),
        MatchedModelConfig("Low rank", "lowrank", pu=False),
        MatchedModelConfig("PU low rank", "lowrank", pu=True),
        MatchedModelConfig("Shared + residual", "joint", pu=False),
        MatchedModelConfig("PU shared + residual", "joint", pu=True),
    ),
    tuning=MatchedTuningConfig(
        direct_l2=(0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
        learning_rates=(0.005, 0.01),
        ranks=(1, 2, 3, 4, 5, 6, 7),
        shared_l2=(1e-6, 1e-5, 1e-4, 1e-3),
        residual_l2=(1e-5, 1e-4, 1e-3, 1e-2),
    ),
    fit=AdaptiveFitConfig(max_epochs=70, batch_size=8192, patience=8),
    checkpoint_dir="/persistent/path/checkpoints",
    unit_context={"outer_fold": 0},
    seed=42,
    code_version="<CORE_COMMIT_SHA>",
)
```

This runner gives every family the same candidate budget. Structured searches
contain a frozen rank-zero copy of the matched direct parent, warm-start ranks
from its coefficient matrix, refine the best nonzero rank, and accept the
refitted structured model only through the paired target-level stability gate.
If the gate rejects, returned fitted parameters and predictions are exactly the
direct parent's. Final training remains on `train`; `validation` is used only
for candidate selection and early stopping.

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
array integrity, candidate/model resume, matched budgets, and exact structured
fallback identity.
