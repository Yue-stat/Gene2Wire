"""Gene2Wire: data-agnostic PU multitask model selection."""

from .adapters import load_npz_bundle, save_npz_bundle
from .adaptive import (
    AdaptiveFitConfig,
    AdaptiveFittedModel,
    AdaptiveModelConfig,
    AdaptivePUModel,
    per_target_validation_metrics,
)
from .checkpoint import (
    AtomicArrayCheckpointStore,
    AtomicCheckpointStore,
    experiment_fingerprint,
    hash_named_files,
    sha256_array,
    sha256_source_tree,
    unit_key,
)
from .config import ExperimentConfig, FitConfig, ModelConfig, TuningConfig, load_config
from .data import DatasetBundle
from .metrics import BrierReport, brier_report, masked_brier, masked_log_loss
from .models import FittedModel, UnifiedPUModel
from .matched import (
    AdaptiveTrialResult,
    MATCHED_API_VERSION,
    MatchedGridRunResult,
    MatchedModelConfig,
    MatchedModelRunResult,
    MatchedTuningConfig,
    run_matched_model_grid,
)
from .runner import CORE_API_VERSION, GridRunResult, ModelRunResult, run_model_grid
from .tuning import TrialResult, TuningResult, full_joint_candidates, tune_model

__all__ = [
    "AtomicArrayCheckpointStore",
    "AtomicCheckpointStore",
    "AdaptiveFitConfig",
    "AdaptiveFittedModel",
    "AdaptiveModelConfig",
    "AdaptivePUModel",
    "AdaptiveTrialResult",
    "BrierReport",
    "CORE_API_VERSION",
    "DatasetBundle",
    "ExperimentConfig",
    "FitConfig",
    "FittedModel",
    "GridRunResult",
    "MATCHED_API_VERSION",
    "MatchedGridRunResult",
    "MatchedModelConfig",
    "MatchedModelRunResult",
    "MatchedTuningConfig",
    "ModelConfig",
    "ModelRunResult",
    "TrialResult",
    "TuningConfig",
    "TuningResult",
    "UnifiedPUModel",
    "brier_report",
    "experiment_fingerprint",
    "full_joint_candidates",
    "hash_named_files",
    "load_config",
    "load_npz_bundle",
    "masked_brier",
    "masked_log_loss",
    "per_target_validation_metrics",
    "run_matched_model_grid",
    "run_model_grid",
    "save_npz_bundle",
    "sha256_array",
    "sha256_source_tree",
    "tune_model",
    "unit_key",
]

__version__ = "0.3.0"
