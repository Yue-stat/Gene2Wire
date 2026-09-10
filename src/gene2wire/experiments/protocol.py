"""The single scientific profile used by every 0908 paper notebook."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import FitConfig, ModelConfig, TuningConfig

PROTOCOL_VERSION = "0908-v2-balanced"
MODEL_ORDER = ("Logistic", "MIRT", "Joint", "PU", "PU-MIRT", "PU-Joint")


@dataclass(frozen=True)
class Settings:
    n_outer_folds: int = 3
    use_location: bool = False
    use_target_features: bool = False
    n_jobs: int = 32
    parallel_unit: str = "scenario"
    n_repetitions: int = 5
    strategy: str = "full_joint"
    seed: int = 20260908
    paired_fraction: float = 0.20
    loss_rates: tuple[float, ...] = (0., .2, .4, .6, .8)
    candidate_budget: int = 32
    penalties: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10.)
    maxiter: int = 500
    retry_maxiter: int = 1000
    tolerance: float = 1e-8
    init_direct_maxiter: int = 120
    run_information_controls: bool = True
    run_random_forest: bool = True
    run_mechanism_controls: bool = True
    run_calibration_controls: bool = True
    run_qiao: bool = True
    calibration_fractions: tuple[float, ...] = (.2,)
    protocol_version: str = PROTOCOL_VERSION
    supervision_profile: str = "paired_reference"

    def __post_init__(self):
        for field in ("n_outer_folds", "n_jobs", "n_repetitions", "candidate_budget",
                      "maxiter", "retry_maxiter", "init_direct_maxiter"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.n_outer_folds < 2:
            raise ValueError("At least two outer folds are required")
        if self.strategy not in {"full_joint", "staged_rank_l2"}:
            raise ValueError("Unknown tuning strategy")
        if self.parallel_unit not in {"scenario", "fold"}:
            raise ValueError("parallel_unit must be 'scenario' or 'fold'")
        if self.supervision_profile not in {"paired_reference", "assay_only"}:
            raise ValueError("supervision_profile must be 'paired_reference' or 'assay_only'")
        minimum_ok = (self.paired_fraction >= 0 if self.supervision_profile == "assay_only"
                      else self.paired_fraction > 0)
        if not minimum_ok or not self.paired_fraction < 1:
            raise ValueError("paired_fraction must lie strictly between zero and one (zero is allowed for assay_only)")
        if (not self.loss_rates or len(set(self.loss_rates)) != len(self.loss_rates)
                or any(not 0 <= r < 1 for r in self.loss_rates)):
            raise ValueError("loss_rates must be unique probabilities below one")
        if not self.penalties or any(not np.isfinite(x) or x <= 0 for x in self.penalties):
            raise ValueError("Penalties must be positive and finite")
        if any(not 0 < f < 1 for f in self.calibration_fractions):
            raise ValueError("Calibration fractions must lie between zero and one")
        if len(set(self.calibration_fractions)) != len(self.calibration_fractions):
            raise ValueError("calibration_fractions must be unique")

    def scientific_dict(self):
        result = asdict(self)
        result.pop("n_jobs")  # Hardware changes must not invalidate learned models.
        result.pop("parallel_unit")  # Scheduling cannot change the scientific identity.
        return result

    def fit_config(self):
        return FitConfig(maxiter=self.maxiter, retry_maxiter=self.retry_maxiter,
                         tolerance=self.tolerance, init_direct_maxiter=self.init_direct_maxiter)

    def tuning_config(self, n_features: int, n_targets: int):
        maximum = min(n_features, n_targets)
        ranks = tuple(sorted({0, maximum, *(r for r in (1, 2, 4, 8, 16) if r <= maximum)}))
        return TuningConfig(strategy=self.strategy, ranks=ranks,
                            shared_l2=self.penalties, residual_l2=self.penalties,
                            target_l2=self.penalties, anchor_shared_l2=self.penalties[1 if len(self.penalties)>1 else 0],
                            anchor_residual_l2=self.penalties[1 if len(self.penalties)>1 else 0],
                            anchor_target_l2=self.penalties[1 if len(self.penalties)>1 else 0],
                            candidate_budget=self.candidate_budget, include_endpoints=True)

    def models(self):
        models = tuple(ModelConfig(name=name, kind=kind, rank=rank, pu=pu,
                                 use_target_features=self.use_target_features)
                     for name, kind, rank, pu in (
                         ("Logistic", "direct", 0, False), ("MIRT", "lowrank", 1, False),
                         ("Joint", "joint", 1, False), ("PU", "direct", 0, True),
                         ("PU-MIRT", "lowrank", 1, True), ("PU-Joint", "joint", 1, True)))
        if self.supervision_profile == "assay_only":
            return tuple(model for model in models if not model.pu)
        return models


def source_hash() -> str:
    """Hash all installed Python sources, including dataset/experiment adapters."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                    separators=(",", ":")).encode()).hexdigest()[:20]


def scenarios(settings: Settings, *, natural: bool, simulation: bool, sharing_strength=None):
    """Predeclared controls; no dataset outcome can change this matrix."""
    if settings.supervision_profile == "assay_only":
        if natural:
            raise ValueError("assay_only requires one assay outcome, not paired natural observations")
        return [{"analysis": "primary", "mechanism": "assay_only", "loss_rate": 0.,
                 "calibration_fraction": 0., "calibration_spec": "not_available"}]
    if natural:
        return [{"analysis": "primary", "mechanism": "natural", "loss_rate": None,
                 "calibration_fraction": settings.paired_fraction, "calibration_spec": "correct"}]
    result = [{"analysis": "primary", "mechanism": "technical_sar", "loss_rate": rate,
               "calibration_fraction": settings.paired_fraction, "calibration_spec": "correct"}
              for rate in settings.loss_rates]
    if simulation and settings.run_mechanism_controls:
        result += [{"analysis": "mechanism", "mechanism": mechanism, "loss_rate": .8,
                    "calibration_fraction": settings.paired_fraction, "calibration_spec": "correct"}
                   for mechanism in ("scar", "target_sar")]
    if simulation and settings.run_calibration_controls and sharing_strength in (0., 1.):
        result += [{"analysis": "calibration_size", "mechanism": "technical_sar", "loss_rate": .8,
                    "calibration_fraction": fraction, "calibration_spec": "correct"}
                   for fraction in settings.calibration_fractions if fraction != settings.paired_fraction]
        result += [{"analysis": "calibration_misspecification", "mechanism": "technical_sar", "loss_rate": .8,
                    "calibration_fraction": settings.paired_fraction, "calibration_spec": spec}
                   for spec in ("omit_technical", "pooled_target")]
    return result
