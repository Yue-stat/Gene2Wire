"""Typed hyperparameter interface and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class FitConfig:
    maxiter: int = 400
    tolerance: float = 1e-8
    initialization: str = "svd"
    init_direct_maxiter: int = 100

    def __post_init__(self) -> None:
        if (
            not isinstance(self.maxiter, int)
            or isinstance(self.maxiter, bool)
            or not isinstance(self.init_direct_maxiter, int)
            or isinstance(self.init_direct_maxiter, bool)
            or self.maxiter < 1
            or self.init_direct_maxiter < 1
        ):
            raise ValueError("iteration limits must be positive")
        if not _finite_number(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if self.initialization not in {"svd", "random"}:
            raise ValueError("initialization must be 'svd' or 'random'")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    kind: str
    rank: int = 0
    shared_l2: float = 0.0
    residual_l2: float = 0.0
    use_target_features: bool = False
    target_l2: float = 0.0
    pu: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model name must be a nonempty string")
        if self.kind not in {"direct", "lowrank", "joint"}:
            raise ValueError("kind must be direct, lowrank, or joint")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        if self.kind == "lowrank" and self.rank < 1:
            raise ValueError("lowrank requires rank >= 1")
        penalties = (self.shared_l2, self.residual_l2, self.target_l2)
        if any(not _finite_number(x) or x < 0 for x in penalties):
            raise ValueError("L2 penalties must be finite and nonnegative")
        if not isinstance(self.pu, bool) or not isinstance(self.use_target_features, bool):
            raise ValueError("pu and use_target_features must be booleans")
        if self.kind == "direct" and self.rank != 0:
            raise ValueError("rank is irrelevant for direct and must be 0")
        if self.kind == "direct" and self.shared_l2 != 0:
            raise ValueError("shared_l2 is irrelevant for direct and must be 0")
        if self.kind == "lowrank" and self.residual_l2 != 0:
            raise ValueError("residual_l2 is irrelevant for lowrank and must be 0")
        if self.kind == "joint" and self.rank == 0 and self.shared_l2 != 0:
            raise ValueError("shared_l2 is irrelevant for rank-0 joint and must be 0")
        if not self.use_target_features and self.target_l2 != 0:
            raise ValueError("target_l2 requires use_target_features=True")

    def with_updates(self, **changes: Any) -> "ModelConfig":
        values = asdict(self)
        values.update(changes)
        return ModelConfig(**values)


@dataclass(frozen=True)
class TuningConfig:
    strategy: str = "full_joint"
    ranks: tuple[int, ...] = (2, 4, 8)
    shared_l2: tuple[float, ...] = (1e-5, 1e-3, 1e-2)
    residual_l2: tuple[float, ...] = (1e-5, 1e-3, 1e-2)
    target_l2: tuple[float, ...] = (1e-5, 1e-3, 1e-2)
    anchor_shared_l2: float = 1e-3
    anchor_residual_l2: float = 1e-3
    anchor_target_l2: float = 1e-3
    metric: str = "observed_log_loss"

    def __post_init__(self) -> None:
        if self.strategy not in {"full_joint", "staged_rank_l2"}:
            raise ValueError("strategy must be full_joint or staged_rank_l2")
        if self.metric != "observed_log_loss":
            raise ValueError("v0.1 supports observed_log_loss tuning only")
        if not self.ranks or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in self.ranks
        ):
            raise ValueError("ranks must be a nonempty sequence of nonnegative integers")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("ranks must not contain duplicates")
        if not self.shared_l2 or not self.residual_l2 or not self.target_l2:
            raise ValueError("L2 grids must be nonempty")
        grids = self.shared_l2 + self.residual_l2 + self.target_l2
        if any(not _finite_number(x) or x < 0 for x in grids):
            raise ValueError("L2 grids must be finite and nonnegative")
        if any(
            len(set(grid)) != len(grid)
            for grid in (self.shared_l2, self.residual_l2, self.target_l2)
        ):
            raise ValueError("L2 grids must not contain duplicates")
        anchors = (
            self.anchor_shared_l2,
            self.anchor_residual_l2,
            self.anchor_target_l2,
        )
        if any(not _finite_number(x) or x < 0 for x in anchors):
            raise ValueError("anchor penalties must be finite and nonnegative")
        if self.strategy == "staged_rank_l2" and (
            self.anchor_shared_l2 not in self.shared_l2
            or self.anchor_residual_l2 not in self.residual_l2
            or self.anchor_target_l2 not in self.target_l2
        ):
            raise ValueError("staged anchors must be present in their corresponding grids")


@dataclass(frozen=True)
class ExperimentConfig:
    version: int
    dataset: Mapping[str, Any]
    split: Mapping[str, Any]
    observation: Mapping[str, Any]
    models: tuple[ModelConfig, ...]
    tuning: TuningConfig
    fit: FitConfig = field(default_factory=FitConfig)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    evaluation: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        required = {"version", "dataset", "split", "observation", "models", "tuning"}
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(f"config missing required keys: {missing}")
        models = tuple(ModelConfig(**item) for item in raw["models"])
        if len({m.name for m in models}) != len(models):
            raise ValueError("model names must be unique")
        tuning_raw = dict(raw["tuning"])
        for key in ("ranks", "shared_l2", "residual_l2", "target_l2"):
            if key in tuning_raw:
                tuning_raw[key] = tuple(tuning_raw[key])
        return cls(
            version=int(raw["version"]),
            dataset=dict(raw["dataset"]),
            split=dict(raw["split"]),
            observation=dict(raw["observation"]),
            models=models,
            tuning=TuningConfig(**tuning_raw),
            fit=FitConfig(**dict(raw.get("fit", {}))),
            runtime=dict(raw.get("runtime", {})),
            evaluation=dict(raw.get("evaluation", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment profile."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("top-level YAML value must be a mapping")
    return ExperimentConfig.from_mapping(raw)
