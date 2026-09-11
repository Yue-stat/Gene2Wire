"""Generated projection truth for the 0908, three-fold simulation protocol.

The defaults retain the previous notebook's signal dimensions and variance
construction, with an explicit change: outcome truth uses genes alone unless
``truth_uses_location=True``. Location availability and use by the predictor are
separate choices. The notebook sets this truth option from USE_LOCATION for its
matched-design experiment; an omitted-location stress test must set it explicitly.

No observation thinning or calibration occurs here. ``reference`` is generated
projection truth, and arrays under ``metadata`` are evaluation/audit material.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from ...seeds import stable_seed
from ..contracts import ExperimentDataset, FeatureSet, Fold


SIMULATION_GENERATOR_VERSION = "0908-gene-truth-v1"


def _zscore(a: np.ndarray, axis: int | None = 0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    scale = a.std(axis=axis, keepdims=True)
    return (a - a.mean(axis=axis, keepdims=True)) / np.where(scale > 1e-12, scale, 1.0)


def _intercept(score: np.ndarray, prevalence: float) -> float:
    # The generated signal can be rescaled by callers, so bracket adaptively.
    lo, hi = -max(40.0, float(np.max(score)) + 40.0), max(40.0, -float(np.min(score)) + 40.0)
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if float(expit(score + mid).mean()) < prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _location_basis(position: np.ndarray, n_features: int) -> np.ndarray:
    """First four columns reproduce the old position basis; extras are harmonics."""
    if n_features == 0:
        return np.empty((len(position), 0), dtype=float)
    columns = [position, position**2, np.sin(np.pi * position), np.cos(np.pi * position)]
    harmonic = 2
    while len(columns) < n_features:
        columns.extend((np.sin(harmonic * np.pi * position),
                        np.cos(harmonic * np.pi * position)))
        harmonic += 1
    return _zscore(np.column_stack(columns[:n_features]))


@dataclass(frozen=True)
class SimulationFeatures:
    """Pickleable, train-fitted feature builder; truth variables are not inputs."""

    genes: np.ndarray
    location: np.ndarray
    target_descriptors: np.ndarray

    def __call__(self, train_rows: np.ndarray, use_location: bool,
                 use_target_features: bool) -> FeatureSet:
        rows = np.asarray(train_rows, dtype=int)
        if rows.ndim != 1 or len(rows) == 0 or len(np.unique(rows)) != len(rows):
            raise ValueError("Feature training rows must be nonempty and unique")
        if np.any(rows < 0) or np.any(rows >= len(self.genes)):
            raise ValueError("Feature training rows are out of range")
        if use_location and self.location.shape[1] == 0:
            raise ValueError("USE_LOCATION=True requires generated location features")
        if use_target_features and self.target_descriptors.shape[1] == 0:
            raise ValueError("USE_TARGET_FEATURES=True requires target descriptors")
        raw = (np.column_stack((self.genes, self.location))
               if use_location else self.genes)
        scaler = StandardScaler().fit(raw[rows])
        p_gene = self.genes.shape[1]
        blocks = {"gene": tuple(range(p_gene))}
        if use_location:
            blocks["location"] = tuple(range(p_gene, raw.shape[1]))
        names = tuple(f"gene_{j:02d}" for j in range(p_gene))
        if use_location:
            names += tuple(f"location_{j:02d}" for j in range(self.location.shape[1]))
        return FeatureSet(
            X=scaler.transform(raw), feature_blocks=blocks,
            Y_target=self.target_descriptors.copy() if use_target_features else None,
            feature_names=names,
            metadata={"scaler_mean": scaler.mean_.tolist(),
                      "scaler_scale": scaler.scale_.tolist(),
                      "training_rows": rows.tolist(),
                      "use_location": bool(use_location),
                      "use_target_features": bool(use_target_features)},
        )


@dataclass(frozen=True)
class SimulationSplits:
    slices: np.ndarray
    position: np.ndarray
    repetition: int
    inner_validation_slices: int = 4

    def __call__(self, n_outer_folds: int = 3, seed: int = 0) -> Sequence[Fold]:
        """Contiguous outer blocks; validation slices spaced within development.

        A single-fold option retains one rotating spatial holdout for smoke runs.
        With >=2 folds every cell is tested exactly once per generated dataset.
        The seed is accepted for the common adapter contract; geometry fixes splits.
        """
        del seed
        ordered = np.asarray(sorted(np.unique(self.slices),
                            key=lambda s: self.position[self.slices == s].mean()))
        if isinstance(n_outer_folds, bool) or int(n_outer_folds) != n_outer_folds:
            raise ValueError("n_outer_folds must be an integer")
        n_outer_folds = int(n_outer_folds)
        if n_outer_folds < 1 or n_outer_folds > len(ordered):
            raise ValueError("n_outer_folds must be between 1 and the slice count")
        if n_outer_folds == 1:
            # One holdout is intentionally not complete cross-validation.
            blocks = np.array_split(ordered, min(5, len(ordered)))
            test_blocks = [blocks[self.repetition % len(blocks)]]
        else:
            test_blocks = np.array_split(ordered, n_outer_folds)
        folds = []
        visits = np.zeros(len(self.slices), dtype=int)
        for k, test_slices in enumerate(test_blocks):
            development = np.setdiff1d(ordered, test_slices)
            if len(development) < 2:
                raise ValueError("Each outer fold needs at least two development slices")
            n_val = min(self.inner_validation_slices, max(1, len(development) // 3))
            idx = np.unique(np.linspace(0, len(development) - 1, n_val).round().astype(int))
            validation = development[idx]
            train = np.setdiff1d(development, validation)
            fold = Fold(
                outer_fold=k,
                train_rows=np.flatnonzero(np.isin(self.slices, train)),
                validation_rows=np.flatnonzero(np.isin(self.slices, validation)),
                test_rows=np.flatnonzero(np.isin(self.slices, test_slices)),
                metadata={"train_slices": train.tolist(),
                          "validation_slices": validation.tolist(),
                          "test_slices": test_slices.tolist(),
                          "independent_unit": "generated_dataset",
                          "repetition": self.repetition},
            )
            fold.validate(len(self.slices))
            visits[fold.test_rows] += 1
            folds.append(fold)
        if n_outer_folds > 1 and not np.all(visits == 1):
            raise RuntimeError("Outer folds must test every generated cell exactly once")
        return folds


def generate_simulation(
    repetition: int,
    sharing_strength: float,
    seed: int = 20260908,
    *,
    n_cells: int = 400,
    n_targets: int = 36,
    n_gene_features: int = 16,
    n_location_features: int = 4,
    n_slices: int = 20,
    true_rank: int = 2,
    n_target_features: int = 4,
    signal_sd: float = 1.35,
    target_feature_noise: float = 0.85,
    target_prevalence_range: tuple[float, float] = (0.1, 0.3),
    truth_uses_location: bool = False,
    technical_feature_correlation: float = -0.65,
    technical_noise_scale: float = 0.75,
    inner_validation_slices: int = 4,
) -> ExperimentDataset:
    """Generate one independent dataset, paired across sharing strengths.

    ``sharing_strength`` is the per-target shared fraction of linear-signal
    variance. Reference outcomes are Bernoulli draws, not thresholded scores.
    ``technical_score`` correlates with a generated cell factor, never with a
    realized outcome label. Thinning is delegated to the common observation code.
    """
    dims = {"n_cells": n_cells, "n_targets": n_targets,
            "n_gene_features": n_gene_features, "n_slices": n_slices,
            "true_rank": true_rank, "inner_validation_slices": inner_validation_slices}
    for name, value in dims.items():
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in {"n_location_features": n_location_features,
                        "n_target_features": n_target_features}.items():
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if n_slices < 3 or n_cells < n_slices or n_targets < 2:
        raise ValueError("Simulation needs at least three nonempty slices and two targets")
    p_truth = n_gene_features + (n_location_features if truth_uses_location else 0)
    if truth_uses_location and n_location_features == 0:
        raise ValueError("Location truth requires at least one location feature")
    if p_truth < 2 or true_rank > min(p_truth, n_targets - 1, n_cells - 1):
        raise ValueError("true_rank is incompatible with the effective truth dimensions")
    rho = float(sharing_strength)
    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("sharing_strength must be in [0, 1]")
    if not np.isfinite(signal_sd) or signal_sd <= 0 or target_feature_noise < 0:
        raise ValueError("signal_sd must be positive and target_feature_noise nonnegative")
    lo, hi = map(float, target_prevalence_range)
    if not 0 < lo <= hi < 1:
        raise ValueError("Prevalence endpoints must satisfy 0 < low <= high < 1")
    if technical_noise_scale <= 0 or not np.isfinite(technical_feature_correlation):
        raise ValueError("Technical noise must be positive and correlation coefficient finite")

    rng = np.random.default_rng(stable_seed(seed, "simulation", "base", repetition))
    slices = np.concatenate([np.full(len(rows), s, dtype=int)
                             for s, rows in enumerate(np.array_split(np.arange(n_cells), n_slices))])
    position = np.linspace(-1.0, 1.0, n_slices)[slices] + rng.normal(0, 0.018, n_cells)
    position_z = _zscore(position, axis=None).ravel()
    factors = rng.normal(size=(n_cells, 7))
    factors[:, 0] = 0.70 * position_z + np.sqrt(1 - 0.70**2) * factors[:, 0]
    factors[:, 1] = 0.55 * np.sin(np.pi * position) + np.sqrt(1 - 0.55**2) * factors[:, 1]
    gene_loadings = rng.normal(size=(7, n_gene_features))
    x_gene = _zscore(factors @ gene_loadings + rng.normal(scale=0.55, size=(n_cells, n_gene_features)))
    x_location = _location_basis(position, n_location_features)
    x_truth = np.column_stack((x_gene, x_location)) if truth_uses_location else x_gene

    cell_loading = rng.normal(size=(p_truth, true_rank))
    target_loading = _zscore(rng.normal(size=(n_targets, true_rank)))
    shared_cell_factor = x_truth @ cell_loading
    shared_coef = cell_loading @ target_loading.T
    shared_coef /= np.std(x_truth @ shared_coef, axis=0)
    low_rank = x_truth @ shared_coef
    direct_coef = rng.normal(size=(p_truth, n_targets))
    direct_coef /= np.std(x_truth @ direct_coef, axis=0)
    overlap = np.mean((x_truth @ direct_coef) * low_rank, axis=0)
    direct_coef -= shared_coef * overlap[None, :]
    residual_sd = np.std(x_truth @ direct_coef, axis=0)
    if np.any(residual_sd < 1e-10):
        raise ValueError("Degenerate orthogonal specific component; increase gene dimension")
    direct_coef /= residual_sd
    direct = x_truth @ direct_coef
    mixed_coef = np.sqrt(rho) * shared_coef + np.sqrt(1.0 - rho) * direct_coef
    coefficient = signal_sd * mixed_coef / np.std(x_truth @ mixed_coef, axis=0)
    signal = x_truth @ coefficient
    requested_prevalence = np.linspace(lo, hi, n_targets)
    intercepts = np.asarray([_intercept(signal[:, t], requested_prevalence[t])
                             for t in range(n_targets)])
    true_probability = expit(signal + intercepts)
    latent_uniform = np.random.default_rng(stable_seed(seed, "simulation", "latent", repetition)).uniform(size=true_probability.shape)
    z = (latent_uniform < true_probability).astype(np.int8)

    nuisance = rng.normal(size=(n_targets, max(0, n_target_features - true_rank)))
    y_signal = np.column_stack((target_loading, nuisance))[:, :n_target_features]
    y_target = _zscore(y_signal + rng.normal(scale=target_feature_noise, size=y_signal.shape))
    row_noise = rng.normal(size=n_cells)
    technical_score = _zscore(
        technical_feature_correlation * _zscore(shared_cell_factor[:, 0], axis=None).ravel()
        + technical_noise_scale * row_noise, axis=None).ravel()

    dataset = ExperimentDataset(
        name="simulation", reference=z, measured=np.ones_like(z, dtype=bool),
        cell_ids=tuple(f"rep{repetition:03d}_cell{i:05d}" for i in range(n_cells)),
        target_ids=tuple(f"target{t:03d}" for t in range(n_targets)),
        feature_builder=SimulationFeatures(x_gene, x_location, y_target),
        split_builder=SimulationSplits(slices, position, repetition, inner_validation_slices),
        gene_matrix=x_gene.copy(),
        gene_names=tuple(f"gene_{j:02d}" for j in range(n_gene_features)),
        groups={"slice": slices}, technical_score=technical_score,
        metadata={
            "generator_version": SIMULATION_GENERATOR_VERSION,
            "repetition": int(repetition), "sharing_strength": rho, "seed": int(seed),
            "n_cells": n_cells, "n_targets": n_targets, "n_gene_features": n_gene_features,
            "n_location_features": n_location_features, "n_target_features": n_target_features,
            "n_slices": n_slices, "true_rank": true_rank, "signal_sd": signal_sd,
            "target_feature_noise": target_feature_noise,
            "target_prevalence_range": [lo, hi],
            "truth_uses_location": bool(truth_uses_location), "truth_feature_count": p_truth,
            "technical_feature_correlation": technical_feature_correlation,
            "technical_noise_scale": technical_noise_scale,
            "independent_unit": "generated_dataset",
            "fold_aggregation": "aggregate_within_repetition_before_uncertainty",
            "true_probability": true_probability,
            "true_coefficient": coefficient, "true_intercept": intercepts,
            "shared_coefficient": shared_coef, "specific_coefficient": direct_coef,
            "shared_specific_target_covariance": np.mean(low_rank * direct, axis=0),
            "requested_prevalence": requested_prevalence,
            "latent_uniform": latent_uniform,
            "target_descriptors": y_target,
            "truth_arrays_role": "evaluation_and_audit_only",
        },
    )
    dataset.validate()
    return dataset
