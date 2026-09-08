"""Controlled, fold-local positive thinning for the 0908 protocol.

These are benchmark generators, not fitted physical models of an assay.  The
reference matrix is deliberately accepted here; it must not be passed onward to
the learner except through the separately authorized paired-reference subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


OBSERVATION_VERSION = "logistic-fold-local-v1"


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _rows(rows: Sequence[int], n_cells: int, name: str = "rows") -> np.ndarray:
    value = np.asarray(rows)
    if value.dtype == bool:
        if value.shape != (n_cells,):
            raise ValueError(f"{name} boolean mask must have length n_cells")
        value = np.flatnonzero(value)
    if value.ndim != 1 or (value.size and not np.issubdtype(value.dtype, np.integer)):
        raise ValueError(f"{name} must contain integer row indices")
    value = value.astype(int)
    if len(np.unique(value)) != len(value) or np.any((value < 0) | (value >= n_cells)):
        raise ValueError(f"{name} contains duplicate or out-of-range rows")
    return value


def canonical_mechanism(mechanism: str) -> str:
    """Accept spelling aliases, but never reinterpret an old scientific design."""
    key = str(mechanism).lower().replace("-", "_")
    if key == "tech_sar":
        key = "technical_sar"
    if key not in {"scar", "target_sar", "technical_sar"}:
        raise ValueError(
            f"Unknown mechanism {mechanism!r}. Use scar, target_sar, or "
            "technical_sar. Old mechanisms are archived designs and are not aliases."
        )
    return key


@dataclass(frozen=True)
class ObservationDesign:
    technical_score: np.ndarray
    target_offsets: np.ndarray
    uniforms: np.ndarray
    technical_coefficient: float = float(np.log(2.0))

    def __post_init__(self):
        u = np.asarray(self.technical_score, dtype=float)
        delta = np.asarray(self.target_offsets, dtype=float)
        draws = np.asarray(self.uniforms, dtype=float)
        if u.ndim != 1 or delta.ndim != 1 or draws.shape != (u.size, delta.size):
            raise ValueError("Design requires score (N,), offsets (T,), uniforms (N,T)")
        if not all(np.all(np.isfinite(x)) for x in (u, delta, draws)):
            raise ValueError("Observation design contains non-finite values")
        if np.any((draws < 0) | (draws >= 1)):
            raise ValueError("Uniform draws must be in [0,1)")
        if not np.isfinite(self.technical_coefficient):
            raise ValueError("technical_coefficient must be finite")
        for name, value in (("technical_score", u), ("target_offsets", delta), ("uniforms", draws)):
            object.__setattr__(self, name, _readonly(value))


@dataclass(frozen=True)
class ObservationResult:
    observed: np.ndarray
    sensitivity: np.ndarray
    gamma: float | None
    normalizer_positive_count: int
    expected_train_retention: float
    mechanism: str
    loss_rate: float
    version: str = OBSERVATION_VERSION


def make_observation_design(
    n_cells: int,
    n_targets: int,
    seed: int,
    technical_score=None,
    target_offsets=None,
    technical_coefficient: float = float(np.log(2.0)),
) -> ObservationDesign:
    """Create common random numbers once per fold/repetition, before rate loops.

    A supplied score is used as-is.  For feature-correlated simulation it should
    be generated explicitly by the simulator; the generator never accesses
    projection features or rescales scores using test outcomes.  The default
    target offsets span +/- log(2), in a seeded random target order.
    """
    if n_cells < 1 or n_targets < 1:
        raise ValueError("n_cells and n_targets must be positive")
    score_seed, offset_seed, draw_seed = np.random.SeedSequence(seed).spawn(3)
    u = np.random.default_rng(score_seed).normal(size=n_cells) if technical_score is None else technical_score
    if target_offsets is None:
        offsets = np.linspace(-np.log(2.0), np.log(2.0), n_targets) if n_targets > 1 else np.zeros(1)
        offsets = np.random.default_rng(offset_seed).permutation(offsets)
    else:
        offsets = target_offsets
    draws = np.random.default_rng(draw_seed).uniform(size=(n_cells, n_targets))
    design = ObservationDesign(u, offsets, draws, technical_coefficient)
    if design.uniforms.shape != (n_cells, n_targets):
        raise ValueError("Supplied score/offset dimensions do not match n_cells/n_targets")
    return design


def thin_reference(
    reference,
    measured,
    train_rows,
    loss_rate: float,
    mechanism: str,
    design: ObservationDesign,
) -> ObservationResult:
    """Thin positives, normalizing retention using inner-training positives only.

    With one fixed design and train_rows, increasing loss_rate produces nested
    masks.  Refit must reuse these observed labels rather than regenerate them
    using newly opened validation references.
    """
    z = np.asarray(reference)
    w_raw = np.asarray(measured)
    if z.ndim != 2 or w_raw.shape != z.shape or design.uniforms.shape != z.shape:
        raise ValueError("reference, measured, and design must share shape (N,T)")
    if not np.all((w_raw == 0) | (w_raw == 1)):
        raise ValueError("measured must be binary")
    w = w_raw.astype(bool)
    if not np.all((z[w] == 0) | (z[w] == 1)):
        raise ValueError("Reference values on measured entries must be binary")
    train = _rows(train_rows, z.shape[0], "train_rows")
    if not len(train):
        raise ValueError("train_rows cannot be empty")
    rate = float(loss_rate)
    if not np.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("loss_rate must be in [0,1)")
    mechanism = canonical_mechanism(mechanism)
    positives = w & (z == 1)
    train_positive = positives[train]
    n_positive = int(train_positive.sum())
    if rate == 0:
        sensitivity = np.ones(z.shape, dtype=float)
        # The no-thinning boundary is known exactly; None is JSON-safe and
        # distinguishes it from a finite fitted normalization intercept.
        gamma, retention = None, 1.0
    else:
        if n_positive == 0:
            raise ValueError("Cannot normalize thinning: inner training has no reference positives")
        eta = np.zeros(z.shape, dtype=float)
        if mechanism != "scar":
            eta += design.target_offsets[None, :]
        if mechanism == "technical_sar":
            eta += design.technical_coefficient * design.technical_score[:, None]
        train_eta = eta[train][train_positive]
        retention = 1.0 - rate
        lower = -float(np.max(train_eta)) - 50.0
        upper = -float(np.min(train_eta)) + 50.0
        gamma = float(brentq(lambda g: expit(g + train_eta).mean() - retention, lower, upper, xtol=1e-12))
        sensitivity = expit(gamma + eta)
        retention = float(sensitivity[train][train_positive].mean())
    observed = positives & (design.uniforms < sensitivity)
    return ObservationResult(
        _readonly(observed, bool), _readonly(sensitivity), gamma, n_positive,
        retention, mechanism, rate,
    )
