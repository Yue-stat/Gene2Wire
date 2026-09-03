"""Masked PU and reference metrics with an explicit Brier baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _mask(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("mask must be numeric and binary") from exc
    if raw.shape != shape:
        raise ValueError("mask shape must match truth and prediction")
    if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("mask must be finite and binary")
    return raw.astype(bool, copy=False)


def _checked(y: Any, prediction: Any, mask: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError("truth and prediction shapes must match")
    if truth.ndim != 2 or pred.ndim != 2:
        raise ValueError("metric inputs must be cell-by-target matrices")
    measured = _mask(mask, truth.shape)
    if not np.any(measured):
        raise ValueError("metric mask contains no entries")
    if not np.all(np.isfinite(truth[measured])) or np.any(
        (truth[measured] < 0) | (truth[measured] > 1)
    ):
        raise ValueError("measured truth values must be finite and in [0, 1]")
    if not np.all(np.isfinite(pred[measured])) or np.any(
        (pred[measured] < 0) | (pred[measured] > 1)
    ):
        raise ValueError("measured predictions must be finite and in [0, 1]")
    return truth, pred, measured


def masked_log_loss(y: Any, prediction: Any, mask: Any) -> float:
    truth, pred, measured = _checked(y, prediction, mask)
    eps = np.finfo(np.float64).eps
    p = np.clip(pred[measured], eps, 1 - eps)
    t = truth[measured]
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log1p(-p)))


def masked_brier(y: Any, prediction: Any, mask: Any) -> float:
    truth, pred, measured = _checked(y, prediction, mask)
    return float(np.mean((truth[measured] - pred[measured]) ** 2))


def target_prevalence(y: Any, mask: Any) -> np.ndarray:
    truth = np.asarray(y, dtype=np.float64)
    if truth.ndim != 2:
        raise ValueError("y and mask must be matching 2-D matrices")
    measured = _mask(mask, truth.shape)
    if not np.all(np.isfinite(truth[measured])) or np.any(
        (truth[measured] < 0) | (truth[measured] > 1)
    ):
        raise ValueError("measured truth values must be finite and in [0, 1]")
    counts = np.sum(measured, axis=0)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"training prevalence undefined for targets {missing}")
    return np.sum(np.where(measured, truth, 0.0), axis=0) / counts


@dataclass(frozen=True)
class BrierReport:
    raw_brier: float
    baseline_brier: float
    skill: float
    train_prevalence: np.ndarray


def brier_report(
    test_y: Any,
    prediction: Any,
    test_mask: Any,
    train_y: Any,
    train_mask: Any,
) -> BrierReport:
    """Report raw Brier and BSS against train-only target prevalence.

    The baseline is estimated exclusively from ``train_y`` and evaluated on
    the test mask.  This avoids using test prevalence in the denominator.
    """

    prevalence = target_prevalence(train_y, train_mask)
    test_truth = np.asarray(test_y, dtype=np.float64)
    baseline = np.broadcast_to(prevalence, test_truth.shape)
    raw = masked_brier(test_truth, prediction, test_mask)
    base = masked_brier(test_truth, baseline, test_mask)
    skill = float("nan") if base == 0 else 1.0 - raw / base
    return BrierReport(raw_brier=raw, baseline_brier=base, skill=skill, train_prevalence=prevalence)
