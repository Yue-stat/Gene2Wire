"""Paired-reference detection calibration and outcome-blind budget sampling.

The calibrator sees observed detections only among authorized paired reference
positives.  It never receives the generator's sensitivity or parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .observation import _readonly, _rows


class CalibrationNotEstimable(ValueError):
    """The authorized paired subset contains no assayed reference positives."""


def sample_paired_rows(eligible_rows, fraction: float, seed: int, groups=None) -> np.ndarray:
    """Select exactly floor(fraction * N) rows, with nested fractions.

    No labels enter this function.  Within-group random permutations are
    interleaved by relative position to approximately preserve group proportions
    without overspending the global cell budget.  Call with the same eligible
    rows, groups, and seed for the 10/20/40% calibration-size experiment.
    ``groups`` is aligned to ``eligible_rows``, not to the full dataset.
    """
    eligible = np.asarray(eligible_rows)
    if eligible.ndim != 1 or (eligible.size and not np.issubdtype(eligible.dtype, np.integer)):
        raise ValueError("eligible_rows must contain integer row indices")
    eligible = eligible.astype(int)
    if np.any(eligible < 0) or len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible_rows must be unique and non-negative")
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    labels = np.zeros(len(eligible), dtype=int) if groups is None else np.asarray(groups)
    if labels.shape != eligible.shape:
        raise ValueError("groups must be aligned to eligible_rows")
    rng = np.random.default_rng(seed)
    group_keys = np.array([f"{type(value).__name__}:{value}" for value in labels])
    keys, selected_rows = [], []
    for group in sorted(set(group_keys)):
        # Sorting before permutation makes row input order irrelevant.
        group_rows = rng.permutation(np.sort(eligible[group_keys == group]))
        size = len(group_rows)
        for rank, row in enumerate(group_rows):
            keys.append(((rank + 0.5) / size, float(rng.uniform())))
            selected_rows.append(row)
    order = sorted(range(len(keys)), key=keys.__getitem__)
    size = int(np.floor(float(fraction) * len(eligible) + 1e-12))
    return np.sort(np.asarray(selected_rows, dtype=int)[order[:size]])


@dataclass(frozen=True)
class DetectionCalibrator:
    n_targets: int
    coefficients: np.ndarray
    use_technical: bool
    use_target_effects: bool
    platform_levels: tuple[str, ...]
    use_platform: bool
    technical_center: float
    technical_scale: float
    diagnostics: dict

    def predict(self, n_cells: int | None = None, technical_score=None, platform=None) -> np.ndarray:
        """Predict relative detection using declared covariates only.

        Platforms absent from the paired fit use the pooled target prediction.
        Targets without paired positives use pooled effects under the same fixed
        regularization as all other targets.
        """
        if n_cells is None:
            source = technical_score if technical_score is not None else platform
            if source is None:
                raise ValueError("Provide n_cells when no row covariates are supplied")
            n_cells = len(source)
        if int(n_cells) != n_cells or n_cells < 0:
            raise ValueError("n_cells must be a non-negative integer")
        n_cells = int(n_cells)
        technical = None
        if self.use_technical:
            technical = np.asarray(technical_score, dtype=float)
            if technical.shape != (n_cells,) or not np.all(np.isfinite(technical)):
                raise ValueError("technical_score must be finite with shape (n_cells,)")
            technical = (technical - self.technical_center) / self.technical_scale
        platforms = None
        if self.use_platform:
            if platform is None or np.asarray(platform).shape != (n_cells,):
                raise ValueError("platform must be supplied with shape (n_cells,)")
            platforms = np.asarray(platform).astype(str)
        eta = np.full((n_cells, self.n_targets), self.coefficients[0], dtype=float)
        cursor = 1
        if self.use_target_effects:
            eta += self.coefficients[cursor:cursor + self.n_targets][None, :]
            cursor += self.n_targets
        if self.use_technical:
            eta += self.coefficients[cursor] * technical[:, None]
            cursor += 1
        for level in self.platform_levels:
            selected = platforms == level
            eta[selected] += self.coefficients[cursor]
            cursor += 1
            if self.use_target_effects:
                eta[selected] += self.coefficients[cursor:cursor + self.n_targets][None, :]
                cursor += self.n_targets
        return expit(eta)

    def to_dict(self) -> dict:
        return {
            "n_targets": self.n_targets, "coefficients": self.coefficients.tolist(),
            "use_technical": self.use_technical, "use_target_effects": self.use_target_effects,
            "platform_levels": list(self.platform_levels), "use_platform": self.use_platform,
            "technical_center": self.technical_center, "technical_scale": self.technical_scale,
            "diagnostics": dict(self.diagnostics),
        }


def fit_detection_calibrator(
    observed,
    reference,
    measured,
    paired_rows,
    technical_score=None,
    platform=None,
    use_technical: bool | None = None,
    use_target_effects: bool = True,
    ridge: float = 1.0,
    intercept_pseudocount: float = 0.5,
    max_iter: int = 1000,
) -> DetectionCalibrator:
    """Fit a common regularized logistic detector on paired reference positives.

    The objective is summed Bernoulli NLL plus ridge/2 times squared non-intercept
    coefficients.  A symmetric Beta pseudocount on the pooled intercept prevents
    all-zero/all-one detections from producing infinite fits.  This is a detector
    regularizer, not a projection-target intercept penalty.

    For synthetic Technical-SAR, supply the observed technical covariate.  SCAR
    omits technical and target effects; Target-SAR omits technical only.  Natural
    paired assays supply platform and omit technical.  The latter uses regularized
    target, platform, and platform-by-target effects.  Missing-covariate ablations
    set use_technical=False or use_target_effects=False explicitly.

    Reference values outside paired_rows are never inspected, even for validation.
    """
    d, z, w_raw = np.asarray(observed), np.asarray(reference), np.asarray(measured)
    if d.ndim != 2 or z.shape != d.shape or w_raw.shape != d.shape:
        raise ValueError("observed, reference, measured must share shape (N,T)")
    if not np.all((w_raw == 0) | (w_raw == 1)):
        raise ValueError("measured must be binary")
    paired = _rows(paired_rows, d.shape[0], "paired_rows")
    w = w_raw.astype(bool)
    zp, dp, wp = z[paired], d[paired], w[paired]
    if not np.all((zp[wp] == 0) | (zp[wp] == 1)) or not np.all((dp[wp] == 0) | (dp[wp] == 1)):
        raise ValueError("Paired measured reference and detection labels must be binary")
    if np.any(wp & (dp == 1) & (zp == 0)):
        raise ValueError("PU calibration requires observed positives to be reference positive")
    local_rows, targets = np.nonzero(wp & (zp == 1))
    if not len(local_rows):
        raise CalibrationNotEstimable("Authorized paired subset contains no assayed reference positives")
    rows = paired[local_rows]
    outcome = d[rows, targets].astype(float)
    if not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(intercept_pseudocount) or intercept_pseudocount <= 0:
        raise ValueError("ridge and intercept_pseudocount must be strictly positive")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if use_technical is True and technical_score is None:
        raise ValueError("use_technical=True requires the declared technical_score covariate")
    use_technical = bool(technical_score is not None) if use_technical is None else bool(use_technical)
    center, scale = 0.0, 1.0
    columns = [np.ones((len(rows), 1))]
    if use_target_effects:
        columns.append(np.eye(d.shape[1])[targets])
    if use_technical:
        all_technical = np.asarray(technical_score, dtype=float)
        if all_technical.shape != (d.shape[0],):
            raise ValueError("technical_score must have shape (N,)")
        values = all_technical[rows]
        if not np.all(np.isfinite(values)):
            raise ValueError("Paired-positive technical scores must be finite")
        center = float(values.mean())
        scale = float(values.std())
        if scale < 1e-12:
            scale = 1.0
        columns.append(((values - center) / scale)[:, None])
    platform_levels = ()
    if platform is not None:
        platforms = np.asarray(platform)
        if platforms.shape != (d.shape[0],):
            raise ValueError("platform must have shape (N,)")
        platforms = platforms[rows].astype(str)
        platform_levels = tuple(sorted(set(platforms)))
        for level in platform_levels:
            selected = (platforms == level).astype(float)
            columns.append(selected[:, None])
            if use_target_effects:
                columns.append(selected[:, None] * np.eye(d.shape[1])[targets])
    design = np.concatenate(columns, axis=1)
    initial = np.zeros(design.shape[1], dtype=float)
    successes = outcome.sum()
    initial[0] = np.log((successes + intercept_pseudocount) / (len(outcome) - successes + intercept_pseudocount))

    def objective(coef):
        eta = design @ coef
        loss = float(np.sum(np.logaddexp(0, eta) - outcome * eta))
        loss += 0.5 * ridge * float(coef[1:] @ coef[1:])
        loss += intercept_pseudocount * float(np.logaddexp(0, coef[0]) + np.logaddexp(0, -coef[0]))
        gradient = design.T @ (expit(eta) - outcome)
        gradient[1:] += ridge * coef[1:]
        gradient[0] += intercept_pseudocount * (2.0 * expit(coef[0]) - 1.0)
        return loss, gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-7})
    # A failed optimizer is a recorded failed experiment, never a silent fallback.
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Detection calibration failed to converge: {result.message}")
    return DetectionCalibrator(
        d.shape[1], _readonly(result.x), use_technical, bool(use_target_effects),
        platform_levels, platform is not None, center, scale,
        {
            "paired_cells": int(len(paired)), "reference_positive_pairs": int(len(rows)),
            "detected_reference_positive_pairs": int(successes),
            "targets_without_paired_positives": int(np.sum(np.bincount(targets, minlength=d.shape[1]) == 0)),
            "ridge": float(ridge), "intercept_pseudocount": float(intercept_pseudocount),
            "converged": True, "iterations": int(result.nit), "objective": float(result.fun),
        },
    )


def detection_diagnostics(
    observed, reference, measured, estimated_sensitivity, rows=None, true_sensitivity=None,
) -> dict:
    """Evaluate detection only among held-out assayed reference positives.

    True sensitivity is optional and belongs exclusively to evaluation.  On real
    paired data, report proper scores; there is no biological e truth for RMSE.
    """
    d, z, w = np.asarray(observed), np.asarray(reference), np.asarray(measured, dtype=bool)
    e = np.asarray(estimated_sensitivity, dtype=float)
    if d.ndim != 2 or z.shape != d.shape or w.shape != d.shape or e.shape != d.shape:
        raise ValueError("Detection diagnostic arrays must share shape (N,T)")
    selected = np.ones(d.shape[0], dtype=bool)
    if rows is not None:
        selected[:] = False
        selected[_rows(rows, d.shape[0])] = True
    valid = selected[:, None] & w & (z == 1)
    n = int(valid.sum())
    output = {"detection_reference_positive_count": n}
    if not n:
        output.update(detection_log_loss=float("nan"), detection_brier=float("nan"), detection_mean=float("nan"), detection_observed_rate=float("nan"))
        if true_sensitivity is not None:
            output.update(sensitivity_mae=float("nan"), sensitivity_rmse=float("nan"))
        return output
    predicted, outcome = e[valid], d[valid].astype(float)
    if not np.all(np.isfinite(predicted)) or np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("Estimated sensitivities must be probabilities")
    if not np.all((outcome == 0) | (outcome == 1)):
        raise ValueError("Detection labels must be binary")
    safe = np.clip(predicted, 1e-12, 1 - 1e-12)
    output.update(
        detection_log_loss=float(np.mean(-outcome * np.log(safe) - (1 - outcome) * np.log1p(-safe))),
        detection_brier=float(np.mean((predicted - outcome) ** 2)),
        detection_mean=float(predicted.mean()), detection_observed_rate=float(outcome.mean()),
        detection_score_clip_fraction=float(np.mean(safe != predicted)),
    )
    if true_sensitivity is not None:
        truth = np.asarray(true_sensitivity, dtype=float)
        if truth.shape != d.shape or not np.all(np.isfinite(truth[valid])) or np.any((truth[valid] < 0) | (truth[valid] > 1)):
            raise ValueError("True sensitivity must match (N,T) and contain probabilities")
        difference = predicted - truth[valid]
        output.update(sensitivity_mae=float(np.mean(np.abs(difference))), sensitivity_rmse=float(np.sqrt(np.mean(difference ** 2))))
    return output
