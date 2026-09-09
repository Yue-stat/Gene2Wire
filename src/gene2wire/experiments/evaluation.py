"""Evaluation against held-out references, with explicit probability semantics.

This module never selects models or estimates a training prevalence. It receives
held-out labels only after fitting. All rows remain attached to their original
fold/repetition in the caller; repetitions are not biological replicates.

``prediction`` is reference probability p for reference-supervised/PU fits and
observed probability q for ordinary fits. The latter is deliberately evaluated
against the reference without silently applying sensitivity correction.
"""
from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

ProbabilitySemantics = Literal["reference", "observed", "mixed"]
_EPS = np.finfo(np.float64).eps


def _binary_mask(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or not np.all(np.isin(raw, [0, 1])):
        raise ValueError(f"{name} must be a binary matrix with shape {shape}")
    return raw.astype(bool)


def _probabilities(value: Any, shape: tuple[int, ...], mask: np.ndarray,
                   name: str) -> np.ndarray:
    try:
        array = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    except ValueError as exc:
        raise ValueError(f"{name} does not broadcast to shape {shape}") from exc
    if np.any(~np.isfinite(array[mask])) or np.any((array[mask] < 0) | (array[mask] > 1)):
        raise ValueError(f"{name} must be finite and in [0, 1] on measured entries")
    array[~mask] = np.nan
    return array


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    return float(np.mean(array[valid])) if np.any(valid) else float("nan")


def expected_top_k_recall(truth: Any, score: Any, k: int) -> float:
    """Recall with fractional inclusion at a tied cutoff.

This equals expected recall under uniform random tie breaking and is invariant
to input row order. If there are no positives, recall is undefined (NaN).
"""
    y = np.asarray(truth, dtype=float)
    s = np.asarray(score, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not np.all(np.isin(y, [0, 1])):
        raise ValueError("truth and score must be aligned binary/score vectors")
    if not np.all(np.isfinite(s)):
        raise ValueError("Ranking scores must be finite")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    positives = int(np.sum(y))
    if positives == 0:
        return float("nan")
    k = min(int(k), len(y))
    if k == 0:
        return 0.0
    cutoff = np.partition(s, len(s) - k)[len(s) - k]
    above, tied = s > cutoff, s == cutoff
    tied_fraction = (k - int(np.sum(above))) / int(np.sum(tied))
    recovered = np.sum(y[above]) + tied_fraction * np.sum(y[tied])
    return float(recovered / positives)


def reliability_bins(truth: Any, probability: Any, n_bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width reliability bins, retaining empty bins with NaN means."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    y, p = np.asarray(truth, dtype=float), np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.all(np.isin(y, [0, 1])):
        raise ValueError("Reliability inputs must be aligned binary/probability vectors")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Reliability probabilities must be finite and in [0, 1]")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        selected = bins == b
        count = int(np.sum(selected))
        rows.append({
            "bin": b, "lower": b / n_bins, "upper": (b + 1) / n_bins,
            "n": count,
            "mean_prediction": float(np.mean(p[selected])) if count else float("nan"),
            "positive_fraction": float(np.mean(y[selected])) if count else float("nan"),
        })
    return rows


def _calibration_fit(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Unpenalized joint calibration intercept/slope; undefined fits stay NaN.

Constant predictions, single-class outcomes, and complete/quasi separation do
not identify a finite two-parameter calibration regression.
"""
    if len(y) < 3 or np.unique(y).size != 2:
        return float("nan"), float("nan")
    x = logit(np.clip(p, _EPS, 1 - _EPS))
    if np.ptp(x) < 1e-10:
        return float("nan"), float("nan")
    if np.max(x[y == 0]) <= np.min(x[y == 1]) or np.max(x[y == 1]) <= np.min(x[y == 0]):
        return float("nan"), float("nan")
    # Standardize the logit covariate for numerical conditioning, then undo it.
    center, scale = float(np.mean(x)), float(np.std(x))
    design = np.column_stack([np.ones(len(x)), (x - center) / scale])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ theta
        return (float(np.mean(np.logaddexp(0, eta) - y * eta)),
                design.T @ (expit(eta) - y) / len(y))

    result = minimize(objective, np.array([logit(np.mean(y)), 0.0]), jac=True,
                      method="BFGS", options={"gtol": 1e-7, "maxiter": 300})
    if not result.success or not np.all(np.isfinite(result.x)):
        return float("nan"), float("nan")
    slope = float(result.x[1] / scale)
    return float(result.x[0] - slope * center), slope


def _metric_vector(y: np.ndarray, p: np.ndarray, *, n_bins: int,
                   ranking_score: np.ndarray | None = None) -> dict[str, float]:
    keys = ("auprc", "auroc", "recall_at_100", "brier", "log_loss", "ece",
            "calibration_intercept", "calibration_slope", "predicted_prevalence",
            "reference_prevalence", "prevalence_bias", "prevalence_absolute_error")
    if not len(y):
        return {key: float("nan") for key in keys}
    clipped = np.clip(p, _EPS, 1 - _EPS)
    score = p if ranking_score is None else ranking_score
    prevalence = float(np.mean(y))
    predicted = float(np.mean(p))
    both_classes = np.unique(y).size == 2
    bins = reliability_bins(y, p, n_bins)
    ece = sum(row["n"] * abs(row["mean_prediction"] - row["positive_fraction"])
              for row in bins if row["n"]) / len(y)
    intercept, slope = _calibration_fit(y, p)
    return {
        "auprc": float(average_precision_score(y, score)) if both_classes else float("nan"),
        "auroc": float(roc_auc_score(y, score)) if both_classes else float("nan"),
        "recall_at_100": expected_top_k_recall(y, score, 100),
        "brier": float(np.mean((y - p) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        "ece": float(ece), "calibration_intercept": intercept, "calibration_slope": slope,
        "predicted_prevalence": predicted, "reference_prevalence": prevalence,
        "prevalence_bias": predicted - prevalence,
        "prevalence_absolute_error": abs(predicted - prevalence),
    }


def _hidden_metrics(y: np.ndarray, score: np.ndarray | None) -> tuple[float, float]:
    if score is None or not len(y):
        return float("nan"), float("nan")
    hidden_count = int(np.sum(y))
    recall = expected_top_k_recall(y, score, hidden_count)
    ap = (float(average_precision_score(y, score)) if np.unique(y).size == 2
          else float("nan"))
    return recall, ap


def evaluate_predictions(
    reference: Any, observed: Any, measured: Any, prediction: Any,
    estimated_sensitivity: Any, *, probability_semantics: ProbabilitySemantics = "reference",
    train_reference_prevalence: Any | None = None, target_ids: Sequence[str] | None = None,
    n_bins: int = 10, ranking_score: Any | None = None,
) -> dict[str, Any]:
    """Compute target-macro, entry-micro, hidden-recovery, and reliability results.

    ``train_reference_prevalence`` must be estimated only from authorized paired
    training/reference labels by the caller. Missing target estimates may be NaN;
    Brier skill then excludes those targets and reports its coverage. No test
    prevalence is substituted. Proper scores remain defined for single-class
    outcomes; discrimination metrics that require both classes are NaN.

    AUPRC uses average precision (step integration), consistently across runs.
    ``hidden_recall_at_h`` is the target-macro primary endpoint: h-ranking for
    reference-probability models, raw q-ranking for observed-probability models.
    Explicit p/h-ranking columns provide a fixed-predictor scoring ablation.
    Observed-probability adapters may supply an unbounded finite ``ranking_score``
    (e.g. a squared-error bilinear score before probability clipping). This score
    affects discrimination/recovery only, and is exported for reproducibility.
    Mixed-label RF scores are evaluated directly against each outcome, without
    sensitivity rescaling or h-ranking. They are neither identified p nor q;
    the exported p/q/h arrays are therefore absent for this baseline.
    """
    if probability_semantics not in {"reference", "observed", "mixed"}:
        raise ValueError("probability_semantics must be 'reference', 'observed', or 'mixed'")
    z, d = np.asarray(reference, dtype=float), np.asarray(observed, dtype=float)
    if z.ndim != 2 or z.shape != d.shape:
        raise ValueError("reference and observed must be aligned cell-by-target matrices")
    w = _binary_mask(measured, z.shape, "measured")
    if not np.all(np.isin(z[w], [0, 1])) or not np.all(np.isin(d[w], [0, 1])):
        raise ValueError("Measured reference and observed labels must be binary")
    if np.any(d[w] > z[w]):
        raise ValueError("PU evaluation requires every detection to be reference-positive")
    prediction = _probabilities(prediction, z.shape, w, "prediction")
    ranking = prediction
    if ranking_score is not None:
        if probability_semantics != "observed":
            raise ValueError("Separate ranking_score is supported only for observed-probability models")
        ranking = np.asarray(ranking_score, dtype=float).copy()
        if ranking.shape != z.shape or not np.all(np.isfinite(ranking[w])):
            raise ValueError("ranking_score must be aligned and finite on measured entries")
        ranking[~w] = np.nan
    e = _probabilities(estimated_sensitivity, z.shape, w, "estimated_sensitivity")
    p = prediction.copy() if probability_semantics == "reference" else None
    q = e * prediction if p is not None else prediction.copy()
    h = None
    if p is not None:
        # h only has a role for D=0. If q=1 then D=0 has zero model probability,
        # so h is undefined, rather than silently assigning a favorable rank.
        h = np.divide((1 - e) * p, 1 - e * p, out=np.full(z.shape, np.nan),
                      where=w & (e * p < 1))
    names = tuple(str(t) for t in target_ids) if target_ids is not None else tuple(map(str, range(z.shape[1])))
    if len(names) != z.shape[1] or len(set(names)) != len(names):
        raise ValueError("target_ids must be unique and aligned with target columns")
    if train_reference_prevalence is None:
        train_pi = np.full(z.shape[1], np.nan)
    else:
        train_pi = np.asarray(train_reference_prevalence, dtype=float)
        if train_pi.shape != (z.shape[1],) or np.any(np.isinf(train_pi)) or np.any(
            (train_pi[np.isfinite(train_pi)] < 0) | (train_pi[np.isfinite(train_pi)] > 1)
        ):
            raise ValueError("train_reference_prevalence must contain target probabilities or NaN")
    rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    for t, target in enumerate(names):
        mask = w[:, t]
        y, pred, obs, q_t = z[mask, t], prediction[mask, t], d[mask, t], q[mask, t]
        row: dict[str, Any] = {"target": target, "target_index": t,
            "n_measured": int(np.sum(mask)), "n_reference_positive": int(np.sum(y)),
            "n_observed_positive": int(np.sum(obs)),
            "probability_semantics": probability_semantics}
        row.update(_metric_vector(y, pred, n_bins=n_bins, ranking_score=ranking[mask, t]))
        row.update({f"observed_{key}": value for key, value in _metric_vector(
            obs, q_t, n_bins=n_bins,
            ranking_score=ranking[mask, t] if probability_semantics == "observed" else None).items()})
        row["train_reference_prevalence"] = float(train_pi[t])
        baseline = float(np.mean((y - train_pi[t]) ** 2)) if len(y) and np.isfinite(train_pi[t]) else float("nan")
        row["baseline_brier"] = baseline
        row["brier_skill"] = 1 - row["brier"] / baseline if baseline > 0 else float("nan")
        candidate_mask = mask & (d[:, t] == 0)
        hidden_y = z[candidate_mask, t]
        hidden_p = ranking[candidate_mask, t]
        hidden_h = h[candidate_mask, t] if h is not None else None
        row["n_unlabeled"] = len(hidden_y)
        row["n_hidden_positive"] = int(np.sum(hidden_y))
        row["n_h_undefined"] = int(np.sum(~np.isfinite(hidden_h))) if hidden_h is not None else len(hidden_y)
        rp, ap = _hidden_metrics(hidden_y, hidden_p)
        rh, ah = _hidden_metrics(hidden_y, hidden_h) if hidden_h is not None and np.all(np.isfinite(hidden_h)) else (float("nan"), float("nan"))
        row.update(hidden_recall_at_h_p_ranking=rp, hidden_auprc_p_ranking=ap,
                   hidden_recall_at_h_h_ranking=rh, hidden_auprc_h_ranking=ah,
                   hidden_recall_at_h=rh if p is not None else rp,
                   hidden_auprc=ah if p is not None else ap)
        positive_mask = mask & (z[:, t] == 1)
        detection_metrics = _metric_vector(d[positive_mask, t], e[positive_mask, t], n_bins=n_bins)
        row.update({f"detection_{key}": value for key, value in detection_metrics.items()})
        rows.append(row)
        for scope, truth, probability in (
            ("reference", y, pred), ("observed", obs, q_t),
            ("detection", d[positive_mask, t], e[positive_mask, t]),
        ):
            reliability.extend({"scope": scope, "target": target, "target_index": t, **b}
                               for b in reliability_bins(truth, probability, n_bins))
    numeric = [key for key in rows[0] if key not in {"target", "target_index", "probability_semantics"}] if rows else []
    summary: dict[str, Any] = {"probability_semantics": probability_semantics,
        "n_targets": len(names), "n_measured": int(np.sum(w)),
        "n_reference_positive": int(np.sum(z[w])), "n_observed_positive": int(np.sum(d[w])),
        "n_hidden_positive": int(np.sum(z[w] - d[w]))}
    for key in numeric:
        if key.startswith("n_"):
            continue
        values = [row[key] for row in rows]
        summary[f"macro_{key}"] = _nanmean(values)
        summary[f"n_targets_{key}"] = int(np.sum(np.isfinite(values)))
    for key in ("hidden_recall_at_h", "hidden_recall_at_h_p_ranking", "hidden_recall_at_h_h_ranking",
                "hidden_auprc", "hidden_auprc_p_ranking", "hidden_auprc_h_ranking"):
        summary[key] = summary.get(f"macro_{key}", float("nan"))
    for scope, truth, probability in (("reference", z[w], prediction[w]), ("observed", d[w], q[w])):
        prefix = "micro_" if scope == "reference" else "micro_observed_"
        rank_values = ranking[w] if scope == "reference" or probability_semantics == "observed" else None
        summary.update({prefix + key: value for key, value in _metric_vector(
            truth, probability, n_bins=n_bins, ranking_score=rank_values).items()})
        reliability.extend({"scope": scope, "target": "__pooled__", "target_index": -1, **b}
                           for b in reliability_bins(truth, probability, n_bins))
    bss_mask = w & np.isfinite(train_pi)[None, :]
    base = np.broadcast_to(train_pi, z.shape)
    base_brier = float(np.mean((z[bss_mask] - base[bss_mask]) ** 2)) if np.any(bss_mask) else float("nan")
    raw_brier = float(np.mean((z[bss_mask] - prediction[bss_mask]) ** 2)) if np.any(bss_mask) else float("nan")
    summary.update(micro_baseline_brier=base_brier,
                   micro_brier_skill=1 - raw_brier / base_brier if base_brier > 0 else float("nan"),
                   n_entries_brier_skill=int(np.sum(bss_mask)))
    hidden_mask = w & (d == 0)
    micro_rp, micro_ap = _hidden_metrics(z[hidden_mask], ranking[hidden_mask])
    micro_rh, micro_ah = (_hidden_metrics(z[hidden_mask], h[hidden_mask])
                         if h is not None and np.all(np.isfinite(h[hidden_mask]))
                         else (float("nan"), float("nan")))
    summary.update(micro_hidden_recall_at_h=micro_rh if p is not None else micro_rp,
                   micro_hidden_recall_at_h_p_ranking=micro_rp,
                   micro_hidden_recall_at_h_h_ranking=micro_rh,
                   micro_hidden_auprc=micro_ah if p is not None else micro_ap)
    return {"summary": summary, "per_target": rows, "reliability": reliability,
            "scores": {"prediction": prediction, "ranking_score": ranking,
                       "p": p, "q": q if probability_semantics != "mixed" else None,
                       "h": h, "e": e,
                       "mixed_label_score": prediction if probability_semantics == "mixed" else None}}


def evaluate_detection_calibration(reference: Any, observed: Any, measured: Any,
                                   estimated_sensitivity: Any, *, true_sensitivity: Any | None = None,
                                   target_ids: Sequence[str] | None = None, n_bins: int = 10) -> dict[str, Any]:
    """Detection proper scores on reference positives; oracle errors only if known.

    Synthetic propensity errors weight realized reference-positive entries
    equally. Natural assays have no known true sensitivity, so MAE/RMSE remain
    NaN rather than comparing to an empirical test-target detection fraction.
    """
    z, d = np.asarray(reference, dtype=float), np.asarray(observed, dtype=float)
    if z.ndim != 2 or d.shape != z.shape:
        raise ValueError("reference and observed must be aligned matrices")
    w = _binary_mask(measured, z.shape, "measured")
    if not np.all(np.isin(z[w], [0, 1])) or not np.all(np.isin(d[w], [0, 1])) or np.any(d[w] > z[w]):
        raise ValueError("Labels must satisfy binary one-sided PU observation")
    e = _probabilities(estimated_sensitivity, z.shape, w, "estimated_sensitivity")
    true_e = None if true_sensitivity is None else _probabilities(true_sensitivity, z.shape, w, "true_sensitivity")
    names = tuple(map(str, target_ids)) if target_ids is not None else tuple(map(str, range(z.shape[1])))
    if len(names) != z.shape[1] or len(set(names)) != len(names):
        raise ValueError("target_ids must be unique and aligned")
    rows, reliability = [], []
    for t, target in enumerate(names):
        mask = w[:, t] & (z[:, t] == 1)
        row = {"target": target, "target_index": t, "n_reference_positive": int(np.sum(mask)),
               **_metric_vector(d[mask, t], e[mask, t], n_bins=n_bins)}
        difference = e[mask, t] - true_e[mask, t] if true_e is not None else np.array([])
        row["sensitivity_mae"] = float(np.mean(np.abs(difference))) if len(difference) else float("nan")
        row["sensitivity_rmse"] = float(np.sqrt(np.mean(difference ** 2))) if len(difference) else float("nan")
        rows.append(row)
        reliability.extend({"scope": "detection", "target": target, "target_index": t, **b}
                           for b in reliability_bins(d[mask, t], e[mask, t], n_bins))
    positive = w & (z == 1)
    summary = {"n_reference_positive": int(np.sum(positive)),
               **_metric_vector(d[positive], e[positive], n_bins=n_bins)}
    difference = e[positive] - true_e[positive] if true_e is not None else np.array([])
    summary["sensitivity_mae"] = float(np.mean(np.abs(difference))) if len(difference) else float("nan")
    summary["sensitivity_rmse"] = float(np.sqrt(np.mean(difference ** 2))) if len(difference) else float("nan")
    for key in _metric_vector(np.array([]), np.array([]), n_bins=n_bins):
        summary[f"macro_{key}"] = _nanmean([row[key] for row in rows])
        summary[f"n_targets_{key}"] = int(np.sum(np.isfinite([row[key] for row in rows])))
    reliability.extend({"scope": "detection", "target": "__pooled__", "target_index": -1, **b}
                       for b in reliability_bins(d[positive], e[positive], n_bins))
    return {"summary": summary, "per_target": rows, "reliability": reliability}
