"""Reference scoring for structurally unmeasured cell--target pairs.

These pairs have no observed detection outcome. Consequently every model is
scored using its raw probability (and Qiao's raw score for discrimination), with
no conditioning on D=0 and no hidden-positive posterior h-ranking.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .evaluation import (_binary_mask, _metric_vector, _nanmean,
                         _probabilities, reliability_bins)


def _scores(truth, probability, ranking, n_bins):
    result = _metric_vector(truth, probability, n_bins=n_bins, ranking_score=ranking)
    # A fixed K=100 is not informative for many of these small blocked panels.
    result.pop("recall_at_100", None)
    return result


def evaluate_block_predictions(reference: Any, evaluation_mask: Any, prediction: Any,
                               *, target_ids: Sequence[str] | None = None,
                               groups: Any | None = None, ranking_score: Any | None = None,
                               train_reference_prevalence: Any | None = None,
                               n_bins: int = 10) -> dict[str, Any]:
    """Evaluate only previously measured entries hidden by the panel design.

    The caller supplies held-out cells only. Single-class target discrimination
    stays undefined, with finite-target coverage exported for every macro score.
    This function neither estimates detection nor generates pseudo outcomes.
    """
    z = np.asarray(reference, dtype=float)
    if z.ndim != 2:
        raise ValueError("reference must be a cell-by-target matrix")
    w = _binary_mask(evaluation_mask, z.shape, "evaluation_mask")
    if not np.all(np.isin(z[w], [0, 1])):
        raise ValueError("Evaluation reference must be binary on scored entries")
    probability = _probabilities(prediction, z.shape, w, "prediction")
    ranking = probability if ranking_score is None else np.asarray(ranking_score, dtype=float)
    if ranking.shape != z.shape or not np.all(np.isfinite(ranking[w])):
        raise ValueError("ranking_score must be aligned and finite on scored entries")
    targets = tuple(map(str, target_ids)) if target_ids is not None else tuple(map(str, range(z.shape[1])))
    if len(targets) != z.shape[1] or len(set(targets)) != len(targets):
        raise ValueError("target_ids must be unique and aligned")
    train_pi = (np.full(z.shape[1], np.nan) if train_reference_prevalence is None
                else np.asarray(train_reference_prevalence, dtype=float))
    if train_pi.shape != (z.shape[1],) or np.any(np.isinf(train_pi)) or np.any(
            (train_pi[np.isfinite(train_pi)] < 0) | (train_pi[np.isfinite(train_pi)] > 1)):
        raise ValueError("train_reference_prevalence must contain target probabilities or NaN")
    rows, reliability = [], []
    for index, target in enumerate(targets):
        mask = w[:, index]
        truth, pred, rank = z[mask, index], probability[mask, index], ranking[mask, index]
        row = {"target": target, "target_index": index,
               "n_evaluated": int(mask.sum()), "n_reference_positive": int(truth.sum()),
               **_scores(truth, pred, rank, n_bins)}
        baseline = float(np.mean((truth - train_pi[index]) ** 2)) if len(truth) and np.isfinite(train_pi[index]) else np.nan
        row.update(train_reference_prevalence=float(train_pi[index]) if len(truth) else np.nan, baseline_brier=baseline,
                   brier_skill=1-row["brier"]/baseline if baseline > 0 else np.nan)
        rows.append(row)
        reliability.extend({"scope": "blocked_reference", "target": target,
                            "target_index": index, **value}
                           for value in reliability_bins(truth, pred, n_bins))
    summary = {"n_targets": len(targets), "n_targets_evaluated": int(np.any(w, axis=0).sum()),
               "n_evaluated": int(w.sum()), "n_reference_positive": int(z[w].sum())}
    numeric = [key for key in rows[0] if key not in ("target", "target_index") and not key.startswith("n_")]
    for key in numeric:
        values = [row[key] for row in rows]
        summary[f"macro_{key}"] = _nanmean(values)
        summary[f"n_targets_{key}"] = int(np.isfinite(values).sum())
    summary.update({f"micro_{key}": value for key, value in
                    _scores(z[w], probability[w], ranking[w], n_bins).items()})
    reliability.extend({"scope": "blocked_reference", "target": "__pooled__",
                        "target_index": -1, **value}
                       for value in reliability_bins(z[w], probability[w], n_bins))
    per_group = []
    if groups is not None:
        groups = np.asarray(groups).astype(str)
        if groups.shape != (z.shape[0],):
            raise ValueError("groups must align with evaluation cells")
        for group in np.unique(groups):
            subset = groups == group
            result = evaluate_block_predictions(z[subset], w[subset], probability[subset],
                target_ids=targets, ranking_score=ranking[subset],
                train_reference_prevalence=train_pi, n_bins=n_bins)
            per_group.append({"block_group": group, **result["summary"]})
    return {"summary": summary, "per_target": rows, "reliability": reliability,
            "per_group": per_group}
