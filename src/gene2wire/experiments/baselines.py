"""Independent-target RF and prevalence baselines under explicit label budgets.

The caller supplies an authorized training label/mask view: detections for an
observed-only fit, paired references only for a reference-only fit. Validation
always uses original detections. A common RF hyperparameter configuration is
selected across all targets by mean measured-entry observed log loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from ..checkpoint import canonical_json, sha256_array, sha256_source_tree


@dataclass(frozen=True)
class BaselineResult:
    prediction: np.ndarray
    observed_prediction: np.ndarray
    selected_config: Mapping[str, Any]
    candidate_records: tuple[Mapping[str, Any], ...]
    probability_semantics: str
    diagnostics: Mapping[str, Any]


def baseline_candidates(kind: str = "random_forest", candidate_budget: int = 32) -> list[dict[str, Any]]:
    """A fixed, dataset-independent grid; unused budget is reported honestly."""
    if isinstance(candidate_budget, bool) or not isinstance(candidate_budget, (int, np.integer)) or candidate_budget < 1:
        raise ValueError("candidate_budget must be a positive integer")
    if kind == "prevalence":
        return [{}]
    if kind != "random_forest":
        raise ValueError("kind must be 'random_forest' or 'prevalence'")
    full = [{"n_estimators": 300, "min_samples_leaf": leaf,
             "max_features": features, "max_depth": depth}
            for leaf in (1, 5, 20) for features in ("sqrt", 0.5)
            for depth in (None, 12)]
    # Even coverage preserves multiple regularization levels under a small cap.
    indices = np.unique(np.linspace(0, len(full) - 1, min(candidate_budget, len(full))).round().astype(int))
    return [full[int(index)] for index in indices]


def _checked_training(X: Any, labels: Any, measured: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, raw_mask = np.asarray(X, dtype=float), np.asarray(labels, dtype=float), np.asarray(measured)
    if x.ndim != 2 or y.ndim != 2 or y.shape[0] != x.shape[0] or raw_mask.shape != y.shape:
        raise ValueError("Features, labels, and masks must have aligned matrix shapes")
    if not np.all(np.isfinite(x)):
        raise ValueError("Baseline features must be finite; impute using training-fitted preprocessing")
    if not np.all(np.isin(raw_mask, [0, 1])):
        raise ValueError("Training measured mask must be binary")
    mask = raw_mask.astype(bool)
    if not np.any(mask):
        raise ValueError("Baseline training requires at least one authorized label")
    if not np.all(np.isin(y[mask], [0, 1])):
        raise ValueError("Authorized baseline training labels must be binary")
    # Neither unassayed labels nor non-authorized reference labels affect fit or cache.
    return x, np.where(mask, y, 0.0), mask


def _prediction_features(X: Any, n_features: int) -> np.ndarray:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or x.shape[1] != n_features or not np.all(np.isfinite(x)):
        raise ValueError("Prediction features must be finite and aligned with training columns")
    return x


def _sensitivity(value: Any, shape: tuple[int, int]) -> np.ndarray:
    try:
        array = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    except ValueError as exc:
        raise ValueError("Sensitivity must broadcast to the prediction shape") from exc
    if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
        raise ValueError("Sensitivity must be finite and in [0, 1]")
    return array


def _fit_predict(x: np.ndarray, y: np.ndarray, mask: np.ndarray, x_predict: np.ndarray,
                 *, kind: str, config: Mapping[str, Any], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.empty((len(x_predict), y.shape[1]), dtype=float)
    pooled_training_prevalence = float(np.mean(y[mask]))
    target_counts = np.sum(mask, axis=0).astype(int)
    empty_targets, single_class_targets = [], []
    for target in range(y.shape[1]):
        selected = mask[:, target]
        target_y = y[selected, target]
        if not len(target_y):
            # Predeclared pooled authorized-training fallback, never test prevalence.
            result[:, target] = pooled_training_prevalence
            empty_targets.append(target)
        elif kind == "prevalence" or np.unique(target_y).size == 1:
            result[:, target] = float(np.mean(target_y))
            if np.unique(target_y).size == 1:
                single_class_targets.append(target)
        else:
            target_seed = int(np.random.SeedSequence([seed, target]).generate_state(1)[0])
            forest = RandomForestClassifier(**dict(config), random_state=target_seed,
                                            n_jobs=1, class_weight=None)
            forest.fit(x[selected], target_y.astype(int))
            # Both classes are present here; find the positive column explicitly.
            positive_column = int(np.flatnonzero(forest.classes_ == 1)[0])
            result[:, target] = forest.predict_proba(x_predict)[:, positive_column]
    return result, {"n_training_labels_per_target": target_counts.tolist(),
                    "empty_training_targets": empty_targets,
                    "single_class_training_targets": single_class_targets,
                    "empty_target_fallback": "pooled_authorized_training_prevalence",
                    "pooled_authorized_training_prevalence": pooled_training_prevalence}


def _fit_identity(x: np.ndarray, y: np.ndarray, mask: np.ndarray, x_predict: np.ndarray,
                  *, kind: str, config: Mapping[str, Any], seed: int) -> str:
    payload = {"kind": kind, "config": dict(config), "seed": int(seed),
               "source_hash": sha256_source_tree(Path(__file__).parent),
               "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
               "arrays": {name: sha256_array(value) for name, value in
                          (("X", x), ("labels", y), ("mask", mask), ("X_predict", x_predict))}}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _cached_fit_predict(x: np.ndarray, y: np.ndarray, mask: np.ndarray, x_predict: np.ndarray,
                        *, kind: str, config: Mapping[str, Any], seed: int,
                        checkpoint_dir: Path | None) -> tuple[np.ndarray, dict[str, Any], bool]:
    identity = _fit_identity(x, y, mask, x_predict, kind=kind, config=config, seed=seed)
    path = checkpoint_dir / f"baseline_{identity}.npz" if checkpoint_dir is not None else None
    if path is not None and path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                prediction = saved["prediction"]
                metadata = json.loads(str(saved["metadata"].item()))
            if (metadata["identity"] == identity and prediction.shape == (len(x_predict), y.shape[1])
                and metadata["prediction_hash"] == sha256_array(prediction)
                and np.all(np.isfinite(prediction)) and np.all((prediction >= 0) & (prediction <= 1))):
                return prediction, metadata["diagnostics"], True
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass  # Incomplete/corrupt caches are deterministically recomputed.
    prediction, diagnostics = _fit_predict(x, y, mask, x_predict, kind=kind, config=config, seed=seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = canonical_json({"identity": identity, "prediction_hash": sha256_array(prediction),
                                   "diagnostics": diagnostics})
        descriptor, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, prediction=prediction, metadata=np.array(metadata))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return prediction, diagnostics, False


def fit_baseline(
    X_train: Any, labels: Any, measured: Any, X_validation: Any,
    validation_observed: Any, validation_measured: Any, validation_sensitivity: Any,
    X_test: Any, test_sensitivity: Any, *,
    kind: Literal["random_forest", "prevalence"] = "random_forest",
    probability_semantics: Literal["reference", "observed"] = "observed",
    candidate_budget: int = 32, seed: int = 0, checkpoint_dir: str | Path | None = None,
    refit_X: Any | None = None, refit_labels: Any | None = None, refit_measured: Any | None = None,
    candidate_configs: Sequence[Mapping[str, Any]] | None = None,
) -> BaselineResult:
    """Tune globally on observed validation labels, then refit the chosen config.

    Optional ``refit_*`` arrays provide an authorized development-set label view
    after selection. They must be supplied together. Original validation D is
    never replaced with reference outcomes for selecting either baseline.

    Candidate-level checkpoints contain numeric predictions and checksummed
    metadata, not executable estimator pickles. Every candidate and final fit is
    keyed by data, actual source, configuration, seed, and library versions.
    """
    if probability_semantics not in {"reference", "observed"}:
        raise ValueError("probability_semantics must be 'reference' or 'observed'")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    candidates = baseline_candidates(kind, candidate_budget)
    if candidate_configs is not None:
        if not candidate_configs:
            raise ValueError("candidate_configs cannot be empty")
        candidates = [dict(config) for config in candidate_configs[:candidate_budget]]
    protected = {"n_jobs", "random_state", "class_weight"}
    for config in candidates:
        if protected.intersection(config):
            raise ValueError("Candidate configurations cannot override n_jobs, seed, or class_weight")
        if kind == "prevalence" and config:
            raise ValueError("The prevalence baseline has no hyperparameters")
    # Avoid counting duplicate configurations as extra tuning opportunities.
    candidates = list({canonical_json(config): config for config in candidates}.values())
    x, y, mask = _checked_training(X_train, labels, measured)
    xv, xt = _prediction_features(X_validation, x.shape[1]), _prediction_features(X_test, x.shape[1])
    dv, mv = np.asarray(validation_observed, dtype=float), np.asarray(validation_measured)
    if dv.shape != (len(xv), y.shape[1]) or mv.shape != dv.shape or not np.all(np.isin(mv, [0, 1])):
        raise ValueError("Validation labels and mask must align with cells and targets")
    mv = mv.astype(bool)
    if not np.any(mv) or not np.all(np.isin(dv[mv], [0, 1])):
        raise ValueError("Validation requires at least one measured, binary detection")
    ev, et = _sensitivity(validation_sensitivity, dv.shape), _sensitivity(test_sensitivity, (len(xt), y.shape[1]))
    supplied = [value is not None for value in (refit_X, refit_labels, refit_measured)]
    if any(supplied) and not all(supplied):
        raise ValueError("Supply refit_X, refit_labels, and refit_measured together")
    if all(supplied):
        xr, yr, mr = _checked_training(refit_X, refit_labels, refit_measured)
        if xr.shape[1] != x.shape[1] or yr.shape[1] != y.shape[1]:
            raise ValueError("Refit features/targets must align with tuning feature/target columns")
    else:
        xr, yr, mr = x, y, mask
    directory = Path(checkpoint_dir) if checkpoint_dir is not None else None
    records = []
    eps = np.finfo(np.float64).eps
    for index, config in enumerate(candidates):
        pred, diagnostics, resumed = _cached_fit_predict(x, y, mask, xv, kind=kind,
            config=config, seed=int(seed), checkpoint_dir=directory)
        qv = ev * pred if probability_semantics == "reference" else pred
        q = np.clip(qv[mv], eps, 1 - eps)
        loss = float(-np.mean(dv[mv] * np.log(q) + (1 - dv[mv]) * np.log1p(-q)))
        records.append({"candidate_index": index, "config": dict(config),
                        "validation_observed_log_loss": loss, "resumed": resumed,
                        "n_validation_entries": int(np.sum(mv)), **diagnostics})
    winner = min(range(len(records)), key=lambda index: (records[index]["validation_observed_log_loss"], index))
    selected = candidates[winner]
    prediction, fit_diagnostics, resumed = _cached_fit_predict(xr, yr, mr, xt,
        kind=kind, config=selected, seed=int(seed), checkpoint_dir=directory)
    q_test = et * prediction if probability_semantics == "reference" else prediction.copy()
    return BaselineResult(prediction=prediction, observed_prediction=q_test,
        selected_config=dict(selected), candidate_records=tuple(records),
        probability_semantics=probability_semantics,
        diagnostics={"kind": kind, "candidate_budget": int(candidate_budget),
                     "n_candidates_evaluated": len(candidates), "selected_candidate_index": winner,
                     "selection_rule": "minimum_mean_entry_observed_log_loss_then_candidate_order",
                     "final_refit_on_supplied_development_view": all(supplied),
                     "final_fit_resumed": resumed, **fit_diagnostics})
