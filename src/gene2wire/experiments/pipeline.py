"""Shared OnDemand execution: explicit information budgets, resume, and exports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import os
import platform
from pathlib import Path
import re
import tempfile
import time
import uuid
from typing import Any, Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from ..checkpoint import AtomicCheckpointStore, sha256_array, sha256_file
from ..data import DatasetBundle
from ..runner import run_model_grid
from ..seeds import stable_seed
from .calibration import (CalibrationNotEstimable, detection_diagnostics,
                          fit_detection_calibrator, sample_paired_rows)
from .contracts import (EvaluationSpec, ExperimentDataset, FeatureSet, Fold,
                        MeasurementEvaluationSpec)
from .evaluation import evaluate_detection_calibration, evaluate_predictions
from .io import atomic_json, atomic_npz, jsonable
from .observation import make_observation_design, thin_reference
from .protocol import Settings, fingerprint, scenarios, source_hash
from .progress import ProgressRelay
from .supervision import compile_training_bundle


@dataclass
class Artifacts:
    tables: dict[str, pd.DataFrame]
    export_dir: Path
    manifest: dict[str, Any]


@dataclass
class PreparedFold:
    name: str
    reference: np.ndarray
    measured: np.ndarray
    cell_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    groups: Mapping[str, np.ndarray]
    natural_observed: np.ndarray | None
    platform: np.ndarray | None
    technical_score: np.ndarray | None
    train_features: FeatureSet
    refit_features: FeatureSet
    fold: Fold
    metadata: dict
    evaluation: EvaluationSpec | None = None
    training_measured: np.ndarray | None = None
    virtual_assays: np.ndarray | None = None
    observation_plan: Any | None = None
    measurement_evaluation: MeasurementEvaluationSpec | None = None


@dataclass(frozen=True)
class DeferredPreparedFold:
    """Lightweight fold plan that builds dense features only in its worker.

    Gene-overlap experiments contain many views of the same cells.  Retaining
    both tuning- and refit-standardized full-N matrices for every view can use
    tens of GiB on SPIDER before training starts.  This proxy exposes only the
    metadata needed for scheduling; `_run_checkpointed_fold` materializes one
    pending fold at execution time.  A separate sequential hash pass below
    preserves the exact run identity without retaining those matrices.
    """

    dataset: ExperimentDataset
    fold: Fold

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.dataset.metadata

    @property
    def natural_observed(self):
        return self.dataset.natural_observed

    def materialize(self, settings: Settings) -> PreparedFold:
        return _prepare(self.dataset, self.fold, settings)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def _prepare(dataset, fold, settings):
    fold.validate(len(dataset.cell_ids))
    development = np.sort(np.r_[fold.train_rows, fold.validation_rows])
    train_features = dataset.feature_builder(fold.train_rows, settings.use_location,
                                            settings.use_target_features)
    refit_features = dataset.feature_builder(development, settings.use_location,
                                            settings.use_target_features)
    for features in (train_features, refit_features):
        if features.X.shape[0] != len(dataset.cell_ids):
            raise ValueError("Feature builders must return all rows in original cell order")
        if not np.all(np.isfinite(features.X)):
            raise ValueError("Feature builders must return finite X values")
        if features.X_nuisance is None:
            if features.nuisance_names:
                raise ValueError("nuisance_names requires FeatureSet.X_nuisance")
        else:
            nuisance = np.asarray(features.X_nuisance, dtype=float)
            if (
                nuisance.ndim != 2
                or nuisance.shape[0] != len(dataset.cell_ids)
                or nuisance.shape[1] < 1
                or not np.all(np.isfinite(nuisance))
            ):
                raise ValueError(
                    "FeatureSet.X_nuisance must be a finite all-row matrix"
                )
            if (
                len(features.nuisance_names) != nuisance.shape[1]
                or len(set(features.nuisance_names)) != nuisance.shape[1]
            ):
                raise ValueError("FeatureSet nuisance schema is invalid")
        if settings.use_target_features and features.Y_target is None:
            raise ValueError("USE_TARGET_FEATURES=True requires aligned target covariates")
        if not settings.use_target_features and features.Y_target is not None:
            raise ValueError("Disabled target features must not enter the learner")
    if train_features.nuisance_names != refit_features.nuisance_names:
        raise ValueError("Tuning and refit nuisance schemas must match")
    # Do not serialize the raw gene-count matrix into every worker.
    metadata = {k: v for k, v in dataset.metadata.items() if not isinstance(v, np.ndarray)}
    return PreparedFold(
        name=dataset.name,
        reference=dataset.reference,
        measured=dataset.measured,
        cell_ids=dataset.cell_ids,
        target_ids=dataset.target_ids,
        groups=dataset.groups,
        natural_observed=dataset.natural_observed,
        platform=dataset.platform,
        technical_score=dataset.technical_score,
        train_features=train_features,
        refit_features=refit_features,
        fold=fold,
        metadata=metadata,
        evaluation=dataset.evaluation,
        training_measured=(dataset.measured if dataset.training_measured is None
                           else dataset.training_measured),
        virtual_assays=dataset.virtual_assays,
        observation_plan=dataset.observation_plan,
        measurement_evaluation=dataset.measurement_evaluation,
    )


def _source_context(prepared, repetition):
    return {"dataset": prepared.name, "repetition": repetition,
            "outer_fold": prepared.fold.outer_fold,
            "sharing_strength": prepared.metadata.get("sharing_strength"),
            **prepared.metadata.get("experiment_context", {})}


def _model_seed_coordinate(prepared, repetition):
    """Separate model initialization from a measurement-panel replicate."""
    return int(prepared.metadata.get("model_seed", repetition))


def _paired_sampling_groups(prepared):
    """Return outcome-blind strata for the authorized paired-reference budget."""
    group_key = next((
        key for key in ("animal", "sample", "animal_id", "sample_id")
        if key in prepared.groups
    ), None)
    biological = None if group_key is None else np.asarray(
        prepared.groups[group_key]).astype(str)
    if prepared.virtual_assays is None:
        return biological, group_key or "none"
    if biological is None:
        biological = np.repeat("all", len(prepared.cell_ids))
    assays = np.asarray(prepared.virtual_assays).astype(str)
    combined = np.asarray([
        f"{bio}\0{assay}" for bio, assay in zip(biological, assays)
    ], dtype=str)
    return combined, f"{group_key or 'all_cells'} x virtual_assay"


def _scenarios(prepared, settings):
    experiment_context = prepared.metadata.get("experiment_context", {})
    if experiment_context.get("experiment") == "measurement_degradation":
        declared = prepared.metadata.get("measurement_scenarios")
        if not declared:
            raise ValueError("Measurement-degradation views require declared scenarios")
        required = {"analysis", "mechanism", "loss_rate", "positive_retention",
                    "calibration_fraction", "calibration_spec", "condition_roles"}
        result = []
        for scenario in declared:
            scenario = dict(scenario)
            missing = required.difference(scenario)
            if missing:
                raise ValueError(f"Measurement scenario is missing {sorted(missing)}")
            if scenario["mechanism"] not in {"assay_target_sar", "scar", "natural"}:
                raise ValueError("Unknown measurement-degradation observation mechanism")
            result.append(scenario)
        return result
    if settings.supervision_profile == "assay_only":
        return scenarios(settings, natural=prepared.natural_observed is not None, simulation=False)
    if experiment_context.get("observation_profile") == "technical_sar_80_sensitivity":
        if prepared.natural_observed is not None or tuple(settings.loss_rates) != (.8,):
            raise ValueError(
                "Technical-SAR 80 overlap sensitivity requires generated observations "
                "and loss_rates=(0.8,)"
            )
        return [{"analysis": "positive_label_loss_sensitivity",
                 "mechanism": "technical_sar", "loss_rate": .8,
                 "calibration_fraction": settings.paired_fraction,
                 "calibration_spec": "correct"}]
    if experiment_context.get("experiment") == "target_block":
        mechanism = "natural" if prepared.natural_observed is not None else "scar"
        rates = (None,) if mechanism == "natural" else settings.loss_rates
        return [{"analysis": "primary", "mechanism": mechanism, "loss_rate": rate,
                 "calibration_fraction": settings.paired_fraction,
                 "calibration_spec": "correct"} for rate in rates]
    return scenarios(settings, natural=prepared.natural_observed is not None,
                     simulation=prepared.metadata.get("independent_unit") == "generated_dataset",
                     sharing_strength=prepared.metadata.get("sharing_strength"))


def _fit_measured(prepared):
    """Learner-visible support, with a fallback for historical test fixtures."""
    value = getattr(prepared, "training_measured", None)
    return np.asarray(prepared.measured if value is None else value, dtype=bool)


def _bundle(prepared, features, observed):
    return DatasetBundle(X_cell=features.X, S_observed=observed,
                         W_measured=_fit_measured(prepared), Y_target=features.Y_target,
                         cell_ids=prepared.cell_ids, target_ids=prepared.target_ids,
                         groups=prepared.groups, feature_blocks=features.feature_blocks,
                         X_nuisance=features.X_nuisance,
                         nuisance_names=features.nuisance_names)


def _reference_view(reference, rows):
    """Only authorized paired labels exist in the learner-facing view."""
    result = np.full(reference.shape, np.nan)
    result[rows] = reference[rows]
    return result


def _train_prevalence(reference, measured, paired_rows):
    z, w = reference[paired_rows], measured[paired_rows]
    count = w.sum(axis=0)
    # A fixed Jeffreys correction remains finite for sparse target strata.
    return ((z * w).sum(axis=0) + .5) / (count + 1.)


def _detector(prepared, observed, paired, scenario, technical_score):
    if scenario["loss_rate"] == 0 and scenario["mechanism"] != "natural":
        return np.ones_like(observed, dtype=float), {"known_retention": 1.0}
    use_technical = (scenario["mechanism"] == "technical_sar"
                     and scenario["calibration_spec"] != "omit_technical")
    fitted = fit_detection_calibrator(
        observed, _reference_view(prepared.reference, paired), _fit_measured(prepared), paired,
        technical_score=technical_score if use_technical else None,
        platform=(prepared.virtual_assays if scenario["mechanism"] == "assay_target_sar"
                  else prepared.platform if scenario["mechanism"] == "natural" else None),
        use_technical=use_technical,
        use_target_effects=(scenario["mechanism"] != "scar"
                            and scenario["calibration_spec"] != "pooled_target"))
    estimated = fitted.predict(n_cells=len(prepared.cell_ids),
                               technical_score=technical_score if use_technical else None,
                               platform=(prepared.virtual_assays
                                         if scenario["mechanism"] == "assay_target_sar"
                                         else prepared.platform
                                         if scenario["mechanism"] == "natural" else None))
    return estimated, fitted.to_dict()


def _compile(prepared, features, observed, estimated, rows, paired, mode):
    base = _bundle(prepared, features, observed)
    compiled, exposure = compile_training_bundle(
        base, rows, paired, None if mode == "observed" else _reference_view(prepared.reference, paired),
        estimated, mode)
    return compiled.subset_rows(rows), exposure[rows]


def _run_fold(prepared, repetition, settings, checkpoint_dir, export_dir, code_hash,
              on_progress=None, scenario=None):
    fold = prepared.fold
    train, validation, test = fold.train_rows, fold.validation_rows, fold.test_rows
    development = np.sort(np.r_[train, validation])
    n, nt = prepared.reference.shape
    assay_only = settings.supervision_profile == "assay_only"
    draw_seed = stable_seed(settings.seed, "observation", prepared.name, repetition)
    measurement_plan = getattr(prepared, "observation_plan", None)
    design = (None if assay_only or measurement_plan is not None else make_observation_design(
        n, nt, seed=draw_seed, technical_score=prepared.technical_score))
    tables = {name: [] for name in ("metrics", "per_target", "reliability", "tuning",
                                   "selected", "detection", "detection_per_target",
                                   "detection_reliability", "thinning_audit",
                                   "observation_diagnostics", "failures", "per_group")}
    group_key = next((
        key for key in ("animal", "sample", "animal_id", "sample_id")
        if key in prepared.groups
    ), None)
    groups = None if group_key is None else prepared.groups[group_key]
    paired_groups, paired_stratification = _paired_sampling_groups(prepared)
    paired_seed = stable_seed(
        settings.seed, "paired", prepared.name,
        _model_seed_coordinate(prepared, repetition), fold.outer_fold,
    )
    source_context = _source_context(prepared, repetition)
    model_checkpoint = Path(checkpoint_dir) / slug(prepared.name)

    selected_scenarios = (_scenarios(prepared, settings)
                          if scenario is None else [scenario])
    for scenario in selected_scenarios:
        context = {**source_context, **scenario}
        def emit(event):
            if on_progress is not None:
                on_progress({**context, **event})
        emit({"event": "scenario_start"})
        paired = (np.array([], dtype=int) if assay_only else sample_paired_rows(
            development, scenario["calibration_fraction"], paired_seed,
            groups=(None if paired_groups is None
                    else paired_groups[development])))
        paired_train = np.intersect1d(paired, train)
        if assay_only:
            observed, true_e, gamma = prepared.reference, None, None
        elif scenario["mechanism"] == "natural":
            observed = np.where(_fit_measured(prepared), prepared.natural_observed, 0)
            true_e, gamma = None, None
        elif measurement_plan is not None:
            if scenario["mechanism"] == "scar":
                _, generated = measurement_plan.generate_pair(
                    prepared.reference, _fit_measured(prepared), train,
                    retention=float(scenario["positive_retention"]),
                )
            else:
                generated = measurement_plan.generate(
                    prepared.reference, _fit_measured(prepared), train,
                    retention=float(scenario["positive_retention"]),
                    mechanism="heterogeneous",
                )
            observed, true_e = generated.observed, generated.sensitivity
            gamma = dict(generated.diagnostics)
        else:
            generated = thin_reference(prepared.reference, _fit_measured(prepared), train,
                                       scenario["loss_rate"], scenario["mechanism"], design)
            observed, true_e, gamma = generated.observed, generated.sensitivity, generated.gamma
        try:
            if assay_only:
                # A single assay has no paired detection calibration. Ones denote
                # the absence of *additional* corruption, not biological sensitivity.
                tuning_e = final_e = np.ones_like(observed, dtype=float)
                tuning_detector = final_detector = None
            else:
                technical = None if design is None else design.technical_score
                tuning_e, tuning_detector = _detector(prepared, observed, paired_train, scenario, technical)
                # Separate refit objects never enter the validation score. Their paired
                # validation references are authorized only for development fitting.
                final_e, final_detector = _detector(prepared, observed, paired, scenario, technical)
        except CalibrationNotEstimable as error:
            tables["failures"].append({**context, "stage": "calibration", "error": str(error)})
            emit({"event": "calibration_failed", "error": str(error)})
            continue
        if gamma is not None:
            visible_e = (None if true_e is None else
                         np.asarray(true_e, dtype=float)[_fit_measured(prepared)])
            generator_diagnostics = (dict(gamma) if isinstance(gamma, Mapping)
                                     else {"generator_gamma": gamma})
            tables["observation_diagnostics"].append({
                **context, **generator_diagnostics,
                "true_sensitivity_min": (np.nan if visible_e is None or not len(visible_e)
                                         else float(np.min(visible_e))),
                "true_sensitivity_mean": (np.nan if visible_e is None or not len(visible_e)
                                          else float(np.mean(visible_e))),
                "true_sensitivity_max": (np.nan if visible_e is None or not len(visible_e)
                                         else float(np.max(visible_e))),
            })
        unit_id = fingerprint(context)
        audit_dir = Path(export_dir) / "units" / unit_id
        audit_dir.mkdir(parents=True, exist_ok=True)
        atomic_json({**context, "generator_gamma": gamma,
                     "train_cell_ids": np.asarray(prepared.cell_ids)[train],
                     "validation_cell_ids": np.asarray(prepared.cell_ids)[validation],
                     "test_cell_ids": np.asarray(prepared.cell_ids)[test],
                     "paired_train_cell_ids": np.asarray(prepared.cell_ids)[paired_train],
                     "paired_development_cell_ids": np.asarray(prepared.cell_ids)[paired],
                     "paired_stratification": paired_stratification,
                     "tuning_detector": tuning_detector, "refit_detector": final_detector,
                     "feature_names_tuning": prepared.train_features.feature_names,
                     "feature_names_refit": prepared.refit_features.feature_names,
                     "nuisance_names_tuning": prepared.train_features.nuisance_names,
                     "nuisance_names_refit": prepared.refit_features.nuisance_names,
                     "features_tuning": prepared.train_features.metadata,
                     "features_refit": prepared.refit_features.metadata,
                     "fold": fold.metadata,
                     **({"supervision_profile": "assay_only", "reference_interpretation": "observed assay outcome",
                         "exposure_interpretation": "no additional corruption; biological sensitivity unestimated"}
                        if assay_only else {})}, audit_dir / "audit.json")
        if not assay_only:
            diagnostic = detection_diagnostics(observed, prepared.reference, _fit_measured(prepared),
                                                final_e, rows=test, true_sensitivity=true_e)
            tables["detection"].append({**context, **diagnostic})
            detection_evaluation = evaluate_detection_calibration(
                prepared.reference[test], observed[test], _fit_measured(prepared)[test], final_e[test],
                true_sensitivity=None if true_e is None else true_e[test], target_ids=prepared.target_ids)
            tables["detection_per_target"].extend({**context, **row} for row in detection_evaluation["per_target"])
            tables["detection_reliability"].extend({**context, **row} for row in detection_evaluation["reliability"])
        observation_arrays = {
            "reference": prepared.reference[test], "observed": observed[test],
            "training_measured": _fit_measured(prepared)[test],
            "source_measured": prepared.measured[test],
            "estimated_sensitivity": final_e[test],
        }
        if design is not None:
            observation_arrays.update(
                technical_score=design.technical_score[test],
                target_offsets=design.target_offsets,
            )
        if prepared.virtual_assays is not None:
            observation_arrays["virtual_assay"] = np.asarray(prepared.virtual_assays)[test].astype(str)
        if true_e is not None:
            observation_arrays["true_sensitivity"] = true_e[test]
        atomic_npz(audit_dir / "observation.npz", **observation_arrays)
        for split_name, rows in (("train", train), ("validation", validation), ("test", test)):
            for target_index, target in enumerate(prepared.target_ids):
                measured_count = int(_fit_measured(prepared)[rows, target_index].sum())
                positive_count = int((prepared.reference[rows, target_index]
                                      * _fit_measured(prepared)[rows, target_index]).sum())
                detected_count = int(observed[rows, target_index].sum())
                tables["thinning_audit"].append({**context, "split": split_name, "target": target,
                    "measured_count": measured_count, "reference_positive_count": positive_count,
                    "detected_positive_count": detected_count,
                    "realized_positive_loss": 1-detected_count/positive_count if positive_count else np.nan})
        prior = _train_prevalence(prepared.reference, _fit_measured(prepared),
                                  development if assay_only else paired)

        def record(name, prediction, semantics, extra=None, *, ranking_score=None,
                   estimated_sensitivity=None):
            prefix = {**context, "model": name, "probability_semantics": semantics,
                      **(extra or {})}
            evaluation_e = final_e if estimated_sensitivity is None else np.asarray(
                estimated_sensitivity, dtype=float)
            if evaluation_e.shape == observed.shape:
                test_e = evaluation_e[test]
            elif evaluation_e.shape == prediction.shape:
                test_e = evaluation_e
            else:
                raise ValueError(
                    "Model-specific sensitivity must match all outcomes or test predictions")
            if prepared.measurement_evaluation is not None:
                from .block_evaluation import evaluate_block_predictions
                spec = prepared.measurement_evaluation
                fit_w = _fit_measured(prepared)
                scopes = {
                    "native_reference": np.asarray(spec.source_measured[test], dtype=bool),
                    "on_panel": np.asarray(fit_w[test], dtype=bool),
                    "off_panel": np.asarray(spec.source_measured[test], dtype=bool)
                                 & ~np.asarray(fit_w[test], dtype=bool),
                    "hidden_candidate": np.asarray(fit_w[test], dtype=bool)
                                        & (np.asarray(observed[test]) == 0),
                }
                for scope, mask in scopes.items():
                    evaluated = evaluate_block_predictions(
                        spec.reference[test], mask, prediction,
                        target_ids=prepared.target_ids, groups=spec.assays[test],
                        ranking_score=ranking_score,
                        train_reference_prevalence=prior)
                    evaluation_prefix = {**prefix, "evaluation_scope": scope}
                    tables["metrics"].append({**evaluation_prefix, **evaluated["summary"]})
                    for key in ("per_target", "reliability", "per_group"):
                        tables[key].extend({**evaluation_prefix, **row}
                                           for row in evaluated[key])
                atomic_npz(
                    audit_dir / f"{slug(name)}_predictions.npz",
                    prediction=prediction,
                    reference=spec.reference[test], observed=observed[test],
                    training_measured=fit_w[test],
                    source_measured=spec.source_measured[test],
                    estimated_sensitivity=test_e,
                    virtual_assay=np.asarray(spec.assays[test], dtype=str),
                    cell_ids=np.asarray(prepared.cell_ids)[test].astype(str),
                    target_ids=np.asarray(prepared.target_ids, dtype=str),
                    **({} if ranking_score is None else {"ranking_score": ranking_score}),
                )
                return
            if prepared.evaluation is not None:
                from .block_evaluation import evaluate_block_predictions
                spec = prepared.evaluation
                for fraction, mask in spec.masks.items():
                    evaluated = evaluate_block_predictions(
                        spec.reference[test], mask[test], prediction,
                        target_ids=prepared.target_ids, groups=spec.groups[test],
                        ranking_score=ranking_score,
                        train_reference_prevalence=prior)
                    evaluation_prefix = {**prefix, "evaluation_scope": "blocked",
                                         "block_fraction": float(fraction)}
                    tables["metrics"].append({**evaluation_prefix, **evaluated["summary"]})
                    for key in ("per_target", "reliability", "per_group"):
                        tables[key].extend({**evaluation_prefix, **row} for row in evaluated[key])
                atomic_npz(audit_dir / f"{slug(name)}_predictions.npz",
                           prediction=prediction, reference=spec.reference[test],
                           training_measured=prepared.measured[test],
                           source_measured=spec.source_measured[test],
                           block_fractions=np.asarray(tuple(spec.masks), dtype=float),
                           evaluation_masks=np.stack([mask[test] for mask in spec.masks.values()]),
                           block_groups=np.asarray(spec.groups[test], dtype=str),
                           cell_ids=np.asarray(prepared.cell_ids)[test].astype(str),
                           target_ids=np.asarray(prepared.target_ids, dtype=str),
                           **({} if ranking_score is None else {"ranking_score": ranking_score}))
                return
            evaluated = evaluate_predictions(prepared.reference[test], observed[test],
                prepared.measured[test], prediction, test_e,
                probability_semantics=semantics, train_reference_prevalence=prior,
                target_ids=prepared.target_ids, ranking_score=ranking_score)
            if assay_only:
                # The shared evaluator also computes paired-detector diagnostics.
                # A single assay provides no evidence for those quantities.
                def assay_fields(row):
                    return {key: value for key, value in row.items()
                            if not key.startswith("detection_") and "_detection_" not in key
                            and "hidden" not in key and key != "n_h_undefined"}
                evaluated["summary"] = assay_fields(evaluated["summary"])
                evaluated["per_target"] = [assay_fields(row) for row in evaluated["per_target"]]
                evaluated["reliability"] = [row for row in evaluated["reliability"]
                                            if row["scope"] != "detection"]
                evaluated["scores"].pop("e", None)
            tables["metrics"].append({**prefix, **evaluated["summary"]})
            tables["per_target"].extend({**prefix, **row} for row in evaluated["per_target"])
            tables["reliability"].extend({**prefix, **row} for row in evaluated["reliability"])
            evaluation_group = prepared.metadata.get("evaluation_group")
            if evaluation_group is not None:
                if evaluation_group not in prepared.groups:
                    raise ValueError(f"Unknown evaluation group {evaluation_group!r}")
                labels = np.asarray(prepared.groups[evaluation_group])[test]
                for label in np.unique(labels):
                    selected_rows = np.flatnonzero(labels == label)
                    group_evaluated = evaluate_predictions(
                        prepared.reference[test][selected_rows], observed[test][selected_rows],
                        prepared.measured[test][selected_rows], prediction[selected_rows],
                        test_e[selected_rows], probability_semantics=semantics,
                        train_reference_prevalence=prior, target_ids=prepared.target_ids,
                        ranking_score=None if ranking_score is None else ranking_score[selected_rows])
                    tables["per_group"].append({**prefix, "group_kind": evaluation_group,
                                                "group": str(label),
                                                **group_evaluated["summary"]})
            scores = {key: value for key, value in evaluated["scores"].items()
                      if value is not None and key != "prediction"}
            if ranking_score is not None:
                # Keep raw off-panel ranking scores for prediction-only exports;
                # the evaluator's score arrays intentionally mask W=0 with NaN.
                scores["ranking_score"] = ranking_score
            atomic_npz(audit_dir / f"{slug(name)}_predictions.npz", prediction=prediction,
                       reference=prepared.reference[test], observed=observed[test],
                       measured=prepared.measured[test],
                       **({"additional_retention": final_e[test]} if assay_only else {
                           "estimated_sensitivity": test_e}),
                       cell_ids=np.asarray(prepared.cell_ids)[test].astype(str),
                       target_ids=np.asarray(prepared.target_ids, dtype=str),
                       **({"group_ids": np.asarray(groups)[test].astype(str)}
                          if assay_only and groups is not None else {}), **scores)

        safe_validation = _bundle(prepared, prepared.train_features, observed).subset_rows(validation)
        training_mode = "observed" if assay_only else "calibrated_pu"
        train_bundle, train_e = _compile(prepared, prepared.train_features, observed, tuning_e,
                                         train, paired_train, training_mode)
        refit_bundle, refit_e = _compile(prepared, prepared.refit_features, observed, final_e,
                                         development, paired, training_mode)
        available_features = min(
            prepared.train_features.X.shape[1],
            prepared.refit_features.X.shape[1],
        )
        declared_rank_cap = prepared.metadata.get("tuning_rank_cap")
        if declared_rank_cap is not None:
            if (isinstance(declared_rank_cap, bool)
                    or not isinstance(declared_rank_cap, (int, np.integer))
                    or int(declared_rank_cap) < 1):
                raise ValueError(
                    "tuning_rank_cap must be a positive integer"
                )
            # Some direct-only controls (notably the zero-overlap
            # intersection arm) deliberately have fewer than K columns.  They
            # do not fit a low-rank candidate, while all comparable union/A
            # ablation arms are capped at exactly K.
            available_features = min(available_features, int(declared_rank_cap))
        tuning = settings.tuning_config(available_features, nt)
        models = settings.models()
        model_allowlist = prepared.metadata.get("model_allowlist")
        if model_allowlist is not None:
            allowed = set(map(str, model_allowlist))
            models = tuple(model for model in models if model.name in allowed)
            if not models:
                raise ValueError("model_allowlist removed every configured model")
        lowrank_feature_groups = tuple(
            prepared.metadata.get("model_lowrank_feature_groups", ())
        )
        if lowrank_feature_groups:
            if not any(model.kind == "lowrank" for model in models):
                raise ValueError(
                    "model_lowrank_feature_groups requires an allowed lowrank model"
                )
            models = tuple(
                model.with_updates(
                    lowrank_feature_groups=lowrank_feature_groups
                )
                if model.kind == "lowrank"
                else model
                for model in models
            )
        if scenario["analysis"].startswith("calibration"):
            models = tuple(m for m in models if m.pu)
        runner_context = {k: v for k, v in context.items() if k != "analysis"}

        def core_run(chosen_models, tb, te, rb, re, role):
            return run_model_grid(train=tb, validation=safe_validation,
                test_X=prepared.train_features.X[test], refit_test_X=prepared.refit_features.X[test],
                test_nuisance=(None if prepared.train_features.X_nuisance is None else
                               prepared.train_features.X_nuisance[test]),
                refit_test_nuisance=(None if prepared.refit_features.X_nuisance is None else
                                     prepared.refit_features.X_nuisance[test]),
                models=chosen_models, tuning=tuning, fit=settings.fit_config(),
                train_exposure=te, validation_exposure=tuning_e[validation],
                test_exposure=final_e[test], refit=rb, refit_exposure=re,
                test_cell_ids=np.asarray(prepared.cell_ids)[test],
                checkpoint_dir=model_checkpoint, unit_context={**runner_context, "supervision": role},
                seed=stable_seed(settings.seed, prepared.name,
                                 _model_seed_coordinate(prepared, repetition),
                                 fold.outer_fold),
                code_version=code_hash, on_progress=emit)

        result = core_run(models, train_bundle, train_e, refit_bundle, refit_e, training_mode)
        for name, model_result in result.models.items():
            semantics = "reference" if model_result.fitted.config.pu else "observed"
            record(name, model_result.latent_probability, semantics)
            tables["selected"].append({**context, **model_result.summary()})
            tables["tuning"].extend({**context, "model": name, **asdict(trial.config),
                                     "stage": trial.stage, "index": trial.index,
                                     "validation_loss": trial.validation_loss,
                                     "converged": trial.converged, "iterations": trial.iterations,
                                     "seed": trial.seed} for trial in model_result.tuning.trials)
        primary = scenario["analysis"] == "primary"
        if primary and settings.run_information_controls and not assay_only:
            pu_models = {model.name: model for model in settings.models() if model.pu}
            controls = (
                ("reference_only", (pu_models["PU"].with_updates(name="Reference-only"),)),
                ("reference_plus_pu", tuple(pu_models[base].with_updates(name=name)
                    for base, name in (("PU", "Reference+PU"),
                                       ("PU-MIRT", "Reference+PU-MIRT"),
                                       ("PU-Joint", "Reference+PU-Joint")))),
            )
            for role, control_models in controls:
                tb, te = _compile(prepared, prepared.train_features, observed, tuning_e,
                                  train, paired_train, role)
                rb, re = _compile(prepared, prepared.refit_features, observed, final_e,
                                  development, paired, role)
                # All mixed structures share exactly the same C/O labels and
                # exposures, as well as candidate and warm-start caches. C's
                # reference outcome replaces D; it is never counted twice.
                fitted_controls = core_run(control_models, tb, te, rb, re, role)
                for name, control in fitted_controls.models.items():
                    record(name, control.latent_probability, "reference",
                           {"supervision_mode": role, "uses_paired_reference": True,
                            "uses_pu_likelihood": role == "reference_plus_pu"})
                    tables["selected"].append({**context, **control.summary()})
                    tables["tuning"].extend({**context, "model": name, **asdict(trial.config),
                                             "stage": trial.stage, "index": trial.index,
                                             "validation_loss": trial.validation_loss,
                                             "converged": trial.converged, "iterations": trial.iterations,
                                             "seed": trial.seed} for trial in control.tuning.trials)
        # Rescaling is only defined here for target-constant detection mechanisms.
        if scenario["mechanism"] in ("scar", "target_sar") and "Logistic" in result.models:
            q = result.models["Logistic"].latent_probability
            unbounded = q / final_e[test]
            record("Logistic-rescaled", np.clip(unbounded, 0, 1), "reference",
                   {"rescaling_clip_fraction": float(np.mean(unbounded[prepared.measured[test]] > 1))})
        if primary and settings.run_random_forest:
            from .baselines import fit_baseline
            kind = "random_forest"
            for semantics, role in _random_forest_roles(settings):
                if role == "observed":
                    tb, rb = train_bundle, refit_bundle
                else:
                    tb, _ = _compile(prepared, prepared.train_features, observed, tuning_e,
                                     train, paired_train, role)
                    rb, _ = _compile(prepared, prepared.refit_features, observed, final_e,
                                     development, paired, role)
                name = f"RF-{semantics}"
                baseline_started = time.monotonic()
                emit({"event": "model_start", "model": name})
                baseline = fit_baseline(tb.X_cell, tb.S_observed, tb.W_measured,
                    safe_validation.X_cell, safe_validation.S_observed, safe_validation.W_measured,
                    tuning_e[validation], prepared.refit_features.X[test], final_e[test],
                    kind=kind, probability_semantics=semantics,
                    candidate_budget=settings.candidate_budget,
                    seed=stable_seed(settings.seed,
                                     _model_seed_coordinate(prepared, repetition),
                                     fold.outer_fold, kind),
                    # Fit keys include actual X/labels/masks/prediction X, seed,
                    # configuration and source. Share predictions across rates;
                    # selection still recomputes loss using this scenario's D/e.
                    checkpoint_dir=model_checkpoint / "baselines" / "shared_predictions",
                    refit_X=rb.X_cell, refit_labels=rb.S_observed, refit_measured=rb.W_measured,
                    on_progress=lambda event, model=name: emit({**event, "model": model}))
                emit({"event": "model_complete", "model": name,
                      "elapsed_seconds": time.monotonic()-baseline_started,
                      "cache_status": "checkpoint" if baseline.diagnostics.get("final_fit_resumed") else "fitted",
                      "summary": dict(baseline.selected_config)})
                record(name, baseline.prediction, semantics,
                       {"supervision_mode": role, "uses_paired_reference": semantics != "observed",
                        "uses_pu_likelihood": False})
                tables["selected"].append({**context, "model": name, **baseline.selected_config,
                                            "tuning_trials": len(baseline.candidate_records),
                                            **baseline.diagnostics})
                tables["tuning"].extend({**context, "model": name, **row} for row in baseline.candidate_records)
        if settings.run_qiao and scenario["analysis"] == "primary":
            _run_qiao_controls(prepared, observed, tuning_e, final_e, settings, context,
                               record, tables, model_checkpoint, on_progress=emit)
        if settings.run_pu_comparators and scenario["analysis"] == "primary":
            _run_pu_comparators(
                prepared, observed, tuning_e, final_e, settings, context,
                record, tables, checkpoint_dir=model_checkpoint,
                on_progress=emit,
            )
    return tables


def _atomic_csv(frame, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".csv")
    try:
        with os.fdopen(fd, "w") as handle:
            frame.to_csv(handle, index=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


_GROUP_COLUMNS = ["dataset", "sharing_strength", "analysis", "mechanism", "loss_rate",
                  "calibration_fraction", "calibration_spec", "model", "probability_semantics",
                  "experiment", "group_mode", "training_panel", "training_block_fraction",
                  "evaluation_scope", "block_fraction", "panel_design", "arm",
                  "requested_overlap", "actual_overlap", "panel_size",
                  "gene_requested_coverage", "gene_coverage",
                  "gene_overlap", "target_requested_coverage", "target_coverage",
                  "target_overlap", "positive_retention", "condition_roles"]


def _summarize(tables, *, simulation):
    metrics = tables["metrics"]
    if metrics.empty:
        return
    excluded = set(_GROUP_COLUMNS + [
        "outer_fold", "repetition", "panel_seed", "data_repetition",
    ])
    value_cols = [c for c in metrics.select_dtypes(include="number") if c not in excluded]
    groups = [c for c in _GROUP_COLUMNS if c in metrics]
    # Equal fold weights within repetition; the independent simulation unit is
    # the generated dataset. Fold*repeat is never the sample size of its CI.
    per_rep = metrics.groupby(groups + ["repetition"], dropna=False, observed=True)[value_cols].mean().reset_index()
    tables["per_repetition"] = per_rep
    tables["aggregate"] = per_rep.groupby(groups, dropna=False, observed=True)[value_cols].mean().reset_index()
    if simulation:
        rows = []
        for key, frame in per_rep.groupby(groups, dropna=False, observed=True):
            prefix = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
            for metric in value_cols:
                values = frame[metric].dropna().to_numpy()
                n = len(values)
                mean = float(values.mean()) if n else np.nan
                half = float(student_t.ppf(.975, n-1) * values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
                rows.append({**prefix, "metric": metric, "mean": mean, "n": n,
                             "ci95_low": mean-half, "ci95_high": mean+half})
        tables["simulation_intervals"] = pd.DataFrame(rows)
    identifiers = [c for c in metrics if c not in value_cols]
    tables["metrics_long"] = metrics.melt(id_vars=identifiers, value_vars=value_cols,
                                            var_name="metric", value_name="value")


def _random_forest_roles(settings):
    roles = (("observed", "observed"), ("reference", "reference_only"),
             ("mixed", "reference_plus_observed"))
    return roles[:1] if settings.supervision_profile == "assay_only" else roles


def _planned_models(prepared, settings):
    """Model evaluations per fold/repetition, before any outcome is examined."""
    rows = []
    for scenario in _scenarios(prepared, settings):
        allowlist = prepared.metadata.get("model_allowlist")
        names = [m.name for m in settings.models()
                 if m.pu or not scenario["analysis"].startswith("calibration")]
        if allowlist is not None:
            allowed = set(map(str, allowlist))
            names = [name for name in names if name in allowed]
        if scenario["analysis"] == "primary":
            if settings.run_information_controls and settings.supervision_profile != "assay_only":
                names += ["Reference-only", "Reference+PU", "Reference+PU-MIRT", "Reference+PU-Joint"]
            if settings.run_random_forest:
                names += [f"RF-{semantics}" for semantics, _ in _random_forest_roles(settings)]
            if settings.run_qiao:
                from .qiao import qiao_model_names
                names += list(qiao_model_names(settings.use_target_features))
            if settings.run_pu_comparators:
                names += ["GenEML-adapted", "Inductive-PU-MC", "SAR-PU"]
        rows.extend({**scenario, "model": name} for name in names)
    return rows


def _load_unit_result(store, key, run_id, export_path):
    cached = store.load(key, run_id)
    if cached is None:
        return None
    payload = cached["payload"]
    # A complete summary is reusable only while its predictions/audits exist
    # unchanged. Missing exports can be rebuilt from the finer model caches.
    for relative, digest in payload["artifact_hashes"].items():
        path = export_path / relative
        if not path.is_file() or sha256_file(path) != digest:
            return None
    tables = payload["tables"]
    for name, columns in payload.get("numeric_columns", {}).items():
        for row in tables[name]:
            for column in columns:
                if column in row and row[column] is None:
                    row[column] = np.nan
    for row in tables["selected"]:
        row["resumed"] = True
    return tables


def _run_checkpointed_fold(prepared, repetition, settings, checkpoint_dir,
                           export_path, fit_version, run_id, key, writer, scenario=None):
    started = time.monotonic()
    writer({"event": "unit_start"})
    if isinstance(prepared, DeferredPreparedFold):
        prepared = prepared.materialize(settings)
    tables = _run_fold(prepared, repetition, settings, checkpoint_dir, export_path,
                       fit_version, on_progress=writer, scenario=scenario)
    if not tables["failures"]:
        prefix = _source_context(prepared, repetition)
        files = []
        selected_scenarios = (_scenarios(prepared, settings)
                              if scenario is None else [scenario])
        for scenario in selected_scenarios:
            files.extend((export_path / "units" / fingerprint({**prefix, **scenario})).glob("*"))
        numeric = {name: list(pd.DataFrame(rows).select_dtypes(include="number").columns)
                   for name, rows in tables.items()}
        payload = {"tables": tables, "numeric_columns": numeric,
                   "artifact_hashes": {p.relative_to(export_path).as_posix(): sha256_file(p)
                                       for p in files if p.is_file()}}
        store = AtomicCheckpointStore(Path(checkpoint_dir) / "experiment_units" / run_id)
        store.save_complete(key, run_id, jsonable(payload))
    writer({"event": "unit_complete", "elapsed_seconds": time.monotonic()-started,
            "failed_scenarios": len(tables["failures"])})
    return tables


def _run_scenario_group(tasks, settings, checkpoint_dir, export_path, fit_version, run_id):
    """Run one or more independent scenarios in a single outer worker.

    Grouping changes scheduling only. Each scenario retains its own completed
    summary, model/candidate caches, event stream and deterministic seeds.
    """
    if not tasks:
        return []
    group_writer = tasks[0][4]
    group_writer({"event": "task_group_start"})
    try:
        return [(index, _run_checkpointed_fold(prepared, repetition, settings,
                  checkpoint_dir, export_path, fit_version, run_id, key, writer,
                  scenario=scenario))
                for index, prepared, repetition, key, writer, scenario in tasks]
    finally:
        group_writer({"event": "task_group_complete"})


def _execute(datasets, settings, checkpoint_dir, export_dir, *, progress=True,
             progress_interval=60., progress_level="summary", worker_status=None,
             export_name=None, supplementary_tables=None, manifest_extra=None):
    code_hash = source_hash()
    contexts = []
    metadata = []
    input_identities = []
    is_simulation = all(d.metadata.get("independent_unit") == "generated_dataset" for d in datasets)
    deferred_feature_preparation = False
    for dataset in datasets:
        dataset.validate()
        folds = dataset.split_builder(settings.n_outer_folds, settings.seed)
        visits = np.zeros(len(dataset.cell_ids), int)
        for fold in folds:
            visits[fold.test_rows] += 1
        if len(folds) != settings.n_outer_folds or not np.all(visits == 1):
            raise ValueError("Requested outer CV must test every cell exactly once")
        reps = ([int(dataset.metadata["experiment_repetition"])]
                if "experiment_repetition" in dataset.metadata else
                ([int(dataset.metadata["repetition"])] if is_simulation
                 else range(settings.n_repetitions)))
        defer_dataset = dataset.metadata.get("experiment_context", {}).get(
            "experiment") in {"gene_overlap", "measurement_degradation"}
        deferred_feature_preparation |= defer_dataset
        fold_inputs = []
        for fold in folds:
            prepared = _prepare(dataset, fold, settings)
            fold_inputs.append({"fold": fold.outer_fold,
                "train_rows": sha256_array(np.asarray(fold.train_rows)),
                "validation_rows": sha256_array(np.asarray(fold.validation_rows)),
                "test_rows": sha256_array(np.asarray(fold.test_rows)),
                "train_X": sha256_array(prepared.train_features.X),
                "refit_X": sha256_array(prepared.refit_features.X),
                "train_nuisance": (None if prepared.train_features.X_nuisance is None else
                                     sha256_array(prepared.train_features.X_nuisance)),
                "refit_nuisance": (None if prepared.refit_features.X_nuisance is None else
                                     sha256_array(prepared.refit_features.X_nuisance)),
                "nuisance_names": list(prepared.train_features.nuisance_names),
                "train_Y": None if prepared.train_features.Y_target is None else sha256_array(prepared.train_features.Y_target),
                "refit_Y": None if prepared.refit_features.Y_target is None else sha256_array(prepared.refit_features.Y_target)})
            scheduled = (DeferredPreparedFold(dataset, fold)
                         if defer_dataset else prepared)
            contexts.extend((scheduled, repetition) for repetition in reps)
            if defer_dataset:
                # Release the full-N matrices from the identity pass before
                # preparing the next view/fold.  Cached units never materialize
                # them again; pending units build them inside active workers.
                del prepared
        input_identities.append({"name": dataset.name,
            "repetition": dataset.metadata.get("experiment_repetition", dataset.metadata.get("repetition")),
            "experiment_context": dataset.metadata.get("experiment_context", {}),
            "model_seed": dataset.metadata.get("model_seed"),
            "model_allowlist": dataset.metadata.get("model_allowlist"),
            "evaluation": None if dataset.evaluation is None else {
                "reference": sha256_array(dataset.evaluation.reference),
                "source_measured": sha256_array(dataset.evaluation.source_measured),
                "groups": sha256_array(np.asarray(dataset.evaluation.groups, dtype=str)),
                "masks": {str(f): sha256_array(m) for f, m in dataset.evaluation.masks.items()}},
            "measurement_evaluation": None if dataset.measurement_evaluation is None else {
                "reference": sha256_array(dataset.measurement_evaluation.reference),
                "source_measured": sha256_array(dataset.measurement_evaluation.source_measured),
                "assays": sha256_array(np.asarray(dataset.measurement_evaluation.assays, dtype=str)),
            },
            "training_measured": (None if dataset.training_measured is None
                                  else sha256_array(dataset.training_measured)),
            "virtual_assays": (None if dataset.virtual_assays is None else
                               sha256_array(np.asarray(dataset.virtual_assays, dtype=str))),
            "observation_plan": (None if dataset.observation_plan is None else
                                 dataset.observation_plan.identity()),
            "sharing_strength": dataset.metadata.get("sharing_strength"),
            "reference": sha256_array(dataset.reference), "measured": sha256_array(dataset.measured),
            "cell_ids": sha256_array(np.asarray(dataset.cell_ids, dtype=str)),
            "target_ids": sha256_array(np.asarray(dataset.target_ids, dtype=str)),
            "groups": {name: sha256_array(np.asarray(values, dtype=str)) for name, values in dataset.groups.items()},
            "natural_observed": None if dataset.natural_observed is None else sha256_array(dataset.natural_observed),
            "technical_score": None if dataset.technical_score is None else sha256_array(dataset.technical_score),
            "fold_inputs": fold_inputs})
        metadata.append({"name": dataset.name, "n_cells": len(dataset.cell_ids),
                         "n_targets": len(dataset.target_ids),
                         "reference_hash": sha256_array(dataset.reference),
                         "measured_hash": sha256_array(dataset.measured),
                         "training_measured_hash": (None if dataset.training_measured is None else
                                                    sha256_array(dataset.training_measured)),
                         "metadata": {k: v for k, v in dataset.metadata.items() if not isinstance(v, np.ndarray)}})
    manifest = {"protocol": settings.scientific_dict(), "source_hash": code_hash,
                "datasets": metadata, "input_identities": input_identities,
                "feature_preparation": ("worker_lazy_after_hash_pass"
                                        if deferred_feature_preparation else "eager"),
                "uncertainty_unit": "generated_dataset" if is_simulation else "descriptive_fold_and_mask_repeat",
                "software": {"python": platform.python_version(), **{package: importlib.metadata.version(package)
                             for package in ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl")}}
                }
    run_id = fingerprint({"protocol": settings.scientific_dict(), "source_hash": code_hash,
                          "input_identities": input_identities, "software": manifest["software"]})
    fit_version = code_hash + ":" + fingerprint(manifest["software"])
    export_path = Path(export_dir) / (export_name or ("simulation_0908" if is_simulation else slug(datasets[0].name) + "_0908")) / run_id
    manifest.update(manifest_extra or {})
    export_path.mkdir(parents=True, exist_ok=True)
    manifest.update(run_id=run_id, requested_n_jobs=settings.n_jobs, completed=False,
                    checkpoint_dir=str(Path(checkpoint_dir).resolve()))
    atomic_json(manifest, export_path / "manifest.json")
    # Every completed-summary checkpoint is a fold/repetition/scenario unit,
    # independently of how those units are grouped for scheduling. The default
    # exposes loss rates to the outer process pool without nesting parallelism.
    store = AtomicCheckpointStore(Path(checkpoint_dir) / "experiment_units" / run_id)
    results, pending, inventory, evaluation_plan = [], [], [], []
    for fold_group, (prepared, repetition) in enumerate(contexts):
        planned_models = _planned_models(prepared, settings)
        for scenario in _scenarios(prepared, settings):
            index = len(results)
            unit_context = {**_source_context(prepared, repetition), **scenario}
            key = fingerprint({"run_id": run_id, **unit_context})
            cached = _load_unit_result(store, key, run_id, export_path)
            model_count = sum(all(row[name] == value for name, value in scenario.items())
                              for row in planned_models)
            inventory.append({**unit_context, "work_id": key, "fully_cached": cached is not None,
                              "planned_model_evaluations": model_count})
            evaluation_plan.extend({**unit_context, "work_id": key, "model": row["model"],
                                    "complete_summary_cached": cached is not None}
                                   for row in planned_models
                                   if all(row[name] == value for name, value in scenario.items()))
            results.append(cached)
            if cached is None:
                pending.append((index, prepared, repetition, key, unit_context, scenario, fold_group))
    planned_model_count = sum(row["planned_model_evaluations"] for row in inventory)
    if len(evaluation_plan) != planned_model_count or pd.DataFrame(evaluation_plan).duplicated(
        ["work_id", "model"]).any():
        raise ValueError("Progress plan contains duplicate or inconsistent model evaluation units")
    _atomic_csv(pd.DataFrame(evaluation_plan), export_path / "model_evaluation_plan.csv")
    cached_model_count = sum(row["planned_model_evaluations"] for row in inventory
                             if row["fully_cached"])
    groups = {}
    for index, p, r, key, context, scenario, fold_group in pending:
        group_key = index if settings.parallel_unit == "scenario" else fold_group
        groups.setdefault(group_key, []).append((index, p, r, key, context, scenario))
    workers = min(settings.n_jobs, len(groups))
    event_dir = export_path / "progress" / uuid.uuid4().hex
    relay = ProgressRelay(event_dir, enabled=progress, interval=progress_interval,
                          level=progress_level, total_units=planned_model_count,
                          cached_units=cached_model_count, worker_status=worker_status,
                          worker_slots=workers, requested_workers=settings.n_jobs)
    task_groups = [[(index, p, r, key, relay.writer(key, **context, work_id=key), scenario)
                    for index, p, r, key, context, scenario in group]
                   for group in groups.values()]
    manifest.update(parallel_unit=settings.parallel_unit, effective_n_jobs=workers,
                    scheduled_tasks=len(task_groups), planned_scenario_units=len(inventory),
                    pending_scenario_units=len(pending), worker_capacity=relay.capacity)
    atomic_json(manifest, export_path / "manifest.json")
    with relay:
        # Prepared dense arrays are shared through joblib memmaps when large.
        # Serial and process execution both use one BLAS thread, and RF/Qiao/core
        # remain serial inside a scenario; N_JOBS is the only process budget.
        with threadpool_limits(limits=1):
            if workers <= 1:
                new_results = [_run_scenario_group(group, settings, checkpoint_dir, export_path,
                               fit_version, run_id) for group in task_groups]
            else:
                with parallel_config(backend="loky", inner_max_num_threads=1):
                    new_results = Parallel(n_jobs=workers, verbose=0, max_nbytes="10M")(
                        delayed(_run_scenario_group)(group, settings, checkpoint_dir, export_path,
                            fit_version, run_id) for group in task_groups)
        for group_results in new_results:
            for index, result in group_results:
                results[index] = result
    names = results[0].keys()
    tables = {name: pd.DataFrame([row for result in results for row in result[name]]) for name in names}
    tables.update(supplementary_tables or {})
    tables["checkpoint_inventory"] = pd.DataFrame(inventory)
    tables["model_evaluation_plan"] = pd.DataFrame(evaluation_plan)
    accounting = [{**row, "accounting": "restored_results"} for row in evaluation_plan
                  if row["complete_summary_cached"]]
    tables["model_cache_accounting"] = pd.DataFrame(accounting + relay.model_accounting)
    expected_units = {(row["work_id"], row["model"]) for row in evaluation_plan}
    processed_units = [(row["work_id"], row["model"])
                       for row in accounting + relay.model_accounting]
    if (len(processed_units) != relay.done_units or len(set(processed_units)) != len(processed_units)
        or not set(processed_units).issubset(expected_units)):
        raise ValueError("Processed model units do not match the exported progress plan")
    tables["progress_events"] = pd.DataFrame(relay.rows)
    if len(datasets) == 1 and datasets[0].natural_observed is not None:
        data = datasets[0]
        standard = (data.natural_observed * data.measured).sum(axis=0)
        reference = (data.reference * data.measured).sum(axis=0)
        tables["paired_audit"] = pd.DataFrame({"target": data.target_ids,
            "standard_positive_count": standard, "reference_positive_count": reference,
            "amplification_only_positive_count": reference-standard,
            "relative_detection": np.divide(standard, reference,
                out=np.full(len(reference), np.nan), where=reference > 0)})
    _summarize(tables, simulation=is_simulation)
    from .reporting import joint_selection_diagnostics
    tables["joint_selection_diagnostics"] = joint_selection_diagnostics(tables)
    for name, table in tables.items():
        _atomic_csv(table, export_path / f"{name}.csv")
    manifest.update(completed=True, status="complete_with_failures" if len(tables["failures"]) else "complete",
                    failed_calibration_units=len(tables["failures"]),
                    planned_model_evaluations=planned_model_count,
                    completed_model_evaluations=relay.done_units,
                    cached_model_evaluations=cached_model_count,
                    reused_fit_model_evaluations=relay.reused_fit_units,
                    new_or_mixed_model_evaluations=relay.new_or_mixed_units,
                    unknown_fit_model_evaluations=relay.unknown_units,
                    metric_rows=len(tables["metrics"]), table_files=[f"{name}.csv" for name in tables])
    atomic_json(manifest, export_path / "manifest.json")
    print(f"All results exported to: {export_path}")
    if len(tables["failures"]):
        print(f"Calibration not estimable in {len(tables['failures'])} units; see failures.csv. Do not omit these from reporting.")
    return Artifacts(tables, export_path, manifest)


def run_experiment(dataset: ExperimentDataset, settings: Settings, *, checkpoint_dir, export_dir,
                   progress=True, progress_interval=60., progress_level="summary", worker_status=None):
    return _execute([dataset], settings, checkpoint_dir, export_dir, progress=progress,
                    progress_interval=progress_interval, progress_level=progress_level,
                    worker_status=worker_status)


def run_simulation_experiments(settings: Settings, *, raw_cache_dir, checkpoint_dir, export_dir,
                               sharing_strengths=(0., .5, 1.), simulation_options=None,
                               progress=True, progress_interval=60., progress_level="summary",
                               worker_status=None):
    from .datasets.simulation import generate_simulation
    options = {"truth_uses_location": settings.use_location, **(simulation_options or {})}
    datasets = []
    raw_cache_dir = Path(raw_cache_dir)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    for rho in sharing_strengths:
        for repetition in range(settings.n_repetitions):
            dataset = generate_simulation(repetition, rho, seed=settings.seed, **options)
            raw_id = fingerprint({"seed": settings.seed, "repetition": repetition, "rho": rho,
                                  "options": options, "source_hash": source_hash()})
            path = raw_cache_dir / f"simulation_{raw_id}.npz"
            if not path.exists():
                features = dataset.feature_builder
                atomic_npz(path, reference=dataset.reference, measured=dataset.measured,
                           genes=features.genes, location=features.location,
                           target_descriptors=features.target_descriptors,
                           technical_score=dataset.technical_score,
                           **{k: v for k, v in dataset.metadata.items()
                              if isinstance(v, np.ndarray) and k != "target_descriptors"})
            datasets.append(dataset)
    return _execute(datasets, settings, checkpoint_dir, export_dir, progress=progress,
                    progress_interval=progress_interval, progress_level=progress_level,
                    worker_status=worker_status)


def _run_qiao_controls(prepared, observed, tuning_e, final_e, settings, context,
                       record, tables, checkpoint_dir, on_progress=None):
    """Same-input bilinear controls with independent candidate/final checkpoints.

    Fits see only measured detections. Selection always uses the original
    validation detections, including for the squared-error response model.
    Checkpoints identify inputs and fitting settings, not unrelated model lists,
    worker counts, calibration settings, or evaluation reference labels.
    """
    from dataclasses import replace
    from .qiao import (QiaoFit, fit_qiao, qiao_candidate_grid,
                       qiao_model_names, qiao_target_inputs)
    from ..checkpoint import AtomicArrayCheckpointStore, unit_key
    # Sensitivities are intentionally not inputs to these observed-label fits.
    del tuning_e, final_e
    train, validation, test = prepared.fold.train_rows, prepared.fold.validation_rows, prepared.fold.test_rows
    dev = np.sort(np.r_[train, validation])
    y, yr, target_input_kind = qiao_target_inputs(
        prepared.train_features.Y_target, prepared.refit_features.Y_target,
        len(prepared.target_ids), use_target_features=settings.use_target_features)
    train_features = replace(prepared.train_features, Y_target=y)
    refit_features = replace(prepared.refit_features, Y_target=yr)
    fit_measured = _fit_measured(prepared)
    if not fit_measured[validation].any():
        raise ValueError("Qiao validation requires measured entries")
    candidates = qiao_candidate_grid(train_features.X.shape[1], refit_features.X.shape[1],
                                     y, yr, penalties=settings.penalties,
                                     candidate_budget=settings.candidate_budget)
    budget = min(settings.candidate_budget, 32)
    store = AtomicArrayCheckpointStore(Path(checkpoint_dir) / "qiao")
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "scipy")}
    code_hash = source_hash()
    cell_ids = np.asarray(prepared.cell_ids, dtype=str)
    target_ids = np.asarray(prepared.target_ids, dtype=str)
    validation_mask = np.asarray(fit_measured[validation], dtype=bool)
    validation_d = np.asarray(observed[validation], dtype=float)[validation_mask]
    if not np.all(np.isin(validation_d, [0, 1])):
        raise ValueError("Qiao validation detections must be binary on measured entries")

    def cached_fit(phase, features, rows, prediction_rows, config):
        x, target = features.X[rows], features.Y_target
        w = np.asarray(fit_measured[rows], dtype=bool)
        d = np.where(w, observed[rows], 0.0)
        prediction_x = features.X[prediction_rows]
        seed = stable_seed(settings.seed, "qiao", prepared.name,
                           _model_seed_coordinate(prepared, context.get("repetition")),
                           context.get("outer_fold"),
                           phase, config["objective"], config["rank"], config["l2"])
        inputs = {name: sha256_array(value) for name, value in (
            ("X", x), ("Y", target), ("D_authorized", d), ("W", w),
            ("X_prediction", prediction_x), ("training_ids", cell_ids[rows]),
            ("prediction_ids", cell_ids[prediction_rows]), ("target_ids", target_ids))}
        base = {"phase": phase, "config": config, "inputs": inputs, "source": code_hash,
                "target_input_kind": target_input_kind,
                "versions": versions, "seed": seed, "tolerance": settings.tolerance,
                "init_direct_maxiter": settings.init_direct_maxiter}

        def attempt(maxiter, initial=None):
            identity = fingerprint({**base, "maxiter": maxiter,
                "initial_fit": None if initial is None else {
                    "cell_loading": sha256_array(initial.cell_loading),
                    "target_loading": sha256_array(initial.target_loading),
                    "target_intercept": sha256_array(initial.target_intercept)}})
            key = unit_key(task="qiao_fit", phase=phase, identity=identity)
            cached = store.load(key, identity)
            if cached is not None:
                arrays, stats = cached["arrays"], cached["payload"]
                fitted = QiaoFit(
                    arrays["cell_loading"], arrays["target_loading"], arrays["target_intercept"],
                    arrays["target_features"], config["objective"], config["rank"], config["l2"],
                    stats["converged"], stats["iterations"], stats["objective_value"],
                    stats["optimizer_message"], stats["n_measured"])
                return fitted, arrays["prediction"], arrays["ranking_score"], True, identity
            fitted = fit_qiao(x, target, d, w, **config, seed=seed, maxiter=maxiter,
                              tolerance=settings.tolerance,
                              init_direct_maxiter=settings.init_direct_maxiter,
                              initial_fit=initial)
            probability = fitted.predict_proba(prediction_x)
            ranking_score = fitted.predict_score(prediction_x)
            if config["objective"] == "logit":
                ranking_score = probability
            stats = {"converged": fitted.converged, "iterations": fitted.iterations,
                     "objective_value": fitted.objective_value,
                     "optimizer_message": fitted.optimizer_message,
                     "n_measured": fitted.n_measured, "seed": seed,
                     "maxiter": maxiter, "continued": initial is not None}
            store.save_complete(key, identity, stats,
                {"prediction": probability, "ranking_score": ranking_score,
                 "cell_loading": fitted.cell_loading, "target_loading": fitted.target_loading,
                 "target_intercept": fitted.target_intercept, "target_features": target})
            return fitted, probability, ranking_score, False, identity

        fitted, prediction, ranking_score, resumed, identity = attempt(settings.maxiter)
        iterations = fitted.iterations
        retried = not fitted.converged and settings.retry_maxiter > 0
        if retried:
            fitted, prediction, ranking_score, retry_resumed, identity = attempt(settings.retry_maxiter, fitted)
            iterations += fitted.iterations
            resumed = resumed and retry_resumed
        diagnostics = {"converged": fitted.converged, "iterations": iterations,
                       "objective_value": fitted.objective_value,
                       "optimizer_message": fitted.optimizer_message,
                       "fit_retried": retried, "fit_attempts": 2 if retried else 1,
                       "seed": seed, "resumed": resumed, "checkpoint_identity": identity,
                       "n_measured": fitted.n_measured}
        return prediction, ranking_score, diagnostics

    for objective, name in zip(("squared_error", "logit"),
                               qiao_model_names(settings.use_target_features)):
        model_started = time.monotonic()
        if on_progress is not None:
            on_progress({"event": "model_start", "model": name})
        trials, best = [], None
        for index, candidate in enumerate(candidates):
            candidate_started = time.monotonic()
            config = {**candidate, "objective": objective}
            probability, _, diagnostics = cached_fit("candidate", train_features,
                                                     train, validation, config)
            q = np.clip(probability[validation_mask], 1e-7, 1 - 1e-7)
            loss = float(np.mean(-validation_d * np.log(q) - (1 - validation_d) * np.log1p(-q)))
            trial = {**config, **diagnostics, "index": index,
                     "target_input_kind": target_input_kind,
                     "validation_loss": loss, "selection_metric": "observed_log_loss",
                     "stage": "qiao_candidate"}
            trials.append(trial)
            if on_progress is not None:
                on_progress({"event": "candidate_complete", "model": name, "index": index+1,
                             "total": len(candidates), "config": config,
                             "cache_status": "checkpoint" if diagnostics["resumed"] else "fitted",
                             "elapsed_seconds": time.monotonic()-candidate_started})
            key = (not diagnostics["converged"], loss, config["rank"], -config["l2"])
            if best is None or key < best[0]:
                best = (key, config)
        if best is None:
            raise ValueError("Qiao has no valid candidates under the current feature dimensions")
        refit_started = time.monotonic()
        if on_progress is not None:
            on_progress({"event": "refit_start", "model": name, "stage": "refit",
                         "config": dict(best[1])})
        prediction, ranking_score, diagnostics = cached_fit("final", refit_features,
                                                           dev, test, best[1])
        if on_progress is not None:
            on_progress({"event": "refit_complete", "model": name, "stage": "refit",
                         "cache_status": "checkpoint" if diagnostics["resumed"] else "fitted",
                         "elapsed_seconds": time.monotonic() - refit_started})
        extra = {"use_target_features": settings.use_target_features,
                 "target_input_kind": target_input_kind,
                 "probability_transform": "sigmoid" if objective == "logit" else "clip_linear_score",
                 "probability_clip_fraction": float(np.mean((ranking_score < 1e-7) |
                                                              (ranking_score > 1 - 1e-7)))
                    if objective == "squared_error" else 0.0}
        record(name, prediction, "observed", extra, ranking_score=ranking_score)
        tables["selected"].append({**context, "model": name, **best[1], **diagnostics,
                                    "final_converged": diagnostics["converged"],
                                    "final_iterations": diagnostics["iterations"],
                                    "final_fit_retried": diagnostics["fit_retried"],
                                    "selected_structure": ("target_covariate_bilinear"
                                        if settings.use_target_features else "target_identity_bilinear"),
                                    "use_target_features": settings.use_target_features,
                                    "target_input_kind": target_input_kind,
                                    "selection_metric": "observed_log_loss",
                                    "validation_loss": best[0][1],
                                    "tuning_trials": len(trials), "candidate_budget": budget})
        tables["tuning"].extend({**context, "model": name, **trial} for trial in trials)
        if on_progress is not None:
            on_progress({"event": "model_complete", "model": name,
                         "elapsed_seconds": time.monotonic()-model_started,
                         "cache_status": "checkpoint" if diagnostics["resumed"] else "fitted",
                         "summary": {**best[1], "final_converged": diagnostics["converged"]}})


def _run_pu_comparators(prepared, observed, tuning_e, final_e, settings, context,
                        record, tables, checkpoint_dir=None, on_progress=None):
    """Tune/refit the three predeclared PU comparators on identical inputs.

    GenEML and SAR-EM estimate contextual exposure from observed PU labels and
    outcome-independent assay/target design.  Inductive-PU-MC receives the same
    paired-calibration exposure as the Gene2Wire PU models.  Completed fits are
    independently checkpointed; evaluation reference values are never inputs
    to either the fit or its checkpoint identity.
    """
    from dataclasses import fields as dataclass_fields

    from ..candidate_design import select_balanced_candidates
    from ..checkpoint import AtomicArrayCheckpointStore, unit_key
    from ..config import ModelConfig
    from .pu_comparators import (
        AssayTargetPropensityEncoder,
        GenEMLFit,
        SAREMFit,
        ShiftIMCFit,
        fit_geneml_adapted,
        fit_sar_em,
        fit_shift_imc_adapted,
    )

    train = np.asarray(prepared.fold.train_rows, dtype=int)
    validation = np.asarray(prepared.fold.validation_rows, dtype=int)
    test = np.asarray(prepared.fold.test_rows, dtype=int)
    development = np.sort(np.r_[train, validation])
    fit_w = _fit_measured(prepared)
    labels = np.asarray(
        prepared.virtual_assays if prepared.virtual_assays is not None
        else prepared.platform if prepared.platform is not None
        else np.repeat("assay", len(prepared.cell_ids)),
        dtype=str,
    )
    target_ids = tuple(map(str, prepared.target_ids))

    def propensity(rows, encoder):
        return encoder.transform(labels[rows], target_ids)

    train_encoder = AssayTargetPropensityEncoder.fit(labels[train], target_ids)
    refit_encoder = AssayTargetPropensityEncoder.fit(labels[development], target_ids)
    train_phi = propensity(train, train_encoder)
    validation_phi = propensity(validation, train_encoder)
    development_phi = propensity(development, refit_encoder)
    test_phi = propensity(test, refit_encoder)

    train_x = np.asarray(prepared.train_features.X[train], dtype=float)
    validation_x = np.asarray(prepared.train_features.X[validation], dtype=float)
    development_x = np.asarray(prepared.refit_features.X[development], dtype=float)
    test_x = np.asarray(prepared.refit_features.X[test], dtype=float)
    train_s, validation_s = np.asarray(observed[train]), np.asarray(observed[validation])
    development_s = np.asarray(observed[development])
    train_w, validation_w = np.asarray(fit_w[train]), np.asarray(fit_w[validation])
    development_w = np.asarray(fit_w[development])
    train_target = (prepared.train_features.Y_target
                    if settings.use_target_features else None)
    refit_target = (prepared.refit_features.Y_target
                    if settings.use_target_features else None)

    if settings.use_target_features:
        if train_target is None or refit_target is None:
            raise ValueError("Comparator target-feature mode requires train/refit target designs")
        train_target = np.asarray(train_target, dtype=float)
        refit_target = np.asarray(refit_target, dtype=float)
        expected_rows = len(target_ids)
        if (train_target.ndim != 2 or refit_target.ndim != 2
                or train_target.shape[0] != expected_rows
                or refit_target.shape[0] != expected_rows
                or train_target.shape[1] == 0 or refit_target.shape[1] == 0
                or not np.all(np.isfinite(train_target))
                or not np.all(np.isfinite(refit_target))):
            raise ValueError("Comparator target designs must align with the known target panel")

    penalties = tuple(sorted(set(float(value) for value in settings.penalties)))
    ranks = tuple(rank for rank in settings.tuning_config(
        min(train_x.shape[1], development_x.shape[1]), len(target_ids)).ranks
                  if rank > 0)
    if not ranks:
        ranks = (1,)
    budget = min(int(settings.candidate_budget), 32)

    def balanced(rows, model_configs):
        rows, model_configs = tuple(rows), tuple(model_configs)
        if not rows or len(rows) != len(model_configs):
            raise ValueError("Comparator candidate grid is empty or internally inconsistent")
        selected = select_balanced_candidates(
            model_configs, min(budget, len(model_configs)))
        return tuple(rows[model_configs.index(candidate)] for candidate in selected)

    geneml_rows = [
        {"rank": rank, "l2_u": penalty, "l2_v": penalty,
         "l2_map": penalty, "l2_exposure": penalty}
        for rank in ranks
        if rank <= min(train_x.shape[1] + 1, development_x.shape[1] + 1,
                       len(target_ids))
        for penalty in penalties
    ]
    geneml_candidates = balanced(geneml_rows, [
        ModelConfig(name="GenEML-grid", kind="lowrank", rank=row["rank"],
                    shared_l2=row["l2_u"])
        for row in geneml_rows
    ])

    shift_rank_cap = min(
        train_x.shape[1] + 1, development_x.shape[1] + 1, len(target_ids))
    if train_target is not None:
        shift_rank_cap = min(
            shift_rank_cap, train_target.shape[1] + 1, refit_target.shape[1] + 1)
    shift_rows = [
        {"rank": rank, "l2": penalty}
        for rank in ranks if rank <= shift_rank_cap
        for penalty in penalties
    ]
    shift_candidates = balanced(shift_rows, [
        ModelConfig(name="ShiftIMC-grid", kind="lowrank", rank=row["rank"],
                    shared_l2=row["l2"])
        for row in shift_rows
    ])

    sar_rows = [
        {"l2_classifier": classifier, "l2_propensity": propensity_penalty}
        for classifier in penalties for propensity_penalty in penalties
    ]
    sar_candidates = balanced(sar_rows, [
        ModelConfig(name="SAR-grid", kind="joint", rank=1,
                    shared_l2=row["l2_classifier"],
                    residual_l2=row["l2_propensity"])
        for row in sar_rows
    ])
    specifications = (
        ("GenEML-adapted", geneml_candidates),
        ("Inductive-PU-MC", shift_candidates),
        ("SAR-PU", sar_candidates),
    )

    base_seed = stable_seed(
        settings.seed, "pu_comparators", prepared.name,
        _model_seed_coordinate(prepared, context.get("repetition")),
        context.get("outer_fold"),
    )
    store = (None if checkpoint_dir is None else AtomicArrayCheckpointStore(
        Path(checkpoint_dir) / "pu_comparators"))
    versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "scipy")
    }
    code_hash = source_hash()
    cell_ids = np.asarray(prepared.cell_ids, dtype=str)
    target_id_array = np.asarray(target_ids, dtype=str)
    fit_classes = {
        "GenEMLFit": GenEMLFit,
        "ShiftIMCFit": ShiftIMCFit,
        "SAREMFit": SAREMFit,
    }

    def checkpoint_coordinates(name, phase, config, seed, options, inputs):
        identity = fingerprint({
            "model": name, "phase": phase, "config": dict(config),
            "seed": int(seed), "options": dict(options),
            "inputs": {key: sha256_array(value)
                       for key, value in sorted(inputs.items())},
            "source": code_hash, "versions": versions,
        })
        key = unit_key(task="pu_comparator_fit", model=name,
                       phase=phase, identity=identity)
        return key, identity

    def restore_fit(cached):
        payload = dict(cached["payload"])
        class_name = payload.pop("fit_class", None)
        if class_name not in fit_classes:
            raise ValueError(f"Unknown comparator fit class in checkpoint: {class_name!r}")
        values = {**payload, **cached["arrays"]}
        expected = {field.name for field in dataclass_fields(fit_classes[class_name])}
        if set(values) != expected:
            raise ValueError("Comparator checkpoint fields do not match the fitted API")
        return fit_classes[class_name](**values)

    def cached_fit(name, phase, config, seed, options, inputs, fitter):
        key, identity = checkpoint_coordinates(
            name, phase, config, seed, options, inputs)
        if store is not None:
            cached = store.load(key, identity)
            if cached is not None:
                return restore_fit(cached), True, identity
        fitted = fitter()
        if store is not None:
            payload, arrays = {"fit_class": type(fitted).__name__}, {}
            for field in dataclass_fields(fitted):
                value = getattr(fitted, field.name)
                if isinstance(value, np.ndarray):
                    arrays[field.name] = value
                else:
                    payload[field.name] = value
            store.save_complete(key, identity, payload, arrays)
        return fitted, False, identity

    def checkpoint_available(name, phase, config, seed, options, inputs):
        if store is None:
            return False
        key, identity = checkpoint_coordinates(
            name, phase, config, seed, options, inputs)
        try:
            return store.is_complete(key, identity)
        except (OSError, ValueError):
            # The actual fit call will surface and retain a corrupt-checkpoint
            # failure as a candidate row rather than hiding it in inventory.
            return False

    def observed_loss(probability):
        q = np.asarray(probability, dtype=float)
        if q.shape != validation_s.shape:
            raise ValueError("Comparator validation probability has the wrong shape")
        mask = validation_w.astype(bool)
        if not np.any(mask):
            raise ValueError("Comparator validation requires measured entries")
        values = q[mask]
        if (not np.all(np.isfinite(values))
                or np.any(values < 0) or np.any(values > 1)):
            raise ValueError("Comparator validation probabilities must lie in [0, 1]")
        values = np.clip(values, 1e-9, 1 - 1e-9)
        labels_on_mask = np.asarray(validation_s[mask], dtype=float)
        return float(np.mean(
            -labels_on_mask * np.log(values)
            - (1 - labels_on_mask) * np.log1p(-values)))

    common_candidate_inputs = {
        "X": train_x,
        "S_authorized": np.where(train_w, train_s, 0),
        "W": train_w,
        "training_ids": cell_ids[train],
        "target_ids": target_id_array,
        "feature_names": np.asarray(prepared.train_features.feature_names, dtype=str),
    }
    common_refit_inputs = {
        "X": development_x,
        "S_authorized": np.where(development_w, development_s, 0),
        "W": development_w,
        "training_ids": cell_ids[development],
        "target_ids": target_id_array,
        "feature_names": np.asarray(prepared.refit_features.feature_names, dtype=str),
    }

    for name, candidates in specifications:
        started = time.monotonic()
        if on_progress is not None:
            on_progress({"event": "model_start", "model": name})

        if name == "GenEML-adapted":
            tolerance = max(settings.tolerance, 1e-5)
            candidate_options = {"maxiter": settings.maxiter, "tolerance": tolerance}
            refit_options = {"maxiter": settings.retry_maxiter, "tolerance": tolerance}
            candidate_inputs = {
                **common_candidate_inputs, "propensity_design": train_phi,
                "propensity_feature_names": np.asarray(
                    train_encoder.feature_names, dtype=str),
            }
            refit_inputs = {
                **common_refit_inputs, "propensity_design": development_phi,
                "propensity_feature_names": np.asarray(
                    refit_encoder.feature_names, dtype=str),
            }

            def fit_candidate(config, seed):
                return fit_geneml_adapted(
                    train_x, train_s, train_w, train_phi, **config,
                    **candidate_options, seed=seed)

            def candidate_probability(fitted):
                return fitted.predict_observed(validation_x, validation_phi)

            def fit_refit(config, seed):
                return fit_geneml_adapted(
                    development_x, development_s, development_w,
                    development_phi, **config, **refit_options, seed=seed)

            def final_probabilities(fitted):
                return (fitted.predict_reference(test_x),
                        fitted.predict_exposure(test_x, test_phi))

            metadata = {
                "comparator_source": "Jain-Modhe-Rai-2017",
                "adapted": True, "underlying_method": "GenEML",
                "adaptation_note": "known-W contextual-exposure Python3 point-EM",
                "target_input_kind": "known_target_identity",
                "information_access": "observed labels; assay-target propensity design",
            }
        elif name == "Inductive-PU-MC":
            candidate_options = {
                "maxiter": settings.maxiter, "tolerance": settings.tolerance}
            refit_options = {
                "maxiter": settings.retry_maxiter, "tolerance": settings.tolerance}
            candidate_inputs = {
                **common_candidate_inputs, "exposure": tuning_e[train],
                **({} if train_target is None else {"target_design": train_target}),
            }
            refit_inputs = {
                **common_refit_inputs, "exposure": final_e[development],
                **({} if refit_target is None else {"target_design": refit_target}),
            }

            def fit_candidate(config, seed):
                return fit_shift_imc_adapted(
                    train_x, train_s, train_w, tuning_e[train], train_target,
                    **config, **candidate_options, seed=seed)

            def candidate_probability(fitted):
                return fitted.predict_observed(
                    validation_x, tuning_e[validation], train_target)

            def fit_refit(config, seed):
                return fit_shift_imc_adapted(
                    development_x, development_s, development_w,
                    final_e[development], refit_target, **config,
                    **refit_options, seed=seed)

            def final_probabilities(fitted):
                return (fitted.predict_reference(test_x, refit_target),
                        np.asarray(final_e[test], dtype=float))

            metadata = {
                "comparator_source": "Hsieh-Natarajan-Dhillon-2015",
                "adapted": True, "underlying_method": "ShiftIMC-adapted",
                "adaptation_note": "known-W entry-specific-e sigmoid factorization",
                "target_input_kind": (
                    "target_features" if train_target is not None
                    else "known_target_identity"),
                "information_access": (
                    "observed labels; estimated paired-calibration exposure"),
            }
        else:
            window = max(2, min(10, settings.maxiter))
            candidate_options = {
                "maxiter": max(window, settings.maxiter),
                "tolerance": max(settings.tolerance, 1e-4),
                "convergence_window": window,
                "inner_maxiter": max(20, min(200, settings.maxiter)),
            }
            refit_options = {
                "maxiter": max(window, settings.retry_maxiter),
                "tolerance": max(settings.tolerance, 1e-4),
                "convergence_window": window,
                "inner_maxiter": max(20, min(200, settings.retry_maxiter)),
            }
            candidate_inputs = {
                **common_candidate_inputs, "propensity_design": train_phi,
                "propensity_feature_names": np.asarray(
                    train_encoder.feature_names, dtype=str),
            }
            refit_inputs = {
                **common_refit_inputs, "propensity_design": development_phi,
                "propensity_feature_names": np.asarray(
                    refit_encoder.feature_names, dtype=str),
            }

            def fit_candidate(config, seed):
                return fit_sar_em(
                    train_x, train_s, train_w, train_phi, **config,
                    **candidate_options, seed=seed)

            def candidate_probability(fitted):
                return fitted.predict_observed(validation_x, validation_phi)

            def fit_refit(config, seed):
                return fit_sar_em(
                    development_x, development_s, development_w,
                    development_phi, **config, **refit_options, seed=seed)

            def final_probabilities(fitted):
                return (fitted.predict_reference(test_x),
                        fitted.predict_exposure(test_x, test_phi))

            metadata = {
                "comparator_source": "Bekker-Davis-2018",
                "adapted": False, "underlying_method": "SAR-EM",
                "adaptation_note": "faithful SAR-EM over measured cell-target dyads",
                "target_input_kind": "known_target_identity",
                "information_access": (
                    "observed labels only; assay-target propensity design"),
            }

        candidate_seeds = [
            stable_seed(base_seed, name, "candidate", fingerprint(config))
            for config in candidates
        ]
        cached_candidates = [
            checkpoint_available(
                name, "candidate", config, seed, candidate_options,
                candidate_inputs)
            for config, seed in zip(candidates, candidate_seeds)
        ]
        if on_progress is not None:
            cached_count = sum(cached_candidates)
            on_progress({
                "event": "candidate_inventory", "model": name,
                "stage": "tuning", "total": len(candidates),
                "cached": cached_count, "memory_cached": 0,
                "checkpoint_cached": cached_count,
                "pending": len(candidates) - cached_count,
            })

        valid_trials = []
        for index, (config, seed) in enumerate(
                zip(candidates, candidate_seeds)):
            candidate_started = time.monotonic()
            if on_progress is not None:
                on_progress({
                    "event": "candidate_start", "model": name,
                    "stage": "tuning", "index": index + 1,
                    "total": len(candidates), "config": dict(config),
                    "cache_status": (
                        "checkpoint" if cached_candidates[index] else "pending"),
                    "elapsed_seconds": 0.0,
                })
            try:
                fitted, resumed, identity = cached_fit(
                    name, "candidate", config, seed, candidate_options,
                    candidate_inputs,
                    lambda config=config, seed=seed: fit_candidate(config, seed),
                )
                loss = observed_loss(candidate_probability(fitted))
                elapsed = time.monotonic() - candidate_started
                row = {
                    **context, **metadata, **dict(config), "model": name,
                    "index": index, "seed": seed,
                    "validation_loss": loss,
                    "selection_metric": "observed_log_loss",
                    "stage": "comparator_candidate", "status": "complete",
                    "converged": bool(fitted.converged),
                    "iterations": int(fitted.iterations),
                    "objective_value": float(fitted.objective_value),
                    "elapsed_seconds": elapsed, "resumed": resumed,
                    "checkpoint_status": "checkpoint" if resumed else (
                        "saved" if store is not None else "disabled"),
                    "checkpoint_identity": identity,
                }
                tables["tuning"].append(row)
                valid_trials.append({
                    "index": index, "config": dict(config), "fit": fitted,
                    "loss": loss, "row": row,
                })
                progress_status = "checkpoint" if resumed else "fitted"
                progress_extra = {"validation_observed_log_loss": loss}
            except Exception as error:
                elapsed = time.monotonic() - candidate_started
                message = f"{type(error).__name__}: {error}"
                tables["tuning"].append({
                    **context, **metadata, **dict(config), "model": name,
                    "index": index, "seed": seed, "validation_loss": None,
                    "selection_metric": "observed_log_loss",
                    "stage": "comparator_candidate", "status": "failed",
                    "converged": False, "iterations": 0,
                    "objective_value": None, "elapsed_seconds": elapsed,
                    "resumed": False, "checkpoint_status": "failed",
                    "error": message,
                })
                tables["failures"].append({
                    **context, **metadata, **dict(config),
                    "stage": "pu_comparator_candidate", "model": name,
                    "candidate_index": index, "error": message,
                })
                progress_status = "failed"
                progress_extra = {"error": message}
            if on_progress is not None:
                on_progress({
                    "event": "candidate_complete", "model": name,
                    "stage": "tuning", "index": index + 1,
                    "total": len(candidates), "config": dict(config),
                    "cache_status": progress_status,
                    "elapsed_seconds": elapsed, **progress_extra,
                })

        if not valid_trials:
            message = "No comparator candidate completed successfully"
            tables["failures"].append({
                **context, **metadata, "stage": "pu_comparator",
                "model": name, "error": message,
            })
            if on_progress is not None:
                on_progress({
                    "event": "model_complete", "model": name,
                    "elapsed_seconds": time.monotonic() - started,
                    "cache_status": "failed", "summary": {"failed": True},
                })
            continue

        selected = min(
            valid_trials,
            key=lambda trial: (
                not bool(trial["fit"].converged), trial["loss"],
                int(trial["config"].get("rank", 0)),
                -float(trial["config"].get(
                    "l2", trial["config"].get(
                        "l2_classifier", trial["config"].get("l2_u", 0.0)))),
                trial["index"],
            ),
        )
        selected_config = dict(selected["config"])
        refit_seed = stable_seed(base_seed, name, "refit")
        refit_cached = checkpoint_available(
            name, "refit", selected_config, refit_seed,
            refit_options, refit_inputs)
        if on_progress is not None:
            on_progress({
                "event": "refit_start", "model": name, "stage": "refit",
                "config": selected_config,
                "cache_status": "checkpoint" if refit_cached else "pending",
                "elapsed_seconds": 0.0,
            })
        refit_started = time.monotonic()
        try:
            final, final_resumed, final_identity = cached_fit(
                name, "refit", selected_config, refit_seed,
                refit_options, refit_inputs,
                lambda: fit_refit(selected_config, refit_seed),
            )
            prediction, model_e = final_probabilities(final)
            refit_elapsed = time.monotonic() - refit_started
        except Exception as error:
            refit_elapsed = time.monotonic() - refit_started
            message = f"{type(error).__name__}: {error}"
            tables["failures"].append({
                **context, **metadata, **selected_config,
                "stage": "pu_comparator_refit", "model": name,
                "error": message,
            })
            if on_progress is not None:
                on_progress({
                    "event": "refit_complete", "model": name,
                    "stage": "refit", "config": selected_config,
                    "cache_status": "failed", "elapsed_seconds": refit_elapsed,
                    "error": message,
                })
                on_progress({
                    "event": "model_complete", "model": name,
                    "elapsed_seconds": time.monotonic() - started,
                    "cache_status": "failed", "summary": {"failed": True},
                })
            continue

        if on_progress is not None:
            on_progress({
                "event": "refit_complete", "model": name, "stage": "refit",
                "config": selected_config,
                "cache_status": "checkpoint" if final_resumed else "fitted",
                "elapsed_seconds": refit_elapsed,
            })
        diagnostics = final.diagnostics()
        record(name, prediction, "reference", metadata,
               estimated_sensitivity=model_e)
        model_elapsed = time.monotonic() - started
        tables["selected"].append({
            **context, **metadata, **selected_config, **diagnostics,
            "model": name, "validation_loss": selected["loss"],
            "selection_metric": "observed_log_loss",
            "tuning_trials": len(candidates),
            "valid_tuning_trials": len(valid_trials),
            "candidate_budget": budget,
            "selected_candidate_index": selected["index"],
            "refit_seed": refit_seed,
            "elapsed_seconds": model_elapsed,
            "refit_elapsed_seconds": refit_elapsed,
            "resumed": final_resumed,
            "final_fit_resumed": final_resumed,
            "checkpoint_status": "checkpoint" if final_resumed else (
                "saved" if store is not None else "disabled"),
            "checkpoint_identity": final_identity,
            "final_fit_attempts": 1, "final_fit_retried": False,
        })
        if on_progress is not None:
            on_progress({
                "event": "model_complete", "model": name,
                "elapsed_seconds": model_elapsed,
                "cache_status": "checkpoint" if final_resumed else "fitted",
                "summary": diagnostics,
            })
