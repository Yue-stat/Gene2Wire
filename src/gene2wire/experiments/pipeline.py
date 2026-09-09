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

from ..checkpoint import AtomicCheckpointStore, sha256_array, sha256_file
from ..data import DatasetBundle
from ..runner import run_model_grid
from ..seeds import stable_seed
from .calibration import (CalibrationNotEstimable, detection_diagnostics,
                          fit_detection_calibrator, sample_paired_rows)
from .contracts import ExperimentDataset, FeatureSet, Fold
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
        if settings.use_target_features and features.Y_target is None:
            raise ValueError("USE_TARGET_FEATURES=True requires aligned target covariates")
        if not settings.use_target_features and features.Y_target is not None:
            raise ValueError("Disabled target features must not enter the learner")
    # Do not serialize the raw gene-count matrix into every worker.
    metadata = {k: v for k, v in dataset.metadata.items() if not isinstance(v, np.ndarray)}
    return PreparedFold(dataset.name, dataset.reference, dataset.measured,
                        dataset.cell_ids, dataset.target_ids, dataset.groups,
                        dataset.natural_observed, dataset.platform, dataset.technical_score,
                        train_features, refit_features, fold, metadata)


def _bundle(prepared, features, observed):
    return DatasetBundle(X_cell=features.X, S_observed=observed,
                         W_measured=prepared.measured, Y_target=features.Y_target,
                         cell_ids=prepared.cell_ids, target_ids=prepared.target_ids,
                         groups=prepared.groups, feature_blocks=features.feature_blocks)


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
        observed, _reference_view(prepared.reference, paired), prepared.measured, paired,
        technical_score=technical_score if use_technical else None,
        platform=prepared.platform if scenario["mechanism"] == "natural" else None,
        use_technical=use_technical,
        use_target_effects=(scenario["mechanism"] != "scar"
                            and scenario["calibration_spec"] != "pooled_target"))
    estimated = fitted.predict(n_cells=len(prepared.cell_ids),
                               technical_score=technical_score if use_technical else None,
                               platform=prepared.platform if scenario["mechanism"] == "natural" else None)
    return estimated, fitted.to_dict()


def _compile(prepared, features, observed, estimated, rows, paired, mode):
    base = _bundle(prepared, features, observed)
    compiled, exposure = compile_training_bundle(
        base, rows, paired, _reference_view(prepared.reference, paired), estimated, mode)
    return compiled.subset_rows(rows), exposure[rows]


def _run_fold(prepared, repetition, settings, checkpoint_dir, export_dir, code_hash,
              on_progress=None):
    fold = prepared.fold
    train, validation, test = fold.train_rows, fold.validation_rows, fold.test_rows
    development = np.sort(np.r_[train, validation])
    n, nt = prepared.reference.shape
    rho = prepared.metadata.get("sharing_strength")
    is_simulation = prepared.metadata.get("independent_unit") == "generated_dataset"
    draw_seed = stable_seed(settings.seed, "observation", prepared.name, repetition)
    design = make_observation_design(n, nt, seed=draw_seed,
                                     technical_score=prepared.technical_score)
    tables = {name: [] for name in ("metrics", "per_target", "reliability", "tuning",
                                   "selected", "detection", "detection_per_target",
                                   "detection_reliability", "thinning_audit", "failures")}
    groups = next((prepared.groups[k] for k in ("animal", "sample", "animal_id", "sample_id")
                   if k in prepared.groups), None)
    paired_seed = stable_seed(settings.seed, "paired", prepared.name, repetition, fold.outer_fold)
    source_context = {"dataset": prepared.name, "repetition": repetition,
                      "outer_fold": fold.outer_fold, "sharing_strength": rho}
    model_checkpoint = Path(checkpoint_dir) / slug(prepared.name)

    for scenario in scenarios(settings, natural=prepared.natural_observed is not None,
                              simulation=is_simulation, sharing_strength=rho):
        context = {**source_context, **scenario}
        def emit(event):
            if on_progress is not None:
                on_progress({**context, **event})
        emit({"event": "scenario_start"})
        paired = sample_paired_rows(development, scenario["calibration_fraction"], paired_seed,
                                    groups=None if groups is None else groups[development])
        paired_train = np.intersect1d(paired, train)
        if scenario["mechanism"] == "natural":
            observed, true_e, gamma = prepared.natural_observed, None, None
        else:
            generated = thin_reference(prepared.reference, prepared.measured, train,
                                       scenario["loss_rate"], scenario["mechanism"], design)
            observed, true_e, gamma = generated.observed, generated.sensitivity, generated.gamma
        try:
            tuning_e, tuning_detector = _detector(prepared, observed, paired_train, scenario, design.technical_score)
            # Separate refit objects never enter the validation score. Their paired
            # validation references are authorized only for development fitting.
            final_e, final_detector = _detector(prepared, observed, paired, scenario, design.technical_score)
        except CalibrationNotEstimable as error:
            tables["failures"].append({**context, "stage": "calibration", "error": str(error)})
            emit({"event": "calibration_failed", "error": str(error)})
            continue
        unit_id = fingerprint(context)
        audit_dir = Path(export_dir) / "units" / unit_id
        audit_dir.mkdir(parents=True, exist_ok=True)
        atomic_json({**context, "generator_gamma": gamma,
                     "train_cell_ids": np.asarray(prepared.cell_ids)[train],
                     "validation_cell_ids": np.asarray(prepared.cell_ids)[validation],
                     "test_cell_ids": np.asarray(prepared.cell_ids)[test],
                     "paired_train_cell_ids": np.asarray(prepared.cell_ids)[paired_train],
                     "paired_development_cell_ids": np.asarray(prepared.cell_ids)[paired],
                     "tuning_detector": tuning_detector, "refit_detector": final_detector,
                     "feature_names_tuning": prepared.train_features.feature_names,
                     "feature_names_refit": prepared.refit_features.feature_names,
                     "features_tuning": prepared.train_features.metadata,
                     "features_refit": prepared.refit_features.metadata,
                     "fold": fold.metadata}, audit_dir / "audit.json")
        diagnostic = detection_diagnostics(observed, prepared.reference, prepared.measured,
                                            final_e, rows=test, true_sensitivity=true_e)
        tables["detection"].append({**context, **diagnostic})
        detection_evaluation = evaluate_detection_calibration(
            prepared.reference[test], observed[test], prepared.measured[test], final_e[test],
            true_sensitivity=None if true_e is None else true_e[test], target_ids=prepared.target_ids)
        tables["detection_per_target"].extend({**context, **row} for row in detection_evaluation["per_target"])
        tables["detection_reliability"].extend({**context, **row} for row in detection_evaluation["reliability"])
        atomic_npz(audit_dir / "observation.npz", reference=prepared.reference[test],
                   observed=observed[test], measured=prepared.measured[test],
                   technical_score=design.technical_score[test],
                   estimated_sensitivity=final_e[test], target_offsets=design.target_offsets,
                   **({} if true_e is None else {"true_sensitivity": true_e[test]}))
        for split_name, rows in (("train", train), ("validation", validation), ("test", test)):
            for target_index, target in enumerate(prepared.target_ids):
                measured_count = int(prepared.measured[rows, target_index].sum())
                positive_count = int((prepared.reference[rows, target_index] * prepared.measured[rows, target_index]).sum())
                detected_count = int(observed[rows, target_index].sum())
                tables["thinning_audit"].append({**context, "split": split_name, "target": target,
                    "measured_count": measured_count, "reference_positive_count": positive_count,
                    "detected_positive_count": detected_count,
                    "realized_positive_loss": 1-detected_count/positive_count if positive_count else np.nan})
        prior = _train_prevalence(prepared.reference, prepared.measured, paired)

        def record(name, prediction, semantics, extra=None, *, ranking_score=None):
            evaluated = evaluate_predictions(prepared.reference[test], observed[test],
                prepared.measured[test], prediction, final_e[test],
                probability_semantics=semantics, train_reference_prevalence=prior,
                target_ids=prepared.target_ids, ranking_score=ranking_score)
            prefix = {**context, "model": name, "probability_semantics": semantics,
                      **(extra or {})}
            tables["metrics"].append({**prefix, **evaluated["summary"]})
            tables["per_target"].extend({**prefix, **row} for row in evaluated["per_target"])
            tables["reliability"].extend({**prefix, **row} for row in evaluated["reliability"])
            scores = {key: value for key, value in evaluated["scores"].items()
                      if value is not None and key != "prediction"}
            atomic_npz(audit_dir / f"{slug(name)}_predictions.npz", prediction=prediction,
                       reference=prepared.reference[test], observed=observed[test],
                       measured=prepared.measured[test], estimated_sensitivity=final_e[test],
                       cell_ids=np.asarray(prepared.cell_ids)[test].astype(str),
                       target_ids=np.asarray(prepared.target_ids, dtype=str), **scores)

        safe_validation = _bundle(prepared, prepared.train_features, observed).subset_rows(validation)
        train_bundle, train_e = _compile(prepared, prepared.train_features, observed, tuning_e,
                                         train, paired_train, "calibrated_pu")
        refit_bundle, refit_e = _compile(prepared, prepared.refit_features, observed, final_e,
                                         development, paired, "calibrated_pu")
        tuning = settings.tuning_config(min(prepared.train_features.X.shape[1],
                                           prepared.refit_features.X.shape[1]), nt)
        models = settings.models()
        if scenario["analysis"].startswith("calibration"):
            models = tuple(m for m in models if m.pu)
        runner_context = {k: v for k, v in context.items() if k != "analysis"}

        def core_run(chosen_models, tb, te, rb, re, role):
            return run_model_grid(train=tb, validation=safe_validation,
                test_X=prepared.train_features.X[test], refit_test_X=prepared.refit_features.X[test],
                models=chosen_models, tuning=tuning, fit=settings.fit_config(),
                train_exposure=te, validation_exposure=tuning_e[validation],
                test_exposure=final_e[test], refit=rb, refit_exposure=re,
                test_cell_ids=np.asarray(prepared.cell_ids)[test],
                checkpoint_dir=model_checkpoint, unit_context={**runner_context, "supervision": role},
                seed=stable_seed(settings.seed, prepared.name, repetition, fold.outer_fold),
                code_version=code_hash, on_progress=emit)

        result = core_run(models, train_bundle, train_e, refit_bundle, refit_e, "calibrated_pu")
        for name, model_result in result.models.items():
            semantics = "reference" if model_result.fitted.config.pu else "observed"
            record(name, model_result.latent_probability, semantics)
            tables["selected"].append({**context, **model_result.summary()})
            tables["tuning"].extend({**context, "model": name, **asdict(trial.config),
                                     "stage": trial.stage, "index": trial.index,
                                     "validation_loss": trial.validation_loss,
                                     "converged": trial.converged, "iterations": trial.iterations,
                                     "seed": trial.seed} for trial in model_result.tuning.trials)
        endpoint = scenario["analysis"] == "primary" and scenario["loss_rate"] in (None, .8)
        reference_bundles = None
        if endpoint and (settings.run_information_controls or settings.run_random_forest):
            clean_train, clean_train_e = _compile(prepared, prepared.train_features, observed, tuning_e,
                                                  train, paired_train, "reference_only")
            clean_refit, clean_refit_e = _compile(prepared, prepared.refit_features, observed, final_e,
                                                  development, paired, "reference_only")
            reference_bundles = (clean_train, clean_train_e, clean_refit, clean_refit_e)
        if endpoint and settings.run_information_controls:
            for role, name in (("reference_only", "Reference-only"), ("reference_plus_pu", "Reference+PU")):
                if role == "reference_only":
                    tb, te, rb, re = reference_bundles
                else:
                    tb, te = _compile(prepared, prepared.train_features, observed, tuning_e,
                                      train, paired_train, role)
                    rb, re = _compile(prepared, prepared.refit_features, observed, final_e,
                                      development, paired, role)
                base = next(m for m in settings.models() if m.name == "PU").with_updates(name=name)
                control = core_run((base,), tb, te, rb, re, role).models[name]
                record(name, control.latent_probability, "reference")
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
        baseline_endpoint = scenario["analysis"] == "primary" and scenario["loss_rate"] in (None, 0., .8)
        if baseline_endpoint:
            from .baselines import fit_baseline
            if reference_bundles is None:
                clean_train, clean_train_e = _compile(prepared, prepared.train_features, observed, tuning_e,
                                                      train, paired_train, "reference_only")
                clean_refit, clean_refit_e = _compile(prepared, prepared.refit_features, observed, final_e,
                                                      development, paired, "reference_only")
                reference_bundles = (clean_train, clean_train_e, clean_refit, clean_refit_e)
            for kind in ("prevalence", "random_forest"):
                if kind == "random_forest" and not settings.run_random_forest:
                    continue
                for semantics in ("observed", "reference"):
                    tb, rb = ((train_bundle, refit_bundle) if semantics == "observed"
                              else (reference_bundles[0], reference_bundles[2]))
                    name = ("RF" if kind == "random_forest" else "Prevalence") + ("-reference" if semantics == "reference" else "-observed")
                    baseline_started = time.monotonic()
                    emit({"event": "model_start", "model": name})
                    baseline = fit_baseline(tb.X_cell, tb.S_observed, tb.W_measured,
                        safe_validation.X_cell, safe_validation.S_observed, safe_validation.W_measured,
                        tuning_e[validation], prepared.refit_features.X[test], final_e[test],
                        kind=kind, probability_semantics=semantics,
                        candidate_budget=settings.candidate_budget,
                        seed=stable_seed(settings.seed, repetition, fold.outer_fold, kind),
                        checkpoint_dir=model_checkpoint / "baselines" / fingerprint({**runner_context, "kind": kind, "semantics": semantics}),
                        refit_X=rb.X_cell, refit_labels=rb.S_observed, refit_measured=rb.W_measured,
                        on_progress=lambda event: emit({**event, "model": name}))
                    emit({"event": "model_complete", "model": name,
                          "elapsed_seconds": time.monotonic()-baseline_started,
                          "cache_status": "checkpoint" if baseline.diagnostics.get("final_fit_resumed") else "fitted",
                          "summary": dict(baseline.selected_config)})
                    record(name, baseline.prediction, semantics)
                    tables["selected"].append({**context, "model": name, **baseline.selected_config,
                                                "tuning_trials": len(baseline.candidate_records)})
                    tables["tuning"].extend({**context, "model": name, **row} for row in baseline.candidate_records)
        if settings.run_qiao and scenario["analysis"] == "primary":
            _run_qiao_controls(prepared, observed, tuning_e, final_e, settings, context,
                               record, tables, model_checkpoint, on_progress=emit)
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
                  "calibration_fraction", "calibration_spec", "model", "probability_semantics"]


def _summarize(tables, *, simulation):
    metrics = tables["metrics"]
    if metrics.empty:
        return
    excluded = set(_GROUP_COLUMNS + ["outer_fold", "repetition"])
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


def _planned_models(prepared, settings):
    """Model evaluations per fold/repetition, before any outcome is examined."""
    rows = []
    simulation = prepared.metadata.get("independent_unit") == "generated_dataset"
    for scenario in scenarios(settings, natural=prepared.natural_observed is not None,
                              simulation=simulation,
                              sharing_strength=prepared.metadata.get("sharing_strength")):
        names = [m.name for m in settings.models()
                 if m.pu or not scenario["analysis"].startswith("calibration")]
        if scenario["analysis"] == "primary":
            if scenario["loss_rate"] in (None, .8) and settings.run_information_controls:
                names += ["Reference-only", "Reference+PU"]
            if scenario["loss_rate"] in (None, 0., .8):
                names += ["Prevalence-observed", "Prevalence-reference"]
                if settings.run_random_forest:
                    names += ["RF-observed", "RF-reference"]
            if settings.run_qiao:
                from .qiao import qiao_model_names
                names += list(qiao_model_names(settings.use_target_features))
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
                           export_path, fit_version, run_id, key, writer):
    started = time.monotonic()
    writer({"event": "unit_start"})
    tables = _run_fold(prepared, repetition, settings, checkpoint_dir, export_path,
                       fit_version, on_progress=writer)
    if not tables["failures"]:
        prefix = {"dataset": prepared.name, "repetition": repetition,
                  "outer_fold": prepared.fold.outer_fold,
                  "sharing_strength": prepared.metadata.get("sharing_strength")}
        files = []
        for scenario in scenarios(settings, natural=prepared.natural_observed is not None,
                                  simulation=prepared.metadata.get("independent_unit") == "generated_dataset",
                                  sharing_strength=prepared.metadata.get("sharing_strength")):
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


def _execute(datasets, settings, checkpoint_dir, export_dir, *, progress=True,
             progress_interval=60., progress_level="summary"):
    code_hash = source_hash()
    contexts = []
    metadata = []
    input_identities = []
    is_simulation = all(d.metadata.get("independent_unit") == "generated_dataset" for d in datasets)
    for dataset in datasets:
        dataset.validate()
        folds = dataset.split_builder(settings.n_outer_folds, settings.seed)
        visits = np.zeros(len(dataset.cell_ids), int)
        for fold in folds:
            visits[fold.test_rows] += 1
        if len(folds) != settings.n_outer_folds or not np.all(visits == 1):
            raise ValueError("Requested outer CV must test every cell exactly once")
        reps = ([int(dataset.metadata["repetition"])] if is_simulation
                else range(settings.n_repetitions))
        fold_inputs = []
        for fold in folds:
            prepared = _prepare(dataset, fold, settings)
            contexts.extend((prepared, repetition) for repetition in reps)
            fold_inputs.append({"fold": fold.outer_fold,
                "train_rows": sha256_array(np.asarray(fold.train_rows)),
                "validation_rows": sha256_array(np.asarray(fold.validation_rows)),
                "test_rows": sha256_array(np.asarray(fold.test_rows)),
                "train_X": sha256_array(prepared.train_features.X),
                "refit_X": sha256_array(prepared.refit_features.X),
                "train_Y": None if prepared.train_features.Y_target is None else sha256_array(prepared.train_features.Y_target),
                "refit_Y": None if prepared.refit_features.Y_target is None else sha256_array(prepared.refit_features.Y_target)})
        input_identities.append({"name": dataset.name,
            "repetition": dataset.metadata.get("repetition"),
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
                         "metadata": {k: v for k, v in dataset.metadata.items() if not isinstance(v, np.ndarray)}})
    manifest = {"protocol": settings.scientific_dict(), "source_hash": code_hash,
                "datasets": metadata, "input_identities": input_identities,
                "uncertainty_unit": "generated_dataset" if is_simulation else "descriptive_fold_and_mask_repeat",
                "software": {"python": platform.python_version(), **{package: importlib.metadata.version(package)
                             for package in ("numpy", "scipy", "pandas", "scikit-learn", "joblib")}}
                }
    run_id = fingerprint({"protocol": settings.scientific_dict(), "source_hash": code_hash,
                          "input_identities": input_identities, "software": manifest["software"]})
    fit_version = code_hash + ":" + fingerprint(manifest["software"])
    export_path = Path(export_dir) / ("simulation_0908" if is_simulation else slug(datasets[0].name) + "_0908") / run_id
    export_path.mkdir(parents=True, exist_ok=True)
    manifest.update(run_id=run_id, requested_n_jobs=settings.n_jobs, completed=False,
                    checkpoint_dir=str(Path(checkpoint_dir).resolve()))
    atomic_json(manifest, export_path / "manifest.json")
    # One parallel level only: workers receive prepared dense arrays, not raw
    # high-dimensional sequencing objects; joblib can memory-map large arrays.
    workers = min(settings.n_jobs, len(contexts))
    store = AtomicCheckpointStore(Path(checkpoint_dir) / "experiment_units" / run_id)
    results, pending, inventory = [None] * len(contexts), [], []
    for index, (prepared, repetition) in enumerate(contexts):
        unit_context = {"dataset": prepared.name, "repetition": repetition,
                        "outer_fold": prepared.fold.outer_fold,
                        "sharing_strength": prepared.metadata.get("sharing_strength")}
        key = fingerprint({"run_id": run_id, **unit_context})
        cached = _load_unit_result(store, key, run_id, export_path)
        inventory.append({**unit_context, "work_id": key, "fully_cached": cached is not None,
                          "planned_model_evaluations": len(_planned_models(prepared, settings))})
        if cached is not None:
            results[index] = cached
        else:
            pending.append((index, prepared, repetition, key, unit_context))
    planned_model_count = sum(row["planned_model_evaluations"] for row in inventory)
    cached_model_count = sum(row["planned_model_evaluations"] for row in inventory
                             if row["fully_cached"])
    event_dir = export_path / "progress" / uuid.uuid4().hex
    relay = ProgressRelay(event_dir, enabled=progress, interval=progress_interval,
                          level=progress_level, total_units=planned_model_count,
                          cached_units=cached_model_count)
    with relay:
        tasks = [(index, p, r, key, relay.writer(key, **context, work_id=key))
                 for index, p, r, key, context in pending]
        if workers == 1:
            new_results = [_run_checkpointed_fold(p, r, settings, checkpoint_dir, export_path,
                           fit_version, run_id, key, writer) for _, p, r, key, writer in tasks]
        else:
            with parallel_config(backend="loky", inner_max_num_threads=1):
                new_results = Parallel(n_jobs=workers, verbose=0, max_nbytes="10M")(
                    delayed(_run_checkpointed_fold)(p, r, settings, checkpoint_dir, export_path,
                        fit_version, run_id, key, writer) for _, p, r, key, writer in tasks)
        for task, result in zip(tasks, new_results):
            results[task[0]] = result
    names = results[0].keys()
    tables = {name: pd.DataFrame([row for result in results for row in result[name]]) for name in names}
    tables["checkpoint_inventory"] = pd.DataFrame(inventory)
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
    for name, table in tables.items():
        _atomic_csv(table, export_path / f"{name}.csv")
    manifest.update(completed=True, status="complete_with_failures" if len(tables["failures"]) else "complete",
                    failed_calibration_units=len(tables["failures"]),
                    planned_model_evaluations=planned_model_count,
                    completed_model_evaluations=relay.done_units,
                    cached_model_evaluations=cached_model_count,
                    metric_rows=len(tables["metrics"]), table_files=[f"{name}.csv" for name in tables])
    atomic_json(manifest, export_path / "manifest.json")
    print(f"All results exported to: {export_path}")
    if len(tables["failures"]):
        print(f"Calibration not estimable in {len(tables['failures'])} units; see failures.csv. Do not omit these from reporting.")
    return Artifacts(tables, export_path, manifest)


def run_experiment(dataset: ExperimentDataset, settings: Settings, *, checkpoint_dir, export_dir,
                   progress=True, progress_interval=60., progress_level="summary"):
    return _execute([dataset], settings, checkpoint_dir, export_dir, progress=progress,
                    progress_interval=progress_interval, progress_level=progress_level)


def run_simulation_experiments(settings: Settings, *, raw_cache_dir, checkpoint_dir, export_dir,
                               sharing_strengths=(0., .5, 1.), simulation_options=None,
                               progress=True, progress_interval=60., progress_level="summary"):
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
                    progress_interval=progress_interval, progress_level=progress_level)


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
    if not prepared.measured[validation].any():
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
    validation_mask = np.asarray(prepared.measured[validation], dtype=bool)
    validation_d = np.asarray(observed[validation], dtype=float)[validation_mask]
    if not np.all(np.isin(validation_d, [0, 1])):
        raise ValueError("Qiao validation detections must be binary on measured entries")

    def cached_fit(phase, features, rows, prediction_rows, config):
        x, target = features.X[rows], features.Y_target
        w = np.asarray(prepared.measured[rows], dtype=bool)
        d = np.where(w, observed[rows], 0.0)
        prediction_x = features.X[prediction_rows]
        seed = stable_seed(settings.seed, "qiao", prepared.name,
                           context.get("repetition"), context.get("outer_fold"),
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
        prediction, ranking_score, diagnostics = cached_fit("final", refit_features,
                                                           dev, test, best[1])
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
