"""Native, incomplete assay panels: evaluated observations and unvalidated forecasts.

The native measurement mask is never artificially changed. Repeated within-group
cell splits test prediction on measured entries and export out-of-fold forecasts
on genuinely unassayed entries without inventing reference labels there.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..seeds import stable_seed
from .block_experiment import _FixedFolds, _MemoizedFeatures
from .contracts import ExperimentDataset
from .evaluation import evaluate_predictions
from .io import atomic_json, atomic_npz
from .pipeline import Artifacts, _atomic_csv, _execute, slug
from .protocol import Settings


def native_panel_tables(dataset: ExperimentDataset) -> dict[str, pd.DataFrame]:
    """Availability comes from the adapter's assay design, independently of calls."""
    dataset.validate()
    if "animal" not in dataset.groups:
        raise ValueError("A native animal-panel experiment requires verified animal IDs")
    animals = np.asarray(dataset.groups["animal"], dtype=str)
    rows = []
    for animal in sorted(set(animals)):
        cells = animals == animal
        for j, target in enumerate(dataset.target_ids):
            support = dataset.measured[cells, j].astype(bool)
            if np.any(support) and not np.all(support):
                raise ValueError(f"Native animal panel is not constant for {animal}/{target}")
            measured = bool(np.all(support))
            rows.append({"animal": animal, "target": target, "measured": measured,
                "n_cells": int(cells.sum()), "n_measured": int(support.sum()),
                "n_positive": int(dataset.reference[cells, j].sum()) if measured else np.nan})
    return {
        "native_panel": pd.DataFrame(rows),
        "native_cells": pd.DataFrame({"cell_id": dataset.cell_ids, "animal": animals}),
    }


def run_native_panel_experiment(dataset: ExperimentDataset, settings: Settings, *,
                               checkpoint_dir, export_dir, progress=True,
                               progress_interval=60., progress_level="summary",
                               worker_status=None, export_cell_forecasts=True) -> Artifacts:
    """Reuse the common estimator/tuner with true panels and repeated cell CV."""
    if settings.supervision_profile != "assay_only":
        raise ValueError("Native unpaired panels require supervision_profile='assay_only'")
    if dataset.natural_observed is not None or dataset.evaluation is not None:
        raise ValueError("This protocol uses single-assay native panels, not paired/block references")
    tables = native_panel_tables(dataset)
    features = _MemoizedFeatures(dataset.feature_builder)
    views, split_rows = [], []
    feature_rows = []
    feature_total = 2 * settings.n_repetitions * settings.n_outer_folds
    feature_done = 0
    feature_started = feature_last = time.monotonic()
    if progress:
        print(f"[features] finished feature sets 0/{feature_total}; "
              "each split fits tuning and final-refit transforms before model scheduling")
    for repetition in range(settings.n_repetitions):
        split_seed = stable_seed(settings.seed, "native_panel_splits", dataset.name, repetition)
        folds = tuple(dataset.split_builder(settings.n_outer_folds, split_seed))
        for fold in folds:
            fold.validate(len(dataset.cell_ids))
            development = np.sort(np.r_[fold.train_rows, fold.validation_rows])
            for fit_role, fit_rows in (("tuning", fold.train_rows),
                                       ("final_refit", development)):
                started = time.monotonic()
                prepared = features(fit_rows, settings.use_location,
                                    settings.use_target_features)
                feature_done += 1
                feature_rows.append({"repetition": repetition,
                    "outer_fold": fold.outer_fold, "feature_fit": fit_role,
                    "n_fit_cells": len(fit_rows),
                    "n_output_features": prepared.X.shape[1],
                    "n_hvg_used": prepared.metadata.get("n_hvg_used"),
                    "elapsed_seconds": time.monotonic() - started})
                now = time.monotonic()
                if progress and (now - feature_last >= progress_interval
                                 or feature_done == feature_total):
                    elapsed_minutes = int((now - feature_started) // 60)
                    timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime(
                        "%Y-%m-%d %H:%M:%S %Z")
                    print(f"[features {elapsed_minutes}min] finished feature sets "
                          f"{feature_done}/{feature_total}, current time {timestamp}")
                    feature_last = now
            for role, indices in (("train", fold.train_rows), ("validation", fold.validation_rows),
                                  ("test", fold.test_rows)):
                split_rows.extend({"repetition": repetition, "outer_fold": fold.outer_fold,
                    "cell_id": dataset.cell_ids[int(i)], "split": role} for i in indices)
        views.append(replace(dataset, feature_builder=features, split_builder=_FixedFolds(folds),
            metadata={**dataset.metadata, "experiment_repetition": repetition,
                "experiment_context": {"experiment": "native_panels"}}))
    tables["native_splits"] = pd.DataFrame(split_rows)
    tables["feature_preparation"] = pd.DataFrame(feature_rows)
    artifacts = _execute(views, settings, checkpoint_dir, export_dir, progress=progress,
        progress_interval=progress_interval, progress_level=progress_level,
        worker_status=worker_status, export_name=slug(dataset.name)+"_native_panels",
        supplementary_tables=tables, manifest_extra={
            "experiment": "native_panels",
            "uncertainty_unit": "descriptive_repeated_within_animal_cell_splits",
            "prediction_task": "new cells within represented animals and known targets",
            "native_mask_policy": "verified injection panel; no artificial target or positive masking",
            "observation_semantics": "single-assay positive-call probability, not identified anatomical connectivity",
            "paired_reference_policy": "none available; none used; no sensitivity estimator fitted",
            "evaluation_policy": "native measured test entries only; naturally unassayed predictions have no truth labels",
            "forecast_policy": "out-of-fold raw assay-score extrapolation; split variability is not a prediction interval",
        })
    return summarize_native_predictions(artifacts, export_cell_forecasts=export_cell_forecasts)


def _metric_means(frame, keys):
    if frame.empty:
        return frame.copy()
    values = [c for c in frame.select_dtypes(include="number")
              if c not in (*keys, "outer_fold", "repetition")]
    per_rep = frame.groupby([*keys, "repetition"], observed=True, dropna=False)[values].mean().reset_index()
    return per_rep.groupby(list(keys), observed=True, dropna=False)[values].mean().reset_index()


def summarize_native_predictions(artifacts: Artifacts, *, export_cell_forecasts=True) -> Artifacts:
    """Recover summaries from cached predictions, including in RESULTS_ONLY mode.

    Forecast matrices contain only W=0 values. CSV rows explicitly carry NA truth
    and evaluation_eligible=False. Raw per-repeat predictions remain in units/.
    """
    if artifacts.manifest.get("experiment") != "native_panels":
        raise ValueError("Expected a native_panels export")
    cells = artifacts.tables.get("native_cells", pd.DataFrame())
    panel = artifacts.tables.get("native_panel", pd.DataFrame())
    if cells.empty or panel.empty or cells.cell_id.duplicated().any():
        raise ValueError("Native cell and panel provenance tables are required")
    cell_ids = cells.cell_id.to_numpy(dtype=str)
    animal_by_cell = cells.animal.to_numpy(dtype=str)
    cell_index = {name: i for i, name in enumerate(cell_ids)}
    # The author-specified adapter order is retained in each prediction file.
    expected_targets = None
    per_animal, per_animal_target = [], []
    accumulated = {}
    visits = set()
    units = sorted((Path(artifacts.export_dir) / "units").glob("*/audit.json"))
    if not units:
        raise ValueError("No completed unit prediction audits were found")
    for audit_path in units:
        audit = json.loads(audit_path.read_text())
        if audit.get("mechanism") != "assay_only":
            raise ValueError("Native panel summaries cannot mix assay-only and artificial-thinning runs")
        repetition, fold = int(audit["repetition"]), int(audit["outer_fold"])
        for path in sorted(audit_path.parent.glob("*_predictions.npz")):
            model = path.name.removesuffix("_predictions.npz")
            with np.load(path, allow_pickle=False) as saved:
                ids, targets = saved["cell_ids"].astype(str), saved["target_ids"].astype(str)
                if expected_targets is None:
                    expected_targets = targets
                if not np.array_equal(targets, expected_targets):
                    raise ValueError("Prediction exports disagree on target order")
                if len(np.unique(ids)) != len(ids):
                    raise ValueError("A prediction unit repeats test cell IDs")
                indices = np.asarray([cell_index[name] for name in ids], dtype=int)
                group = animal_by_cell[indices]
                p = np.asarray(saved["prediction"], dtype=float)
                w, z = saved["measured"].astype(bool), saved["reference"].astype(bool)
                score = saved["ranking_score"].copy() if "ranking_score" in saved else None
            if p.shape != w.shape or z.shape != w.shape or p.shape != (len(ids), len(targets)):
                raise ValueError("Misaligned native prediction arrays")
            if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
                raise ValueError("Native forecasts must be finite probabilities, including off-panel")
            available = panel.pivot(index="animal", columns="target", values="measured")
            expected_w = available.loc[group, targets].to_numpy(dtype=bool)
            if not np.array_equal(w, expected_w):
                raise ValueError("Prediction measurement mask disagrees with verified native panel")
            for name in ids:
                key = (model, repetition, name)
                if key in visits:
                    raise ValueError("A cell appears in multiple test folds within a repetition")
                visits.add(key)
            prefix = {"dataset": audit["dataset"], "model": model,
                      "repetition": repetition, "outer_fold": fold}
            for animal in sorted(set(group)):
                selected = group == animal
                result = evaluate_predictions(z[selected], z[selected], w[selected], p[selected],
                    np.ones_like(p[selected]), probability_semantics="observed", target_ids=targets,
                    ranking_score=None if score is None else score[selected])
                def assay_metric(k):
                    return ("hidden" not in k and "baseline" not in k and "skill" not in k
                            and "detection_" not in k and k != "n_h_undefined")
                summary = {k: v for k, v in result["summary"].items() if assay_metric(k)}
                per_animal.append({**prefix, "animal": animal, **summary})
                per_animal_target.extend({**prefix, "animal": animal,
                    **{k: v for k, v in row.items() if assay_metric(k)}}
                    for row in result["per_target"] if row["n_measured"] > 0)
            if model not in accumulated:
                shape = (len(cell_ids), len(targets))
                accumulated[model] = [np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=np.int32)]
            total, squares, count = accumulated[model]
            forecast = np.where(~w, p, 0.)
            total[indices] += forecast
            squares[indices] += forecast ** 2
            count[indices] += ~w
    if not accumulated:
        raise ValueError("No completed model predictions found")
    artifacts.tables["per_animal_metrics"] = pd.DataFrame(per_animal)
    artifacts.tables["per_animal_aggregate"] = _metric_means(
        artifacts.tables["per_animal_metrics"], ("dataset", "animal", "model"))
    artifacts.tables["per_animal_target"] = pd.DataFrame(per_animal_target)
    artifacts.tables["per_animal_target_aggregate"] = _metric_means(
        artifacts.tables["per_animal_target"], ("dataset", "animal", "target", "model"))
    forecast_dir = Path(artifacts.export_dir) / "native_unassayed"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model, (total, squares, count) in accumulated.items():
        mean = np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)
        variance = np.divide(squares - np.divide(total ** 2, count,
            out=np.zeros_like(total), where=count > 0), count-1,
            out=np.full(total.shape, np.nan), where=count > 1)
        sd = np.sqrt(np.maximum(variance, 0.))
        atomic_npz(forecast_dir / f"{model}_oof_forecasts.npz", cell_ids=cell_ids,
            animals=animal_by_cell, target_ids=expected_targets,
            mean_assay_probability=mean, split_standard_deviation=sd, repeat_count=count,
            reference=np.full(mean.shape, np.nan), evaluation_eligible=np.zeros(mean.shape, bool))
        if export_cell_forecasts:
            i, j = np.nonzero(count > 0)
            frame = pd.DataFrame({"cell_id": cell_ids[i], "animal": animal_by_cell[i],
                "target": expected_targets[j], "model": model,
                "mean_assay_probability": mean[i, j], "split_standard_deviation": sd[i, j],
                "repeat_count": count[i, j], "measured": False,
                "reference": np.nan, "evaluation_eligible": False})
            # Compression is explicit because _atomic_csv's staging suffix is not .gz.
            temporary = forecast_dir / f".{model}_cell_forecasts.csv.gz.tmp"
            frame.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(forecast_dir / f"{model}_cell_forecasts.csv.gz")
        for animal in sorted(set(animal_by_cell)):
            selected = animal_by_cell == animal
            for j, target in enumerate(expected_targets):
                valid = selected & (count[:, j] > 0)
                if valid.any():
                    finite_sd = sd[valid, j]
                    summaries.append({"animal": animal, "target": target, "model": model,
                        "mean_prediction": float(mean[valid, j].mean()), "n_cells": int(valid.sum()),
                        "mean_repeat_sd": float(np.nanmean(finite_sd)) if np.isfinite(finite_sd).any() else np.nan,
                        "min_repeats": int(count[valid, j].min()), "max_repeats": int(count[valid, j].max()),
                        "measured": False, "reference": np.nan, "evaluation_eligible": False})
    artifacts.tables["unassayed_summary"] = pd.DataFrame(summaries)
    summary_tables = ("per_animal_metrics", "per_animal_aggregate", "per_animal_target",
                      "per_animal_target_aggregate", "unassayed_summary")
    for name in summary_tables:
        _atomic_csv(artifacts.tables[name], Path(artifacts.export_dir) / f"{name}.csv")
    atomic_json({"forecast_semantics": "unvalidated assay-positive probability extrapolation",
        "reference_available": False, "evaluation_eligible": False,
        "uncertainty": "sample SD across repeated training splits, not a confidence/prediction interval",
        "model_files": sorted(accumulated), "expected_repetitions": artifacts.manifest.get("protocol", {}).get("n_repetitions"),
        "raw_per_fold_predictions": "../units/", "cell_csv_exported": bool(export_cell_forecasts)},
        forecast_dir / "README.json")
    # These tables are produced after the shared runner writes its manifest.
    # Register them so a direct export reload can reproduce every native-panel plot.
    artifacts.manifest["table_files"] = sorted(set(artifacts.manifest.get("table_files", []))
        | {f"{name}.csv" for name in summary_tables})
    artifacts.manifest["native_predictions_completed"] = True
    atomic_json(artifacts.manifest, Path(artifacts.export_dir) / "manifest.json")
    return artifacts
