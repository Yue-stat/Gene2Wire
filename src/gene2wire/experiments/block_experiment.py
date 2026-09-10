"""Group-by-target panel masking, using the shared experiment fitting pipeline.

All gene inputs are retained. Panel design depends only on IDs and original
assay availability. Blocked outcomes are removed from every learner-facing
array, including the reference labels available to the paired calibration arm.
Original outcomes are retained in an evaluation-only contract.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..seeds import stable_seed
from .block_design import (BlockMaskConfig, make_block_design, make_block_folds,
                           make_block_groups)
from .contracts import EvaluationSpec, ExperimentDataset
from .io import atomic_json, atomic_npz, file_hash, jsonable
from .pipeline import _execute, slug
from .protocol import Settings, fingerprint, source_hash


class _MemoizedFeatures:
    """Share identical train-fitted features across all panel views of a source."""
    def __init__(self, builder):
        self.builder = builder
        self.cache = {}

    def __call__(self, train_rows, use_location, use_target_features):
        rows = np.asarray(train_rows, dtype=np.int64)
        key = (rows.tobytes(), bool(use_location), bool(use_target_features))
        if key not in self.cache:
            self.cache[key] = self.builder(rows, use_location, use_target_features)
        return self.cache[key]


class _FixedFolds:
    def __init__(self, folds):
        self.folds = tuple(folds)

    def __call__(self, n_outer_folds, seed):
        if len(self.folds) != n_outer_folds:
            raise ValueError("Block folds were built for a different outer-fold count")
        return self.folds


def preview_block_experiment(dataset: ExperimentDataset, settings: Settings,
                             block_config: BlockMaskConfig, repetition: int = 0):
    """Show the exact outcome-blind groups, folds and masks used for a repetition."""
    dataset.validate()
    groups = make_block_groups(dataset, block_config,
        stable_seed(settings.seed, "block_groups", dataset.name, repetition))
    design = make_block_design(dataset.measured, groups, dataset.target_ids,
        block_config.fractions, stable_seed(settings.seed, "block_targets", dataset.name, repetition))
    folds = make_block_folds(groups, settings.n_outer_folds,
        stable_seed(settings.seed, "block_folds", dataset.name), block_config.validation_fraction)
    return {"groups": groups, "folds": folds, "design": design}


def _block_views(dataset, settings, config, repetition, feature_builder=None):
    """Build sanitized fitting views and an outcome-blind design audit.

    This helper has no fitting side effects, which permits direct leakage-boundary
    tests: changing only blocked outcomes must not alter any fitting view.
    """
    preview = preview_block_experiment(dataset, settings, config, repetition)
    groups, folds, design = preview["groups"], preview["folds"], preview["design"]
    features = feature_builder or _MemoizedFeatures(dataset.feature_builder)
    prefix = {"dataset": dataset.name, "repetition": repetition,
              "sharing_strength": dataset.metadata.get("sharing_strength"),
              "group_mode": config.group_mode}
    tables = {
        "block_assignment": design.assignment.assign(**prefix),
        "block_summary": design.summary.assign(**prefix),
        "block_groups": pd.DataFrame({**prefix, "cell_id": dataset.cell_ids,
                                       "block_group": groups}),
        "block_splits": pd.DataFrame([
            {**prefix, "outer_fold": fold.outer_fold, "cell_id": dataset.cell_ids[int(row)],
             "block_group": groups[row], "split": role}
            for fold in folds
            for role, rows in (("train", fold.train_rows), ("validation", fold.validation_rows),
                               ("test", fold.test_rows)) for row in rows]),
    }
    views = []
    panels = [("masked", float(fraction), np.asarray(mask, dtype=bool), {fraction: mask})
              for fraction, mask in design.masks.items()]
    if config.include_full_panel_control:
        panels.append(("full", 0., np.zeros_like(dataset.measured, dtype=bool), design.masks))
    for panel, fraction, hidden, evaluation_masks in panels:
        visible = np.asarray(dataset.measured, dtype=bool) & ~hidden
        reference = np.asarray(dataset.reference, dtype=bool) & visible
        observed = (None if dataset.natural_observed is None else
                    np.asarray(dataset.natural_observed, dtype=bool) & visible)
        evaluation = EvaluationSpec(reference=dataset.reference, masks=evaluation_masks,
                                    groups=groups, source_measured=dataset.measured)
        context = {"experiment": "target_block", "training_panel": panel,
                   "training_block_fraction": fraction, "group_mode": config.group_mode}
        view = replace(dataset, reference=reference, measured=visible,
            natural_observed=observed, feature_builder=features, split_builder=_FixedFolds(folds),
            groups={**dataset.groups, "block_group": groups}, evaluation=evaluation,
            metadata={**dataset.metadata, "experiment_repetition": repetition,
                      "experiment_context": context})
        view.validate()
        views.append(view)
    return views, tables


def _combine_tables(collection):
    return {key: pd.concat([tables[key] for tables in collection], ignore_index=True)
            for key in collection[0]}


def _manifest(config):
    return {
        "experiment": "target_block",
        "block_protocol": asdict(config),
        "prediction_task": "new cells within represented groups, on structurally hidden target blocks",
        "observation_design": "Natural detections if paired assay; otherwise optional SCAR at configured loss rates",
        "scoring": "raw model probability against original assay reference on blocked test entries; no D=0 conditioning",
        "full_panel_control": "one fit per source/repetition/fold/scenario; evaluated on every same blocked test mask",
        "paired_policy": "same paired cell IDs within each split; all blocked entries excluded in every supervision arm",
        "simulation_extra_controls": "mechanism/calibration-control suites are separate primary experiments; this protocol uses configured loss rates and paired_fraction",
    }


def run_block_experiment(dataset: ExperimentDataset, settings: Settings,
                         block_config: BlockMaskConfig, *, checkpoint_dir, export_dir,
                         progress=True, progress_interval=60., progress_level="summary",
                         worker_status=None):
    """Fit repeated panel designs for a real dataset; full controls fit once."""
    if dataset.metadata.get("independent_unit") == "generated_dataset":
        raise ValueError("Use run_block_simulation_experiments to generate independent repetitions")
    views, tables = [], []
    features = _MemoizedFeatures(dataset.feature_builder)
    for repetition in range(settings.n_repetitions):
        current, audit = _block_views(dataset, settings, block_config, repetition, features)
        views.extend(current)
        tables.append(audit)
    return _execute(views, settings, checkpoint_dir, export_dir, progress=progress,
                    progress_interval=progress_interval, progress_level=progress_level,
                    worker_status=worker_status, export_name=slug(dataset.name)+"_target_block",
                    supplementary_tables=_combine_tables(tables), manifest_extra=_manifest(block_config))


def _cached_block_simulation(raw_cache_dir, repetition, rho, seed, options):
    """Restore an integrity-checked, pickle-free generated source before fitting.

    A lock protects the NPZ/JSON pair from concurrent notebook writers. Each file
    is atomically replaced, with JSON published last. An interrupted incomplete
    pair or a checksum mismatch is reported rather than silently regenerated.
    """
    from .datasets.simulation import (SimulationFeatures, SimulationSplits,
                                      generate_simulation)
    identity = {"seed": seed, "repetition": repetition, "rho": rho,
                "options": options, "source_hash": source_hash()}
    raw_id = fingerprint(identity)
    raw_path = Path(raw_cache_dir) / f"simulation_{raw_id}.npz"
    manifest_path = raw_path.with_suffix(".json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if raw_path.exists() or manifest_path.exists():
            if not raw_path.is_file() or not manifest_path.is_file():
                raise ValueError(f"Incomplete simulation raw cache: {raw_path}; preserve or remove the pair explicitly")
            try:
                manifest = json.loads(manifest_path.read_text())
            except (ValueError, OSError) as error:
                raise ValueError(f"Cannot read simulation raw-cache manifest: {manifest_path}") from error
            if (manifest.get("schema") != "block-simulation-raw-v1"
                    or manifest.get("identity") != jsonable(identity)):
                raise ValueError(f"Simulation raw-cache identity mismatch: {manifest_path}")
            if file_hash(raw_path) != manifest.get("sha256"):
                raise ValueError(f"Simulation raw-cache checksum mismatch: {raw_path}; cached data were not overwritten")
            try:
                with np.load(raw_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                metadata = {**manifest["metadata"],
                            **{key: arrays[array_key] for key, array_key in manifest["metadata_arrays"].items()}}
                features = SimulationFeatures(arrays["genes"], arrays["location"],
                                              arrays["target_descriptors"])
                splits = SimulationSplits(arrays["split_slices"], arrays["split_position"],
                                          **manifest["split_parameters"])
                dataset = ExperimentDataset(name=manifest["name"],
                    reference=arrays["reference"], measured=arrays["measured"],
                    cell_ids=tuple(arrays["cell_ids"].astype(str)),
                    target_ids=tuple(arrays["target_ids"].astype(str)),
                    feature_builder=features, split_builder=splits,
                    groups={key: arrays[array_key] for key, array_key in manifest["groups"].items()},
                    technical_score=arrays["technical_score"], metadata=metadata)
                dataset.validate()
                if (features.genes.shape[0] != len(dataset.cell_ids)
                        or features.location.shape[0] != len(dataset.cell_ids)
                        or features.target_descriptors.shape[0] != len(dataset.target_ids)):
                    raise ValueError("Cached simulation features do not align with outcomes")
            except (ValueError, KeyError, OSError) as error:
                raise ValueError(f"Invalid simulation raw cache: {raw_path}") from error
            return dataset
        dataset = generate_simulation(repetition, rho, seed=seed, **options)
        features, splits = dataset.feature_builder, dataset.split_builder
        arrays = {"reference": dataset.reference, "measured": dataset.measured,
                  "cell_ids": np.asarray(dataset.cell_ids, dtype=str),
                  "target_ids": np.asarray(dataset.target_ids, dtype=str),
                  "genes": features.genes, "location": features.location,
                  "target_descriptors": features.target_descriptors,
                  "technical_score": dataset.technical_score,
                  "split_slices": splits.slices, "split_position": splits.position}
        metadata_arrays = {}
        for key, value in dataset.metadata.items():
            if isinstance(value, np.ndarray):
                array_key = f"metadata_{len(metadata_arrays)}"
                arrays[array_key] = value
                metadata_arrays[key] = array_key
        groups = {}
        for key, value in dataset.groups.items():
            array_key = f"group_{len(groups)}"
            arrays[array_key] = np.asarray(value)
            groups[key] = array_key
        if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
            raise ValueError("Simulation raw caches require numeric or string arrays, never pickled objects")
        atomic_npz(raw_path, **arrays)
        atomic_json({"schema": "block-simulation-raw-v1", "identity": identity,
                     "name": dataset.name, "sha256": file_hash(raw_path),
                     "metadata": {key: value for key, value in dataset.metadata.items()
                                  if not isinstance(value, np.ndarray)},
                     "metadata_arrays": metadata_arrays, "groups": groups,
                     "split_parameters": {"repetition": splits.repetition,
                                          "inner_validation_slices": splits.inner_validation_slices}},
                    manifest_path)
        return dataset


def run_block_simulation_experiments(settings: Settings, block_config: BlockMaskConfig,
                                     *, raw_cache_dir, checkpoint_dir, export_dir,
                                     sharing_strengths=(0., .5, 1.), simulation_options=None,
                                     progress=True, progress_interval=60., progress_level="summary",
                                     worker_status=None):
    """Generate independent datasets and apply the same shared block protocol."""
    if block_config.group_mode != "artificial":
        raise ValueError("Simulation has no biological animals; use group_mode='artificial'")
    options = dict(simulation_options or {})
    if "truth_uses_location" in options and bool(options["truth_uses_location"]) != settings.use_location:
        raise ValueError("Simulation truth_uses_location must match USE_LOCATION")
    options["truth_uses_location"] = settings.use_location
    raw_cache_dir = Path(raw_cache_dir)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    views, tables = [], []
    if not sharing_strengths or len(set(sharing_strengths)) != len(sharing_strengths):
        raise ValueError("sharing_strengths must be nonempty and distinct")
    for rho in sharing_strengths:
        for repetition in range(settings.n_repetitions):
            dataset = _cached_block_simulation(raw_cache_dir, repetition, rho,
                                                settings.seed, options)
            current, audit = _block_views(dataset, settings, block_config, repetition)
            views.extend(current)
            tables.append(audit)
    return _execute(views, settings, checkpoint_dir, export_dir, progress=progress,
                    progress_interval=progress_interval, progress_level=progress_level,
                    worker_status=worker_status, export_name="simulation_target_block",
                    supplementary_tables=_combine_tables(tables), manifest_extra=_manifest(block_config))
