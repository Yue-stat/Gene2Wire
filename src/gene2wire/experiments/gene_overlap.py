"""Leak-safe partial gene-panel overlap experiments.

Only cell-gene inputs are hidden. Projection targets, outcomes, assay masks and
outer splits remain unchanged. Missing expression is centered to the fold's
visible training mean and represented by zero after standardization; hidden raw
values are never read while fitting preprocessing statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..seeds import stable_seed
from .contracts import ExperimentDataset, FeatureSet, Fold
from .pipeline import Artifacts, _execute, _atomic_csv
from .io import atomic_json
from .protocol import Settings


PRIMARY_MODELS = ("PU", "PU-MIRT", "PU-Joint")


@dataclass(frozen=True)
class PanelDraw:
    panel_a: tuple[int, ...]
    panel_b: tuple[int, ...]
    common: tuple[int, ...]
    requested_overlap: float

    @property
    def panel_size(self) -> int:
        return len(self.panel_a)

    @property
    def actual_overlap(self) -> float:
        return len(self.common) / self.panel_size


def validate_overlap_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if not grid or len(set(grid)) != len(grid) or any(not np.isfinite(x) or x < 0 or x > 1 for x in grid):
        raise ValueError("OVERLAP_GRID must be a nonempty list of unique values in [0, 1]")
    return grid


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + .5))


def draw_gene_panels(n_genes: int, panel_size: int, overlap_grid: Sequence[float],
                     *, seed: int) -> dict[float, PanelDraw]:
    """Draw nested equal-budget panels without consulting projection labels."""
    grid = validate_overlap_grid(overlap_grid)
    if isinstance(panel_size, bool) or not isinstance(panel_size, (int, np.integer)):
        raise TypeError("panel_size must be an integer")
    if not 1 <= int(panel_size) <= int(n_genes):
        raise ValueError("panel_size must lie between one and the gene count")
    counts = [_round_half_up(int(panel_size) * overlap) for overlap in grid]
    if len(set(counts)) != len(counts):
        raise ValueError("OVERLAP_GRID maps to duplicate overlap-gene counts; use a coarser grid")
    if 2 * int(panel_size) - min(counts) > int(n_genes):
        raise ValueError("The requested panel size/overlap requires more genes than available")
    order = np.random.default_rng(seed).permutation(int(n_genes))
    panel_a_order = order[:int(panel_size)]
    outside_order = order[int(panel_size):]
    output: dict[float, PanelDraw] = {}
    for requested, count in zip(grid, counts):
        panel_a = tuple(sorted(map(int, panel_a_order)))
        panel_b = tuple(sorted(map(int, np.r_[panel_a_order[:count], outside_order[:int(panel_size)-count]])))
        common = tuple(sorted(set(panel_a).intersection(panel_b)))
        output[requested] = PanelDraw(panel_a, panel_b, common, requested)
    return output


def crossed_panel_assignment(strata: Sequence[object], *, seed: int) -> np.ndarray:
    """Balance pseudo-panels within every outcome-independent stratum."""
    values = np.asarray(strata).astype(str)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("strata must be a one-dimensional cell-aligned vector")
    panels = np.empty(len(values), dtype="U1")
    rng = np.random.default_rng(seed)
    for stratum in np.unique(values):
        rows = np.flatnonzero(values == stratum)
        shuffled = rng.permutation(rows)
        panels[shuffled[::2]] = "A"
        panels[shuffled[1::2]] = "B"
    if set(panels) != {"A", "B"}:
        raise ValueError("Panel assignment must contain both A and B")
    return panels


def composite_strata(dataset: ExperimentDataset, names: Sequence[str]) -> np.ndarray:
    if not names or any(name not in dataset.groups for name in names):
        raise ValueError("Every requested crossed-panel stratum must exist in dataset.groups")
    columns = [np.asarray(dataset.groups[name]).astype(str) for name in names]
    return np.asarray(["|".join(items) for items in zip(*columns)], dtype=str)


def aligned_panel_assignment(groups: Sequence[object], mapping: Mapping[object, str]) -> np.ndarray:
    values = np.asarray(groups)
    result = np.asarray([mapping.get(value, mapping.get(str(value), "")) for value in values], dtype="U1")
    if np.any(~np.isin(result, ["A", "B"])) or set(result) != {"A", "B"}:
        raise ValueError("Every aligned group must map to A or B and both panels must occur")
    return result


def paired_group_folds(dataset: ExperimentDataset, group_name: str,
                       pairs: Sequence[Sequence[object]]) -> ExperimentDataset:
    """Use one complete A/B group pair for each train/validation/test role."""
    if group_name not in dataset.groups or len(pairs) != 3:
        raise ValueError("paired_group_folds requires three pairs from a declared group")
    groups = np.asarray(dataset.groups[group_name])
    normalized = tuple(tuple(pair) for pair in pairs)
    flat = [value for pair in normalized for value in pair]
    if any(len(pair) != 2 for pair in normalized) or len(set(map(str, flat))) != 6:
        raise ValueError("pairs must contain six distinct groups")
    if set(map(str, np.unique(groups))) != set(map(str, flat)):
        raise ValueError("pairs must cover every dataset group exactly once")

    def split_builder(n_outer_folds=3, seed=0):
        del seed
        if int(n_outer_folds) != 3:
            raise ValueError("The paired A/B animal design is fixed at three outer folds")
        output = []
        for outer in range(3):
            test_groups = normalized[outer]
            validation_groups = normalized[(outer + 1) % 3]
            train_groups = normalized[(outer + 2) % 3]
            fold = Fold(
                outer, np.flatnonzero(np.isin(groups, train_groups)),
                np.flatnonzero(np.isin(groups, validation_groups)),
                np.flatnonzero(np.isin(groups, test_groups)),
                {"split": "paired_group_train_validation_test",
                 "train_groups": list(map(str, train_groups)),
                 "validation_groups": list(map(str, validation_groups)),
                 "test_groups": list(map(str, test_groups))},
            )
            fold.validate(len(groups))
            output.append(fold)
        return tuple(output)
    result = replace(dataset, split_builder=split_builder)
    result.validate()
    return result


def common_target_dataset(dataset: ExperimentDataset, group_name: str) -> ExperimentDataset:
    """Keep only targets whose measurement mask covers every cell in every group."""
    if group_name not in dataset.groups:
        raise ValueError(f"Unknown target-coverage group {group_name!r}")
    groups = np.asarray(dataset.groups[group_name])
    keep = np.ones(len(dataset.target_ids), dtype=bool)
    for group in np.unique(groups):
        keep &= np.asarray(dataset.measured)[groups == group].all(axis=0)
    if np.sum(keep) < 2:
        raise ValueError("Fewer than two targets have identical assay coverage across all groups")
    indices = np.flatnonzero(keep)
    original_features = dataset.feature_builder

    def features(train_rows, use_location=False, use_target_features=False):
        result = original_features(train_rows, use_location, use_target_features)
        target = None if result.Y_target is None else np.asarray(result.Y_target)[indices]
        return replace(result, Y_target=target)

    metadata = dict(dataset.metadata)
    metadata["common_target_audit"] = {
        "group": group_name, "source_target_count": len(dataset.target_ids),
        "retained_target_count": int(np.sum(keep)),
        "retained_targets": [dataset.target_ids[index] for index in indices],
    }
    result = replace(
        dataset, reference=np.asarray(dataset.reference)[:, indices],
        measured=np.asarray(dataset.measured)[:, indices],
        target_ids=tuple(dataset.target_ids[index] for index in indices),
        natural_observed=(None if dataset.natural_observed is None else
                          np.asarray(dataset.natural_observed)[:, indices]),
        feature_builder=features, metadata=metadata,
    )
    result.validate()
    return result


def _availability(panels: np.ndarray, draw: PanelDraw, n_genes: int) -> np.ndarray:
    available = np.zeros((len(panels), n_genes), dtype=bool)
    available[np.ix_(panels == "A", np.asarray(draw.panel_a, dtype=int))] = True
    available[np.ix_(panels == "B", np.asarray(draw.panel_b, dtype=int))] = True
    return available


def _visible_standardize(raw: np.ndarray, availability: np.ndarray,
                         train_rows: np.ndarray, columns: Sequence[int]) -> np.ndarray:
    columns = tuple(map(int, columns))
    output = np.zeros((len(raw), len(columns)), dtype=float)
    train_mask = np.zeros(len(raw), dtype=bool)
    train_mask[np.asarray(train_rows, dtype=int)] = True
    for out_index, gene_index in enumerate(columns):
        fit = train_mask & availability[:, gene_index]
        if not np.any(fit):
            raise ValueError(f"Gene {gene_index} has no visible outer-training cells")
        fit_values = raw[fit, gene_index]
        center = float(np.mean(fit_values))
        scale = float(np.std(fit_values))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        visible = availability[:, gene_index]
        # Deliberately index only visible values: hidden raw expression never
        # enters preprocessing, even transiently before being overwritten.
        output[visible, out_index] = (raw[visible, gene_index] - center) / scale
    return output


def _panel_feature_builder(dataset: ExperimentDataset, panels: np.ndarray, draw: PanelDraw,
                           arm: str):
    if dataset.gene_matrix is None:
        raise ValueError("The dataset adapter does not expose a gene_matrix")
    raw = np.asarray(dataset.gene_matrix, dtype=float)
    n_genes = raw.shape[1]
    available = _availability(panels, draw, n_genes)

    def builder(train_rows, use_location=False, use_target_features=False):
        if use_location or use_target_features:
            raise ValueError("Gene-overlap primary notebooks require USE_LOCATION=False and USE_TARGET_FEATURES=False")
        train = np.asarray(train_rows, dtype=int)
        if train.ndim != 1 or not len(train) or len(np.unique(train)) != len(train):
            raise ValueError("Training rows must be a nonempty unique integer vector")
        nuisance = (panels == "B").astype(float)[:, None]
        if arm == "union":
            columns = tuple(sorted(set(draw.panel_a).union(draw.panel_b)))
            x_gene = _visible_standardize(raw, available, train, columns)
            names = tuple(dataset.gene_names[index] for index in columns)
            blocks = {"gene_union": tuple(range(len(columns)))}
        elif arm == "intersection":
            columns = draw.common
            if columns:
                x_gene = _visible_standardize(raw, np.ones_like(available), train, columns)
                names = tuple(dataset.gene_names[index] for index in columns)
            else:
                x_gene = np.zeros((len(raw), 1), dtype=float)
                names = ("no_common_gene_intercept_proxy",)
            blocks = {"gene_intersection": tuple(range(x_gene.shape[1]))}
        elif arm == "separate_panel":
            pieces, names_list = [], []
            for label, columns in (("A", draw.panel_a), ("B", draw.panel_b)):
                panel_available = np.zeros_like(available)
                panel_available[np.ix_(panels == label, np.asarray(columns, dtype=int))] = True
                pieces.append(_visible_standardize(raw, panel_available, train, columns))
                names_list.extend(f"panel_{label}:{dataset.gene_names[index]}" for index in columns)
            x_gene = np.column_stack(pieces)
            names = tuple(names_list)
            blocks = {"panel_A_gene": tuple(range(draw.panel_size)),
                      "panel_B_gene": tuple(range(draw.panel_size, 2 * draw.panel_size))}
        elif arm == "all_gene_oracle":
            columns = tuple(range(n_genes))
            x_gene = _visible_standardize(raw, np.ones_like(available), train, columns)
            names = tuple(dataset.gene_names)
            blocks = {"all_gene_oracle": tuple(range(n_genes))}
        else:
            raise ValueError(f"Unknown overlap arm {arm!r}")
        x = np.column_stack((x_gene, nuisance))
        blocks = {**blocks, "panel_nuisance": (x.shape[1] - 1,)}
        return FeatureSet(
            X=x, feature_blocks=blocks, feature_names=names + ("panel_B_nuisance",),
            metadata={"preprocessing": "visible-training-only standardization; hidden values are mean-zero",
                      "panel_A_genes": [dataset.gene_names[i] for i in draw.panel_a],
                      "panel_B_genes": [dataset.gene_names[i] for i in draw.panel_b],
                      "common_genes": [dataset.gene_names[i] for i in draw.common],
                      "arm": arm, "fit_rows": train.tolist()},
        )
    return builder


def overlap_view(dataset: ExperimentDataset, panels: Sequence[str], draw: PanelDraw, *,
                 repetition: int, panel_design: str, arm: str) -> ExperimentDataset:
    panels = np.asarray(panels).astype(str)
    if panels.shape != (len(dataset.cell_ids),) or set(panels) != {"A", "B"}:
        raise ValueError("panels must align to cells and contain A/B")
    allowlist = {"union": PRIMARY_MODELS, "intersection": ("PU",),
                 "separate_panel": ("PU",), "all_gene_oracle": ("PU",)}[arm]
    context = {
        "experiment": "gene_overlap", "panel_design": panel_design, "arm": arm,
        "requested_overlap": float(draw.requested_overlap),
        "actual_overlap": float(draw.actual_overlap), "panel_size": draw.panel_size,
    }
    metadata = dict(dataset.metadata)
    metadata.update({"experiment_context": context, "experiment_repetition": int(repetition),
                     "evaluation_group": "overlap_panel", "model_allowlist": allowlist,
                     "gene_overlap": {"panel_A": list(draw.panel_a), "panel_B": list(draw.panel_b),
                                      "common": list(draw.common)}})
    groups = {**dataset.groups, "overlap_panel": panels}
    result = replace(dataset, feature_builder=_panel_feature_builder(dataset, panels, draw, arm),
                     groups=groups, metadata=metadata)
    result.validate()
    return result


def build_overlap_views(dataset: ExperimentDataset, overlap_grid: Sequence[float], *,
                        panel_size: int | None = None, n_repetitions: int = 5,
                        panel_design: str = "crossed", strata: Sequence[str] = ("slice",),
                        aligned_mapping: Mapping[object, str] | None = None,
                        aligned_group: str = "animal",
                        include_controls: bool = True, seed: int = 20260908) -> tuple[ExperimentDataset, ...]:
    dataset.validate()
    if dataset.gene_matrix is None:
        raise ValueError("Gene-overlap experiments require adapter gene_matrix/gene_names")
    p = np.asarray(dataset.gene_matrix).shape[1]
    k = p // 2 if panel_size is None else int(panel_size)
    grid = validate_overlap_grid(overlap_grid)
    views = []
    repetitions = ((int(dataset.metadata["repetition"]),)
                   if dataset.metadata.get("independent_unit") == "generated_dataset"
                   and "repetition" in dataset.metadata
                   else range(int(n_repetitions)))
    for repetition in repetitions:
        if panel_design == "crossed":
            panels = crossed_panel_assignment(
                composite_strata(dataset, strata),
                seed=stable_seed(seed, dataset.name, panel_design, "assignment", repetition))
        elif panel_design == "animal_aligned":
            if aligned_mapping is None or aligned_group not in dataset.groups:
                raise ValueError("animal_aligned design requires an aligned group mapping")
            panels = aligned_panel_assignment(dataset.groups[aligned_group], aligned_mapping)
        else:
            raise ValueError("panel_design must be crossed or animal_aligned")
        draws = draw_gene_panels(p, k, grid,
            seed=stable_seed(seed, dataset.name, panel_design, "genes", repetition))
        for requested in grid:
            draw = draws[requested]
            views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                      panel_design=panel_design, arm="union"))
            if include_controls:
                views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                          panel_design=panel_design, arm="intersection"))
                views.append(overlap_view(dataset, panels, draw, repetition=repetition,
                                          panel_design=panel_design, arm="separate_panel"))
        if include_controls:
            oracle_draw = PanelDraw(tuple(range(k)), tuple(range(k)), tuple(range(k)), 1.0)
            views.append(overlap_view(dataset, panels, oracle_draw, repetition=repetition,
                                      panel_design=panel_design, arm="all_gene_oracle"))
    return tuple(views)


_OVERLAP_SCENARIO_CONTEXT = (
    "dataset", "sharing_strength", "analysis", "mechanism", "loss_rate",
    "calibration_fraction", "calibration_spec", "experiment", "group_mode",
    "training_panel", "training_block_fraction", "evaluation_scope",
    "block_fraction", "panel_design", "panel_size", "probability_semantics",
)


def _write_overlap_summaries(artifacts: Artifacts) -> None:
    per_group = artifacts.tables.get("per_group", pd.DataFrame())
    if not per_group.empty:
        keys = [column for column in (
            *_OVERLAP_SCENARIO_CONTEXT, "arm", "requested_overlap", "actual_overlap",
            "model", "repetition", "outer_fold",
        ) if column in per_group]
        metrics = [column for column in ("macro_auprc", "macro_log_loss", "macro_brier")
                   if column in per_group]
        worst_rows = []
        for values, frame in per_group.groupby(keys, dropna=False, observed=True):
            prefix = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
            row = dict(prefix)
            for metric in metrics:
                row[f"worst_panel_{metric}"] = (frame[metric].min() if metric == "macro_auprc"
                                                    else frame[metric].max())
            worst_rows.append(row)
        artifacts.tables["worst_panel"] = pd.DataFrame(worst_rows)
        _atomic_csv(artifacts.tables["worst_panel"], artifacts.export_dir / "worst_panel.csv")

    per_rep = artifacts.tables.get("per_repetition", pd.DataFrame())
    required = {"arm", "model", "actual_overlap", "repetition"}
    if per_rep.empty or not required.issubset(per_rep):
        return
    # Keep every fixed scenario coordinate in the matching key.  In particular,
    # simulation artifacts contain several sharing strengths; dropping rho makes
    # a baseline lookup return a Series instead of one scalar and corrupts R_M.
    context = [column for column in (*_OVERLAP_SCENARIO_CONTEXT, "repetition")
               if column in per_rep]
    union = per_rep[per_rep["arm"] == "union"].copy()
    rows = []
    for values, frame in union.groupby(context, dropna=False, observed=True):
        prefix = dict(zip(context, values if isinstance(values, tuple) else (values,)))
        baseline = frame[frame["model"] == "PU"].set_index("actual_overlap")
        endpoints = frame[np.isclose(frame["actual_overlap"], 1.0)].set_index("model")
        if baseline.index.has_duplicates or endpoints.index.has_duplicates:
            raise ValueError(
                "Gene-overlap summaries require one PU baseline and one model endpoint "
                "per scenario/repetition/actual_overlap; include all scenario keys."
            )
        if "PU" not in endpoints.index:
            continue
        for _, candidate in frame[frame["model"].isin(("PU-MIRT", "PU-Joint"))].iterrows():
            overlap = candidate["actual_overlap"]
            if overlap not in baseline.index or candidate["model"] not in endpoints.index:
                continue
            for metric, direction in (("macro_auprc", 1.), ("macro_log_loss", -1.), ("macro_brier", -1.)):
                if metric not in frame:
                    continue
                current = direction * (candidate[metric] - baseline.at[overlap, metric])
                endpoint = direction * (
                    endpoints.at[candidate["model"], metric] - endpoints.at["PU", metric]
                )
                rows.append({**prefix, "model": candidate["model"], "metric": metric,
                             "actual_overlap": overlap, "joint_or_mirt_minus_pu": current,
                             "difference_vs_100pct_overlap": current - endpoint})
    artifacts.tables["overlap_contrasts"] = pd.DataFrame(rows)
    _atomic_csv(artifacts.tables["overlap_contrasts"], artifacts.export_dir / "overlap_contrasts.csv")


def rebuild_overlap_summaries(artifacts: Artifacts) -> Artifacts:
    """Rebuild overlap-specific tables and persist them in an existing export.

    This is intentionally safe for ``RESULTS_ONLY`` notebooks: it only reads
    the saved per-group/per-repetition tables and rewrites derived summaries;
    it never launches model fitting.
    """
    _write_overlap_summaries(artifacts)
    table_files = {str(name) + ".csv" for name in artifacts.tables}
    artifacts.manifest["table_files"] = sorted(table_files)
    atomic_json(artifacts.manifest, artifacts.export_dir / "manifest.json")
    return artifacts


def run_gene_overlap_experiment(views: Sequence[ExperimentDataset], settings: Settings, *,
                                checkpoint_dir: str | Path, export_dir: str | Path,
                                export_name: str, progress: bool = True,
                                progress_interval: float = 60., progress_level: str = "summary",
                                worker_status=None) -> Artifacts:
    """Run overlap views through the shared model/tuning/evaluation pipeline."""
    if settings.run_random_forest:
        raise ValueError("Gene-overlap notebooks intentionally require RUN_RANDOM_FOREST=False")
    if settings.run_information_controls or settings.run_qiao:
        raise ValueError("Overlap controls are explicit views; disable information/Qiao controls")
    artifacts = _execute(
        tuple(views), settings, checkpoint_dir=checkpoint_dir, export_dir=export_dir,
        progress=progress, progress_interval=progress_interval, progress_level=progress_level,
        worker_status=worker_status, export_name=export_name,
        manifest_extra={"experiment": "gene_overlap", "targets_unchanged": True,
                        "missing_expression_representation": "visible-train standardized; hidden=0"},
    )
    return rebuild_overlap_summaries(artifacts)
