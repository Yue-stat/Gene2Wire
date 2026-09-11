"""Outcome-blind group-by-target panels for the block-completion experiment.

These functions accept assay availability, IDs, and a seed. They never inspect
projection outcomes or gene expression. ``BlockDesign.masks`` mark reference
entries withheld from *every* training channel; they do not turn labels into
observed negatives. A fraction is the fraction of eligible targets assigned a
partial panel, not the fraction of positive labels or cell-target pairs hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import ExperimentDataset, Fold


def _fractions(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if (not result or len(set(result)) != len(result)
            or any(not np.isfinite(value) or not 0 <= value <= 1 for value in result)):
        raise ValueError("fractions must be distinct finite probabilities from zero to one")
    return result


def _labels(values, *, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0 or np.any(pd.isna(values)):
        raise ValueError(f"{name} must be a nonempty vector without missing IDs")
    labels = values.astype(str)
    if np.any(labels == ""):
        raise ValueError(f"{name} cannot contain empty IDs")
    return labels


@dataclass(frozen=True)
class BlockMaskConfig:
    fractions: tuple[float, ...] = (.2, .4, .6, .8)
    group_mode: str = "animal"
    n_artificial_groups: int = 2
    validation_fraction: float = .2
    include_full_panel_control: bool = True

    def __post_init__(self):
        _fractions(self.fractions)
        if self.group_mode not in {"animal", "sample", "artificial"}:
            raise ValueError("group_mode must be 'animal', 'sample', or 'artificial'")
        if (isinstance(self.n_artificial_groups, bool)
                or not isinstance(self.n_artificial_groups, (int, np.integer))
                or self.n_artificial_groups < 2):
            raise ValueError("n_artificial_groups must be an integer of at least two")
        if not np.isfinite(self.validation_fraction) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie strictly between zero and one")
        if not isinstance(self.include_full_panel_control, (bool, np.bool_)):
            raise ValueError("include_full_panel_control must be boolean")


@dataclass(frozen=True)
class BlockDesign:
    masks: Mapping[float, np.ndarray]
    assignment: pd.DataFrame
    summary: pd.DataFrame
    eligible_targets: tuple[str, ...]


def make_block_groups(dataset: ExperimentDataset, config: BlockMaskConfig,
                      seed: int) -> np.ndarray:
    """Use supplied animal IDs, or balanced seed-and-cell-ID artificial groups.

    Artificial allocation is invariant to reordering dataset rows. No biological
    animal identity or natural sample metadata are inferred from a cell name.
    """
    ids = _labels(dataset.cell_ids, name="cell_ids")
    if len(set(ids)) != len(ids):
        raise ValueError("cell_ids must be unique")
    if config.group_mode in {"animal", "sample"}:
        aliases = ("animal", "animal_id") if config.group_mode == "animal" else ("sample",)
        key = next((key for key in aliases if key in dataset.groups), None)
        if key is None:
            choices = "animal or animal_id" if config.group_mode == "animal" else "sample"
            raise ValueError(f"{config.group_mode.capitalize()} block masking requires explicit {choices} metadata")
        groups = _labels(dataset.groups[key], name=f"{config.group_mode} groups")
        if groups.shape != ids.shape:
            raise ValueError(f"{config.group_mode.capitalize()} groups must align with cell_ids")
        if len(np.unique(groups)) < 2:
            plural = "animals" if config.group_mode == "animal" else "samples"
            raise ValueError(f"{config.group_mode.capitalize()} block masking requires at least two {plural}")
        return groups.copy()
    if len(ids) < config.n_artificial_groups:
        raise ValueError("There must be at least one cell per artificial group")
    keyed = sorted((hashlib.sha256(f"{int(seed)}\0{cell_id}".encode()).hexdigest(),
                    cell_id, row) for row, cell_id in enumerate(ids))
    groups = np.empty(len(ids), dtype=object)
    for order, (_, _, row) in enumerate(keyed):
        groups[row] = f"artificial_{order % config.n_artificial_groups + 1}"
    return groups.astype(str)


def make_block_folds(groups, n_outer_folds: int, seed: int,
                     validation_fraction: float = .2) -> tuple[Fold, ...]:
    """Partition cells within every group; these are not held-out-animal folds.

    Each cell is test data once. Every group is present in all three roles of
    every fold. Twenty percent of each development group is validation by
    default (rounded upward, with at least one cell left for training).
    """
    groups = _labels(groups, name="groups")
    if (isinstance(n_outer_folds, bool) or not isinstance(n_outer_folds, (int, np.integer))
            or n_outer_folds < 2):
        raise ValueError("n_outer_folds must be an integer of at least two")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_parts = {}
    for group in unique_groups:
        rows = rng.permutation(np.flatnonzero(groups == group))
        if len(rows) < n_outer_folds or len(rows) - int(np.ceil(len(rows) / n_outer_folds)) < 2:
            raise ValueError(f"Group {group!r} has too few cells for every train/validation/test role")
        group_parts[group] = tuple(np.array_split(rows, n_outer_folds))
    folds = []
    for outer_fold in range(n_outer_folds):
        train, validation, test = [], [], []
        for parts in group_parts.values():
            test.append(parts[outer_fold])
            development = rng.permutation(np.concatenate(
                [part for number, part in enumerate(parts) if number != outer_fold]))
            n_validation = min(len(development) - 1,
                               max(1, int(np.ceil(validation_fraction * len(development)))))
            validation.append(development[:n_validation])
            train.append(development[n_validation:])
        fold = Fold(outer_fold=outer_fold, train_rows=np.sort(np.concatenate(train)),
                    validation_rows=np.sort(np.concatenate(validation)),
                    test_rows=np.sort(np.concatenate(test)), metadata={
                        "split_type": "within_group_new_cell", "group_labels": unique_groups.tolist(),
                        "validation_fraction": float(validation_fraction), "seed": int(seed)})
        fold.validate(len(groups))
        folds.append(fold)
    return tuple(folds)


def _selected_count(fraction: float, n_eligible: int) -> int:
    # Round half upward explicitly; tiny positive fractions still exercise one target.
    return 0 if fraction == 0 else min(n_eligible, max(1, int(np.floor(fraction * n_eligible + .5))))


def make_block_design(measured, groups, target_ids: Sequence[str],
                      fractions: Sequence[float], seed: int) -> BlockDesign:
    """Construct nested group-target masks solely from native assay availability.

    A target is eligible only if measured in at least two groups. For each
    selected target, half its measured groups (rounded down) are hidden. Each
    selected target retains at least one observed group, and each group retains
    at least one measured target. Infeasible designs raise rather than adapting
    to reference outcomes. Target selection is a single seeded prefix schedule.
    """
    fractions = _fractions(fractions)
    groups = _labels(groups, name="groups")
    targets = _labels(target_ids, name="target_ids")
    if len(set(targets)) != len(targets):
        raise ValueError("target_ids must be unique")
    native = np.asarray(measured)
    if native.shape != (len(groups), len(targets)) or not np.all(np.isin(native, [0, 1])):
        raise ValueError("measured must be a binary matrix aligned with groups and target_ids")
    native = native.astype(bool, copy=True)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Block masking requires at least two groups")
    native_counts = np.asarray([native[groups == group].sum(axis=0) for group in unique_groups])
    native_blocks = native_counts > 0
    if np.any(native_blocks.sum(axis=1) == 0):
        raise ValueError("Every group must have at least one natively measured target")
    eligible = native_blocks.sum(axis=0) >= 2
    n_eligible = int(eligible.sum())
    if n_eligible == 0 and any(fraction > 0 for fraction in fractions):
        raise ValueError("No target is measured in at least two groups; no block can be hidden")
    rng = np.random.default_rng(seed)
    # Sort IDs before permutation: reordering matrix rows or target columns cannot
    # silently change the experimental design for the same named observations.
    ordered_targets = rng.permutation(np.asarray(
        sorted(np.flatnonzero(eligible), key=lambda i: targets[i]), dtype=int))
    max_selected = max(_selected_count(fraction, n_eligible) for fraction in fractions)
    capacities = native_blocks.sum(axis=1) - 1
    hidden_counts = np.zeros(len(unique_groups), dtype=int)
    hidden_blocks = np.zeros_like(native_blocks)
    priority = rng.permutation(len(unique_groups))

    def assign_one(target: int, seen_targets: set[int], seen_groups: set[int]) -> bool:
        """Augment a capacity-constrained matching without consulting outcomes.

        Direct free-capacity assignments are balanced first. Reassignment is
        needed only for sparse native panels where an early assignment would
        otherwise occupy the only feasible group for a later target.
        """
        if target in seen_targets:
            return False
        seen_targets.add(target)
        number = int(np.flatnonzero(ordered_targets == target)[0])
        rotated = np.roll(priority, -(number % len(priority)))
        positions = {int(group): index for index, group in enumerate(rotated)}
        candidates = sorted((int(group) for group in np.flatnonzero(native_blocks[:, target])
                             if capacities[group] > 0 and not hidden_blocks[group, target]
                             and group not in seen_groups),
                            key=lambda group: (hidden_counts[group] >= capacities[group],
                                               hidden_counts[group], positions[group]))
        for group in candidates:
            if hidden_counts[group] < capacities[group]:
                hidden_blocks[group, target] = True
                hidden_counts[group] += 1
                return True
            seen_groups.add(group)
            for displaced in ordered_targets:
                if hidden_blocks[group, displaced] and assign_one(int(displaced), seen_targets, seen_groups):
                    hidden_blocks[group, displaced] = False
                    hidden_blocks[group, target] = True
                    # The recursive assignment consumed one capacity elsewhere;
                    # replacing this group's old edge leaves its load unchanged.
                    return True
        return False

    for number, target in enumerate(ordered_targets[:max_selected]):
        available = np.flatnonzero(native_blocks[:, target])
        required = len(available) // 2
        for _ in range(required):
            if not assign_one(int(target), set(), set()):
                raise ValueError("Requested block mask cannot retain a measured target in every group; "
                                 "reduce fractions or use groups with more overlapping targets")
    order_lookup = {int(target): number + 1 for number, target in enumerate(ordered_targets)}
    assignment = pd.DataFrame([
        {"group": str(group), "target_id": str(target),
         "native_count": int(native_counts[group_index, target_index]),
         "eligible": bool(eligible[target_index]),
         "selection_order": order_lookup.get(target_index, pd.NA),
         "selected_at_max_fraction": bool(order_lookup.get(target_index, max_selected + 1) <= max_selected),
         "hidden_when_selected": bool(hidden_blocks[group_index, target_index])}
        for group_index, group in enumerate(unique_groups)
        for target_index, target in enumerate(targets)])
    assignment["selection_order"] = assignment["selection_order"].astype("Int64")
    masks, summary, seen_counts = {}, [], {}
    n_native_pairs = int(native.sum())
    n_native_blocks = int(native_blocks.sum())
    for fraction in sorted(fractions):
        n_selected = _selected_count(fraction, n_eligible)
        active = np.zeros(len(targets), dtype=bool)
        active[ordered_targets[:n_selected]] = True
        partial_blocks = hidden_blocks & active[None, :]
        hidden = np.zeros_like(native)
        for group_index, group in enumerate(unique_groups):
            rows = groups == group
            hidden[rows] = native[rows] & partial_blocks[group_index][None, :]
        masks[fraction] = hidden
        n_hidden_pairs = int(hidden.sum())
        n_hidden_blocks = int(partial_blocks.sum())
        summary.append({"block_fraction": fraction, "n_cells": len(groups), "n_groups": len(unique_groups),
                        "n_targets": len(targets), "n_eligible_targets": n_eligible,
                        "n_selected_targets": n_selected,
                        "realized_eligible_target_fraction": n_selected / n_eligible if n_eligible else 0.,
                        "realized_all_target_fraction": n_selected / len(targets),
                        "n_native_pairs": n_native_pairs, "n_hidden_pairs": n_hidden_pairs,
                        "realized_pair_fraction": n_hidden_pairs / n_native_pairs,
                        "n_native_blocks": n_native_blocks, "n_hidden_blocks": n_hidden_blocks,
                        "realized_block_fraction": n_hidden_blocks / n_native_blocks,
                        "duplicate_of_fraction": seen_counts.get(n_selected, np.nan)})
        seen_counts.setdefault(n_selected, fraction)
    return BlockDesign(masks=masks, assignment=assignment, summary=pd.DataFrame(summary),
                       eligible_targets=tuple(str(targets[index]) for index in ordered_targets))
