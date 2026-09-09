"""Outcome-independent designs for bounded Cartesian hyperparameter grids.

Selecting evenly spaced indices of a flattened Cartesian product can alias its
last coordinates.  For example, a 13-point subset of a 6 x 6 x 6 grid can put
12 points at the largest residual penalty.  Balance the coordinates themselves
before using geometric separation to choose among equally balanced candidates.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from .config import ModelConfig


_COORDINATES = ("rank", "shared_l2", "residual_l2", "target_l2")


def _coordinates(config: ModelConfig) -> tuple[float, ...]:
    return tuple(float(getattr(config, name)) for name in _COORDINATES)


def select_balanced_candidates(
    candidates: Sequence[ModelConfig], count: int
) -> tuple[ModelConfig, ...]:
    """Select unique configurations within one structure, without outcomes.

    Greedily minimize the increase in the sum of squared marginal level counts.
    This prioritizes unused/underrepresented levels in *each* active coordinate.
    Among equally balanced candidates, prefer underrepresented pairs of levels,
    then maximize the minimum squared distance from the points already selected.
    Start nearest the geometric center. Pair coverage prevents repeated penalty
    combinations at different ranks from dominating an otherwise balanced design.

    Rank is scaled linearly; positive penalties are scaled in log10 space.  If
    a penalty grid includes zero, zero gets a separate position one decade
    below the smallest positive level.  Each coordinate is then scaled to [0,1].
    Ties use the numeric hyperparameter tuple, making the design independent of
    input enumeration, model name, PU status, random seed, and evaluation data.

    A bounded design does not cover every joint combination or guarantee that
    another model's best candidate is included.  On incomplete Cartesian grids,
    perfect marginal balance need not be feasible.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    values = tuple(candidates)
    if not values:
        raise ValueError("cannot select candidates from an empty grid")
    if len({(value.kind, value.use_target_features) for value in values}) != 1:
        raise ValueError("candidate design requires one structure and feature specification")
    ordered = tuple(sorted(values, key=_coordinates))
    raw = np.asarray([_coordinates(value) for value in ordered], dtype=float)
    if len(np.unique(raw, axis=0)) != len(ordered):
        raise ValueError("candidate grid contains duplicate hyperparameter tuples")
    if count >= len(ordered):
        return ordered

    levels = []
    codes = []
    normalized = []
    for column, name in enumerate(_COORDINATES):
        unique, inverse = np.unique(raw[:, column], return_inverse=True)
        if len(unique) == 1:
            continue
        distances = unique.copy()
        if name != "rank":
            positive = distances > 0
            distances[positive] = np.log10(distances[positive])
            if not np.all(positive):
                distances[~positive] = distances[positive].min() - 1.0
        distances = (distances - distances.min()) / np.ptp(distances)
        levels.append(np.zeros(len(unique), dtype=np.int64))
        codes.append(inverse)
        normalized.append(distances[inverse])

    # Distinct configurations imply at least one active coordinate.
    points = np.column_stack(normalized)
    pairs = []
    for left, right in combinations(range(len(codes)), 2):
        pair_code = codes[left] * len(levels[right]) + codes[right]
        pair_frequency = np.zeros(len(levels[left]) * len(levels[right]), dtype=np.int64)
        pairs.append((pair_frequency, pair_code))
    available = np.ones(len(ordered), dtype=bool)
    nearest = np.full(len(ordered), np.inf)
    selected = []
    for _ in range(count):
        # Adding a point changes sum(n_level**2) by sum(2*n_level + 1).
        imbalance = sum(2 * frequency[code] + 1 for frequency, code in zip(levels, codes))
        eligible = np.flatnonzero(available)
        eligible = eligible[imbalance[eligible] == imbalance[eligible].min()]
        if pairs:
            pair_imbalance = sum(2 * frequency[code] + 1 for frequency, code in pairs)
            eligible = eligible[pair_imbalance[eligible] == pair_imbalance[eligible].min()]
        separation = (nearest[eligible] if selected else
                      -np.sum((points[eligible] - 0.5) ** 2, axis=1))
        best = separation.max()
        # Stable numerical tie handling; ordered indices break any remaining tie.
        chosen = int(eligible[np.flatnonzero(np.isclose(separation, best, rtol=0, atol=1e-12))[0]])
        selected.append(chosen)
        available[chosen] = False
        for frequency, code in zip(levels, codes):
            frequency[code[chosen]] += 1
        for frequency, code in pairs:
            frequency[code[chosen]] += 1
        nearest = np.minimum(nearest, np.sum((points - points[chosen]) ** 2, axis=1))
    return tuple(ordered[index] for index in selected)
