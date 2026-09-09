from collections import Counter
from itertools import product

import numpy as np
import pytest

from gene2wire.candidate_design import select_balanced_candidates
from gene2wire.config import ModelConfig


PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10.)


def joint_grid(ranks=(1, 2, 4, 8, 16, 23), *, target=False):
    return tuple(ModelConfig(name="PU-Joint", kind="joint", rank=rank,
                             shared_l2=shared, residual_l2=residual,
                             use_target_features=target, target_l2=target_l2)
                 for rank, shared, residual, target_l2 in
                 product(ranks, PENALTIES, PENALTIES, PENALTIES if target else (0.,)))


def identity(value):
    return value.rank, value.shared_l2, value.residual_l2, value.target_l2


@pytest.mark.parametrize("ranks", [(1, 2, 4, 8, 11), (1, 2, 4, 8, 16, 23)])
def test_barseq_budget_balances_rank_and_both_penalties(ranks):
    selected = select_balanced_candidates(joint_grid(ranks), 13)
    assert len({identity(value) for value in selected}) == 13
    for name, support in (("rank", ranks), ("shared_l2", PENALTIES),
                          ("residual_l2", PENALTIES)):
        counts = Counter(getattr(value, name) for value in selected)
        assert set(counts) == set(support)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_m1_flattened_linspace_alias_is_removed():
    candidates = joint_grid()
    previous = tuple(candidates[index] for index in np.linspace(0, len(candidates)-1, 13, dtype=int))
    assert sum(value.residual_l2 == 10 for value in previous) == 12
    selected = select_balanced_candidates(candidates, 13)
    assert max(Counter(value.residual_l2 for value in selected).values()) <= 3
    assert len({(value.shared_l2, value.residual_l2) for value in selected}) == 13


def test_target_feature_penalty_is_balanced_too():
    selected = select_balanced_candidates(joint_grid(target=True), 13)
    for name in ("shared_l2", "residual_l2", "target_l2"):
        counts = Counter(getattr(value, name) for value in selected)
        assert set(counts) == set(PENALTIES)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_design_ignores_enumeration_names_and_pu_status():
    grid = joint_grid()
    selected = tuple(map(identity, select_balanced_candidates(grid, 13)))
    renamed = tuple(value.with_updates(name="Joint", pu=False) for value in reversed(grid))
    assert tuple(map(identity, select_balanced_candidates(renamed, 13))) == selected
    assert tuple(map(identity, select_balanced_candidates(grid, 13))) == selected


def test_zero_penalty_is_distinct_and_full_budget_keeps_all():
    grid = tuple(ModelConfig(name="PU", kind="direct", residual_l2=penalty)
                 for penalty in (0., 1e-6, 1e-3, 1.))
    assert {value.residual_l2 for value in select_balanced_candidates(grid, 10)} == {0., 1e-6, 1e-3, 1.}
    subset = select_balanced_candidates(grid, 3)
    assert len(set(map(identity, subset))) == 3
    assert all(value in grid for value in subset)


def test_rejects_invalid_or_mixed_candidates():
    grid = joint_grid()
    for count in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            select_balanced_candidates(grid, count)
    with pytest.raises(ValueError, match="empty grid"):
        select_balanced_candidates((), 1)
    with pytest.raises(ValueError, match="duplicate"):
        select_balanced_candidates((grid[0], grid[0]), 1)
    with pytest.raises(ValueError, match="one structure"):
        select_balanced_candidates((grid[0], ModelConfig(name="PU", kind="direct")), 1)
