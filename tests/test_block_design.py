"""Scientific invariants of outcome-blind block-completion designs."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.block_design import (
    BlockMaskConfig, make_block_design, make_block_folds, make_block_groups,
)


def test_artificial_groups_are_balanced_and_use_only_stable_cell_ids():
    data = SimpleNamespace(cell_ids=tuple(f"cell-{i}" for i in range(41)), groups={})
    config = BlockMaskConfig(group_mode="artificial", n_artificial_groups=3)
    original = make_block_groups(data, config, seed=14)
    counts = np.unique(original, return_counts=True)[1]
    assert counts.max() - counts.min() == 1
    permutation = np.random.default_rng(9).permutation(len(original))
    reordered = SimpleNamespace(cell_ids=np.asarray(data.cell_ids)[permutation], groups={})
    np.testing.assert_array_equal(make_block_groups(reordered, config, seed=14), original[permutation])
    assert np.any(make_block_groups(data, config, seed=15) != original)


def test_actual_animal_groups_require_explicit_metadata_and_two_animals():
    data = SimpleNamespace(cell_ids=("a", "b", "c", "d"), groups={"animal_id": ["A", "A", "B", "B"]})
    np.testing.assert_array_equal(make_block_groups(data, BlockMaskConfig(), 2), ["A", "A", "B", "B"])
    data.groups = {"animal": ["A"] * 4}
    with pytest.raises(ValueError, match="at least two animals"):
        make_block_groups(data, BlockMaskConfig(), 2)
    data.groups = {"sample": ["A", "A", "B", "B"]}
    with pytest.raises(ValueError, match="explicit animal"):
        make_block_groups(data, BlockMaskConfig(), 2)


def test_recorded_sample_groups_are_explicit_and_distinct_from_animals():
    data = SimpleNamespace(cell_ids=("a", "b", "c", "d"),
                           groups={"sample": ["S1", "S1", "S2", "S2"]})
    config = BlockMaskConfig(group_mode="sample")
    np.testing.assert_array_equal(make_block_groups(data, config, 2),
                                  ["S1", "S1", "S2", "S2"])
    data.groups = {"animal": ["A", "A", "B", "B"]}
    with pytest.raises(ValueError, match="explicit sample"):
        make_block_groups(data, config, 2)


def test_cell_folds_cover_every_cell_once_and_keep_every_group_in_every_role():
    groups = np.repeat(["a", "b", "c"], [31, 35, 39])
    folds = make_block_folds(groups, 3, seed=7)
    repeat = make_block_folds(groups, 3, seed=7)
    visits = np.zeros(len(groups), dtype=int)
    for fold, repeated in zip(folds, repeat):
        fold.validate(len(groups))
        assert fold.metadata["split_type"] == "within_group_new_cell"
        for role in ("train_rows", "validation_rows", "test_rows"):
            rows = getattr(fold, role)
            assert set(groups[rows]) == {"a", "b", "c"}
            np.testing.assert_array_equal(rows, getattr(repeated, role))
        visits[fold.test_rows] += 1
    np.testing.assert_array_equal(visits, 1)
    with pytest.raises(ValueError, match="too few cells"):
        make_block_folds(["a", "a", "b", "b"], 2, seed=7)


def test_complementary_panels_are_nested_balanced_and_preserve_training_support():
    groups = np.repeat(["a", "b"], 12)
    measured = np.ones((24, 10), dtype=bool)
    design = make_block_design(measured, groups, tuple(map(str, range(10))), (0, .2, .4, .6, .8, 1), 42)
    previous = np.zeros_like(measured)
    for fraction, hidden in design.masks.items():
        assert np.all(previous <= hidden)
        visible = measured & ~hidden
        assert visible.any(axis=0).all()
        for group in ("a", "b"):
            mask = hidden[groups == group]
            assert np.array_equal(mask, np.broadcast_to(mask[0], mask.shape))
            assert visible[groups == group].any()
        counts = [hidden[groups == group].any(axis=0).sum() for group in ("a", "b")]
        assert abs(counts[0] - counts[1]) <= 1
        assert hidden.sum() / measured.sum() == pytest.approx(fraction / 2)
        previous = hidden
    # Every target is hidden in precisely one group at full fragmentation.
    assert design.assignment.groupby("target_id").hidden_when_selected.sum().eq(1).all()
    assert design.summary.iloc[-1].realized_pair_fraction == .5
    assert measured.all(), "Source assay availability was modified"


def test_native_off_panel_pairs_and_single_group_targets_are_never_masked():
    groups = np.repeat(["a", "b", "c"], 8)
    measured = np.ones((24, 5), dtype=bool)
    measured[groups != "a", 0] = False  # Target 0 is not eligible.
    measured[groups == "c", 1] = False
    measured[:, 4] = False  # Entirely unassayed target.
    design = make_block_design(measured, groups, tuple("ABCDE"), (.25, .5, 1), 33)
    assert set(design.eligible_targets) == {"B", "C", "D"}
    for hidden in design.masks.values():
        assert not np.any(hidden & ~measured)
        assert not hidden[:, [0, 4]].any()
        assert (measured & ~hidden).any(axis=0)[:4].all()
    assert not design.assignment.loc[~design.assignment.eligible, "hidden_when_selected"].any()
    assert (design.summary.n_eligible_targets == 3).all()


def test_reordering_cells_or_target_columns_preserves_the_named_design():
    groups = np.repeat(["a", "b", "c"], [10, 11, 12])
    measured = np.ones((len(groups), 7), dtype=bool)
    measured[groups == "c", 0] = False
    targets = np.asarray(["t7", "t2", "t9", "t1", "t4", "t8", "t6"])
    original = make_block_design(measured, groups, targets, (.2, .8), 3)
    rng = np.random.default_rng(5)
    rows, columns = rng.permutation(len(groups)), rng.permutation(len(targets))
    other = make_block_design(measured[rows][:, columns], groups[rows], targets[columns], (.2, .8), 3)
    for fraction in (.2, .8):
        np.testing.assert_array_equal(other.masks[fraction], original.masks[fraction][rows][:, columns])


def test_rounded_duplicate_rates_are_reported_and_keep_identical_masks():
    groups = np.repeat(["a", "b"], 8)
    design = make_block_design(np.ones((16, 3)), groups, ("a", "b", "c"), (.1, .2, .4, .8), 42)
    counts = design.summary.set_index("block_fraction").n_selected_targets
    assert counts.to_dict() == {.1: 1, .2: 1, .4: 1, .8: 2}
    duplicate = design.summary.set_index("block_fraction").duplicate_of_fraction
    assert duplicate.loc[.2] == .1 and duplicate.loc[.4] == .1
    assert pd.isna(duplicate.loc[.8])
    np.testing.assert_array_equal(design.masks[.1], design.masks[.4])


def test_infeasible_design_rejected_without_consulting_projection_outcomes():
    with pytest.raises(ValueError, match="retain a measured target"):
        make_block_design(np.ones((6, 1)), ["a"] * 3 + ["b"] * 3, ["single"], [.8], 0)
    native = np.zeros((6, 2), dtype=bool)
    native[:3, 0], native[3:, 1] = True, True
    with pytest.raises(ValueError, match="No target"):
        make_block_design(native, ["a"] * 3 + ["b"] * 3, ["A", "B"], [.8], 0)
    zero = make_block_design(native, ["a"] * 3 + ["b"] * 3, ["A", "B"], [0.], 0)
    assert not zero.masks[0.].any()


def test_sparse_cyclic_overlap_uses_feasible_assignment_instead_of_greedy_failure():
    # A greedy early choice can exhaust a group needed by the last target.
    native = np.asarray([[0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool)
    design = make_block_design(native, ["a", "b", "c"], ["A", "B", "C"], [.4, 1.], 1)
    hidden = design.masks[1.]
    assert np.all(hidden.sum(axis=0) == 1)
    assert np.all((native & ~hidden).sum(axis=1) == 1)
    assert np.all(design.masks[.4] <= hidden)


@pytest.mark.parametrize("overrides", [
    {"fractions": ()}, {"fractions": (.2, .2)}, {"fractions": (float("nan"),)},
    {"fractions": (1.1,)}, {"group_mode": "batch"}, {"n_artificial_groups": 1},
    {"n_artificial_groups": True}, {"validation_fraction": 0.},
])
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        replace(BlockMaskConfig(), **overrides)
