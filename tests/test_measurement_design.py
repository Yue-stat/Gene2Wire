"""Scientific invariants of the combined measurement-degradation masks."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gene2wire.experiments.measurement_design import (
    apply_censoring_pair,
    apply_heterogeneous_censoring,
    apply_target_panel,
    assign_virtual_assays,
    audit_measurement_support,
    draw_cyclic_panel_family,
    make_censoring_design,
    make_observation_plan,
    panel_design_tables,
    row_panel_mask,
)


PRIMARY_COVERAGES = (1.0, 5 / 6, 2 / 3, 1 / 2)


def test_barseq_23_gene_full_endpoint_and_cyclic_golden_counts():
    family = draw_cyclic_panel_family(
        [f"gene-{index}" for index in range(23)],
        (*PRIMARY_COVERAGES, 1 / 3), seed=17,
    )
    expected = {
        1.0: (23, [23, 23, 23], 1.0),
        5 / 6: (19, [15, 15, 15], 15 / 19),
        2 / 3: (15, [7, 7, 8], 22 / 45),
        1 / 2: (12, [4, 4, 5], 13 / 36),
        1 / 3: (8, [0, 0, 1], 1 / 24),
    }
    for coverage, (panel_size, intersections, mean_overlap) in expected.items():
        draw = family.draw(coverage)
        assert draw.panel_size == panel_size
        assert np.all(draw.incidence.sum(axis=1) == panel_size)
        assert draw.union_count == 23 and draw.union_coverage == 1
        upper = np.triu_indices(3, 1)
        assert sorted(draw.pairwise_intersections[upper]) == intersections
        assert draw.mean_pairwise_overlap == pytest.approx(mean_overlap)
    assert family.draw(1.0).incidence.all(), (
        "At coverage one, every BARseq panel must measure all 23 genes"
    )


@pytest.mark.parametrize("n_items", [3, 4, 5, 7, 8, 11, 23, 24, 35])
def test_union_equal_budget_and_nestedness_for_arbitrary_item_counts(n_items):
    family = draw_cyclic_panel_family(
        [f"x{index}" for index in range(n_items)],
        (*PRIMARY_COVERAGES, 1 / 3), seed=44,
    )
    previous = None
    for coverage in sorted(family.draws):
        draw = family.draw(coverage)
        assert draw.union_coverage == 1
        assert np.all(draw.incidence.sum(axis=1) == draw.panel_size)
        assert draw.panel_size >= int(np.ceil(n_items / 3))
        if previous is not None:
            assert np.all(previous.incidence <= draw.incidence)
        previous = draw


def test_named_design_is_seeded_and_invariant_to_input_order():
    item_ids = np.asarray([f"gene-{index:02d}" for index in range(31)])
    original = draw_cyclic_panel_family(item_ids, PRIMARY_COVERAGES, seed=4)
    repeated = draw_cyclic_panel_family(item_ids, PRIMARY_COVERAGES, seed=4)
    permutation = np.random.default_rng(9).permutation(len(item_ids))
    reordered = draw_cyclic_panel_family(item_ids[permutation], PRIMARY_COVERAGES, seed=4)
    changed = draw_cyclic_panel_family(item_ids, PRIMARY_COVERAGES, seed=5)
    for coverage in PRIMARY_COVERAGES:
        for assay in original.assay_ids:
            expected = set(original.draw(coverage).panel_items(assay))
            assert expected == set(repeated.draw(coverage).panel_items(assay))
            assert expected == set(reordered.draw(coverage).panel_items(assay))
    assert any(
        set(original.draw(1 / 2).panel_items(assay))
        != set(changed.draw(1 / 2).panel_items(assay))
        for assay in original.assay_ids
    )


def test_realized_duplicate_is_disclosed_and_not_rejected_for_small_target_sets():
    family = draw_cyclic_panel_family(list("abcde"), PRIMARY_COVERAGES, seed=3)
    high = family.draw(2 / 3)
    low = family.draw(1 / 2)
    assert high.panel_size == low.panel_size == 3
    np.testing.assert_array_equal(high.incidence, low.incidence)
    assert low.duplicate_of_coverage == pytest.approx(2 / 3)
    conditions = panel_design_tables(family)["conditions"].set_index("requested_coverage")
    assert conditions.loc[1 / 2, "duplicate_of_coverage"] == pytest.approx(2 / 3)


def test_overlap_is_intersection_over_k_not_jaccard():
    draw = draw_cyclic_panel_family(
        [f"x{i}" for i in range(24)], [1 / 2], seed=8,
    ).draw(1 / 2)
    upper = np.triu_indices(3, 1)
    np.testing.assert_array_equal(draw.pairwise_intersections[upper], [4, 4, 4])
    np.testing.assert_allclose(draw.pairwise_overlap[upper], 1 / 3)
    assert 1 / 3 != pytest.approx(4 / (12 + 12 - 4))  # Jaccard would be 0.2.


def test_optional_minimum_coverage_reports_integer_residual_overlap():
    divisible = draw_cyclic_panel_family(range(24), [1 / 3], seed=2).draw(1 / 3)
    assert divisible.mean_pairwise_overlap == 0
    indivisible = draw_cyclic_panel_family(range(23), [1 / 3], seed=2).draw(1 / 3)
    assert indivisible.actual_coverage == pytest.approx(8 / 23)
    assert indivisible.mean_pairwise_overlap == pytest.approx(1 / 24)
    assert indivisible.union_coverage == 1


@pytest.mark.parametrize("coverages", [(), (0.2,), (1.1,), (np.nan,), (.5, .5)])
def test_invalid_coverage_grids_fail(coverages):
    with pytest.raises(ValueError):
        draw_cyclic_panel_family(range(12), coverages, seed=1)


def test_invalid_ids_and_too_few_items_fail():
    with pytest.raises(ValueError, match="unique"):
        draw_cyclic_panel_family(["a", "a", "b"], [1], seed=1)
    with pytest.raises(ValueError, match="at least 3"):
        draw_cyclic_panel_family(["a", "b"], [1], seed=1)
    with pytest.raises(TypeError, match="seed"):
        draw_cyclic_panel_family(range(3), [1], seed=True)


def test_virtual_assay_assignment_is_balanced_stratified_and_row_invariant():
    cells = np.asarray([f"cell-{i}" for i in range(41)])
    strata = np.repeat(["animal-1", "animal-2", "animal-3"], [7, 15, 19])
    assignment = assign_virtual_assays(cells, strata, seed=29)
    assert set(assignment) == {"A", "B", "C"}
    for stratum in np.unique(strata):
        counts = np.unique(assignment[strata == stratum], return_counts=True)[1]
        assert counts.max() - counts.min() <= 1
    assert np.ptp(np.unique(assignment, return_counts=True)[1]) <= 1
    permutation = np.random.default_rng(3).permutation(len(cells))
    reordered = assign_virtual_assays(cells[permutation], strata[permutation], seed=29)
    reverse = np.empty_like(permutation)
    reverse[permutation] = np.arange(len(permutation))
    np.testing.assert_array_equal(assignment, reordered[reverse])
    assert np.any(assign_virtual_assays(cells, strata, seed=30) != assignment)


def test_gene_row_mask_and_target_native_intersection_have_distinct_semantics():
    family = draw_cyclic_panel_family(["t1", "t2", "t3", "t4", "t5"], [1, .5], seed=5)
    assays = np.asarray(["A", "B", "C", "A", "B", "C"])
    full = family.draw(1)
    assert row_panel_mask(assays, full).all()
    native = np.ones((len(assays), 5), dtype=bool)
    native[0, 0] = False
    native[2, 4] = False
    np.testing.assert_array_equal(apply_target_panel(native, assays, full), native)
    partial = family.draw(.5)
    effective = apply_target_panel(native, assays, partial)
    assert np.all(effective <= native)
    np.testing.assert_array_equal(effective, native & row_panel_mask(assays, partial))


def test_fold_support_audit_counts_connectivity_and_native_failures():
    family = draw_cyclic_panel_family([f"t{i}" for i in range(12)], [.5, 1 / 3], seed=4)
    assays = np.repeat(["A", "B", "C"], 6)
    native = np.ones((18, 12), dtype=bool)
    primary_w = apply_target_panel(native, assays, family.draw(.5))
    audit = audit_measurement_support(primary_w, assays, family.item_ids)
    np.testing.assert_array_equal(audit.effective_incidence, family.draw(.5).incidence)
    np.testing.assert_array_equal(
        audit.co_measurement_counts,
        audit.effective_incidence.astype(int) @ audit.effective_incidence.astype(int).T,
    )
    assert audit.graph_connected and audit.union_coverage == 1

    extreme_w = apply_target_panel(native, assays, family.draw(1 / 3))
    extreme = audit_measurement_support(extreme_w, assays, family.item_ids)
    assert not extreme.graph_connected and extreme.n_components == 3

    failed = primary_w.copy()
    failed[:, 0] = False
    failed_audit = audit_measurement_support(failed, assays, family.item_ids)
    assert family.item_ids[0] in failed_audit.unsupported_items
    assert failed_audit.union_coverage == pytest.approx(11 / 12)


def _censoring_example(seed=12):
    rng = np.random.default_rng(91)
    n_cells, n_targets = 180, 5
    assays = np.resize(np.asarray(["A", "B", "C"]), n_cells)
    reference = rng.random((n_cells, n_targets)) < np.asarray([.45, .35, .25, .18, .12])
    measured = rng.random(reference.shape) < .9
    plan = make_observation_plan(
        assays, [f"target-{j}" for j in range(n_targets)], seed=seed,
        heterogeneity_delta=1.0,
    )
    return reference, measured, assays, plan


def test_observation_plan_is_immutable_identified_and_owns_h_u():
    reference, _, assays, plan = _censoring_example()
    assert plan.h.shape == (3, reference.shape[1])
    assert plan.u.shape == reference.shape
    assert plan.h.mean() == pytest.approx(0, abs=1e-14)
    assert plan.h.std() == pytest.approx(1, abs=1e-14)
    assert not plan.h.flags.writeable and not plan.u.flags.writeable
    identity = plan.identity()
    json.dumps(identity, sort_keys=True)
    assert len(identity["identity_sha256"]) == 64
    repeated = make_observation_plan(
        assays, plan.target_ids, seed=12, heterogeneity_delta=1.0,
    )
    assert repeated.identity() == identity
    with pytest.raises(FrozenInstanceError):
        plan.row_assays = tuple(reversed(plan.row_assays))


def test_target_specific_fold_local_normalization_and_outside_fold_isolation():
    reference, measured, assays, plan = _censoring_example()
    train = np.arange(100)
    result = plan.generate(
        reference, measured, train, retention=.4, mechanism="heterogeneous")
    train_positive = reference[train] & measured[train]
    assert result.target_normalizers.shape == (reference.shape[1],)
    assert result.target_train_positive_counts.tolist() == train_positive.sum(axis=0).tolist()
    for target in range(reference.shape[1]):
        if train_positive[:, target].any():
            actual = result.sensitivity[train, target][train_positive[:, target]].mean()
            assert actual == pytest.approx(.4, abs=1e-11)
    assert result.expected_train_retention == pytest.approx(.4, abs=1e-11)

    altered = reference.copy()
    altered[100:] = ~altered[100:]
    other = plan.generate(altered, measured, train, retention=.4, mechanism="heterogeneous")
    np.testing.assert_array_equal(result.target_normalizers, other.target_normalizers)
    np.testing.assert_array_equal(result.sensitivity, other.sensitivity)
    np.testing.assert_array_equal(result.observed[train], other.observed[train])


def test_target_without_training_positive_uses_disclosed_pooled_fallback_only():
    reference, measured, _, plan = _censoring_example()
    train = np.arange(90)
    reference[train, -1] = False
    result = plan.generate(
        reference, measured, train, retention=.5, mechanism="heterogeneous")
    assert result.targets_without_train_positives == (plan.target_ids[-1],)
    assert result.target_train_positive_counts[-1] == 0
    assert result.target_normalizers[-1] == pytest.approx(result.normalizer)
    assert result.diagnostics["targets_without_train_positives"] == [plan.target_ids[-1]]


def test_heterogeneous_masks_are_nested_and_never_create_off_panel_detections():
    reference, measured, _, plan = _censoring_example()
    train = np.arange(100)
    previous = np.zeros_like(reference, dtype=bool)
    for retention in (.1, .25, .5, .75, 1.0):
        result = plan.generate(
            reference, measured, train,
            retention=retention, mechanism="heterogeneous")
        assert np.all(previous <= result.observed)
        assert not np.any(result.observed & ~reference)
        assert not np.any(result.observed & ~measured)
        previous = result.observed


def test_matched_uniform_arm_exactly_matches_realized_training_count():
    reference, measured, assays, plan = _censoring_example()
    train = np.arange(110)
    heterogeneous, uniform = plan.generate_pair(
        reference, measured, train, retention=.35)
    assert heterogeneous.train_retained_count == uniform.train_retained_count
    assert uniform.matched_to_train_retained_count == heterogeneous.train_retained_count
    assert np.unique(uniform.sensitivity).size == 1
    assert uniform.mechanism == "matched_uniform"
    assert not np.any(uniform.observed & ~reference)
    assert not np.any(uniform.observed & ~measured)
    direct_h, direct_u = apply_censoring_pair(
        reference, measured, assays, train, .35, plan.design)
    np.testing.assert_array_equal(heterogeneous.observed, direct_h.observed)
    np.testing.assert_array_equal(uniform.observed, direct_u.observed)


def test_zero_positive_fold_and_unknown_mechanism_fail_without_other_role_access():
    reference, measured, _, plan = _censoring_example()
    reference[:30] = False
    with pytest.raises(ValueError, match="no measured reference positives"):
        plan.generate(reference, measured, np.arange(30), retention=.5,
                      mechanism="heterogeneous")
    with pytest.raises(ValueError, match="mechanism"):
        plan.generate(reference, measured, np.arange(30), retention=1,
                      mechanism="mystery")


def test_low_level_design_and_api_validation():
    design = make_censoring_design(6, ["x", "y"], seed=3)
    reference = np.ones((6, 2), dtype=bool)
    measured = np.ones_like(reference)
    assays = np.asarray(["A", "B", "C", "A", "B", "C"])
    result = apply_heterogeneous_censoring(
        reference, measured, assays, [0, 1, 2], .5, design)
    assert result.expected_train_retention == pytest.approx(.5)
    with pytest.raises(ValueError, match="retention"):
        apply_heterogeneous_censoring(reference, measured, assays, [0, 1], 0, design)
