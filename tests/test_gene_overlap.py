from dataclasses import replace

import numpy as np

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.gene_overlap import (
    build_overlap_views,
    crossed_panel_assignment,
    draw_gene_panels,
    overlap_view,
    validate_overlap_grid,
)


def _simulation():
    return generate_simulation(
        0, .5, seed=3, n_cells=60, n_targets=4, n_gene_features=6,
        n_location_features=0, n_target_features=2, n_slices=6,
    )


def test_panel_draw_has_equal_budget_and_barseq_rounding():
    draws = draw_gene_panels(23, 11, [0, .25, .5, .75, 1], seed=4)
    assert [len(draw.common) for draw in draws.values()] == [0, 3, 6, 8, 11]
    for draw in draws.values():
        assert len(draw.panel_a) == len(draw.panel_b) == 11
        assert len(set(draw.panel_a).intersection(draw.panel_b)) == len(draw.common)


def test_grid_rejects_duplicate_rounded_gene_counts():
    validate_overlap_grid([0, .1])
    try:
        draw_gene_panels(6, 3, [0, .1], seed=1)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate realized overlaps must be rejected")


def test_hidden_raw_values_do_not_affect_features():
    base = _simulation()
    panels = crossed_panel_assignment(base.groups["slice"], seed=8)
    draw = draw_gene_panels(6, 3, [0], seed=9)[0.0]
    view = overlap_view(base, panels, draw, repetition=0,
                        panel_design="crossed", arm="union")
    fold = view.split_builder(2, 0)[0]
    expected = view.feature_builder(fold.train_rows, False, False).X
    available = np.zeros(base.gene_matrix.shape, dtype=bool)
    available[np.ix_(panels == "A", draw.panel_a)] = True
    available[np.ix_(panels == "B", draw.panel_b)] = True
    poisoned_raw = base.gene_matrix.copy()
    poisoned_raw[~available] = 1e100
    poisoned = replace(base, gene_matrix=poisoned_raw)
    actual = overlap_view(poisoned, panels, draw, repetition=0,
                          panel_design="crossed", arm="union").feature_builder(
                              fold.train_rows, False, False).X
    np.testing.assert_allclose(actual, expected)


def test_views_keep_targets_outcomes_and_outer_splits_unchanged():
    base = _simulation()
    views = build_overlap_views(base, [0, 1], panel_size=3, n_repetitions=1,
                                strata=("slice",), include_controls=True, seed=10)
    for view in views:
        np.testing.assert_array_equal(view.reference, base.reference)
        np.testing.assert_array_equal(view.measured, base.measured)
        assert view.target_ids == base.target_ids
        original = base.split_builder(2, 10)
        changed = view.split_builder(2, 10)
        for left, right in zip(original, changed):
            np.testing.assert_array_equal(left.test_rows, right.test_rows)

