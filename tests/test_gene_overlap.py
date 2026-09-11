from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.gene_overlap import (
    build_overlap_views,
    canonical_overlap_arm,
    combine_partition_predictions,
    crossed_panel_assignment,
    draw_gene_panels,
    overlap_view,
    panel_feature_subset,
    panel_fit_spec,
    paired_group_folds,
    partition_rows,
    run_gene_overlap_experiment,
    shared_a_contrasts,
    validate_overlap_artifact,
    validate_overlap_grid,
    validate_partition_fold,
)
from gene2wire.experiments.protocol import Settings


def _simulation():
    return generate_simulation(
        0, .5, seed=3, n_cells=60, n_targets=4, n_gene_features=6,
        n_location_features=0, n_target_features=2, n_slices=6,
    )


def _completed_v2_artifact():
    arm_models = {
        "union": ("PU", "PU-MIRT", "PU-Joint"),
        "intersection": ("PU",),
        "disjoint_coefficient": ("PU",),
        "shared_a": ("PU-MIRT",),
        "separate_a": ("PU-MIRT",),
        "all_gene_oracle": ("PU",),
    }
    rows = []
    contexts = []
    for arm, models in arm_models.items():
        overlaps = (1.0,) if arm == "all_gene_oracle" else (0.0, 1.0)
        for overlap in overlaps:
            contexts.append({
                "experiment": "gene_overlap", "panel_design": "crossed",
                "arm": arm, "requested_overlap": overlap,
            })
            for model in models:
                rows.append({
                    "arm": arm, "model": model,
                    "requested_overlap": overlap, "actual_overlap": overlap,
                    "loss_rate": 0.0,
                })
    return SimpleNamespace(
        manifest={
            "experiment": "gene_overlap", "completed": True,
            "protocol": {"nuisance_l2": 1e-4},
            "datasets": [{"metadata": {"experiment_context": context}}
                         for context in contexts],
        },
        tables={
            "aggregate": pd.DataFrame(rows),
            "overlap_contrasts": pd.DataFrame({"value": [0.0]}),
            "shared_a_contrasts": pd.DataFrame({"value": [0.0]}),
        },
    )


def test_results_only_release_contract_rejects_legacy_or_incomplete_exports():
    artifact = _completed_v2_artifact()
    validate_overlap_artifact(
        artifact, expected_panel_design="crossed",
        expected_overlap_grid=(0.0, 1.0), require_release_v2=True,
    )

    legacy_nuisance = _completed_v2_artifact()
    legacy_nuisance.manifest["protocol"]["nuisance_l2"] = 0.0
    with pytest.raises(ValueError, match="nuisance_l2"):
        validate_overlap_artifact(
            legacy_nuisance, expected_panel_design="crossed",
            expected_overlap_grid=(0.0, 1.0), require_release_v2=True,
        )

    legacy_arms = _completed_v2_artifact()
    legacy_arms.tables["aggregate"] = legacy_arms.tables["aggregate"].loc[
        ~legacy_arms.tables["aggregate"]["arm"].isin(("shared_a", "separate_a"))
    ]
    with pytest.raises(ValueError, match="complete v2 arm/model contract"):
        validate_overlap_artifact(
            legacy_arms, expected_panel_design="crossed",
            expected_overlap_grid=(0.0, 1.0), require_release_v2=True,
        )

    wrong_grid = _completed_v2_artifact()
    with pytest.raises(ValueError, match="does not match"):
        validate_overlap_artifact(
            wrong_grid, expected_panel_design="crossed",
            expected_overlap_grid=(0.0, 0.5, 1.0), require_release_v2=True,
        )

    no_shared_contrast = _completed_v2_artifact()
    no_shared_contrast.tables["shared_a_contrasts"] = pd.DataFrame()
    with pytest.raises(ValueError, match="Shared-A/Separate-A"):
        validate_overlap_artifact(
            no_shared_contrast, expected_panel_design="crossed",
            expected_overlap_grid=(0.0, 1.0), require_release_v2=True,
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


def test_v2_arm_specs_distinguish_coefficient_detector_and_target_sharing():
    assert panel_fit_spec("disjoint_coefficient").fit_mode == "single"
    assert panel_fit_spec("disjoint_coefficient").model_allowlist == ("PU",)
    assert panel_fit_spec("shared_a").fit_mode == "single"
    assert panel_fit_spec("shared_a").model_allowlist == ("PU-MIRT",)
    assert panel_fit_spec("shared_a").lowrank_feature_groups == ()
    assert panel_fit_spec("separate_a").fit_mode == "single"
    assert panel_fit_spec("separate_a").detector_pooling == "pooled"
    assert panel_fit_spec("separate_a").lowrank_feature_groups == (
        "panel_A_gene", "panel_B_gene",
    )
    assert panel_fit_spec("independent_panel").fit_mode == "partitioned"
    assert panel_fit_spec("independent_panel").detector_pooling == "by_partition"
    with pytest.raises(ValueError, match="Unknown overlap arm"):
        canonical_overlap_arm("archived_separate_panel_meaning")


def test_extended_views_have_fixed_rank_cap_and_json_safe_partition_specs():
    base = _simulation()
    views = build_overlap_views(
        base, [0, 1], panel_size=3, n_repetitions=1, strata=("slice",),
        include_controls=True, include_shared_a_ablation=True,
        include_independent_panel_sensitivity=True, seed=10,
    )
    expected = {
        "union", "intersection", "disjoint_coefficient",
        "shared_a", "separate_a", "independent_panel", "all_gene_oracle",
    }
    assert {view.metadata["experiment_context"]["arm"] for view in views} == expected
    assert {view.metadata["tuning_rank_cap"] for view in views} == {3}
    assert {view.metadata["experiment_context"]["rank_cap"] for view in views} == {3}
    for view in views:
        context = view.metadata["experiment_context"]
        spec = view.metadata["panel_fit_spec"]
        assert context["fit_mode"] == spec["fit_mode"]
        assert context["detector_pooling"] == spec["detector_pooling"]
        if context["arm"] == "independent_panel":
            assert view.metadata["fit_partition"] == spec
            assert spec["group"] == "overlap_panel"
            assert spec["feature_blocks"] == {
                "A": "panel_A_gene", "B": "panel_B_gene",
            }
        else:
            assert view.metadata["fit_partition"] is None
        expected_groups = (("panel_A_gene", "panel_B_gene")
                           if context["arm"] == "separate_a" else ())
        assert view.metadata["model_lowrank_feature_groups"] == expected_groups


def test_gene_draws_are_paired_across_crossed_and_aligned_designs():
    base = _simulation()
    animals = np.repeat(("mouse_1", "mouse_2"), len(base.cell_ids) // 2)
    aligned_base = replace(base, groups={**base.groups, "animal": animals})
    crossed = build_overlap_views(
        aligned_base, [0, .5, 1], panel_size=3, n_repetitions=1,
        panel_design="crossed", strata=("slice",), include_controls=False, seed=22,
    )
    aligned = build_overlap_views(
        aligned_base, [0, .5, 1], panel_size=3, n_repetitions=1,
        panel_design="animal_aligned", aligned_group="animal",
        aligned_mapping={"mouse_1": "A", "mouse_2": "B"},
        include_controls=False, seed=22,
    )
    crossed_draws = {
        view.metadata["experiment_context"]["requested_overlap"]: view.metadata["gene_overlap"]
        for view in crossed
    }
    aligned_draws = {
        view.metadata["experiment_context"]["requested_overlap"]: view.metadata["gene_overlap"]
        for view in aligned
    }
    assert crossed_draws == aligned_draws


def test_partitioned_sensitivity_fails_before_unsupported_execution(tmp_path, monkeypatch):
    from gene2wire.experiments import pipeline

    monkeypatch.setattr(pipeline, "SUPPORTS_PARTITIONED_PANEL_FITS", False, raising=False)
    base = _simulation()
    panels = crossed_panel_assignment(base.groups["slice"], seed=8)
    draw = draw_gene_panels(6, 3, [0], seed=9)[0.0]
    view = overlap_view(base, panels, draw, repetition=0,
                        panel_design="crossed", arm="independent_panel")
    settings = Settings(
        n_outer_folds=2, n_repetitions=1, n_jobs=1, loss_rates=(0.0,),
        run_information_controls=False, run_random_forest=False,
        run_mechanism_controls=False, run_calibration_controls=False, run_qiao=False,
    )
    with pytest.raises(RuntimeError, match="partition-aware pipeline executor"):
        run_gene_overlap_experiment(
            (view,), settings, checkpoint_dir=tmp_path / "checkpoints",
            export_dir=tmp_path / "exports", export_name="unsupported", progress=False,
        )


def test_partition_helpers_use_only_panel_gene_block_and_restore_test_order():
    base = _simulation()
    panels = crossed_panel_assignment(base.groups["slice"], seed=8)
    draw = draw_gene_panels(6, 3, [0], seed=9)[0.0]
    view = overlap_view(base, panels, draw, repetition=0,
                        panel_design="crossed", arm="independent_panel")
    fold = view.split_builder(2, 0)[0]
    validate_partition_fold(fold, panels)
    features = view.feature_builder(fold.train_rows, False, False)
    assert features.X.shape == (len(base.cell_ids), 6)
    np.testing.assert_array_equal(features.X_nuisance[:, 0], panels == "B")
    assert features.nuisance_names == ("overlap_panel[B]",)
    spec = panel_fit_spec("independent_panel")
    for label in ("A", "B"):
        subset = panel_feature_subset(features, label, spec)
        assert subset.X.shape == (len(base.cell_ids), 3)
        assert subset.feature_blocks == {"gene_panel": (0, 1, 2)}
        assert "panel_B_nuisance" not in subset.feature_names
        assert subset.metadata["fit_panel"] == label
        assert subset.X_nuisance is None
        assert subset.nuisance_names == ()

    test_parts = partition_rows(fold.test_rows, panels)
    predictions = {
        "A": np.full((len(test_parts["A"]), 2), 0.2),
        "B": np.full((len(test_parts["B"]), 2), 0.8),
    }
    combined = combine_partition_predictions(fold.test_rows, panels, predictions)
    expected = np.where(panels[fold.test_rows, None] == "A", 0.2, 0.8)
    np.testing.assert_array_equal(combined, np.broadcast_to(expected, combined.shape))


def test_shared_a_contrast_is_direction_normalized_and_endpoint_subtracted():
    rows = []
    for overlap in (0.0, 1.0):
        for arm, auprc, loss, brier in (
            ("shared_a", 0.50 + .1 * overlap, 0.30, 0.15),
            ("separate_a", 0.45 + .1 * overlap, 0.34, 0.17),
        ):
            rows.append({
                "dataset": "fixture", "analysis": "primary", "mechanism": "technical_sar",
                "loss_rate": 0.0, "calibration_fraction": 0.2,
                "calibration_spec": "correct", "panel_design": "crossed",
                "panel_size": 3, "probability_semantics": "reference",
                "arm": arm, "model": "PU-MIRT", "actual_overlap": overlap,
                "repetition": 0, "macro_auprc": auprc,
                "macro_log_loss": loss, "macro_brier": brier,
            })
    result = shared_a_contrasts(pd.DataFrame(rows))
    assert len(result) == 6
    assert (result["shared_a_minus_separate_a"] > 0).all()
    endpoint = result[np.isclose(result.actual_overlap, 1.0)]
    np.testing.assert_allclose(endpoint["difference_vs_100pct_overlap"], 0.0)


def test_paired_group_folds_put_a_complete_pair_in_each_role():
    base = _simulation()
    groups = np.repeat(np.arange(1, 7), 10)
    grouped = replace(base, groups={**base.groups, "animal": groups})
    paired = paired_group_folds(grouped, "animal", ((1, 4), (2, 5), (3, 6)))
    folds = paired.split_builder(3, 0)
    for fold in folds:
        assert len(np.unique(groups[fold.train_rows])) == 2
        assert len(np.unique(groups[fold.validation_rows])) == 2
        assert len(np.unique(groups[fold.test_rows])) == 2
