"""Scientific invariants for the v2 gene-panel overlap extensions."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.gene_overlap import (
    build_overlap_views,
    combine_partition_predictions,
    panel_feature_subset,
    panel_fit_spec,
    partition_rows,
)
from gene2wire.experiments.gene_overlap_plotting import plot_overlap_results
from gene2wire.experiments.gene_overlap_sensitivity import (
    TECH_SAR_80_EXPORT_NAME,
    TECH_SAR_80_LABEL,
    prepare_tech_sar80_sensitivity,
)
from gene2wire.experiments.protocol import Settings


OVERLAPS = (0.0, 0.5, 1.0)


def _simulation(*, repetition=0, sharing=0.5):
    return generate_simulation(
        repetition,
        sharing,
        seed=20260910,
        n_cells=72,
        n_targets=12,
        n_gene_features=8,
        n_location_features=0,
        n_target_features=2,
        n_slices=6,
    )


def _views(**changes):
    options = dict(
        panel_size=4,
        n_repetitions=1,
        panel_design="crossed",
        strata=("slice",),
        include_controls=True,
        include_shared_a_ablation=True,
        include_independent_panel_sensitivity=False,
        seed=19,
    )
    options.update(changes)
    return build_overlap_views(_simulation(), OVERLAPS, **options)


def _by_arm(views, arm):
    return [view for view in views
            if view.metadata["experiment_context"]["arm"] == arm]


def test_arm_contracts_separate_coefficient_model_and_detector_pooling():
    expected = {
        "union": ("single", "pooled", ("PU", "PU-MIRT", "PU-Joint")),
        "intersection": ("single", "pooled", ("PU",)),
        "disjoint_coefficient": ("single", "pooled", ("PU",)),
        "shared_a": ("single", "pooled", ("PU-MIRT",)),
        "separate_a": ("single", "pooled", ("PU-MIRT",)),
        "independent_panel": ("partitioned", "by_partition", ("PU",)),
        "all_gene_oracle": ("single", "pooled", ("PU",)),
    }
    for arm, (mode, detector, models) in expected.items():
        spec = panel_fit_spec(arm)
        assert (spec.fit_mode, spec.detector_pooling, spec.model_allowlist) == (
            mode, detector, models)
        payload = spec.to_dict()
        assert payload["combine_test_predictions"] is (mode == "partitioned")
        if mode == "partitioned":
            assert payload["group"] == "overlap_panel"
            assert payload["labels"] == ["A", "B"]
            assert set(payload["feature_blocks"]) == {"A", "B"}
    assert panel_fit_spec("shared_a").lowrank_feature_groups == ()
    assert panel_fit_spec("separate_a").lowrank_feature_groups == (
        "panel_A_gene", "panel_B_gene")


def test_constant_gene_budget_and_rank_support_across_overlap():
    views = _views()
    settings = Settings(
        n_repetitions=1,
        loss_rates=(0.0,),
        run_information_controls=False,
        run_random_forest=False,
        run_qiao=False,
        run_mechanism_controls=False,
        run_calibration_controls=False,
    )
    rank_grids = set()
    for view in views:
        context = view.metadata["experiment_context"]
        if context["arm"] == "all_gene_oracle":
            continue
        assert context["panel_size"] == 4
        assert context["rank_cap"] == 4
        assert view.metadata["tuning_rank_cap"] == 4
        assert len(view.metadata["gene_overlap"]["panel_A"]) == 4
        assert len(view.metadata["gene_overlap"]["panel_B"]) == 4
        # The shared pipeline must derive candidates from the declared assay
        # budget, not from the overlap-dependent union column count.
        rank_grids.add(settings.tuning_config(
            view.metadata["tuning_rank_cap"], len(view.target_ids)).ranks)
    assert rank_grids == {(0, 1, 2, 4)}
    union_dimensions = set()
    for view in _by_arm(views, "union"):
        fold = view.split_builder(3, 0)[0]
        union_dimensions.add(view.feature_builder(fold.train_rows, False, False).X.shape[1])
    assert len(union_dimensions) > 1  # The invariant is nontrivial.


def test_pipeline_consumes_rank_cap_instead_of_overlap_dependent_union_width(
        tmp_path, monkeypatch):
    """Metadata alone is insufficient: the actual tuner must receive fixed ranks."""
    from gene2wire.experiments import pipeline

    selected_views = [view for view in _by_arm(_views(), "union")
                      if view.metadata["experiment_context"]["requested_overlap"] in (0.0, 1.0)]
    settings = Settings(
        n_outer_folds=3,
        n_repetitions=1,
        n_jobs=1,
        loss_rates=(0.0,),
        run_information_controls=False,
        run_random_forest=False,
        run_qiao=False,
        run_mechanism_controls=False,
        run_calibration_controls=False,
    )
    observed_rank_grids = []

    def fake_core(**kwargs):
        observed_rank_grids.append(kwargs["tuning"].ranks)
        n_test = len(kwargs["test_X"])
        n_targets = kwargs["train"].n_targets
        models = {}
        for config in kwargs["models"]:
            models[config.name] = SimpleNamespace(
                fitted=SimpleNamespace(config=config),
                latent_probability=np.full((n_test, n_targets), 0.25),
                summary=lambda config=config: {
                    "model": config.name,
                    "kind": config.kind,
                    "rank": config.rank,
                    "nuisance_l2": config.nuisance_l2,
                },
                tuning=SimpleNamespace(trials=()),
            )
        return SimpleNamespace(models=models)

    monkeypatch.setattr(pipeline, "run_model_grid", fake_core)
    for index, view in enumerate(selected_views):
        fold = view.split_builder(settings.n_outer_folds, settings.seed)[0]
        prepared = pipeline._prepare(view, fold, settings)
        pipeline._run_fold(
            prepared,
            0,
            settings,
            tmp_path / "checkpoints",
            tmp_path / f"export_{index}",
            "test-source",
        )
    assert observed_rank_grids
    assert set(observed_rank_grids) == {(0, 1, 2, 4)}


def test_panel_nuisance_is_not_a_gene_and_is_model_identical():
    views = _views(include_controls=False, include_shared_a_ablation=False)
    schemas = set()
    for view in views:
        fold = view.split_builder(3, 0)[0]
        features = view.feature_builder(fold.train_rows, False, False)
        assert features.X_nuisance is not None
        assert features.X_nuisance.shape == (len(view.cell_ids), 1)
        assert features.nuisance_names == ("overlap_panel[B]",)
        assert not any("nuisance" in name.lower() for name in features.feature_names)
        assert not any("nuisance" in name.lower() for name in features.feature_blocks)
        assert features.X.shape[1] == len(features.feature_names)
        panels = np.asarray(view.groups["overlap_panel"])
        np.testing.assert_array_equal(features.X_nuisance[:, 0], panels == "B")
        schemas.add((features.nuisance_names, features.X_nuisance.shape[1]))
    assert schemas == {(('overlap_panel[B]',), 1)}
    nuisance_l2 = 1e-4
    primary = [model for model in Settings(nuisance_l2=nuisance_l2).models()
               if model.name in ("PU", "PU-MIRT", "PU-Joint")]
    assert len(primary) == 3
    assert {model.nuisance_l2 for model in primary} == {nuisance_l2}


def test_shared_and_separate_a_use_matched_panel_features_and_pooled_detector():
    views = _views()
    for overlap in OVERLAPS:
        shared = next(view for view in _by_arm(views, "shared_a")
                      if np.isclose(view.metadata["experiment_context"]["requested_overlap"], overlap))
        separate = next(view for view in _by_arm(views, "separate_a")
                        if np.isclose(view.metadata["experiment_context"]["requested_overlap"], overlap))
        fold = shared.split_builder(3, 0)[0]
        shared_features = shared.feature_builder(fold.train_rows, False, False)
        separate_features = separate.feature_builder(fold.train_rows, False, False)
        np.testing.assert_array_equal(shared_features.X, separate_features.X)
        assert shared_features.feature_blocks == separate_features.feature_blocks
        np.testing.assert_array_equal(
            shared_features.X_nuisance, separate_features.X_nuisance)
        assert shared_features.nuisance_names == separate_features.nuisance_names
        assert panel_fit_spec("shared_a").detector_pooling == "pooled"
        assert panel_fit_spec("separate_a").detector_pooling == "pooled"
        # The two arms differ only in whether the target loading is shared
        # between the two declared, K-column gene blocks.
        assert panel_fit_spec("shared_a").lowrank_feature_groups == ()
        groups = panel_fit_spec("separate_a").lowrank_feature_groups
        assert groups == ("panel_A_gene", "panel_B_gene")
        assert all(len(separate_features.feature_blocks[name]) == 4 for name in groups)


def test_pipeline_does_not_reduce_separate_a_to_a_metadata_alias(tmp_path, monkeypatch):
    """The ablation must change the fitted factorization, not just table labels."""
    from gene2wire.experiments import pipeline

    views = _views()
    chosen = [
        next(view for view in _by_arm(views, arm)
             if np.isclose(view.metadata["experiment_context"]["requested_overlap"], 0.5))
        for arm in ("shared_a", "separate_a")
    ]
    settings = Settings(
        n_outer_folds=3,
        n_repetitions=1,
        n_jobs=1,
        loss_rates=(0.0,),
        run_information_controls=False,
        run_random_forest=False,
        run_qiao=False,
        run_mechanism_controls=False,
        run_calibration_controls=False,
    )
    calls = []

    def fake_core(**kwargs):
        # A grouped factorization may be represented in ModelConfig or as an
        # explicit runner argument. Context/arm labels are deliberately excluded.
        structure = {
            "models": tuple(asdict(model) for model in kwargs["models"]),
            "lowrank_feature_groups": kwargs.get("lowrank_feature_groups"),
        }
        calls.append((
            structure,
            kwargs["train"].X_cell.copy(),
            dict(kwargs["train"].feature_blocks),
            None if kwargs["train"].X_nuisance is None
            else kwargs["train"].X_nuisance.copy(),
            kwargs["train"].nuisance_names,
        ))
        config = kwargs["models"][0]
        n_test, n_targets = len(kwargs["test_X"]), kwargs["train"].n_targets
        fitted = SimpleNamespace(config=config)
        result = SimpleNamespace(
            fitted=fitted,
            latent_probability=np.full((n_test, n_targets), 0.25),
            summary=lambda: {"model": config.name, "kind": config.kind,
                             "rank": config.rank,
                             "nuisance_l2": config.nuisance_l2},
            tuning=SimpleNamespace(trials=()),
        )
        return SimpleNamespace(models={config.name: result})

    monkeypatch.setattr(pipeline, "run_model_grid", fake_core)
    for index, view in enumerate(chosen):
        fold = view.split_builder(3, settings.seed)[0]
        prepared = pipeline._prepare(view, fold, settings)
        pipeline._run_fold(
            prepared, 0, settings, tmp_path / "checkpoints",
            tmp_path / f"ablation_{index}", "test-source")
    assert len(calls) == 2
    # The input and feature partition are exactly matched.
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    assert calls[0][2] == calls[1][2]
    np.testing.assert_array_equal(calls[0][3], calls[1][3])
    assert calls[0][4] == calls[1][4]
    # But the runner-visible statistical structure must differ.
    assert calls[0][0] != calls[1][0]
    shared_config = calls[0][0]["models"][0]
    separate_config = calls[1][0]["models"][0]
    assert tuple(shared_config["lowrank_feature_groups"]) == ()
    assert tuple(separate_config["lowrank_feature_groups"]) == (
        "panel_A_gene", "panel_B_gene")


def test_independent_panel_component_features_are_exactly_k_and_have_own_intercept():
    view = _by_arm(_views(include_independent_panel_sensitivity=True),
                   "independent_panel")[0]
    fold = view.split_builder(3, 0)[0]
    features = view.feature_builder(fold.train_rows, False, False)
    for label in ("A", "B"):
        component = panel_feature_subset(
            features, label, panel_fit_spec("independent_panel"))
        assert component.X.shape[1] == 4
        assert component.feature_blocks == {"gene_panel": (0, 1, 2, 3)}
        # A fully independent component already has its own target intercept;
        # retaining a constant panel dummy would be non-identifiable.
        assert component.X_nuisance is None
        assert component.nuisance_names == ()


def test_partition_rows_and_stitching_cover_test_once_in_original_order():
    panels = np.asarray(["B", "A", "A", "B", "A", "B"])
    test_rows = np.asarray([5, 1, 3, 2])
    split = partition_rows(test_rows, panels)
    assert set(split) == {"A", "B"}
    assert set(split["A"]).isdisjoint(split["B"])
    assert set(np.r_[split["A"], split["B"]]) == set(test_rows)
    prediction = combine_partition_predictions(
        test_rows,
        panels,
        {
            "A": np.column_stack((split["A"], split["A"] + 100)),
            "B": np.column_stack((split["B"], split["B"] + 100)),
        },
    )
    np.testing.assert_array_equal(prediction[:, 0], test_rows)
    np.testing.assert_array_equal(prediction[:, 1], test_rows + 100)


def test_tech_sar80_is_disabled_by_default_and_simulation_union_only():
    views = _views()
    primary = Settings(
        n_repetitions=1,
        loss_rates=(0.0,),
        run_information_controls=False,
        run_random_forest=False,
        run_qiao=False,
        run_mechanism_controls=False,
        run_calibration_controls=False,
    )
    assert prepare_tech_sar80_sensitivity(views, primary) is None
    plan = prepare_tech_sar80_sensitivity(views, primary, enabled=True)
    assert plan is not None
    assert plan.label == TECH_SAR_80_LABEL
    assert plan.export_name == TECH_SAR_80_EXPORT_NAME
    assert plan.export_name != "simulation_gene_overlap_0910"
    assert plan.settings.loss_rates == (0.8,)
    assert not plan.settings.run_mechanism_controls
    assert not plan.settings.run_calibration_controls
    assert {view.metadata["experiment_context"]["arm"] for view in plan.views} == {"union"}
    assert all(view.metadata["sensitivity_analysis"]["role"] == "secondary"
               for view in plan.views)
    assert plan.expected_scenario_units == len(plan.views) * primary.n_outer_folds
    assert plan.expected_model_evaluations == plan.expected_scenario_units * 3


def test_projection_or_other_natural_observation_cannot_enter_synthetic_sensitivity():
    views = _views(include_controls=False, include_shared_a_ablation=False)
    natural = tuple(replace(
        view,
        natural_observed=np.asarray(view.reference, dtype=bool),
        metadata={**dict(view.metadata), "independent_unit": "paired_real_dataset"},
    ) for view in views)
    settings = Settings(
        n_repetitions=1,
        loss_rates=(0.0,),
        run_information_controls=False,
        run_random_forest=False,
        run_qiao=False,
        run_mechanism_controls=False,
        run_calibration_controls=False,
    )
    with pytest.raises(ValueError, match="simulation-only|not eligible"):
        prepare_tech_sar80_sensitivity(natural, settings, enabled=True)


def _scenario_frames():
    aggregate, contrasts = [], []
    for loss_rate in (0.0, 0.8):
        for overlap in OVERLAPS:
            for model, offset in (("PU", 0.0), ("PU-MIRT", 0.01), ("PU-Joint", 0.02)):
                row = dict(
                    dataset="simulation",
                    sharing_strength=0.5,
                    panel_design="crossed",
                    analysis="primary" if loss_rate == 0 else "sensitivity",
                    mechanism="technical_sar",
                    loss_rate=loss_rate,
                    calibration_fraction=0.2,
                    calibration_spec="correct",
                    arm="union",
                    model=model,
                    actual_overlap=overlap,
                    macro_auprc=0.3 + offset,
                    macro_log_loss=0.4 - offset,
                    macro_brier=0.2 - offset,
                )
                aggregate.append(row)
                if model != "PU":
                    for metric in ("macro_auprc", "macro_log_loss", "macro_brier"):
                        contrasts.append({
                            **{key: row[key] for key in (
                                "dataset", "sharing_strength", "panel_design", "analysis",
                                "mechanism", "loss_rate", "calibration_fraction",
                                "calibration_spec", "model", "actual_overlap")},
                            "metric": metric,
                            "difference_vs_100pct_overlap": offset,
                        })
    return pd.DataFrame(aggregate), pd.DataFrame(contrasts)


def test_plot_titles_and_paths_keep_primary_and_sensitivity_separate(tmp_path, monkeypatch):
    aggregate, contrasts = _scenario_frames()
    artifacts = SimpleNamespace(
        tables={"aggregate": aggregate, "overlap_contrasts": contrasts},
        export_dir=tmp_path,
        manifest={},
    )
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    paths = plot_overlap_results(artifacts, tmp_path, prefix="paired", display=True)
    # There are no extra control arms: primary and sensitivity each produce
    # one performance figure and one R_M figure.
    assert len(paths) == 4
    assert len(set(path.name for path in paths)) == 4
    titles = [figure._suptitle.get_text() for figure in shown]
    assert sum("Primary: no added positive-label loss" in title for title in titles) == 2
    assert sum("Sensitivity:" in title and "80%" in title for title in titles) == 2
    assert sum("sensitivity_technical_sar_loss80" in path.name for path in paths) == 2
    assert all(axis.get_title() for figure in shown for axis in figure.axes)
    for figure in shown:
        plt.close(figure)
