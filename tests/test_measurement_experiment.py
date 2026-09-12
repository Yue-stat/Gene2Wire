"""Integration contracts for combined measurement-degradation experiment views."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gene2wire.experiments.contracts import ExperimentDataset, FeatureSet, Fold
from gene2wire.experiments.measurement_experiment import (
    MeasurementConfig,
    build_measurement_views,
)
from gene2wire.experiments import pipeline
from gene2wire.experiments.protocol import Settings


def _dataset() -> ExperimentDataset:
    n_cells, n_genes, n_targets = 45, 6, 6
    rows = np.arange(n_cells)[:, None]
    columns = np.arange(n_targets)[None, :]
    measured = (rows + 2 * columns) % 7 != 0
    reference = ((3 * rows + columns) % 5 == 0) & measured
    gene_matrix = (
        np.sin(np.arange(n_cells * n_genes).reshape(n_cells, n_genes) / 7.0)
        + np.arange(n_genes)[None, :] / 5.0
    )
    gene_names = tuple(f"g{index}" for index in range(n_genes))

    def source_features(train_rows, use_location, use_target_features):
        del train_rows
        if use_location or use_target_features:
            raise AssertionError("This fixture has only gene features")
        return FeatureSet(
            X=gene_matrix.copy(),
            feature_blocks={"gene": tuple(range(n_genes))},
            feature_names=gene_names,
        )

    folds = (
        Fold(0, np.arange(0, 24), np.arange(24, 33), np.arange(33, 45)),
        Fold(1, np.arange(21, 45), np.arange(12, 21), np.arange(0, 12)),
    )

    def split_builder(n_outer_folds, seed):
        del seed
        assert n_outer_folds == 2
        return folds

    return ExperimentDataset(
        name="measurement-fixture",
        reference=reference,
        measured=measured,
        cell_ids=tuple(f"cell-{index}" for index in range(n_cells)),
        target_ids=tuple(f"t{index}" for index in range(n_targets)),
        feature_builder=source_features,
        split_builder=split_builder,
        groups={"sample": np.repeat(("s1", "s2", "s3"), 15)},
        metadata={"independent_unit": "paired_real_dataset"},
        gene_matrix=gene_matrix,
        gene_names=gene_names,
    )


def _settings(**updates) -> Settings:
    values = dict(
        n_outer_folds=2,
        n_jobs=1,
        n_repetitions=1,
        seed=41,
        paired_fraction=0.4,
        loss_rates=(0.0,),
        candidate_budget=1,
        penalties=(0.1,),
        maxiter=5,
        retry_maxiter=5,
        init_direct_maxiter=5,
        run_information_controls=False,
        run_random_forest=False,
        run_mechanism_controls=False,
        run_calibration_controls=False,
        run_qiao=False,
        run_pu_comparators=False,
        calibration_fractions=(0.4,),
    )
    values.update(updates)
    return Settings(**values)


def _full_config() -> MeasurementConfig:
    return MeasurementConfig(
        gene_coverages=(1.0,),
        target_coverages=(1.0,),
        retentions=(1.0,),
        anchor_gene_coverage=1.0,
        anchor_target_coverage=1.0,
        anchor_retention=1.0,
        include_matched_uniform=False,
        include_natural_recovery=False,
    )


def _small_grid_config() -> MeasurementConfig:
    return MeasurementConfig(
        gene_coverages=(1.0, 0.5),
        target_coverages=(1.0, 0.5),
        retentions=(1.0, 0.5),
        anchor_gene_coverage=0.5,
        anchor_target_coverage=0.5,
        anchor_retention=0.5,
        include_matched_uniform=True,
        include_natural_recovery=False,
    )


def _view(views, gene_coverage, target_coverage):
    matches = [
        view for view in views
        if np.isclose(
            view.metadata["experiment_context"]["gene_requested_coverage"],
            gene_coverage,
        )
        and np.isclose(
            view.metadata["experiment_context"]["target_requested_coverage"],
            target_coverage,
        )
    ]
    assert len(matches) == 1
    return matches[0]


def test_full_gene_endpoint_exposes_all_source_genes_in_every_assay():
    dataset = _dataset()
    views, tables = build_measurement_views(
        dataset, _settings(), _full_config(), repetition=0)
    assert len(views) == 1
    view = views[0]
    context = view.metadata["experiment_context"]
    assert context["gene_panel_size"] == len(dataset.gene_names)
    assert context["gene_coverage"] == 1.0
    assert context["gene_union_coverage"] == 1.0

    membership = tables["measurement_gene_panel_membership"]
    endpoint = membership.loc[np.isclose(membership["requested_coverage"], 1.0)]
    assert set(endpoint["assay"]) == {"A", "B", "C"}
    assert endpoint.groupby("assay")["measured"].sum().eq(len(dataset.gene_names)).all()

    fold = view.split_builder(2, 999)[0]
    features = view.feature_builder(fold.train_rows, False, False)
    p = len(dataset.gene_names)
    np.testing.assert_array_equal(features.X[:, p:2 * p], 1.0)
    expected_center = dataset.gene_matrix[fold.train_rows].mean(axis=0)
    expected_scale = dataset.gene_matrix[fold.train_rows].std(axis=0)
    expected_scale[expected_scale < 1e-12] = 1.0
    np.testing.assert_allclose(
        features.X[:, :p],
        (dataset.gene_matrix - expected_center) / expected_scale,
    )
    np.testing.assert_array_equal(view.training_measured, dataset.measured)


def test_masked_features_and_training_target_support_follow_assay_panels():
    dataset = _dataset()
    views, tables = build_measurement_views(
        dataset, _settings(), _small_grid_config(), repetition=0)
    view = _view(views, 0.5, 0.5)
    fold = view.split_builder(2, 999)[0]
    features = view.feature_builder(fold.train_rows, False, False)
    assays = np.asarray(view.virtual_assays).astype(str)
    p = len(dataset.gene_names)

    gene_membership = tables["measurement_gene_panel_membership"]
    gene_membership = gene_membership.loc[
        np.isclose(gene_membership["requested_coverage"], 0.5)]
    gene_lookup = {
        (row.assay, row.item): bool(row.measured)
        for row in gene_membership.itertuples(index=False)
    }
    expected_gene_mask = np.asarray([
        [gene_lookup[(assay, gene)] for gene in dataset.gene_names]
        for assay in assays
    ])
    observed_block = np.asarray(features.X[:, features.feature_blocks["gene_observed"]],
                                dtype=bool)
    np.testing.assert_array_equal(observed_block, expected_gene_mask)
    values = features.X[:, features.feature_blocks["gene_value"]]
    np.testing.assert_array_equal(values[~expected_gene_mask], 0.0)
    for gene in range(p):
        visible_train = fold.train_rows[expected_gene_mask[fold.train_rows, gene]]
        center = dataset.gene_matrix[visible_train, gene].mean()
        scale = dataset.gene_matrix[visible_train, gene].std()
        scale = 1.0 if scale < 1e-12 else scale
        np.testing.assert_allclose(
            values[expected_gene_mask[:, gene], gene],
            (dataset.gene_matrix[expected_gene_mask[:, gene], gene] - center) / scale,
        )

    levels = tuple(sorted(set(assays)))
    expected_assay_block = np.column_stack([
        (assays == assay).astype(float) for assay in levels[1:]
    ])
    np.testing.assert_array_equal(
        features.X[:, features.feature_blocks["assay"]], expected_assay_block)
    assert features.X.shape == (len(dataset.cell_ids), 2 * p + 2)

    target_membership = tables["measurement_target_panel_membership"]
    target_membership = target_membership.loc[
        np.isclose(target_membership["requested_coverage"], 0.5)]
    target_lookup = {
        (row.assay, row.item): bool(row.measured)
        for row in target_membership.itertuples(index=False)
    }
    panel_support = np.asarray([
        [target_lookup[(assay, target)] for target in dataset.target_ids]
        for assay in assays
    ])
    np.testing.assert_array_equal(
        view.training_measured, np.asarray(dataset.measured, bool) & panel_support)
    assert np.any(dataset.measured & ~view.training_measured)

    support_assays = tables["measurement_support_assays"]
    assert {"effective_target_count", "effective_target_coverage",
            "measured_entry_count"}.issubset(support_assays)
    assert set(support_assays["virtual_assay"]) == {"A", "B", "C"}
    support_pairs = tables["measurement_support_pairs"]
    assert {"assay_a", "assay_b", "co_measured_target_count",
            "connected_edge"}.issubset(support_pairs)
    assert len(support_pairs) == 4 * 2 * 3 * 3

    evaluation = view.measurement_evaluation
    np.testing.assert_array_equal(evaluation.reference, dataset.reference)
    np.testing.assert_array_equal(evaluation.source_measured, dataset.measured)
    np.testing.assert_array_equal(evaluation.assays, assays)


def test_gene_target_joint_support_is_fold_local_and_discloses_unsupported_pairs():
    source = _dataset()
    source = replace(source, measured=np.ones_like(source.measured, dtype=bool))
    config = MeasurementConfig(
        gene_coverages=(1.0, 2 / 3, 0.5),
        target_coverages=(1.0, 2 / 3, 0.5),
        retentions=(1.0, 0.5),
        anchor_gene_coverage=2 / 3,
        anchor_target_coverage=2 / 3,
        anchor_retention=0.5,
        include_matched_uniform=False,
        include_natural_recovery=False,
    )
    views, tables = build_measurement_views(
        source, _settings(), config, repetition=0, panel_seed=0)

    summary = tables["measurement_gene_target_support"]
    required = {
        "outer_fold", "split", "gene_target_pair_count",
        "supported_gene_target_pair_count",
        "unsupported_gene_target_pair_count",
        "unsupported_gene_target_pair_fraction",
        "all_gene_target_pairs_supported", "minimum_co_measured_cells",
        "median_co_measured_cells", "maximum_co_measured_cells",
    }
    assert required.issubset(summary)
    assert set(summary["split"]) == {"train", "validation", "test"}
    assert summary["gene_target_pair_count"].eq(
        len(source.gene_names) * len(source.target_ids)).all()

    connected = summary.loc[
        np.isclose(summary["gene_requested_coverage"], 2 / 3)
        & np.isclose(summary["target_requested_coverage"], 2 / 3)
    ]
    assert connected["all_gene_target_pairs_supported"].all()
    assert connected["unsupported_gene_target_pair_count"].eq(0).all()

    sparse = summary.loc[
        np.isclose(summary["gene_requested_coverage"], 0.5)
        & np.isclose(summary["target_requested_coverage"], 0.5)
    ]
    assert sparse["unsupported_gene_target_pair_count"].gt(0).all()
    unsupported = tables["measurement_unsupported_gene_target_pairs"]
    assert {"gene", "target", "co_measured_cell_count"}.issubset(unsupported)
    assert set(unsupported["split"]) == {"train"}
    assert unsupported["co_measured_cell_count"].eq(0).all()
    expected_rows = int(summary.loc[
        summary["split"].eq("train"),
        "unsupported_gene_target_pair_count",
    ].sum())
    assert len(unsupported) == expected_rows

    sparse_view = _view(views, 0.5, 0.5)
    fold = sparse_view.split_builder(2, 0)[0]
    cell_level_counts = (
        sparse_view.feature_builder.observed_mask[fold.train_rows].astype(np.int64).T
        @ sparse_view.training_measured[fold.train_rows].astype(np.int64)
    )
    audited = sparse.loc[
        sparse["outer_fold"].eq(fold.outer_fold)
        & sparse["split"].eq("train")
    ].iloc[0]
    assert audited["unsupported_gene_target_pair_count"] == int(
        np.sum(cell_level_counts == 0))
    assert audited["minimum_co_measured_cells"] == int(cell_level_counts.min())
    assert audited["median_co_measured_cells"] == float(
        np.median(cell_level_counts))
    assert audited["maximum_co_measured_cells"] == int(cell_level_counts.max())


def test_optional_train_only_pca_occurs_after_mask_and_retains_gene_mask():
    source = _dataset()
    source = replace(
        source,
        metadata={**source.metadata, "n_gene_components": 2},
    )
    settings = _settings()
    views, _ = build_measurement_views(
        source, settings, _small_grid_config(), repetition=0)
    view = _view(views, 0.5, 0.5)
    fold = view.split_builder(2, 0)[0]
    features = view.feature_builder(fold.train_rows, False, False)

    assert len(features.feature_blocks["gene_value"]) == 2
    assert len(features.feature_blocks["gene_observed"]) == 6
    assert features.X.shape[1] == 2 + 6 + 2
    assert features.metadata["n_gene_components_used"] == 2
    assert features.metadata["pca_fit_rows"] == fold.train_rows.tolist()
    assert "train-only PCA" in features.metadata["preprocessing"]

    gene_mask = np.asarray(view.feature_builder.observed_mask, dtype=bool)
    altered_matrix = np.asarray(source.gene_matrix).copy()
    altered_matrix[~gene_mask] += 1e6
    altered_source = replace(source, gene_matrix=altered_matrix)
    altered_views, _ = build_measurement_views(
        altered_source, settings, _small_grid_config(), repetition=0)
    altered = _view(altered_views, 0.5, 0.5).feature_builder(
        fold.train_rows, False, False)
    np.testing.assert_allclose(altered.X, features.X)


def test_condition_registry_deduplicates_coordinates_and_merges_roles():
    views, tables = build_measurement_views(
        _dataset(), _settings(), _small_grid_config(), repetition=0)
    assert len(views) == 4
    scenarios = tables["measurement_scenarios"]
    coordinates = [
        "gene_requested_coverage", "target_requested_coverage",
        "positive_retention", "mechanism",
    ]
    assert len(scenarios) == 7
    assert not scenarios.duplicated(coordinates).any()

    merged = scenarios.loc[
        np.isclose(scenarios["gene_requested_coverage"], 0.5)
        & np.isclose(scenarios["target_requested_coverage"], 0.5)
        & np.isclose(scenarios["positive_retention"], 0.5)
        & scenarios["mechanism"].eq("assay_target_sar")
    ]
    assert len(merged) == 1
    assert merged.iloc[0]["condition_roles"] == "coverage_heatmap+retention_curve"

    for view in views:
        prepared = pipeline._prepare(
            view, view.split_builder(2, 0)[0], _settings())
        declared = [dict(item) for item in view.metadata["measurement_scenarios"]]
        assert pipeline._scenarios(prepared, _settings()) == declared
        keys = [
            (item["mechanism"], item["positive_retention"], item["condition_roles"])
            for item in declared
        ]
        assert len(keys) == len(set(keys))
        np.testing.assert_array_equal(
            pipeline._fit_measured(prepared), view.training_measured)


def test_paired_reference_budget_is_stratified_by_biology_and_virtual_assay():
    settings = _settings()
    views, _ = build_measurement_views(
        _dataset(), settings, _full_config(), repetition=0)
    prepared = pipeline._prepare(
        views[0], views[0].split_builder(2, 0)[0], settings)

    groups, description = pipeline._paired_sampling_groups(prepared)

    assert description == "sample x virtual_assay"
    expected = {
        f"{sample}\0{assay}"
        for sample in ("s1", "s2", "s3")
        for assay in ("A", "B", "C")
    }
    assert set(groups) == expected
    assert all(np.sum(groups == label) == 5 for label in expected)


def test_simulation_repetition_nests_multiple_panel_seeds_with_common_random_numbers():
    generated = replace(
        _dataset(),
        metadata={"independent_unit": "generated_dataset", "sharing_strength": 0.5},
    )
    settings = _settings()
    first, _ = build_measurement_views(
        generated, settings, _small_grid_config(), repetition=2, panel_seed=0)
    second, _ = build_measurement_views(
        generated, settings, _small_grid_config(), repetition=2, panel_seed=1)
    first_view = _view(first, 0.5, 0.5)
    second_view = _view(second, 0.5, 0.5)

    assert first_view.metadata["experiment_repetition"] == 2
    assert second_view.metadata["experiment_repetition"] == 2
    assert first_view.metadata["experiment_context"]["data_repetition"] == 2
    assert second_view.metadata["experiment_context"]["data_repetition"] == 2
    assert first_view.metadata["experiment_context"]["panel_seed"] == 0
    assert second_view.metadata["experiment_context"]["panel_seed"] == 1
    np.testing.assert_array_equal(first_view.virtual_assays, second_view.virtual_assays)
    np.testing.assert_array_equal(first_view.observation_plan.h,
                                  second_view.observation_plan.h)
    np.testing.assert_array_equal(first_view.observation_plan.u,
                                  second_view.observation_plan.u)
    assert not np.array_equal(first_view.feature_builder.observed_mask,
                              second_view.feature_builder.observed_mask)


def test_primary_plan_retains_fifteen_methods_and_adds_exactly_three_comparators():
    settings = _settings(
        run_information_controls=True,
        run_random_forest=True,
        run_qiao=True,
        run_pu_comparators=True,
    )
    views, _ = build_measurement_views(
        _dataset(), settings, _full_config(), repetition=0)
    prepared = pipeline._prepare(
        views[0], views[0].split_builder(2, 0)[0], settings)
    names = [row["model"] for row in pipeline._planned_models(prepared, settings)]
    assert len(names) == 18
    assert len(set(names)) == 18
    assert {"PU-Joint", "GenEML-adapted", "Inductive-PU-MC", "SAR-PU"}.issubset(names)


def test_comparator_wrapper_reports_requested_public_names_and_probabilities():
    settings = _settings(run_pu_comparators=True, maxiter=20, retry_maxiter=20)
    views, _ = build_measurement_views(
        _dataset(), settings, _full_config(), repetition=0)
    fold = views[0].split_builder(2, 0)[0]
    prepared = pipeline._prepare(views[0], fold, settings)
    observed = np.asarray(prepared.reference, dtype=bool)
    exposure = np.ones_like(observed, dtype=float)
    records = []
    tables = {"selected": [], "tuning": [], "failures": []}

    def record(name, prediction, semantics, extra=None, **kwargs):
        records.append((name, np.asarray(prediction), semantics,
                        np.asarray(kwargs["estimated_sensitivity"]), extra))

    pipeline._run_pu_comparators(
        prepared, observed, exposure, exposure, settings,
        {"repetition": 0, "outer_fold": 0, "mechanism": "assay_target_sar",
         "positive_retention": 1.0, "panel_id": "full"},
        record, tables,
    )
    expected = {"GenEML-adapted", "Inductive-PU-MC", "SAR-PU"}
    assert {item[0] for item in records} == expected
    assert {row["model"] for row in tables["selected"]} == expected
    assert {row["model"] for row in tables["tuning"]} == expected
    assert not tables["failures"]
    for _, prediction, semantics, sensitivity, _ in records:
        assert prediction.shape == (len(fold.test_rows), len(prepared.target_ids))
        assert np.isfinite(prediction).all() and ((0 <= prediction) & (prediction <= 1)).all()
        assert np.isfinite(sensitivity).all() and ((0 <= sensitivity) & (sensitivity <= 1)).all()
        assert semantics == "reference"


def test_tiny_fold_uses_training_mask_but_scores_fixed_native_reference(
        tmp_path, monkeypatch):
    dataset = _dataset()
    config = MeasurementConfig(
        gene_coverages=(1.0,),
        target_coverages=(1.0, 0.5),
        retentions=(1.0,),
        anchor_gene_coverage=1.0,
        anchor_target_coverage=0.5,
        anchor_retention=1.0,
        include_matched_uniform=False,
        include_natural_recovery=False,
    )
    settings = _settings()
    views, _ = build_measurement_views(dataset, settings, config, repetition=0)
    view = _view(views, 1.0, 0.5)
    fold = view.split_builder(2, 0)[0]
    prepared = pipeline._prepare(view, fold, settings)
    scenario = next(
        item for item in pipeline._scenarios(prepared, settings)
        if item["positive_retention"] == 1.0)

    calls = []

    def fake_grid(**kwargs):
        calls.append(kwargs)
        predictions = np.full(
            (len(kwargs["test_X"]), kwargs["train"].n_targets), 0.35)
        records = {}
        for model in kwargs["models"]:
            records[model.name] = SimpleNamespace(
                fitted=SimpleNamespace(config=model),
                latent_probability=predictions.copy(),
                summary=lambda name=model.name: {"model": name, "tuning_trials": 0},
                tuning=SimpleNamespace(trials=[]),
            )
        return SimpleNamespace(models=records)

    monkeypatch.setattr(pipeline, "run_model_grid", fake_grid)
    tables = pipeline._run_fold(
        prepared, 0, settings,
        tmp_path / "checkpoints", tmp_path / "exports", "test-code",
        scenario=scenario,
    )

    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_array_equal(
        call["train"].W_measured, view.training_measured[fold.train_rows])
    np.testing.assert_array_equal(
        call["validation"].W_measured,
        view.training_measured[fold.validation_rows])
    assert settings.run_pu_comparators is False

    metrics = pd.DataFrame(tables["metrics"])
    assert set(metrics["model"]) == {model.name for model in settings.models()}
    assert set(metrics["evaluation_scope"]) == {
        "native_reference", "on_panel", "off_panel", "hidden_candidate",
    }
    assert len(metrics) == len(settings.models()) * 4

    test = fold.test_rows
    native = np.asarray(dataset.measured[test], dtype=bool)
    on_panel = np.asarray(view.training_measured[test], dtype=bool)
    expected_counts = {
        "native_reference": int(native.sum()),
        "on_panel": int(on_panel.sum()),
        "off_panel": int((native & ~on_panel).sum()),
        "hidden_candidate": int((on_panel & ~dataset.reference[test]).sum()),
    }
    for scope, expected in expected_counts.items():
        selected = metrics.loc[metrics["evaluation_scope"].eq(scope)]
        assert selected["n_evaluated"].eq(expected).all()
    assert expected_counts["native_reference"] > expected_counts["on_panel"]
