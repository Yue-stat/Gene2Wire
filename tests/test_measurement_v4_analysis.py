"""Contracts for the measurement-degradation V4 plot-only helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "notebooks" / "measurement_v4_analysis.py"


def helper():
    spec = importlib.util.spec_from_file_location(
        "measurement_v4_analysis", HELPER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses inspects the defining module while decorating classes.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _per_repetition() -> pd.DataFrame:
    rows = []
    models = (
        "PU",
        "PU-MIRT",
        "PU-Joint",
        "GenEML-authors-mask",
        "Inductive-PU-MC",
        "SAR-PU",
    )
    conditions = []
    for gene in (1.0, 0.7, 0.4):
        for target in (1.0, 0.7, 0.4):
            for retention in (1.0, 0.8, 0.6, 0.4, 0.2):
                roles = {"combined_mask_curve"}
                if gene == target == 1.0:
                    roles.add("positive_label_loss_only")
                if target == retention == 1.0:
                    roles.add("gene_coverage_only")
                if gene == retention == 1.0:
                    roles.add("target_coverage_only")
                if gene == target == retention == 1.0:
                    roles.add("full_control")
                conditions.append(("+".join(sorted(roles)), gene, target, retention))
    for sharing in (0.0, 1.0):
        for repetition in (0, 1):
            for condition_index, (role, gene, target, retention) in enumerate(
                conditions
            ):
                for model_index, model in enumerate(models):
                    # The two facets occupy disjoint raw ranges.  Correct V4
                    # normalization must nevertheless span [0, 1] in each.
                    base = 0.30 + 0.30 * sharing + 0.01 * condition_index
                    advantage = 0.01 * model_index
                    rows.append(
                        {
                            "dataset": "simulation",
                            "sharing_strength": sharing,
                            "repetition": repetition,
                            "model": model,
                            "condition_roles": role,
                            "evaluation_scope": "native_reference",
                            "mechanism": "assay_target_sar",
                            "gene_requested_coverage": gene,
                            # Actual values deliberately differ; plots must use
                            # the requested/theoretical coordinate.
                            "gene_coverage": gene if gene == 1 else 5 / 7,
                            "target_requested_coverage": target,
                            "target_coverage": target if target == 1 else 5 / 7,
                            "positive_retention": retention,
                            "probability_semantics": (
                                "observed"
                                if model == "SAR-PU"
                                else "reference_probability"
                            ),
                            "macro_auprc": base + advantage + repetition * 0.001,
                            "macro_auroc": base + 0.3 + advantage,
                            "macro_log_loss": 0.9 - base - advantage,
                            "macro_brier": 0.4 - 0.2 * base - advantage,
                            "hidden_recall_at_h": (
                                np.nan
                                if retention == 1.0
                                else base + advantage
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_curve_coordinates_use_theoretical_values_and_retained_harmonic_mean():
    module = helper()
    rows = _per_repetition()
    positive = module.prepare_v4_curve_table(
        rows, family="positive_label_loss", metrics=("auprc",)
    )
    assert set(np.round(positive["plot_x"], 12)) == {0.0, 0.2, 0.4, 0.6, 0.8}
    assert positive["gene_requested_coverage"].eq(1.0).all()
    assert positive["target_requested_coverage"].eq(1.0).all()

    gene = module.prepare_v4_curve_table(
        rows, family="gene_coverage", metrics=("auprc",)
    )
    assert set(np.round(gene["plot_x"], 12)) == {0.4, 0.7, 1.0}
    assert np.isclose(gene.loc[gene["plot_x"].eq(0.7), "plot_x"].iloc[0], 0.7)
    assert gene["target_requested_coverage"].eq(1.0).all()
    assert gene["positive_retention"].eq(1.0).all()

    target = module.prepare_v4_curve_table(
        rows, family="target_coverage", metrics=("auprc",)
    )
    assert set(np.round(target["plot_x"], 12)) == {0.4, 0.7, 1.0}
    assert target["gene_requested_coverage"].eq(1.0).all()
    assert target["positive_retention"].eq(1.0).all()

    combined = module.prepare_v4_curve_table(
        rows, family="combined_masks", metrics=("auprc",)
    )
    assert len(combined[
        ["gene_requested_coverage", "target_requested_coverage", "positive_retention"]
    ].drop_duplicates()) == 45
    expected = 3.0 / (1 / 0.7 + 1 / 0.7 + 1 / 0.4)
    assert np.any(np.isclose(combined["plot_x"], expected))
    assert module.CURVE_FAMILY_SPECS["combined_masks"].x_label == (
        "Harmonic mean retained measurement"
    )


def test_metric_grid_has_sharing_rows_metric_columns_and_gene2wire_alias():
    module = helper()
    figure, axes = module.plot_v4_metric_grid(
        _per_repetition(),
        family="positive_label_loss",
        models=module.GENE2WIRE_FAMILY_MODELS,
        metrics=("auprc", "log_loss"),
    )
    assert axes.shape == (2, 2)
    assert axes[0, 0].get_xlim()[0] < axes[0, 0].get_xlim()[1]
    assert axes[0, 0].get_ylim()[0] > 0
    legend_labels = [text.get_text() for legend in figure.legends for text in legend.texts]
    assert "Gene2Wire" in legend_labels
    assert "PU-Joint" not in legend_labels
    gene2wire_line = next(
        line for line in axes[0, 0].lines if line.get_label() == "Gene2Wire"
    )
    pu_line = next(line for line in axes[0, 0].lines if line.get_label() == "PU")
    mirt_line = next(
        line for line in axes[0, 0].lines if line.get_label() == "PU-MIRT"
    )
    np.testing.assert_allclose(gene2wire_line.get_xdata(), [0.0, 0.2, 0.4, 0.6, 0.8])
    assert pu_line.get_markerfacecolor() == "white"
    assert mirt_line.get_markerfacecolor() == "white"
    assert gene2wire_line.get_markerfacecolor() == "#D55E00"
    plt.close(figure)


def test_funky_normalization_is_within_sharing_and_reverses_losses():
    module = helper()
    scorecard = module.prepare_v4_funky_table(
        _per_repetition(),
        metrics=("auprc", "log_loss"),
        # Keep this directionality test separate from the fail-closed
        # observed-only SAR semantics exercised below.
        models=module.FUNKY_MODELS[:-1],
    )
    normalized = module.normalize_v4_funky_table(scorecard)
    for _, group in normalized.groupby(
        ["facet", "family", "condition", "metric"], observed=True
    ):
        finite = group.loc[np.isfinite(group["value"])]
        if len(finite) >= 2 and finite["value"].max() != finite["value"].min():
            assert finite["desirability"].min() == 0.0
            assert finite["desirability"].max() == 1.0

    selected = normalized.loc[
        normalized["family"].eq("positive_label_loss")
        & normalized["condition"].eq("positive_loss_0.2")
        & normalized["metric"].eq("auprc")
    ]
    assert set(selected.groupby("facet")["desirability"].min()) == {0.0}
    assert set(selected.groupby("facet")["desirability"].max()) == {1.0}

    loss = normalized.loc[
        normalized["facet"].eq("sharing=0")
        & normalized["family"].eq("positive_label_loss")
        & normalized["condition"].eq("positive_loss_0.2")
        & normalized["metric"].eq("log_loss")
    ].set_index("model")
    assert loss.loc["Inductive-PU-MC", "desirability"] > loss.loc[
        "PU-Joint", "desirability"
    ]


def test_funky_partitions_nine_one_factor_scenarios_into_forty_five_columns():
    module = helper()
    scorecard = module.prepare_v4_funky_table(_per_repetition())
    catalog = scorecard[
        [
            "family",
            "condition",
            "metric",
            "gene_requested_coverage",
            "target_requested_coverage",
            "positive_retention",
        ]
    ].drop_duplicates()
    triples = catalog[
        [
            "gene_requested_coverage",
            "target_requested_coverage",
            "positive_retention",
        ]
    ].drop_duplicates()
    assert len(triples) == 9
    assert len(catalog[["family", "condition", "metric"]].drop_duplicates()) == 45
    assert (
        catalog.groupby(
            [
                "gene_requested_coverage",
                "target_requested_coverage",
                "positive_retention",
                "metric",
            ],
            observed=True,
        )["family"].nunique().max()
        == 1
    )
    assert "combined_masks" not in set(catalog["family"])
    assert "gene_target_coverage" not in set(catalog["family"])
    assert set(catalog["family"]) == set(module.DEFAULT_V4_FUNKY_FAMILIES)


def test_funky_tiny_nonzero_span_still_gets_exact_minimum_and_maximum():
    module = helper()
    scorecard = pd.DataFrame(
        [
            {
                "facet": "sharing=0.5",
                "family": "positive_label_loss",
                "condition": "positive_loss_0.2",
                "metric": "auprc",
                "higher_is_better": True,
                "value": value,
            }
            for value in (0.5, 0.5 + 1e-12)
        ]
    )
    normalized = module.normalize_v4_funky_table(scorecard)
    assert normalized["desirability"].tolist() == [0.0, 1.0]


def test_native_inputs_keep_na_blank_and_use_object_id_columns():
    module = helper()
    scorecard = module.prepare_v4_funky_table(_per_repetition())
    # Hidden recall is undefined at 100% retention, producing one all-NA
    # physical column that funkyheatmappy 0.7 cannot concatenate.
    inputs = module.build_funkyheatmappy_inputs(scorecard)
    catalog = inputs["catalog"]
    assert not (
        catalog["metric"].eq("hidden_recall")
        & catalog["condition"].eq("positive_loss_0")
    ).any()
    assert inputs["dropped_all_na_plot_ids"]
    assert inputs["dropped_all_na_columns"]
    assert sum(
        "On-panel hidden Recall" in label
        for label in inputs["dropped_all_na_columns"]
    ) == 5
    assert all(
        not label.startswith("metric_")
        and " / " in label
        for label in inputs["dropped_all_na_columns"]
    )

    column_info = inputs["column_info"]
    assert column_info["id_color"].dtype == object
    assert column_info["id_size"].dtype == object
    assert set(column_info.loc[column_info["geom"].eq("funkyrect"), "geom"]) == {
        "funkyrect"
    }
    assert inputs["palettes"] == {"metric_blue": ["#2F80ED"]}
    assert inputs["legends"][0]["enabled"] is True

    # SAR-PU was marked observed-only in the fixture, so no four-model proper
    # score comparison has a complete repetition.  The entire physical column
    # must remain N/A, never rescale the other three models.
    loss = scorecard.loc[scorecard["metric"].eq("log_loss")]
    assert loss["value"].isna().all()
    assert loss["n_matched_repetitions"].eq(0).all()
    assert loss["n_expected_repetitions"].eq(2).all()
    assert loss["incomplete_models"].str.contains("SAR-PU").all()


def test_funky_labels_use_only_requested_theoretical_coverages():
    module = helper()
    scorecard = module.prepare_v4_funky_table(
        _per_repetition(), metrics=("auprc",)
    )
    labels = set(scorecard["condition_label"])
    assert "Loss 20%" in labels
    assert "G70%" in labels
    assert "T70%" in labels
    assert all("→" not in label for label in labels)
    assert all("71.4" not in label for label in labels)


def test_notebook_source_excludes_loader_only_boundary():
    source = helper().notebook_source()
    assert "def plot_v4_metric_grid(" in source
    assert "def plot_v4_funky_heatmap(" in source
    assert "def notebook_source(" not in source


def test_curve_means_use_only_repetitions_complete_for_every_requested_model():
    module = helper()
    rows = _per_repetition()
    missing = (
        rows["sharing_strength"].eq(0.0)
        & rows["repetition"].eq(1)
        & rows["positive_retention"].eq(0.8)
        & rows["gene_requested_coverage"].eq(1.0)
        & rows["target_requested_coverage"].eq(1.0)
        & rows["model"].eq("PU-MIRT")
    )
    rows.loc[missing, "macro_auprc"] = np.nan
    comparison = module.prepare_v4_curve_comparison(
        rows,
        family="positive_label_loss",
        models=module.GENE2WIRE_FAMILY_MODELS,
        metrics=("auprc",),
    )
    point = comparison.loc[
        comparison["facet"].eq("sharing=0")
        & np.isclose(comparison["plot_x"], 0.2)
    ]
    assert point["n_matched_repetitions"].eq(1).all()
    assert point["n_expected_repetitions"].eq(2).all()
    assert point["availability"].eq("partial").all()
    assert point["incomplete_models"].str.contains(r"PU-MIRT \(1/2\)").all()

    gene2wire = point.loc[point["model"].eq("PU-Joint"), "value"].item()
    expected = rows.loc[
        rows["sharing_strength"].eq(0.0)
        & rows["repetition"].eq(0)
        & rows["positive_retention"].eq(0.8)
        & rows["gene_requested_coverage"].eq(1.0)
        & rows["target_requested_coverage"].eq(1.0)
        & rows["model"].eq("PU-Joint"),
        "macro_auprc",
    ].item()
    assert gene2wire == expected


def test_curve_keeps_requested_legend_and_na_when_one_model_is_absent(capsys):
    module = helper()
    rows = _per_repetition().loc[
        lambda frame: frame["model"].ne("GenEML-authors-mask")
    ]
    figure, axes = module.plot_v4_metric_grid(
        rows,
        family="gene_coverage",
        models=module.EXTERNAL_CURVE_MODELS,
        metrics=("auprc",),
    )
    labels = [text.get_text() for legend in figure.legends for text in legend.texts]
    assert labels == ["Gene2Wire", "GenEML", "PU matrix completion"]
    assert all(not axis.lines for axis in axes.ravel())
    assert all(
        any("N/A" in text.get_text() for text in axis.texts)
        for axis in axes.ravel()
    )
    assert "incomplete models: GenEML (0/2)" in capsys.readouterr().out
    assert figure.v4_curve_audit["value"].isna().all()
    plt.close(figure)


def test_combined_curve_records_physical_conditions_collapsed_at_same_h():
    module = helper()
    comparison = module.prepare_v4_curve_comparison(
        _per_repetition(),
        family="combined_masks",
        models=module.GENE2WIRE_FAMILY_MODELS,
        metrics=("auprc",),
    )
    collision = comparison.loc[
        comparison["facet"].eq("sharing=0")
        & comparison["n_physical_conditions_at_x"].eq(2)
        & comparison["model"].eq("PU-Joint")
    ]
    assert not collision.empty
    row = collision.loc[
        collision["physical_conditions_at_x"].str.contains("G70%/T40%/R40%")
    ].iloc[0]
    assert row["n_expected_physical_units"] == 4
    assert row["n_matched_physical_units"] == 4
    assert "G70%/T40%/R40%" in row["physical_conditions_at_x"]
    assert "G40%/T70%/R40%" in row["physical_conditions_at_x"]


def test_combined_curve_drops_a_repetition_if_one_collapsed_condition_is_incomplete():
    module = helper()
    rows = _per_repetition()
    missing = (
        rows["sharing_strength"].eq(0.0)
        & rows["repetition"].eq(1)
        & rows["gene_requested_coverage"].eq(0.7)
        & rows["target_requested_coverage"].eq(0.4)
        & rows["positive_retention"].eq(0.4)
        & rows["model"].eq("PU-MIRT")
    )
    rows.loc[missing, "macro_auprc"] = np.nan
    comparison = module.prepare_v4_curve_comparison(
        rows,
        family="combined_masks",
        models=module.GENE2WIRE_FAMILY_MODELS,
        metrics=("auprc",),
    )
    point = comparison.loc[
        comparison["facet"].eq("sharing=0")
        & comparison["physical_conditions_at_x"].str.contains(
            "G70%/T40%/R40%", regex=False
        )
    ]
    assert point["n_matched_repetitions"].eq(1).all()
    assert point["n_expected_repetitions"].eq(2).all()
    assert point["n_matched_physical_units"].eq(2).all()
    assert point["n_expected_physical_units"].eq(4).all()


def test_funky_raw_means_are_paired_and_partial_support_is_audited():
    module = helper()
    rows = _per_repetition()
    missing = (
        rows["sharing_strength"].eq(0.0)
        & rows["repetition"].eq(1)
        & rows["positive_retention"].eq(0.8)
        & rows["gene_requested_coverage"].eq(1.0)
        & rows["target_requested_coverage"].eq(1.0)
        & rows["model"].eq("SAR-PU")
    )
    rows.loc[missing, "macro_auprc"] = np.nan
    scorecard = module.prepare_v4_funky_table(rows, metrics=("auprc",))
    point = scorecard.loc[
        scorecard["facet"].eq("sharing=0")
        & scorecard["condition"].eq("positive_loss_0.2")
    ]
    assert point["n_matched_repetitions"].eq(1).all()
    assert point["n_expected_repetitions"].eq(2).all()
    assert point["availability"].eq("partial").all()
    assert point["incomplete_models"].str.contains(r"SAR-PU \(1/2\)").all()
    expected = rows.loc[
        rows["sharing_strength"].eq(0.0)
        & rows["repetition"].eq(0)
        & rows["positive_retention"].eq(0.8)
        & rows["gene_requested_coverage"].eq(1.0)
        & rows["target_requested_coverage"].eq(1.0)
        & rows["model"].eq("PU-Joint"),
        "macro_auprc",
    ].item()
    assert point.loc[point["model"].eq("PU-Joint"), "value"].item() == expected


def test_funky_missing_requested_model_makes_entire_column_na():
    module = helper()
    rows = _per_repetition().loc[lambda frame: frame["model"].ne("SAR-PU")]
    scorecard = module.prepare_v4_funky_table(rows, metrics=("auprc",))
    assert scorecard["value"].isna().all()
    assert scorecard["n_matched_repetitions"].eq(0).all()
    assert scorecard["incomplete_models"].str.contains(r"SAR-PU \(0/2\)").all()
    normalized = module.normalize_v4_funky_table(scorecard)
    assert normalized["desirability"].isna().all()


@pytest.mark.parametrize(
    "prepare",
    (
        lambda module, rows: module.prepare_v4_curve_comparison(
            rows,
            family="positive_label_loss",
            models=module.GENE2WIRE_FAMILY_MODELS,
            metrics=("auprc",),
        ),
        lambda module, rows: module.prepare_v4_funky_table(
            rows, metrics=("auprc",)
        ),
    ),
)
def test_v4_plot_helpers_reject_an_entirely_missing_physical_condition(prepare):
    module = helper()
    rows = _per_repetition()
    missing_condition = (
        rows["sharing_strength"].eq(0.0)
        & rows["repetition"].eq(1)
        & rows["gene_requested_coverage"].eq(1.0)
        & rows["target_requested_coverage"].eq(1.0)
        & rows["positive_retention"].eq(0.8)
    )
    rows = rows.loc[~missing_condition]
    with pytest.raises(
        ValueError, match="missing declared factorial conditions"
    ):
        prepare(module, rows)
