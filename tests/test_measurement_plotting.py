import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.measurement_plotting import (
    KEY_MODELS,
    plot_accuracy_panel_sensitivity,
    plot_gene_target_brier_heatmap,
    plot_projection_tags_budget_recall,
    plot_retention_auprc,
)


def test_retention_plot_has_key_and_full_model_modes_and_averages_repetitions():
    rows = []
    for repetition in (0, 1):
        for retention in (0.1, 0.5, 1.0):
            for model, offset in (("PU", 0.0), ("PU-MIRT", 0.02),
                                  ("GenEML", 0.01), ("Logistic", -0.03)):
                rows.append({
                    "repetition": repetition,
                    "retention": retention,
                    "model": model,
                    "macro_auprc": 0.2 + 0.2 * retention + offset + .01 * repetition,
                })
    table = pd.DataFrame(rows)

    figure, axis = plot_retention_auprc(table, mode="key")
    assert figure is axis.figure
    assert [line.get_label() for line in axis.lines] == [
        model for model in KEY_MODELS if model in {"PU", "PU-MIRT", "GenEML"}
    ]
    pu = next(line for line in axis.lines if line.get_label() == "PU")
    np.testing.assert_allclose(pu.get_xdata(), [0.1, 0.5, 1.0])
    np.testing.assert_allclose(pu.get_ydata(), [0.225, 0.305, 0.405])
    assert axis.get_xlabel() == "Positive-label retention"
    plt.close(figure)

    figure, axis = plot_retention_auprc(table, mode="full")
    assert {line.get_label() for line in axis.lines} == {
        "PU", "PU-MIRT", "GenEML", "Logistic"
    }
    plt.close(figure)


def test_retention_plot_rejects_unknown_mode_and_invalid_probabilities():
    table = pd.DataFrame({"retention": [0.5], "model": ["PU"],
                          "macro_auprc": [0.3]})
    with pytest.raises(ValueError, match="mode must"):
        plot_retention_auprc(table, mode="short")
    bad = table.assign(retention=1.5)
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        plot_retention_auprc(bad)


def test_brier_heatmap_uses_one_fixed_baseline_and_zero_centered_scale():
    rows = []
    expected = np.empty((2, 2))
    for row, target_coverage in enumerate((0.5, 1.0)):
        for column, gene_coverage in enumerate((0.25, 1.0)):
            difference = (row - column) * 0.02 + 0.01
            expected[row, column] = difference
            for repetition in (0, 1):
                pu_mirt = 0.18 + .005 * repetition
                rows.extend((
                    {"gene_coverage": gene_coverage,
                     "target_coverage": target_coverage, "model": "Fixed baseline",
                     "macro_brier": pu_mirt + difference},
                    {"gene_coverage": gene_coverage,
                     "target_coverage": target_coverage, "model": "PU-MIRT",
                     "macro_brier": pu_mirt},
                    # A stronger per-cell model must not be selected in place of
                    # the predeclared fixed comparator.
                    {"gene_coverage": gene_coverage,
                     "target_coverage": target_coverage, "model": "Other",
                     "macro_brier": pu_mirt - 0.1},
                ))
    figure, axis = plot_gene_target_brier_heatmap(
        pd.DataFrame(rows), baseline_model="Fixed baseline")
    image = axis.images[0]
    np.testing.assert_allclose(np.asarray(image.get_array()), expected)
    assert isinstance(image.norm, TwoSlopeNorm)
    assert image.norm.vcenter == 0.0
    assert np.isclose(abs(image.norm.vmin), image.norm.vmax)
    assert len(figure.axes) == 2  # heatmap plus colorbar
    assert "Fixed baseline Brier" in figure.axes[1].get_ylabel()
    plt.close(figure)


def test_brier_heatmap_marks_incomplete_grid_cells_as_na():
    table = pd.DataFrame([
        {"gene_coverage": .5, "target_coverage": .5,
         "model": "PU", "macro_brier": .20},
        {"gene_coverage": .5, "target_coverage": .5,
         "model": "PU-MIRT", "macro_brier": .18},
        {"gene_coverage": 1., "target_coverage": 1.,
         "model": "PU", "macro_brier": .19},
        {"gene_coverage": 1., "target_coverage": 1.,
         "model": "PU-MIRT", "macro_brier": .18},
    ])
    figure, axis = plot_gene_target_brier_heatmap(table)
    assert sum(text.get_text() == "NA" for text in axis.texts) == 2
    plt.close(figure)


def test_accuracy_sensitivity_scatter_aggregates_one_point_per_model_and_reuses_axis():
    table = pd.DataFrame({
        "model": ["PU", "PU", "PU-MIRT", "PU-MIRT"],
        "macro_auprc": [.30, .34, .35, .37],
        "panel_sensitivity": [.10, .14, .08, .10],
    })
    figure, supplied = plt.subplots()
    returned_figure, axis = plot_accuracy_panel_sensitivity(table, ax=supplied)
    assert returned_figure is figure and axis is supplied
    assert len(axis.collections) == 2
    offsets = np.vstack([collection.get_offsets()[0] for collection in axis.collections])
    np.testing.assert_allclose(offsets, [[.12, .32], [.09, .36]])
    assert {text.get_text() for text in axis.texts} == {"PU", "PU-MIRT"}
    plt.close(figure)


def test_projection_tags_budget_recall_curve_sorts_budgets_and_checks_columns():
    table = pd.DataFrame({
        "budget_fraction": [.20, .01, .05, .20, .01, .05],
        "model": ["PU", "PU", "PU", "PU-MIRT", "PU-MIRT", "PU-MIRT"],
        "amplification_confirmed_recall": [.60, .08, .25, .68, .12, .30],
    })
    figure, axis = plot_projection_tags_budget_recall(table)
    assert [line.get_label() for line in axis.lines] == ["PU", "PU-MIRT"]
    np.testing.assert_allclose(axis.lines[0].get_xdata(), [.01, .05, .20])
    assert axis.get_ylim() == (0.0, 1.0)
    plt.close(figure)

    with pytest.raises(ValueError, match="missing required columns"):
        plot_projection_tags_budget_recall(table.drop(columns="budget_fraction"))


@pytest.mark.parametrize(
    "function,table",
    [
        (plot_retention_auprc,
         pd.DataFrame({"retention": [], "model": [], "macro_auprc": []})),
        (plot_accuracy_panel_sensitivity,
         pd.DataFrame({"model": ["PU"], "macro_auprc": [np.nan],
                       "panel_sensitivity": [.1]})),
    ],
)
def test_measurement_plots_reject_empty_or_nonfinite_inputs(function, table):
    with pytest.raises(ValueError):
        function(table)
