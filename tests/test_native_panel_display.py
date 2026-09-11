"""Natural panels separate measured-assay evaluation from unvalidated predictions."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.native_panel_plotting import plot_native_panel_results
from gene2wire.experiments.reporting import display_diagnostics, notebook_summaries


def artifacts(tmp_path):
    models = ("Logistic", "MIRT", "Joint", "RF-observed")
    aggregate = pd.DataFrame([{"dataset": "SPIDER-Seq Adult", "mechanism": "assay_only",
        "analysis": "primary", "model": model, "macro_auprc": .3 + .01*index,
        "macro_log_loss": .4 - .01*index, "macro_brier": .1, "macro_auroc": .75,
        "hidden_recall_at_h": 999.} for index, model in enumerate(models)])
    per_animal = pd.concat([aggregate.assign(animal=animal) for animal in ("Adult1", "Adult2")],
                          ignore_index=True)
    panel = pd.DataFrame([{"animal": animal, "target": target,
        "measured": str((animal == "Adult1" and target != "C") or
                        (animal == "Adult2" and target != "A")),
        "n_cells": 20, "n_positive": 4}
        for animal in ("Adult1", "Adult2") for target in ("A", "B", "C")])
    predictions = pd.DataFrame([{"animal": animal, "target": target,
        "model": model, "mean_prediction": .2 + .05*index, "n_cells": 20,
        "mean_repeat_sd": .01} for index, model in enumerate(models[:3])
        for animal, target in (("Adult1", "C"), ("Adult2", "A"))])
    selected = pd.DataFrame([{"dataset": "SPIDER-Seq Adult", "analysis": "primary",
        "mechanism": "assay_only", "model": model, "repetition": rep, "outer_fold": fold,
        "kind": "direct" if model == "Logistic" else "lowrank", "rank": 0 if model == "Logistic" else 2,
        "residual_l2": .01, "shared_l2": .001, "final_converged": True}
        for model in models[:3] for rep in range(2) for fold in range(3)])
    return SimpleNamespace(tables={"aggregate": aggregate, "metrics": aggregate.copy(),
        "selected": selected, "failures": pd.DataFrame(), "native_panel": panel,
        "per_animal_aggregate": per_animal, "unassayed_summary": predictions},
        export_dir=tmp_path, manifest={"protocol": {"supervision_profile": "assay_only",
            "n_repetitions": 2, "paired_fraction": .2}})


def test_native_figures_display_pdf_only_and_never_score_unassayed_entries(tmp_path, monkeypatch):
    result = artifacts(tmp_path)
    before = {name: frame.copy(deep=True) for name, frame in result.tables.items()}
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    paths = plot_native_panel_results(result, tmp_path)
    assert len(paths) == len(figures) == 4
    assert all(path.suffix == ".pdf" and path.read_bytes().startswith(b"%PDF") for path in paths.values())
    assert not list(tmp_path.rglob("*.png"))
    assert not plt.get_fignums()
    for name, original in before.items():
        pd.testing.assert_frame_equal(result.tables[name], original)
    availability, measured, per_animal, predictions = figures
    np.testing.assert_array_equal(availability.axes[0].images[0].get_array(), [[1., 1., 0.], [0., 1., 1.]])
    assert len(measured.axes) == 3
    assert len(per_animal.axes) == 2
    assert "unvalidated" in predictions._suptitle.get_text()
    assert "within-animal" in per_animal._suptitle.get_text()
    for figure in figures:
        assert all("Hidden" not in axis.get_title() and "PU" not in axis.get_title()
                   for axis in figure.axes)
    for index, axis in enumerate(predictions.axes[:3]):
        values = axis.images[1].get_array()
        assert values.count() == 2
        assert float(values[0, 2]) == pytest.approx(.2 + .05*index)
        assert values.mask[0, 0]


def test_native_figures_do_not_fill_missing_predictions_or_metrics(tmp_path, monkeypatch):
    result = artifacts(tmp_path)
    result.tables["unassayed_summary"] = pd.DataFrame()
    result.tables["aggregate"]["macro_brier"] = np.nan
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    paths = plot_native_panel_results(result, tmp_path, include_auroc=True)
    assert "unassayed_predictions" not in paths
    assert len(figures[1].axes) == 3  # AUPRC / LL / AUROC; Brier stays absent.
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("show=False must not display"))
    plot_native_panel_results(result, tmp_path, show=False)
    assert plot_native_panel_results(SimpleNamespace(tables={}), tmp_path, show=False) == {}


def test_native_figures_reject_duplicate_conditions_and_wrong_semantics(tmp_path):
    result = artifacts(tmp_path)
    result.tables["aggregate"] = pd.concat([result.tables["aggregate"], result.tables["aggregate"].iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate measured aggregate"):
        plot_native_panel_results(result, tmp_path, show=False)
    result = artifacts(tmp_path)
    result.tables["aggregate"]["mechanism"] = "technical_sar"
    with pytest.raises(ValueError, match="assay_only"):
        plot_native_panel_results(result, tmp_path, show=False)
    result = artifacts(tmp_path)
    result.tables["unassayed_summary"].loc[0, "mean_prediction"] = 2.
    with pytest.raises(ValueError, match="probabilities"):
        plot_native_panel_results(result, tmp_path, show=False)


def test_native_compact_reports_show_observed_structure_choices_without_pu_claims(tmp_path, capsys):
    result = artifacts(tmp_path)
    summaries = notebook_summaries(result.tables)
    assert "pu_curves" not in summaries
    assert "per_animal_measured_metrics" in summaries
    choices = summaries["selected_hyperparameters_by_repetition"]
    assert set(choices.model) == {"Logistic", "MIRT", "Joint"}
    assert len(choices) == 6
    assert "K=2" in choices.loc[choices.model.eq("Joint"), "fold_configurations"].iloc[0]
    assert "hidden_recall_at_h" not in summaries["primary_endpoint_metrics"]
    shown = []
    display_diagnostics(result, display_fn=shown.append)
    output = capsys.readouterr().out
    assert "supervision_profile" in output
    assert "paired_fraction" not in output
    assert "PU curves" not in output
    assert "No paired reference or detector calibration is used" in output
    assert "unvalidated assay-score extrapolations" in output
    assert len(shown) <= 8 and sum(map(len, shown)) <= 300


def test_native_missing_columns_warn_instead_of_inventing_heatmaps(tmp_path):
    result = artifacts(tmp_path)
    result.tables["native_panel"] = result.tables["native_panel"].drop(columns="measured")
    with pytest.warns(UserWarning, match="missing columns"):
        paths = plot_native_panel_results(result, tmp_path, show=False)
    assert "native_panel_coverage" not in paths and "unassayed_predictions" not in paths
