"""Block plots preserve evaluation scope, information use and panel controls."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.block_plotting import (
    _block_aggregate,
    _extra_sharing_gain_auprc,
    plot_block_results,
)


def _artifacts(*, natural=False):
    rows = []
    models = ("Logistic", "MIRT", "Joint", "PU", "PU-MIRT", "PU-Joint",
              "Reference-only", "Reference+PU", "Reference+PU-MIRT", "Reference+PU-Joint",
              "RF-observed", "RF-reference", "RF-mixed", "Qiao-ID-logit", "Logistic-rescaled")
    for panel in ("masked", "full"):
        for fraction in (.2, .4, .6, .8):
            for model in models:
                rows.append({"dataset": "Projection_TAGs" if natural else "BARseq_A1",
                             "sharing_strength": np.nan, "analysis": "target_blocks",
                             "mechanism": "natural" if natural else "technical_sar",
                             "loss_rate": np.nan if natural else .8,
                             "calibration_fraction": .2, "calibration_spec": "correct",
                             "training_panel": panel, "block_fraction": fraction,
                             "training_block_fraction": fraction if panel == "masked" else 0.,
                             "evaluation_scope": "blocked", "model": model,
                             "macro_auprc": .30 if panel == "masked" else .40,
                             "macro_log_loss": .45, "macro_brier": .12,
                             "hidden_recall_at_h": 999.})
    return SimpleNamespace(tables={"aggregate": pd.DataFrame(rows)})


def _gain_artifacts():
    artifacts = _artifacts()
    template = artifacts.tables["aggregate"].iloc[0].to_dict()
    rows = []
    for repetition in range(3):
        for outer_fold in range(2):
            for fraction in (.2, .4):
                for model, full_advantage, multiplier in (
                    ("PU", 0., 0.), ("PU-MIRT", .01, .5), ("PU-Joint", .01, 1.)):
                    extra_gain = multiplier * (fraction / 10 + repetition / 100)
                    for panel, baseline_score in (("masked", .30), ("full", .40)):
                        advantage = full_advantage + (extra_gain if panel == "masked" else 0.)
                        rows.append({**template, "model": model, "training_panel": panel,
                                     "training_block_fraction": fraction if panel == "masked" else 0.,
                                     "block_fraction": fraction, "repetition": repetition,
                                     "outer_fold": outer_fold,
                                     "probability_semantics": "p",
                                     "macro_auprc": baseline_score + advantage})
    artifacts.tables["metrics"] = pd.DataFrame(rows)
    return artifacts


def test_block_plots_keep_panel_controls_reference_budgets_and_pdf_display(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    artifacts = _artifacts()
    original = artifacts.tables["aggregate"].copy(deep=True)
    distractors = original.assign(evaluation_scope="observed", macro_auprc=.99)
    artifacts.tables["aggregate"] = pd.concat([original, distractors], ignore_index=True)
    paths = plot_block_results(artifacts, tmp_path)
    assert len(paths) == len(figures) == 4
    assert all(path.suffix == ".pdf" and path.read_bytes().startswith(b"%PDF")
               for path in paths.values())
    assert not list(tmp_path.rglob("*.png"))
    pd.testing.assert_frame_equal(artifacts.tables["aggregate"].iloc[:len(original)], original)
    primary, benchmark, detection_only, same_budget = figures
    assert len(primary.axes) == 6
    for row, value in ((0, .3), (1, .4)):
        lines = primary.axes[3 * row].lines
        assert {line.get_label() for line in lines} == {"PU logistic", "PU-MIRT", "PU-Joint"}
        assert all(np.allclose(line.get_ydata(), value) for line in lines)
        assert all(np.allclose(line.get_xdata(), [.2, .4, .6, .8]) for line in lines)
    assert "Masked training" in primary.axes[0].get_ylabel()
    assert "Full training" in primary.axes[3].get_ylabel()
    for figure in figures:
        assert all("Hidden" not in axis.get_title() for axis in figure.axes)
        assert all(axis.get_xlabel() != "Positive-label loss" for axis in figure.axes)
        assert all("Targets with masked group blocks" == axis.get_xlabel() for axis in figure.axes[3:])
        assert len(figure.axes[0].collections) == 0
    excluded = {"Reference-only logistic", "Reference + PU logistic", "Reference + PU-MIRT",
                "Reference + PU-Joint", "RF (paired references)", "RF (observed + paired references)"}
    detection_labels = {line.get_label() for line in detection_only.axes[0].lines}
    assert not excluded & detection_labels
    assert {"PU logistic", "RF (observed labels)", "Logistic + sensitivity rescaling"} <= detection_labels
    assert {line.get_label() for line in same_budget.axes[0].lines} == {
        "Reference + PU logistic", "RF (observed + paired references)", "Reference + PU-Joint"}
    legend = benchmark.legends[0]
    legend_lines = dict(zip([text.get_text() for text in legend.get_texts()], legend.get_lines()))
    for label in ("PU logistic", "PU-Joint", "Reference + PU logistic", "RF (observed + paired references)"):
        assert legend_lines[label].get_linestyle() == "-"
    for label in ("Logistic", "RF (observed labels)", "Qiao-ID-logit"):
        assert legend_lines[label].get_linestyle() == "--"
    plt.close("all")


def test_natural_labels_still_make_block_fraction_curves(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    paths = plot_block_results(_artifacts(natural=True), tmp_path, include_benchmarks=False)
    assert len(paths) == len(figures) == 1
    assert "natural" in figures[0]._suptitle.get_text()
    assert "positive-label loss" not in figures[0]._suptitle.get_text()
    assert all(np.allclose(line.get_xdata(), [.2, .4, .6, .8]) for line in figures[0].axes[0].lines)


def test_extra_sharing_gain_is_paired_then_gets_repetition_level_se(tmp_path, monkeypatch):
    artifacts = _gain_artifacts()
    summary = _extra_sharing_gain_auprc(artifacts.tables)
    joint = summary.loc[summary["model"].eq("PU-Joint")].sort_values("block_fraction")
    np.testing.assert_allclose(joint["extra_sharing_gain_auprc"], [.03, .05])
    np.testing.assert_allclose(joint["extra_sharing_gain_auprc_se"],
                               np.repeat(.01 / np.sqrt(3), 2))
    assert joint["n_repetitions"].tolist() == [3, 3]

    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    paths = plot_block_results(artifacts, tmp_path, include_benchmarks=False)
    assert len(paths) == len(figures) == 2
    gain_key = next(key for key in paths if key.endswith("extra_sharing_gain_auprc"))
    assert "extra_sharing_gain" in paths[gain_key].parts
    assert paths[gain_key].read_bytes().startswith(b"%PDF")
    gain_figure = figures[-1]
    assert len(gain_figure.axes[0].containers) == 2
    assert "Extra sharing gain" in gain_figure.axes[0].get_ylabel()
    assert "±1 SE" in " ".join(text.get_text() for text in gain_figure.texts)


def test_block_settings_never_average_across_loss_mechanism_or_rho(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    artifacts = _artifacts()
    frame = artifacts.tables["aggregate"]
    frames = [frame.assign(dataset="simulation", sharing_strength=rho, mechanism=mechanism,
                           loss_rate=loss, macro_auprc=.1 + .1 * index)
              for index, (rho, mechanism, loss) in enumerate(
                  ((0., "scar", .0), (0., "scar", .8), (.5, "scar", .8), (.5, "target_sar", .8)))]
    artifacts.tables["aggregate"] = pd.concat(frames, ignore_index=True)
    paths = plot_block_results(artifacts, tmp_path, include_benchmarks=False)
    assert len(paths) == len(figures) == 4
    values = sorted(float(figure.axes[0].lines[0].get_ydata()[0]) for figure in figures)
    np.testing.assert_allclose(values, [.1, .2, .3, .4])


def test_distinct_block_group_modes_and_experiments_stay_separate(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    artifacts = _artifacts()
    frame = artifacts.tables["aggregate"]
    artifacts.tables["aggregate"] = pd.concat([
        frame.assign(experiment="target_block", group_mode="animal", macro_auprc=.1),
        frame.assign(experiment="target_block", group_mode="artificial", macro_auprc=.2),
        frame.assign(experiment="another_protocol", group_mode="artificial", macro_auprc=.3),
    ], ignore_index=True)
    paths = plot_block_results(artifacts, tmp_path, include_benchmarks=False)
    assert len(paths) == len(figures) == 3
    np.testing.assert_allclose(sorted(float(fig.axes[0].lines[0].get_ydata()[0]) for fig in figures),
                               [.1, .2, .3])


def test_missing_fraction_breaks_curve_and_silent_mode_does_not_display(tmp_path, monkeypatch):
    artifacts = _artifacts()
    frame = artifacts.tables["aggregate"]
    artifacts.tables["aggregate"] = frame.loc[~(frame["model"].eq("PU") & frame["block_fraction"].eq(.4))]
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    plot_block_results(artifacts, tmp_path, include_benchmarks=False)
    line = next(line for line in figures[0].axes[0].lines if line.get_label() == "PU logistic")
    np.testing.assert_array_equal(np.isfinite(line.get_ydata()), [True, False, True, True])
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("show=False must not display"))
    plot_block_results(artifacts, tmp_path, show=False, include_benchmarks=False)


def test_block_aggregation_weights_repetitions_equally_and_keeps_panels():
    rows = []
    template = _artifacts().tables["aggregate"].iloc[0].to_dict()
    for panel in ("masked", "full"):
        for rep, folds in ((0, (0, 1, 2)), (1, (0,))):
            for fold in folds:
                rows.append({**template, "training_panel": panel, "repetition": rep,
                             "outer_fold": fold, "macro_auprc": float(rep) + (panel == "full")})
    frame = _block_aggregate({"metrics": pd.DataFrame(rows)})
    assert len(frame) == 2
    np.testing.assert_allclose(frame.sort_values("training_panel")["macro_auprc"], [1.5, .5])


def test_duplicate_block_estimates_and_wrong_coordinates_fail(tmp_path):
    artifacts = _artifacts()
    frame = artifacts.tables["aggregate"]
    artifacts.tables["aggregate"] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate block estimates"):
        plot_block_results(artifacts, tmp_path, show=False)
    artifacts.tables["aggregate"] = frame.assign(block_fraction=-.2)
    with pytest.raises(ValueError, match="block_fraction"):
        plot_block_results(artifacts, tmp_path, show=False)
    artifacts.tables["aggregate"] = frame.drop(columns="evaluation_scope")
    with pytest.raises(ValueError, match="coordinates"):
        plot_block_results(artifacts, tmp_path, show=False)
    assert plot_block_results(SimpleNamespace(tables={}), tmp_path) == {}


def test_zero_block_fraction_is_valid_but_has_no_scored_curve(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    artifacts = _artifacts()
    positive = artifacts.tables["aggregate"]
    zero = positive.loc[positive["block_fraction"].eq(.2)].assign(
        block_fraction=0., macro_auprc=np.nan, macro_log_loss=np.nan, macro_brier=np.nan)
    artifacts.tables["aggregate"] = zero
    assert plot_block_results(artifacts, tmp_path) == {}
    assert figures == []
    artifacts.tables["aggregate"] = pd.concat([zero, positive], ignore_index=True)
    paths = plot_block_results(artifacts, tmp_path, include_benchmarks=False)
    assert len(paths) == len(figures) == 1
    for axis in figures[0].axes:
        for line in axis.lines:
            np.testing.assert_allclose(line.get_xdata(), [.2, .4, .6, .8])
