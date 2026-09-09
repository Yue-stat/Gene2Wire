"""Primary-only paper figures, correct independent units, PDF plus display."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.plotting import MODEL_ORDER, _aggregate, plot_results


def _artifacts(*, simulation=False, natural=False, datasets=("SPIDER",), n=5):
    rows, intervals = [], []
    for dataset in datasets:
        for rho in ((0., .5, 1.) if simulation else (np.nan,)):
            for rate in ((np.nan,) if natural else (0., .2, .4, .6, .8)):
                for index, model in enumerate(MODEL_ORDER):
                    row = {"dataset": dataset, "sharing_strength": rho,
                           "analysis": "primary", "mechanism": "natural" if natural else "technical_sar",
                           "calibration_spec": "correct", "calibration_fraction": .2,
                           "loss_rate": rate, "model": model,
                           "macro_auprc": .2 + index * .01,
                           "macro_log_loss": .4 - index * .01,
                           "hidden_recall_at_h": np.nan if rate == 0 else .1 + index * .01,
                           "macro_predicted_prevalence": .15 - index * .005,
                           "macro_reference_prevalence": .15}
                    rows.append(row)
                    for metric in ("macro_auprc", "macro_log_loss", "hidden_recall_at_h"):
                        intervals.append({**row, "metric": metric, "mean": row[metric],
                                          "ci95_low": row[metric] - .01, "ci95_high": row[metric] + .01, "n": n})
    # A sensitivity-analysis endpoint must not leak into the primary curves.
    rows.append({**rows[-1], "analysis": "calibration_misspecification", "macro_auprc": .99})
    return SimpleNamespace(tables={"aggregate": pd.DataFrame(rows),
                                   "simulation_intervals": pd.DataFrame(intervals)},
                           manifest={"uncertainty_unit": "generated_dataset" if simulation else "descriptive_fold_and_mask_repeat"})


def _capture_show(monkeypatch):
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    return figures


def test_simulation_all_three_rhos_pdf_only_display_and_primary_filter(tmp_path, monkeypatch):
    figures = _capture_show(monkeypatch)
    paths = plot_results(_artifacts(simulation=True, datasets=("simulation",)), tmp_path)
    assert len(paths) == 2
    assert len(figures) == 2
    assert all(path.suffix == ".pdf" and path.read_bytes().startswith(b"%PDF") for path in paths.values())
    assert not list(tmp_path.rglob("*.png"))
    primary = figures[0]
    assert len(primary.axes) == 9
    assert [primary.axes[i].get_ylabel() for i in (0, 3, 6)] == [r"Sharing $\rho=0$", r"Sharing $\rho=0.5$", r"Sharing $\rho=1$"]
    for axis in primary.axes:
        assert {line.get_label() for line in axis.lines} == {"Logistic", "PU logistic", "PU-MIRT", "PU-Joint"}
        assert all(np.nanmax(line.get_ydata()) < .5 for line in axis.lines)
    for index in (2, 5, 8):
        assert all(np.all(line.get_xdata() > 0) for line in primary.axes[index].lines)


def test_n_one_simulation_has_no_fabricated_interval(tmp_path, monkeypatch):
    figures = _capture_show(monkeypatch)
    plot_results(_artifacts(simulation=True, datasets=("simulation",), n=1), tmp_path)
    assert all(len(axis.collections) == 0 for axis in figures[0].axes)


def test_real_curves_keep_three_pu_models_and_bar_panels_separate(tmp_path, monkeypatch):
    figures = _capture_show(monkeypatch)
    paths = plot_results(_artifacts(datasets=("BARseq_A1", "BARseq_M1")), tmp_path)
    assert len(paths) == 4
    assert {path.name for path in paths.values()} == {
        "BARseq_A1_results_0908.pdf", "BARseq_A1_probability_0908.pdf",
        "BARseq_M1_results_0908.pdf", "BARseq_M1_probability_0908.pdf"}
    for figure in (figures[0], figures[2]):
        assert len(figure.axes) == 3
        for axis in figure.axes:
            assert {line.get_label() for line in axis.lines} == {"PU logistic", "PU-MIRT", "PU-Joint"}
            assert len(axis.collections) == 0  # No biological CI inferred from masks.


def test_natural_audit_pooled_ratio_and_no_artificial_loss_axis(tmp_path, monkeypatch):
    figures = _capture_show(monkeypatch)
    artifacts = _artifacts(natural=True, datasets=("Projection_TAGs",))
    artifacts.tables["paired_audit"] = pd.DataFrame({
        "target": ["t1", "t2"], "standard_positive_count": [1, 3],
        "reference_positive_count": [2, 10], "relative_detection": [.5, .3]})
    plot_results(artifacts, tmp_path)
    assert len(figures[0].axes) == 4
    assert any("33.3%" in text.get_text() for text in figures[0].axes[0].texts)
    assert not any(axis.get_xlabel() == "Positive-label loss" for figure in figures for axis in figure.axes)


def test_all_nan_hidden_metrics_and_empty_results_are_handled(tmp_path, monkeypatch):
    figures = _capture_show(monkeypatch)
    artifacts = _artifacts()
    artifacts.tables["aggregate"]["hidden_recall_at_h"] = np.nan
    plot_results(artifacts, tmp_path)
    assert any(text.get_text() == "No defined estimates" for text in figures[0].axes[2].texts)
    assert plot_results(SimpleNamespace(tables={}, manifest={}), tmp_path) == {}


def test_raw_metric_fallback_averages_folds_before_repetitions():
    frame = pd.DataFrame({"dataset": ["d"] * 6, "analysis": ["primary"] * 6,
                          "mechanism": ["technical_sar"] * 6, "model": ["PU"] * 6,
                          "loss_rate": [.8] * 6, "outer_fold": [0, 1, 2, 3, 4, 0],
                          "repetition": [0, 0, 0, 0, 0, 1], "macro_auprc": [0, 0, 0, 0, 0, 1]})
    result = _aggregate({"metrics": frame})
    assert result.iloc[0]["macro_auprc"] == pytest.approx(.5)


def test_benchmark_figures_add_every_available_model_without_replacing_paper_plots(tmp_path, monkeypatch):
    from gene2wire.experiments.plotting import plot_benchmark_results
    figures = _capture_show(monkeypatch)
    artifacts = _artifacts()
    primary = artifacts.tables["aggregate"].query("analysis == 'primary'").copy()
    extras = []
    for model in ("RF-observed", "RF-reference", "Prevalence-observed", "Prevalence-reference",
                  "Reference-only", "Reference+PU", "Qiao-squared", "Qiao-logit", "Future baseline"):
        for rate in ((0., .8) if model.startswith(("RF", "Prevalence")) else (.8,)):
            extras.append({**primary.iloc[0].to_dict(), "model": model, "loss_rate": rate})
    artifacts.tables["aggregate"] = pd.concat([primary, pd.DataFrame(extras)], ignore_index=True)
    old_paths = plot_results(artifacts, tmp_path)
    old_files = {key: path.read_bytes() for key, path in old_paths.items()}
    paths = plot_benchmark_results(artifacts, tmp_path)
    assert len(paths) == 1
    assert len(figures) == 3  # Two unchanged paper figures, then the additional diagnostic.
    assert {key: path.read_bytes() for key, path in old_paths.items()} == old_files
    assert all(path.suffix == ".pdf" and path.parent.name == "all_benchmarks" for path in paths.values())
    assert not list(tmp_path.rglob("*.png"))
    benchmark = figures[-1]
    assert len(benchmark.legends[0].get_texts()) == len(MODEL_ORDER) + 9
    legend_labels = {text.get_text() for text in benchmark.legends[0].get_texts()}
    assert {"PU-Joint", "Future baseline", "RF (paired references)", "Reference + PU logistic"} <= legend_labels
    curves = {line.get_label(): line for line in benchmark.axes[0].lines}
    forest = curves["RF (observed labels)"]
    assert forest.get_linestyle() == "None"
    assert np.isfinite(forest.get_ydata()).sum() == 2
    assert np.isnan(forest.get_ydata()[1:-1]).all()
    assert curves["PU-Joint"].get_linestyle() == "-"
    assert len(curves["PU-Joint"].get_ydata()) == 5
    legend = benchmark.legends[0]
    legend_lines = dict(zip([text.get_text() for text in legend.get_texts()], legend.get_lines()))
    assert legend_lines["PU-Joint"].get_linestyle() == "-"
    assert legend_lines["Logistic"].get_linestyle() == "--"
    assert legend_lines["RF (observed labels)"].get_linestyle() == "None"
    assert legend_lines["Reference + PU logistic"].get_linestyle() == "None"
    for label in ("PU-Joint", "Logistic", "RF (observed labels)"):
        assert legend_lines[label].get_marker() == curves[label].get_marker()
        assert legend_lines[label].get_color() == curves[label].get_color()
        assert legend_lines[label].get_linewidth() == curves[label].get_linewidth()


def test_benchmark_legend_uses_visible_segments_across_metrics_and_respects_gaps(tmp_path, monkeypatch):
    from gene2wire.experiments.plotting import plot_benchmark_results
    figures = _capture_show(monkeypatch)
    artifacts = _artifacts()
    frame = artifacts.tables["aggregate"].query("analysis == 'primary'").copy()
    # AUPRC is absent, but the other panels still contain real PU curves.
    frame.loc[frame["model"].eq("PU"), "macro_auprc"] = np.nan
    # Three isolated measured points are not a connected curve.
    isolated = frame.loc[frame["model"].eq("Joint") & frame["loss_rate"].isin([0., .4, .8])].assign(
        model="Isolated baseline")
    artifacts.tables["aggregate"] = pd.concat([frame, isolated], ignore_index=True)
    plot_benchmark_results(artifacts, tmp_path)
    legend = figures[0].legends[0]
    lines = dict(zip([text.get_text() for text in legend.get_texts()], legend.get_lines()))
    assert lines["PU logistic"].get_linestyle() == "-"
    assert lines["Isolated baseline"].get_linestyle() == "None"
    isolated_curve = next(line for line in figures[0].axes[0].lines
                          if line.get_label() == "Isolated baseline")
    np.testing.assert_array_equal(np.isfinite(isolated_curve.get_ydata()), [True, False, True, False, True])


def test_benchmark_controls_are_separate_by_rho_mechanism_fraction_and_spec(tmp_path, monkeypatch):
    from gene2wire.experiments.plotting import plot_benchmark_results
    figures = _capture_show(monkeypatch)
    row = _artifacts().tables["aggregate"].query("model == 'PU-Joint'").iloc[0].to_dict()
    rows = [
        {**row, "sharing_strength": 0., "loss_rate": .8, "macro_auprc": .1},
        {**row, "sharing_strength": .5, "loss_rate": .8, "macro_auprc": .2},
        {**row, "sharing_strength": 1., "loss_rate": .8, "macro_auprc": .3},
        {**row, "sharing_strength": 1., "loss_rate": .8, "macro_auprc": .4,
         "analysis": "mechanism", "mechanism": "target_sar"},
        {**row, "sharing_strength": 1., "loss_rate": .8, "macro_auprc": .5,
         "analysis": "calibration_size", "calibration_fraction": .1},
        {**row, "sharing_strength": 1., "loss_rate": .8, "macro_auprc": .6,
         "analysis": "calibration_misspecification", "calibration_spec": "omit_technical"},
    ]
    artifacts = SimpleNamespace(tables={"aggregate": pd.DataFrame(rows)}, manifest={})
    paths = plot_benchmark_results(artifacts, tmp_path)
    assert len(paths) == len(figures) == 6
    assert {round(float(figure.axes[0].lines[0].get_xdata()[0]), 5) for figure in figures} == {.1, .2, .3, .4, .5, .6}
    assert all(figure.axes[0].get_xlabel() != "Positive-label loss" for figure in figures)


def test_benchmark_natural_all_models_no_loss_axis_and_undefined_scores(tmp_path, monkeypatch):
    from gene2wire.experiments.plotting import plot_benchmark_results
    figures = _capture_show(monkeypatch)
    artifacts = _artifacts(natural=True, datasets=("Projection_TAGs",))
    artifacts.tables["aggregate"] = artifacts.tables["aggregate"].query("analysis == 'primary'").copy()
    artifacts.tables["aggregate"]["hidden_recall_at_h"] = np.nan
    paths = plot_benchmark_results(artifacts, tmp_path)
    assert len(paths) == 1
    assert all(axis.get_xlabel() != "Positive-label loss" for axis in figures[0].axes)
    assert [tick.get_text() for tick in figures[0].axes[2].get_yticklabels()] == [
        "Logistic", "MIRT", "Joint", "PU logistic", "PU-MIRT", "PU-Joint"]
    assert any(text.get_text() == "No defined estimates" for text in figures[0].axes[2].texts)


def test_benchmark_aggregation_respects_repeat_units_and_refuses_mixed_semantics(tmp_path):
    from gene2wire.experiments.plotting import _benchmark_aggregate, plot_benchmark_results
    metrics = pd.DataFrame({"dataset": ["d"] * 6, "analysis": ["mechanism"] * 6,
        "mechanism": ["target_sar"] * 6, "model": ["Logistic-rescaled"] * 6,
        "loss_rate": [.8] * 6, "outer_fold": [0, 1, 2, 3, 4, 0],
        "repetition": [0, 0, 0, 0, 0, 1], "macro_auprc": [0, 0, 0, 0, 0, 1]})
    summary = _benchmark_aggregate({"metrics": metrics})
    assert summary.iloc[0]["macro_auprc"] == pytest.approx(.5)
    assert summary.iloc[0]["analysis"] == "mechanism"
    mixed = pd.concat([summary.assign(probability_semantics="reference"),
                       summary.assign(probability_semantics="observed")], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate model/loss estimates"):
        plot_benchmark_results(SimpleNamespace(tables={"aggregate": mixed}), tmp_path,
                               show=False, metrics=("macro_auprc",))


def test_benchmark_curves_accept_readonly_numpy_views_without_mutating_results(tmp_path, monkeypatch):
    """Emulate the immutable arrays returned by pandas copy-on-write modes."""
    from gene2wire.experiments.plotting import plot_benchmark_results
    figures = _capture_show(monkeypatch)
    artifacts = _artifacts()
    frame = artifacts.tables["aggregate"].query("analysis == 'primary'").copy()
    frame.loc[frame["model"].eq("PU") & frame["loss_rate"].eq(.4), "macro_auprc"] = np.inf
    artifacts.tables["aggregate"] = frame
    original = frame.copy(deep=True)
    to_numpy = pd.Series.to_numpy

    def readonly(series, *args, **kwargs):
        result = to_numpy(series, *args, **kwargs).copy()
        result.setflags(write=False)
        return result

    monkeypatch.setattr(pd.Series, "to_numpy", readonly)
    paths = plot_benchmark_results(artifacts, tmp_path)
    assert len(paths) == 1
    pu = next(line for line in figures[0].axes[0].lines if line.get_label() == "PU logistic")
    assert np.isnan(pu.get_ydata()[2])
    assert np.isfinite(pu.get_ydata()).sum() == 4
    pd.testing.assert_frame_equal(frame, original)


def _information_artifacts(*, simulation=False, natural=False):
    artifacts = _artifacts(simulation=simulation, natural=natural,
                           datasets=("simulation" if simulation else "Projection_TAGs" if natural else "BARseq_M1",))
    base = artifacts.tables["aggregate"].query("analysis == 'primary' and model == 'PU'")
    endpoint = base.loc[base["loss_rate"].isna() | base["loss_rate"].eq(.8)]
    extras = [endpoint.assign(model="Reference-only", macro_auprc=.31),
              endpoint.assign(model="Reference+PU", macro_auprc=.32)]
    artifacts.tables["aggregate"] = pd.concat([artifacts.tables["aggregate"], *extras], ignore_index=True)
    return artifacts


def test_information_budget_compares_three_independent_arms_at_80_percent(tmp_path, monkeypatch):
    from gene2wire.experiments.plotting import plot_information_budget_results
    figures = _capture_show(monkeypatch)
    artifacts = _information_artifacts()
    # A low-loss PU score or another structure must never enter this contrast.
    frame = artifacts.tables["aggregate"]
    frame.loc[frame["loss_rate"].eq(.6), "macro_auprc"] = .99
    paths = plot_information_budget_results(artifacts, tmp_path)
    assert len(paths) == len(figures) == 1
    assert all(path.parent.name == "information_budget" and path.suffix == ".pdf"
               and path.read_bytes().startswith(b"%PDF") for path in paths.values())
    assert len(figures[0].axes) == 3
    axis = figures[0].axes[0]
    assert [tick.get_text() for tick in axis.get_yticklabels()] == [
        "Reference-only", "Calibrated PU", "Reference + PU"]
    assert [line.get_label() for line in axis.lines] == [
        "Reference-only", "Calibrated PU", "Reference + PU"]
    assert [float(line.get_xdata()[0]) for line in axis.lines] == pytest.approx([.31, .23, .32])
    assert "80%" in figures[0]._suptitle.get_text()
    assert not list(tmp_path.rglob("*.png"))


def test_information_budget_separates_simulation_rhos_and_paired_fractions(tmp_path, monkeypatch):
    from gene2wire.experiments.plotting import plot_information_budget_results
    figures = _capture_show(monkeypatch)
    artifacts = _information_artifacts(simulation=True)
    frame = artifacts.tables["aggregate"]
    extra = frame.loc[frame["analysis"].eq("primary") & frame["loss_rate"].eq(.8)].assign(
        calibration_fraction=.4, macro_auprc=.7)
    artifacts.tables["aggregate"] = pd.concat([frame, extra], ignore_index=True)
    paths = plot_information_budget_results(artifacts, tmp_path)
    assert len(paths) == len(figures) == 6
    assert sum("paired = 40%" in figure._suptitle.get_text() for figure in figures) == 3
    for figure in figures:
        if "paired = 40%" in figure._suptitle.get_text():
            assert all(float(line.get_xdata()[0]) == .7 for line in figure.axes[0].lines)


def test_information_budget_natural_missing_arms_and_undefined_metrics_are_explicit(tmp_path, monkeypatch):
    from gene2wire.experiments.plotting import plot_information_budget_results
    figures = _capture_show(monkeypatch)
    artifacts = _information_artifacts(natural=True)
    frame = artifacts.tables["aggregate"]
    artifacts.tables["aggregate"] = frame.loc[frame["model"].ne("Reference+PU")].assign(
        hidden_recall_at_h=np.nan)
    paths = plot_information_budget_results(artifacts, tmp_path)
    assert len(paths) == 1
    assert "natural paired evaluation" in figures[0]._suptitle.get_text()
    assert "80%" not in figures[0]._suptitle.get_text()
    assert any(text.get_text() == "Not run" for text in figures[0].axes[0].texts)
    assert len(figures[0].axes[0].lines) == 2
    assert len(figures[0].axes[2].lines) == 0
    assert sum(text.get_text() == "Undefined / unavailable" for text in figures[0].axes[2].texts) == 2


def test_information_budget_never_substitutes_a_different_loss_or_mixes_semantics(tmp_path):
    from gene2wire.experiments.plotting import plot_information_budget_results
    artifacts = _information_artifacts()
    frame = artifacts.tables["aggregate"]
    artifacts.tables["aggregate"] = frame.loc[frame["loss_rate"].lt(.8)]
    assert plot_information_budget_results(artifacts, tmp_path, show=False) == {}
    duplicate = frame.loc[frame["analysis"].eq("primary") & frame["model"].eq("PU")
                          & frame["loss_rate"].eq(.8)].assign(probability_semantics="other")
    artifacts.tables["aggregate"] = pd.concat([frame, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate model estimates"):
        plot_information_budget_results(artifacts, tmp_path, show=False)
