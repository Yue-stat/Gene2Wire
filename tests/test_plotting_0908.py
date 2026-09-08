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
