"""Unit contracts for results-only measurement-degradation V3 analysis."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts" / "notebooks" / "measurement_v3_analysis.py"


def helper():
    spec = importlib.util.spec_from_file_location("measurement_v3_analysis", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_degradation_pair_runs_from_100_to_zero_and_keeps_large_log_loss():
    rows = []
    for repetition in (0, 1):
        for retention, auprc, log_loss in (
            (1.0, .40, .35), (.7, .34, .48), (.4, .28, .72), (.1, .18, 1.25)
        ):
            rows.extend((
                {
                    "repetition": repetition,
                    "positive_retention": retention,
                    "model": "PU",
                    "macro_auprc": auprc + .01 * repetition,
                    "macro_log_loss": log_loss + .02 * repetition,
                    "probability_semantics": "reference_probability",
                },
                {
                    "repetition": repetition,
                    "positive_retention": retention,
                    "model": "SAR-PU",
                    "macro_auprc": auprc - .01,
                    "macro_log_loss": .2,
                    "probability_semantics": "observed_probability",
                },
            ))
    figure, axes = helper().plot_degradation_pair(
        pd.DataFrame(rows),
        x_col="positive_retention",
        x_label="Positive retention",
        key_models=("PU", "SAR-PU"),
    )
    assert tuple(axes[0].get_xlim()) == (1.0, 0.0)
    assert tuple(axes[1].get_xlim()) == (1.0, 0.0)
    assert axes[0].get_ylabel() == "Macro AUPRC ↑"
    assert axes[1].get_ylabel() == "Macro log loss ↓"
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [1.0, .7, .4, .1])
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [.405, .345, .285, .185])
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [.36, .49, .73, 1.26])
    assert [line.get_label() for line in axes[1].lines] == ["PU"]
    plt.close(figure)


def test_projection_budget_metrics_distinguish_macro_and_micro_false_positives():
    summary = pd.DataFrame([{
        "dataset": "Projection-TAGs",
        "repetition": 0,
        "model": "PU-Joint",
        "budget_fraction": .1,
        "selected_candidates": 4,
        "candidate_count": 20,
        "amplification_confirmed_positive_count": 4,
        "recovered_positive_count": 2,
        "amplification_confirmed_recall": .5,
    }])
    per_target = pd.DataFrame([
        {
            "dataset": "Projection-TAGs", "repetition": 0,
            "model": "PU-Joint", "target": "A", "budget_fraction": .1,
            "selected_candidates": 1, "candidate_count": 10,
            "amplification_confirmed_positive_count": 1,
            "recovered_positive_count": 1,
            "amplification_confirmed_recall": 1.0,
        },
        {
            "dataset": "Projection-TAGs", "repetition": 0,
            "model": "PU-Joint", "target": "B", "budget_fraction": .1,
            "selected_candidates": 3, "candidate_count": 10,
            "amplification_confirmed_positive_count": 3,
            "recovered_positive_count": 1,
            "amplification_confirmed_recall": 1 / 3,
        },
    ])
    augmented, target_augmented = helper().augment_projection_budget_metrics(
        summary, per_target=per_target
    )
    row = augmented.iloc[0]
    assert row["amplification_reference_false_positive_count"] == 2
    assert np.isclose(row["micro_amplification_confirmed_recall"], .5)
    assert np.isclose(row["micro_amplification_confirmed_precision"], .5)
    assert np.isclose(row["micro_amplification_reference_fdr"], .5)
    assert np.isclose(row["micro_amplification_reference_fpr"], 2 / 16)
    assert np.isclose(row["amplification_confirmed_precision"], (1 + 1 / 3) / 2)
    assert np.isclose(row["amplification_reference_fdr"], (0 + 2 / 3) / 2)
    assert len(target_augmented) == 2


def test_projection_candidate_posterior_conditions_on_standard_nondetection():
    posterior = helper()._posterior_after_nondetection(
        np.asarray([.5, .8]), np.asarray([.8, .25])
    )
    np.testing.assert_allclose(posterior, [1 / 6, .75])
    assert np.isnan(helper()._posterior_after_nondetection(
        np.asarray([1.0]), np.asarray([1.0])
    )[0])


def test_projection_natural_reanalysis_reuses_npz_and_collapses_panel_seed_copies(tmp_path):
    units = tmp_path / "units"
    prediction = np.asarray([[.8, .2], [.3, .7]])
    reference = np.asarray([[1, 0], [0, 1]], dtype=bool)
    observed = np.zeros((2, 2), dtype=bool)
    measured = np.ones((2, 2), dtype=bool)
    sensitivity = np.full((2, 2), .5)
    for panel_seed in (0, 1):
        directory = units / f"unit_{panel_seed}"
        directory.mkdir(parents=True)
        (directory / "audit.json").write_text(json.dumps({
            "dataset": "Projection-TAGs",
            # Real-data measurement runs expose panel_seed through the generic
            # repetition coordinate and retain data_repetition=0.
            "repetition": panel_seed,
            "data_repetition": 0,
            "panel_seed": panel_seed,
            "outer_fold": 0,
            "mechanism": "natural",
            "condition_roles": "natural_recovery",
        }))
        np.savez_compressed(
            directory / "PU_predictions.npz",
            prediction=prediction,
            reference=reference,
            observed=observed,
            source_measured=measured,
            estimated_sensitivity=sensitivity,
            cell_ids=np.asarray(["c1", "c2"]),
            target_ids=np.asarray(["t1", "t2"]),
        )
    artifacts = SimpleNamespace(
        export_dir=tmp_path,
        manifest={
            "measurement_protocol": {"n_panel_seeds": 2},
            "protocol": {"n_outer_folds": 1},
        },
        tables={"metrics": pd.DataFrame([{
            "model": "PU",
            "mechanism": "natural",
            "condition_roles": "natural_recovery",
            "probability_semantics": "reference_probability",
        }])},
    )
    tables = helper().derive_projection_natural_metrics(artifacts, budgets=(.5,))
    summary = tables["summary"].iloc[0]
    assert summary["candidate_count"] == 4
    assert summary["duplicate_candidates_collapsed"] == 4
    assert summary["source_panel_seed_count"] == 2
    assert summary["candidate_probability"] == "h_after_standard_nondetection"
    assert tables["audit"].iloc[0]["panel_seed_copy_check"] == "identical"
    expected = (1 - .5) * prediction / (1 - .5 * prediction)
    expected_brier = np.mean([
        np.mean((reference[:, column] - expected[:, column]) ** 2)
        for column in range(2)
    ])
    assert np.isclose(summary["macro_brier"], expected_brier)
    budget = tables["budget"].iloc[0]
    assert budget["selected_candidates"] == 2
    assert budget["recovered_positive_count"] == 2
    assert budget["amplification_reference_false_positive_count"] == 0


def test_projection_natural_reanalysis_rejects_a_missing_panel_seed(tmp_path):
    units = tmp_path / "units" / "unit_0"
    units.mkdir(parents=True)
    (units / "audit.json").write_text(json.dumps({
        "dataset": "Projection-TAGs",
        "repetition": 0,
        "data_repetition": 0,
        "panel_seed": 0,
        "outer_fold": 0,
        "mechanism": "natural",
        "condition_roles": "natural_recovery",
    }))
    array = np.asarray([[.5]])
    np.savez_compressed(
        units / "PU_predictions.npz",
        prediction=array,
        reference=np.asarray([[True]]),
        observed=np.asarray([[False]]),
        source_measured=np.asarray([[True]]),
        estimated_sensitivity=array,
        cell_ids=np.asarray(["c1"]),
        target_ids=np.asarray(["t1"]),
    )
    artifacts = SimpleNamespace(
        export_dir=tmp_path,
        manifest={
            "measurement_protocol": {"n_panel_seeds": 2},
            "protocol": {"n_outer_folds": 1},
        },
        tables={"metrics": pd.DataFrame([{
            "model": "PU",
            "mechanism": "natural",
            "condition_roles": "natural_recovery",
            "probability_semantics": "reference_probability",
        }])},
    )
    with np.testing.assert_raises_regex(ValueError, "incomplete"):
        helper().derive_projection_natural_metrics(artifacts, budgets=(.5,))


def _scorecard_rows():
    rows = []
    base = {
        "dataset": "example",
        "sharing_strength": np.nan,
        "evaluation_scope": "native_reference",
        "mechanism": "assay_target_sar",
        "gene_requested_coverage": .7,
        "gene_coverage": .7,
        "target_requested_coverage": .7,
        "target_coverage": .7,
        "positive_retention": .4,
    }
    scenarios = [("full_control", 1., 1., 1.)]
    scenarios.extend(
        ("retention_curve", .7, .7, retention)
        for retention in (1., .7, .4, .1)
    )
    scenarios.extend(
        ("coverage_heatmap", gene, target, .4)
        for gene in (1., .7, .4)
        for target in (1., .7, .4)
    )
    for model, semantics, offset in (
        ("PU", "reference_probability", 0.),
        ("PU-Joint", "reference_probability", .01),
        ("SAR-PU", "observed_probability", .02),
    ):
        for role, gene, target, retention in scenarios:
            rows.append({
                **base,
                "model": model,
                "probability_semantics": semantics,
                "condition_roles": role,
                "gene_requested_coverage": gene,
                "gene_coverage": gene,
                "target_requested_coverage": target,
                "target_coverage": target,
                "positive_retention": retention,
                "macro_auprc": .25 + offset,
                "macro_auroc": .65 + offset,
                "macro_brier": .18 - offset,
                "macro_log_loss": .52 - offset,
            })
        rows.append({
            **base,
            "model": model,
            "probability_semantics": semantics,
            "condition_roles": "matched_uniform_control",
            "mechanism": "scar",
            "macro_auprc": .24 + offset,
            "macro_auroc": .64 + offset,
            "macro_brier": .19 - offset,
            "macro_log_loss": .54 - offset,
        })
    return pd.DataFrame(rows)


def test_funky_table_preserves_model_order_semantics_and_common_direction():
    module = helper()
    scorecard = module.prepare_measurement_funky_table(
        _scorecard_rows(),
        measurement_protocol={
            "anchor_gene_coverage": .7,
            "anchor_target_coverage": .7,
            "anchor_retention": .4,
            "gene_coverages": (1., .7, .4),
            "target_coverages": (1., .7, .4),
            "retentions": (1., .7, .4, .1),
            "include_matched_uniform": True,
        },
    )
    sar_losses = scorecard.loc[
        scorecard["model"].eq("SAR-PU")
        & scorecard["metric"].isin(("macro_brier", "macro_log_loss"))
    ]
    assert sar_losses["value"].isna().all()
    ranking = scorecard.loc[
        scorecard["model"].eq("SAR-PU") & scorecard["metric"].eq("macro_auprc")
    ]
    assert ranking["value"].notna().all()
    coverage_conditions = scorecard.loc[
        scorecard["family"].eq("Gene × target coverage"), "condition"
    ].unique()
    assert len(coverage_conditions) == 9
    direct = scorecard.loc[
        scorecard["condition"].eq("full")
        & scorecard["metric"].eq("macro_brier")
        & scorecard["model"].isin(("PU", "PU-Joint"))
    ].set_index("model")
    assert direct.loc["PU-Joint", "desirability"] > direct.loc["PU", "desirability"]
    figure, axes = module.plot_measurement_funky_summary(
        scorecard, mode="key", key_models=("PU", "PU-Joint", "SAR-PU")
    )
    assert [tick.get_text() for tick in axes[0].get_yticklabels()] == [
        "PU", "PU-Joint", "SAR-PU"
    ]
    assert any(text.get_text() == "NA" for text in axes[0].texts)
    plt.close(figure)


def test_funky_key_and_full_keep_fully_absent_models_as_na_rows():
    module = helper()
    scorecard = module.prepare_measurement_funky_table(
        _scorecard_rows().loc[lambda frame: frame["model"].eq("PU")],
        measurement_protocol={
            "anchor_gene_coverage": .7,
            "anchor_target_coverage": .7,
            "anchor_retention": .4,
            "gene_coverages": (1., .7, .4),
            "target_coverages": (1., .7, .4),
            "retentions": (1., .7, .4, .1),
            "include_matched_uniform": True,
        },
    )
    figure, axes = module.plot_measurement_funky_summary(
        scorecard, mode="key", key_models=("PU", "PU-Joint")
    )
    assert [tick.get_text() for tick in axes[0].get_yticklabels()] == [
        "PU", "PU-Joint"
    ]
    assert sum(text.get_text() == "NA" for text in axes[0].texts) >= 1
    plt.close(figure)

    full_figure, full_axes = module.plot_measurement_funky_summary(
        scorecard, mode="full"
    )
    assert len(full_axes[0].get_yticklabels()) == len(module.FULL_MODEL_ORDER)
    assert len(full_axes[0].get_xticklabels()) == len(axes[0].get_xticklabels())
    plt.close(full_figure)
