"""Scientific contracts for measurement-degradation derived tables."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.measurement_experiment import (
    _add_heatmap_contrasts,
    _add_panel_stability,
    _add_projection_budget_recall,
    _information_access,
)
from gene2wire.experiments.pipeline import Artifacts, slug


def _artifacts(tmp_path, *, metrics=None, tuning=None, manifest=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return Artifacts(
        tables={
            "metrics": pd.DataFrame() if metrics is None else metrics,
            "tuning": pd.DataFrame() if tuning is None else tuning,
        },
        export_dir=tmp_path,
        manifest={} if manifest is None else manifest,
    )


def _write_unit(export_dir, unit, context, model, **arrays):
    destination = export_dir / "units" / unit
    destination.mkdir(parents=True)
    (destination / "audit.json").write_text(json.dumps(context))
    np.savez(destination / f"{slug(model)}_predictions.npz", **arrays)


def test_heatmap_contrast_preserves_scientific_coordinates(tmp_path):
    tuning_rows = []
    metric_rows = []
    conditions = (
        ("data-a", 0.0, "assay_target_sar", 0.50, 0.30, 0.20),
        ("data-b", 1.0, "scar", 0.25, 0.50, 0.10),
    )
    for dataset, sharing, mechanism, retention, baseline_brier, mirt_brier in conditions:
        for model, loss in (("PU", 0.1), ("PU-Joint", 0.2)):
            tuning_rows.append({
                "dataset": dataset, "sharing_strength": sharing,
                "mechanism": mechanism, "panel_seed": 0, "repetition": 0,
                "outer_fold": 0, "gene_requested_coverage": 2 / 3,
                "gene_coverage": 0.67, "target_requested_coverage": 2 / 3,
                "target_coverage": 0.67, "positive_retention": retention,
                "condition_roles": "coverage_heatmap", "model": model,
                "validation_loss": loss,
            })
        for model, brier in (("PU", baseline_brier), ("PU-MIRT", mirt_brier)):
            metric_rows.append({
                "dataset": dataset, "sharing_strength": sharing,
                "experiment": "measurement_degradation", "analysis": "primary",
                "mechanism": mechanism, "loss_rate": 1 - retention,
                "positive_retention": retention, "calibration_fraction": 0.2,
                "calibration_spec": "correct", "evaluation_scope": "native_reference",
                "condition_roles": "coverage_heatmap",
                "gene_requested_coverage": 2 / 3, "gene_coverage": 0.67,
                "gene_overlap": 0.5, "gene_panel_size": 4,
                "gene_union_coverage": 1.0,
                "target_requested_coverage": 2 / 3, "target_coverage": 0.67,
                "target_overlap": 0.5, "target_panel_size": 4,
                "target_union_coverage": 1.0,
                "model": model, "probability_semantics": "reference",
                "macro_brier": brier,
            })
    artifacts = _artifacts(
        tmp_path, metrics=pd.DataFrame(metric_rows), tuning=pd.DataFrame(tuning_rows))

    _add_heatmap_contrasts(artifacts)

    selection = artifacts.tables["heatmap_baseline_selection"]
    assert selection.loc[selection["selected"], "model"].tolist() == ["PU"]
    contrast = artifacts.tables["heatmap_contrasts"].sort_values("dataset")
    assert contrast["dataset"].tolist() == ["data-a", "data-b"]
    assert contrast["sharing_strength"].tolist() == [0.0, 1.0]
    assert contrast["mechanism"].tolist() == ["assay_target_sar", "scar"]
    assert contrast["positive_retention"].tolist() == [0.5, 0.25]
    np.testing.assert_allclose(contrast["brier_contrast"], [0.1, 0.4])
    assert contrast["baseline_probability_semantics"].eq("reference").all()


def test_panel_stability_retains_anchor_coordinates_and_semantics(tmp_path):
    metrics = []
    base_context = {
        "dataset": "real", "sharing_strength": None, "outer_fold": 0,
        "gene_requested_coverage": 2 / 3, "gene_coverage": 0.67,
        "target_requested_coverage": 2 / 3, "target_coverage": 0.67,
        "positive_retention": 0.5, "mechanism": "assay_target_sar",
        "condition_roles": "coverage_heatmap", "repetition": 0,
    }
    for panel_seed, prediction, auprc in (
        (0, np.array([[0.1], [0.3]]), 0.30),
        (1, np.array([[0.3], [0.7]]), 0.50),
    ):
        context = {**base_context, "panel_seed": panel_seed,
                   "repetition": panel_seed}
        _write_unit(
            tmp_path, f"panel-{panel_seed}", context, "PU",
            prediction=prediction,
            source_measured=np.ones_like(prediction, dtype=bool),
            cell_ids=np.array(["c0", "c1"]), target_ids=np.array(["t0"]),
        )
        metrics.append({
            **context, "model": "PU", "evaluation_scope": "native_reference",
            "probability_semantics": "reference", "macro_auprc": auprc,
        })
    artifacts = _artifacts(tmp_path, metrics=pd.DataFrame(metrics))

    _add_panel_stability(artifacts)

    result = artifacts.tables["panel_stability"]
    assert len(result) == 1
    row = result.iloc[0]
    assert np.isclose(row["gene_requested_coverage"], 2 / 3)
    assert np.isclose(row["target_coverage"], 0.67)
    assert row["positive_retention"] == 0.5
    assert row["mechanism"] == "assay_target_sar"
    assert row["probability_semantics"] == "reference"
    assert bool(row["is_reference_probability"])
    assert np.isclose(row["panel_sensitivity"], 0.15)
    assert np.isclose(row["macro_auprc"], 0.40)
    assert row["n_panel_seeds"] == 2
    assert row["n_common_native_pairs"] == 2


def test_projection_recovery_pools_oof_and_macro_averages_per_target(tmp_path):
    metrics = pd.DataFrame({"model": ["PU"]})
    contexts = [
        {"dataset": "projection", "repetition": 0, "outer_fold": 0,
         "mechanism": "natural", "condition_roles": "natural_recovery"},
        {"dataset": "projection", "repetition": 0, "outer_fold": 1,
         "mechanism": "natural", "condition_roles": "natural_recovery"},
    ]
    references = (
        np.array([[0, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [0, 0]], dtype=bool),
    )
    rankings = (
        np.array([[0.3, 0.3], [0.2, 0.2]]),
        np.array([[0.99, 0.1], [0.1, 0.99]]),
    )
    # Deliberately make probability rank differently; the raw ranking_score
    # must drive the recovery curve.
    predictions = (
        np.array([[0.9, 0.9], [0.8, 0.8]]),
        np.array([[0.1, 0.7], [0.99, 0.1]]),
    )
    for fold, context in enumerate(contexts):
        _write_unit(
            tmp_path, f"natural-{fold}", context, "PU",
            prediction=predictions[fold], ranking_score=rankings[fold],
            reference=references[fold], observed=np.zeros((2, 2), dtype=bool),
            source_measured=np.ones((2, 2), dtype=bool),
            cell_ids=np.array([f"c{2 * fold}", f"c{2 * fold + 1}"]),
            target_ids=np.array(["t0", "t1"]),
        )
    artifacts = _artifacts(tmp_path, metrics=metrics)

    _add_projection_budget_recall(artifacts)

    result = artifacts.tables["projection_budget_recall"]
    assert len(result) == 4
    assert "outer_fold" not in result
    row = result.loc[np.isclose(result["budget_fraction"], 0.20)].iloc[0]
    assert row["candidate_count"] == 8
    assert row["selected_candidates"] == 2  # one 20% allocation per target
    assert row["amplification_confirmed_positive_count"] == 4
    assert row["recovered_positive_count"] == 1
    assert np.isclose(row["amplification_confirmed_recall"], 0.5)
    assert np.isclose(row["micro_amplification_confirmed_recall"], 0.25)
    assert row["n_targets"] == 2 and row["n_evaluable_targets"] == 2
    assert row["ranking_source"] == "ranking_score"
    assert row["aggregation"] == "pooled_oof_within_repetition"

    per_target = artifacts.tables["projection_budget_recall_per_target"]
    selected = per_target.loc[np.isclose(per_target["budget_fraction"], 0.20)]
    assert set(selected["target"]) == {"t0", "t1"}
    assert selected.set_index("target")["amplification_confirmed_recall"].to_dict() == {
        "t0": 1.0, "t1": 0.0,
    }


def test_projection_recovery_rejects_duplicate_oof_pairs(tmp_path):
    metrics = pd.DataFrame({"model": ["PU"]})
    for fold in (0, 1):
        context = {
            "dataset": "projection", "repetition": 0, "outer_fold": fold,
            "mechanism": "natural", "condition_roles": "natural_recovery",
        }
        _write_unit(
            tmp_path, f"duplicate-{fold}", context, "PU",
            prediction=np.array([[0.5]]), reference=np.array([[True]]),
            observed=np.array([[False]]), source_measured=np.array([[True]]),
            cell_ids=np.array(["same-cell"]), target_ids=np.array(["same-target"]),
        )
    artifacts = _artifacts(tmp_path, metrics=metrics)

    with pytest.raises(ValueError, match="multiple outer folds"):
        _add_projection_budget_recall(artifacts)


def test_new_comparator_information_access_matches_implemented_target_inputs():
    table = _information_access({}, use_target_features=True).set_index("model")
    assert table.loc["Inductive-PU-MC", "target_input"] == "declared_target_features"
    assert table.loc["GenEML-adapted", "target_input"] == "target_identity"
    assert table.loc["SAR-PU", "target_input"] == "target_identity"
