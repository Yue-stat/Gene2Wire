"""Native incomplete panels have one assay, no manufactured paired references."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments import baselines, pipeline
from gene2wire.experiments.block_design import make_block_folds
from gene2wire.experiments.contracts import ExperimentDataset, FeatureSet
from gene2wire.experiments.evaluation import evaluate_predictions
from gene2wire.experiments.protocol import Settings, fingerprint, scenarios
from gene2wire.models import UnifiedPUModel


def _dataset():
    rng = np.random.default_rng(905)
    groups = np.repeat(("Adult1", "Adult2", "Adult3"), 24)
    x = rng.normal(size=(72, 3))
    panel = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], bool)
    w = np.repeat(panel, 24, axis=0)
    z = (rng.random(w.shape) < .45) & w

    def features(rows, use_location, use_target_features):
        assert not use_location and not use_target_features
        transformed = x - x[rows].mean(axis=0)
        return FeatureSet(transformed, {"gene": (0, 1, 2)}, feature_names=("g0", "g1", "g2"))

    return ExperimentDataset(
        "native_three_animals", z, w, tuple(f"cell_{i}" for i in range(72)),
        ("shared", "Adult1_only", "Adult2_only", "Adult3_only"), features,
        lambda folds, seed: make_block_folds(groups, folds, seed),
        groups={"animal": groups}, metadata={"reference": "observed assay calls"})


def _settings(**kwargs):
    return Settings(supervision_profile="assay_only", paired_fraction=0., n_jobs=1,
                    n_repetitions=1, **kwargs)


def test_profile_plans_no_reference_methods_even_when_old_control_flags_enabled():
    settings = _settings()
    prepared = SimpleNamespace(metadata={}, natural_observed=None)
    plan = pipeline._planned_models(prepared, settings)
    assert {r["model"] for r in plan} == {
        "Logistic", "MIRT", "Joint", "RF-observed", "Qiao-ID-logit", "Qiao-ID-squared"}
    assert len(plan) == 6
    assert {r["mechanism"] for r in plan} == {"assay_only"}
    assert {r["loss_rate"] for r in plan} == {0.}
    assert {r["calibration_fraction"] for r in plan} == {0.}
    assert {r["calibration_spec"] for r in plan} == {"not_available"}
    assert len(Settings().models()) == 6  # Original default remains intact.
    assert fingerprint(settings.scientific_dict()) != fingerprint(
        replace(settings, supervision_profile="paired_reference", paired_fraction=.2).scientific_dict())
    with pytest.raises(ValueError, match="one assay outcome"):
        scenarios(settings, natural=True, simulation=False)
    with pytest.raises(ValueError, match="supervision_profile"):
        Settings(supervision_profile="unknown")


def test_assay_run_has_no_pairing_thinning_or_detector_and_preserves_native_predictions(tmp_path, monkeypatch):
    dataset, settings = _dataset(), _settings()
    fold = dataset.split_builder(3, settings.seed)[0]
    prepared = pipeline._prepare(dataset, fold, settings)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("A native assay-only run must not fabricate detection calibration")

    for name in ("sample_paired_rows", "make_observation_design", "thin_reference",
                 "fit_detection_calibrator", "detection_diagnostics", "evaluate_detection_calibration"):
        monkeypatch.setattr(pipeline, name, forbidden)

    def fake_core(**kwargs):
        calls.append(kwargs)
        assert kwargs["unit_context"]["supervision"] == "observed"
        for role, rows in (("train", fold.train_rows),
                           ("refit", np.sort(np.r_[fold.train_rows, fold.validation_rows]))):
            bundle = kwargs[role]
            assert bundle.Z_reference is None
            assert bundle.metadata["paired_training_row_count"] == 0
            np.testing.assert_array_equal(bundle.W_measured, dataset.measured[rows])
            np.testing.assert_array_equal(bundle.S_observed, dataset.reference[rows])
            np.testing.assert_array_equal(kwargs[f"{role}_exposure"], 1.)
        return SimpleNamespace(models={model.name: SimpleNamespace(
            fitted=SimpleNamespace(config=model),
            latent_probability=np.full((len(fold.test_rows), 4), .4),
            summary=lambda name=model.name: {"model": name}, tuning=SimpleNamespace(trials=[]))
            for model in kwargs["models"]})

    def fake_rf(*args, **kwargs):
        assert kwargs["probability_semantics"] == "observed"
        calls.append("RF-observed")
        return SimpleNamespace(prediction=np.full((len(fold.test_rows), 4), .4),
                               selected_config={}, diagnostics={}, candidate_records=[])

    raw_ranking = np.linspace(-2, 2, len(fold.test_rows) * 4).reshape(-1, 4)

    def fake_qiao(prepared, observed, tuning_e, final_e, settings, context,
                  record, tables, checkpoint_dir, on_progress=None):
        for name in ("Qiao-ID-logit", "Qiao-ID-squared"):
            record(name, np.clip(raw_ranking, 0, 1), "observed", ranking_score=raw_ranking)

    monkeypatch.setattr(pipeline, "run_model_grid", fake_core)
    monkeypatch.setattr(baselines, "fit_baseline", fake_rf)
    monkeypatch.setattr(pipeline, "_run_qiao_controls", fake_qiao)
    tables = pipeline._run_fold(prepared, 0, settings, tmp_path / "checkpoints", tmp_path / "exports", "test")
    assert len(calls) == 2
    assert len(tables["metrics"]) == len(pipeline._planned_models(prepared, settings))
    assert not tables["detection"] and not tables["detection_per_target"] and not tables["detection_reliability"]
    assert not any("detection_" in key for row in tables["metrics"] + tables["per_target"] for key in row)
    assert not any(row["scope"] == "detection" for row in tables["reliability"])
    audit_path = next((tmp_path / "exports").rglob("audit.json"))
    audit = json.loads(audit_path.read_text())
    assert audit["paired_train_cell_ids"] == audit["paired_development_cell_ids"] == []
    assert audit["tuning_detector"] is None and audit["refit_detector"] is None
    assert "biological sensitivity unestimated" in audit["exposure_interpretation"]
    dev = np.sort(np.r_[fold.train_rows, fold.validation_rows])
    expected_prior = ((dataset.reference[dev] * dataset.measured[dev]).sum(axis=0) + .5) / (
        dataset.measured[dev].sum(axis=0) + 1.)
    logistic_targets = [r for r in tables["per_target"] if r["model"] == "Logistic"]
    np.testing.assert_allclose([r["train_reference_prevalence"] for r in logistic_targets], expected_prior)
    for row in tables["metrics"]:
        assert row["n_measured"] == int(dataset.measured[fold.test_rows].sum())
        assert "hidden_recall_at_h" not in row
    with np.load(audit_path.parent / "Logistic_predictions.npz") as saved:
        assert np.isfinite(saved["prediction"][~saved["measured"]]).all()
        assert "estimated_sensitivity" not in saved and "additional_retention" in saved
        np.testing.assert_array_equal(saved["group_ids"], dataset.groups["animal"][fold.test_rows])
    with np.load(audit_path.parent / "Qiao-ID-squared_predictions.npz") as saved:
        np.testing.assert_array_equal(saved["ranking_score"], raw_ranking)


def test_native_metrics_cannot_score_unassayed_placeholders():
    dataset = _dataset()
    p = np.full(dataset.reference.shape, .4)
    first = evaluate_predictions(dataset.reference, dataset.reference, dataset.measured, p,
                                 np.ones_like(p), probability_semantics="observed")
    changed_labels = dataset.reference.copy()
    changed_labels[~dataset.measured] = True
    p[~dataset.measured] = np.nan
    second = evaluate_predictions(changed_labels, changed_labels, dataset.measured, p,
                                  np.ones_like(p), probability_semantics="observed")
    pd.testing.assert_series_equal(pd.Series(first["summary"]), pd.Series(second["summary"]))


def test_native_measured_fit_and_complete_checkpoint_resume(tmp_path):
    dataset = _dataset()
    settings = _settings(run_random_forest=False, run_qiao=False, candidate_budget=3,
                         penalties=(.1,), maxiter=40, retry_maxiter=80, tolerance=1e-4)
    kwargs = dict(checkpoint_dir=tmp_path / "checkpoints", export_dir=tmp_path / "exports", progress=False)
    first = pipeline.run_experiment(dataset, settings, **kwargs)
    assert first.manifest["planned_model_evaluations"] == 9
    assert len(first.tables["metrics"]) == 9
    assert set(first.tables["metrics"]["model"]) == {"Logistic", "MIRT", "Joint"}
    assert first.tables["detection"].empty
    assert "paired_audit" not in first.tables
    with patch.object(UnifiedPUModel, "fit", side_effect=AssertionError("Unexpected refit")):
        again = pipeline.run_experiment(dataset, settings, **kwargs)
    assert again.manifest["run_id"] == first.manifest["run_id"]
    assert again.tables["checkpoint_inventory"]["fully_cached"].all()
    pd.testing.assert_frame_equal(first.tables["aggregate"], again.tables["aggregate"], check_like=True)
