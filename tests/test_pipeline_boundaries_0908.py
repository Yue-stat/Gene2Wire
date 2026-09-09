"""Instrument pipeline information boundaries without running model searches."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from gene2wire.experiments import baselines, pipeline
from gene2wire.experiments.contracts import ExperimentDataset, FeatureSet, Fold
from gene2wire.experiments.observation import thin_reference as actual_thin
from gene2wire.experiments.protocol import Settings


def _dataset(natural=False):
    n = 60
    reference = np.ones((n, 2), dtype=bool)
    measured = np.ones_like(reference)
    measured[::4, 1] = False
    reference &= measured
    fold = Fold(0, np.arange(30), np.arange(30, 45), np.arange(45, 60))
    built_with = []

    def features(rows, use_location, use_target_features):
        built_with.append(np.array(rows))
        raw = np.column_stack((np.arange(n) / n, np.sin(np.arange(n))))
        x = raw - raw[rows].mean(axis=0)
        return FeatureSet(x, {"gene": (0, 1)}, feature_names=("g0", "g1"))

    data = ExperimentDataset(
        "boundary_example", reference, measured,
        tuple(f"cell_{i}" for i in range(n)), ("t0", "t1"),
        features, lambda count, seed: [fold],
        natural_observed=np.zeros_like(reference) if natural else None,
        technical_score=np.linspace(-1, 1, n),
    )
    return data, fold, built_with


def _instrument(monkeypatch):
    calls = {"detectors": [], "core": [], "baselines": [], "thinning": []}

    def detector(observed, reference, measured, paired_rows, **kwargs):
        paired = np.array(paired_rows)
        outside = np.ones(len(reference), bool)
        outside[paired] = False
        assert np.isnan(reference[outside]).all()
        assert np.isfinite(reference[paired]).all()
        value = .2 + np.sum(reference[paired] * measured[paired]) / 200
        calls["detectors"].append({"paired": paired, "reference": reference.copy(),
                                   "value": value, "kwargs": kwargs})
        return SimpleNamespace(
            predict=lambda n_cells, **unused: np.full((n_cells, observed.shape[1]), value),
            to_dict=lambda: {"fake_detector": True, "value": value},
        )

    def thin(reference, measured, train_rows, loss_rate, mechanism, design):
        result = actual_thin(reference, measured, train_rows, loss_rate, mechanism, design)
        calls["thinning"].append({"train": np.array(train_rows), "result": result})
        return result

    def core(**kwargs):
        calls["core"].append(kwargs)
        for key in ("train", "validation", "refit"):
            assert kwargs[key].Z_reference is None
            assert kwargs[key].reference_mask is None
        records = {}
        for model in kwargs["models"]:
            records[model.name] = SimpleNamespace(
                fitted=SimpleNamespace(config=model),
                latent_probability=np.full((len(kwargs["test_X"]), kwargs["train"].n_targets), .3),
                summary=lambda name=model.name: {"model": name},
                tuning=SimpleNamespace(trials=[]),
            )
        return SimpleNamespace(models=records)

    def baseline(*args, **kwargs):
        calls["baselines"].append((args, kwargs))
        return SimpleNamespace(prediction=np.full((len(args[7]), args[1].shape[1]), .3),
                               selected_config={}, candidate_records=[], diagnostics={})

    def evaluation(*args, **kwargs):
        return {"summary": {"macro_auprc": .5}, "per_target": [],
                "reliability": [], "scores": {}}

    monkeypatch.setattr(pipeline, "fit_detection_calibrator", detector)
    monkeypatch.setattr(pipeline, "thin_reference", thin)
    monkeypatch.setattr(pipeline, "run_model_grid", core)
    monkeypatch.setattr(pipeline, "evaluate_predictions", evaluation)
    monkeypatch.setattr(baselines, "fit_baseline", baseline)
    return calls


def _settings():
    return Settings(n_jobs=1, n_repetitions=1, loss_rates=(.8,),
                    run_random_forest=False, run_mechanism_controls=False,
                    run_calibration_controls=False, run_qiao=False)


def test_pipeline_compiles_disjoint_information_views_and_reuses_fold_local_masks(tmp_path, monkeypatch):
    data, fold, built_with = _dataset()
    settings = _settings()
    prepared = pipeline._prepare(data, fold, settings)
    np.testing.assert_array_equal(built_with[0], fold.train_rows)
    np.testing.assert_array_equal(built_with[1], np.arange(45))
    calls = _instrument(monkeypatch)
    pipeline._run_fold(prepared, 0, settings, tmp_path / "checkpoints", tmp_path / "exports", "test-source")
    assert len(calls["thinning"]) == 1  # No regeneration on development refit.
    np.testing.assert_array_equal(calls["thinning"][0]["train"], fold.train_rows)
    tuning, final = calls["detectors"]
    assert len(final["paired"]) == int(.2 * 45)
    np.testing.assert_array_equal(tuning["paired"], np.intersect1d(final["paired"], fold.train_rows))
    assert not np.intersect1d(final["paired"], fold.test_rows).size
    assert set(call["unit_context"]["supervision"] for call in calls["core"]) == {
        "calibrated_pu", "reference_only", "reference_plus_pu",
    }
    observed = calls["thinning"][0]["result"].observed
    for call in calls["core"]:
        np.testing.assert_array_equal(call["validation"].S_observed, observed[fold.validation_rows])
        np.testing.assert_array_equal(call["validation_exposure"], tuning["value"])
        np.testing.assert_array_equal(call["test_exposure"], final["value"])
        for key, rows, paired in (("train", fold.train_rows, tuning["paired"]),
                                  ("refit", np.arange(45), final["paired"])):
            bundle = call[key]
            paired_local = np.isin(rows, paired)[:, None]
            role = call["unit_context"]["supervision"]
            if role == "reference_only":
                np.testing.assert_array_equal(bundle.W_measured, data.measured[rows] & paired_local)
            else:
                np.testing.assert_array_equal(bundle.W_measured, data.measured[rows])
            expected_labels = observed[rows].copy()
            if role in ("reference_only", "reference_plus_pu"):
                expected_labels[np.isin(rows, paired)] = data.reference[paired]
            expected_labels &= bundle.W_measured
            np.testing.assert_array_equal(bundle.S_observed, expected_labels)
    # Both prevalence baseline roles are retained when RF itself is disabled.
    assert {kw["probability_semantics"] for _, kw in calls["baselines"]} == {"observed", "reference"}
    for args, kwargs in calls["baselines"]:
        np.testing.assert_array_equal(args[4], observed[fold.validation_rows])
        np.testing.assert_array_equal(args[6], tuning["value"])
        if kwargs["probability_semantics"] == "reference":
            np.testing.assert_array_equal(args[2], data.measured[fold.train_rows] & np.isin(fold.train_rows, tuning["paired"])[:, None])


def test_natural_validation_references_affect_refit_but_not_tuning_views(tmp_path, monkeypatch):
    data, fold, _ = _dataset(natural=True)
    settings = _settings()
    first = pipeline._prepare(data, fold, settings)
    calls = _instrument(monkeypatch)
    pipeline._run_fold(first, 0, settings, tmp_path / "checkpoints", tmp_path / "first", "test-source")
    old = calls["core"][0]
    old_final_value = calls["detectors"][1]["value"]
    data.reference[fold.validation_rows] = 0  # D is fixed and remains PU-consistent.
    second = pipeline._prepare(data, fold, settings)
    calls = _instrument(monkeypatch)
    pipeline._run_fold(second, 0, settings, tmp_path / "checkpoints", tmp_path / "second", "test-source")
    new = calls["core"][0]
    assert calls["detectors"][1]["value"] < old_final_value
    np.testing.assert_array_equal(old["train"].S_observed, new["train"].S_observed)
    np.testing.assert_array_equal(old["train_exposure"], new["train_exposure"])
    np.testing.assert_array_equal(old["validation"].S_observed, new["validation"].S_observed)
    np.testing.assert_array_equal(old["validation_exposure"], new["validation_exposure"])
    assert not np.array_equal(old["refit_exposure"], new["refit_exposure"])


def test_summary_simulation_ci_counts_generated_datasets_not_folds():
    rows = [{"dataset": "simulation", "sharing_strength": 1., "analysis": "primary",
             "mechanism": "technical_sar", "loss_rate": .8, "calibration_fraction": .2,
             "calibration_spec": "correct", "model": "PU", "probability_semantics": "reference",
             "repetition": repetition, "outer_fold": fold, "macro_auprc": repetition + fold / 10}
            for repetition in range(5) for fold in range(3)]
    tables = {"metrics": pd.DataFrame(rows)}
    pipeline._summarize(tables, simulation=True)
    assert len(tables["per_repetition"]) == 5
    interval = tables["simulation_intervals"].query("metric == 'macro_auprc'").iloc[0]
    assert interval["n"] == 5
    np.testing.assert_allclose(interval["mean"], 2.1)
    empirical = {"metrics": pd.DataFrame(rows)}
    pipeline._summarize(empirical, simulation=False)
    assert "simulation_intervals" not in empirical


def test_scar_correct_calibrator_does_not_fit_target_heterogeneity(monkeypatch):
    data, fold, _ = _dataset()
    prepared = pipeline._prepare(data, fold, _settings())
    calls = _instrument(monkeypatch)
    scenario = {"mechanism": "scar", "loss_rate": .8, "calibration_spec": "correct"}
    pipeline._detector(prepared, np.zeros_like(data.reference), np.arange(10), scenario, data.technical_score)
    kwargs = calls["detectors"][0]["kwargs"]
    assert kwargs["use_technical"] is False
    assert kwargs["use_target_effects"] is False
