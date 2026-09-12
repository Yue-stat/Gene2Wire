from __future__ import annotations

import inspect

import numpy as np
import pytest

from gene2wire.experiments import pipeline
from gene2wire.experiments.contracts import FeatureSet, Fold
from gene2wire.experiments.pipeline import PreparedFold
from gene2wire.experiments.protocol import Settings
from gene2wire.experiments.pu_comparators import (
    AssayTargetPropensityEncoder,
    fit_geneml_adapted,
    fit_sar_em,
    fit_shift_imc_adapted,
    shift_imc_unbiased_entry_loss,
    tune_geneml_adapted,
)


def _small_pu_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(417)
    x = rng.normal(size=(18, 3))
    beta = np.array([[0.9, -0.6, 0.3], [-0.3, 0.8, 0.5], [0.4, 0.2, -0.7]])
    truth = (x @ beta > np.array([0.0, -0.15, 0.2])).astype(float)
    exposure = np.broadcast_to(np.array([0.8, 0.65, 0.9]), truth.shape).copy()
    uniforms = rng.uniform(size=truth.shape)
    observed = truth * (uniforms < exposure)
    measured = np.ones_like(observed, dtype=bool)
    measured[[0, 3, 6], 0] = False
    measured[[1, 4, 7], 1] = False
    measured[[2, 5, 8], 2] = False
    observed[~measured] = 0
    assays = np.asarray(["A", "B", "C"] * 6)
    encoder = AssayTargetPropensityEncoder.fit(assays, ["t0", "t1", "t2"])
    phi = encoder.transform(assays, ["t0", "t1", "t2"])
    return x, observed, measured, exposure, phi


def test_public_fit_apis_cannot_accept_reference_truth() -> None:
    forbidden = {"reference", "truth", "hidden", "y_ref", "generator_propensity"}
    for function in (fit_geneml_adapted, fit_shift_imc_adapted, fit_sar_em):
        parameters = {name.lower() for name in inspect.signature(function).parameters}
        assert not parameters.intersection(forbidden)


def test_propensity_encoder_is_outcome_blind_and_schema_locked() -> None:
    assays = ["B", "A", "B", "C"]
    qc = np.array([[1.0], [2.0], [4.0], [8.0]])
    encoder = AssayTargetPropensityEncoder.fit(
        assays, ["left", "right"], qc, ["depth"]
    )
    assert encoder.assay_levels == ("A", "B", "C")
    assert "assay=B:target=right" in encoder.feature_names
    assert "depth:target=right" in encoder.feature_names
    design = encoder.transform(assays, ["left", "right"], qc)
    assert design.shape == (4, 2, len(encoder.feature_names))
    assert np.all(np.isfinite(design))
    with pytest.raises(ValueError, match="exactly"):
        encoder.transform(assays, ["right", "left"], qc)
    with pytest.raises(ValueError, match="unknown assay"):
        encoder.transform(["A", "A", "D", "A"], ["left", "right"], qc)
    assert "observed" not in inspect.signature(AssayTargetPropensityEncoder.fit).parameters


def test_observed_positive_outside_measured_entries_is_rejected() -> None:
    x, observed, measured, exposure, _ = _small_pu_problem()
    observed = observed.copy()
    observed[0, 0] = 1
    assert not measured[0, 0]
    with pytest.raises(ValueError, match="cannot occur"):
        fit_shift_imc_adapted(x, observed, measured, exposure, rank=1, maxiter=2)


@pytest.mark.parametrize("clean_label", [0.0, 1.0])
def test_shift_imc_entry_loss_is_unbiased(clean_label: float) -> None:
    p = np.array([0.13, 0.41, 0.82])
    e = np.array([0.2, 0.55, 0.9])
    loss_if_hidden = shift_imc_unbiased_entry_loss(p, np.zeros(3), e)
    if clean_label:
        loss_if_seen = shift_imc_unbiased_entry_loss(p, np.ones(3), e)
        expectation = e * loss_if_seen + (1 - e) * loss_if_hidden
    else:
        expectation = loss_if_hidden
    np.testing.assert_allclose(expectation, (p - clean_label) ** 2, atol=1e-12)


def test_all_comparators_return_distinct_reference_and_observed_probabilities() -> None:
    x, observed, measured, exposure, phi = _small_pu_problem()
    geneml = fit_geneml_adapted(
        x, observed, measured, phi, rank=1, maxiter=4, seed=11
    )
    shift = fit_shift_imc_adapted(
        x, observed, measured, exposure, rank=1, maxiter=20, seed=12
    )
    sar = fit_sar_em(
        x,
        observed,
        measured,
        phi,
        maxiter=4,
        convergence_window=3,
        inner_maxiter=30,
        seed=13,
    )

    for fitted in (geneml, sar):
        p = fitted.predict_reference(x)
        e = fitted.predict_exposure(x, phi)
        q = fitted.predict_observed(x, phi)
        assert p.shape == observed.shape
        assert np.all((0 <= p) & (p <= 1))
        assert np.all((0 <= e) & (e <= 1))
        np.testing.assert_allclose(q, p * e)
        assert fitted.diagnostics()["probability_semantics"].startswith("reference_p")

    p = shift.predict_reference(x)
    q = shift.predict_observed(x, exposure)
    np.testing.assert_allclose(q, p * exposure)
    assert shift.target_input_kind == "known_target_identity"
    # Prediction-time zero means no chance of observing a label and is valid,
    # although a positive training label may never have zero exposure.
    zero_exposure = exposure.copy()
    zero_exposure[:, 0] = 0
    np.testing.assert_allclose(shift.predict_observed(x, zero_exposure)[:, 0], 0)


def test_propensity_prediction_cannot_silently_switch_design_schema() -> None:
    x, observed, measured, _, phi = _small_pu_problem()
    contextual = fit_sar_em(
        x,
        observed,
        measured,
        phi,
        maxiter=2,
        convergence_window=2,
        inner_maxiter=20,
    )
    with pytest.raises(ValueError, match="fitted schema"):
        contextual.predict_exposure(x)

    target_only = fit_geneml_adapted(
        x, observed, measured, rank=1, maxiter=2, seed=9
    )
    with pytest.raises(ValueError, match="fitted schema"):
        target_only.predict_exposure(x, phi)


def test_unmeasured_payload_does_not_change_geneml_fit() -> None:
    x, observed, measured, _, phi = _small_pu_problem()
    noisy_observed = observed.copy()
    noisy_observed[~measured] = 8.0
    noisy_phi = phi.copy()
    noisy_phi[~measured] = 1234.0
    first = fit_geneml_adapted(
        x, observed, measured, phi, rank=1, maxiter=3, seed=100
    )
    second = fit_geneml_adapted(
        x, noisy_observed, measured, noisy_phi, rank=1, maxiter=3, seed=100
    )
    np.testing.assert_allclose(first.predict_reference(x), second.predict_reference(x))
    np.testing.assert_allclose(
        first.predict_exposure(x, phi), second.predict_exposure(x, phi)
    )


def test_shift_imc_is_deterministic_and_excludes_unmeasured_payload() -> None:
    x, observed, measured, exposure, _ = _small_pu_problem()
    noisy_observed = observed.copy()
    noisy_observed[~measured] = -4.0
    noisy_exposure = exposure.copy()
    noisy_exposure[~measured] = -99.0
    first = fit_shift_imc_adapted(
        x, observed, measured, exposure, rank=1, maxiter=20, seed=71
    )
    second = fit_shift_imc_adapted(
        x, noisy_observed, measured, noisy_exposure, rank=1, maxiter=20, seed=71
    )
    np.testing.assert_allclose(first.predict_reference(x), second.predict_reference(x))


def test_sar_em_excludes_unmeasured_label_and_propensity_payload() -> None:
    x, observed, measured, _, phi = _small_pu_problem()
    noisy_observed = observed.copy()
    noisy_observed[~measured] = -7.0
    noisy_phi = phi.copy()
    noisy_phi[~measured] = -321.0
    options = dict(maxiter=3, convergence_window=2, inner_maxiter=20, seed=19)
    first = fit_sar_em(x, observed, measured, phi, **options)
    second = fit_sar_em(x, noisy_observed, measured, noisy_phi, **options)
    np.testing.assert_allclose(first.predict_reference(x), second.predict_reference(x))
    np.testing.assert_allclose(first.predict_exposure(x, phi), second.predict_exposure(x, phi))


def test_tuner_deduplicates_configs_and_enforces_candidate_budget() -> None:
    x, observed, measured, _, phi = _small_pu_problem()
    result = tune_geneml_adapted(
        x[:12],
        observed[:12],
        measured[:12],
        x[12:],
        observed[12:],
        measured[12:],
        train_propensity_design=phi[:12],
        validation_propensity_design=phi[12:],
        candidates=[{"rank": 1}, {"rank": 1}],
        maxiter=2,
        seed=44,
    )
    assert len(result.trials) == 1
    assert result.selection_metric == "observed_log_loss"
    assert result.trials[0].seed >= 0

    too_many = [{"rank": 1, "l2_u": 0.01 + index / 1000} for index in range(33)]
    with pytest.raises(ValueError, match="at most 32"):
        tune_geneml_adapted(
            x[:12],
            observed[:12],
            measured[:12],
            x[12:],
            observed[12:],
            measured[12:],
            candidates=too_many,
            maxiter=1,
        )


def test_sar_em_requires_support_for_every_target() -> None:
    x, observed, measured, _, _ = _small_pu_problem()
    measured = measured.copy()
    observed = observed.copy()
    measured[:, 2] = False
    observed[:, 2] = 0
    with pytest.raises(ValueError, match="every target"):
        fit_sar_em(x, observed, measured, maxiter=2, convergence_window=2)


def _prepared_comparator_problem(*, target_features: bool = False) -> PreparedFold:
    x, observed, measured, _, _ = _small_pu_problem()
    target = np.arange(observed.shape[1], dtype=float)[:, None] if target_features else None
    features = FeatureSet(
        X=x,
        Y_target=target,
        feature_blocks={"gene": tuple(range(x.shape[1]))},
        feature_names=tuple(f"g{index}" for index in range(x.shape[1])),
    )
    fold = Fold(
        outer_fold=0,
        train_rows=np.arange(0, 9),
        validation_rows=np.arange(9, 14),
        test_rows=np.arange(14, 18),
    )
    return PreparedFold(
        name="comparator-fixture",
        reference=observed.astype(bool),
        measured=measured,
        cell_ids=tuple(f"cell-{index}" for index in range(len(x))),
        target_ids=tuple(f"target-{index}" for index in range(observed.shape[1])),
        groups={},
        natural_observed=None,
        platform=None,
        technical_score=None,
        train_features=features,
        refit_features=features,
        fold=fold,
        metadata={"model_seed": 0},
        training_measured=measured,
        virtual_assays=np.asarray(["A", "B", "C"] * 6),
    )


def _comparator_settings(**updates) -> Settings:
    values = dict(
        n_outer_folds=2,
        n_jobs=1,
        n_repetitions=1,
        seed=80,
        paired_fraction=0.3,
        loss_rates=(0.0,),
        candidate_budget=32,
        penalties=(0.1,),
        maxiter=3,
        retry_maxiter=3,
        tolerance=1e-4,
        init_direct_maxiter=3,
        run_information_controls=False,
        run_random_forest=False,
        run_mechanism_controls=False,
        run_calibration_controls=False,
        run_qiao=False,
        run_pu_comparators=True,
        calibration_fractions=(0.3,),
    )
    values.update(updates)
    return Settings(**values)


def _run_comparator_wrapper(
    prepared: PreparedFold,
    settings: Settings,
    *,
    checkpoint_dir=None,
    progress=None,
    context_updates=None,
):
    observed = np.asarray(prepared.reference, dtype=bool)
    exposure = np.ones_like(observed, dtype=float)
    records: list[str] = []
    tables = {"selected": [], "tuning": [], "failures": []}
    context = {
        "repetition": 0,
        "outer_fold": 0,
        "mechanism": "assay_target_sar",
        "positive_retention": 1.0,
        "panel_id": "full",
    }
    context.update(context_updates or {})
    pipeline._run_pu_comparators(
        prepared,
        observed,
        exposure,
        exposure,
        settings,
        context,
        lambda name, *args, **kwargs: records.append(name),
        tables,
        checkpoint_dir=checkpoint_dir,
        on_progress=None if progress is None else progress.append,
    )
    return records, tables


def test_wrapper_initialization_seeds_do_not_change_with_mask_severity() -> None:
    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(candidate_budget=1)
    _, first = _run_comparator_wrapper(prepared, settings)
    _, second = _run_comparator_wrapper(
        prepared,
        settings,
        context_updates={
            "mechanism": "scar",
            "positive_retention": 0.25,
            "panel_id": "another-panel-draw",
        },
    )
    first_candidate = {
        row["model"]: row["seed"] for row in first["tuning"]
    }
    second_candidate = {
        row["model"]: row["seed"] for row in second["tuning"]
    }
    first_refit = {
        row["model"]: row["refit_seed"] for row in first["selected"]
    }
    second_refit = {
        row["model"]: row["refit_seed"] for row in second["selected"]
    }
    assert first_candidate == second_candidate
    assert first_refit == second_refit


def test_wrapper_caps_shift_rank_by_target_descriptor_dimension() -> None:
    prepared = _prepared_comparator_problem(target_features=True)
    settings = _comparator_settings(use_target_features=True)
    records, tables = _run_comparator_wrapper(prepared, settings)
    assert "Inductive-PU-MC" in records
    assert not [
        row for row in tables["failures"]
        if row["model"] == "Inductive-PU-MC"
    ]
    shift_rows = [
        row for row in tables["tuning"]
        if row["model"] == "Inductive-PU-MC"
    ]
    assert {row["rank"] for row in shift_rows} == {1, 2}
    assert all(row["rank"] <= prepared.train_features.Y_target.shape[1] + 1
               for row in shift_rows)
    selected_inputs = {
        row["model"]: row["target_input_kind"] for row in tables["selected"]
    }
    assert selected_inputs["Inductive-PU-MC"] == "target_features"
    assert selected_inputs["GenEML-adapted"] == "known_target_identity"
    assert selected_inputs["SAR-PU"] == "known_target_identity"


def test_wrapper_retains_failed_candidate_and_continues_with_valid_fit(
    monkeypatch,
) -> None:
    from gene2wire.experiments import pu_comparators

    prepared = _prepared_comparator_problem(target_features=True)
    settings = _comparator_settings(use_target_features=True)
    original = pu_comparators.fit_shift_imc_adapted

    def fail_rank_one(*args, rank, **kwargs):
        if rank == 1:
            raise FloatingPointError("deliberate candidate failure")
        return original(*args, rank=rank, **kwargs)

    monkeypatch.setattr(
        pu_comparators, "fit_shift_imc_adapted", fail_rank_one)
    progress = []
    records, tables = _run_comparator_wrapper(
        prepared, settings, progress=progress)
    assert "Inductive-PU-MC" in records
    failures = [
        row for row in tables["failures"]
        if row["model"] == "Inductive-PU-MC"
    ]
    assert len(failures) == 1
    assert failures[0]["stage"] == "pu_comparator_candidate"
    assert failures[0]["rank"] == 1
    trials = [
        row for row in tables["tuning"]
        if row["model"] == "Inductive-PU-MC"
    ]
    assert {row["status"] for row in trials} == {"failed", "complete"}
    selected = next(
        row for row in tables["selected"]
        if row["model"] == "Inductive-PU-MC"
    )
    assert selected["rank"] == 2
    assert selected["underlying_method"] == "ShiftIMC-adapted"
    assert selected["adapted"] is True
    shift_events = [
        row["event"] for row in progress
        if row.get("model") == "Inductive-PU-MC"
    ]
    assert "candidate_inventory" in shift_events
    assert "candidate_start" in shift_events
    assert "candidate_complete" in shift_events
    assert "refit_start" in shift_events
    assert "refit_complete" in shift_events
    assert all(
        "elapsed_seconds" in row
        for row in progress
        if row.get("model") == "Inductive-PU-MC"
        and row["event"] in {"candidate_complete", "refit_complete"}
    )


def test_wrapper_balances_bounded_candidate_coordinates() -> None:
    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(
        candidate_budget=4, penalties=(0.001, 0.01, 0.1, 1.0))
    _, tables = _run_comparator_wrapper(prepared, settings)
    geneml = [
        row for row in tables["tuning"]
        if row["model"] == "GenEML-adapted"
    ]
    assert len(geneml) == 4
    assert {row["rank"] for row in geneml} == {1, 2, 3}
    assert len({row["l2_u"] for row in geneml}) == 4
    sar = [row for row in tables["tuning"] if row["model"] == "SAR-PU"]
    assert len(sar) == 4
    assert len({row["l2_classifier"] for row in sar}) == 4
    assert len({row["l2_propensity"] for row in sar}) == 4


def test_wrapper_resumes_candidate_and_refit_checkpoints(tmp_path) -> None:
    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(candidate_budget=1)
    first_progress: list[dict] = []
    _, first = _run_comparator_wrapper(
        prepared, settings, checkpoint_dir=tmp_path, progress=first_progress)
    assert not first["failures"]
    assert all(not row["resumed"] for row in first["tuning"])
    assert all(not row["final_fit_resumed"] for row in first["selected"])

    second_progress: list[dict] = []
    _, second = _run_comparator_wrapper(
        prepared, settings, checkpoint_dir=tmp_path, progress=second_progress)
    assert not second["failures"]
    assert all(row["resumed"] for row in second["tuning"])
    assert all(row["final_fit_resumed"] for row in second["selected"])
    assert all(row["checkpoint_status"] == "checkpoint"
               for row in second["selected"])
    completed = [
        row for row in second_progress if row["event"] == "model_complete"
    ]
    assert len(completed) == 3
    assert all(row["cache_status"] == "checkpoint" for row in completed)
