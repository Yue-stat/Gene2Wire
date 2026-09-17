from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from gene2wire.experiments import pipeline
from gene2wire.experiments.contracts import FeatureSet, Fold
from gene2wire.experiments.pipeline import PreparedFold
from gene2wire.experiments.protocol import Settings
from gene2wire.experiments.pu_comparators import (
    AssayTargetPropensityEncoder,
    PerTargetAssayPropensityEncoder,
    fit_geneml_adapted,
    fit_sar_em,
    fit_shift_imc_adapted,
    shift_imc_unbiased_entry_loss,
    tune_geneml_adapted,
    tune_shift_imc_adapted,
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


def test_per_target_sar_encoder_is_full_rank_and_target_permutation_equivariant() -> None:
    assays = np.asarray(["A", "B", "C"] * 6)
    targets = ("alpha", "beta", "gamma")
    measured = np.column_stack(
        (
            np.isin(assays, ("A", "B")),
            np.isin(assays, ("B", "C")),
            assays == "A",
        )
    )
    depth = np.linspace(-2.0, 2.0, len(assays))
    qc = np.column_stack(
        (depth, (assays == "B").astype(float), np.ones(len(assays)))
    )
    names = ("depth", "duplicate_assay_B", "constant")
    encoder = PerTargetAssayPropensityEncoder.fit(
        assays, targets, measured, qc, names
    )
    design = encoder.transform(assays, targets, qc)

    assert all(
        "target=" not in name and ":target" not in name
        for schema in encoder.feature_names_by_target
        for name in schema
    )
    for target, count in enumerate(encoder.active_feature_counts):
        rows = measured[:, target]
        active = design[rows, target, :count]
        augmented = np.column_stack((np.ones(rows.sum()), active))
        assert np.linalg.matrix_rank(augmented) == augmented.shape[1]
        assert not np.any(design[:, target, count:])

    permutation = np.asarray([2, 0, 1])
    permuted_targets = tuple(targets[index] for index in permutation)
    permuted_encoder = PerTargetAssayPropensityEncoder.fit(
        assays, permuted_targets, measured[:, permutation], qc, names
    )
    permuted_design = permuted_encoder.transform(assays, permuted_targets, qc)
    np.testing.assert_allclose(permuted_design, design[:, permutation])
    assert permuted_encoder.active_feature_counts == tuple(
        encoder.active_feature_counts[index] for index in permutation
    )
    assert permuted_encoder.feature_names_by_target == tuple(
        encoder.feature_names_by_target[index] for index in permutation
    )
    assert "observed" not in inspect.signature(
        PerTargetAssayPropensityEncoder.fit
    ).parameters


def test_per_target_sar_assay_contrasts_are_relabeling_equivariant_under_l2() -> None:
    from gene2wire.experiments.sarpu_authors import fit_sarpu_authors

    rng = np.random.default_rng(71)
    n_rows = 240
    assays = np.asarray(["A", "B", "C"] * (n_rows // 3))
    relabeling = {"A": "Z", "B": "A", "C": "B"}
    relabeled_assays = np.asarray([relabeling[value] for value in assays])
    targets = ("response",)
    measured = np.ones((n_rows, 1), dtype=bool)

    encoder = PerTargetAssayPropensityEncoder.fit(assays, targets, measured)
    relabeled_encoder = PerTargetAssayPropensityEncoder.fit(
        relabeled_assays, targets, measured
    )
    design = encoder.transform(assays, targets)
    relabeled_design = relabeled_encoder.transform(relabeled_assays, targets)

    # Every deterministic orthonormal basis of the centered K-level space has
    # the same row kernel. A category relabeling can therefore only rotate the
    # coefficient coordinates, which preserves an L2 penalty.
    np.testing.assert_allclose(
        design[:, 0] @ design[:, 0].T,
        relabeled_design[:, 0] @ relabeled_design[:, 0].T,
        rtol=0,
        atol=5e-15,
    )
    basis = PerTargetAssayPropensityEncoder._assay_contrast_basis(3)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), rtol=0, atol=5e-15)
    np.testing.assert_allclose(basis.sum(axis=0), 0.0, rtol=0, atol=5e-15)

    x = rng.normal(size=(n_rows, 2))
    reference_probability = 1.0 / (
        1.0 + np.exp(-(x[:, 0] - 0.3 * x[:, 1]))
    )
    assay_exposure = {"A": 0.30, "B": 0.55, "C": 0.80}
    exposure = np.asarray([assay_exposure[value] for value in assays])
    reference = rng.binomial(1, reference_probability)
    observed = (reference * rng.binomial(1, exposure))[:, None]

    fitted = fit_sarpu_authors(
        x,
        observed,
        measured,
        design,
        target_ids=targets,
        max_its=100,
        seed=2,
    )
    relabeled_fitted = fit_sarpu_authors(
        x,
        observed,
        measured,
        relabeled_design,
        target_ids=targets,
        max_its=100,
        seed=2,
    )
    assert fitted.converged and relabeled_fitted.converged
    np.testing.assert_allclose(
        fitted.predict_reference(x),
        relabeled_fitted.predict_reference(x),
        rtol=0,
        atol=5e-14,
    )
    np.testing.assert_allclose(
        fitted.predict_exposure(x, design),
        relabeled_fitted.predict_exposure(x, relabeled_design),
        rtol=0,
        atol=5e-14,
    )


def test_per_target_sar_encoder_rejects_required_target_unseen_assays() -> None:
    assays = np.asarray(["A", "B", "C", "A", "B", "C"])
    targets = ("left", "right")
    measured = np.column_stack(
        (np.isin(assays, ("A", "B")), np.isin(assays, ("B", "C")))
    )
    encoder = PerTargetAssayPropensityEncoder.fit(assays, targets, measured)

    required = np.zeros_like(measured, dtype=bool)
    required[assays == "C", 0] = True
    with pytest.raises(
        ValueError,
        match=r"unseen on fitted W support for target 'left'.*C",
    ):
        encoder.transform(assays, targets, required_mask=required)

    # A target-unseen level is harmless on rows that the caller declares out
    # of support; its padded contrast coordinates are never consumed.
    design = encoder.transform(assays, targets, required_mask=np.zeros_like(measured))
    np.testing.assert_array_equal(design[assays == "C", 0], 0.0)

    # A globally new assay is also safe only when every corresponding target
    # entry is explicitly outside the caller's required/support mask.
    off_support = encoder.transform(
        ["new-assay"], targets, required_mask=np.zeros((1, len(targets)))
    )
    np.testing.assert_array_equal(off_support, 0.0)
    required_new = np.zeros((1, len(targets)), dtype=bool)
    required_new[0, 1] = True
    with pytest.raises(
        ValueError,
        match=r"unseen on fitted W support for target 'right'.*new-assay",
    ):
        encoder.transform(["new-assay"], targets, required_mask=required_new)
    with pytest.raises(ValueError, match="unknown assay levels"):
        encoder.transform(["new-assay"], targets)

    with pytest.raises(ValueError, match="required_mask must align"):
        encoder.transform(assays, targets, required_mask=required[:, :1])
    invalid = required.astype(float)
    invalid[0, 0] = 0.5
    with pytest.raises(ValueError, match="required_mask must be binary"):
        encoder.transform(assays, targets, required_mask=invalid)


def test_per_target_sar_encoder_represents_zero_support_target_without_crashing() -> None:
    assays = np.asarray(["A", "B", "A", "B"])
    targets = ("supported", "unsupported")
    measured = np.column_stack(
        (np.ones(len(assays), dtype=bool), np.zeros(len(assays), dtype=bool))
    )

    encoder = PerTargetAssayPropensityEncoder.fit(
        assays, targets, measured
    )
    design = encoder.transform(
        assays, targets, required_mask=measured
    )

    assert encoder.active_feature_counts == (1, 0)
    np.testing.assert_array_equal(design[:, 1], 0.0)


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


def test_shift_imc_identity_is_target_permutation_equivariant() -> None:
    rng = np.random.default_rng(105)
    n_cells, n_targets, n_features = 35, 7, 4
    x = rng.normal(size=(n_cells, n_features))
    coefficients = rng.normal(size=(n_features, n_targets))
    probability = 1.0 / (1.0 + np.exp(-(x @ coefficients)))
    reference = rng.binomial(1, probability)
    exposure = rng.uniform(0.2, 0.9, size=(n_cells, n_targets))
    observed = reference * rng.binomial(1, exposure)
    measured = rng.random((n_cells, n_targets)) > 0.15
    observed = np.where(measured, observed, 0)
    target_ids = tuple(f"target-{index}" for index in range(n_targets))
    permutation = np.asarray([4, 1, 6, 3, 2, 0, 5])
    inverse = np.argsort(permutation)

    first = fit_shift_imc_adapted(
        x,
        observed,
        measured,
        exposure,
        target_ids=target_ids,
        rank=2,
        l2=1e-3,
        maxiter=1000,
        seed=12,
    )
    second = fit_shift_imc_adapted(
        x,
        observed[:, permutation],
        measured[:, permutation],
        exposure[:, permutation],
        target_ids=tuple(target_ids[index] for index in permutation),
        rank=2,
        l2=1e-3,
        maxiter=1000,
        seed=12,
    )

    assert first.converged and second.converged
    assert first.augment_target_intercept is False
    assert np.linalg.matrix_rank(first.target_design) == n_targets
    np.testing.assert_allclose(
        second.predict_reference(x)[:, inverse],
        first.predict_reference(x),
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        second.objective_value, first.objective_value, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        first.predict_reference(
            x, target_ids=tuple(target_ids[index] for index in permutation)
        )[:, inverse],
        first.predict_reference(x),
        rtol=0,
        atol=0,
    )


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


def test_shiftimc_public_tuner_rejects_all_nonconverged_trials(
    monkeypatch,
) -> None:
    from gene2wire.experiments import pu_comparators

    x, observed, measured, exposure, _ = _small_pu_problem()
    original = pu_comparators.fit_shift_imc_adapted

    def finite_but_nonconverged(*args, **kwargs):
        return replace(original(*args, **kwargs), converged=False)

    monkeypatch.setattr(
        pu_comparators, "fit_shift_imc_adapted", finite_but_nonconverged
    )
    with pytest.raises(ValueError, match="no converged finite candidate"):
        tune_shift_imc_adapted(
            x[:12], observed[:12], measured[:12], exposure[:12],
            x[12:], observed[12:], measured[12:], exposure[12:],
            candidates=({"rank": 1, "l2": 0.1},),
            maxiter=500,
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
        # The 3x3 Latin schedule makes every assay estimable for every target
        # on training W support; the original mod-3 schedule confounded each
        # target's missing rows with exactly one assay level.
        virtual_assays=np.asarray(
            ["A", "B", "C", "B", "C", "A", "C", "A", "B"] * 2
        ),
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
    settings = _comparator_settings(
        use_target_features=True, maxiter=500, retry_maxiter=500
    )
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
    assert selected_inputs["GenEML-authors-mask"] == "known_target_identity"
    assert selected_inputs["SAR-PU"] == "known_target_identity"


def test_wrapper_retains_failed_candidate_and_continues_with_valid_fit(
    monkeypatch,
) -> None:
    from gene2wire.experiments import pu_comparators

    prepared = _prepared_comparator_problem(target_features=True)
    settings = _comparator_settings(
        use_target_features=True, maxiter=500, retry_maxiter=500
    )
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
    assert selected["underlying_method"] == (
        "ShiftIMC-inspired paper-based reimplementation"
    )
    assert selected["adapted"] is True
    assert selected["authors_code"] is False
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
    sar_target_events = [
        row
        for row in progress
        if row.get("model") == "SAR-PU"
        and row["event"] in {"target_start", "target_complete"}
    ]
    assert len(sar_target_events) == 4 * len(prepared.target_ids)
    assert {row["stage"] for row in sar_target_events} == {"tuning", "refit"}
    assert {
        row["target_id"] for row in sar_target_events
    } == set(map(str, prepared.target_ids))


def test_wrapper_balances_bounded_candidate_coordinates() -> None:
    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(
        candidate_budget=4, penalties=(0.001, 0.01, 0.1, 1.0))
    _, tables = _run_comparator_wrapper(prepared, settings)
    geneml = [
        row for row in tables["tuning"]
        if row["model"] == "GenEML-authors-mask"
    ]
    assert len(geneml) == 4
    assert {row["rank"] for row in geneml} == {1, 2, 3}
    assert {row["lam_u"] for row in geneml} == {1e-3}
    assert {row["lam_v"] for row in geneml} == {1e-3}
    assert len({row["lam_w"] for row in geneml}) == 4
    assert all(row["authors_code"] is False for row in geneml)
    assert all(row["authors_source"] is True for row in geneml)
    assert all(
        row["executes_unmodified_authors_code"] is False
        for row in geneml
    )
    assert all(
        row["verified_vendor_sha256"] == row["vendor_sha256"]
        for row in geneml
    )
    assert all(
        row["implementation_variant"]
        == "authors-source-python3-measurement-mask-port"
        for row in geneml
    )
    sar = [row for row in tables["tuning"] if row["model"] == "SAR-PU"]
    assert len(sar) == 1
    assert "l2_classifier" not in sar[0]
    assert "l2_propensity" not in sar[0]
    assert sar[0]["implementation_variant"] == (
        "authors_sar_em_per_target_measurement_wrapper"
    )
    assert sar[0]["authors_code"] is False
    assert sar[0]["authors_source"] is True
    assert sar[0]["executes_unmodified_authors_code"] is False
    assert sar[0]["uses_unmodified_authors_em_kernel"] is True
    assert sar[0]["verified_vendor_sha256"] == (
        sar[0]["vendored_source_sha256"]
    )
    assert sar[0]["sar_candidate_max_its"] == 2000
    assert sar[0]["sar_target_retry_max_its"] == 4000
    assert sar[0]["sar_slope_eps"] == pytest.approx(1e-3)
    assert sar[0]["sar_ll_eps"] == pytest.approx(1e-3)
    selected_sar = next(
        row for row in tables["selected"] if row["model"] == "SAR-PU"
    )
    assert selected_sar["max_its"] == 2000
    assert selected_sar["retry_max_its"] == 4000


def test_wrapper_fails_only_sar_for_validation_assay_without_target_support() -> None:
    prepared = _prepared_comparator_problem()
    assays = np.asarray(prepared.virtual_assays)
    train = np.asarray(prepared.fold.train_rows)
    fit_w = np.asarray(prepared.training_measured, dtype=bool).copy()
    # Target 0 has only A/B support during candidate fitting, while validation
    # contains a measured C row. That propensity is not identified and must
    # not be silently mapped to the all-zero/reference contrast.
    train_c = train[assays[train] == "C"]
    fit_w[train_c, 0] = False
    reference = np.asarray(prepared.reference, dtype=bool).copy()
    reference[~fit_w] = False
    prepared = replace(
        prepared,
        reference=reference,
        measured=fit_w,
        training_measured=fit_w,
    )
    settings = _comparator_settings(
        candidate_budget=1, maxiter=500, retry_maxiter=500
    )

    records, tables = _run_comparator_wrapper(prepared, settings)

    assert "SAR-PU" not in records
    assert {"GenEML-authors-mask", "Inductive-PU-MC"}.issubset(records)
    sar_failure = next(
        row
        for row in tables["failures"]
        if row["model"] == "SAR-PU"
        and row["stage"] == "pu_comparator_candidate"
    )
    assert "validation propensity support failed" in sar_failure["error"]
    assert "unseen on fitted W support" in sar_failure["error"]
    sar_trial = next(
        row for row in tables["tuning"] if row["model"] == "SAR-PU"
    )
    assert sar_trial["status"] == "failed"
    assert sar_trial["validation_unseen_required_entries"] > 0
    assert "fail on W-supported rows" in sar_trial[
        "propensity_unseen_level_policy"
    ]


def test_wrapper_fails_only_sar_for_test_assay_without_refit_support() -> None:
    prepared = _prepared_comparator_problem()
    assays = np.asarray(prepared.virtual_assays)
    development = np.sort(
        np.r_[prepared.fold.train_rows, prepared.fold.validation_rows]
    )
    fit_w = np.asarray(prepared.training_measured, dtype=bool).copy()
    development_c = development[assays[development] == "C"]
    fit_w[development_c, 0] = False
    reference = np.asarray(prepared.reference, dtype=bool).copy()
    reference[~fit_w] = False
    prepared = replace(
        prepared,
        reference=reference,
        measured=fit_w,
        training_measured=fit_w,
    )
    settings = _comparator_settings(
        candidate_budget=1, maxiter=500, retry_maxiter=500
    )

    records, tables = _run_comparator_wrapper(prepared, settings)

    assert "SAR-PU" not in records
    assert {"GenEML-authors-mask", "Inductive-PU-MC"}.issubset(records)
    assert any(
        row["model"] == "SAR-PU"
        and row["status"] == "complete"
        for row in tables["tuning"]
    )
    sar_failure = next(
        row
        for row in tables["failures"]
        if row["model"] == "SAR-PU"
        and row["stage"] == "pu_comparator_refit"
    )
    assert "test propensity support failed" in sar_failure["error"]
    assert "unseen on fitted W support" in sar_failure["error"]


def test_authors_sar_failure_is_explicit_and_never_uses_legacy_fallback(
    monkeypatch,
) -> None:
    from gene2wire.experiments import pu_comparators, sarpu_authors

    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(candidate_budget=1)

    def authors_failure(*args, **kwargs):
        raise sarpu_authors.SARPUAuthorsConvergenceError(
            "deliberate authors-code nonconvergence"
        )

    def forbidden_legacy(*args, **kwargs):
        raise AssertionError("legacy clean-room SAR must not be called")

    monkeypatch.setattr(sarpu_authors, "fit_sarpu_authors", authors_failure)
    monkeypatch.setattr(pu_comparators, "fit_sar_em", forbidden_legacy)
    records, tables = _run_comparator_wrapper(prepared, settings)

    assert "SAR-PU" not in records
    assert not [row for row in tables["selected"] if row["model"] == "SAR-PU"]
    sar_trials = [row for row in tables["tuning"] if row["model"] == "SAR-PU"]
    assert len(sar_trials) == 1
    assert sar_trials[0]["status"] == "failed"
    assert "deliberate authors-code nonconvergence" in sar_trials[0]["error"]
    assert any(
        row["model"] == "SAR-PU" and row["stage"] == "pu_comparator"
        for row in tables["failures"]
    )


def test_sar_unsupported_class_support_does_not_block_other_comparators() -> None:
    prepared = _prepared_comparator_problem()
    reference = np.asarray(prepared.reference, dtype=bool).copy()
    reference[:, 0] = False
    prepared = replace(prepared, reference=reference)
    settings = _comparator_settings(
        candidate_budget=1, maxiter=500, retry_maxiter=500
    )

    records, tables = _run_comparator_wrapper(prepared, settings)

    assert "SAR-PU" not in records
    assert {"GenEML-authors-mask", "Inductive-PU-MC"}.issubset(records)
    sar_trial = next(
        row for row in tables["tuning"] if row["model"] == "SAR-PU"
    )
    assert sar_trial["status"] == "failed"
    assert "requires measured positive and unlabeled support" in sar_trial["error"]
    assert not [
        row for row in tables["failures"]
        if row["model"] in {"GenEML-authors-mask", "Inductive-PU-MC"}
    ]


def test_wrapper_resumes_candidate_and_refit_checkpoints(tmp_path) -> None:
    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(
        candidate_budget=1, maxiter=500, retry_maxiter=500
    )
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


def test_cached_comparator_restore_still_fails_on_fresh_vendor_verification(
    tmp_path, monkeypatch,
) -> None:
    from gene2wire.experiments import geneml_authors

    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(
        candidate_budget=1, maxiter=500, retry_maxiter=500
    )
    _run_comparator_wrapper(
        prepared, settings, checkpoint_dir=tmp_path
    )

    def reject_changed_source():
        raise RuntimeError("deliberate fresh GenEML source verification failure")

    monkeypatch.setattr(
        geneml_authors, "verify_geneml_vendor", reject_changed_source
    )
    with pytest.raises(RuntimeError, match="fresh GenEML source verification"):
        _run_comparator_wrapper(
            prepared, settings, checkpoint_dir=tmp_path
        )


def test_shiftimc_all_nonconverged_candidates_are_not_selectable(
    monkeypatch,
) -> None:
    from gene2wire.experiments import pu_comparators

    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(
        candidate_budget=2, maxiter=500, retry_maxiter=500
    )
    original = pu_comparators.fit_shift_imc_adapted

    def force_nonconvergence(*args, **kwargs):
        fitted = original(*args, **kwargs)
        return replace(
            fitted, converged=False,
            optimizer_message="deliberate finite non-convergence",
        )

    monkeypatch.setattr(
        pu_comparators, "fit_shift_imc_adapted", force_nonconvergence
    )
    records, tables = _run_comparator_wrapper(prepared, settings)
    assert "Inductive-PU-MC" not in records
    assert not [
        row for row in tables["selected"]
        if row["model"] == "Inductive-PU-MC"
    ]
    trials = [
        row for row in tables["tuning"]
        if row["model"] == "Inductive-PU-MC"
    ]
    assert trials and all(row["status"] == "failed" for row in trials)
    failures = [
        row for row in tables["failures"]
        if row["model"] == "Inductive-PU-MC"
    ]
    assert failures
    assert any("did not converge" in row["error"] for row in failures)


def test_shiftimc_nonconverged_refit_exports_no_prediction(monkeypatch) -> None:
    from gene2wire.experiments import pu_comparators

    prepared = _prepared_comparator_problem()
    settings = _comparator_settings(
        candidate_budget=1, maxiter=500, retry_maxiter=500
    )
    original = pu_comparators.fit_shift_imc_adapted
    n_candidate_rows = len(prepared.fold.train_rows)

    def fail_only_refit(*args, **kwargs):
        fitted = original(*args, **kwargs)
        if len(np.asarray(args[0])) > n_candidate_rows:
            return replace(
                fitted, converged=False,
                optimizer_message="deliberate finite refit non-convergence",
            )
        return replace(fitted, converged=True)

    monkeypatch.setattr(
        pu_comparators, "fit_shift_imc_adapted", fail_only_refit
    )
    records, tables = _run_comparator_wrapper(prepared, settings)
    assert "Inductive-PU-MC" not in records
    assert not [
        row for row in tables["selected"]
        if row["model"] == "Inductive-PU-MC"
    ]
    refit_failures = [
        row for row in tables["failures"]
        if row["model"] == "Inductive-PU-MC"
        and row["stage"] == "pu_comparator_refit"
    ]
    assert len(refit_failures) == 1
    assert "predictions are not exportable" in refit_failures[0]["error"]
