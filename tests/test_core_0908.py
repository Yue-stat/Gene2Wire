"""Regression gates for the frozen 0908 protocol's statistical boundaries."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from gene2wire import DatasetBundle, FitConfig, ModelConfig, TuningConfig, run_model_grid
from gene2wire.models import UnifiedPUModel, model_identity
from gene2wire.tuning import full_joint_candidates, tune_model


def partitions():
    rng = np.random.default_rng(9208)
    x = rng.normal(size=(100, 4))
    p = 1 / (1 + np.exp(-(x @ rng.normal(scale=.5, size=(4, 3)) - .3)))
    z = rng.random(p.shape) < p
    d = z & (rng.random(p.shape) < .65)

    def bundle(rows, features=None):
        return DatasetBundle(
            x[rows] if features is None else features,
            d[rows], np.ones_like(d[rows]),
            cell_ids=[f"cell_{i}" for i in rows], target_ids=("a", "b", "c"),
        )

    return bundle(range(50)), bundle(range(50, 75)), x[75:], z, bundle


def tuning(**changes):
    return TuningConfig(
        ranks=(0, 1, 2), shared_l2=(.03, .1), residual_l2=(.03, .1),
        target_l2=(.03, .1), anchor_shared_l2=.1, anchor_residual_l2=.1,
        anchor_target_l2=.1, **changes,
    )


def fit_config():
    return FitConfig(maxiter=200, retry_maxiter=300, tolerance=1e-7)


def run_kwargs():
    train, validation, test, _, _ = partitions()
    return dict(
        train=train, validation=validation, test_X=test,
        train_exposure=.65, validation_exposure=.65, test_exposure=.65,
        test_cell_ids=[f"cell_{i}" for i in range(75, 100)],
        fit=fit_config(), seed=91, run_fingerprint="caller-static-label",
    )


def test_budget_keeps_exact_endpoints_with_target_features():
    base = ModelConfig(name="PU-Joint", kind="joint", rank=1, use_target_features=True)
    grid = full_joint_candidates(base, tuning(include_endpoints=True, candidate_budget=9))
    assert len(grid) == 9
    assert {c.kind for c in grid} == {"direct", "lowrank", "joint"}
    assert all(c.residual_l2 == 0 for c in grid if c.kind == "lowrank")
    assert all(c.rank == 0 for c in grid if c.kind == "direct")
    assert len({str(model_identity(c)) for c in grid}) == len(grid)
    assert grid == full_joint_candidates(base, tuning(include_endpoints=True, candidate_budget=9))
    with pytest.raises(ValueError, match="too small"):
        full_joint_candidates(base, tuning(include_endpoints=True, candidate_budget=2))


def test_joint_native_budget_and_explicit_winner_endpoints():
    train, validation, _, _, _ = partitions()
    direct_winner = ModelConfig(name="Logistic", kind="direct", residual_l2=0.007)
    lowrank_winner = ModelConfig(name="MIRT", kind="lowrank", rank=2,
                                 shared_l2=0.013)
    result = tune_model(
        train, validation, .65, .65,
        ModelConfig(name="Joint", kind="joint", rank=1),
        tuning(candidate_budget=3, include_endpoints=True),
        fit=fit_config(), seed=92,
        required_endpoints=(direct_winner, lowrank_winner),
    )
    assert len(result.trials) == 5
    assert sum(trial.config.kind == "joint" for trial in result.trials) == 3
    assert sum(trial.stage == "inherited_endpoint" for trial in result.trials) == 2
    identities = {str(model_identity(trial.config)) for trial in result.trials}
    assert str(model_identity(direct_winner)) in identities
    assert str(model_identity(lowrank_winner)) in identities


def test_required_joint_endpoints_reject_different_nuisance_or_grouped_structure():
    train, validation, _, _, _ = partitions()
    base = ModelConfig(name="Joint", kind="joint", rank=1, nuisance_l2=.1)
    kwargs = dict(
        train=train, validation=validation, train_exposure=.65,
        validation_exposure=.65, base_model=base,
        tuning=tuning(candidate_budget=1, include_endpoints=True),
        fit=fit_config(), seed=93,
    )
    with pytest.raises(ValueError, match="nuisance_l2"):
        tune_model(required_endpoints=(
            ModelConfig(name="Logistic", kind="direct", nuisance_l2=.2),
        ), **kwargs)
    with pytest.raises(ValueError, match="grouped low-rank"):
        tune_model(required_endpoints=(
            ModelConfig(
                name="Separate-A", kind="lowrank", rank=1,
                nuisance_l2=.1,
                lowrank_feature_groups=("panel_A_gene", "panel_B_gene"),
            ),
        ), **kwargs)


def test_staged_joint_native_budget_appends_exact_endpoints():
    train, validation, _, _, _ = partitions()
    direct_winner = ModelConfig(name="Logistic", kind="direct", residual_l2=.007)
    lowrank_winner = ModelConfig(name="MIRT", kind="lowrank", rank=2,
                                 shared_l2=.013)
    result = tune_model(
        train, validation, .65, .65,
        ModelConfig(name="Joint", kind="joint", rank=1),
        tuning(strategy="staged_rank_l2", candidate_budget=4,
               include_endpoints=True),
        fit=fit_config(), seed=94,
        required_endpoints=(direct_winner, lowrank_winner),
    )
    assert len(result.trials) == 6
    assert sum(trial.config.kind == "joint" for trial in result.trials) == 4
    assert {trial.stage for trial in result.trials} == {
        "rank", "penalty", "inherited_endpoint"
    }


def test_rank_zero_is_exact_direct_even_with_finite_iterations():
    train, _, _, _, _ = partitions()
    direct = ModelConfig(name="PU", kind="direct", residual_l2=.1)
    joint = direct.with_updates(name="PU-Joint", kind="joint")
    limited = FitConfig(maxiter=2, initialization="svd", init_direct_maxiter=1)
    first = UnifiedPUModel(direct, limited).fit(train, .65, seed=8)
    second = UnifiedPUModel(joint, limited).fit(train, .65, seed=8)
    np.testing.assert_array_equal(first.predict_proba(train.X_cell), second.predict_proba(train.X_cell))
    assert second.config.kind == "direct"
    assert first.iterations == second.iterations


def test_aliases_share_trials_and_final_fit_and_resume(tmp_path):
    models = (
        ModelConfig(name="PU", kind="direct"),
        ModelConfig(name="PU-Joint", kind="joint", rank=0),
    )
    grid = TuningConfig(ranks=(0,), shared_l2=(.1,), residual_l2=(.1,), target_l2=(.1,), include_endpoints=True)
    kwargs = {**run_kwargs(), "models": models, "tuning": grid, "checkpoint_dir": tmp_path}
    original_fit = UnifiedPUModel.fit
    calls = []

    def recording(self, *args, **kw):
        calls.append(self.config.kind)
        return original_fit(self, *args, **kw)

    with patch.object(UnifiedPUModel, "fit", recording):
        result = run_model_grid(**kwargs)
    assert calls == ["direct", "direct"]  # one validation fit, one final refit
    np.testing.assert_array_equal(result.models["PU"].latent_probability, result.models["PU-Joint"].latent_probability)
    assert result.models["PU-Joint"].summary()["selected_structure"] == "direct"
    with patch.object(UnifiedPUModel, "fit", side_effect=AssertionError("refitted")):
        again = run_model_grid(**kwargs)
    assert all(item.resumed for item in again.models.values())
    # A new reporting alias has no completed-model checkpoint. Both its tuning
    # trial and its final fit must still load from the canonical disk caches.
    renamed = models[1].with_updates(name="new endpoint alias")
    with patch.object(UnifiedPUModel, "fit", side_effect=AssertionError("alias refitted")):
        alias_result = run_model_grid(**{**kwargs, "models": (renamed,)})
    np.testing.assert_array_equal(
        alias_result.models[renamed.name].latent_probability,
        result.models["PU"].latent_probability,
    )


def test_adding_model_does_not_invalidate_existing_completed_fit(tmp_path):
    direct = ModelConfig(name="PU", kind="direct")
    grid = tuning()
    kwargs = {**run_kwargs(), "models": (direct,), "tuning": grid, "checkpoint_dir": tmp_path}
    run_model_grid(**kwargs)
    changed = run_model_grid(**{**kwargs, "models": (direct, direct.with_updates(name="Logistic", pu=False))})
    assert changed.models["PU"].resumed
    assert not changed.models["Logistic"].resumed


def test_caller_fingerprint_cannot_hide_data_or_source_changes(tmp_path):
    kwargs = {**run_kwargs(), "models": (ModelConfig(name="PU", kind="direct"),),
              "tuning": tuning(), "checkpoint_dir": tmp_path}
    first = run_model_grid(**kwargs)
    changed = run_model_grid(**{**kwargs, "test_X": kwargs["test_X"] + .001})
    assert first.fingerprint != changed.fingerprint
    with patch("gene2wire.runner.sha256_source_tree", return_value="changed-source"):
        source_changed = run_model_grid(**kwargs)
    assert source_changed.fingerprint != first.fingerprint
    assert not source_changed.models["PU"].resumed


def test_development_refit_can_use_explicit_new_feature_schema():
    train, val, test, _, bundle = partitions()
    dev_x = np.vstack((train.X_cell, val.X_cell))
    refit_x = np.column_stack((dev_x, dev_x[:, 0] ** 2))
    refit = bundle(range(75), refit_x)
    kwargs = {**run_kwargs(), "models": (ModelConfig(name="PU", kind="direct"),),
              "tuning": tuning(), "refit": refit, "refit_exposure": .65}
    with pytest.raises(ValueError, match="feature counts"):
        run_model_grid(**kwargs)
    result = run_model_grid(**kwargs, refit_test_X=np.column_stack((test, test[:, 0] ** 2)))
    assert result.models["PU"].fitted.residual.shape == (5, 3)
    assert result.models["PU"].latent_probability.shape == (25, 3)


def test_staged_search_obeys_same_total_candidate_budget():
    train, val, _, _, _ = partitions()
    result = tune_model(
        train, val, .65, .65, ModelConfig(name="PU-Joint", kind="joint", rank=1),
        tuning(strategy="staged_rank_l2", candidate_budget=8, include_endpoints=True),
        fit=fit_config(),
    )
    assert len(result.trials) <= 8
    assert {trial.config.kind for trial in result.trials} == {"direct", "lowrank", "joint"}


def test_hidden_posterior_is_monotone_for_constant_exposure():
    train, _, test, _, _ = partitions()
    fitted = UnifiedPUModel(ModelConfig(name="PU", kind="direct", residual_l2=.1), fit_config()).fit(train, .65)
    p = fitted.predict_proba(test)
    h = fitted.predict_hidden(test, .65)
    np.testing.assert_array_equal(np.argsort(p, axis=0), np.argsort(h, axis=0))
    assert np.all((h >= 0) & (h <= 1))
    assert np.all(fitted.predict_hidden(test, 1.0) == 0)
    nonpu = replace(fitted, config=fitted.config.with_updates(pu=False))
    with pytest.raises(ValueError, match="requires"):
        nonpu.predict_hidden(test, .65)


def test_mixed_clean_and_pu_objective_counts_clean_entries_once():
    train, _, _, truth, _ = partitions()
    clean = np.arange(train.n_cells) < 10
    labels = np.array(train.S_observed, copy=True)
    labels[clean] = truth[:50][clean]
    exposure = np.full(labels.shape, .65)
    exposure[clean] = 1.0
    mixed = DatasetBundle(train.X_cell, labels, train.W_measured,
                          cell_ids=train.cell_ids, target_ids=train.target_ids)
    model = UnifiedPUModel(ModelConfig(name="Reference+PU", kind="direct"), fit_config())
    fitted = model.fit(mixed, exposure)
    p = fitted.predict_proba(mixed.X_cell)
    q = exposure * p
    expected = -np.mean(labels * np.log(q) + (1 - labels) * np.log1p(-q))
    assert fitted.objective == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("initialization", ["svd", "random"])
def test_pu_and_nonpu_are_identical_at_unit_exposure(initialization):
    kwargs = run_kwargs()
    kwargs.update(train_exposure=1.0, validation_exposure=1.0, test_exposure=1.0,
                  fit=replace(fit_config(), initialization=initialization))
    models = []
    for kind, rank in (("direct", 0), ("lowrank", 1), ("joint", 1)):
        models.extend((ModelConfig(name=kind, kind=kind, rank=rank, pu=False),
                       ModelConfig(name=f"PU-{kind}", kind=kind, rank=rank, pu=True)))
    result = run_model_grid(**kwargs, models=models, tuning=TuningConfig(
        ranks=(1,), shared_l2=(.1,), residual_l2=(.1,), target_l2=(.1,)))
    for kind in ("direct", "lowrank", "joint"):
        first = result.models[kind]
        second = result.models[f"PU-{kind}"]
        np.testing.assert_array_equal(first.latent_probability, second.latent_probability)
        assert first.tuning.best_validation_loss == second.tuning.best_validation_loss
