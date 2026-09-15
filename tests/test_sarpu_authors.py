from __future__ import annotations

import hashlib
import inspect
import json
import warnings
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from gene2wire.seeds import stable_seed
from gene2wire.experiments import sarpu_authors as sar


def _problem(*, n: int = 100, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    propensity = rng.normal(size=(n, 1, 1))
    p = 1.0 / (1.0 + np.exp(-(x[:, 0] + 0.5 * x[:, 1])))
    e = 1.0 / (1.0 + np.exp(-(-0.2 + 0.2 * propensity[:, 0, 0])))
    reference = rng.binomial(1, p)
    observed = (reference * rng.binomial(1, e))[:, None]
    measured = np.ones_like(observed, dtype=bool)
    assert 0 < observed.sum() < n
    return x, observed, measured, propensity


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_vendored_authors_sources_and_provenance_are_pinned():
    root = Path(sar._VENDOR_PACKAGE.__file__).resolve().parent
    actual = {
        name: _digest(root / name)
        for name in sar.SARPU_AUTHORS_PROVENANCE["vendored_source_sha256"]
    }
    assert actual == sar.SARPU_AUTHORS_PROVENANCE["vendored_source_sha256"]
    expected_blobs = {
        "pu_learning.py": "f218fe5ffd6a4d1615b06cd78cfc19d957963eac",
        # The vendored final LF is the sole difference from the upstream blob.
        "PUmodels.py": "9f54be35700f03aace32e401a736149abfb5d0eb",
        "LICENSE": "87bd8db562a5849d3ca7f19f7782b1bc9ae1785a",
    }
    for name, expected in expected_blobs.items():
        payload = (root / name).read_bytes()
        header = f"blob {len(payload)}\0".encode()
        assert hashlib.sha1(header + payload).hexdigest() == expected
    assert sar.SARPU_AUTHORS_PROVENANCE["commit"] == (
        "6e4fc3d8c84ac3512669a4e36ffcb5086f5b42a7"
    )
    assert sar.SARPU_AUTHORS_PROVENANCE["license"] == "MIT"
    assert sar.SARPU_AUTHORS_PROVENANCE["authors_code"] is False
    assert sar.SARPU_AUTHORS_PROVENANCE["authors_source"] is True
    assert (
        sar.SARPU_AUTHORS_PROVENANCE["executes_unmodified_authors_code"]
        is False
    )
    assert sar.SARPU_AUTHORS_PROVENANCE["uses_unmodified_authors_em_kernel"] is True
    assert sar.SARPU_AUTHORS_PROVENANCE["wrapper_source_sha256"] == _digest(
        Path(sar.__file__)
    )
    json.dumps(sar.SARPU_AUTHORS_PROVENANCE, allow_nan=False)


def test_all_measured_wrapper_matches_direct_upstream_call():
    x, observed, measured, propensity = _problem()
    fitted = sar.fit_sarpu_authors(
        x, observed, measured, propensity, max_its=300, seed=13
    )

    augmented = np.concatenate([x, propensity[:, 0]], axis=1)
    classifier_seed = stable_seed(13, "sarpu_authors", 0, "classifier")
    propensity_seed = stable_seed(13, "sarpu_authors", 0, "propensity")
    direct_classifier, direct_propensity, direct_info = sar._UPSTREAM.pu_learn_sar_em(
        augmented,
        observed[:, 0],
        [x.shape[1]],
        classification_attributes=list(range(x.shape[1])),
        classification_model=sar.CompatibleLogisticRegressionPU(
            random_state=classifier_seed
        ),
        propensity_model=sar.CompatibleLogisticRegressionPU(
            random_state=propensity_seed
        ),
        max_its=300,
        slope_eps=1e-4,
        ll_eps=1e-4,
        convergence_window=10,
        refit_classifier=True,
    )

    np.testing.assert_allclose(
        fitted.predict_p(x)[:, 0], direct_classifier.predict_proba(augmented), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        fitted.predict_e(x, propensity)[:, 0],
        direct_propensity.predict_proba(augmented),
        rtol=0,
        atol=1e-14,
    )
    assert fitted.target_iterations[0] == direct_info["nb_iterations"] + 1


def test_compatibility_estimator_pins_upstream_l2_liblinear_semantics():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(80, 3))
    y = (x[:, 0] - 0.4 * x[:, 1] > 0).astype(int)
    seed = 42
    compatible = sar.CompatibleLogisticRegressionPU(random_state=seed).fit(x, y)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"'penalty' was deprecated.*", category=FutureWarning
        )
        explicit = LogisticRegression(
            penalty="l2",
            C=1.0,
            dual=False,
            tol=1e-4,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=seed,
            solver="liblinear",
            max_iter=100,
            verbose=0,
            warm_start=False,
        ).fit(x, y)
    assert compatible.get_params()["penalty"] == "l2"
    np.testing.assert_array_equal(compatible.coef_, explicit.coef_)
    np.testing.assert_array_equal(compatible.intercept_, explicit.intercept_)


def test_duplicate_propensity_columns_are_reduced_to_effective_full_rank():
    x, observed, measured, propensity = _problem(seed=22)
    duplicated = np.concatenate(
        (
            propensity,
            propensity,
            np.ones_like(propensity),
        ),
        axis=2,
    )
    fitted = sar.fit_sarpu_authors(
        x, observed, measured, duplicated, max_its=300, seed=17
    )
    assert fitted.target_propensity_design_ranks.tolist() == [2]
    assert fitted.propensity_active_mask[:, 0].tolist() == [True, False, False]
    reduced = sar.fit_sarpu_authors(
        x, observed, measured, propensity, max_its=300, seed=17
    )
    np.testing.assert_allclose(
        fitted.predict_reference(x), reduced.predict_reference(x), rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        fitted.predict_exposure(x, duplicated),
        reduced.predict_exposure(x, propensity),
        rtol=0,
        atol=1e-14,
    )


def test_named_targets_make_independent_sar_fit_permutation_equivariant():
    x, first_observed, measured_one, first_propensity = _problem(n=120, seed=31)
    observed = np.column_stack(
        (first_observed[:, 0], np.roll(first_observed[:, 0], 5))
    )
    measured = np.broadcast_to(measured_one, observed.shape).copy()
    propensity = np.stack(
        (first_propensity[:, 0], np.roll(first_propensity[:, 0], 7, axis=0)),
        axis=1,
    )
    target_ids = ("left", "right")
    first = sar.fit_sarpu_authors(
        x,
        observed,
        measured,
        propensity,
        target_ids=target_ids,
        max_its=300,
        seed=23,
    )
    permutation = np.asarray([1, 0])
    second = sar.fit_sarpu_authors(
        x,
        observed[:, permutation],
        measured[:, permutation],
        propensity[:, permutation],
        target_ids=tuple(target_ids[index] for index in permutation),
        max_its=300,
        seed=23,
    )
    np.testing.assert_allclose(
        second.predict_reference(x), first.predict_reference(x)[:, permutation]
    )
    np.testing.assert_allclose(
        second.predict_exposure(x, propensity[:, permutation]),
        first.predict_exposure(x, propensity)[:, permutation],
    )
    np.testing.assert_array_equal(
        second.classifier_seeds, first.classifier_seeds[permutation]
    )
    np.testing.assert_array_equal(
        second.propensity_seeds, first.propensity_seeds[permutation]
    )


def test_unmeasured_payload_cannot_affect_per_target_fit():
    x, observed, measured, propensity = _problem(n=140, seed=12)
    measured[100:, 0] = False
    observed[100:, 0] = 0
    first = sar.fit_sarpu_authors(
        x, observed, measured, propensity, max_its=300, seed=4
    )

    changed_x = x.astype(object)
    changed_observed = observed.astype(object)
    changed_propensity = propensity.astype(object)
    changed_x[100:] = "outside-W poison"
    changed_observed[100:, 0] = "outside-W poison"
    changed_propensity[100:] = "outside-W poison"
    second = sar.fit_sarpu_authors(
        changed_x,
        changed_observed,
        measured,
        changed_propensity,
        max_its=300,
        seed=4,
    )

    for name in (
        "classifier_coefficients",
        "classifier_intercepts",
        "propensity_coefficients",
        "propensity_intercepts",
        "target_iterations",
        "target_objective_values",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_probability_semantics_and_checkpoint_fields_are_finite_and_json_safe():
    x, observed, measured, propensity = _problem(seed=42)
    fitted = sar.fit_sarpu_authors(
        x, observed, measured, propensity, max_its=300, seed=9
    )
    p = fitted.predict_reference(x)
    e = fitted.predict_exposure(x, propensity)
    q = fitted.predict_observed(x, propensity)
    np.testing.assert_array_equal(q, p * e)
    assert np.all(np.isfinite(p)) and np.all((0 <= p) & (p <= 1))
    assert np.all(np.isfinite(e)) and np.all((0 <= e) & (e <= 1))

    payload = {}
    arrays = {}
    for field in fields(fitted):
        value = getattr(fitted, field.name)
        if isinstance(value, np.ndarray):
            arrays[field.name] = value
        else:
            payload[field.name] = value
    json.dumps(payload, allow_nan=False)
    assert arrays
    assert all(np.all(np.isfinite(value)) for value in arrays.values())
    diagnostics = fitted.diagnostics()
    assert diagnostics["source_faithful"] is False
    assert diagnostics["authors_code"] is False
    assert diagnostics["authors_source"] is True
    assert diagnostics["executes_unmodified_authors_code"] is False
    assert diagnostics["uses_unmodified_authors_em_kernel"] is True
    assert diagnostics["max_its"] == 300
    assert diagnostics["slope_eps"] == pytest.approx(1e-4)
    assert diagnostics["ll_eps"] == pytest.approx(1e-4)
    assert diagnostics["convergence_window"] == 10
    assert diagnostics["refit_classifier"] is True


def test_per_target_progress_reports_target_identity_and_completion():
    x, observed, measured, propensity = _problem(seed=43)
    events: list[dict] = []
    fitted = sar.fit_sarpu_authors(
        x,
        observed,
        measured,
        propensity,
        target_ids=("projection-A",),
        max_its=300,
        seed=10,
        on_progress=events.append,
    )

    assert fitted.converged
    assert [event["event"] for event in events] == [
        "target_start",
        "target_complete",
    ]
    assert all(event["target_id"] == "projection-A" for event in events)
    assert all(event["target_index"] == 1 for event in events)
    assert all(event["target_total"] == 1 for event in events)
    assert events[-1]["iterations"] == fitted.target_iterations[0]
    assert events[-1]["elapsed_seconds"] >= 0


def test_unsupported_target_and_hidden_positive_fail_before_upstream_fit(monkeypatch):
    x, observed, measured, propensity = _problem()
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("upstream must not be called")

    monkeypatch.setattr(sar._UPSTREAM, "pu_learn_sar_em", forbidden)
    no_positive = observed.copy()
    no_positive[:, 0] = 0
    with pytest.raises(sar.SARPUAuthorsSupportError, match="unsupported targets"):
        sar.fit_sarpu_authors(x, no_positive, measured, propensity)
    assert not called

    hidden_positive = observed.copy()
    row = int(np.flatnonzero(hidden_positive[:, 0])[0])
    hidden_measurement = measured.copy()
    hidden_measurement[row, 0] = False
    with pytest.raises(ValueError, match="outside the measurement mask"):
        sar.fit_sarpu_authors(x, hidden_positive, hidden_measurement, propensity)
    assert not called


def test_nonconvergence_fails_closed_and_api_has_no_reference_or_true_e_inputs():
    x, observed, measured, propensity = _problem()
    with pytest.raises(sar.SARPUAuthorsConvergenceError, match="did not satisfy"):
        sar.fit_sarpu_authors(
            x,
            observed,
            measured,
            propensity,
            max_its=12,
            convergence_window=10,
        )

    parameters = inspect.signature(sar.fit_sarpu_authors).parameters
    assert not ({"reference", "truth", "true_e", "e"} & set(parameters))
