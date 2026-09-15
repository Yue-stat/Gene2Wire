from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from gene2wire.experiments.geneml_authors import (
    GENEML_AUTHORS_PROVENANCE,
    GENEML_PORT_SOURCE_SHA256,
    GENEML_PORT_VERSION,
    GENEML_UPSTREAM_COMMIT,
    GenEMLAuthorsMaskFit,
    fit_geneml_authors_mask,
    fit_geneml_authors_port,
    verify_geneml_vendor,
)


def _problem() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(12, 3))
    observed = (rng.uniform(size=(12, 4)) < 0.2).astype(float)
    return x, observed


def _fit_kwargs() -> dict[str, object]:
    return {
        "rank": 2,
        "num_epochs": 2,
        "normalize_features": False,
        "seed": 4,
    }


def test_vendored_geneml_source_and_provenance_are_pinned() -> None:
    verified = verify_geneml_vendor()
    root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "gene2wire"
        / "_vendor"
        / "geneml"
    )
    expected_blobs = {
        "LICENSE": "775c1cc73d9108424ffa5ee6471ef2a6a5643a72",
        "README.upstream.md": "e727de456d44b0dfd5c5af6544925f59672e13a3",
        "inputs.py2": "398b75da21cd3a15e69b6d16bc39b164e34c2aeb",
        "main.py2": "7962bad437dff0354fcbaf38f400d5131b5a7977",
        "model.py2": "637848adc3ed4a4bc4e968504b849429e1f6db25",
    }
    for name, expected in expected_blobs.items():
        payload = (root / name).read_bytes()
        header = f"blob {len(payload)}\0".encode()
        assert hashlib.sha1(header + payload).hexdigest() == expected
    assert "MIT License" in (root / "LICENSE").read_text()
    assert GENEML_UPSTREAM_COMMIT == (
        "f6c08c8f3c69a009b565955231509633eda611d1"
    )
    assert GENEML_AUTHORS_PROVENANCE["port_version"] == GENEML_PORT_VERSION
    assert GENEML_AUTHORS_PROVENANCE["authors_code"] is False
    assert GENEML_AUTHORS_PROVENANCE["authors_source"] is True
    assert (
        GENEML_AUTHORS_PROVENANCE["executes_unmodified_authors_code"]
        is False
    )
    assert GENEML_AUTHORS_PROVENANCE["adapted"] is True
    assert GENEML_AUTHORS_PROVENANCE["source_commit"] == (
        GENEML_UPSTREAM_COMMIT
    )
    assert GENEML_AUTHORS_PROVENANCE["license"] == "MIT"
    assert GENEML_AUTHORS_PROVENANCE["port_source_sha256"] == (
        GENEML_PORT_SOURCE_SHA256
    )
    assert verified == GENEML_AUTHORS_PROVENANCE["vendor_sha256"]
    assert "cell features X" in GENEML_AUTHORS_PROVENANCE["training_inputs"]
    assert "measurement mask W" in GENEML_AUTHORS_PROVENANCE["training_inputs"]
    # The cache/manifest payload must remain JSON-safe.
    json.dumps(GENEML_AUTHORS_PROVENANCE, sort_keys=True)


def test_all_measured_mask_reduces_exactly_to_unmasked_author_port() -> None:
    x, observed = _problem()
    direct = fit_geneml_authors_port(x, observed, **_fit_kwargs())
    masked = fit_geneml_authors_mask(
        x,
        observed,
        np.ones_like(observed, dtype=bool),
        **_fit_kwargs(),
    )
    np.testing.assert_array_equal(masked.feature_map, direct.feature_map)
    np.testing.assert_array_equal(masked.target_factors, direct.target_factors)
    np.testing.assert_array_equal(
        masked.target_exposure, direct.target_exposure
    )
    assert masked.objective_value == direct.objective_value


def test_one_update_matches_pinned_author_source_translation_golden() -> None:
    # These values were independently produced by running one full-batch
    # update from the pinned Python-2 model.py after syntax-only 2to3
    # translation. This catches drift in the all-measured numerical equations.
    x, observed = _problem()
    fitted = fit_geneml_authors_port(
        x,
        observed,
        rank=2,
        num_epochs=1,
        normalize_features=False,
        seed=4,
    )
    expected_feature_map = np.array(
        [
            [-1.5501826494908262, -0.24079584156857708, 1.0712902605965138],
            [-0.9742377742855621, -0.8374117820081963, 1.1478298890758198],
        ]
    )
    expected_target_factors = np.array(
        [
            [-0.23592641949653625, 0.02319187484681606],
            [-0.40476712584495544, 0.00721003254875541],
            [-0.2847641110420227, -0.2656620442867279],
            [-0.21784956753253937, 0.06086056306958199],
        ]
    )
    expected_target_exposure = np.array(
        [
            0.7296048925924525,
            0.7861944603591162,
            0.8258585169886581,
            0.27073006713310527,
        ]
    )
    np.testing.assert_allclose(
        fitted.feature_map, expected_feature_map, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        fitted.target_factors,
        expected_target_factors,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        fitted.target_exposure,
        expected_target_exposure,
        rtol=2e-13,
        atol=2e-13,
    )
    assert fitted.objective_value == pytest.approx(
        0.5105402490080707, rel=2e-13, abs=2e-13
    )


def test_unmeasured_label_payload_and_fully_unmeasured_row_are_excluded() -> None:
    x, observed = _problem()
    measured = np.ones_like(observed, dtype=bool)
    measured[0] = False
    measured[[1, 2], 0] = False
    measured[[3, 4], 1] = False
    measured[[5, 6], 2] = False
    measured[[7, 8], 3] = False

    noisy_observed = observed.copy()
    noisy_observed[~measured] = np.nan
    changed_x = x.copy()
    changed_x[0] = np.array([1e6, -2e6, 3e6])
    first = fit_geneml_authors_mask(
        x, observed, measured, **_fit_kwargs()
    )
    second = fit_geneml_authors_mask(
        changed_x, noisy_observed, measured, **_fit_kwargs()
    )
    np.testing.assert_array_equal(first.feature_map, second.feature_map)
    np.testing.assert_array_equal(first.target_factors, second.target_factors)
    np.testing.assert_array_equal(
        first.target_exposure, second.target_exposure
    )


def test_predictions_keep_author_global_target_exposure_semantics() -> None:
    x, observed = _problem()
    fitted = fit_geneml_authors_port(x, observed, **_fit_kwargs())
    assert isinstance(fitted, GenEMLAuthorsMaskFit)
    reference = fitted.predict_reference(x)
    exposure = fitted.predict_exposure(x)
    observed_probability = fitted.predict_observed(x)
    assert reference.shape == observed.shape
    assert exposure.shape == observed.shape
    assert np.all((reference >= 0.0) & (reference <= 1.0))
    assert np.all((exposure >= 0.0) & (exposure <= 1.0))
    np.testing.assert_allclose(
        exposure,
        np.broadcast_to(fitted.target_exposure, exposure.shape),
    )
    np.testing.assert_allclose(
        observed_probability, reference * exposure
    )
    np.testing.assert_array_equal(
        fitted.predict_exposure(), fitted.target_exposure
    )
    diagnostics = fitted.diagnostics()
    assert diagnostics["exposure_model"] == (
        "one_global_probability_per_target"
    )
    assert diagnostics["upstream_commit"] == GENEML_UPSTREAM_COMMIT
    assert diagnostics["port_version"] == GENEML_PORT_VERSION
    assert diagnostics["port_source_sha256"] == GENEML_PORT_SOURCE_SHA256
    assert diagnostics["authors_code"] is False
    assert diagnostics["authors_source"] is True
    assert diagnostics["executes_unmodified_authors_code"] is False
    assert diagnostics["adapted"] is True
    assert "measurement mask W" in diagnostics["training_inputs"]


def test_sparse_author_style_inputs_match_dense_adapter() -> None:
    x, observed = _problem()
    measured = np.ones_like(observed, dtype=bool)
    dense = fit_geneml_authors_mask(
        x, observed, measured, **_fit_kwargs()
    )
    sparse_fit = fit_geneml_authors_mask(
        sparse.csr_matrix(x),
        sparse.csr_matrix(observed),
        sparse.csr_matrix(measured),
        **_fit_kwargs(),
    )
    np.testing.assert_array_equal(dense.feature_map, sparse_fit.feature_map)
    np.testing.assert_array_equal(
        dense.target_factors, sparse_fit.target_factors
    )
    np.testing.assert_array_equal(
        dense.target_exposure, sparse_fit.target_exposure
    )


def test_fit_api_is_truth_blind_and_unsupported_masks_fail_closed() -> None:
    forbidden = {
        "reference",
        "truth",
        "hidden",
        "y_ref",
        "test",
        "generator_propensity",
    }
    for function in (fit_geneml_authors_mask, fit_geneml_authors_port):
        parameters = {
            name.lower() for name in inspect.signature(function).parameters
        }
        assert not parameters.intersection(forbidden)

    x, observed = _problem()
    measured = np.ones_like(observed, dtype=bool)
    measured[:, 2] = False
    with pytest.raises(ValueError, match="no measured entries"):
        fit_geneml_authors_mask(
            x, observed, measured, **_fit_kwargs()
        )
    with pytest.raises(ValueError, match="must divide"):
        fit_geneml_authors_mask(
            x,
            observed,
            np.ones_like(observed, dtype=bool),
            batch_size=5,
            rank=2,
            num_epochs=1,
        )


@pytest.mark.parametrize("invalid", [np.nan, -1.0, 0.5, 2.0])
def test_measurement_mask_must_be_finite_binary(invalid: float) -> None:
    x, observed = _problem()
    measured = np.ones_like(observed, dtype=float)
    measured[0, 0] = invalid
    with pytest.raises(ValueError, match="finite and binary"):
        fit_geneml_authors_mask(
            x, observed, measured, **_fit_kwargs()
        )
