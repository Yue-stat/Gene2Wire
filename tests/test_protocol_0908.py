"""Freeze-matrix and persisted identity checks for the common 0908 protocol."""

from dataclasses import replace
import json

import numpy as np
import pytest

from gene2wire.experiments import io, protocol
from gene2wire.experiments.protocol import Settings, fingerprint, scenarios


def test_user_defaults_and_hardware_independent_scientific_identity():
    settings = Settings()
    assert (settings.n_outer_folds, settings.n_repetitions, settings.n_jobs) == (3, 5, 32)
    assert settings.use_location is False and settings.use_target_features is False
    assert settings.strategy == "full_joint"
    assert settings.paired_fraction == .2
    original = fingerprint(settings.scientific_dict())
    assert original == fingerprint(replace(settings, n_jobs=1).scientific_dict())
    for altered in (
        replace(settings, use_location=True), replace(settings, use_target_features=True),
        replace(settings, n_repetitions=6), replace(settings, paired_fraction=.4),
        replace(settings, strategy="staged_rank_l2"), replace(settings, seed=settings.seed + 1),
    ):
        assert original != fingerprint(altered.scientific_dict())


def test_dimension_driven_ranks_and_uniform_target_feature_switch():
    settings = Settings()
    assert settings.tuning_config(3, 35).ranks == (0, 1, 2, 3)
    assert settings.tuning_config(16, 36).ranks == (0, 1, 2, 4, 8, 16)
    assert settings.tuning_config(23, 11).ranks == (0, 1, 2, 4, 8, 11)
    for enabled in (False, True):
        chosen = replace(settings, use_target_features=enabled)
        assert len(chosen.models()) == 6
        assert all(model.use_target_features is enabled for model in chosen.models())
        assert sum(model.pu for model in chosen.models()) == 3
        assert chosen.tuning_config(16, 36).include_endpoints
    with pytest.raises(ValueError, match="require USE_TARGET_FEATURES"):
        replace(settings, run_qiao=True)


def test_real_data_scenarios_do_not_expand_to_control_cartesian_product():
    settings = Settings()
    natural = scenarios(settings, natural=True, simulation=False)
    assert len(natural) == 1
    assert natural[0]["mechanism"] == "natural" and natural[0]["loss_rate"] is None
    empirical = scenarios(settings, natural=False, simulation=False)
    assert len(empirical) == 5
    assert {row["analysis"] for row in empirical} == {"primary"}
    assert {row["mechanism"] for row in empirical} == {"technical_sar"}
    assert {row["loss_rate"] for row in empirical} == {0., .2, .4, .6, .8}
    expanded_fractions = replace(settings, calibration_fractions=(.05, .1, .2, .4, .8))
    assert scenarios(expanded_fractions, natural=False, simulation=False) == empirical


@pytest.mark.parametrize("rho, expected_count", [(0., 11), (.5, 7), (1., 11)])
def test_simulation_controls_have_predeclared_endpoint_scope(rho, expected_count):
    matrix = scenarios(Settings(), natural=False, simulation=True, sharing_strength=rho)
    assert len(matrix) == expected_count
    assert len({fingerprint(row) for row in matrix}) == len(matrix)
    for row in matrix:
        if row["analysis"] != "primary":
            assert row["loss_rate"] == .8
        if row["analysis"].startswith("calibration"):
            assert rho in (0., 1.)
            assert row["mechanism"] == "technical_sar"
        if row["analysis"] == "calibration_misspecification":
            assert row["calibration_fraction"] == .2
        if row["analysis"] == "calibration_size":
            assert row["calibration_fraction"] in (.1, .4)
            assert row["calibration_spec"] == "correct"
    specs = {row["calibration_spec"] for row in matrix}
    assert specs == ({"correct"} if rho == .5 else {"correct", "omit_technical", "pooled_target"})


def test_control_switches_remove_only_the_corresponding_axis():
    settings = Settings(run_mechanism_controls=False, run_calibration_controls=False)
    assert len(scenarios(settings, natural=False, simulation=True, sharing_strength=0.)) == 5
    with_calibration = replace(settings, run_calibration_controls=True)
    matrix = scenarios(with_calibration, natural=False, simulation=True, sharing_strength=0.)
    assert len(matrix) == 9
    assert not any(row["analysis"] == "mechanism" for row in matrix)


def test_source_manifest_hash_changes_for_nested_adapter_source(tmp_path, monkeypatch):
    package = tmp_path / "gene2wire"
    nested = package / "experiments" / "datasets"
    nested.mkdir(parents=True)
    marker = package / "experiments" / "protocol.py"
    marker.write_text("version = 1\n")
    adapter = nested / "example.py"
    adapter.write_text("setting = 1\n")
    monkeypatch.setattr(protocol, "__file__", str(marker))
    before = protocol.source_hash()
    adapter.write_text("setting = 2\n")
    assert protocol.source_hash() != before
    # Non-source output artifacts are not scientific source changes.
    current = protocol.source_hash()
    (nested / "preview.pdf").write_bytes(b"temporary preview")
    assert protocol.source_hash() == current


def test_raw_cache_reuses_verified_bytes_without_network_and_detects_tampering(tmp_path, monkeypatch):
    path = tmp_path / "raw.dat"
    path.write_bytes(b"already downloaded scientific data")

    def forbidden_network(*args, **kwargs):
        raise AssertionError("Validated cache must not contact the internet")

    monkeypatch.setattr(io, "urlopen", forbidden_network)
    assert io.cached_download("https://example.invalid/raw", path) == path
    sidecar = path.with_name(path.name + ".download.json")
    recorded = json.loads(sidecar.read_text())
    assert recorded["sha256"] == io.file_hash(path)
    assert recorded["adopted_existing_file"] is True
    assert io.cached_download("https://example.invalid/raw", path) == path
    path.write_bytes(b"silently changed input")
    with pytest.raises(ValueError, match="Raw cache changed"):
        io.cached_download("https://example.invalid/raw", path)


def test_manifests_are_strict_json_and_atomic_arrays_reopen(tmp_path):
    manifest = tmp_path / "manifest.json"
    io.atomic_json({"gamma": None, "undefined_metric": np.nan,
                    "count": np.int64(3), "rows": np.array([1, 4, 6])}, manifest)
    text = manifest.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"gamma": None, "undefined_metric": None, "count": 3, "rows": [1, 4, 6]}
    arrays = tmp_path / "predictions.npz"
    expected = np.arange(12).reshape(4, 3)
    io.atomic_npz(arrays, prediction=expected, measured=expected > 3)
    with np.load(arrays, allow_pickle=False) as restored:
        np.testing.assert_array_equal(restored["prediction"], expected)
        np.testing.assert_array_equal(restored["measured"], expected > 3)
