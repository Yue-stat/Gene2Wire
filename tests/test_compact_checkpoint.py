"""Compact checkpoint indexes retain exact resume semantics without file storms."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import multiprocessing
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from gene2wire.checkpoint import (
    AtomicArrayCheckpointStore,
    AtomicCheckpointStore,
    CompactArrayCheckpointStore,
    CompactCheckpointStore,
)
from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.pipeline import run_experiment
from gene2wire.experiments.protocol import Settings
from gene2wire import DatasetBundle, FitConfig, ModelConfig, TuningConfig, run_model_grid


def _process_save(arguments):
    root, index = arguments
    CompactCheckpointStore(root).save_complete(
        f"process-unit-{index}", "process-fingerprint", {"value": index}
    )


def _process_array_save(arguments):
    root, index = arguments
    CompactArrayCheckpointStore(root).save_complete(
        f"array-unit-{index}", "array-fingerprint", {"value": index},
        {"prediction": np.full((20, 4), index, dtype=np.float64)},
    )


def test_compact_store_indexes_many_payloads_in_one_file(tmp_path):
    root = tmp_path / "compact"
    store = CompactCheckpointStore(root)
    for index in range(100):
        store.save_complete(
            f"trial-{index}",
            "source-and-protocol-fingerprint",
            {"validation_loss": index / 100, "candidate": {"rank": index % 5}},
        )

    assert {path.name for path in root.iterdir()} == {"index.sqlite3"}
    restored = store.load("trial-37", "source-and-protocol-fingerprint")
    assert restored is not None
    assert restored["payload"] == {
        "candidate": {"rank": 2}, "validation_loss": 0.37,
    }
    assert store.load("trial-37", "different-fingerprint") is None


def test_compact_store_is_concurrent_and_checksummed(tmp_path):
    store = CompactCheckpointStore(tmp_path)

    def save(index):
        store.save_complete(f"unit-{index}", "fingerprint", {"value": index})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(64)))
    assert all(store.load(f"unit-{index}", "fingerprint")["payload"]["value"] == index
               for index in range(64))

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE completed_checkpoint SET payload_sha256 = ? WHERE unit_key = ?",
            ("0" * 64, "unit-3"),
        )
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.load("unit-3", "fingerprint")


def test_compact_store_serializes_independent_process_writers(tmp_path):
    # Construct the schema in the parent, exactly as the experiment scheduler
    # does before Loky workers start writing completed scenario units.
    store = CompactCheckpointStore(tmp_path)
    arguments = [(str(tmp_path), index) for index in range(24)]
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        pool.map(_process_save, arguments)
    assert all(
        store.load(f"process-unit-{index}", "process-fingerprint")["payload"]
        == {"value": index}
        for index in range(24)
    )


def test_compact_store_reads_but_does_not_delete_legacy_json(tmp_path):
    legacy = AtomicCheckpointStore(tmp_path)
    legacy_path = legacy.save_complete("old-unit", "old-fingerprint", {"value": 9})

    compact = CompactCheckpointStore(tmp_path)
    restored = compact.load("old-unit", "old-fingerprint")
    assert restored is not None and restored["payload"] == {"value": 9}
    assert legacy_path.exists()

    compact.save_complete("new-unit", "new-fingerprint", {"value": 10})
    assert compact.load("new-unit", "new-fingerprint")["payload"] == {"value": 10}
    assert legacy_path.exists()

    # A newer indexed record for the same semantic key must not hide a valid
    # legacy record when the caller explicitly requests the legacy fingerprint.
    compact.save_complete("old-unit", "replacement-fingerprint", {"value": 11})
    assert compact.load("old-unit", "old-fingerprint")["payload"] == {"value": 9}


def test_compact_array_store_keeps_blobs_separate_and_reads_legacy(tmp_path):
    root = tmp_path / "arrays"
    compact = CompactArrayCheckpointStore(root)
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    compact.save_complete("new", "fingerprint", {"kind": "prediction"}, {"p": values})
    restored = compact.load("new", "fingerprint")
    assert restored["payload"] == {"kind": "prediction"}
    np.testing.assert_array_equal(restored["arrays"]["p"], values)
    assert not list(root.glob("*.json"))
    assert len(list(root.glob("arrays--*.npz"))) == 1
    assert (root / "manifest_index" / "index.sqlite3").is_file()

    legacy = AtomicArrayCheckpointStore(root)
    legacy_manifest = legacy.save_complete(
        "old", "old-fingerprint", {"kind": "fit"}, {"coef": values + 1}
    )
    old = compact.load("old", "old-fingerprint")
    assert old["payload"] == {"kind": "fit"}
    np.testing.assert_array_equal(old["arrays"]["coef"], values + 1)
    assert legacy_manifest.exists()


def test_compact_array_store_handles_process_writers_and_rejects_wrong_identity(tmp_path):
    store = CompactArrayCheckpointStore(tmp_path)
    arguments = [(str(tmp_path), index) for index in range(12)]
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        pool.map(_process_array_save, arguments)
    for index in range(12):
        restored = store.load(f"array-unit-{index}", "array-fingerprint")
        np.testing.assert_array_equal(
            restored["arrays"]["prediction"], np.full((20, 4), index)
        )
        assert store.load(f"array-unit-{index}", "wrong-fingerprint") is None

    restored = store.load("array-unit-3", "array-fingerprint")
    restored["manifest"]  # Compact manifests are indexed rather than per-fit JSON.
    def first_value(path):
        with np.load(path, allow_pickle=False) as archive:
            return archive["prediction"][0, 0]

    blob = next(path for path in tmp_path.glob("arrays--*.npz") if first_value(path) == 3)
    with blob.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="array checksum mismatch"):
        store.load("array-unit-3", "array-fingerprint")
    # A deterministic recomputation must replace the damaged content-addressed
    # blob instead of discarding the good temporary file because its name exists.
    expected = np.full((20, 4), 3, dtype=np.float64)
    store.save_complete(
        "array-unit-3", "array-fingerprint", {"value": 3},
        {"prediction": expected},
    )
    np.testing.assert_array_equal(
        store.load("array-unit-3", "array-fingerprint")["arrays"]["prediction"],
        expected,
    )


def test_raw_progress_transport_is_retained_when_final_export_fails(tmp_path):
    data = generate_simulation(
        0, 0.5, seed=9208, n_cells=90, n_targets=4,
        n_gene_features=4, n_slices=9,
    )
    settings = replace(
        Settings(), n_jobs=1, n_repetitions=1, penalties=(0.01,),
        candidate_budget=2, maxiter=20, retry_maxiter=30,
        init_direct_maxiter=10, tolerance=1e-4, loss_rates=(0.0,),
        run_information_controls=False, run_random_forest=False,
        run_mechanism_controls=False, run_calibration_controls=False,
        run_qiao=False,
    )
    with patch(
        "gene2wire.experiments.pipeline._summarize",
        side_effect=RuntimeError("deliberate export failure"),
    ), pytest.raises(RuntimeError, match="deliberate export failure"):
        run_experiment(
            data, settings, checkpoint_dir=tmp_path / "checkpoints",
            export_dir=tmp_path / "exports", progress=False,
        )
    retained = set((tmp_path / "exports").glob("**/progress/*/*.jsonl"))
    assert retained
    run_experiment(
        data, settings, checkpoint_dir=tmp_path / "checkpoints",
        export_dir=tmp_path / "exports", progress=False,
    )
    # The successful rerun removes only its fresh transport directory; it does
    # not erase the earlier failed-run diagnostics.
    assert set((tmp_path / "exports").glob("**/progress/*/*.jsonl")) == retained


def test_completed_model_wrapper_stores_prediction_once_and_repairs_refit(tmp_path):
    rng = np.random.default_rng(37)
    x = rng.normal(size=(72, 5))
    beta = rng.normal(scale=0.4, size=(5, 4))
    p = 1.0 / (1.0 + np.exp(-(x @ beta)))
    truth = rng.random(p.shape) < p
    exposure = np.full(truth.shape, 0.7)
    observed = truth & (rng.random(p.shape) < exposure)

    def bundle(rows):
        return DatasetBundle(
            x[rows], observed[rows], np.ones_like(observed[rows], dtype=bool),
            cell_ids=tuple(f"cell-{index}" for index in rows),
            target_ids=tuple(f"target-{index}" for index in range(4)),
        )

    kwargs = dict(
        train=bundle(range(42)), validation=bundle(range(42, 60)),
        test_X=x[60:], test_cell_ids=tuple(f"cell-{index}" for index in range(60, 72)),
        models=(ModelConfig(name="PU", kind="direct", pu=True),),
        tuning=TuningConfig(
            ranks=(1,), shared_l2=(0.1,), residual_l2=(0.1,),
            target_l2=(0.1,), anchor_shared_l2=0.1,
            anchor_residual_l2=0.1, anchor_target_l2=0.1,
            candidate_budget=1,
        ),
        fit=FitConfig(maxiter=100, retry_maxiter=150, tolerance=1e-6),
        train_exposure=exposure[:42], validation_exposure=exposure[42:60],
        test_exposure=exposure[60:], checkpoint_dir=tmp_path / "checkpoints",
        unit_context={"fold": 0, "condition": "compact-test"}, seed=19,
        code_version="compact-wrapper-test",
    )
    first = run_model_grid(**kwargs).models["PU"]
    for legacy_name in ("trials", "models", "refits", "warm_starts"):
        assert not (tmp_path / "checkpoints" / legacy_name).exists()
    array_files = list((tmp_path / "checkpoints").glob(
        "compact_units/*/arrays/arrays--*.npz"
    ))
    archived_names = {}
    for path in array_files:
        with np.load(path, allow_pickle=False) as archive:
            archived_names[path] = set(archive.files)
    wrappers = [path for path, names in archived_names.items()
                if names == {"latent_probability"}]
    refits = [path for path, names in archived_names.items()
              if any(name.startswith("fitted__") for name in names)]
    assert len(wrappers) == len(refits) == 1
    assert all("observed_probability" not in names for names in archived_names.values())
    np.testing.assert_array_equal(
        first.observed_probability, exposure[60:] * first.latent_probability
    )

    second = run_model_grid(**kwargs).models["PU"]
    assert second.resumed
    np.testing.assert_array_equal(second.latent_probability, first.latent_probability)
    np.testing.assert_array_equal(
        second.observed_probability, exposure[60:] * first.latent_probability
    )

    with refits[0].open("ab") as handle:
        handle.write(b"damaged-refit")
    repaired = run_model_grid(**kwargs).models["PU"]
    assert not repaired.resumed
    np.testing.assert_allclose(
        repaired.latent_probability, first.latent_probability, atol=1e-12, rtol=1e-12
    )
    assert run_model_grid(**kwargs).models["PU"].resumed
