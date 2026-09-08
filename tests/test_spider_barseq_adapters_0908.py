"""Scientific adapter checks without downloads or full experiment training."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.datasets.spider import SpiderData, spider_dataset
from gene2wire.experiments.datasets.barseq import (
    BarseqData, TARGET_FEATURE_COLUMNS, barseq_datasets, _load_barseq2_panel,
)


def _spider():
    rng = np.random.default_rng(42)
    n = 72
    location = rng.normal(size=(n, 3))
    location[:, 2] = np.repeat(np.arange(12), 6)
    z = rng.random((n, 3)) < .2
    return SpiderData(
        X_gene_raw=rng.poisson(4, (n, 4)).astype(float), X_loc_raw=location,
        Z_reference=z, W_measured=np.ones_like(z),
        cell_ids=tuple(f"s{i}" for i in range(n)), target_ids=("a", "b", "c"),
        slice_ids=np.repeat([f"slice{i}" for i in range(12)], 6),
        ap_coordinate=location[:, 2], gene_names=("g1", "g2", "g3", "g4"),
        source_sha256="synthetic_fixture",
    )


def _barseq():
    rng = np.random.default_rng(43)
    # Three animals, with two cells in each of 12 depth bins.
    n = 72
    animals = np.repeat(["A1_animal_1", "A1_animal_2", "M1_animal_3"], 24)
    panels = np.repeat(["A1", "A1", "M1"], 24)
    bins = np.tile(np.repeat(np.arange(12), 2), 3)
    w = np.zeros((n, 4), dtype=bool)
    w[:48, :2] = True
    w[48:, 2:] = True
    y = np.zeros((4, len(TARGET_FEATURE_COLUMNS)))
    y[:2, 0] = 1
    y[2:, 1] = 1
    y[:, 9] = [1, 0, 1, 0]
    y[:, 10] = [0, 1, 0, 1]
    return BarseqData(
        X_gene_raw=rng.poisson(4, (n, 3)).astype(float),
        X_loc_raw=np.column_stack((bins, rng.normal(size=(n, 2)))),
        Y_target_raw=y, Z_reference=(rng.random(w.shape) < .2) & w, W_measured=w,
        cell_ids=tuple(f"b{i}" for i in range(n)),
        target_ids=("A1::OFC", "A1::AudC", "M1::Str-r-i", "M1::Str-r-c"),
        target_panels=("A1", "A1", "M1", "M1"),
        target_regions=("OFC", "AudC", "Str-r-i", "Str-r-c"),
        target_feature_names=TARGET_FEATURE_COLUMNS, animal_ids=animals,
        panel_ids=panels,
        slice_ids=np.array([f"{a}|{b}" for a, b in zip(animals, bins)]),
        depth_bins=bins, gene_names=("g1", "g2", "g3"),
        source_sha256={"A1": "synthetic_a1", "M1": "synthetic_m1"},
    )


@pytest.mark.parametrize("n_folds", [3, 4, 6])
def test_spatial_folds_partition_and_cover_test_once(n_folds):
    datasets = [spider_dataset(_spider()), *barseq_datasets(_barseq()).values()]
    for dataset in datasets:
        folds = dataset.split_builder(n_folds, 123)
        assert len(folds) == n_folds
        test_counts = np.zeros(len(dataset.cell_ids), dtype=int)
        for fold in folds:
            fold.validate(len(dataset.cell_ids))
            test_counts[fold.test_rows] += 1
            if "animal" in dataset.groups:
                animals = set(dataset.groups["animal"])
                for rows in (fold.train_rows, fold.validation_rows, fold.test_rows):
                    assert set(dataset.groups["animal"][rows]) == animals
        np.testing.assert_array_equal(test_counts, 1)


def test_barseq_separates_panels_and_target_features_are_real_descriptors():
    datasets = barseq_datasets(_barseq())
    assert datasets["A1"].reference.shape == (48, 2)
    assert datasets["M1"].reference.shape == (24, 2)
    for panel, data in datasets.items():
        assert data.measured.all()
        assert all(t.startswith(panel + "::") for t in data.target_ids)
        train = data.split_builder(3, 1)[0].train_rows
        gene = data.feature_builder(train, False, False)
        full = data.feature_builder(train, True, True)
        assert gene.X.shape[1] == 3 and gene.Y_target is None
        assert full.X.shape[1] > 3 and "location_spline" in full.feature_blocks
        assert full.Y_target.shape[0] == 2
        assert full.metadata["target_feature_names"]
        np.testing.assert_allclose(gene.X, full.X[:, :3])
        np.testing.assert_allclose(gene.X[train].mean(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize("dataset_name", ["spider", "barseq"])
def test_feature_transforms_ignore_heldout_values_when_fitting(dataset_name):
    raw = _spider() if dataset_name == "spider" else _barseq()
    adapt = (spider_dataset if dataset_name == "spider"
             else lambda d: barseq_datasets(d)["A1"])
    data = adapt(raw)
    fold = data.split_builder(3, 0)[0]
    before = data.feature_builder(fold.train_rows, True, False)
    # A1's local row mapping equals its first 48 global rows in this fixture.
    x, loc = raw.X_gene_raw.copy(), raw.X_loc_raw.copy()
    x[fold.test_rows] += 10_000
    loc[fold.test_rows] += 100
    after = adapt(replace(raw, X_gene_raw=x, X_loc_raw=loc)).feature_builder(
        fold.train_rows, True, False
    )
    np.testing.assert_allclose(before.X[fold.train_rows], after.X[fold.train_rows])
    np.testing.assert_allclose(before.X[fold.validation_rows], after.X[fold.validation_rows])


def test_spider_external_target_descriptors_are_required_and_id_aligned(tmp_path):
    raw = _spider()
    dataset = spider_dataset(raw)
    train = dataset.split_builder(3, 0)[0].train_rows
    with pytest.raises(ValueError, match="target_features_csv"):
        dataset.feature_builder(train, False, True)
    path = tmp_path / "targets.csv"
    pd.DataFrame({"descriptor": [3., 1., 2.]}, index=["c", "a", "b"]).to_csv(path)
    supplied = spider_dataset(raw, target_features_csv=path)
    features = supplied.feature_builder(train, False, True)
    np.testing.assert_array_equal(features.Y_target[:, 0], [1., 2., 3.])
    assert supplied.feature_builder(train, True, False).X.shape[1] > 4


def test_barseq_author_filter_and_ob_control_removal(tmp_path):
    from scipy.io import savemat
    expression = np.ones((5, 23))
    expression[:, 20] = [6, 4, 6, 6, 7]
    expression[:, 21] = [3, 2, 7, 3, 1]
    projection_raw = np.array([[0, 3, 0], [0, 4, 1], [0, 5, 0],
                               [0, 2, 0], [0, 0, 8.]])
    path = tmp_path / "panel.mat"
    savemat(path, {
        "genelabels": np.array([f"g{i}" for i in range(23)], dtype=object),
        "labels": np.array(["OB", "target1", "target2"], dtype=object),
        "allbcexpmat": expression, "proj": projection_raw,
        "projraw": projection_raw, "allbcdepths": np.tile([2., 3.], (5, 1)),
        "allbcangles": np.zeros(5), "brainidx": np.ones(5),
        "allbcid": np.arange(5), "allbcseq": np.array([f"bc{i}" for i in range(5)], dtype=object),
    })
    result = _load_barseq2_panel("A1", path)
    np.testing.assert_array_equal(result["source_row"], [0, 4])
    assert result["target_names"].tolist() == ["target1", "target2"]
    np.testing.assert_allclose(result["cortical_depth_um"], 480.)
