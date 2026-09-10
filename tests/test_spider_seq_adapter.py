"""Native panel, sparse preprocessing leakage, and cache integrity checks."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from gene2wire.experiments.datasets import spider_seq as seq


def _fixture(n_per_animal=18):
    rng = np.random.default_rng(42)
    animals = np.repeat(tuple(seq.SPIDER_SEQ_PANELS), n_per_animal)
    cells = tuple(f"{animal}_cell{i}" for i, animal in enumerate(animals))
    targets = seq.SPIDER_SEQ_TARGETS
    measured = seq.make_native_panel_mask(animals, targets)
    calls = np.where(measured, rng.choice([0., 0., .01, 1.], measured.shape), np.nan)
    calls[0, measured[0]] = 0.  # This barcode-negative cell must be retained.
    frame = pd.DataFrame(calls, columns=targets, index=cells)
    frame["sample"] = animals
    frame["BC_num"] = np.sum(calls > 0, axis=1)
    x = sparse.csr_matrix(rng.poisson(2, (len(cells), 20)).astype(float))
    return x, tuple(f"g{j}" for j in range(20)), cells, frame


def _data():
    return seq.spider_seq_from_arrays(*_fixture())


def test_verified_panels_have_published_sizes_and_partial_overlap():
    a, b, c = map(set, seq.SPIDER_SEQ_PANELS.values())
    assert list(map(len, (a, b, c))) == [12, 14, 16]
    assert len(a | b | c) == 24
    assert [(len(x & y), len(x - y), len(y - x))
            for x, y in ((a, b), (a, c), (b, c))] == [(6, 6, 8), (6, 6, 10), (12, 2, 4)]


def test_measurement_mask_is_injection_design_not_positive_support():
    x, genes, cells, meta = _fixture()
    before = seq.spider_seq_from_arrays(x, genes, cells, meta)
    # Remove all detections for an actually assayed target, preserving availability.
    meta.loc[:, "CP-I"] = 0.
    meta["BC_num"] = np.sum(meta.loc[:, list(seq.SPIDER_SEQ_TARGETS)].to_numpy() > 0, axis=1)
    after = seq.spider_seq_from_arrays(x, genes, cells, meta)
    np.testing.assert_array_equal(before.W_measured, after.W_measured)
    assert after.W_measured[:, after.target_ids.index("CP-I")].all()
    assert not after.Z_reference[:, after.target_ids.index("CP-I")].any()
    assert after.cell_ids == cells and len(after.cell_ids) == 54
    assert not after.Z_reference[0].any()
    assert after.metadata["cells_without_detected_barcode"] >= 1


def test_metadata_is_aligned_by_exact_cell_ids():
    x, genes, cells, frame = _fixture()
    expected = seq.spider_seq_from_arrays(x, genes, cells, frame)
    got = seq.spider_seq_from_arrays(x, genes, cells, frame.iloc[::-1])
    np.testing.assert_array_equal(got.Z_reference, expected.Z_reference)
    with pytest.raises(ValueError, match="identical unique IDs"):
        seq.spider_seq_from_arrays(x, genes, cells, frame.iloc[:-1])


@pytest.mark.parametrize("problem", ["unknown_animal", "offpanel_positive", "missing_onpanel", "bc_num"])
def test_source_inconsistencies_fail_before_fitting(problem):
    x, genes, cells, frame = _fixture()
    if problem == "unknown_animal":
        frame.loc[cells[0], "sample"] = "unverified_sample"
    elif problem == "offpanel_positive":
        frame.loc[cells[0], "AId-C"] = .2
    elif problem == "missing_onpanel":
        frame.loc[cells[0], "CP-I"] = np.nan
    else:
        frame.loc[cells[0], "BC_num"] = 999
    with pytest.raises(ValueError):
        seq.spider_seq_from_arrays(x, genes, cells, frame)


def test_no_spatial_or_target_features_are_invented(tmp_path):
    data = _data()
    dataset = seq.spider_seq_dataset(data, n_hvg=8, n_gene_components=3)
    rows = dataset.split_builder(3, 100)[0].train_rows
    with pytest.raises(ValueError, match="location_features_csv"):
        dataset.feature_builder(rows, True, False)
    with pytest.raises(ValueError, match="target_features_csv"):
        dataset.feature_builder(rows, False, True)
    loc = tmp_path / "locations.csv"
    targets = tmp_path / "targets.csv"
    pd.DataFrame({"depth": np.arange(len(data.cell_ids))}, index=data.cell_ids).iloc[::-1].to_csv(loc)
    pd.DataFrame({"independent_descriptor": np.arange(len(data.target_ids))},
                 index=data.target_ids).iloc[::-1].to_csv(targets)
    supplied = seq.spider_seq_dataset(data, n_hvg=8, n_gene_components=3,
                                      location_features_csv=loc, target_features_csv=targets)
    feat = supplied.feature_builder(rows, True, True)
    assert feat.X.shape == (54, 4)
    np.testing.assert_allclose(feat.X[rows].mean(axis=0), 0., atol=1e-7)
    np.testing.assert_array_equal(feat.Y_target[:, 0], np.arange(24))


@pytest.mark.parametrize("n_components", [3, None])
def test_gene_selection_and_transform_fit_only_training_cells(n_components):
    data = _data()
    dataset = seq.spider_seq_dataset(data, n_hvg=8, n_gene_components=n_components)
    fold = dataset.split_builder(3, 100)[0]
    before = dataset.feature_builder(fold.train_rows, False, False)
    # Change only held-out cells, including which genes are globally variable.
    changed = data.X_gene_raw.toarray()
    changed[fold.test_rows] = 0
    changed[fold.test_rows, -1] = np.arange(len(fold.test_rows)) * 100_000
    after_data = replace(data, X_gene_raw=sparse.csr_matrix(changed))
    after = seq.spider_seq_dataset(after_data, n_hvg=8, n_gene_components=n_components).feature_builder(
        fold.train_rows, False, False)
    assert before.metadata["selected_gene_names"] == after.metadata["selected_gene_names"]
    np.testing.assert_allclose(before.X[fold.train_rows], after.X[fold.train_rows], atol=1e-8)
    np.testing.assert_allclose(before.X[fold.validation_rows], after.X[fold.validation_rows], atol=1e-8)
    np.testing.assert_allclose(before.X[fold.train_rows].mean(axis=0), 0., atol=1e-7)
    assert before.X.shape[1] == (3 if n_components else 8)


def test_preprocessing_never_densifies_all_genes(monkeypatch):
    data = _data()
    train = np.arange(15)
    original = sparse.csr_matrix.toarray
    allocations = []

    def guarded(matrix, *args, **kwargs):
        allocations.append(matrix.shape)
        assert matrix.shape[1] <= 7
        assert matrix.shape[0] <= len(train)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", guarded)
    features = seq.prepare_spider_seq_features(data, train, n_hvg=7,
                                               n_gene_components=3, transform_batch_size=4)
    assert features.X.shape == (54, 3)
    assert allocations[0] == (15, 7)
    assert max(n for n, _ in allocations[1:]) == 4


def test_within_animal_folds_are_seeded_and_cover_test_once():
    data = _data()
    dataset = seq.spider_seq_dataset(data)
    folds = dataset.split_builder(3, 55)
    same = dataset.split_builder(3, 55)
    changed = dataset.split_builder(3, 56)
    visits = np.zeros(len(data.cell_ids), dtype=int)
    for a, b in zip(folds, same):
        a.validate(len(data.cell_ids))
        np.testing.assert_array_equal(a.test_rows, b.test_rows)
        visits[a.test_rows] += 1
        for rows in (a.train_rows, a.validation_rows, a.test_rows):
            assert set(data.animal_ids[rows]) == set(seq.SPIDER_SEQ_PANELS)
    np.testing.assert_array_equal(visits, 1)
    assert not np.array_equal(folds[0].test_rows, changed[0].test_rows)


def test_processed_cache_is_pickle_free_validated_and_reusable(tmp_path, monkeypatch):
    data = _data()
    processed = tmp_path / "processed_adult_ex_v1"
    seq._save_processed(data, processed)
    restored = seq._load_processed(processed)
    np.testing.assert_array_equal(restored.X_gene_raw.toarray(), data.X_gene_raw.toarray())
    np.testing.assert_array_equal(restored.W_measured, data.W_measured)
    with np.load(processed / "adult_ex_arrays.npz", allow_pickle=False) as arrays:
        assert all(arrays[key].dtype.kind != "O" for key in arrays.files)
    # No raw parsing is needed on the second run. Raw checksum validation remains.
    import rdata
    monkeypatch.setattr(rdata.parser, "parse_file", lambda *_: pytest.fail("processed cache reparsed RDS"))
    validated = []
    monkeypatch.setattr(seq, "cached_download", lambda url, path, **kw: validated.append(kw))
    assert seq.load_spider_seq_data(tmp_path).cell_ids == data.cell_ids
    assert validated == [{"sha256": seq.SPIDER_SEQ_SHA256}]
    assert seq._load_processed(processed, source_sha256="new-source") is None
    manifest_path = processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = -1
    manifest_path.write_text(json.dumps(manifest))
    assert seq._load_processed(processed) is None


def test_corrupted_processed_cache_is_not_silently_used(tmp_path):
    seq._save_processed(_data(), tmp_path)
    with (tmp_path / "adult_ex_arrays.npz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        seq._load_processed(tmp_path)


@pytest.mark.parametrize("name,value", [("n_hvg", 0), ("n_hvg", True),
                                        ("n_gene_components", 0), ("n_gene_components", 1.5)])
def test_invalid_preprocessing_parameters_fail(name, value):
    with pytest.raises(ValueError, match=name):
        seq.spider_seq_dataset(_data(), **{name: value})
