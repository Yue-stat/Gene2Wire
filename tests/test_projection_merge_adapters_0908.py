"""Dataset-level invariants using local fixtures; no network or fitting needed."""
import gzip
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

from gene2wire.experiments.datasets import merge_seq as merge
from gene2wire.experiments.datasets import projection_tags as projection


def projection_fixture(tmp_path=None, **options):
    rng = np.random.default_rng(18)
    animals = np.repeat(np.arange(1, 7), 12)
    meta = pd.DataFrame({"Animal_ID": animals,
                         "Platform": np.where(animals <= 4, "10X RNA", "10X multiome"),
                         "Origin": np.where(animals % 2 == 1, "MO", "SSC")},
                        index=[f"cell_{i}" for i in range(len(animals))])
    w = np.ones((len(meta), len(projection.TARGETS)), dtype=bool)
    w[animals <= 4, 1] = False
    z = (rng.uniform(size=w.shape) < .3) & w
    d = z & (rng.uniform(size=w.shape) < .5)
    genes = rng.poisson(5, size=(len(meta), len(projection.GENE_FEATURES)))
    library = genes.sum(axis=1) + 100
    dataset = projection._make_dataset(gene_counts=genes, library_size=library, meta=meta,
                                      standard=d, reference=z, measured=w, metadata={}, **options)
    return dataset, genes, library, meta, d, z, w


def merge_fixture(**options):
    rng = np.random.default_rng(29)
    samples = np.repeat(merge.SAMPLES, [21, 25, 29, 32])
    meta = pd.DataFrame({"sample": samples, "barcoded": "Barcoded"},
                        index=[f"{s}_{i}" for i, s in enumerate(samples)])
    for target in merge.TARGETS:
        meta[target] = np.where(rng.uniform(size=len(meta)) < .2, target, "Others")
    meta.loc[meta.index[::11], ["barcoded", *merge.TARGETS]] = ["Non-barcoded", *("Others" for _ in merge.TARGETS)]
    expression = sparse.csr_matrix(rng.uniform(.1, 5, size=(len(meta), 8)))
    dataset = merge._make_dataset(normalized_expression=expression, gene_names=tuple(f"g{i}" for i in range(8)),
                                 meta=meta, metadata={}, n_gene_features=4, **options)
    return dataset, expression, meta


def test_projection_default_gene_only_and_exact_animal_split():
    dataset, *_ = projection_fixture()
    folds = dataset.split_builder(3, 42)
    assert [f.metadata["test_groups"] for f in folds] == [["1", "4"], ["2", "6"], ["3", "5"]]
    assert [f.metadata["validation_groups"] for f in folds] == [["2"], ["4"], ["1"]]
    x = dataset.feature_builder(folds[0].train_rows, False, False)
    loc = dataset.feature_builder(folds[0].train_rows, True, False)
    assert x.X.shape[1] == 32 and loc.X.shape[1] == 33
    assert np.allclose(x.X, loc.X[:, :32])
    assert np.allclose(x.X[folds[0].train_rows].mean(axis=0), 0, atol=1e-12)
    assert x.Y_target is None
    with pytest.raises(ValueError, match="no native target"):
        dataset.feature_builder(folds[0].train_rows, False, True)
    assert np.sum(dataset.natural_observed) <= np.sum(dataset.reference)


def test_projection_scaler_is_fitted_on_training_only():
    first, genes, library, meta, d, z, w = projection_fixture()
    fold = first.split_builder(3, 0)[0]
    changed = genes.copy()
    changed[fold.test_rows] *= 100
    second = projection._make_dataset(gene_counts=changed, library_size=library, meta=meta,
                                      standard=d, reference=z, measured=w, metadata={})
    x1 = first.feature_builder(fold.train_rows, False, False).X
    x2 = second.feature_builder(fold.train_rows, False, False).X
    assert np.array_equal(x1[fold.train_rows], x2[fold.train_rows])


@pytest.mark.parametrize("n_folds", [2, 4, 5, 6])
def test_projection_other_group_fold_counts_preserve_assay_coverage(n_folds):
    dataset, *_ = projection_fixture()
    for fold in dataset.split_builder(n_folds, 31):
        assert dataset.measured[fold.train_rows].any(axis=0).all()
        assert set(dataset.platform[fold.train_rows]) == set(dataset.platform)


def test_external_features_align_by_id_and_toggle(tmp_path):
    base, *_ = projection_fixture()
    target_path = tmp_path / "targets.csv"
    values = pd.DataFrame({"feature": np.arange(len(base.target_ids), dtype=float)}, index=base.target_ids)
    values.iloc[::-1].to_csv(target_path)
    dataset, *_ = projection_fixture(target_features_csv=target_path)
    fitted = dataset.feature_builder(dataset.split_builder(3, 0)[0].train_rows, False, True)
    assert np.array_equal(fitted.Y_target[:, 0], np.arange(len(base.target_ids)))
    values.iloc[:-1].to_csv(target_path)
    with pytest.raises(ValueError, match="missing aligned IDs"):
        projection_fixture(target_features_csv=target_path)


@pytest.mark.parametrize("n_folds", [3, 4])
def test_merge_group_splits_are_complete_disjoint_and_label_blind(n_folds):
    dataset, expression, meta = merge_fixture()
    folds = dataset.split_builder(n_folds, 73)
    visits = np.zeros(len(meta), dtype=int)
    for fold in folds:
        fold.validate(len(meta))
        visits[fold.test_rows] += 1
        groups = [set(meta.iloc[rows]["sample"]) for rows in (fold.train_rows, fold.validation_rows, fold.test_rows)]
        assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])
        assert fold.metadata["assignment_uses_outcomes"] is False
    assert np.all(visits == 1)
    assert sorted(len(f.metadata["test_groups"]) for f in folds) == ([1, 1, 2] if n_folds == 3 else [1, 1, 1, 1])
    meta.loc[:, list(merge.TARGETS)] = "Others"
    changed = merge._make_dataset(normalized_expression=expression, gene_names=tuple(f"g{i}" for i in range(8)),
                                  meta=meta, metadata={}, n_gene_features=4)
    assert all(np.array_equal(a.test_rows, b.test_rows) for a, b in zip(folds, changed.split_builder(n_folds, 73)))


def test_merge_train_only_gene_selection_and_unassayed_mask():
    dataset, expression, meta = merge_fixture()
    fold = dataset.split_builder(3, 0)[0]
    x = dataset.feature_builder(fold.train_rows, False, False)
    dense = expression.toarray()
    dense[fold.test_rows, 0] *= 1000
    other = merge._make_dataset(normalized_expression=sparse.csr_matrix(dense), gene_names=tuple(f"g{i}" for i in range(8)),
                                meta=meta, metadata={}, n_gene_features=4)
    x2 = other.feature_builder(fold.train_rows, False, False)
    assert x.feature_names == x2.feature_names
    assert np.array_equal(x.X[fold.train_rows], x2.X[fold.train_rows])
    assert np.allclose(x.X[fold.train_rows].mean(axis=0), 0, atol=1e-12)
    assert not dataset.measured[::11].any()
    assert not dataset.reference[::11].any()
    with pytest.raises(ValueError, match="pseudotime"):
        dataset.feature_builder(fold.train_rows, True, False)
    with pytest.raises(ValueError, match="no native target"):
        dataset.feature_builder(fold.train_rows, False, True)


def test_merge_geo_alignment_supports_trailing_barcode_suffix(tmp_path):
    files = {kind: tmp_path / f"{kind}.gz" for kind in ("genes", "barcodes", "matrix")}
    with gzip.open(files["genes"], "wt") as handle:
        handle.write("id1\tg1\nid2\tg2\n")
    with gzip.open(files["barcodes"], "wt") as handle:
        handle.write("AA-1\nBB-1\nCC-1\n")
    with gzip.open(files["matrix"], "wb") as handle:
        mmwrite(handle, sparse.coo_matrix([[1, 2, 3], [4, 5, 6]]))
    x, names, n = merge._load_selected_10x_sample(files, "pfc_1", ["pfc_1_CC", "pfc_1_AA-1"])
    assert n == 3 and names == ("g1", "g2")
    assert np.array_equal(x.toarray(), [[3, 6], [1, 4]])
    with pytest.raises(ValueError, match="one-to-one"):
        merge._load_selected_10x_sample(files, "pfc_1", ["pfc_1_MISSING"])


def test_id_canonicalization_and_gene_name_collisions():
    assert projection.canonical_standard_cell_id("RNA_19_SSC__ABC-1") == "GEX__ABC-4"
    assert projection.canonical_standard_cell_id("Parse_07_SSC__ABC") == "Parse_1__ABC"
    assert merge._make_unique_gene_names(["A", "A", "A.1", "A"]) == ("A", "A.2", "A.1", "A.3")


def test_merge_full_loader_reuses_raw_and_processed_cache_offline(tmp_path, monkeypatch):
    from gene2wire.experiments import io

    def no_network(*args, **kwargs):
        raise AssertionError("A complete local raw cache must not access the network")

    monkeypatch.setattr(io, "urlopen", no_network)
    meta_rows = []
    geo = tmp_path / "GSE210172"
    geo.mkdir()
    for sample, (_, stem) in merge.GEO_SAMPLE_FILES.items():
        for i in range(12):
            row = {"cell": f"{sample}_BC{i}", "sample": sample, "barcoded": "Barcoded"}
            row.update({target: target if i % 2 == 0 else "Others" for target in merge.TARGETS})
            meta_rows.append(row)
        for suffix, text in (("genes.tsv.gz", "id1\tg1\nid2\tg2\nid3\tg3\n"),
                             ("barcodes.tsv.gz", "".join(f"BC{i}-1\n" for i in range(12)))):
            with gzip.open(geo / f"{stem}_{suffix}", "wt") as handle:
                handle.write(text)
        with gzip.open(geo / f"{stem}_matrix.mtx.gz", "wb") as handle:
            mmwrite(handle, sparse.coo_matrix(np.tile(np.arange(1, 13), (3, 1))))
    metadata_path = tmp_path / "exn_meta_valid.csv"
    pd.DataFrame(meta_rows).set_index("cell").to_csv(metadata_path)
    monkeypatch.setattr(merge, "METADATA_BLOB_SHA", io.file_hash(metadata_path, git_blob=True))
    first = merge.load_merge_seq(tmp_path, n_gene_features=2)
    assert first.reference.shape == (48, 5)
    assert len(first.split_builder(3, 0)) == 3
    assert len(list(tmp_path.rglob("*.download.json"))) == 13
    monkeypatch.setattr(merge, "mmread", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Processed cache must bypass matrix parsing")))
    second = merge.load_merge_seq(tmp_path, n_gene_features=2)
    assert np.array_equal(first.reference, second.reference)
    rows = first.split_builder(3, 0)[0].train_rows
    assert np.array_equal(first.feature_builder(rows, False, False).X,
                          second.feature_builder(rows, False, False).X)


def test_projection_full_loader_preserves_filters_and_offline_cache(tmp_path, monkeypatch):
    from gene2wire.experiments import io

    monkeypatch.setattr(io, "urlopen", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Unexpected network")))
    ids = [f"cell{i}" for i in range(8)]
    standard = pd.DataFrame({f"BC{j}.counts": np.ones(8, dtype=int) for j in range(1, 8)}, index=ids)
    union = pd.DataFrame({target: np.ones(8, dtype=int) for target in projection.TARGETS}, index=ids)
    union["Animal_ID"] = [1, 2, 3, 4, 5, 6, 1, 2]
    union["Platform"] = ["10X RNA"] * 4 + ["10X multiome"] * 2 + ["Parse", "10X RNA"]
    union["Celltype"] = ["L2/3 IT"] * 7 + ["Astro"]
    union["Origin"] = ["MO", "SSC"] * 4
    animal_rows = []
    for animal in range(1, 7):
        row = {"Animal ID": f"Animal_SEQ_{animal}"}
        row.update({f"BC injected into {t}": (None if t == "SSp" and animal <= 4 else f"BC{j}")
                    for j, t in enumerate(projection.TARGETS, start=1)})
        animal_rows.append(row)
    files = {key: tmp_path / source[0] for key, source in projection.SOURCES.items()}
    standard.to_csv(files["standard"])
    union.to_csv(files["union"])
    # The official workbook contains a second Animal ID header for an HCR
    # cohort.  Preserve that layout so the loader cannot rely on a unique
    # header cell or accidentally select non-sequencing animals.
    with pd.ExcelWriter(files["animals"]) as writer:
        pd.DataFrame([["Animals used for sequencing"]]).to_excel(
            writer, sheet_name="Animals", index=False, header=False)
        pd.DataFrame(animal_rows).to_excel(
            writer, sheet_name="Animals", index=False, startrow=1)
        pd.DataFrame([["Animals used for HCR"]]).to_excel(
            writer, sheet_name="Animals", index=False, header=False, startrow=9)
        pd.DataFrame([
            {"Animal ID": "Animal_FISH_1", **{
                f"BC injected into {target}": f"BC{j}"
                for j, target in enumerate(projection.TARGETS, start=1)
            }}
        ]).to_excel(writer, sheet_name="Animals", index=False, startrow=10)
    files["expression"].write_bytes(b"fixture RDS placeholder")
    sources = {key: (path.name, projection.SOURCES[key][1], io.file_hash(path)) for key, path in files.items()}
    monkeypatch.setattr(projection, "SOURCES", sources)
    genes = (*projection.GENE_FEATURES, *projection.VECTOR_FEATURES, "background")
    matrix = sparse.csc_matrix(np.ones((len(genes), len(ids))))
    obj = SimpleNamespace(Dimnames=(np.array(genes), np.array(ids)), Dim=matrix.shape,
                          x=matrix.data, i=matrix.indices, p=matrix.indptr)
    parser = SimpleNamespace(parse_file=lambda path: obj)
    monkeypatch.setitem(sys.modules, "rdata", SimpleNamespace(parser=parser, conversion=SimpleNamespace(convert=lambda x: x)))
    first = projection.load_projection_tags(tmp_path)
    assert first.reference.shape == (6, 7)
    assert first.metadata["strict_matched_cells"] == 8
    assert first.metadata["excluded_parse_matched_cells"] == 1
    assert first.metadata["excluded_celltype_after_platform"] == 1
    assert np.array_equal(first.measured[:, 1], [0, 0, 0, 0, 1, 1])
    parser.parse_file = lambda *a: (_ for _ in ()).throw(AssertionError("Processed cache must bypass RDS parsing"))
    second = projection.load_projection_tags(tmp_path)
    assert np.array_equal(first.reference, second.reference)
    assert len(list(tmp_path.glob("*.download.json"))) == 4
