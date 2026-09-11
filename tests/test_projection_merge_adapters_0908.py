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


def merge_overlap_fixture(*, expression=None, seed=41):
    rng = np.random.default_rng(901)
    samples = np.repeat(merge.SAMPLES, 12)
    ids = [f"{sample}_cell_{i:02d}" for sample in merge.SAMPLES for i in range(12)]
    status = np.tile(["Barcoded"] * 6 + ["Non-barcoded"] * 6, len(merge.SAMPLES))
    meta = pd.DataFrame({"sample": samples, "barcoded": status}, index=ids)
    for target_index, target in enumerate(merge.TARGETS):
        positive = (np.arange(len(meta)) + target_index) % (target_index + 3) == 0
        positive &= status == "Barcoded"
        meta[target] = np.where(positive, target, "Others")
    if expression is None:
        expression = sparse.csr_matrix(
            rng.lognormal(mean=0.0, sigma=.7, size=(len(meta), 7))
        )
    genes = ("g0", "g1", "g2", "g3", "g4", "g5", "barcode")
    dataset = merge._make_overlap_dataset(
        normalized_expression=expression,
        gene_names=genes,
        meta=meta,
        metadata={"fixture": True},
        source_gene_count=4,
        panel_design_fraction=.34,
        seed=seed,
    )
    return dataset, sparse.csr_matrix(expression), meta, genes


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


def test_merge_overlap_reserves_isolated_unassayed_design_cells_and_preserves_targets():
    dataset, _, meta, _ = merge_overlap_fixture()
    reference, measured = merge._labels_from_metadata(meta)
    design_ids = tuple(dataset.metadata["overlap_design_cell_ids"])
    assert len(design_ids) == 2 * len(merge.SAMPLES)
    assert set(design_ids).isdisjoint(dataset.cell_ids)
    design_rows = meta.index.get_indexer(design_ids)
    assert np.all(design_rows >= 0)
    assert not measured[design_rows].any()
    assert not reference[design_rows].any()
    assert (meta.iloc[design_rows]["barcoded"] == "Non-barcoded").all()
    analysis_rows = meta.index.get_indexer(dataset.cell_ids)
    np.testing.assert_array_equal(dataset.reference, reference[analysis_rows])
    np.testing.assert_array_equal(dataset.measured, measured[analysis_rows])
    assert dataset.target_ids == merge.TARGETS
    assert set(dataset.groups) == {"sample", "assay_status"}
    np.testing.assert_array_equal(
        dataset.groups["assay_status"],
        np.where(dataset.measured.any(axis=1), "assayed", "unassayed"),
    )
    assert dataset.metadata["overlap_design_counts"] == {
        sample: 2 for sample in merge.SAMPLES
    }
    assert dataset.metadata["overlap_design_rows_removed"] is True
    assert dataset.metadata["overlap_selection_reads_analysis_expression"] is False


def test_merge_overlap_source_selection_is_deterministic_and_excludes_assay_genes():
    first, expression, meta, genes = merge_overlap_fixture(seed=72)
    second, _, _, _ = merge_overlap_fixture(seed=72)
    assert first.cell_ids == second.cell_ids
    assert first.gene_names == second.gene_names
    np.testing.assert_array_equal(first.gene_matrix, second.gene_matrix)
    assert first.metadata["overlap_design_cell_ids"] == second.metadata["overlap_design_cell_ids"]
    order = np.arange(len(meta))[::-1]
    reordered = merge._make_overlap_dataset(
        normalized_expression=expression[order],
        gene_names=genes,
        meta=meta.iloc[order],
        metadata={"fixture": True},
        source_gene_count=4,
        panel_design_fraction=.34,
        seed=72,
    )
    assert set(first.metadata["overlap_design_cell_ids"]) == set(
        reordered.metadata["overlap_design_cell_ids"]
    )
    assert first.gene_names == reordered.gene_names
    assert "barcode" not in first.gene_names
    assert "barcode" in first.metadata["excluded_assay_derived_genes"]
    assert first.gene_matrix.shape == (len(first.cell_ids), 4)


def test_merge_overlap_renormalization_removes_barcode_from_library_denominator():
    matrix, excluded = merge._renormalize_without_assay_features(
        sparse.csr_matrix(np.log1p([[1.0, 3.0, 6.0], [2.0, 2.0, 0.0]])),
        ("g0", "g1", "barcode"),
    )
    linear = np.expm1(matrix.toarray())
    np.testing.assert_allclose(linear[:, :2].sum(axis=1), 1e4, rtol=1e-6)
    np.testing.assert_allclose(linear[0, :2], [2500.0, 7500.0], rtol=1e-6)
    np.testing.assert_array_equal(linear[:, 2], 0.0)
    np.testing.assert_array_equal(excluded, [False, False, True])


def test_merge_overlap_gene_selection_never_reads_analysis_expression():
    first, expression, meta, genes = merge_overlap_fixture(seed=19)
    design_ids = set(first.metadata["overlap_design_cell_ids"])
    analysis_rows = np.asarray([cell not in design_ids for cell in meta.index], dtype=bool)
    poisoned = expression.toarray()
    poisoned[analysis_rows] = np.clip(
        poisoned[analysis_rows] + np.arange(len(genes)), 0.0, 20.0
    )
    second = merge._make_overlap_dataset(
        normalized_expression=sparse.csr_matrix(poisoned),
        gene_names=genes,
        meta=meta,
        metadata={"fixture": True},
        source_gene_count=4,
        panel_design_fraction=.34,
        seed=19,
    )
    assert first.metadata["overlap_design_cell_ids"] == second.metadata["overlap_design_cell_ids"]
    assert first.metadata["overlap_source_gene_indices"] == second.metadata["overlap_source_gene_indices"]
    assert first.gene_names == second.gene_names
    assert not np.array_equal(first.gene_matrix, second.gene_matrix)


def test_merge_overlap_design_and_source_genes_are_outcome_blind():
    first, expression, meta, genes = merge_overlap_fixture(seed=37)
    changed_meta = meta.copy()
    assayed = changed_meta["barcoded"].eq("Barcoded")
    for target in merge.TARGETS:
        values = changed_meta.loc[assayed, target]
        changed_meta.loc[assayed, target] = np.where(
            values.eq(target), "Others", target
        )
    second = merge._make_overlap_dataset(
        normalized_expression=expression,
        gene_names=genes,
        meta=changed_meta,
        metadata={"fixture": True},
        source_gene_count=4,
        panel_design_fraction=.34,
        seed=37,
    )
    assert not np.array_equal(first.reference, second.reference)
    np.testing.assert_array_equal(first.measured, second.measured)
    assert first.metadata["overlap_design_cell_ids"] == second.metadata[
        "overlap_design_cell_ids"
    ]
    assert first.metadata["overlap_source_gene_indices"] == second.metadata[
        "overlap_source_gene_indices"
    ]
    assert first.gene_names == second.gene_names
    np.testing.assert_array_equal(first.gene_matrix, second.gene_matrix)


def test_merge_overlap_adapter_builds_balanced_label_blind_views():
    from gene2wire.experiments.gene_overlap import build_overlap_views

    dataset, _, _, _ = merge_overlap_fixture(seed=19)
    views = build_overlap_views(
        dataset,
        [0.0, 0.5, 1.0],
        panel_size=2,
        n_repetitions=1,
        panel_design="crossed",
        strata=("sample", "assay_status"),
        include_controls=False,
        nuisance_groups=(),
        seed=73,
    )
    assert len(views) == 3
    base_folds = dataset.split_builder(3, 11)
    for view in views:
        np.testing.assert_array_equal(view.reference, dataset.reference)
        np.testing.assert_array_equal(view.measured, dataset.measured)
        assert view.target_ids == dataset.target_ids
        assert view.metadata["experiment_context"]["nuisance_groups"] == []
        panels = np.asarray(view.groups["overlap_panel"])
        for sample in merge.SAMPLES:
            for status in ("assayed", "unassayed"):
                rows = ((np.asarray(dataset.groups["sample"]) == sample)
                        & (np.asarray(dataset.groups["assay_status"]) == status))
                counts = [int(np.sum(panels[rows] == panel)) for panel in ("A", "B")]
                assert abs(counts[0] - counts[1]) <= 1
        for original, changed in zip(base_folds, view.split_builder(3, 11)):
            np.testing.assert_array_equal(original.train_rows, changed.train_rows)
            np.testing.assert_array_equal(original.validation_rows, changed.validation_rows)
            np.testing.assert_array_equal(original.test_rows, changed.test_rows)


def test_merge_overlap_cannot_reserve_every_unassayed_cell_in_a_sample():
    _, expression, meta, genes = merge_overlap_fixture()
    with pytest.raises(ValueError, match="all-W=0 non-barcoded cell"):
        merge._make_overlap_dataset(
            normalized_expression=expression,
            gene_names=genes,
            meta=meta,
            metadata={},
            source_gene_count=4,
            panel_design_fraction=.99,
            seed=1,
        )


def test_merge_public_loaders_forward_to_separate_dataset_builders(monkeypatch, tmp_path):
    expression = object()
    genes = ("g0", "g1")
    meta = object()
    audit = {
        "feature_selection": "training_only_variance_with_detection_filter",
        "fixture": True,
    }
    raw_paths = {"metadata": tmp_path / "meta.csv", "geo_dir": tmp_path / "geo"}
    load_calls = []
    standard_calls = []
    overlap_calls = []
    standard_result = object()
    overlap_result = object()

    def fake_load_arrays(cache_dir, *, raw_paths=None):
        load_calls.append((cache_dir, raw_paths))
        return expression, genes, meta, audit

    def fake_make_dataset(**kwargs):
        standard_calls.append(kwargs)
        return standard_result

    def fake_make_overlap_dataset(**kwargs):
        overlap_calls.append(kwargs)
        return overlap_result

    monkeypatch.setattr(merge, "_load_merge_seq_arrays", fake_load_arrays)
    monkeypatch.setattr(merge, "_make_dataset", fake_make_dataset)
    monkeypatch.setattr(merge, "_make_overlap_dataset", fake_make_overlap_dataset)

    location_path = tmp_path / "location.csv"
    target_path = tmp_path / "target.csv"
    assert merge.load_merge_seq(
        tmp_path,
        n_gene_features=17,
        location_features_csv=location_path,
        target_features_csv=target_path,
        raw_paths=raw_paths,
    ) is standard_result
    assert merge.load_merge_seq_overlap(
        tmp_path,
        source_gene_count=128,
        panel_design_fraction=.25,
        seed=91,
        raw_paths=raw_paths,
    ) is overlap_result

    assert load_calls == [(tmp_path, raw_paths), (tmp_path, raw_paths)]
    assert len(standard_calls) == len(overlap_calls) == 1
    standard = standard_calls[0]
    assert standard["normalized_expression"] is expression
    assert standard["gene_names"] is genes
    assert standard["meta"] is meta
    assert standard["metadata"] is audit
    assert standard["n_gene_features"] == 17
    assert standard["location_features_csv"] == location_path
    assert standard["target_features_csv"] == target_path
    assert "source_gene_count" not in standard

    overlap = overlap_calls[0]
    assert overlap["normalized_expression"] is expression
    assert overlap["gene_names"] is genes
    assert overlap["meta"] is meta
    assert overlap["metadata"] == {
        **audit,
        "feature_selection":
            "reserved_all-W-zero_non-barcoded_top_variance_then_train_only_scaling",
    }
    assert overlap["source_gene_count"] == 128
    assert overlap["panel_design_fraction"] == .25
    assert overlap["seed"] == 91
    assert "n_gene_features" not in overlap


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
