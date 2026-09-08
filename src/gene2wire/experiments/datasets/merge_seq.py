"""MERGE-seq excitatory cohort aligned to cached GEO filtered 10x matrices.

Non-barcoded rows remain structurally unassayed, not negative labels. Variable
gene selection/scaling is trained only on supplied training rows. Sample splits
never inspect projection outcomes, and no notebook-local model patch is used.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.preprocessing import StandardScaler

from ..contracts import ExperimentDataset, FeatureSet
from .projection_tags import _aligned_numeric_csv, _group_folds, _training_rows

METADATA_URL = ("https://raw.githubusercontent.com/MichaelPeibo/MERGE-seq-analysis/"
                "refs/heads/revised/Fig2%26S2/exn_meta_valid.csv")
METADATA_BLOB_SHA = "2d0ab3336d4518ac00353b96618ca7beb797fed2"
GEO_SAMPLE_FILES = {
    "pfc_1": ("GSM6422992", "GSM6422992_mouse_1"),
    "pfc_2": ("GSM6422993", "GSM6422993_mouse_2"),
    "pfc_3": ("GSM6422994", "GSM6422994_mouse_3"),
    "pfc_4": ("GSM6422995", "GSM6422995_mouse_4-6"),
}
TARGETS = ("AI_valid", "DMS_valid", "MD_valid", "BLA_valid", "LH_valid")
SAMPLES = tuple(GEO_SAMPLE_FILES)


def _read_gzip_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [x.rstrip("\r\n") for x in handle if x.strip()]


def _make_unique_gene_names(names: Sequence[str]) -> tuple[str, ...]:
    # Reserve existing names so duplicate A cannot take an original A.1 name.
    reserved, used, result = set(map(str, names)), set(), []
    for value in names:
        name, suffix = str(value), 1
        candidate = name
        if candidate in used:
            while True:
                candidate = f"{name}.{suffix}"
                suffix += 1
                if candidate not in used and candidate not in reserved:
                    break
        used.add(candidate)
        result.append(candidate)
    return tuple(result)


def _canonical_10x_barcode(value):
    text = str(value).strip()
    return text[:-2] if text.endswith("-1") else text


def _load_selected_10x_sample(files, sample, requested_cell_ids, expected_gene_names=None):
    barcodes = _read_gzip_lines(files["barcodes"])
    index = pd.Index([_canonical_10x_barcode(x) for x in barcodes])
    if not index.is_unique:
        raise ValueError(f"{sample} GEO cell barcodes are not unique")
    prefix = sample + "_"
    requested = []
    for cell in requested_cell_ids:
        if not str(cell).startswith(prefix):
            raise ValueError(f"Unexpected {sample} metadata cell ID: {cell}")
        requested.append(_canonical_10x_barcode(str(cell)[len(prefix):]))
    positions = index.get_indexer(requested)
    if np.any(positions < 0) or len(np.unique(positions)) != len(positions):
        missing = np.asarray(requested, dtype=str)[positions < 0][:5].tolist()
        raise ValueError(f"{sample} metadata does not map one-to-one to GEO columns: {missing}")
    gene_lines = _read_gzip_lines(files["genes"])
    genes = _make_unique_gene_names([x.split("\t")[1] if "\t" in x else x for x in gene_lines])
    if expected_gene_names is not None and tuple(expected_gene_names) != genes:
        raise ValueError(f"{sample} gene order differs from previous GEO samples")
    with gzip.open(files["matrix"], "rb") as handle:
        matrix = sparse.csc_matrix(mmread(handle), dtype=np.float32)
    if matrix.shape != (len(genes), len(barcodes)):
        raise ValueError(f"{sample} matrix dimensions do not match genes/barcodes")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0) or not np.allclose(matrix.data, np.round(matrix.data)):
        raise ValueError(f"{sample} matrix must contain finite nonnegative counts")
    selected = matrix[:, positions].T.tocsr()
    selected.eliminate_zeros()
    return selected, genes, len(barcodes)


def _normalize_counts(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("Retained MERGE-seq cells must have positive expression library size")
    normalized = (sparse.diags((1e4 / totals).astype(np.float32)) @ matrix).tocsr()
    normalized.data = np.log1p(normalized.data)
    normalized.eliminate_zeros()
    return normalized


def _labels_from_metadata(metadata):
    required = {"sample", "barcoded", *TARGETS}
    if required - set(metadata):
        raise ValueError(f"MERGE-seq metadata is missing columns: {sorted(required - set(metadata))}")
    if not metadata.index.is_unique or set(metadata["sample"].astype(str)) != set(SAMPLES):
        raise ValueError("MERGE-seq requires unique cells in the four audited pfc samples")
    status = metadata["barcoded"].astype(str).str.strip()
    if not set(status).issubset({"Barcoded", "Non-barcoded"}):
        raise ValueError("Unknown MERGE-seq barcoding status")
    measured = np.repeat(status.eq("Barcoded").to_numpy()[:, None], len(TARGETS), axis=1)
    reference = np.zeros_like(measured)
    for j, target in enumerate(TARGETS):
        values = metadata[target].astype(str).str.strip()
        if not set(values).issubset({target, "Others"}):
            raise ValueError(f"Unknown {target} reference value")
        reference[:, j] = values.eq(target)
    if np.any(reference & ~measured):
        raise ValueError("Reference positives cannot occur in non-barcoded cells")
    return reference, measured


def _make_dataset(*, normalized_expression, gene_names, meta, metadata,
                  n_gene_features=128, location_features_csv=None, target_features_csv=None):
    expression = sparse.csr_matrix(normalized_expression, dtype=np.float32)
    genes = tuple(map(str, gene_names))
    if expression.shape != (len(meta), len(genes)) or len(set(genes)) != len(genes):
        raise ValueError("MERGE-seq expression and unique genes must align to metadata")
    if not np.isfinite(expression.data).all() or np.any(expression.data < 0):
        raise ValueError("MERGE-seq normalized expression is invalid")
    if int(n_gene_features) != n_gene_features or not 1 <= n_gene_features <= len(genes):
        raise ValueError("n_gene_features must be positive and no larger than the available panel")
    n_gene_features = int(n_gene_features)
    reference, measured = _labels_from_metadata(meta)
    ids = tuple(meta.index.astype(str))
    samples = meta["sample"].astype(str).to_numpy()
    location, location_names = _aligned_numeric_csv(location_features_csv, ids, "Location")
    target, target_names = _aligned_numeric_csv(target_features_csv, TARGETS, "Target feature")

    def features(train_rows, use_location=False, use_target_features=False):
        train = _training_rows(train_rows, len(ids))
        if use_location and location is None:
            raise ValueError("MERGE-seq has no native physical cell coordinates; pseudotime is "
                             "not location. Supply location_features_csv indexed by cell ID")
        if use_target_features and target is None:
            raise ValueError("MERGE-seq has no native target expression descriptors; supply "
                             "target_features_csv indexed by the five target IDs")
        fitted = expression[train]
        mean = np.asarray(fitted.mean(axis=0)).ravel().astype(float)
        variance = np.maximum(np.asarray(fitted.multiply(fitted).mean(axis=0)).ravel() - mean**2, 0)
        detected = np.asarray(fitted.getnnz(axis=0)).ravel()
        minimum_detected = max(10, int(math.ceil(.01 * len(train))))
        eligible = np.flatnonzero((detected >= minimum_detected) & np.isfinite(variance))
        if len(eligible) < n_gene_features:
            raise ValueError(f"Only {len(eligible)} genes meet training detection >= {minimum_detected}; "
                             f"requested n_gene_features={n_gene_features}")
        selected = eligible[np.lexsort((eligible, -variance[eligible]))[:n_gene_features]]
        dense = expression[:, selected].toarray().astype(float)
        x = StandardScaler().fit(dense[train]).transform(dense)
        names = tuple(genes[j] for j in selected)
        blocks = {"gene": tuple(range(len(names)))}
        if use_location:
            loc = StandardScaler().fit(location[train]).transform(location)
            blocks["location"] = tuple(range(x.shape[1], x.shape[1] + loc.shape[1]))
            x = np.column_stack((x, loc))
            names += location_names
        return FeatureSet(x, blocks, target if use_target_features else None, names,
                          {"preprocessing_fit_rows": train.tolist(), "selected_gene_indices": selected.tolist(),
                           "minimum_detected_train_cells": minimum_detected,
                           "target_feature_names": list(target_names) if use_target_features else []})

    def splits(n_outer_folds=3, seed=0):
        # Four complete samples -> three count-balanced test groups (2/1/1),
        # or four leave-one-sample-out folds. No sample is split across roles.
        return _group_folds(samples, n_outer_folds, seed)

    audit = dict(metadata)
    audit.update({"retained_cells": len(meta), "barcoded_cells": int(measured.any(axis=1).sum()),
                  "unassayed_cells": int((~measured.any(axis=1)).sum()),
                  "reference_positives": int(reference.sum()), "n_gene_features": n_gene_features,
                  "sample_counts": {s: int(np.sum(samples == s)) for s in SAMPLES},
                  "location_available": location is not None, "native_target_features_available": False,
                  "reference_kind": "pre_thinning_assay", "cohort_version": "authors_validated_excitatory_v1"})
    dataset = ExperimentDataset("MERGE-seq", reference, measured, ids, TARGETS,
                                features, splits, {"sample": samples}, metadata=audit)
    dataset.validate()
    return dataset


def load_merge_seq(cache_dir, *, n_gene_features=128, location_features_csv=None,
                   target_features_csv=None, raw_paths: Mapping[str, str | Path] | None = None):
    """Load audited metadata + 12 cached GEO files without repeated downloads.

    ``raw_paths`` accepts ``metadata`` and ``geo_dir``. Existing environment
    variables MERGESEQ_METADATA_PATH/MERGESEQ_GEO_MATRIX_DIR remain supported.
    Overrides are still validated. Download manifests pin each raw file's bytes.
    """
    from ..io import atomic_json, cached_download, file_hash

    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    supplied = dict(raw_paths or {})
    metadata_path = Path(supplied.get("metadata", os.environ.get("MERGESEQ_METADATA_PATH", cache / "exn_meta_valid.csv"))).expanduser()
    metadata_path = cached_download(METADATA_URL, metadata_path, git_blob_sha=METADATA_BLOB_SHA)
    meta = pd.read_csv(metadata_path, index_col=0)
    meta.index = meta.index.astype(str)
    _labels_from_metadata(meta)
    source_dir = Path(supplied.get("geo_dir", os.environ.get("MERGESEQ_GEO_MATRIX_DIR", cache / "GSE210172"))).expanduser()
    hashes, paths = {"metadata": file_hash(metadata_path)}, {}
    for sample, (accession, stem) in GEO_SAMPLE_FILES.items():
        paths[sample] = {}
        for kind, suffix in (("barcodes", "barcodes.tsv.gz"), ("genes", "genes.tsv.gz"), ("matrix", "matrix.mtx.gz")):
            filename = f"{stem}_{suffix}"
            url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{accession[:7]}nnn/{accession}/suppl/{filename}"
            path = cached_download(url, source_dir / filename)
            paths[sample][kind] = path
            hashes[filename] = file_hash(path)
    key = hashlib.sha256(json.dumps({"hashes": hashes, "processing": "aligned_log1p_1e4_v1"},
                                    sort_keys=True).encode()).hexdigest()[:20]
    matrix_cache, info_cache = cache / f"expression_{key}.npz", cache / f"expression_{key}.json"
    if matrix_cache.exists() and info_cache.exists():
        info = json.loads(info_cache.read_text())
        if info.get("matrix_sha256") != file_hash(matrix_cache):
            raise ValueError("Processed MERGE-seq cache checksum mismatch; remove the processed pair to rebuild")
        expression = sparse.load_npz(matrix_cache)
        genes, raw_cells = tuple(info["gene_names"]), int(info["raw_input_cells"])
    else:
        genes, blocks, row_blocks, raw_cells = None, [], [], 0
        samples = meta["sample"].astype(str).to_numpy()
        for sample in SAMPLES:
            rows = np.flatnonzero(samples == sample)
            block, genes, total = _load_selected_10x_sample(paths[sample], sample, meta.index[rows], genes)
            blocks.append(block)
            row_blocks.append(rows)
            raw_cells += total
        all_rows = np.concatenate(row_blocks)
        order = np.argsort(all_rows, kind="stable")
        if not np.array_equal(all_rows[order], np.arange(len(meta))):
            raise ValueError("GEO sample blocks do not partition the retained metadata")
        expression = _normalize_counts(sparse.vstack(blocks, format="csr")[order])
        temporary = matrix_cache.with_suffix(".tmp.npz")
        sparse.save_npz(temporary, expression)
        temporary.replace(matrix_cache)
        atomic_json({"gene_names": genes, "raw_input_cells": raw_cells,
                     "matrix_sha256": file_hash(matrix_cache)}, info_cache)
    audit = {"source_sha256": hashes, "raw_input_cells": raw_cells,
             "expression_normalization": "per_cell_library_1e4_log1p",
             "feature_selection": "training_only_variance_with_detection_filter"}
    return _make_dataset(normalized_expression=expression, gene_names=genes, meta=meta,
                         metadata=audit, n_gene_features=n_gene_features,
                         location_features_csv=location_features_csv, target_features_csv=target_features_csv)
