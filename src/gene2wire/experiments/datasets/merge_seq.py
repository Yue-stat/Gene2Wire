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
ASSAY_DERIVED_GENE_PREFIXES = ("barcode",)


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


def _load_merge_seq_arrays(
    cache_dir,
    *,
    raw_paths: Mapping[str, str | Path] | None = None,
):
    """Return aligned metadata and normalized expression from the verified cache."""

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
    return expression, genes, meta, audit


def _deterministic_unassayed_design_rows(meta, measured, *, fraction, seed):
    """Reserve a fixed fraction of label-free design cells within every sample.

    The candidates must be both explicitly ``Non-barcoded`` and all-W=0. Cell
    IDs are sorted before seeded sampling so the result does not depend on the
    input row order. These rows are used only to choose a fixed source-gene
    panel and are removed from the returned analysis cohort.
    """

    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("panel_design_fraction must lie strictly between zero and one")
    w = np.asarray(measured, dtype=bool)
    if w.shape != (len(meta), len(TARGETS)):
        raise ValueError("MERGE-seq measurement mask is not aligned to metadata")
    status = meta["barcoded"].astype(str).str.strip().to_numpy()
    candidate = (status == "Non-barcoded") & ~w.any(axis=1)
    ids = meta.index.astype(str).to_numpy()
    samples = meta["sample"].astype(str).to_numpy()
    selected = []
    counts = {}
    for sample in SAMPLES:
        rows = np.flatnonzero(candidate & (samples == sample))
        if len(rows) < 2:
            raise ValueError(
                f"MERGE-seq sample {sample!r} has only {len(rows)} all-W=0 "
                "non-barcoded cells; at least two are required"
            )
        sample_count = max(1, int(np.floor(len(rows) * fraction + .5)))
        if sample_count >= len(rows):
            raise ValueError(
                f"panel_design_fraction={fraction:g} would reserve every all-W=0 "
                f"non-barcoded cell in MERGE-seq sample {sample!r}"
            )
        rows = rows[np.argsort(ids[rows], kind="stable")]
        payload = f"{int(seed)}|MERGE-seq-overlap-design|{sample}".encode("utf-8")
        sample_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        chosen = rows[np.random.default_rng(sample_seed).permutation(len(rows))[:sample_count]]
        selected.extend(map(int, chosen))
        counts[sample] = int(len(chosen))
    result = np.asarray(sorted(selected), dtype=int)
    if len(result) != sum(counts.values()) or len(np.unique(result)) != len(result):
        raise RuntimeError("MERGE-seq design rows were not selected one-to-one")
    if np.any(w[result]) or np.any(status[result] != "Non-barcoded"):
        raise RuntimeError("MERGE-seq source-gene design rows must be label-free")
    return result, counts


def _assay_derived_gene(name: str) -> bool:
    value = str(name).strip().casefold()
    return any(value.startswith(prefix) for prefix in ASSAY_DERIVED_GENE_PREFIXES)


def _renormalize_without_assay_features(expression, genes):
    """Remove assay-derived rows from the log-normalization denominator.

    The shared processed cache stores ``log1p(1e4 * count / all-count total)``.
    Exponentiating recovers the normalized linear proportions, so a second
    row normalization after zeroing ``barcode*`` columns is algebraically the
    same as normalizing the biological-gene counts without those columns.  The
    ordinary MERGE adapter deliberately retains its historical normalization;
    this stricter transformation is overlap-only.
    """

    matrix = sparse.csr_matrix(expression, dtype=np.float64).copy()
    if matrix.shape[1] != len(genes):
        raise ValueError("MERGE-seq expression and genes are not aligned")
    assay_derived = np.asarray(
        [_assay_derived_gene(name) for name in genes], dtype=bool
    )
    matrix.data = np.expm1(matrix.data)
    if assay_derived.any():
        matrix = matrix.multiply((~assay_derived).astype(float)).tocsr()
        matrix.eliminate_zeros()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError(
            "MERGE-seq cells need positive non-assay expression library size"
        )
    matrix = (sparse.diags(1e4 / totals) @ matrix).tocsr()
    matrix.data = np.log1p(matrix.data)
    matrix.eliminate_zeros()
    return matrix.astype(np.float32), assay_derived


def _select_overlap_source_genes(expression, genes, design_rows, *, n_source_genes):
    """Choose fixed high-variance genes using reserved design expression only."""

    if isinstance(n_source_genes, bool) or int(n_source_genes) != n_source_genes:
        raise TypeError("n_source_genes must be an integer")
    n_source_genes = int(n_source_genes)
    if n_source_genes < 2:
        raise ValueError("n_source_genes must be at least two")
    design = np.asarray(design_rows, dtype=int)
    if design.ndim != 1 or not len(design) or len(np.unique(design)) != len(design):
        raise ValueError("design_rows must be a nonempty unique integer vector")
    fitted = sparse.csr_matrix(expression)[design]
    mean = np.asarray(fitted.mean(axis=0)).ravel().astype(float)
    variance = np.maximum(
        np.asarray(fitted.multiply(fitted).mean(axis=0)).ravel() - mean**2,
        0,
    )
    detected = np.asarray(fitted.getnnz(axis=0)).ravel()
    minimum_detected = max(2, int(math.ceil(.01 * len(design))))
    assay_derived = np.asarray([_assay_derived_gene(name) for name in genes], dtype=bool)
    eligible = np.flatnonzero(
        (detected >= minimum_detected) & np.isfinite(variance) & ~assay_derived
    )
    if len(eligible) < n_source_genes:
        raise ValueError(
            f"Only {len(eligible)} genes meet the reserved-design detection filter; "
            f"requested n_source_genes={n_source_genes}"
        )
    selected = eligible[np.lexsort((eligible, -variance[eligible]))[:n_source_genes]]
    return np.asarray(selected, dtype=int), minimum_detected, tuple(
        str(genes[index]) for index in np.flatnonzero(assay_derived)
    )


def _make_overlap_dataset(
    *,
    normalized_expression,
    gene_names,
    meta,
    metadata,
    source_gene_count=128,
    panel_design_fraction=.20,
    seed=20260910,
):
    """Build the opt-in MERGE-seq source panel for gene-overlap experiments.

    Unlike the ordinary adapter's fold-specific variable-gene selection, this
    adapter needs one fixed source panel so all overlap levels and folds refer
    to the same gene coordinates. It chooses that panel using only a reserved
    subset of structurally unassayed cells and then removes those cells.
    """

    expression = sparse.csr_matrix(normalized_expression, dtype=np.float32)
    genes = tuple(map(str, gene_names))
    if expression.shape != (len(meta), len(genes)) or len(set(genes)) != len(genes):
        raise ValueError("MERGE-seq expression and unique genes must align to metadata")
    if not np.isfinite(expression.data).all() or np.any(expression.data < 0):
        raise ValueError("MERGE-seq normalized expression is invalid")
    # Prevent an excluded barcode-capture feature from leaking back indirectly
    # through the library-size denominator used by the shared processed cache.
    expression, assay_derived = _renormalize_without_assay_features(expression, genes)
    reference, measured = _labels_from_metadata(meta)
    design_rows, design_counts = _deterministic_unassayed_design_rows(
        meta, measured, fraction=panel_design_fraction, seed=seed
    )
    selected, minimum_detected, excluded_assay_genes = _select_overlap_source_genes(
        expression, genes, design_rows, n_source_genes=source_gene_count
    )
    keep = np.ones(len(meta), dtype=bool)
    keep[design_rows] = False
    analysis_rows = np.flatnonzero(keep)
    analysis_meta = meta.iloc[analysis_rows].copy()
    source_matrix = expression[analysis_rows][:, selected].toarray().astype(np.float32)
    analysis_reference = reference[analysis_rows]
    analysis_measured = measured[analysis_rows]
    ids = tuple(analysis_meta.index.astype(str))
    samples = analysis_meta["sample"].astype(str).to_numpy()
    assay_status = np.where(analysis_measured.any(axis=1), "assayed", "unassayed")

    def features(train_rows, use_location=False, use_target_features=False):
        train = _training_rows(train_rows, len(ids))
        if use_location:
            raise ValueError(
                "MERGE-seq has no native physical cell coordinates; pseudotime is not location"
            )
        if use_target_features:
            raise ValueError(
                "MERGE-seq has no native target expression descriptors; "
                "the overlap adapter requires USE_TARGET_FEATURES=False"
            )
        x = StandardScaler().fit(source_matrix[train]).transform(source_matrix)
        return FeatureSet(
            x,
            {"gene": tuple(range(len(selected)))},
            None,
            tuple(genes[index] for index in selected),
            {
                "preprocessing_fit_rows": train.tolist(),
                "source_gene_selection": "reserved_all-W-zero_non-barcoded_top_variance_v1",
                "source_gene_selection_reads_analysis_expression": False,
            },
        )

    def splits(n_outer_folds=3, seed=0):
        return _group_folds(samples, n_outer_folds, seed)

    audit = dict(metadata)
    audit.update({
        "retained_cells": len(ids),
        "barcoded_cells": int(analysis_measured.any(axis=1).sum()),
        "unassayed_cells": int((~analysis_measured.any(axis=1)).sum()),
        "reference_positives": int(analysis_reference.sum()),
        "sample_counts": {sample: int(np.sum(samples == sample)) for sample in SAMPLES},
        "location_available": False,
        "native_target_features_available": False,
        "reference_kind": "pre_thinning_assay",
        "cohort_version": "authors_validated_excitatory_v1_overlap_reserved_design_v1",
        "overlap_source_gene_count": int(len(selected)),
        "overlap_source_gene_indices": selected.tolist(),
        "overlap_source_gene_names": [genes[index] for index in selected],
        "overlap_source_selection": "reserved_all-W-zero_non-barcoded_top_variance_v1",
        "overlap_expression_normalization":
            "per_cell_library_1e4_log1p_excluding_assay_derived_features",
        "overlap_normalization_reads_analysis_labels": False,
        "overlap_source_minimum_detected_design_cells": int(minimum_detected),
        "overlap_design_seed": int(seed),
        "overlap_panel_design_fraction": float(panel_design_fraction),
        "overlap_design_counts": design_counts,
        "overlap_design_cell_ids": meta.index.astype(str).to_numpy()[design_rows].tolist(),
        "overlap_design_rows_removed": True,
        "overlap_selection_reads_analysis_expression": False,
        "excluded_assay_derived_genes": list(excluded_assay_genes),
        "excluded_assay_derived_gene_count": int(assay_derived.sum()),
    })
    dataset = ExperimentDataset(
        "MERGE-seq",
        analysis_reference,
        analysis_measured,
        ids,
        TARGETS,
        features,
        splits,
        {"sample": samples, "assay_status": assay_status},
        metadata=audit,
        gene_matrix=source_matrix,
        gene_names=tuple(genes[index] for index in selected),
    )
    dataset.validate()
    if not np.array_equal(dataset.measured, np.repeat(
        dataset.measured[:, :1], len(TARGETS), axis=1
    )):
        raise ValueError("MERGE-seq overlap targets do not share one assay mask")
    return dataset


def load_merge_seq(cache_dir, *, n_gene_features=128, location_features_csv=None,
                   target_features_csv=None, raw_paths: Mapping[str, str | Path] | None = None):
    """Load audited metadata + 12 cached GEO files without repeated downloads.

    ``raw_paths`` accepts ``metadata`` and ``geo_dir``. Existing environment
    variables MERGESEQ_METADATA_PATH/MERGESEQ_GEO_MATRIX_DIR remain supported.
    Overrides are still validated. Download manifests pin each raw file's bytes.
    """

    expression, genes, meta, audit = _load_merge_seq_arrays(
        cache_dir, raw_paths=raw_paths
    )
    return _make_dataset(normalized_expression=expression, gene_names=genes, meta=meta,
                         metadata=audit, n_gene_features=n_gene_features,
                         location_features_csv=location_features_csv, target_features_csv=target_features_csv)


def load_merge_seq_overlap(
    cache_dir,
    *,
    source_gene_count=128,
    panel_design_fraction=.20,
    seed=20260910,
    raw_paths: Mapping[str, str | Path] | None = None,
):
    """Load the opt-in, leak-isolated MERGE-seq gene-overlap adapter.

    A deterministic fraction of all-W=0 non-barcoded cells within each sample
    defines a fixed top-variance source panel. Those design cells are
    removed before splitting, calibration, fitting, and evaluation. Projection
    targets and the remaining cells' reference/measurement arrays are unchanged.
    The ordinary :func:`load_merge_seq` adapter retains its fold-local feature
    selection and is not altered by this opt-in workflow.
    """

    expression, genes, meta, audit = _load_merge_seq_arrays(
        cache_dir, raw_paths=raw_paths
    )
    return _make_overlap_dataset(
        normalized_expression=expression,
        gene_names=genes,
        meta=meta,
        metadata={**audit, "feature_selection":
                  "reserved_all-W-zero_non-barcoded_top_variance_then_train_only_scaling"},
        source_gene_count=source_gene_count,
        panel_design_fraction=panel_design_fraction,
        seed=seed,
    )
