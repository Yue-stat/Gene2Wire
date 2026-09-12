"""SPIDER-Seq Adult.Ex: native animal target panels and train-fitted sparse RNA.

W is constructed from the authors' verified injection design, independently of
outcomes. All author-processed Adult.Ex excitatory cells are retained, including
barcode-negative cells. Reference calls are assay outcomes, not anatomical truth.
"""
from __future__ import annotations
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.sparsefuncs import mean_variance_axis

from ..block_design import make_block_folds
from ..contracts import ExperimentDataset, FeatureSet
from ..io import atomic_json, atomic_npz, cached_download, file_hash

SPIDER_SEQ_REVISION = "22e37e94bc5a5ecda2185f2af4eae5cf49c6092a"
SPIDER_SEQ_URL = (
    "https://huggingface.co/spaces/TigerZheng/SPIDER-web/resolve/"
    f"{SPIDER_SEQ_REVISION}/data/Adult.Ex.rds?download=true"
)
SPIDER_SEQ_SHA256 = "5fd40731b5f415e5aa700df3e677b5a71652898a7c681f71ae70878adc5ade10"
SPIDER_SEQ_SOURCE_COMMIT = "047eb5aaa15087c241570745bd6df07882b0dd78"
SPIDER_SEQ_S2_SHA256 = "3d535880f31c51e065d955e5982e6a5d700c85833a8e6844195fbc7ea968c95d"
# FigureS3.R lists cross-checked with Supplementary Table S2's scRNA-seq
# mouse1/2/3 injections. Table S2 Re/Ect/Lent/Rsp spelling is canonicalized.
SPIDER_SEQ_PANELS = {
    "Adult1": ("ACB-C", "ACB-I", "BLA-I", "CP-C", "CP-I", "DR-I", "LHA-I",
               "MD-I", "RE-I", "SC-I", "VIS-I", "VTA-I"),
    "Adult2": ("ACB-C", "ACB-I", "AId-C", "AId-I", "AUD-I", "BLA-I", "CP-C",
               "CP-I", "ECT-I", "ENTl-I", "PL-C", "RSP-I", "SSp-I", "VIS-I"),
    "Adult3": ("ACB-C", "ACB-I", "AId-C", "AId-I", "BLA-C", "BLA-I", "CP-C",
               "CP-I", "ECT-C", "ECT-I", "ENTl-C", "ENTl-I", "RSP-C", "RSP-I",
               "SSp-I", "VIS-I"),
}
SPIDER_SEQ_TARGETS = tuple(sorted(set().union(*map(set, SPIDER_SEQ_PANELS.values()))))
CACHE_SCHEMA_VERSION = 1
SPIDER_SEQ_MEASUREMENT_GENE_POOL_SIZE = 2000
SPIDER_SEQ_MEASUREMENT_GENE_POOL_VERSION = "gene-id-sha256-v1"
_SPIDER_SEQ_MEASUREMENT_GENE_POOL_SALT = (
    "Gene2Wire::SPIDER-Seq::measurement-gene-pool::v1"
)


def panel_manifest_hash(panels: Mapping[str, Sequence[str]] = SPIDER_SEQ_PANELS) -> str:
    payload = {str(k): sorted(map(str, v)) for k, v in panels.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_native_panel_mask(animal_ids: Sequence[str], target_ids: Sequence[str], *,
                           panels=SPIDER_SEQ_PANELS) -> np.ndarray:
    """Construct assay availability from injection lists, never positive counts."""
    animals = np.asarray(animal_ids, dtype=str)
    targets = tuple(map(str, target_ids))
    if animals.ndim != 1 or len(set(targets)) != len(targets):
        raise ValueError("Animal IDs must be a vector and target IDs must be unique")
    unknown = set(animals).difference(panels)
    if unknown:
        raise ValueError(f"Unrecognized SPIDER-Seq animal IDs: {sorted(unknown)}")
    if set(targets) != set().union(*map(set, panels.values())):
        raise ValueError("Target IDs must equal the union of verified injection panels")
    mask = np.zeros((len(animals), len(targets)), dtype=bool)
    for animal, panel in panels.items():
        mask[animals == animal] = np.isin(targets, tuple(panel))
    return mask


@dataclass(frozen=True)
class SpiderSeqData:
    """Sparse counts plus validated outcomes in one canonical cell order."""
    X_gene_raw: sparse.csr_matrix
    Z_reference: np.ndarray
    W_measured: np.ndarray
    cell_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    animal_ids: np.ndarray
    gene_names: tuple[str, ...]
    source_sha256: str
    metadata: Mapping[str, Any]

    def __post_init__(self):
        x = self.X_gene_raw
        n, p = len(self.cell_ids), len(self.gene_names)
        if not sparse.isspmatrix_csr(x) or x.shape != (n, p):
            raise ValueError("SPIDER-Seq counts must be aligned cell-by-gene CSR")
        if len(set(self.cell_ids)) != n or len(set(self.gene_names)) != p:
            raise ValueError("SPIDER-Seq cell and gene IDs must be unique")
        if not np.isfinite(x.data).all() or np.any(x.data < 0):
            raise ValueError("RNA counts must be finite and nonnegative")
        z, w = np.asarray(self.Z_reference), np.asarray(self.W_measured)
        if z.shape != (n, len(self.target_ids)) or w.shape != z.shape:
            raise ValueError("SPIDER-Seq outcomes and assay mask must align")
        if not np.isin(z, [0, 1]).all() or not np.isin(w, [0, 1]).all():
            raise ValueError("SPIDER-Seq reference and measurement arrays must be binary")
        if np.any(z.astype(bool) & ~w.astype(bool)):
            raise ValueError("A positive call occurs outside the verified injection panel")
        if np.asarray(self.animal_ids).shape != (n,):
            raise ValueError("SPIDER-Seq animal IDs must align with cells")
        if set(self.gene_names).intersection(self.target_ids):
            raise ValueError("Projection barcode channels cannot be gene predictors")


def spider_seq_from_arrays(counts, gene_names, cell_ids, metadata: pd.DataFrame, *,
                           panels=SPIDER_SEQ_PANELS,
                           source_sha256=SPIDER_SEQ_SHA256) -> SpiderSeqData:
    """Validate parsed author data without outcome-based cell filtering."""
    cell_ids, gene_names = tuple(map(str, cell_ids)), tuple(map(str, gene_names))
    meta = metadata.copy()
    meta.index = meta.index.astype(str)
    if meta.index.has_duplicates or set(meta.index) != set(cell_ids):
        raise ValueError("RNA cells and metadata must have identical unique IDs")
    meta = meta.loc[list(cell_ids)]
    if "sample" not in meta:
        raise ValueError("Adult.Ex metadata must contain the author sample field")
    if meta["sample"].isna().any():
        raise ValueError("SPIDER-Seq sample IDs cannot be missing")
    animals = meta["sample"].astype(str).to_numpy()
    targets = tuple(sorted(set().union(*map(set, panels.values()))))
    missing = set(targets).difference(meta.columns)
    if missing:
        raise ValueError(f"Processed target columns are missing: {sorted(missing)}")
    barcode = meta.loc[:, list(targets)].to_numpy(dtype=float)
    measured = make_native_panel_mask(animals, targets, panels=panels)
    if not np.isfinite(barcode[measured]).all() or np.any(barcode[measured] < 0):
        raise ValueError("Measured target calls must be finite and nonnegative")
    if np.any(np.isfinite(barcode[~measured]) & (barcode[~measured] > 0)):
        raise ValueError("Processed positive barcode conflicts with the injection panel")
    reference = measured & (barcode > 0)
    # BC_num is an alignment audit, not an inclusion criterion.
    if "BC_num" in meta and not np.array_equal(reference.sum(axis=1),
                                               meta["BC_num"].to_numpy(dtype=float)):
        raise ValueError("Processed target calls disagree with author BC_num")
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    counts.sum_duplicates()
    counts.sort_indices()
    summary = {
        "cohort": "all cells in author-processed Adult.Ex; no added outcome-dependent filter",
        "sample_counts": {str(k): int(v) for k, v in meta["sample"].value_counts().items()},
        "cells_without_detected_barcode": int((reference.sum(axis=1) == 0).sum()),
        "zero_library_cells": int((np.asarray(counts.sum(axis=1)).ravel() == 0).sum()),
        "native_W_matches_processed_finite": bool(np.array_equal(measured, np.isfinite(barcode))),
        "n_cells": len(cell_ids), "n_genes_raw": len(gene_names),
        "n_assayed_entries": int(measured.sum()),
        "n_positive_entries": int(reference.sum()),
        "panels": {str(k): list(v) for k, v in panels.items()},
        "panel_manifest_sha256": panel_manifest_hash(panels),
        "label_rule": "author-processed metadata target > 0 within verified injection panel",
    }
    return SpiderSeqData(counts, reference, measured, cell_ids, targets, animals,
                         gene_names, source_sha256, summary)


def _save_processed(data: SpiderSeqData, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "adult_ex_arrays.npz"
    x = data.X_gene_raw
    atomic_npz(destination, counts_data=x.data, counts_indices=x.indices,
               counts_indptr=x.indptr, counts_shape=np.asarray(x.shape),
               reference=data.Z_reference, measured=data.W_measured,
               cells=np.asarray(data.cell_ids), genes=np.asarray(data.gene_names),
               targets=np.asarray(data.target_ids), animals=np.asarray(data.animal_ids, dtype=str))
    atomic_json({"schema_version": CACHE_SCHEMA_VERSION, "source_sha256": data.source_sha256,
                 "panel_manifest_sha256": data.metadata["panel_manifest_sha256"],
                 "arrays_sha256": file_hash(destination), "metadata": dict(data.metadata)},
                directory / "manifest.json")


def _load_processed(directory: Path, *, source_sha256=SPIDER_SEQ_SHA256,
                    panels=SPIDER_SEQ_PANELS) -> SpiderSeqData | None:
    manifest_path, arrays_path = directory / "manifest.json", directory / "adult_ex_arrays.npz"
    if not manifest_path.exists() or not arrays_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    expected = {"schema_version": CACHE_SCHEMA_VERSION, "source_sha256": source_sha256,
                "panel_manifest_sha256": panel_manifest_hash(panels)}
    if any(manifest.get(k) != v for k, v in expected.items()):
        return None  # A new extraction schema/source/panel needs fresh extraction.
    if file_hash(arrays_path) != manifest.get("arrays_sha256"):
        raise ValueError("SPIDER-Seq processed cache failed checksum; remove that processed directory explicitly")
    with np.load(arrays_path, allow_pickle=False) as a:
        x = sparse.csr_matrix((a["counts_data"], a["counts_indices"], a["counts_indptr"]),
                              shape=tuple(a["counts_shape"]))
        data = SpiderSeqData(x, a["reference"], a["measured"], tuple(a["cells"].tolist()),
                             tuple(a["targets"].tolist()), a["animals"], tuple(a["genes"].tolist()),
                             source_sha256, manifest["metadata"])
    if not np.array_equal(data.W_measured, make_native_panel_mask(data.animal_ids,
                           data.target_ids, panels=panels)):
        raise ValueError("Processed assay mask disagrees with verified injection panels")
    return data


def _load_spider_seq_data_unlocked(cache_dir: str | Path, *, raw_path: str | Path | None = None) -> SpiderSeqData:
    """Caller owns the cache lock during validation, extraction and publication."""
    directory = Path(cache_dir).expanduser().resolve()
    path = Path(raw_path).expanduser().resolve() if raw_path is not None else directory / "Adult.Ex.rds"
    cached_download(SPIDER_SEQ_URL, path, sha256=SPIDER_SEQ_SHA256)
    processed = directory / "processed_adult_ex_v1"
    data = _load_processed(processed)
    if data is not None:
        return data
    try:
        import rdata
    except ImportError as exc:
        raise ImportError("Initial SPIDER-Seq RDS extraction requires rdata==1.1.0") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(path)
        seurat = rdata.conversion.convert(parsed)
    meta = seurat.__dict__["meta.data"]
    counts = seurat.assays["RNA"].counts
    gene_by_cell = sparse.csc_matrix((counts.x, counts.i, counts.p),
                                     shape=tuple(int(v) for v in counts.Dim))
    data = spider_seq_from_arrays(gene_by_cell.T.tocsr(), counts.Dimnames[0],
                                  counts.Dimnames[1], meta)
    del parsed, seurat, counts, gene_by_cell
    _save_processed(data, processed)
    return data


def load_spider_seq_data(cache_dir: str | Path, *, raw_path: str | Path | None = None) -> SpiderSeqData:
    """Reuse verified raw/processed caches; serialize initial download/extraction.

    The lock covers both the atomic NPZ and its manifest, so a second OnDemand
    kernel cannot read a half-published pair or duplicate the large RDS parse.
    """
    directory = Path(cache_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".spider_seq_cache.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _load_spider_seq_data_unlocked(directory, raw_path=raw_path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _aligned_numeric_csv(path, ids, label):
    frame = pd.read_csv(Path(path).expanduser(), index_col=0)
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates or not frame.columns.is_unique:
        raise ValueError(f"{label} CSV must have unique IDs and columns")
    missing = sorted(set(ids).difference(frame.index))
    if missing:
        raise ValueError(f"{label} CSV lacks required IDs: {missing[:5]}")
    try:
        values = frame.loc[list(ids)].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} descriptors must be numeric") from exc
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f"{label} descriptors must contain finite numeric columns")
    return values, tuple(map(str, frame.columns))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def select_spider_seq_measurement_gene_pool(
    gene_names: Sequence[str],
    pool_size: int = SPIDER_SEQ_MEASUREMENT_GENE_POOL_SIZE,
) -> tuple[int, ...]:
    """Select a fixed source-gene pool using gene IDs alone.

    Genes are ranked by a versioned SHA256 hash of the exact gene ID and a
    public fixed salt.  The rule is invariant to source-column order and never
    inspects expression, projection outcomes, assay availability, animals, or
    train/validation/test membership.  If fewer than ``pool_size`` genes are
    available, every gene is retained in the same deterministic hash order.

    The returned indices follow that hash order, which makes both membership
    and output column order reproducible from the declared gene IDs.
    """
    size = _positive_integer(pool_size, "pool_size")
    names = tuple(map(str, gene_names))
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("gene_names must be a nonempty sequence of unique nonempty IDs")

    def key(index: int):
        payload = (
            f"{_SPIDER_SEQ_MEASUREMENT_GENE_POOL_SALT}\0{names[index]}"
        ).encode("utf-8")
        return hashlib.sha256(payload).digest(), names[index]

    ranked = sorted(range(len(names)), key=key)
    return tuple(ranked[:min(size, len(ranked))])


def _normalize_rna_counts(counts):
    """Apply the cell-local library-size transform once for all CV splits."""
    x = counts.copy().astype(np.float32)
    library = np.asarray(x.sum(axis=1)).ravel()
    factors = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    x = x.multiply(factors[:, None]).tocsr()
    np.log1p(x.data, out=x.data)
    return x


def _prepare_normalized_spider_seq_features(x, gene_names, train_rows, *,
                                             n_hvg=2000, n_gene_components=50,
                                             transform_batch_size=1024) -> FeatureSet:
    """Fit split-specific gene selection/PCA from normalized sparse counts."""
    n_hvg = _positive_integer(n_hvg, "n_hvg")
    if n_gene_components is not None:
        n_gene_components = _positive_integer(n_gene_components, "n_gene_components")
    transform_batch_size = _positive_integer(transform_batch_size, "transform_batch_size")
    train = np.asarray(train_rows, dtype=int)
    n = x.shape[0]
    if (train.ndim != 1 or len(train) < 2 or len(np.unique(train)) != len(train)
            or np.any(train < 0) or np.any(train >= n)):
        raise ValueError("At least two unique, in-range training rows are required")
    if not sparse.isspmatrix_csr(x) or x.shape[1] != len(gene_names):
        raise ValueError("Normalized RNA matrix and gene names are misaligned")
    train_sparse = x[train]
    _, variance = mean_variance_axis(train_sparse, axis=0)
    order = np.lexsort((np.arange(x.shape[1]), -variance))
    selected = order[variance[order] > 1e-10][:n_hvg]
    if len(selected) == 0:
        raise ValueError("No RNA gene varies in the training subset")
    selected_names = tuple(gene_names[j] for j in selected)
    train_dense = train_sparse[:, selected].toarray()
    metadata = {
        "fit_rows": train.tolist(), "n_hvg_requested": n_hvg,
        "n_hvg_used": len(selected), "n_gene_components_requested": n_gene_components,
        "selected_gene_names": list(selected_names),
    }
    if n_gene_components is None:
        scaler = StandardScaler().fit(train_dense)
        scores = np.empty((n, len(selected)), dtype=np.float64)
        for start in range(0, n, transform_batch_size):
            stop = min(n, start + transform_batch_size)
            scores[start:stop] = scaler.transform(x[start:stop, selected].toarray())
        names = selected_names
        metadata.update({"preprocessing": "per-cell counts/total*10000; log1p; train-variance genes; train gene standardization",
                         "n_gene_components_used": None})
    else:
        n_components = min(n_gene_components, len(selected), len(train) - 1)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
        pca.fit(train_dense)
        scores = np.empty((n, n_components), dtype=np.float64)
        for start in range(0, n, transform_batch_size):
            stop = min(n, start + transform_batch_size)
            scores[start:stop] = pca.transform(x[start:stop, selected].toarray())
        scaler = StandardScaler().fit(scores[train])
        scores = scaler.transform(scores)
        names = tuple(f"gene_PC_{j+1}" for j in range(n_components))
        metadata.update({
            "preprocessing": "per-cell counts/total*10000; log1p; train-variance genes; train-centered randomized PCA; train PC standardization",
            "n_gene_components_used": n_components,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
            "pca_random_state": 0,
        })
    if not np.isfinite(scores).all():
        raise FloatingPointError("SPIDER-Seq RNA features are non-finite")
    return FeatureSet(X=scores, feature_blocks={"gene": tuple(range(scores.shape[1]))},
                      feature_names=names, metadata=metadata)


def prepare_spider_seq_features(data: SpiderSeqData, train_rows, *, n_hvg=2000,
                                n_gene_components=50, transform_batch_size=1024) -> FeatureSet:
    """Select genes and fit centered PCA on train only; transform in batches.

    Sparse full-RNA counts remain sparse. Dense allocation is bounded by the
    training-by-selected-gene matrix plus one batch, never all genes by all cells.
    Per-cell library normalization does not learn across cells.
    """
    return _prepare_normalized_spider_seq_features(
        _normalize_rna_counts(data.X_gene_raw), data.gene_names, train_rows,
        n_hvg=n_hvg, n_gene_components=n_gene_components,
        transform_batch_size=transform_batch_size)


def spider_seq_dataset(data: SpiderSeqData, *, n_hvg=2000, n_gene_components=50,
                       location_features_csv=None, target_features_csv=None) -> ExperimentDataset:
    """Expose the common API with native W and within-animal held-out cells."""
    n_hvg = _positive_integer(n_hvg, "n_hvg")
    if n_gene_components is not None:
        n_gene_components = _positive_integer(n_gene_components, "n_gene_components")
    location = (None if location_features_csv is None else
                _aligned_numeric_csv(location_features_csv, data.cell_ids, "Cell location"))
    target = (None if target_features_csv is None else
              _aligned_numeric_csv(target_features_csv, data.target_ids, "Target feature"))
    # This transform is cell-local and learns no population parameter. Computing
    # it once avoids copying and normalizing the 79-million-entry sparse matrix
    # for every fold. HVG selection and PCA below remain fitted on train rows.
    normalized_counts = _normalize_rna_counts(data.X_gene_raw)
    gene_names = data.gene_names
    animal_ids = np.asarray(data.animal_ids, dtype=str)

    def features(train_rows, use_location=False, use_target_features=False):
        if not isinstance(use_location, bool) or not isinstance(use_target_features, bool):
            raise TypeError("Feature switches must be booleans")
        if use_location and location is None:
            raise ValueError("SPIDER-Seq has no bundled spatial coordinates; supply location_features_csv or use USE_LOCATION=False")
        if use_target_features and target is None:
            raise ValueError("Supply independent target_features_csv descriptors or use USE_TARGET_FEATURES=False")
        base = _prepare_normalized_spider_seq_features(
            normalized_counts, gene_names, train_rows, n_hvg=n_hvg,
            n_gene_components=n_gene_components)
        x, blocks, names = base.X, dict(base.feature_blocks), base.feature_names
        if use_location:
            loc = StandardScaler().fit(location[0][train_rows]).transform(location[0])
            blocks["location"] = tuple(range(x.shape[1], x.shape[1] + loc.shape[1]))
            x = np.column_stack((x, loc))
            names += tuple(f"location::{name}" for name in location[1])
        return FeatureSet(x, blocks, target[0].copy() if use_target_features else None,
                          names, {**dict(base.metadata),
                                  "location_feature_names": location[1] if use_location else (),
                                  "target_feature_names": target[1] if use_target_features else ()})

    def splits(n_outer_folds=3, seed=0):
        return make_block_folds(animal_ids, n_outer_folds, seed)

    result = ExperimentDataset(
        name="SPIDER-Seq", reference=data.Z_reference, measured=data.W_measured,
        cell_ids=data.cell_ids, target_ids=data.target_ids,
        feature_builder=features, split_builder=splits,
        groups={"animal": animal_ids},
        metadata={**dict(data.metadata), "source_url": SPIDER_SEQ_URL,
                  "source_sha256": data.source_sha256, "source_revision": SPIDER_SEQ_REVISION,
                  "source_code_commit": SPIDER_SEQ_SOURCE_COMMIT,
                  "supplementary_table_s2_sha256": SPIDER_SEQ_S2_SHA256,
                  "reference": "author-processed assay calls; not anatomical truth",
                  "native_fragmented_panels": True, "gene_names": data.gene_names,
                  "n_hvg": n_hvg, "n_gene_components": n_gene_components,
                  "location_feature_source": str(location_features_csv) if location else None,
                  "target_feature_source": str(target_features_csv) if target else None,
                  "location_feature_sha256": file_hash(Path(location_features_csv).expanduser()) if location else None,
                  "target_feature_sha256": file_hash(Path(target_features_csv).expanduser()) if target else None,
                  "target_features_available": target is not None,
                  "split": "within-animal held-out cells; not unseen-animal evaluation"},
    )
    result.validate()
    return result


def spider_seq_measurement_dataset(
    data: SpiderSeqData,
    *,
    gene_pool_size: int = SPIDER_SEQ_MEASUREMENT_GENE_POOL_SIZE,
    n_gene_components=50,
    location_features_csv=None,
    target_features_csv=None,
) -> ExperimentDataset:
    """Expose a fixed, outcome-blind source pool for panel-degradation studies.

    This adapter deliberately differs from :func:`spider_seq_dataset`.  The
    ordinary adapter chooses fold-specific highly variable genes from the full
    transcriptome.  A panel experiment instead needs one immutable source pool
    so that every overlap/coverage condition refers to the same named genes.
    Membership and order of this pool are determined only from exact gene IDs
    by :func:`select_spider_seq_measurement_gene_pool`.

    ``gene_matrix`` is dense float32 on the cell-local
    ``counts / library_size * 10000 -> log1p`` scale.  The denominator uses the
    original full RNA library and does not fit any cross-cell statistic.  No
    centering, scaling, HVG selection, or PCA has been fitted in this matrix.
    A downstream measurement wrapper must first apply its gene-observation
    panel to ``gene_matrix`` and only then fit any train-only scaling or PCA.
    The feature builder here represents the unmasked full-pool control.

    Native target availability, target outcomes, animal groups, and within-
    animal folds are copied without modification.
    """
    size = _positive_integer(gene_pool_size, "gene_pool_size")
    if n_gene_components is not None:
        n_gene_components = _positive_integer(
            n_gene_components, "n_gene_components"
        )
    selected = select_spider_seq_measurement_gene_pool(data.gene_names, size)
    selected_names = tuple(data.gene_names[index] for index in selected)
    location = (
        None
        if location_features_csv is None
        else _aligned_numeric_csv(
            location_features_csv, data.cell_ids, "Cell location"
        )
    )
    target = (
        None
        if target_features_csv is None
        else _aligned_numeric_csv(
            target_features_csv, data.target_ids, "Target feature"
        )
    )

    # Library normalization is cell-local and therefore safe to compute once.
    # Restrict to the declared source pool before any train-fitted operation.
    normalized = _normalize_rna_counts(data.X_gene_raw)
    source_sparse = normalized[:, np.asarray(selected, dtype=int)].tocsr()
    source_matrix = source_sparse.toarray().astype(np.float32, copy=False)
    animal_ids = np.asarray(data.animal_ids, dtype=str)
    pool_manifest = {
        "version": SPIDER_SEQ_MEASUREMENT_GENE_POOL_VERSION,
        "requested_size": size,
        "source_gene_count": len(data.gene_names),
        "gene_ids": selected_names,
    }
    pool_hash = hashlib.sha256(
        json.dumps(pool_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def features(train_rows, use_location=False, use_target_features=False):
        if not isinstance(use_location, bool) or not isinstance(
            use_target_features, bool
        ):
            raise TypeError("Feature switches must be booleans")
        if use_location and location is None:
            raise ValueError(
                "SPIDER-Seq has no bundled spatial coordinates; supply "
                "location_features_csv or use USE_LOCATION=False"
            )
        if use_target_features and target is None:
            raise ValueError(
                "Supply independent target_features_csv descriptors or use "
                "USE_TARGET_FEATURES=False"
            )
        base = _prepare_normalized_spider_seq_features(
            source_sparse,
            selected_names,
            train_rows,
            n_hvg=len(selected_names),
            n_gene_components=n_gene_components,
        )
        x, blocks, names = base.X, dict(base.feature_blocks), base.feature_names
        if use_location:
            train = np.asarray(train_rows, dtype=int)
            loc = StandardScaler().fit(location[0][train]).transform(location[0])
            blocks["location"] = tuple(
                range(x.shape[1], x.shape[1] + loc.shape[1])
            )
            x = np.column_stack((x, loc))
            names += tuple(f"location::{name}" for name in location[1])
        return FeatureSet(
            x,
            blocks,
            target[0].copy() if use_target_features else None,
            names,
            {
                **dict(base.metadata),
                "source_gene_pool_sha256": pool_hash,
                "source_gene_pool_size": len(selected_names),
                "source_gene_pool_selection": SPIDER_SEQ_MEASUREMENT_GENE_POOL_VERSION,
                "source_gene_matrix_scale": (
                    "full-library per-cell counts/total*10000; log1p; "
                    "no cross-cell fit"
                ),
                "feature_role": "unmasked full measurement-gene-pool control",
                "location_feature_names": location[1] if use_location else (),
                "target_feature_names": target[1] if use_target_features else (),
            },
        )

    def splits(n_outer_folds=3, seed=0):
        return make_block_folds(animal_ids, n_outer_folds, seed)

    result = ExperimentDataset(
        name="SPIDER-Seq",
        reference=data.Z_reference,
        measured=data.W_measured,
        cell_ids=data.cell_ids,
        target_ids=data.target_ids,
        feature_builder=features,
        split_builder=splits,
        groups={"animal": animal_ids},
        gene_matrix=source_matrix,
        gene_names=selected_names,
        metadata={
            **dict(data.metadata),
            "source_url": SPIDER_SEQ_URL,
            "source_sha256": data.source_sha256,
            "source_revision": SPIDER_SEQ_REVISION,
            "source_code_commit": SPIDER_SEQ_SOURCE_COMMIT,
            "supplementary_table_s2_sha256": SPIDER_SEQ_S2_SHA256,
            "reference": "author-processed assay calls; not anatomical truth",
            "native_fragmented_panels": True,
            "measurement_gene_pool_version": SPIDER_SEQ_MEASUREMENT_GENE_POOL_VERSION,
            "measurement_gene_pool_requested_size": size,
            "measurement_gene_pool_size": len(selected_names),
            "measurement_gene_pool_source_gene_count": len(data.gene_names),
            "measurement_gene_pool_indices": list(selected),
            "measurement_gene_pool_names": list(selected_names),
            "measurement_gene_pool_sha256": pool_hash,
            "measurement_gene_pool_selection": (
                "versioned SHA256 rank of exact gene IDs; expression, outcomes, "
                "assay availability, animals, and split roles are not read"
            ),
            "gene_matrix_stage": (
                "post cell-local full-library normalization/log1p; pre "
                "train-fitted scaling, HVG selection, and PCA"
            ),
            "gene_matrix_scale": "counts/total_RNA_library*10000 then log1p",
            "gene_matrix_cross_cell_fit": False,
            "measurement_wrapper_contract": (
                "apply the gene-observation panel to gene_matrix before every "
                "train-only scaling or PCA fit"
            ),
            "n_gene_components": n_gene_components,
            "location_feature_source": (
                str(location_features_csv) if location else None
            ),
            "target_feature_source": (
                str(target_features_csv) if target else None
            ),
            "location_feature_sha256": (
                file_hash(Path(location_features_csv).expanduser())
                if location
                else None
            ),
            "target_feature_sha256": (
                file_hash(Path(target_features_csv).expanduser())
                if target
                else None
            ),
            "target_features_available": target is not None,
            "split": "within-animal held-out cells; not unseen-animal evaluation",
        },
    )
    result.validate()
    return result


def load_spider_seq(cache_dir, *, n_hvg=2000, n_gene_components=50,
                    location_features_csv=None, target_features_csv=None,
                    raw_path=None) -> ExperimentDataset:
    return spider_seq_dataset(load_spider_seq_data(cache_dir, raw_path=raw_path),
                              n_hvg=n_hvg, n_gene_components=n_gene_components,
                              location_features_csv=location_features_csv,
                              target_features_csv=target_features_csv)


def load_spider_seq_measurement(
    cache_dir,
    *,
    gene_pool_size: int = SPIDER_SEQ_MEASUREMENT_GENE_POOL_SIZE,
    n_gene_components=50,
    location_features_csv=None,
    target_features_csv=None,
    raw_path=None,
) -> ExperimentDataset:
    """Load SPIDER-Seq with the fixed measurement-degradation gene pool."""
    return spider_seq_measurement_dataset(
        load_spider_seq_data(cache_dir, raw_path=raw_path),
        gene_pool_size=gene_pool_size,
        n_gene_components=n_gene_components,
        location_features_csv=location_features_csv,
        target_features_csv=target_features_csv,
    )
