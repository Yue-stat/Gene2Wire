"""Audited SPIDER spatial cohort, persistent raw cache, and train-fitted features.

Raw pre-thinning assay calls form an imperfect evaluation reference. Source
cohort and barcode thresholds follow SPIDER_no_loc.ipynb; no outcome-dependent
feature or hyperparameter choices are implemented in this adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, StandardScaler

from ..contracts import ExperimentDataset, FeatureSet, Fold
from ..io import cached_download

from scipy.sparse import csc_matrix

SPIDER_RDS_URL = (
    "https://huggingface.co/spaces/TigerZheng/"
    "SPIDER-web/resolve/main/data/sp.PFC.rds?download=true"
)


SPIDER_RDS_SHA256 = "9069585cadd37b22edb236a5c0fb84c1326a0a9be1f9bc67f4068a8295d45cae"


SPIDER_TARGETS = (
    "VIS-I", "ACB-I", "CP-C", "AId-I", "CP-I",
    "ECT-C", "AId-C", "ECT-I", "BLA-I", "AUD-I",
    "RSP-C", "SSp-I", "RSP-I", "ACB-C", "LHA-I",
)


SPIDER_LOCATION_COLUMNS = ("ML_new", "DV_new", "AP_new")


SPIDER_INCLUDED_LAYERS = ("L2/3 IT", "L4/5 IT", "L5 IT", "L6 IT", "L5 PT")


def download_spider_rds(cache_dir: str | Path) -> tuple[Path, str]:
    """Use the checksum-verified raw cache without contacting the network."""

    path = Path(cache_dir).expanduser().resolve() / "sp.PFC.rds"
    cached_download(SPIDER_RDS_URL, path, sha256=SPIDER_RDS_SHA256)
    return path, SPIDER_RDS_SHA256


@dataclass(frozen=True)
class SpiderData:
    """Validated pre-hide SPIDER arrays in the canonical cell order."""

    X_gene_raw: np.ndarray
    X_loc_raw: np.ndarray
    Z_reference: np.ndarray
    W_measured: np.ndarray
    cell_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    slice_ids: np.ndarray
    ap_coordinate: np.ndarray
    gene_names: tuple[str, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        n_cells = len(self.cell_ids)
        if self.X_gene_raw.ndim != 2 or self.X_gene_raw.shape[0] != n_cells:
            raise ValueError("X_gene_raw must be cell-by-gene")
        if self.X_loc_raw.shape != (n_cells, 3):
            raise ValueError("X_loc_raw must contain the three SPIDER coordinates")
        expected = (n_cells, len(self.target_ids))
        if self.Z_reference.shape != expected or self.W_measured.shape != expected:
            raise ValueError("SPIDER label and measurement arrays are misaligned")
        if len(self.slice_ids) != n_cells or len(self.ap_coordinate) != n_cells:
            raise ValueError("slice/AP arrays must align with cells")
        if len(set(self.cell_ids)) != n_cells:
            raise ValueError("SPIDER cell IDs must be unique")
        if not np.all(np.isfinite(self.X_gene_raw)) or not np.all(np.isfinite(self.X_loc_raw)):
            raise ValueError("SPIDER features contain non-finite values")
        if np.any(self.Z_reference & ~self.W_measured):
            raise ValueError("a positive label cannot occur outside W_measured")

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_targets(self) -> int:
        return len(self.target_ids)


@dataclass(frozen=True)
class SpiderFold:
    outer_fold: int
    train_rows: np.ndarray
    validation_rows: np.ndarray
    test_rows: np.ndarray
    train_slices: tuple[str, ...]
    validation_slices: tuple[str, ...]
    test_slices: tuple[str, ...]


def load_spider_data(cache_dir: str | Path) -> SpiderData:
    """Load the audited SPIDER object and construct X_gene, X_loc, Z, and W."""

    try:
        import rdata
    except ImportError as exc:
        raise ImportError("SPIDER loading requires rdata==1.1.0") from exc

    rds_path, source_hash = download_spider_rds(cache_dir)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = rdata.parser.parse_file(rds_path)
        seurat = rdata.conversion.convert(parsed)

    meta_all = seurat.__dict__["meta.data"].copy()
    meta_all.columns = meta_all.columns.astype(str)
    rna_assay = next(iter(seurat.assays.values())).counts
    gene_names = np.asarray(rna_assay.Dimnames[0]).astype(str)
    cell_names = np.asarray(rna_assay.Dimnames[1]).astype(str)
    gene_by_cell = csc_matrix(
        (rna_assay.x, rna_assay.i, rna_assay.p),
        shape=tuple(int(value) for value in rna_assay.Dim),
    )
    if gene_by_cell.shape != (32, 124_829):
        raise AssertionError(f"unexpected SPIDER count shape: {gene_by_cell.shape}")
    if meta_all.shape[0] != gene_by_cell.shape[1]:
        raise AssertionError("SPIDER metadata and count matrix have different cell counts")
    if not np.array_equal(meta_all.index.astype(str).to_numpy(), cell_names):
        raise AssertionError("SPIDER metadata order does not match RNA cell order")

    keep = (
        meta_all["ABA_hemisphere"].eq("Left")
        & meta_all["SubType_Layer"].isin(SPIDER_INCLUDED_LAYERS)
    ).to_numpy()
    meta = meta_all.loc[keep].copy()
    x_gene = gene_by_cell[:, keep].T.toarray().astype(np.float64)
    x_loc = meta[list(SPIDER_LOCATION_COLUMNS)].to_numpy(dtype=np.float64)
    barcode = meta[list(SPIDER_TARGETS)].to_numpy(dtype=np.float64)
    measured = np.isfinite(barcode)
    reference = np.zeros(barcode.shape, dtype=bool)
    reference[measured] = barcode[measured] > 0
    retained_ids = tuple(meta.index.astype(str).tolist())
    observed_positive_count = np.sum(reference, axis=1).astype(int)

    if not set(gene_names).isdisjoint(SPIDER_TARGETS):
        raise AssertionError("barcode target labels leaked into gene features")
    if not np.array_equal(observed_positive_count, meta["BC_num"].to_numpy(dtype=int)):
        raise AssertionError("constructed SPIDER labels disagree with BC_num")
    if not measured.all():
        raise AssertionError("the audited SPIDER object should have all 15 channels measured")
    if len(np.unique(meta["slice"].astype(str))) != 36:
        raise AssertionError("the audited SPIDER subset should contain 36 slices")

    del parsed, seurat, rna_assay, gene_by_cell
    print(
        f"Loaded SPIDER in {time.time() - start:.1f}s: "
        f"{len(retained_ids):,} neurons, {x_gene.shape[1]} genes, "
        f"{len(SPIDER_TARGETS)} targets"
    )
    return SpiderData(
        X_gene_raw=x_gene,
        X_loc_raw=x_loc,
        Z_reference=reference,
        W_measured=measured,
        cell_ids=retained_ids,
        target_ids=SPIDER_TARGETS,
        slice_ids=meta["slice"].astype(str).to_numpy(),
        ap_coordinate=meta["AP_new"].to_numpy(dtype=np.float64),
        gene_names=tuple(gene_names.tolist()),
        source_sha256=source_hash,
    )


def make_spatial_folds(
    data: SpiderData,
    n_outer_folds: int = 3,
    n_inner_validation_slices: int = 4,
    *,
    require_36_slices: bool = True,
) -> tuple[SpiderFold, ...]:
    """Create contiguous AP outer blocks and spaced inner validation slices."""

    if n_outer_folds < 2 or n_inner_validation_slices < 1:
        raise ValueError("SPIDER CV needs >=2 outer folds and >=1 validation slice")
    slice_frame = pd.DataFrame(
        {"slice": data.slice_ids.astype(str), "ap": data.ap_coordinate}
    )
    slice_ap = slice_frame.groupby("slice", observed=True)["ap"].median().sort_values()
    ordered_slices = slice_ap.index.astype(str).tolist()
    if require_36_slices and len(ordered_slices) != 36:
        raise AssertionError(f"expected 36 SPIDER slices, found {len(ordered_slices)}")
    blocks = [tuple(map(str, block)) for block in np.array_split(ordered_slices, n_outer_folds)]
    if any(not block for block in blocks):
        raise ValueError("more outer folds than slices")

    folds: list[SpiderFold] = []
    test_visit_count = np.zeros(data.n_cells, dtype=int)
    for outer_fold, test_tuple in enumerate(blocks):
        test_slices = set(test_tuple)
        remaining = [name for name in ordered_slices if name not in test_slices]
        if len(remaining) <= n_inner_validation_slices:
            raise ValueError("too few remaining slices for the requested validation set")
        positions = np.unique(
            np.linspace(
                0,
                len(remaining) - 1,
                num=n_inner_validation_slices,
                dtype=int,
            )
        )
        if len(positions) != n_inner_validation_slices:
            raise AssertionError("inner validation slice selection produced duplicates")
        validation_slices = {remaining[int(position)] for position in positions}
        training_slices = set(remaining).difference(validation_slices)
        train_mask = np.isin(data.slice_ids, sorted(training_slices))
        validation_mask = np.isin(data.slice_ids, sorted(validation_slices))
        test_mask = np.isin(data.slice_ids, sorted(test_slices))
        membership = train_mask.astype(int) + validation_mask.astype(int) + test_mask.astype(int)
        if not np.all(membership == 1):
            raise AssertionError("SPIDER fold does not partition every neuron exactly once")
        test_visit_count += test_mask.astype(int)
        folds.append(
            SpiderFold(
                outer_fold=outer_fold,
                train_rows=np.flatnonzero(train_mask),
                validation_rows=np.flatnonzero(validation_mask),
                test_rows=np.flatnonzero(test_mask),
                train_slices=tuple(name for name in ordered_slices if name in training_slices),
                validation_slices=tuple(name for name in ordered_slices if name in validation_slices),
                test_slices=tuple(name for name in ordered_slices if name in test_slices),
            )
        )
    if not np.all(test_visit_count == 1):
        raise AssertionError("each neuron must be outer-test exactly once")
    return tuple(folds)


def prepare_fold_features(
    data: SpiderData,
    train_rows: Sequence[int],
    *,
    use_location: bool = False,
) -> tuple[np.ndarray, dict[str, tuple[int, ...]]]:
    """Fit selected feature transforms on inner train and apply to every row."""

    train = np.asarray(train_rows, dtype=int)
    if train.ndim != 1 or len(train) < 2 or len(np.unique(train)) != len(train):
        raise ValueError("At least two distinct training rows are required")
    if np.any(train < 0) or np.any(train >= data.n_cells):
        raise ValueError("Training rows are outside the dataset")
    gene_log = np.log1p(np.clip(data.X_gene_raw, 0, None))
    gene_scaler = StandardScaler().fit(gene_log[train])
    gene = gene_scaler.transform(gene_log)
    n_gene = gene.shape[1]
    blocks = {"gene": tuple(range(n_gene))}

    if use_location:
        spline = SplineTransformer(
            n_knots=5,
            degree=3,
            include_bias=False,
            extrapolation="linear",
        ).fit(data.X_loc_raw[train])
        location_basis = spline.transform(data.X_loc_raw)
        location_scaler = StandardScaler().fit(location_basis[train])
        location = location_scaler.transform(location_basis)
        x = np.concatenate([gene, location], axis=1).astype(np.float64)
        blocks["location_spline"] = tuple(range(n_gene, x.shape[1]))
    else:
        x = gene.astype(np.float64, copy=False)

    if not np.all(np.isfinite(x)):
        raise FloatingPointError("fold-specific SPIDER features are non-finite")
    return x, blocks


def _external_target_features(
    path: str | Path, target_ids: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read a target_id-indexed numeric CSV and align it without using outcomes."""

    frame = pd.read_csv(Path(path).expanduser(), index_col=0)
    frame.index = frame.index.astype(str)
    if frame.index.has_duplicates or not frame.columns.is_unique:
        raise ValueError("Target feature CSV must have unique target IDs and columns")
    missing = sorted(set(target_ids).difference(frame.index))
    if missing:
        raise ValueError(f"Target feature CSV lacks targets: {missing}")
    try:
        values = frame.loc[list(target_ids)].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Target descriptor columns must all be numeric") from exc
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Target features must have finite numeric descriptor columns")
    return values, tuple(map(str, frame.columns))


def spider_dataset(
    data: SpiderData, *, target_features_csv: str | Path | None = None
) -> ExperimentDataset:
    """Adapt validated arrays; preprocessing is fitted afresh for each train split.

    SPIDER's audited spatial object has no postsynaptic gene-expression features.
    A supplied target descriptor CSV must contain independent anatomical or
    molecular metadata, never summaries of projection labels. Its first column
    contains the exact target IDs and the remaining columns numeric descriptors.
    """

    external = (None if target_features_csv is None else
                _external_target_features(target_features_csv, data.target_ids))

    def features(train_rows, use_location=False, use_target_features=False):
        if not isinstance(use_location, bool) or not isinstance(use_target_features, bool):
            raise TypeError("Feature switches must be booleans")
        if use_target_features and external is None:
            raise ValueError(
                "SPIDER has no bundled target descriptors. Set "
                "USE_TARGET_FEATURES=False or pass target_features_csv=Path(...) "
                "to load_spider. The CSV needs target_id plus numeric independent "
                "descriptors; target one-hot IDs are not biological features."
            )
        x, blocks = prepare_fold_features(data, train_rows, use_location=use_location)
        names = data.gene_names + tuple(
            f"location_spline_{j}" for j in range(x.shape[1] - len(data.gene_names))
        )
        return FeatureSet(
            X=x, feature_blocks=blocks,
            Y_target=external[0].copy() if use_target_features else None,
            feature_names=names,
            metadata={
                "preprocessing": "log1p_gene_standardize; optional_location_spline",
                "fit_rows": np.asarray(train_rows, dtype=int).tolist(),
                "target_feature_names": external[1] if use_target_features else (),
                "target_feature_source": str(target_features_csv) if use_target_features else None,
            },
        )

    def splits(n_outer_folds=3, seed=0):
        # Spatial ordering determines this split. The seed is accepted for the
        # common interface but does not randomize spatial blocks.
        original = make_spatial_folds(data, n_outer_folds, require_36_slices=False)
        result = tuple(Fold(
            f.outer_fold, f.train_rows, f.validation_rows, f.test_rows,
            {"train_slices": f.train_slices, "validation_slices": f.validation_slices,
             "test_slices": f.test_slices, "split": "contiguous_ap_test_spaced_validation"},
        ) for f in original)
        for fold in result:
            fold.validate(data.n_cells)
        return result

    result = ExperimentDataset(
        name="SPIDER", reference=data.Z_reference, measured=data.W_measured,
        cell_ids=data.cell_ids, target_ids=data.target_ids,
        feature_builder=features, split_builder=splits,
        groups={"slice": data.slice_ids},
        metadata={
            "source_url": SPIDER_RDS_URL, "source_sha256": data.source_sha256,
            "reference": "pre-thinning assay outcome; not complete anatomical truth",
            "cohort": {"ABA_hemisphere": "Left", "SubType_Layer": SPIDER_INCLUDED_LAYERS},
            "gene_names": data.gene_names, "location_columns": SPIDER_LOCATION_COLUMNS,
            "target_features_available": external is not None,
            "target_feature_source": str(target_features_csv) if external is not None else None,
        },
    )
    result.validate()
    return result


def load_spider(
    cache_dir: str | Path, *, target_features_csv: str | Path | None = None
) -> ExperimentDataset:
    """Load the persistent audited raw RDS and expose dataset-independent contracts."""

    return spider_dataset(load_spider_data(cache_dir), target_features_csv=target_features_csv)
