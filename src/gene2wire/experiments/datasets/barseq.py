"""BARseq2 v2 paired expression/projection cohorts, fitted separately by panel.

A1 contains two animals and M1 one animal. Depth-block holdout is within-animal.
Off-panel targets are excluded structurally, never converted to negative labels.
Anatomical descriptors are parsed from target labels without projection outcomes.
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

from scipy.io import loadmat

BARSEQ2_DATASET_DOI = "10.17632/jnx89bmv4s.2"


BARSEQ2_DATASET_PAGE = "https://data.mendeley.com/datasets/jnx89bmv4s/2"


BARSEQ2_FILE_SPECS = {
    "A1pooleddata.mat": {
        "panel": "A1",
        "file_id": "bec3fbb5-a757-4882-9f03-835e7da3d3c6",
        "sha256": "d4b1ac270b90c15ccb21a5ce06e9baf02eb38eebd52388c1067147be8d8ffa31",
    },
    "M1pooleddata.mat": {
        "panel": "M1",
        "file_id": "f0a55ee3-adcd-4659-b994-7a56392e55b4",
        "sha256": "5ab81c2ea22482c0bf9a574885b5cf3f05abbafce8c737f377f4d5194b12c12c",
    },
}


PANEL_NAMES = ("A1", "M1")


LOCATION_COLUMNS = ("cortical_depth_um", "angle_sin", "angle_cos")


DEPTH_BINS_PER_ANIMAL = 12


TARGET_FEATURE_COLUMNS = (
    "family_cortex",
    "family_striatum",
    "family_thalamus",
    "family_amygdala",
    "family_claustrum",
    "family_midbrain_brainstem",
    "family_medulla",
    "family_spinal",
    "family_other_subcortical",
    "laterality_ipsilateral",
    "laterality_contralateral",
    "rostrocaudal_rostral",
    "rostrocaudal_intermediate",
    "rostrocaudal_caudal",
    "mediolateral_medial",
    "mediolateral_lateral",
)


def download_barseq2_files(cache_dir: str | Path) -> dict[str, tuple[Path, str]]:
    """Read verified cache files first; there is no mandatory metadata API call."""

    cache = Path(cache_dir).expanduser().resolve()
    resolved = {}
    for filename, specification in BARSEQ2_FILE_SPECS.items():
        path = cache / filename
        url = (
            "https://data.mendeley.com/public-files/datasets/jnx89bmv4s/files/"
            f"{specification['file_id']}/file_downloaded"
        )
        cached_download(url, path, sha256=specification["sha256"])
        resolved[specification["panel"]] = (path, specification["sha256"])
    return resolved


@dataclass(frozen=True)
class BarseqData:
    """Validated pre-hide BARseq2 arrays in one canonical neuron order."""

    X_gene_raw: np.ndarray
    X_loc_raw: np.ndarray
    Y_target_raw: np.ndarray
    Z_reference: np.ndarray
    W_measured: np.ndarray
    cell_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    target_panels: tuple[str, ...]
    target_regions: tuple[str, ...]
    target_feature_names: tuple[str, ...]
    animal_ids: np.ndarray
    panel_ids: np.ndarray
    slice_ids: np.ndarray
    depth_bins: np.ndarray
    gene_names: tuple[str, ...]
    source_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        n_cells = len(self.cell_ids)
        n_targets = len(self.target_ids)
        if self.X_gene_raw.ndim != 2 or self.X_gene_raw.shape[0] != n_cells:
            raise ValueError("X_gene_raw must be neuron-by-gene")
        if self.X_loc_raw.shape != (n_cells, len(LOCATION_COLUMNS)):
            raise ValueError("X_loc_raw must contain depth and angle sine/cosine")
        if self.Y_target_raw.shape != (n_targets, len(self.target_feature_names)):
            raise ValueError("Y_target_raw is not aligned to target_ids")
        if self.Z_reference.shape != (n_cells, n_targets):
            raise ValueError("Z_reference is not neuron-by-target")
        if self.W_measured.shape != self.Z_reference.shape:
            raise ValueError("W_measured is not aligned to Z_reference")
        if np.any(self.Z_reference & ~self.W_measured):
            raise ValueError("positive labels cannot occur outside the measured panel")
        if len(self.target_panels) != n_targets or len(self.target_regions) != n_targets:
            raise ValueError("target metadata is not aligned")
        for values, name in (
            (self.animal_ids, "animal_ids"),
            (self.panel_ids, "panel_ids"),
            (self.slice_ids, "slice_ids"),
            (self.depth_bins, "depth_bins"),
        ):
            if len(values) != n_cells:
                raise ValueError(f"{name} is not aligned to cells")
        if len(set(self.cell_ids)) != n_cells:
            raise ValueError("BARseq2 cell IDs must be unique")
        if not np.all(np.isfinite(self.X_gene_raw)) or not np.all(np.isfinite(self.X_loc_raw)):
            raise ValueError("BARseq2 cell features contain non-finite values")
        if not np.all(np.isfinite(self.Y_target_raw)):
            raise ValueError("BARseq2 target features contain non-finite values")

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_targets(self) -> int:
        return len(self.target_ids)

    def row_indices(self, panel: str) -> np.ndarray:
        if panel not in PANEL_NAMES:
            raise ValueError(f"unknown BARseq2 panel {panel!r}")
        return np.flatnonzero(self.panel_ids.astype(str) == panel)

    def target_indices(self, panel: str) -> np.ndarray:
        if panel not in PANEL_NAMES:
            raise ValueError(f"unknown BARseq2 panel {panel!r}")
        return np.flatnonzero(np.asarray(self.target_panels, dtype=str) == panel)

    def panel_target_features(self, panel: str) -> tuple[np.ndarray, tuple[str, ...]]:
        indices = self.target_indices(panel)
        values = self.Y_target_raw[indices]
        keep = np.any(np.abs(values) > 0, axis=0)
        names = tuple(
            name for name, selected in zip(self.target_feature_names, keep) if selected
        )
        return values[:, keep].astype(np.float64, copy=False), names


@dataclass(frozen=True)
class BarseqFold:
    outer_fold: int
    train_rows: np.ndarray
    validation_rows: np.ndarray
    test_rows: np.ndarray
    train_slices: tuple[str, ...]
    validation_slices: tuple[str, ...]
    test_slices: tuple[str, ...]
    train_depth_bins: tuple[int, ...]
    validation_depth_bins: tuple[int, ...]
    test_depth_bins: tuple[int, ...]


def _matlab_strings(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=object).reshape(-1).astype(str)


def _load_barseq2_panel(panel: str, path: Path) -> dict[str, Any]:
    source = loadmat(path, squeeze_me=True, struct_as_record=False)
    gene_labels = _matlab_strings(source["genelabels"])
    target_labels = _matlab_strings(source["labels"])
    expression = np.asarray(source["allbcexpmat"], dtype=np.float64)
    projection = np.asarray(source["proj"], dtype=np.float64)
    projection_raw = np.asarray(source["projraw"], dtype=np.float64)
    depths = np.asarray(source["allbcdepths"], dtype=np.float64)
    angles = np.asarray(source["allbcangles"], dtype=np.float64).reshape(-1)
    brain_index = np.asarray(source["brainidx"], dtype=int).reshape(-1)
    cell_id = np.asarray(source["allbcid"], dtype=float).reshape(-1)
    barcode = _matlab_strings(source["allbcseq"])

    n_rows = expression.shape[0]
    if expression.shape != (n_rows, 23):
        raise AssertionError(f"unexpected {panel} expression shape: {expression.shape}")
    if projection.shape != projection_raw.shape or projection.shape[0] != n_rows:
        raise AssertionError(f"unexpected {panel} projection shape")
    if len(gene_labels) != expression.shape[1] or len(target_labels) != projection.shape[1]:
        raise ValueError(f"{panel} expression/projection column labels are misaligned")
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise ValueError(f"{panel} gene counts must be finite and nonnegative")
    if not np.isfinite(projection).all() or not np.isfinite(projection_raw).all():
        raise ValueError(f"{panel} contains missing within-panel projection measurements")
    if depths.shape != (n_rows, 2):
        raise AssertionError(f"unexpected {panel} depth shape: {depths.shape}")
    if not all(len(values) == n_rows for values in (angles, brain_index, cell_id, barcode)):
        raise AssertionError(f"{panel} metadata is not row-aligned")
    if not np.isfinite(depths).all() or not np.isfinite(angles).all():
        raise ValueError(f"{panel} spatial coordinates must be finite")

    # Exact author preprocessing for the excitatory, projection-positive paired cohort.
    keep = (
        (projection_raw.max(axis=1) > 2)
        & (expression[:, 20] >= 5)
        & (expression[:, 20] >= expression[:, 21])
    )
    normalized_depths = (
        depths[keep]
        / np.maximum(depths[keep].sum(axis=1, keepdims=True), 1e-8)
        * 1200.0
    )
    return {
        "panel": panel,
        "gene_names": gene_labels,
        # The first projection column is the OB control and is not modeled.
        "target_names": target_labels[1:],
        "X_gene": expression[keep],
        "projection_signal": projection[keep, 1:],
        "cortical_depth_um": normalized_depths[:, 0],
        "cortical_depth_complement_um": normalized_depths[:, 1],
        "angle_deg": angles[keep],
        "brain_index": brain_index[keep],
        "cell_id": cell_id[keep],
        "barcode": barcode[keep],
        "source_row": np.flatnonzero(keep),
        "raw_row_count": n_rows,
    }


def _target_feature_row(panel: str, label: str) -> dict[str, float]:
    row = {name: 0.0 for name in TARGET_FEATURE_COLUMNS}
    if panel == "A1":
        if label in {"OFC", "M", "SS", "VisIp", "VisC", "AudC"}:
            row["family_cortex"] = 1.0
        elif label in {"Rstr", "Cstr"}:
            row["family_striatum"] = 1.0
        elif label == "Amyg":
            row["family_amygdala"] = 1.0
        elif label == "Thal":
            row["family_thalamus"] = 1.0
        elif label == "Tect":
            row["family_midbrain_brainstem"] = 1.0
        else:
            row["family_other_subcortical"] = 1.0
        if label in {"OFC", "M", "SS", "VisIp"}:
            row["laterality_ipsilateral"] = 1.0
        elif label in {"VisC", "AudC"}:
            row["laterality_contralateral"] = 1.0
        if label == "Rstr":
            row["rostrocaudal_rostral"] = 1.0
        elif label == "Cstr":
            row["rostrocaudal_caudal"] = 1.0
    elif panel == "M1":
        prefix = label.split("-")[0]
        if prefix in {"MOs", "ORB", "SS", "MOp", "SSp", "TEa"}:
            row["family_cortex"] = 1.0
        elif prefix == "CLA":
            row["family_claustrum"] = 1.0
        elif prefix == "Str":
            row["family_striatum"] = 1.0
        elif prefix == "Thal":
            row["family_thalamus"] = 1.0
        elif prefix == "MY":
            row["family_medulla"] = 1.0
        elif prefix == "Sp":
            row["family_spinal"] = 1.0
        elif prefix in {"MRN", "PG", "SC"}:
            row["family_midbrain_brainstem"] = 1.0
        else:
            row["family_other_subcortical"] = 1.0
        if label.endswith("-i"):
            row["laterality_ipsilateral"] = 1.0
        elif label.endswith("-c"):
            row["laterality_contralateral"] = 1.0
        if label.startswith("Str-r-") or label.startswith("Thal-mr-") or label.startswith("Thal-lr-"):
            row["rostrocaudal_rostral"] = 1.0
        elif label.startswith("Str-i-"):
            row["rostrocaudal_intermediate"] = 1.0
        elif label.startswith("Str-c-") or label.startswith("Thal-mc-") or label.startswith("Thal-lc-"):
            row["rostrocaudal_caudal"] = 1.0
        if label.startswith("Thal-m") or label.startswith("MY-m-"):
            row["mediolateral_medial"] = 1.0
        elif label.startswith("Thal-l") or label.startswith("MY-l-"):
            row["mediolateral_lateral"] = 1.0
    else:
        raise ValueError(panel)
    return row


def load_barseq2_data(cache_dir: str | Path) -> BarseqData:
    """Load BARseq2 and construct X_gene, X_loc, Y_target, Z, and panel mask W."""

    paths = download_barseq2_files(cache_dir)
    start = time.time()
    panels = {
        panel: _load_barseq2_panel(panel, paths[panel][0]) for panel in PANEL_NAMES
    }
    if not np.array_equal(panels["A1"]["gene_names"], panels["M1"]["gene_names"]):
        raise AssertionError("A1 and M1 gene panels differ")
    if panels["A1"]["X_gene"].shape != (593, 23):
        raise AssertionError(f"unexpected retained A1 shape: {panels['A1']['X_gene'].shape}")
    if panels["M1"]["X_gene"].shape != (749, 23):
        raise AssertionError(f"unexpected retained M1 shape: {panels['M1']['X_gene'].shape}")
    if len(panels["A1"]["target_names"]) != 11 or len(panels["M1"]["target_names"]) != 35:
        raise AssertionError("unexpected BARseq2 target counts")

    panel_metadata: list[pd.DataFrame] = []
    for panel in PANEL_NAMES:
        values = panels[panel]
        angle = np.deg2rad(values["angle_deg"])
        animal_ids = np.asarray(
            [f"{panel}_animal_{int(value)}" for value in values["brain_index"]]
        )
        cell_ids = np.asarray(
            [
                f"{panel}|animal{int(brain)}|row{int(row)}|id{cell_id:g}"
                for brain, row, cell_id in zip(
                    values["brain_index"], values["source_row"], values["cell_id"]
                )
            ]
        )
        panel_metadata.append(
            pd.DataFrame(
                {
                    "panel": panel,
                    "animal_id": animal_ids,
                    "cortical_depth_um": values["cortical_depth_um"],
                    "angle_sin": np.sin(angle),
                    "angle_cos": np.cos(angle),
                },
                index=cell_ids,
            )
        )
    metadata = pd.concat(panel_metadata, axis=0)
    if not metadata.index.is_unique:
        raise AssertionError("BARseq2 cell identifiers are not unique")

    x_gene = np.vstack([panels[panel]["X_gene"] for panel in PANEL_NAMES])
    x_loc = metadata[list(LOCATION_COLUMNS)].to_numpy(dtype=np.float64)
    target_regions = tuple(
        str(target)
        for panel in PANEL_NAMES
        for target in panels[panel]["target_names"].tolist()
    )
    target_panels = tuple(
        panel
        for panel in PANEL_NAMES
        for _ in range(len(panels[panel]["target_names"]))
    )
    target_ids = tuple(
        f"{panel}::{target}" for panel, target in zip(target_panels, target_regions)
    )
    y_target = np.asarray(
        [
            [row[name] for name in TARGET_FEATURE_COLUMNS]
            for row in (
                _target_feature_row(panel, region)
                for panel, region in zip(target_panels, target_regions)
            )
        ],
        dtype=np.float64,
    )

    n_a1 = len(panels["A1"]["X_gene"])
    n_cells = len(metadata)
    n_targets = len(target_ids)
    n_a1_targets = len(panels["A1"]["target_names"])
    reference = np.zeros((n_cells, n_targets), dtype=bool)
    measured = np.zeros_like(reference)
    measured[:n_a1, :n_a1_targets] = True
    measured[n_a1:, n_a1_targets:] = True
    reference[:n_a1, :n_a1_targets] = panels["A1"]["projection_signal"] > 0
    reference[n_a1:, n_a1_targets:] = panels["M1"]["projection_signal"] > 0

    depth_bins = np.full(n_cells, -1, dtype=int)
    animal_array = metadata["animal_id"].astype(str).to_numpy()
    depth_array = metadata["cortical_depth_um"].to_numpy(dtype=np.float64)
    for animal in sorted(np.unique(animal_array)):
        positions = np.flatnonzero(animal_array == animal)
        ordered = positions[np.argsort(depth_array[positions], kind="mergesort")]
        chunks = np.array_split(ordered, DEPTH_BINS_PER_ANIMAL)
        if any(len(chunk) == 0 for chunk in chunks):
            raise AssertionError(f"too few cells to create depth bins for {animal}")
        for depth_bin, chunk in enumerate(chunks):
            depth_bins[chunk] = depth_bin
    slice_ids = np.asarray(
        [
            f"{animal}|depth_bin_{depth_bin:02d}"
            for animal, depth_bin in zip(animal_array, depth_bins)
        ]
    )

    if sorted(np.unique(animal_array).tolist()) != [
        "A1_animal_1",
        "A1_animal_2",
        "M1_animal_3",
    ]:
        raise AssertionError("unexpected BARseq2 animal identifiers")
    if len(np.unique(slice_ids)) != 3 * DEPTH_BINS_PER_ANIMAL:
        raise AssertionError("BARseq2 pseudo-slice construction failed")
    if not np.all(np.sum(reference, axis=1) > 0):
        raise AssertionError("retained paired cohort should have a detected projection")

    print(
        f"Loaded BARseq2 in {time.time() - start:.1f}s: {n_cells:,} neurons, "
        f"{x_gene.shape[1]} genes, {n_targets} targets, 3 animals"
    )
    return BarseqData(
        X_gene_raw=x_gene,
        X_loc_raw=x_loc,
        Y_target_raw=y_target,
        Z_reference=reference,
        W_measured=measured,
        cell_ids=tuple(metadata.index.astype(str).tolist()),
        target_ids=target_ids,
        target_panels=target_panels,
        target_regions=target_regions,
        target_feature_names=TARGET_FEATURE_COLUMNS,
        animal_ids=animal_array,
        panel_ids=metadata["panel"].astype(str).to_numpy(),
        slice_ids=slice_ids,
        depth_bins=depth_bins,
        gene_names=tuple(panels["A1"]["gene_names"].astype(str).tolist()),
        source_sha256={panel: paths[panel][1] for panel in PANEL_NAMES},
    )


def make_spatial_folds(
    data: BarseqData,
    n_outer_folds: int = 3,
    n_validation_slices_per_animal: int = 2,
    *,
    require_36_slices: bool = True,
) -> tuple[BarseqFold, ...]:
    """Use the same contiguous depth block as test in every BARseq2 animal."""

    if not isinstance(n_outer_folds, int) or isinstance(n_outer_folds, bool) or not 2 <= n_outer_folds <= DEPTH_BINS_PER_ANIMAL:
        raise ValueError("BARseq2 outer folds must be between 2 and 12")
    if n_validation_slices_per_animal < 1:
        raise ValueError("at least one validation slice per animal is required")
    animals = tuple(sorted(np.unique(data.animal_ids.astype(str)).tolist()))
    if require_36_slices and len(np.unique(data.slice_ids.astype(str))) != 36:
        raise AssertionError("expected 36 BARseq2 pseudo-slices")
    test_blocks = tuple(
        tuple(int(value) for value in block)
        for block in np.array_split(np.arange(DEPTH_BINS_PER_ANIMAL), n_outer_folds)
    )
    folds: list[BarseqFold] = []
    test_visit_count = np.zeros(data.n_cells, dtype=int)
    for outer_fold, test_depth_bins in enumerate(test_blocks):
        test_depth_set = set(test_depth_bins)
        remaining_depth_bins = [
            value for value in range(DEPTH_BINS_PER_ANIMAL) if value not in test_depth_set
        ]
        if len(remaining_depth_bins) <= n_validation_slices_per_animal:
            raise ValueError("too few remaining depth bins for validation")
        validation_positions = np.unique(
            np.linspace(
                0,
                len(remaining_depth_bins) - 1,
                num=n_validation_slices_per_animal,
                dtype=int,
            )
        )
        if len(validation_positions) != n_validation_slices_per_animal:
            raise AssertionError("validation depth-bin selection produced duplicates")
        validation_depth_bins = tuple(
            remaining_depth_bins[int(position)] for position in validation_positions
        )
        validation_depth_set = set(validation_depth_bins)
        train_depth_bins = tuple(
            value for value in remaining_depth_bins if value not in validation_depth_set
        )

        test_mask = np.isin(data.depth_bins, test_depth_bins)
        validation_mask = np.isin(data.depth_bins, validation_depth_bins)
        train_mask = np.isin(data.depth_bins, train_depth_bins)
        membership = train_mask.astype(int) + validation_mask.astype(int) + test_mask.astype(int)
        if not np.all(membership == 1):
            raise AssertionError("BARseq2 fold does not partition every neuron exactly once")
        for role_mask in (train_mask, validation_mask, test_mask):
            if set(data.animal_ids[role_mask].astype(str)) != set(animals):
                raise AssertionError("every BARseq2 split role must contain every animal")
            if set(data.panel_ids[role_mask].astype(str)) != set(PANEL_NAMES):
                raise AssertionError("every BARseq2 split role must contain both panels")
        test_visit_count += test_mask.astype(int)
        folds.append(
            BarseqFold(
                outer_fold=outer_fold,
                train_rows=np.flatnonzero(train_mask),
                validation_rows=np.flatnonzero(validation_mask),
                test_rows=np.flatnonzero(test_mask),
                train_slices=tuple(sorted(np.unique(data.slice_ids[train_mask].astype(str)))),
                validation_slices=tuple(
                    sorted(np.unique(data.slice_ids[validation_mask].astype(str)))
                ),
                test_slices=tuple(sorted(np.unique(data.slice_ids[test_mask].astype(str)))),
                train_depth_bins=train_depth_bins,
                validation_depth_bins=validation_depth_bins,
                test_depth_bins=test_depth_bins,
            )
        )
    if not np.all(test_visit_count == 1):
        raise AssertionError("each BARseq2 neuron must be outer-test exactly once")
    return tuple(folds)


def prepare_fold_features(
    data: BarseqData,
    train_rows: Sequence[int],
    *,
    use_location: bool = False,
) -> tuple[np.ndarray, dict[str, tuple[int, ...]]]:
    """Fit gene/location transforms on panel-specific inner training rows only."""

    train = np.asarray(train_rows, dtype=int)
    if train.ndim != 1 or len(train) < 2 or len(np.unique(train)) != len(train):
        raise ValueError("At least two distinct training rows are required")
    if np.any(train < 0) or np.any(train >= data.n_cells):
        raise ValueError("Training rows are outside the dataset")
    gene_log = np.log1p(np.clip(data.X_gene_raw, 0, None))
    gene_scaler = StandardScaler().fit(gene_log[train])
    gene = gene_scaler.transform(gene_log)
    n_gene = gene.shape[1]
    feature_blocks: dict[str, tuple[int, ...]] = {"gene": tuple(range(n_gene))}
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
        x_all = np.concatenate([gene, location], axis=1).astype(np.float64)
        feature_blocks["location_spline"] = tuple(range(n_gene, x_all.shape[1]))
    else:
        x_all = gene.astype(np.float64, copy=False)
    if not np.all(np.isfinite(x_all)):
        raise FloatingPointError("fold-specific BARseq2 features are non-finite")
    return x_all, feature_blocks


def _panel_dataset(data: BarseqData, panel: str) -> ExperimentDataset:
    rows, targets = data.row_indices(panel), data.target_indices(panel)
    local_to_global = rows.copy()
    global_to_local = np.full(data.n_cells, -1, dtype=int)
    global_to_local[rows] = np.arange(len(rows))
    target_features, target_names = data.panel_target_features(panel)

    def features(train_rows, use_location=False, use_target_features=False):
        if not isinstance(use_location, bool) or not isinstance(use_target_features, bool):
            raise TypeError("Feature switches must be booleans")
        train = np.asarray(train_rows, dtype=int)
        if train.ndim != 1 or np.any(train < 0) or np.any(train >= len(rows)):
            raise ValueError("Training rows are outside this BARseq panel")
        all_x, blocks = prepare_fold_features(
            data, local_to_global[train], use_location=use_location
        )
        x = all_x[rows]
        return FeatureSet(
            X=x, feature_blocks=blocks,
            Y_target=target_features.copy() if use_target_features else None,
            feature_names=data.gene_names + tuple(
                f"location_spline_{j}" for j in range(x.shape[1] - len(data.gene_names))
            ),
            metadata={
                "preprocessing": "log1p_gene_standardize; optional_depth_angle_spline",
                "fit_rows": train.tolist(),
                "target_feature_names": target_names if use_target_features else (),
                "target_feature_source": "target-label-derived anatomy; no projection labels",
            },
        )

    def splits(n_outer_folds=3, seed=0):
        original = make_spatial_folds(data, n_outer_folds, require_36_slices=False)

        def local(global_rows):
            indices = global_to_local[global_rows]
            return indices[indices >= 0]

        result = tuple(Fold(
            f.outer_fold, local(f.train_rows), local(f.validation_rows), local(f.test_rows),
            {"split": "within_animal_contiguous_depth_holdout",
             "train_depth_bins": f.train_depth_bins,
             "validation_depth_bins": f.validation_depth_bins,
             "test_depth_bins": f.test_depth_bins},
        ) for f in original)
        for fold in result:
            fold.validate(len(rows))
        return result

    result = ExperimentDataset(
        name=f"BARseq {panel}",
        reference=data.Z_reference[np.ix_(rows, targets)],
        measured=data.W_measured[np.ix_(rows, targets)],
        cell_ids=tuple(data.cell_ids[i] for i in rows),
        target_ids=tuple(data.target_ids[j] for j in targets),
        feature_builder=features, split_builder=splits,
        gene_matrix=np.log1p(np.clip(data.X_gene_raw[rows], 0, None)),
        gene_names=tuple(data.gene_names),
        groups={"animal": data.animal_ids[rows], "slice": data.slice_ids[rows],
                "panel": data.panel_ids[rows]},
        metadata={
            "source_doi": BARSEQ2_DATASET_DOI, "source_url": BARSEQ2_DATASET_PAGE,
            "source_sha256": data.source_sha256[panel], "panel": panel,
            "reference": "pre-thinning assay outcome; not complete anatomical truth",
            "cohort": "max(projraw)>2; Slc17a7>=5; Slc17a7>=Gad1; remove OB control",
            "gene_names": data.gene_names, "location_columns": LOCATION_COLUMNS,
            "target_features_available": True,
            "target_feature_names": target_names,
            "target_feature_source": "target-label-derived anatomy; not postsynaptic expression",
            "animal_count": len(np.unique(data.animal_ids[rows])),
            "evaluation_scope": "within-animal depth-held-out; A1 and M1 fitted separately",
        },
    )
    result.validate()
    return result


def barseq_datasets(data: BarseqData) -> dict[str, ExperimentDataset]:
    """Expose two independent panel datasets; never train on off-panel zeros."""

    return {panel: _panel_dataset(data, panel) for panel in PANEL_NAMES}


def load_barseq(cache_dir: str | Path) -> dict[str, ExperimentDataset]:
    """Load the persistent BARseq2 raw MAT files and return A1/M1 adapters."""

    return barseq_datasets(load_barseq2_data(cache_dir))
