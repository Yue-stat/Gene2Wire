"""Projection-TAGs paired assay adapter; no model fitting or label-based features.

The retained cohort matches the audited notebook: strict matched IDs, non-Parse
platforms, and five prespecified excitatory classes. The source-origin covariate
is optional; UMAP, barcode/vector expression and projection labels are excluded.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from ..contracts import ExperimentDataset, FeatureSet, Fold

TARGETS = ("MOp", "SSp", "VP", "PAG", "MY", "SCL", "SCS")
GENE_FEATURES = (
    "Pou3f1", "Rorb", "S100b", "Hap1", "Dio3", "Cbln2", "Penk", "Bdnf",
    "Dlk1", "Otof", "Sema3d", "Adamts2", "Cpne7", "Tnnc1", "Nnat", "Cux2",
    "Calb1", "Etv1", "Tshz2", "Deptor", "Syt6", "Oprk1", "Grp", "Ddit4l",
    "Rspo1", "Fstl5", "Slc24a2", "Scn4b", "Ptn", "Pcp4", "Efnb3", "Cdh13",
)
INCLUDED_CELLTYPES = ("L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L5 PT")
VECTOR_FEATURES = ("Sun1-GFP", "WPRE", *(f"BC{i}" for i in range(1, 13)))
SOURCES = {
    "expression": ("standard_gene_matrix.rds", "https://digitalcommonsdata.wustl.edu/public-files/datasets/d4htjwvcmg/files/c2a9c2d6-5861-46b4-9994-e7d3a128a72d/file_downloaded", "61201ac270a2e6db889db7bc1c8ea0cd05ffea9f57b23607c31283dd9c11825b"),
    "standard": ("standard_meta.csv", "https://digitalcommonsdata.wustl.edu/public-files/datasets/d4htjwvcmg/files/36566623-deef-4115-ba91-58275004e7b6/file_downloaded", "3b1f0888fe73e7fe1f98fe4453cac95330bf5faa88eb25a56b25c0cbeb32e3a1"),
    "union": ("GSE277718_snRNAseq_metadata.csv.gz", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE277nnn/GSE277718/suppl/GSE277718_snRNAseq_metadata.csv.gz", "2abcfb4c60f1cf974d52a8ebd6ddb0c3ca0967a0c8961965bf6527da48da37ef"),
    "animals": ("supplementary_data_2.xlsx", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-60360-w/MediaObjects/41467_2025_60360_MOESM4_ESM.xlsx", "bf0000b787268ce545ebd252671e12fcb287eb4012d0f1ab9b998216fb15f156"),
}


def canonical_standard_cell_id(cell_id: str) -> str:
    """Map old processed-file IDs to final GEO release IDs, one-to-one."""
    rules = (("RNA_16_MO__", "GEX__", 1), ("RNA_16_SSC__", "GEX__", 2),
             ("RNA_19_MO__", "GEX__", 3), ("RNA_19_SSC__", "GEX__", 4),
             ("Parse_07_SSC__", "Parse_1__", None),
             ("Parse_09_SSC__", "Parse_2__", None))
    for old, new, suffix in rules:
        if str(cell_id).startswith(old):
            remainder = str(cell_id)[len(old):]
            if suffix is not None:
                remainder = re.sub(r"-1$", f"-{suffix}", remainder)
            return new + remainder
    return str(cell_id)


def _sequencing_animal_table(path: str | Path) -> pd.DataFrame:
    """Extract the six sequencing-animal barcode rows from Supplementary Data 2.

    The official ``Animals`` worksheet contains a second ``Animal ID`` header
    for the HCR cohort.  Locate candidate headers by content and accept the
    unique one followed by ``Animal_SEQ_1`` through ``Animal_SEQ_6``.  This is
    stricter than selecting the first header while remaining insensitive to
    title rows, blank rows, or harmless surrounding whitespace.
    """
    sheet = pd.read_excel(path, sheet_name="Animals", header=None)

    def normalized(value) -> str:
        if pd.isna(value):
            return ""
        return " ".join(str(value).replace("\xa0", " ").split())

    expected_ids = {f"Animal_SEQ_{animal}" for animal in range(1, 7)}
    required = ("Animal ID", *(f"BC injected into {target}" for target in TARGETS))
    candidates = []
    for header_row in range(len(sheet)):
        labels = [normalized(value) for value in sheet.iloc[header_row].tolist()]
        if "Animal ID" not in labels or any(label not in labels for label in required):
            continue
        positions = {label: labels.index(label) for label in required}
        animal_column = sheet.iloc[header_row + 1:, positions["Animal ID"]].map(normalized)
        selected_rows = animal_column.index[animal_column.isin(expected_ids)]
        selected_ids = animal_column.loc[selected_rows]
        if set(selected_ids) != expected_ids or not selected_ids.is_unique:
            continue
        records = {
            label: [normalized(sheet.iat[row, column]) or np.nan for row in selected_rows]
            for label, column in positions.items()
        }
        candidates.append(pd.DataFrame(records))
    if len(candidates) != 1:
        raise ValueError(
            "Cannot identify a unique Supplementary Data 2 sequencing-animal barcode table"
        )
    return candidates[0]


def _aligned_numeric_csv(path, ids: Sequence[str], role: str):
    if path is None:
        return None, ()
    table = pd.read_csv(Path(path).expanduser(), index_col=0)
    table.index = table.index.astype(str)
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(f"{role} CSV must have unique row IDs and column names")
    missing = set(map(str, ids)) - set(table.index)
    if missing:
        raise ValueError(f"{role} CSV is missing aligned IDs: {sorted(missing)[:5]}")
    try:
        values = table.loc[list(map(str, ids))].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{role} CSV columns must be numeric") from exc
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f"{role} CSV must contain finite feature columns")
    return values, tuple(map(str, table.columns))


def _training_rows(rows, n):
    values = np.asarray(rows)
    if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
        raise ValueError("Feature fitting requires nonempty integer training-row indices")
    if len(np.unique(values)) != len(values) or values.min() < 0 or values.max() >= n:
        raise ValueError("Training rows must be unique valid cell indices")
    return values.astype(int)


def _group_folds(groups, n_outer_folds, seed, *, fixed_test=None, fixed_validation=None,
                 measured=None, platform=None):
    """Outcome-free balanced group partitions; optional assay-coverage checks.

    Counts, measured panels and platform are design information, not outcomes.
    Seed resolves equal-size group ties reproducibly. Each cell is tested once.
    """
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    k = int(n_outer_folds)
    if k != n_outer_folds or k < 2 or k > len(unique):
        raise ValueError(f"n_outer_folds must be an integer in [2, {len(unique)}]")
    if len(unique) < 3:
        raise ValueError("At least three groups are needed for train/validation/test")
    all_targets = None if measured is None else np.asarray(measured).any(axis=0)

    def supports_training(test, validation):
        training = np.flatnonzero(~np.isin(groups, [*test, validation]))
        if not len(training):
            return False
        if measured is not None and not np.array_equal(
                np.asarray(measured)[training].any(axis=0), all_targets):
            return False
        if platform is not None and set(np.asarray(platform)[training]) != set(platform):
            return False
        return True

    if fixed_test is None:
        rng = np.random.default_rng(seed)
        jitter = rng.random(len(unique))
        order = np.lexsort((jitter, -counts))
        bins, loads = [[] for _ in range(k)], np.zeros(k, dtype=int)
        for j in order:
            b = int(np.argmin(loads))
            bins[b].append(unique[j])
            loads[b] += counts[j]
        # With six Projection-TAGs animals a count-only greedy split can put
        # both SSp-assayed animals in one test fold. Search the small set of
        # complete group partitions using only counts/panels/platforms instead.
        def feasible(partition):
            return all(any(supports_training(test, val) for val in unique if val not in test)
                       for test in partition)
        if (measured is not None or platform is not None) and not feasible(bins):
            if len(unique) > 8:
                raise ValueError("No supported greedy group split; supply an explicit design partition")

            def partitions(values, count):
                blocks = []

                def visit(position):
                    if position == len(values):
                        if len(blocks) == count:
                            yield [list(block) for block in blocks]
                        return
                    if len(blocks) + len(values) - position < count:
                        return
                    value = values[position]
                    for block in blocks:
                        block.append(value)
                        yield from visit(position + 1)
                        block.pop()
                    if len(blocks) < count:
                        blocks.append([value])
                        yield from visit(position + 1)
                        blocks.pop()
                yield from visit(0)

            count_by_group = dict(zip(unique, counts))
            valid = []
            for partition in partitions(unique, k):
                if feasible(partition):
                    sizes = [sum(count_by_group[g] for g in block) for block in partition]
                    valid.append((float(np.var(sizes)), float(rng.random()), partition))
            if not valid:
                raise ValueError("No group partition preserves training target/platform coverage")
            bins = min(valid, key=lambda item: item[:2])[2]
    else:
        bins = [list(x) for x in fixed_test]
        if len(bins) != k:
            raise ValueError("Fixed test groups do not match n_outer_folds")
    output = []
    for f, test in enumerate(bins):
        remaining = [g for g in unique if g not in test]
        if len(remaining) < 2:
            raise ValueError("This group partition leaves no separate training group")
        candidates = (remaining[f % len(remaining):] + remaining[:f % len(remaining)])
        if fixed_validation is not None:
            candidates = [fixed_validation[f]]
        val = None
        for candidate in candidates:
            if supports_training(test, candidate):
                val = candidate
                break
        if val is None:
            raise ValueError("Group split leaves a training target/platform unsupported; "
                             "use the audited 3-fold Projection-TAGs design")
        train_groups = [g for g in remaining if g != val]
        fold = Fold(f, np.flatnonzero(np.isin(groups, train_groups)),
                    np.flatnonzero(groups == val), np.flatnonzero(np.isin(groups, test)),
                    {"train_groups": list(map(str, train_groups)),
                     "validation_groups": [str(val)], "test_groups": list(map(str, test)),
                     "assignment_uses_outcomes": False})
        fold.validate(len(groups))
        output.append(fold)
    visits = np.bincount(np.concatenate([f.test_rows for f in output]), minlength=len(groups))
    if not np.all(visits == 1):
        raise ValueError("Outer test groups must cover each cell exactly once")
    return tuple(output)


def _make_dataset(*, gene_counts, library_size, meta, standard, reference, measured,
                  metadata, location_features_csv=None, target_features_csv=None):
    """Construct an adapter from audited, aligned retained-cell arrays."""
    counts = np.asarray(gene_counts, dtype=float)
    library = np.asarray(library_size, dtype=float)
    if counts.shape != (len(meta), len(GENE_FEATURES)) or library.shape != (len(meta),):
        raise ValueError("Projection-TAGs expression/library sizes are misaligned")
    if not np.isfinite(counts).all() or np.any(counts < 0) or not np.isfinite(library).all() or np.any(library <= 0):
        raise ValueError("Expression must be nonnegative and library sizes positive")
    gene_log = np.log1p(1e4 * counts / library[:, None])
    ids = tuple(meta.index.astype(str))
    location, location_names = _aligned_numeric_csv(location_features_csv, ids, "Location")
    if location is None:
        origins = meta["Origin"].astype(str)
        if not set(origins).issubset({"MO", "SSC"}):
            raise ValueError("Unexpected Projection-TAGs source origin")
        location = origins.eq("MO").to_numpy(float)[:, None]
        location_names = ("source_origin_MO",)
    target, target_names = _aligned_numeric_csv(target_features_csv, TARGETS, "Target feature")
    animals = meta["Animal_ID"].to_numpy(int)
    platforms = meta["Platform"].astype(str).to_numpy()

    def features(train_rows, use_location=False, use_target_features=False):
        train = _training_rows(train_rows, len(ids))
        x = StandardScaler().fit(gene_log[train]).transform(gene_log)
        blocks = {"gene": tuple(range(x.shape[1]))}
        names = GENE_FEATURES
        if use_location:
            loc = StandardScaler().fit(location[train]).transform(location)
            blocks["location"] = tuple(range(x.shape[1], x.shape[1] + loc.shape[1]))
            x = np.column_stack((x, loc))
            names += location_names
        if use_target_features and target is None:
            raise ValueError("Projection-TAGs has no native target expression descriptors; "
                             "set target_features_csv with rows indexed by target ID")
        return FeatureSet(x, blocks, target if use_target_features else None, names,
                          {"preprocessing_fit_rows": train.tolist(),
                           "location_kind": "external" if location_features_csv else "source_origin",
                           "target_feature_names": list(target_names) if use_target_features else []})

    def splits(n_outer_folds=3, seed=0):
        fixed = None
        val = None
        if int(n_outer_folds) == 3 and set(animals) == set(range(1, 7)):
            fixed, val = ((1, 4), (2, 6), (3, 5)), (2, 4, 1)
        return _group_folds(animals, n_outer_folds, seed, fixed_test=fixed,
                            fixed_validation=val, measured=measured, platform=platforms)

    audit = dict(metadata)
    audit.update({"standard_positives": int(np.sum(standard)),
                  "reference_positives": int(np.sum(reference)),
                  "hidden_positives": int(np.sum(np.asarray(reference, bool) & ~np.asarray(standard, bool))),
                  "target_audit": [{"target": t, "measured_cells": int(np.sum(measured[:, j])),
                                    "standard_positives": int(np.sum(standard[:, j])),
                                    "reference_positives": int(np.sum(reference[:, j]))}
                                   for j, t in enumerate(TARGETS)],
                  "included_celltypes": list(INCLUDED_CELLTYPES), "excluded_platforms": ["Parse"],
                  "gene_features": list(GENE_FEATURES), "location_available": True,
                  "native_target_features_available": False})
    dataset = ExperimentDataset("Projection-TAGs", np.asarray(reference, bool), np.asarray(measured, bool),
                                ids, TARGETS, features, splits, {"animal": animals},
                                natural_observed=np.asarray(standard, bool), platform=platforms,
                                metadata=audit)
    dataset.validate()
    return dataset


def load_projection_tags(cache_dir, *, location_features_csv=None, target_features_csv=None,
                         raw_paths: Mapping[str, str | Path] | None = None):
    """Load/cached-download the four audited raw files and preserve cohort rules.

    ``raw_paths`` may map expression/standard/union/animals to already downloaded
    official files. They are checksum verified. Location CSV rows are cell IDs;
    target CSV rows are the exact seven TARGETS. Neither CSV may use outcomes.
    """
    from ..io import atomic_json, atomic_npz, cached_download, file_hash

    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, url, sha) in SOURCES.items():
        path = Path((raw_paths or {}).get(key, cache / name)).expanduser()
        paths[key] = Path(cached_download(url, path, sha256=sha))
    standard_raw = pd.read_csv(paths["standard"], index_col=0, low_memory=False)
    union_raw = pd.read_csv(paths["union"], index_col=0, low_memory=False)
    standard_raw.index = standard_raw.index.astype(str)
    union_raw.index = union_raw.index.astype(str)
    standard_raw["_rds_position"] = np.arange(len(standard_raw))
    standard = standard_raw.copy()
    standard.index = [canonical_standard_cell_id(x) for x in standard.index]
    if not standard.index.is_unique or not union_raw.index.is_unique:
        raise ValueError("Raw paired cell IDs are not one-to-one")
    matched = union_raw.index[union_raw.index.isin(standard.index)]
    union = union_raw.loc[matched]
    keep_platform = union["Platform"].ne("Parse")
    keep_class = union["Celltype"].isin(INCLUDED_CELLTYPES)
    meta = union.loc[keep_platform & keep_class].copy()
    standard = standard.loc[meta.index]
    if len(meta) == 0:
        raise ValueError("No cells survive the prespecified paired cohort filters")
    table = _sequencing_animal_table(paths["animals"])
    table = table[table["Animal ID"].astype(str).str.match(r"Animal_SEQ_[1-6]$")].copy()
    table["animal"] = table["Animal ID"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    if set(table["animal"]) != set(range(1, 7)) or not table["animal"].is_unique:
        raise ValueError("Animal-specific barcode map must contain animals 1--6 exactly once")
    n, t = len(meta), len(TARGETS)
    measured = np.zeros((n, t), bool)
    observed = np.zeros((n, t), bool)
    reference = np.zeros((n, t), bool)
    for _, animal in table.iterrows():
        rows = meta["Animal_ID"].to_numpy(int) == animal["animal"]
        for j, target in enumerate(TARGETS):
            barcode = animal[f"BC injected into {target}"]
            if pd.isna(barcode) or str(barcode).strip().upper() == "NA":
                continue
            values = meta.loc[rows, target].to_numpy(float)
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f"Invalid reference labels for animal {animal['animal']}, {target}")
            standard_values = standard.loc[rows, f"{str(barcode).strip()}.counts"].to_numpy(float)
            if not np.isfinite(standard_values).all() or np.any(standard_values < 0):
                raise ValueError("Invalid standard barcode counts")
            measured[rows, j] = True
            observed[rows, j] = standard_values > 0
            reference[rows, j] = values.astype(bool)
    positions = standard["_rds_position"].to_numpy(int)
    key = hashlib.sha256(json.dumps({"expression_sha": SOURCES["expression"][2],
        "positions": positions.tolist(), "genes": GENE_FEATURES, "vector": VECTOR_FEATURES},
        sort_keys=True).encode()).hexdigest()[:20]
    compact = cache / f"retained_gene_counts_{key}.npz"
    compact_manifest = compact.with_suffix(".json")
    if compact.exists() and compact_manifest.exists():
        recorded = json.loads(compact_manifest.read_text())
        if recorded.get("sha256") != file_hash(compact):
            raise ValueError("Processed Projection-TAGs cache checksum mismatch; remove the processed pair to rebuild")
        with np.load(compact, allow_pickle=False) as data:
            counts, library = data["counts"], data["library"]
    else:
        try:
            import rdata
        except ImportError as exc:
            raise ImportError("Projection-TAGs raw RDS reading requires the optional rdata dependency") from exc
        obj = rdata.conversion.convert(rdata.parser.parse_file(paths["expression"]))
        genes = np.asarray(obj.Dimnames[0]).astype(str)
        cells = np.asarray(obj.Dimnames[1]).astype(str)
        if not np.array_equal(cells, standard_raw.index.to_numpy()):
            raise ValueError("RDS columns and standard metadata cells are not aligned")
        matrix = sparse.csc_matrix((obj.x, obj.i, obj.p), shape=tuple(map(int, obj.Dim)))
        def unique_indices(names):
            indices = []
            for name in names:
                matches = np.flatnonzero(genes == name)
                if len(matches) != 1:
                    raise ValueError(f"Expected one RDS expression row for {name}, found {len(matches)}")
                indices.append(int(matches[0]))
            return np.asarray(indices, dtype=int)
        selected, vectors = unique_indices(GENE_FEATURES), unique_indices(VECTOR_FEATURES)
        counts = matrix[selected][:, positions].T.toarray().astype(np.float32)
        library = (np.asarray(matrix.sum(axis=0)).ravel() -
                   np.asarray(matrix[vectors].sum(axis=0)).ravel())[positions]
        atomic_npz(compact, counts=counts, library=library)
        atomic_json({"sha256": file_hash(compact), "source_key": key}, compact_manifest)
    audit = {"source_sha256": {k: v[2] for k, v in SOURCES.items()},
             "standard_raw_cells": len(standard_raw), "union_raw_cells": len(union_raw),
             "strict_matched_cells": len(matched), "retained_cells": len(meta),
             "excluded_parse_matched_cells": int((~keep_platform).sum()),
             "excluded_celltype_after_platform": int((keep_platform & ~keep_class).sum()),
             "reference_kind": "standard_or_target_amplified_assay", "cohort_version": "matched_nonparse_five_classes_v1"}
    return _make_dataset(gene_counts=counts, library_size=library, meta=meta,
                         standard=observed, reference=reference, measured=measured, metadata=audit,
                         location_features_csv=location_features_csv, target_features_csv=target_features_csv)
