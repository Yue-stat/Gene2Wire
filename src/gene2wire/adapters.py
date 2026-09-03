"""Portable NPZ adapter and explicit boundaries for raw-data migration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import DatasetBundle


REQUIRED_NPZ_KEYS = {"X_cell", "S_observed", "W_measured"}


def load_npz_bundle(path: str | Path) -> DatasetBundle:
    """Load the canonical, pickle-free ``.npz`` interchange format.

    Optional keys are ``Y_target``, ``Z_reference``, ``reference_mask``,
    ``cell_ids``, ``target_ids``, ``semantics_json``, ``metadata_json``, plus
    keys prefixed with ``group__`` and ``feature_block__``.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_NPZ_KEYS.difference(archive.files))
        if missing:
            raise ValueError(f"{source} is missing required arrays: {missing}")
        values: dict[str, Any] = {name: archive[name] for name in archive.files}

    groups = {
        key.removeprefix("group__"): value
        for key, value in values.items()
        if key.startswith("group__")
    }
    feature_blocks = {
        key.removeprefix("feature_block__"): tuple(int(i) for i in value.tolist())
        for key, value in values.items()
        if key.startswith("feature_block__")
    }

    def json_value(name: str) -> dict[str, Any]:
        if name not in values:
            return {}
        raw = values[name]
        text = str(raw.item()) if np.asarray(raw).ndim == 0 else str(raw.tolist()[0])
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} must encode a JSON object")
        return parsed

    return DatasetBundle(
        X_cell=values["X_cell"],
        S_observed=values["S_observed"],
        W_measured=values["W_measured"],
        Y_target=values.get("Y_target"),
        Z_reference=values.get("Z_reference"),
        reference_mask=values.get("reference_mask"),
        cell_ids=None if "cell_ids" not in values else values["cell_ids"].tolist(),
        target_ids=None if "target_ids" not in values else values["target_ids"].tolist(),
        groups=groups,
        feature_blocks=feature_blocks,
        semantics=json_value("semantics_json"),
        metadata=json_value("metadata_json"),
    )


def save_npz_bundle(data: DatasetBundle, path: str | Path) -> Path:
    """Write a validated bundle to the portable interchange format."""

    destination = Path(path)
    arrays: dict[str, Any] = {
        "X_cell": data.X_cell,
        "S_observed": data.S_observed.astype(np.uint8),
        "W_measured": data.W_measured.astype(np.uint8),
        "cell_ids": np.asarray(data.cell_ids, dtype=str),
        "target_ids": np.asarray(data.target_ids, dtype=str),
        "semantics_json": np.asarray(json.dumps(dict(data.semantics), sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(dict(data.metadata), sort_keys=True)),
    }
    if data.Y_target is not None:
        arrays["Y_target"] = data.Y_target
    if data.Z_reference is not None:
        arrays["Z_reference"] = data.Z_reference.astype(np.uint8)
        arrays["reference_mask"] = data.reference_mask.astype(np.uint8)
    for name, values in data.groups.items():
        arrays[f"group__{name}"] = np.asarray(values, dtype=str)
    for name, indices in data.feature_blocks.items():
        arrays[f"feature_block__{name}"] = np.asarray(indices, dtype=np.int64)
    np.savez_compressed(destination, **arrays)
    return destination


def load_raw_dataset(dataset_name: str, *args: Any, **kwargs: Any) -> DatasetBundle:
    """Explicit boundary: raw dataset parsing belongs in experiment adapters."""

    del args, kwargs
    raise NotImplementedError(
        f"raw loader for {dataset_name!r} is not part of the core package; "
        "convert it to DatasetBundle in an experiment adapter"
    )
