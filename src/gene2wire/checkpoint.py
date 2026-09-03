"""Machine-independent fingerprints and atomic JSON checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


RUNTIME_LOCAL_KEYS = frozenset(
    {
        "checkpoint_dir",
        "device",
        "hardware",
        "host",
        "hostname",
        "num_workers",
        "output_dir",
        "path",
        "paths",
        "timestamp",
        "wall_time",
        "workers",
    }
)

DATASET_LOCAL_PATH_KEYS = frozenset(
    {
        "cache_dir",
        "file",
        "files",
        "local_path",
        "local_paths",
        "npz_path",
        "path",
        "paths",
        "raw_dir",
    }
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """Return a lossless canonical JSON representation.

    This function deliberately performs no semantic-key filtering, so it is
    safe for checkpoint identities and checksums.
    """

    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint_config(value: Any) -> Any:
    """Remove only documented machine-local fields at known config locations."""

    config = _jsonable(value)
    if not isinstance(config, dict):
        return config
    result = dict(config)
    runtime = result.get("runtime")
    if isinstance(runtime, dict):
        result["runtime"] = {
            key: item
            for key, item in runtime.items()
            if key.lower() not in RUNTIME_LOCAL_KEYS
        }
    dataset = result.get("dataset")
    if isinstance(dataset, dict):
        result["dataset"] = {
            key: item
            for key, item in dataset.items()
            if key.lower() not in DATASET_LOCAL_PATH_KEYS
        }
    return result


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def hash_named_files(files: Mapping[str, str | Path]) -> dict[str, str]:
    """Return content hashes keyed by stable logical names, never local paths."""

    if not files:
        raise ValueError("files mapping must not be empty")
    return {str(name): sha256_file(path) for name, path in sorted(files.items())}


def sha256_array(value: Any) -> str:
    """Hash an array's dtype, shape, and values without pickle serialization."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("object arrays are not supported; convert IDs to strings")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    if array.dtype.kind in {"U", "S"}:
        digest.update(canonical_json(array.astype(str).ravel().tolist()).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_source_tree(
    root: str | Path,
    suffixes: tuple[str, ...] = (".py",),
) -> str:
    """Hash source bytes and relative filenames in a machine-independent order."""

    source_root = Path(root)
    if not source_root.is_dir():
        raise ValueError(f"source root is not a directory: {source_root}")
    files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError("source tree contains no matching files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def experiment_fingerprint(
    config: Any,
    *,
    input_hashes: Mapping[str, str],
    code_version: str,
    seeds: Mapping[str, int],
    source_hash: str | None = None,
) -> str:
    """Hash semantic config, content hashes, code version, and seed policy.

    Machine paths and runtime hardware fields are intentionally removed.  Data
    identity therefore comes from content hashes or stable IDs supplied through
    ``input_hashes``, not from machine-local filenames.
    """

    if not input_hashes:
        raise ValueError("input_hashes must identify the data or generator")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, str)
        or not value
        for name, value in input_hashes.items()
    ):
        raise ValueError("input_hashes must map nonempty logical names to nonempty hashes")
    if not isinstance(code_version, str) or not code_version:
        raise ValueError("code_version must be a nonempty string")
    if source_hash is not None and (not isinstance(source_hash, str) or not source_hash):
        raise ValueError("source_hash must be None or a nonempty string")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(value, int)
        or isinstance(value, bool)
        for name, value in seeds.items()
    ):
        raise ValueError("seeds must map nonempty names to integers")
    payload = {
        "config": _fingerprint_config(config),
        "input_hashes": dict(sorted(input_hashes.items())),
        "code_version": str(code_version),
        "source_hash": source_hash,
        "seeds": {str(k): int(v) for k, v in sorted(seeds.items())},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def unit_key(**parts: Any) -> str:
    """Create a stable checkpoint unit key from semantic coordinates."""

    if not parts:
        raise ValueError("unit_key requires at least one coordinate")
    return canonical_json(parts)


class AtomicCheckpointStore:
    """JSON checkpoint store using write-fsync-replace atomic commits."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        hint = re.sub(r"[^a-zA-Z0-9_-]+", "-", key).strip("-")[:48] or "unit"
        return self.root / f"{hint}--{digest}.json"

    def save_complete(self, key: str, fingerprint: str, payload: Any) -> Path:
        safe_payload = _jsonable(payload)
        record = {
            "status": "complete",
            "unit_key": key,
            "fingerprint": fingerprint,
            "payload": safe_payload,
            "payload_sha256": hashlib.sha256(
                canonical_json(safe_payload).encode("utf-8")
            ).hexdigest(),
        }
        destination = self._path(key)
        text = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return destination

    def load(self, key: str, fingerprint: str | None = None) -> dict[str, Any] | None:
        source = self._path(key)
        if not source.exists():
            return None
        with source.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("unit_key") != key or record.get("status") != "complete":
            raise ValueError(f"invalid checkpoint record: {source}")
        expected_checksum = record.get("payload_sha256")
        actual_checksum = hashlib.sha256(
            canonical_json(record.get("payload")).encode("utf-8")
        ).hexdigest()
        if not isinstance(expected_checksum, str) or expected_checksum != actual_checksum:
            raise ValueError(f"checkpoint payload checksum mismatch: {source}")
        if fingerprint is not None and record.get("fingerprint") != fingerprint:
            return None
        return record

    def is_complete(self, key: str, fingerprint: str) -> bool:
        return self.load(key, fingerprint=fingerprint) is not None


class AtomicArrayCheckpointStore:
    """Atomic metadata + compressed-array checkpoints for completed model units.

    Array files are content-addressed and the small JSON manifest is committed
    last.  A runtime interruption can therefore leave an unused array file, but
    never a manifest that silently points to a partial file.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        hint = re.sub(r"[^a-zA-Z0-9_-]+", "-", key).strip("-")[:48] or "unit"
        return self.root / f"{hint}--{digest}.json"

    def save_complete(
        self,
        key: str,
        fingerprint: str,
        payload: Any,
        arrays: Mapping[str, Any],
    ) -> Path:
        if not fingerprint:
            raise ValueError("fingerprint must be nonempty")
        if any(not isinstance(name, str) or not name for name in arrays):
            raise ValueError("array names must be nonempty strings")
        safe_arrays: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(f"checkpoint array {name!r} has object dtype")
            safe_arrays[name] = array

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".arrays.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **safe_arrays)
                handle.flush()
                os.fsync(handle.fileno())
            arrays_sha256 = sha256_file(temporary)
            arrays_path = self.root / f"arrays--{arrays_sha256[:24]}.npz"
            if arrays_path.exists():
                temporary.unlink()
            else:
                os.replace(temporary, arrays_path)

            safe_payload = _jsonable(payload)
            record = {
                "status": "complete",
                "unit_key": key,
                "fingerprint": fingerprint,
                "payload": safe_payload,
                "payload_sha256": hashlib.sha256(
                    canonical_json(safe_payload).encode("utf-8")
                ).hexdigest(),
                "arrays_file": arrays_path.name,
                "arrays_sha256": arrays_sha256,
                "array_names": sorted(safe_arrays),
            }
            manifest = self._manifest_path(key)
            text = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
            json_descriptor, json_temporary_name = tempfile.mkstemp(
                prefix=f".{manifest.name}.", suffix=".tmp", dir=self.root
            )
            try:
                with os.fdopen(json_descriptor, "w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(json_temporary_name, manifest)
            except BaseException:
                try:
                    os.unlink(json_temporary_name)
                except FileNotFoundError:
                    pass
                raise
            return manifest
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def load(self, key: str, fingerprint: str | None = None) -> dict[str, Any] | None:
        manifest = self._manifest_path(key)
        if not manifest.exists():
            return None
        with manifest.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("unit_key") != key or record.get("status") != "complete":
            raise ValueError(f"invalid array checkpoint record: {manifest}")
        if fingerprint is not None and record.get("fingerprint") != fingerprint:
            return None
        safe_payload = record.get("payload")
        payload_sha256 = hashlib.sha256(
            canonical_json(safe_payload).encode("utf-8")
        ).hexdigest()
        if record.get("payload_sha256") != payload_sha256:
            raise ValueError(f"checkpoint payload checksum mismatch: {manifest}")

        arrays_name = record.get("arrays_file")
        if not isinstance(arrays_name, str) or Path(arrays_name).name != arrays_name:
            raise ValueError(f"invalid checkpoint array filename: {manifest}")
        arrays_path = self.root / arrays_name
        if not arrays_path.exists() or sha256_file(arrays_path) != record.get("arrays_sha256"):
            raise ValueError(f"checkpoint array checksum mismatch: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as archive:
            names = sorted(archive.files)
            if names != sorted(record.get("array_names", [])):
                raise ValueError(f"checkpoint array index mismatch: {arrays_path}")
            arrays = {name: np.array(archive[name], copy=True) for name in names}
        return {"payload": safe_payload, "arrays": arrays, "manifest": manifest}

    def is_complete(self, key: str, fingerprint: str) -> bool:
        return self.load(key, fingerprint=fingerprint) is not None
