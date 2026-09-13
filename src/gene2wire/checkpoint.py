"""Machine-independent fingerprints and atomic compact checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import zlib
from contextlib import closing
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


class CompactCheckpointStore:
    """Concurrent, compact index for small completed checkpoint records.

    ``AtomicCheckpointStore`` deliberately creates one JSON file per unit.  That
    is convenient for a handful of records, but candidate-level and
    scenario-level experiments can otherwise leave tens of thousands of tiny
    files on a shared filesystem.  This store provides the same ``save/load``
    contract in one SQLite index per logical checkpoint directory.  Payloads
    are canonical-JSON encoded, checksummed, compressed, and committed in a
    transaction; SQLite supplies process-safe atomicity and crash recovery.

    Existing JSON checkpoints are read as a compatibility fallback, but new
    records are written only to ``index.sqlite3``.  Nothing in an existing
    checkpoint directory is deleted or silently adopted across fingerprints.
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        read_legacy: bool = True,
        legacy_root: str | Path | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "index.sqlite3"
        legacy_path = self.root if legacy_root is None else Path(legacy_root)
        # An explicit fallback is read-only compatibility. Do not manufacture
        # empty legacy trials/models/refits/warm-start directories in a fresh
        # compact run merely by constructing a reader.
        self._legacy = (
            AtomicCheckpointStore(legacy_path)
            if read_legacy and (legacy_root is None or legacy_path.exists())
            else None
        )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS completed_checkpoint (
                        key_sha256 TEXT PRIMARY KEY,
                        unit_key TEXT NOT NULL,
                        fingerprint TEXT NOT NULL,
                        payload_zlib BLOB NOT NULL,
                        payload_sha256 TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK(status = 'complete')
                    )
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        # Keep SQLite's network-filesystem-friendly default rollback journal
        # (do not opt into WAL). Each operation is intentionally short, while
        # the busy timeout lets independent scenario workers serialize commits.
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    @staticmethod
    def _key_sha256(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def save_complete(self, key: str, fingerprint: str, payload: Any) -> Path:
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint key must be nonempty")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("fingerprint must be nonempty")
        safe_payload = _jsonable(payload)
        payload_text = canonical_json(safe_payload)
        checksum = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        compressed = sqlite3.Binary(zlib.compress(payload_text.encode("utf-8"), level=6))
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO completed_checkpoint (
                        key_sha256, unit_key, fingerprint, payload_zlib,
                        payload_sha256, schema_version, status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'complete')
                    ON CONFLICT(key_sha256) DO UPDATE SET
                        unit_key=excluded.unit_key,
                        fingerprint=excluded.fingerprint,
                        payload_zlib=excluded.payload_zlib,
                        payload_sha256=excluded.payload_sha256,
                        schema_version=excluded.schema_version,
                        status='complete'
                    """,
                    (
                        self._key_sha256(key), key, fingerprint, compressed,
                        checksum, self._SCHEMA_VERSION,
                    ),
                )
        return self.path

    def load(self, key: str, fingerprint: str | None = None) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT unit_key, fingerprint, payload_zlib, payload_sha256,
                       schema_version, status
                FROM completed_checkpoint WHERE key_sha256 = ?
                """,
                (self._key_sha256(key),),
            ).fetchone()
        if row is None:
            return (None if self._legacy is None else
                    self._legacy.load(key, fingerprint=fingerprint))
        unit_key_value, stored_fingerprint, compressed, checksum, version, status = row
        if unit_key_value != key or status != "complete" or version != self._SCHEMA_VERSION:
            raise ValueError(f"invalid compact checkpoint record: {self.path}")
        if fingerprint is not None and stored_fingerprint != fingerprint:
            return (None if self._legacy is None else
                    self._legacy.load(key, fingerprint=fingerprint))
        try:
            payload_text = zlib.decompress(bytes(compressed)).decode("utf-8")
            payload = json.loads(payload_text)
        except (UnicodeDecodeError, json.JSONDecodeError, zlib.error) as error:
            raise ValueError(f"invalid compact checkpoint payload: {self.path}") from error
        actual_checksum = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        if not isinstance(checksum, str) or checksum != actual_checksum:
            raise ValueError(f"compact checkpoint payload checksum mismatch: {self.path}")
        return {
            "status": "complete",
            "unit_key": key,
            "fingerprint": stored_fingerprint,
            "payload": payload,
            "payload_sha256": checksum,
        }

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
            if arrays_path.exists() and sha256_file(arrays_path) == arrays_sha256:
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


class CompactArrayCheckpointStore:
    """Content-addressed arrays with one compact manifest index.

    Large numeric payloads remain ordinary compressed NPZ blobs, so concurrent
    workers never rewrite a monolithic array database.  Only the small mapping
    from a semantic checkpoint key to its checksummed blob is consolidated.
    Existing ``AtomicArrayCheckpointStore`` manifests remain readable and are
    never removed.
    """

    _PAYLOAD_SCHEMA = 1

    def __init__(self, root: str | Path, *, legacy_root: str | Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = CompactCheckpointStore(
            self.root / "manifest_index", read_legacy=False
        )
        legacy_path = self.root if legacy_root is None else Path(legacy_root)
        self._legacy = (
            AtomicArrayCheckpointStore(legacy_path)
            if legacy_root is None or legacy_path.exists()
            else None
        )

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
            if arrays_path.exists() and sha256_file(arrays_path) == arrays_sha256:
                temporary.unlink()
            else:
                os.replace(temporary, arrays_path)
            self._index.save_complete(
                key,
                fingerprint,
                {
                    "schema": self._PAYLOAD_SCHEMA,
                    "checkpoint_payload": _jsonable(payload),
                    "arrays_file": arrays_path.name,
                    "arrays_sha256": arrays_sha256,
                    "array_names": sorted(safe_arrays),
                },
            )
            return self._index.path
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def load(self, key: str, fingerprint: str | None = None) -> dict[str, Any] | None:
        record = self._index.load(key, fingerprint=fingerprint)
        if record is None:
            return (None if self._legacy is None else
                    self._legacy.load(key, fingerprint=fingerprint))
        indexed = record["payload"]
        if indexed.get("schema") != self._PAYLOAD_SCHEMA:
            raise ValueError(f"invalid compact array checkpoint schema: {self._index.path}")
        arrays_name = indexed.get("arrays_file")
        if not isinstance(arrays_name, str) or Path(arrays_name).name != arrays_name:
            raise ValueError(f"invalid compact checkpoint array filename: {self._index.path}")
        arrays_path = self.root / arrays_name
        if (not arrays_path.exists()
                or sha256_file(arrays_path) != indexed.get("arrays_sha256")):
            raise ValueError(f"checkpoint array checksum mismatch: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as archive:
            names = sorted(archive.files)
            if names != sorted(indexed.get("array_names", [])):
                raise ValueError(f"checkpoint array index mismatch: {arrays_path}")
            arrays = {name: np.array(archive[name], copy=True) for name in names}
        return {
            "payload": indexed.get("checkpoint_payload"),
            "arrays": arrays,
            "manifest": self._index.path,
        }

    def is_complete(self, key: str, fingerprint: str) -> bool:
        return self.load(key, fingerprint=fingerprint) is not None
